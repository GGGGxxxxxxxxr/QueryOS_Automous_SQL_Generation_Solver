from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
CLEANED_QUERYOS_ROOT = SCRIPT_DIR.parent
if str(CLEANED_QUERYOS_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEANED_QUERYOS_ROOT))

from extractor import dedupe_pattern_tuples, extract_and_route_record, select_unnecessary_proposals  # noqa: E402
from io_utils import append_jsonl, filter_records, load_jsonl, load_processed_question_ids, write_json  # noqa: E402
from query_os.config import cfg_get, load_yaml_config, pick  # noqa: E402
from query_os.llm import create_llm_backend, safe_llm_error  # noqa: E402
from taxonomy import (  # noqa: E402
    advance_taxonomy_sample,
    apply_taxonomy_routing,
    build_general_mistake_set,
    drop_proposals_by_ids,
    load_taxonomy,
    normalize_atomic_items,
    normalize_pattern_tuples,
    prune_stale_proposals,
)
from text_utils import clean_text, safe_int, sanitize_output_obj  # noqa: E402


def main() -> int:
    args = parse_args()
    paths = build_output_paths(args)

    if args.reset:
        for path in generated_output_paths(paths):
            if path.exists():
                path.unlink()

    config = load_yaml_config(args.config)
    provider, model, client = build_llm(args, config)
    temperature = float(args.temperature if args.temperature is not None else cfg_get(config, "temperature", 0.2))
    taxonomy = load_taxonomy(paths["taxonomy"])
    processed_question_ids = load_processed_question_ids(paths["atomic"]) | load_processed_question_ids(paths["ignored"])

    records = list(load_jsonl(paths["error_bank"]))
    records = filter_records(records, question_ids=args.question_id)
    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]

    print(f"[general-mistakes] error bank: {paths['error_bank']}")
    print(f"[general-mistakes] output dir: {paths['out_dir']}")
    print(f"[general-mistakes] selected records: {len(records)}")
    print(f"[general-mistakes] provider={provider} model={model}")
    print(f"[general-mistakes] promotion threshold: {args.promotion_threshold}")
    print(f"[general-mistakes] proposal stale after: {args.proposal_stale_after}")
    print(f"[general-mistakes] max proposed types: {args.max_proposed_types}")

    counters = {
        "processed": 0,
        "skipped": 0,
        "ignored": 0,
        "atomic": 0,
        "promoted": 0,
        "discarded_proposals": 0,
        "capacity_prune_runs": 0,
        "errors": 0,
    }
    for idx, record in enumerate(records, start=1):
        process_one_record(
            args=args,
            idx=idx,
            total=len(records),
            record=record,
            client=client,
            model=model,
            temperature=temperature,
            taxonomy=taxonomy,
            processed_question_ids=processed_question_ids,
            paths=paths,
            counters=counters,
        )

    counters["tuple_dedupe_runs"] = dedupe_active_type_patterns(args, client, model, temperature, taxonomy)
    write_json(paths["taxonomy"], sanitize_output_obj(taxonomy))
    write_json(paths["mistake_set"], build_general_mistake_set(taxonomy, pattern_limit=args.max_tuples_per_type))
    print(
        "[general-mistakes] done | "
        f"processed={counters['processed']} skipped={counters['skipped']} "
        f"ignored={counters['ignored']} atomic={counters['atomic']} "
        f"promoted={counters['promoted']} discarded_proposals={counters['discarded_proposals']} "
        f"capacity_prune_runs={counters['capacity_prune_runs']} "
        f"tuple_dedupe_runs={counters['tuple_dedupe_runs']} errors={counters['errors']}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-expanding general mistake set from a QueryOS error memory bank. "
            "Outputs are isolated from the QueryOS runtime package."
        )
    )
    parser.add_argument("--error-bank", required=True, help="Path to error_bank.jsonl.")
    parser.add_argument("--out", required=True, help="Output directory for generated artifacts.")
    parser.add_argument("--config", default=str(CLEANED_QUERYOS_ROOT / "queryos_config.yaml"))
    parser.add_argument("--provider", choices=["openai", "vllm"], help="LLM provider override.")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--model", help="Model override. Defaults to YAML model.")
    parser.add_argument("--temperature", type=float, help="LLM temperature override.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--llm-timeout-seconds", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--question-id", type=int, action="append")
    parser.add_argument("--promotion-threshold", type=int, default=3)
    parser.add_argument(
        "--proposal-stale-after",
        type=int,
        default=50,
        help="Discard a proposed type if this many later processed samples do not vote for it. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-proposed-types",
        type=int,
        default=100,
        help="Start LLM cleanup when proposed type count exceeds this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--proposal-capacity-review-limit",
        type=int,
        default=200,
        help="Maximum number of proposed type candidates shown to the capacity cleanup LLM.",
    )
    parser.add_argument(
        "--tuple-dedupe-threshold",
        type=int,
        default=8,
        help=(
            "Run final intra-type tuple dedupe when an active type has at least this many raw pattern tuples. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-tuples-per-type",
        type=int,
        default=5,
        help="Maximum representative pattern tuples kept per active type in general_mistake_set.json.",
    )
    parser.add_argument(
        "--tuple-dedupe-review-limit",
        type=int,
        default=80,
        help="Maximum raw pattern tuples shown to the final tuple-dedupe LLM for one active type.",
    )
    parser.add_argument("--active-preview-limit", type=int, default=40)
    parser.add_argument("--proposed-preview-limit", type=int, default=80)
    parser.add_argument("--record-max-chars", type=int, default=16000)
    parser.add_argument("--reset", action="store_true", help="Rebuild outputs instead of resuming.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def build_output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    error_bank = Path(args.error_bank).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "error_bank": error_bank,
        "out_dir": out_dir,
        "atomic": out_dir / "atomic_mistakes.jsonl",
        "taxonomy": out_dir / "taxonomy.json",
        "mistake_set": out_dir / "general_mistake_set.json",
        "ignored": out_dir / "ignored_traces.jsonl",
        "run_log": out_dir / "run_log.jsonl",
    }


def generated_output_paths(paths: Dict[str, Path]) -> list[Path]:
    return [
        paths["atomic"],
        paths["taxonomy"],
        paths["mistake_set"],
        paths["ignored"],
        paths["run_log"],
    ]


def build_llm(args: argparse.Namespace, config: Dict[str, Any]) -> tuple[str, str, Any]:
    router_config = cfg_get(config, "llm_router", {})
    if not isinstance(router_config, dict):
        router_config = {}
    provider_default = "vllm" if router_config.get("enabled") else "openai"
    provider = pick(args.provider, config, "provider", provider_default)
    model = pick(args.model, config, "models.general_mistake_extractor", None) or pick(
        args.model,
        config,
        "model",
        "gpt-4.1-mini",
    )
    timeout = args.llm_timeout_seconds or cfg_get(config, "llm_timeout_seconds", None)
    client = create_llm_backend(
        provider=provider,
        api_key=pick(args.api_key, config, "api_key", None),
        base_url=pick(args.base_url, config, "base_url", None),
        model=model,
        router_config=router_config,
        timeout=timeout,
    )
    return str(provider), str(model), client


def process_one_record(
    *,
    args: argparse.Namespace,
    idx: int,
    total: int,
    record: Dict[str, Any],
    client: Any,
    model: str,
    temperature: float,
    taxonomy: Dict[str, Any],
    processed_question_ids: set[int],
    paths: Dict[str, Path],
    counters: Dict[str, int],
) -> None:
    question_id = safe_int(record.get("question_id"), default=-1)
    if question_id in processed_question_ids:
        counters["skipped"] += 1
        print(f"[{idx}/{total}] q{question_id} skip existing")
        return

    print(f"[{idx}/{total}] q{question_id} extract")
    try:
        result = extract_and_route_record(
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=args.max_tokens,
            record=record,
            taxonomy=taxonomy,
            active_preview_limit=args.active_preview_limit,
            proposed_preview_limit=args.proposed_preview_limit,
            record_max_chars=args.record_max_chars,
        )
        sample_idx = advance_taxonomy_sample(taxonomy)
        skip_info = normalized_skip_info(result)
        if skip_info["skip"]:
            discarded = discard_proposals(args, client, model, temperature, taxonomy, counters)
            append_jsonl(
                paths["ignored"],
                {
                    "source_question_id": question_id,
                    "db_id": str(record.get("db_id") or ""),
                    "difficulty": str(record.get("difficulty") or ""),
                    "reason": skip_info["reason"],
                },
            )
            write_json(paths["taxonomy"], sanitize_output_obj(taxonomy))
            write_json(paths["mistake_set"], build_general_mistake_set(taxonomy, pattern_limit=args.max_tuples_per_type))
            counters["discarded_proposals"] += len(discarded)
            append_run_log(
                paths["run_log"],
                question_id,
                "ignored_unreliable_trace",
                sample_idx=sample_idx,
                reason=skip_info["reason"],
                discarded_proposals=discarded,
            )
            counters["ignored"] += 1
            counters["processed"] += 1
            processed_question_ids.add(question_id)
            return

        atomic_items = normalize_atomic_items(result, record)
        if not atomic_items:
            discarded = discard_proposals(args, client, model, temperature, taxonomy, counters)
            write_json(paths["taxonomy"], sanitize_output_obj(taxonomy))
            write_json(paths["mistake_set"], build_general_mistake_set(taxonomy, pattern_limit=args.max_tuples_per_type))
            counters["discarded_proposals"] += len(discarded)
            append_run_log(
                paths["run_log"],
                question_id,
                "no_atomic_mistakes",
                sample_idx=sample_idx,
                discarded_proposals=discarded,
            )
            counters["processed"] += 1
            processed_question_ids.add(question_id)
            return

        for item in atomic_items:
            update = apply_taxonomy_routing(
                taxonomy=taxonomy,
                atomic=item,
                promotion_threshold=args.promotion_threshold,
            )
            item["routing_result"] = update
            append_jsonl(paths["atomic"], sanitize_output_obj(item))
            counters["atomic"] += 1
            if update.get("promoted"):
                counters["promoted"] += 1

        discarded = discard_proposals(args, client, model, temperature, taxonomy, counters)
        counters["discarded_proposals"] += len(discarded)
        write_json(paths["taxonomy"], sanitize_output_obj(taxonomy))
        write_json(paths["mistake_set"], build_general_mistake_set(taxonomy, pattern_limit=args.max_tuples_per_type))
        append_run_log(
            paths["run_log"],
            question_id,
            "processed",
            sample_idx=sample_idx,
            atomic_count=len(atomic_items),
            discarded_proposals=discarded,
        )
        counters["processed"] += 1
        processed_question_ids.add(question_id)
    except Exception as exc:
        counters["errors"] += 1
        err = safe_llm_error(exc)
        append_run_log(paths["run_log"], question_id, "error", error=err)
        print(f"[{idx}/{total}] q{question_id} error: {err}")
        if args.stop_on_error:
            raise


def discard_proposals(
    args: argparse.Namespace,
    client: Any,
    model: str,
    temperature: float,
    taxonomy: Dict[str, Any],
    counters: Dict[str, int],
) -> list[Dict[str, Any]]:
    discarded = prune_stale_proposals(taxonomy, stale_after=max(0, int(args.proposal_stale_after or 0)))
    discarded.extend(discard_excess_proposals(args, client, model, temperature, taxonomy, counters))
    return discarded


def dedupe_active_type_patterns(
    args: argparse.Namespace,
    client: Any,
    model: str,
    temperature: float,
    taxonomy: Dict[str, Any],
) -> int:
    threshold = max(0, int(args.tuple_dedupe_threshold or 0))
    if threshold <= 0:
        return 0
    max_patterns = max(1, int(args.max_tuples_per_type or 1))
    review_limit = max(max_patterns, int(args.tuple_dedupe_review_limit or 0))
    runs = 0
    for item in taxonomy.get("active_types", []):
        if not isinstance(item, dict):
            continue
        raw_patterns = normalize_pattern_tuples(item.get("pattern_tuples"))
        if len(raw_patterns) < threshold:
            continue
        try:
            deduped = dedupe_pattern_tuples(
                client=client,
                model=model,
                temperature=temperature,
                max_tokens=args.max_tokens,
                type_item=item,
                review_limit=review_limit,
                max_patterns=max_patterns,
            )
        except Exception as exc:
            print(f"[general-mistakes] tuple dedupe skipped for {item.get('id')}: {safe_llm_error(exc)}")
            continue
        if deduped:
            item["deduped_pattern_tuples"] = deduped
            runs += 1
            print(
                "[general-mistakes] tuple dedupe "
                f"{item.get('id')} | raw={len(raw_patterns)} representative={len(deduped)}"
            )
    return runs


def discard_excess_proposals(
    args: argparse.Namespace,
    client: Any,
    model: str,
    temperature: float,
    taxonomy: Dict[str, Any],
    counters: Dict[str, int],
) -> list[Dict[str, Any]]:
    max_proposed = max(0, int(args.max_proposed_types or 0))
    proposed_count = len(taxonomy.get("proposed_types", []))
    if max_proposed <= 0 or proposed_count <= max_proposed:
        return []
    target_drop_count = proposed_count - max_proposed
    drop_items = select_unnecessary_proposals(
        client=client,
        model=model,
        temperature=temperature,
        max_tokens=args.max_tokens,
        taxonomy=taxonomy,
        max_proposed_types=max_proposed,
        target_drop_count=target_drop_count,
        review_limit=max(target_drop_count, int(args.proposal_capacity_review_limit or 0)),
    )
    counters["capacity_prune_runs"] += 1
    discarded = drop_proposals_by_ids(taxonomy, drop_items)
    if discarded:
        print(
            "[general-mistakes] capacity prune discarded "
            f"{len(discarded)} proposed types ({len(taxonomy.get('proposed_types', []))}/{max_proposed} remain)"
        )
    return discarded


def append_run_log(path: Path, question_id: int, status: str, **extra: Any) -> None:
    payload = {"question_id": question_id, "status": status, "created_at": int(time.time())}
    payload.update(extra)
    append_jsonl(path, payload)


def normalized_skip_info(result: Dict[str, Any]) -> Dict[str, Any]:
    raw = result.get("skip_failure_trace")
    if isinstance(raw, bool):
        return {"skip": raw, "reason": "LLM marked this trace as unsafe for taxonomy learning."}
    if not isinstance(raw, dict):
        return {"skip": False, "reason": ""}
    return {
        "skip": bool(raw.get("skip")),
        "reason": clean_text(raw.get("reason")) or "Trace marked unsafe for taxonomy learning.",
    }
