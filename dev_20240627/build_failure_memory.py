from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CLEANED_QUERYOS_ROOT = REPO_ROOT / "cleaned_query_os"
if str(CLEANED_QUERYOS_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEANED_QUERYOS_ROOT))

from query_os.config import cfg_get, load_yaml_config, pick  # noqa: E402
from query_os.llm import create_chat_completion, create_llm_backend, safe_llm_error  # noqa: E402
from query_os.sql_agent import QueryOS, result_to_dict  # noqa: E402
from recheck_true_errors import recheck_record  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run QueryOS over dev.json, compare against each golden SQL, apply relaxed full-result "
            "rechecks, and build a JSONL failure-memory file for true errors."
        )
    )
    parser.add_argument("--dev-json", default=str(SCRIPT_DIR / "dev.json"))
    parser.add_argument(
        "--dataset-root",
        help=(
            "Dataset root containing dev_databases/. Defaults to the parent directory of --dev-json. "
            "Use this when the question JSON lives outside the database directory tree."
        ),
    )
    parser.add_argument("--config", default=str(CLEANED_QUERYOS_ROOT / "queryos_config.yaml"))
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "traces" / "failure_memory_runs"))
    parser.add_argument("--output-jsonl", default=str(SCRIPT_DIR / "failure_memory" / "error_bank.jsonl"))
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--provider", choices=["openai", "vllm"], help="LLM provider backend.")
    parser.add_argument("--model", help="Override QueryOS model from YAML.")
    parser.add_argument("--reason-model", help="Model used with --llm-reason. Defaults to QueryOS model.")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validation", choices=["off", "auto"])
    parser.add_argument("--limit", type=int, help="Maximum number of selected dev examples to run.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many selected examples before running.")
    parser.add_argument("--question-id", type=int, action="append", help="Run only these question_id values.")
    parser.add_argument("--db-id", action="append", help="Run only these db_id values.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of dev examples to run concurrently. Effective LLM concurrency is roughly "
            "workers * sql_writer.parallel_workers, plus planner/SDA/SVA/reason calls."
        ),
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between examples.")
    parser.add_argument("--live-trace", action="store_true", help="Show QueryOS live trace while batch runs.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip existing output records/results.")
    parser.add_argument(
        "--overwrite-results",
        dest="overwrite_results",
        action="store_true",
        default=True,
        help="Regenerate q*_result.json even if present. This is the default.",
    )
    parser.add_argument(
        "--no-overwrite-results",
        dest="overwrite_results",
        action="store_false",
        help="Reuse existing q*_result.json files when present.",
    )
    parser.add_argument("--llm-reason", action="store_true", help="Use an LLM to summarize true-error reasons.")
    parser.add_argument("--no-llm-reason", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-relaxed-recheck", action="store_true", help="Disable full SQL re-execution relaxed matching.")
    parser.add_argument("--relaxed-output-jsonl", help="Optional JSONL for mismatches filtered out by relaxed recheck.")
    parser.add_argument("--recheck-max-rows", type=int, default=2000, help="Rows to fetch per SQL during relaxed recheck.")
    parser.add_argument("--recheck-timeout", type=int, default=60, help="Seconds per SQL during relaxed recheck.")
    parser.add_argument("--recheck-max-projection-combinations", type=int, default=20000)
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the batch when one example raises an exception.")
    args = parser.parse_args()

    dev_json = Path(args.dev_json).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else dev_json.parent
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    relaxed_output_jsonl = Path(args.relaxed_output_jsonl).expanduser().resolve() if args.relaxed_output_jsonl else None
    results_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if relaxed_output_jsonl:
        relaxed_output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    config = load_yaml_config(args.config)
    reason_model = (
        args.reason_model
        or cfg_get(config, "models.reason_summarizer", None)
        or pick(args.model, config, "model", "gpt-4.1-mini")
    )

    samples = load_dev_samples(dev_json)
    samples = list(filter_samples(samples, question_ids=args.question_id, db_ids=args.db_id))
    if args.offset:
        samples = samples[args.offset :]
    if args.limit is not None:
        samples = samples[: args.limit]

    resume = not args.no_resume
    completed_failure_ids = load_existing_failure_ids(output_jsonl) if resume else set()
    if resume and relaxed_output_jsonl:
        completed_failure_ids.update(load_existing_failure_ids(relaxed_output_jsonl))

    print(f"[failure-memory] selected examples: {len(samples)}")
    print(f"[failure-memory] dev json: {dev_json}")
    print(f"[failure-memory] dataset root: {dataset_root}")
    print(f"[failure-memory] results dir: {results_dir}")
    print(f"[failure-memory] output jsonl: {output_jsonl}")
    if relaxed_output_jsonl:
        print(f"[failure-memory] relaxed output jsonl: {relaxed_output_jsonl}")
    print(f"[failure-memory] relaxed recheck: {not args.no_relaxed_recheck}")
    print(f"[failure-memory] llm reason: {bool(args.llm_reason and not args.no_llm_reason)}")
    print(f"[failure-memory] outer workers: {max(1, args.workers)}")
    print(f"[failure-memory] inner schema workers: {int(cfg_get(config, 'schema_discovery.parallel_workers', 1))}")
    print(f"[failure-memory] inner sql writers: {int(cfg_get(config, 'sql_writer.parallel_workers', 1))}")
    if args.live_trace and args.workers > 1:
        print("[failure-memory] warning: --live-trace with --workers > 1 can interleave terminal logs")

    counters = {"written": 0, "skipped": 0, "matched": 0, "relaxed_matched": 0, "failed": 0}
    print_lock = threading.Lock()
    write_lock = threading.Lock()
    workers = max(1, int(args.workers or 1))

    if workers == 1:
        for idx, sample in enumerate(samples, start=1):
            status = process_sample(
                args=args,
                config=config,
                dataset_root=dataset_root,
                results_dir=results_dir,
                output_jsonl=output_jsonl,
                relaxed_output_jsonl=relaxed_output_jsonl,
                sample=sample,
                idx=idx,
                total=len(samples),
                resume=resume,
                completed_failure_ids=completed_failure_ids,
                reason_model=reason_model,
                print_lock=print_lock,
                write_lock=write_lock,
            )
            update_counters(counters, status)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    process_sample,
                    args=args,
                    config=config,
                    dataset_root=dataset_root,
                    results_dir=results_dir,
                    output_jsonl=output_jsonl,
                    relaxed_output_jsonl=relaxed_output_jsonl,
                    sample=sample,
                    idx=idx,
                    total=len(samples),
                    resume=resume,
                    completed_failure_ids=completed_failure_ids,
                    reason_model=reason_model,
                    print_lock=print_lock,
                    write_lock=write_lock,
                )
                for idx, sample in enumerate(samples, start=1)
            ]
            for future in as_completed(futures):
                status = future.result()
                update_counters(counters, status)

    print(
        "[failure-memory] done | "
        f"matched={counters['matched']} relaxed_filtered={counters['relaxed_matched']} "
        f"mismatched_or_error={counters['failed']} "
        f"written={counters['written']} skipped={counters['skipped']}"
    )
    return 0


def build_queryos_agent(args: argparse.Namespace, config: Dict[str, Any]) -> QueryOS:
    trace_max_chars = cfg_get(config, "trace.max_chars", None)
    if trace_max_chars is not None:
        trace_max_chars = int(trace_max_chars)
    if trace_max_chars == 0:
        trace_max_chars = None
    router_config = cfg_get(config, "llm_router", {})
    if not isinstance(router_config, dict):
        router_config = {}
    provider_default = "vllm" if router_config.get("enabled") else "openai"

    return QueryOS(
        provider=pick(args.provider, config, "provider", provider_default),
        api_key=pick(args.api_key, config, "api_key", None),
        model=pick(args.model, config, "model", "gpt-4.1-mini"),
        base_url=pick(args.base_url, config, "base_url", None),
        llm_router_config=router_config,
        llm_timeout=cfg_get(config, "llm_timeout_seconds", None),
        planner_model=cfg_get(config, "models.planner", None),
        schema_model=cfg_get(config, "models.schema_discovery", None),
        sql_model=cfg_get(config, "models.sql_writer", None),
        validator_model=cfg_get(config, "models.sql_validator", None),
        temperature=float(cfg_get(config, "temperature", 0.2)),
        max_steps=int(pick(args.max_steps, config, "workflow.max_steps", 8)),
        planner_max_tokens=int(cfg_get(config, "planner.max_tokens", 1024)),
        schema_max_tokens=int(cfg_get(config, "schema_discovery.max_tokens", 4096)),
        schema_max_turns=int(cfg_get(config, "schema_discovery.max_turns", 6)),
        schema_max_tool_calls_per_turn=int(cfg_get(config, "schema_discovery.max_tool_calls_per_turn", 4)),
        schema_read_table_summary_max_cols=int(cfg_get(config, "schema_discovery.read_table_summary_max_cols", 30)),
        schema_trace_column_preview_limit=int(cfg_get(config, "schema_discovery.trace_column_preview_limit", 8)),
        schema_parallel_workers=int(cfg_get(config, "schema_discovery.parallel_workers", 1)),
        sql_max_tokens=int(cfg_get(config, "sql_writer.max_tokens", 4096)),
        sql_max_turns=int(cfg_get(config, "sql_writer.max_turns", 8)),
        sql_parallel_workers=int(cfg_get(config, "sql_writer.parallel_workers", 1)),
        sql_chatgroup_enabled=bool(cfg_get(config, "sql_writer.chatgroup.enabled", True)),
        sql_chatgroup_max_rounds=int(cfg_get(config, "sql_writer.chatgroup.max_rounds", 2)),
        sql_consensus_require_same_columns=bool(cfg_get(config, "sql_writer.consensus.require_same_columns", False)),
        validator_max_tokens=int(cfg_get(config, "sql_validator.max_tokens", 2048)),
        live_trace=False,
        trace_style=cfg_get(config, "trace.style", "pretty"),
        trace_color=cfg_get(config, "trace.color", "auto"),
        trace_max_chars=trace_max_chars,
        trace_sql_preview_rows=int(cfg_get(config, "trace.sql_preview_rows", 3)),
        trace_gold_preview_rows=int(cfg_get(config, "trace.gold_preview_rows", 3)),
        trace_result_cell_max_width=int(cfg_get(config, "trace.result_cell_max_width", 32)),
        state_view=cfg_get(config, "trace.state_view", "diff"),
        planner_context=cfg_get(config, "planner.context", "dispatch"),
        validation_mode=pick(args.validation, config, "workflow.validation", "auto"),
        auto_finish_on_sql=bool(cfg_get(config, "workflow.auto_finish_on_sql", False)),
    )


def process_sample(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    dataset_root: Path,
    results_dir: Path,
    output_jsonl: Path,
    relaxed_output_jsonl: Optional[Path],
    sample: Dict[str, Any],
    idx: int,
    total: int,
    resume: bool,
    completed_failure_ids: Set[int],
    reason_model: str,
    print_lock: threading.Lock,
    write_lock: threading.Lock,
) -> str:
    question_id = int(sample["question_id"])
    db_id = str(sample["db_id"])
    result_path = results_dir / f"q{question_id}_result.json"
    trace_path = results_dir / f"q{question_id}_trace.json"

    if resume and question_id in completed_failure_ids:
        log(print_lock, f"[{idx}/{total}] q{question_id} skip existing failure-memory record")
        return "skipped"

    log(print_lock, f"[{idx}/{total}] q{question_id} db={db_id}")

    try:
        if resume and result_path.exists() and not args.overwrite_results:
            result_doc = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            agent = build_queryos_agent(args, config)
            db_path = find_sqlite_path(dataset_root, db_id)
            metadata_path = dataset_root / "dev_databases" / db_id / "database_description"
            result = agent.generate(
                question=str(sample.get("question") or ""),
                db_path=str(db_path),
                schema_metadata_path=str(metadata_path) if metadata_path.exists() else None,
                external_knowledge=str(sample.get("evidence") or ""),
                db_id=db_id,
                live_trace=args.live_trace,
                trace_json_path=str(trace_path),
                golden_sql=str(sample.get("SQL") or ""),
            )
            result_doc = result_to_dict(result)
            result_path.write_text(
                json.dumps(result_doc, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        if result_doc.get("gold_match") is True:
            log(print_lock, f"[{idx}/{total}] q{question_id} match=true")
            status = "matched"
        else:
            relaxed_recheck = None
            if not args.no_relaxed_recheck:
                relaxed_recheck = recheck_record(
                    record={
                        "question_id": question_id,
                        "db_id": db_id,
                        "predicted_sql": result_doc.get("final_sql", ""),
                        "gold_sql": sample.get("SQL", ""),
                    },
                    dataset_root=dataset_root,
                    max_rows=args.recheck_max_rows,
                    timeout=args.recheck_timeout,
                    preview_rows=10,
                    max_projection_combinations=args.recheck_max_projection_combinations,
                )
                if not relaxed_recheck.get("true_error"):
                    record = build_failure_record(
                        sample=sample,
                        result_doc=result_doc,
                        error_reason=deterministic_error_reason(result_doc, relaxed_recheck),
                        result_path=result_path,
                        trace_path=trace_path,
                    )
                    record["relaxed_recheck"] = relaxed_recheck
                    if relaxed_output_jsonl:
                        append_jsonl(relaxed_output_jsonl, record, lock=write_lock)
                    log(
                        print_lock,
                        f"[{idx}/{total}] q{question_id} relaxed_non_error "
                        f"cluster={relaxed_recheck.get('cluster')}",
                    )
                    status = "relaxed_matched"
                    if args.sleep > 0:
                        time.sleep(args.sleep)
                    return status

            if args.llm_reason and not args.no_llm_reason:
                router_config = cfg_get(config, "llm_router", {})
                if not isinstance(router_config, dict):
                    router_config = {}
                provider_default = "vllm" if router_config.get("enabled") else "openai"
                reason_client = create_llm_backend(
                    provider=pick(args.provider, config, "provider", provider_default),
                    api_key=pick(args.api_key, config, "api_key", None),
                    base_url=pick(args.base_url, config, "base_url", None),
                    model=pick(args.model, config, "model", "gpt-4.1-mini"),
                    router_config=router_config,
                    timeout=cfg_get(config, "llm_timeout_seconds", None),
                )
                error_reason = summarize_error_reason(reason_client, reason_model, sample, result_doc)
            else:
                error_reason = deterministic_error_reason(result_doc, relaxed_recheck)

            record = build_failure_record(
                sample=sample,
                result_doc=result_doc,
                error_reason=error_reason,
                result_path=result_path,
                trace_path=trace_path,
            )
            if relaxed_recheck:
                record["relaxed_recheck"] = relaxed_recheck
            append_jsonl(output_jsonl, record, lock=write_lock)
            cluster_text = f" cluster={relaxed_recheck.get('cluster')}" if relaxed_recheck else ""
            log(print_lock, f"[{idx}/{total}] q{question_id} true_error{cluster_text} | wrote failure")
            status = "written"

    except KeyboardInterrupt:
        raise
    except Exception as exc:
        error_reason = f"QueryOS run or failure summarization failed before comparison: {safe_llm_error(exc)}"
        record = build_failure_record(
            sample=sample,
            result_doc={
                "ok": False,
                "final_sql": "",
                "columns": [],
                "rows": [],
                "gold_sql": sample.get("SQL", ""),
                "gold_match": False,
                "gold_comparison": {"run_error": safe_llm_error(exc)},
            },
            error_reason=error_reason,
            result_path=result_path,
            trace_path=trace_path,
        )
        append_jsonl(output_jsonl, record, lock=write_lock)
        log(print_lock, f"[{idx}/{total}] q{question_id} error | wrote run_error reason: {safe_llm_error(exc)}")
        if args.stop_on_error:
            raise
        status = "error_written"

    if args.sleep > 0:
        time.sleep(args.sleep)
    return status


def update_counters(counters: Dict[str, int], status: str) -> None:
    if status == "matched":
        counters["matched"] += 1
    elif status == "skipped":
        counters["skipped"] += 1
    elif status == "relaxed_matched":
        counters["relaxed_matched"] += 1
    elif status == "written":
        counters["written"] += 1
        counters["failed"] += 1
    elif status == "error_written":
        counters["written"] += 1
        counters["failed"] += 1


def log(lock: threading.Lock, message: str) -> None:
    with lock:
        print(message, flush=True)


def load_dev_samples(dev_json: Path) -> List[Dict[str, Any]]:
    with dev_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("dev.json root must be a list")
    return [item for item in data if isinstance(item, dict)]


def filter_samples(
    samples: Iterable[Dict[str, Any]],
    *,
    question_ids: Optional[List[int]],
    db_ids: Optional[List[str]],
) -> Iterable[Dict[str, Any]]:
    qid_set = set(question_ids or [])
    db_set = set(db_ids or [])
    for sample in samples:
        if qid_set and int(sample.get("question_id", -1)) not in qid_set:
            continue
        if db_set and str(sample.get("db_id", "")) not in db_set:
            continue
        yield sample


def find_sqlite_path(dataset_root: Path, db_id: str) -> Path:
    db_dir = dataset_root / "dev_databases" / db_id
    preferred = db_dir / f"{db_id}.sqlite"
    if preferred.exists():
        return preferred
    matches = sorted(db_dir.glob("*.sqlite"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"SQLite database not found for db_id={db_id}: {db_dir}")


def load_existing_failure_ids(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    out: Set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out.add(int(obj["question_id"]))
            except Exception:
                continue
    return out


def summarize_error_reason(
    client: Any,
    model: str,
    sample: Dict[str, Any],
    result_doc: Dict[str, Any],
) -> str:
    predicted_rows = result_doc.get("rows") or []
    gold_result = result_doc.get("gold_result") or {}
    gold_body = gold_result.get("result") or {}
    payload = {
        "question_id": sample.get("question_id"),
        "db_id": sample.get("db_id"),
        "question": sample.get("question"),
        "evidence": sample.get("evidence", ""),
        "gold_sql": sample.get("SQL", ""),
        "predicted_sql": result_doc.get("final_sql", ""),
        "predicted_result": {
            "columns": result_doc.get("columns", []),
            "row_count": len(predicted_rows),
            "rows_preview": predicted_rows[:10],
        },
        "gold_result": {
            "columns": gold_body.get("columns", []),
            "row_count": len(gold_body.get("rows") or []),
            "rows_preview": (gold_body.get("rows") or [])[:10],
            "error": gold_result.get("error", ""),
        },
        "comparison": result_doc.get("gold_comparison", {}),
        "validation_attempts": result_doc.get("validation_attempts", [])[-3:],
    }
    response = create_chat_completion(
        client,
        role="reason_summarizer",
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize text-to-SQL failures for a retrieval memory. "
                    "Write one unified error_reason paragraph in English. "
                    "Do not output a category label, bullets, JSON, or remediation checklist. "
                    "Focus on the concrete semantic difference between predicted SQL/result and golden SQL/result."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            },
        ],
        temperature=0.0,
        max_tokens=700,
    )
    content = response.choices[0].message.content or ""
    return " ".join(content.split())


def deterministic_error_reason(result_doc: Dict[str, Any], relaxed_recheck: Optional[Dict[str, Any]] = None) -> str:
    comparison = result_doc.get("gold_comparison") or {}
    base = (
        "The predicted SQL result does not match the golden SQL result. "
        f"Predicted row count is {comparison.get('predicted_row_count')} and gold row count is "
        f"{comparison.get('gold_row_count')}; predicted columns are {comparison.get('predicted_columns')} "
        f"and gold columns are {comparison.get('gold_columns')}. "
    )
    if relaxed_recheck:
        return (
            base
            + f"Relaxed full-result recheck cluster: {relaxed_recheck.get('cluster')}; "
            + f"reason: {relaxed_recheck.get('reason')}."
        )
    return base + "Inspect the predicted SQL and golden SQL to identify the semantic mismatch."


def build_failure_record(
    *,
    sample: Dict[str, Any],
    result_doc: Dict[str, Any],
    error_reason: str,
    result_path: Path,
    trace_path: Path,
) -> Dict[str, Any]:
    predicted_rows = result_doc.get("rows") or []
    gold_result = result_doc.get("gold_result") or {}
    gold_body = gold_result.get("result") or {}
    return {
        "question_id": sample.get("question_id"),
        "db_id": sample.get("db_id"),
        "difficulty": sample.get("difficulty", ""),
        "question": sample.get("question", ""),
        "evidence": sample.get("evidence", ""),
        "predicted_sql": result_doc.get("final_sql", ""),
        "gold_sql": sample.get("SQL", ""),
        "predicted_columns": result_doc.get("columns", []),
        "predicted_rows_preview": predicted_rows[:10],
        "gold_columns": gold_body.get("columns", []),
        "gold_rows_preview": (gold_body.get("rows") or [])[:10],
        "gold_match": result_doc.get("gold_match"),
        "gold_comparison": result_doc.get("gold_comparison", {}),
        "error_reason": error_reason,
        "result_json": str(result_path),
        "trace_json": str(trace_path),
    }


def append_jsonl(path: Path, record: Dict[str, Any], lock: Optional[threading.Lock] = None) -> None:
    if lock is None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
