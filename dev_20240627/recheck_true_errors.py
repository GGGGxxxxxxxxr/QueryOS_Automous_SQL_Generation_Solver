from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CLEANED_QUERYOS_ROOT = REPO_ROOT / "cleaned_query_os"
if str(CLEANED_QUERYOS_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEANED_QUERYOS_ROOT))

from query_os.sqlite_executor import WRITE_ACTIONS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-execute predicted_sql and gold_sql from an error_bank.jsonl, then separate true "
            "semantic errors from display-only mismatches such as extra columns or irrelevant row order."
        )
    )
    parser.add_argument("--error-bank", default=str(SCRIPT_DIR / "failure_memory" / "error_bank.jsonl"))
    parser.add_argument(
        "--dataset-root",
        default=str(SCRIPT_DIR),
        help="Dataset root containing dev_databases/. Defaults to this dev_20240627 directory.",
    )
    parser.add_argument(
        "--output-jsonl",
        help="True-error JSONL output. Defaults to <error-bank-stem>_true_error.jsonl.",
    )
    parser.add_argument(
        "--resolved-jsonl",
        help="Optional JSONL for relaxed matches that are not counted as true errors.",
    )
    parser.add_argument(
        "--summary-json",
        help="Cluster summary output. Defaults to <error-bank-stem>_true_error_summary.json.",
    )
    parser.add_argument("--question-id", type=int, action="append", help="Only recheck these question IDs.")
    parser.add_argument("--limit", type=int, help="Maximum number of records to recheck after filters.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=2000, help="0 means fetch all rows.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1, help="Number of records to recheck concurrently.")
    parser.add_argument("--preview-rows", type=int, default=10)
    parser.add_argument("--max-projection-combinations", type=int, default=20000)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print per-record progress every N records. 0 disables progress logs.",
    )
    args = parser.parse_args()

    error_bank = Path(args.error_bank).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else default_output_path(error_bank)
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else default_summary_path(error_bank)
    resolved_jsonl = Path(args.resolved_jsonl).expanduser().resolve() if args.resolved_jsonl else None

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    if resolved_jsonl:
        resolved_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records = list(load_jsonl(error_bank))
    if args.question_id:
        wanted = set(args.question_id)
        records = [record for record in records if int(record.get("question_id", -1)) in wanted]
    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]

    counters: Dict[str, int] = defaultdict(int)
    clusters: Dict[str, int] = defaultdict(int)
    resolved_clusters: Dict[str, int] = defaultdict(int)
    print(f"[recheck] selected records: {len(records)}")
    print(f"[recheck] workers: {max(1, int(args.workers or 1))}")

    with output_jsonl.open("w", encoding="utf-8") as true_handle:
        resolved_handle = resolved_jsonl.open("w", encoding="utf-8") if resolved_jsonl else None
        try:
            run_recheck_pool(
                records=records,
                args=args,
                dataset_root=dataset_root,
                true_handle=true_handle,
                resolved_handle=resolved_handle,
                counters=counters,
                clusters=clusters,
                resolved_clusters=resolved_clusters,
            )
        finally:
            if resolved_handle:
                resolved_handle.close()

    summary = {
        "error_bank": str(error_bank),
        "dataset_root": str(dataset_root),
        "output_jsonl": str(output_jsonl),
        "resolved_jsonl": str(resolved_jsonl) if resolved_jsonl else "",
        "processed": counters["processed"],
        "true_error": counters["true_error"],
        "relaxed_non_error": counters["relaxed_non_error"],
        "true_error_clusters": dict(sorted(clusters.items())),
        "relaxed_non_error_clusters": dict(sorted(resolved_clusters.items())),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[recheck] true errors: {output_jsonl}")
    if resolved_jsonl:
        print(f"[recheck] relaxed non-errors: {resolved_jsonl}")
    print(f"[recheck] summary: {summary_json}")
    return 0


def run_recheck_pool(
    *,
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    dataset_root: Path,
    true_handle: Any,
    resolved_handle: Optional[Any],
    counters: Dict[str, int],
    clusters: Dict[str, int],
    resolved_clusters: Dict[str, int],
) -> None:
    total = len(records)
    workers = max(1, int(args.workers or 1))
    next_pos = 0
    in_flight: Dict[Any, Tuple[int, Dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while next_pos < total or in_flight:
            while next_pos < total and len(in_flight) < workers:
                idx = next_pos + 1
                record = records[next_pos]
                next_pos += 1
                if should_print_progress(idx, total, args.progress_every):
                    print(
                        f"[recheck] start {idx}/{total} qid={record.get('question_id')} db={record.get('db_id')}",
                        flush=True,
                    )
                future = pool.submit(
                    recheck_record,
                    record=record,
                    dataset_root=dataset_root,
                    max_rows=args.max_rows,
                    timeout=args.timeout,
                    preview_rows=args.preview_rows,
                    max_projection_combinations=args.max_projection_combinations,
                )
                in_flight[future] = (idx, record)

            if not in_flight:
                continue

            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                idx, record = in_flight.pop(future)
                analysis = future.result()
                write_recheck_result(
                    idx=idx,
                    total=total,
                    record=record,
                    analysis=analysis,
                    args=args,
                    true_handle=true_handle,
                    resolved_handle=resolved_handle,
                    counters=counters,
                    clusters=clusters,
                    resolved_clusters=resolved_clusters,
                )


def write_recheck_result(
    *,
    idx: int,
    total: int,
    record: Dict[str, Any],
    analysis: Dict[str, Any],
    args: argparse.Namespace,
    true_handle: Any,
    resolved_handle: Optional[Any],
    counters: Dict[str, int],
    clusters: Dict[str, int],
    resolved_clusters: Dict[str, int],
) -> None:
    annotated = dict(record)
    annotated["relaxed_recheck"] = analysis
    counters["processed"] += 1

    cluster = str(analysis.get("cluster") or "unknown")
    if analysis.get("true_error"):
        counters["true_error"] += 1
        clusters[cluster] += 1
        true_handle.write(json.dumps(annotated, ensure_ascii=False, default=str) + "\n")
        true_handle.flush()
    else:
        counters["relaxed_non_error"] += 1
        resolved_clusters[cluster] += 1
        if resolved_handle:
            resolved_handle.write(json.dumps(annotated, ensure_ascii=False, default=str) + "\n")
            resolved_handle.flush()

    if should_print_progress(idx, total, args.progress_every):
        print(
            f"[recheck] done {idx}/{total} qid={record.get('question_id')} cluster={cluster} "
            f"true_error={bool(analysis.get('true_error'))}; "
            f"true={counters['true_error']} relaxed={counters['relaxed_non_error']}",
            flush=True,
        )


def recheck_record(
    *,
    record: Dict[str, Any],
    dataset_root: Path,
    max_rows: int,
    timeout: int,
    preview_rows: int,
    max_projection_combinations: int,
) -> Dict[str, Any]:
    db_id = str(record.get("db_id") or "")
    db_path = find_db_path(dataset_root, db_id)
    predicted_sql = str(record.get("predicted_sql") or "")
    gold_sql = str(record.get("gold_sql") or "")
    predicted = execute_sql(db_path, predicted_sql, max_rows=max_rows, timeout=timeout)
    gold = execute_sql(db_path, gold_sql, max_rows=max_rows, timeout=timeout)

    base = {
        "db_path": str(db_path),
        "predicted_ok": predicted["ok"],
        "gold_ok": gold["ok"],
        "predicted_error": predicted.get("error", ""),
        "gold_error": gold.get("error", ""),
        "predicted_columns": predicted.get("columns", []),
        "gold_columns": gold.get("columns", []),
        "predicted_row_count": len(predicted.get("rows", [])),
        "gold_row_count": len(gold.get("rows", [])),
        "predicted_truncated": predicted.get("truncated", False),
        "gold_truncated": gold.get("truncated", False),
        "gold_has_order_by": has_order_by(gold_sql),
        "predicted_preview": predicted.get("rows", [])[:preview_rows],
        "gold_preview": gold.get("rows", [])[:preview_rows],
    }

    if not predicted["ok"]:
        return true_cluster(base, "predicted_execution_error", "Predicted SQL failed during re-execution.")
    if not gold["ok"]:
        return true_cluster(base, "gold_execution_error", "Gold SQL failed during re-execution.")
    if predicted.get("truncated") or gold.get("truncated"):
        return true_cluster(base, "recheck_row_limit_reached", "Recheck hit max row limit, so relaxed comparison is unsafe.")

    pred_rows = predicted["rows"]
    gold_rows = gold["rows"]
    gold_ordered = has_order_by(gold_sql)
    percent_context = has_percent_context(record, predicted["columns"], gold["columns"])
    base["percent_context"] = percent_context

    if rows_equal(pred_rows, gold_rows):
        return relaxed_cluster(base, "exact_full_match_after_reexecution", "Full rows match after re-execution.")

    if rows_unordered_equal(pred_rows, gold_rows):
        if not gold_ordered:
            return relaxed_cluster(
                base,
                "row_order_only_gold_has_no_order_by",
                "Rows only differ by order, and gold SQL has no ORDER BY.",
            )
        return true_cluster(base, "row_order_mismatch_gold_has_order_by", "Rows match unordered, but gold SQL has ORDER BY.")

    projection = find_matching_projection(
        pred_rows=pred_rows,
        gold_rows=gold_rows,
        pred_columns=predicted["columns"],
        gold_columns=gold["columns"],
        allow_unordered=not gold_ordered,
        max_combinations=max_projection_combinations,
        percent_context=False,
    )
    if projection:
        pred_width = len(predicted["columns"])
        gold_width = len(gold["columns"])
        if pred_width > gold_width:
            cluster = "extra_columns_only"
            if projection["match_type"] == "unordered":
                cluster = "extra_columns_and_row_order_only"
            return relaxed_cluster(base, cluster, "Gold output is a projection of predicted output.", projection)
        cluster = "column_display_order_only"
        if projection["match_type"] == "unordered":
            cluster = "column_display_order_and_row_order_only"
        return relaxed_cluster(base, cluster, "Predicted columns can be reordered to match gold output.", projection)

    if percent_context:
        if rows_equal_scaled(pred_rows, gold_rows):
            return relaxed_cluster(
                base,
                "percentage_scale_only",
                "Rows differ only by percentage scale, e.g. 0.12 versus 12 percent.",
                details=percent_payload(),
            )
        if not gold_ordered and rows_unordered_equal_scaled(pred_rows, gold_rows):
            return relaxed_cluster(
                base,
                "percentage_scale_and_row_order_only",
                "Rows differ only by percentage scale and row order; gold SQL has no ORDER BY.",
                details=percent_payload(),
            )
        projection = find_matching_projection(
            pred_rows=pred_rows,
            gold_rows=gold_rows,
            pred_columns=predicted["columns"],
            gold_columns=gold["columns"],
            allow_unordered=not gold_ordered,
            max_combinations=max_projection_combinations,
            percent_context=True,
        )
        if projection:
            pred_width = len(predicted["columns"])
            gold_width = len(gold["columns"])
            if pred_width > gold_width:
                cluster = "extra_columns_and_percentage_scale_only"
                if projection["match_type"] == "unordered":
                    cluster = "extra_columns_percentage_scale_and_row_order_only"
                return relaxed_cluster(
                    base,
                    cluster,
                    "Gold output is a percentage-scale projection of predicted output.",
                    projection,
                    details=percent_payload(),
                )
            cluster = "column_order_and_percentage_scale_only"
            if projection["match_type"] == "unordered":
                cluster = "column_order_percentage_scale_and_row_order_only"
            return relaxed_cluster(
                base,
                cluster,
                "Predicted columns can be reordered and percentage-scaled to match gold output.",
                projection,
                details=percent_payload(),
            )

    if len(pred_rows) != len(gold_rows):
        return true_cluster(base, "row_count_mismatch", "Full row counts differ after re-execution.")
    if row_width(pred_rows) != row_width(gold_rows):
        return true_cluster(base, "output_shape_mismatch", "Column counts differ and no matching projection was found.")
    return true_cluster(base, "value_mismatch", "Rows have the same shape but different values.")


def execute_sql(db_path: Path, sql: str, *, max_rows: int, timeout: int) -> Dict[str, Any]:
    sql = sql.strip()
    if not sql:
        return {"ok": False, "columns": [], "rows": [], "error": "empty SQL", "truncated": False}
    conn: Optional[sqlite3.Connection] = None
    timer: Optional[threading.Timer] = None
    deadline = time.monotonic() + max(1, timeout)
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")
        conn.set_authorizer(read_only_authorizer)
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        cursor = conn.cursor()
        timer = threading.Timer(timeout, conn.interrupt)
        timer.daemon = True
        timer.start()
        cursor.execute(sql)
        if cursor.description is None:
            return {"ok": False, "columns": [], "rows": [], "error": "query returned no rows", "truncated": False}
        columns = [desc[0] for desc in cursor.description]
        if max_rows and max_rows > 0:
            raw_rows = cursor.fetchmany(max_rows + 1)
            truncated = len(raw_rows) > max_rows
            raw_rows = raw_rows[:max_rows]
        else:
            raw_rows = cursor.fetchall()
            truncated = False
        rows = [list(row) for row in raw_rows]
        return {"ok": True, "columns": columns, "rows": rows, "error": "", "truncated": truncated}
    except Exception as exc:
        error = str(exc)
        if time.monotonic() > deadline and "interrupted" in error.lower():
            error = f"SQL execution timeout after {timeout}s"
        return {"ok": False, "columns": [], "rows": [], "error": error, "truncated": False}
    finally:
        if timer:
            timer.cancel()
        if conn:
            conn.set_progress_handler(None, 0)
            conn.close()


def read_only_authorizer(action: int, arg1: str, arg2: str, dbname: str, source: str) -> int:
    if action in WRITE_ACTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def should_print_progress(idx: int, total: int, progress_every: int) -> bool:
    if progress_every <= 0:
        return False
    return idx == 1 or idx == total or idx % progress_every == 0


def find_matching_projection(
    *,
    pred_rows: List[List[Any]],
    gold_rows: List[List[Any]],
    pred_columns: List[str],
    gold_columns: List[str],
    allow_unordered: bool,
    max_combinations: int,
    percent_context: bool,
) -> Optional[Dict[str, Any]]:
    pred_width = len(pred_columns)
    gold_width = len(gold_columns)
    if gold_width <= 0 or pred_width < gold_width:
        return None

    checked = 0
    for indices in permutations(range(pred_width), gold_width):
        checked += 1
        if checked > max_combinations:
            return None
        projected = project_rows(pred_rows, indices)
        if rows_equal(projected, gold_rows) or (percent_context and rows_equal_scaled(projected, gold_rows)):
            return projection_payload(indices, pred_columns, checked, "exact")
        if allow_unordered and (
            rows_unordered_equal(projected, gold_rows)
            or (percent_context and rows_unordered_equal_scaled(projected, gold_rows))
        ):
            return projection_payload(indices, pred_columns, checked, "unordered")
    return None


def projection_payload(indices: Tuple[int, ...], columns: List[str], checked: int, match_type: str) -> Dict[str, Any]:
    return {
        "projection_indices": list(indices),
        "projection_columns": [columns[idx] for idx in indices],
        "projection_checked": checked,
        "match_type": match_type,
    }


def project_rows(rows: List[List[Any]], indices: Tuple[int, ...]) -> List[List[Any]]:
    return [[row[idx] for idx in indices] for row in rows]


def rows_equal(left: List[List[Any]], right: List[List[Any]]) -> bool:
    return canonical_rows(left) == canonical_rows(right)


def rows_unordered_equal(left: List[List[Any]], right: List[List[Any]]) -> bool:
    return Counter(canonical_rows(left)) == Counter(canonical_rows(right))


def rows_equal_scaled(left: List[List[Any]], right: List[List[Any]]) -> bool:
    if len(left) != len(right):
        return False
    return all(row_equal_scaled(left_row, right_row) for left_row, right_row in zip(left, right))


def rows_unordered_equal_scaled(left: List[List[Any]], right: List[List[Any]]) -> bool:
    if len(left) != len(right):
        return False
    unmatched = list(right)
    for left_row in left:
        match_idx = None
        for idx, right_row in enumerate(unmatched):
            if row_equal_scaled(left_row, right_row):
                match_idx = idx
                break
        if match_idx is None:
            return False
        unmatched.pop(match_idx)
    return True


def row_equal_scaled(left: List[Any], right: List[Any]) -> bool:
    if len(left) != len(right):
        return False
    return all(cell_equal_scaled(left_cell, right_cell) for left_cell, right_cell in zip(left, right))


def cell_equal_scaled(left: Any, right: Any) -> bool:
    if canonical_cell(left) == canonical_cell(right):
        return True
    if is_number(left) and is_number(right):
        left_num = float(left)
        right_num = float(right)
        return numeric_close(left_num * 100.0, right_num) or numeric_close(left_num, right_num * 100.0)
    return False


def numeric_close(left: float, right: float) -> bool:
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    return abs(left - right) <= max(1e-6, 1e-6 * max(abs(left), abs(right)))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def canonical_rows(rows: List[List[Any]]) -> List[Tuple[Any, ...]]:
    return [tuple(canonical_cell(cell) for cell in row) for row in rows]


def canonical_cell(cell: Any) -> Any:
    if cell is None:
        return ("null", None)
    if isinstance(cell, bool):
        return ("bool", bool(cell))
    if isinstance(cell, (int, float)):
        value = float(cell)
        if math.isfinite(value):
            return ("num", round(value, 10))
    if isinstance(cell, bytes):
        return ("bytes", cell.hex())
    return ("text", str(cell))


def row_width(rows: List[List[Any]]) -> int:
    if not rows:
        return 0
    return len(rows[0])


def has_order_by(sql: str) -> bool:
    return bool(re.search(r"\border\s+by\b", sql or "", flags=re.IGNORECASE))


def has_percent_context(record: Dict[str, Any], predicted_columns: List[str], gold_columns: List[str]) -> bool:
    text = " ".join(
        [
            str(record.get("question") or ""),
            str(record.get("evidence") or ""),
            " ".join(str(col) for col in predicted_columns),
            " ".join(str(col) for col in gold_columns),
        ]
    )
    return bool(
        re.search(
            r"(%|\bpercent\b|\bpercentage\b|\bpct\b)",
            text,
            flags=re.IGNORECASE,
        )
    )


def percent_payload() -> Dict[str, Any]:
    return {"scale_equivalence": "0-1_fraction_vs_0-100_percent"}


def true_cluster(base: Dict[str, Any], cluster: str, reason: str) -> Dict[str, Any]:
    return {**base, "true_error": True, "cluster": cluster, "reason": reason}


def relaxed_cluster(
    base: Dict[str, Any],
    cluster: str,
    reason: str,
    projection: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {**base, "true_error": False, "cluster": cluster, "reason": reason}
    if projection:
        payload["projection"] = projection
    if details:
        payload.update(details)
    return payload


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def find_db_path(dataset_root: Path, db_id: str) -> Path:
    candidates = [
        dataset_root / "dev_databases" / db_id / f"{db_id}.sqlite",
        dataset_root / "bird_mini_dev" / "dev_databases" / db_id / f"{db_id}.sqlite",
        dataset_root / "bird_mini_dev" / "mini_dev_databases" / db_id / f"{db_id}.sqlite",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(dataset_root.rglob(f"{db_id}.sqlite"))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"Could not find SQLite database for db_id={db_id} under {dataset_root}")


def default_output_path(error_bank: Path) -> Path:
    stem = error_bank.name[:-6] if error_bank.name.endswith(".jsonl") else error_bank.stem
    return error_bank.with_name(f"{stem}_true_error.jsonl")


def default_summary_path(error_bank: Path) -> Path:
    stem = error_bank.name[:-6] if error_bank.name.endswith(".jsonl") else error_bank.stem
    return error_bank.with_name(f"{stem}_true_error_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
