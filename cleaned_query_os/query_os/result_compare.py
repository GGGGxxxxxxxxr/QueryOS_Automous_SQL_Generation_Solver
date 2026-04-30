from __future__ import annotations

from itertools import permutations
import json
from typing import Any, Dict, List


def compare_sql_execution_results(
    predicted: Dict[str, Any],
    gold: Dict[str, Any],
    *,
    relaxed: bool = True,
    max_projection_permutations: int = 20000,
) -> Dict[str, Any]:
    pred_body = predicted.get("result") or {}
    gold_body = gold.get("result") or {}
    pred_rows = pred_body.get("rows") or []
    gold_rows = gold_body.get("rows") or []
    pred_columns = pred_body.get("columns", [])
    gold_columns = gold_body.get("columns", [])

    exact = exact_rows_match(pred_rows, gold_rows)
    unordered = unordered_rows_match(pred_rows, gold_rows)
    gold_all_null = result_rows_all_null(gold_rows)
    if relaxed and gold_all_null:
        relaxed_payload = {
            "cluster": "reference_result_all_null_ignored",
            "reason": "Reference SQL returned only NULL values; this comparison is ignored.",
            "projection": {},
        }
    elif relaxed:
        relaxed_payload = relaxed_result_match(
            pred_rows=pred_rows,
            gold_rows=gold_rows,
            pred_columns=pred_columns,
            gold_columns=gold_columns,
            max_projection_permutations=max_projection_permutations,
        )
    else:
        relaxed_payload = {}
    predicted_ok = bool(predicted.get("ok"))
    gold_ok = bool(gold.get("ok"))
    relaxed_ok = bool(predicted_ok and gold_ok and relaxed_payload)
    return {
        "predicted_ok": predicted_ok,
        "gold_ok": gold_ok,
        "match": bool(predicted_ok and gold_ok and (exact or unordered or relaxed_ok)),
        "comparison_mode": "relaxed" if relaxed else "exact",
        "exact_rows_match": bool(predicted_ok and gold_ok and exact),
        "unordered_rows_match": bool(predicted_ok and gold_ok and unordered),
        "relaxed_match": relaxed_ok,
        "relaxed_cluster": relaxed_payload.get("cluster") if relaxed_payload else "",
        "relaxed_reason": relaxed_payload.get("reason") if relaxed_payload else "",
        "projection": relaxed_payload.get("projection") if relaxed_payload else {},
        "predicted_row_count": len(pred_rows),
        "gold_row_count": len(gold_rows),
        "predicted_columns": pred_columns,
        "gold_columns": gold_columns,
        "gold_error": gold.get("error", ""),
    }


def exact_rows_match(left: List[List[Any]], right: List[List[Any]]) -> bool:
    return normalize_rows_for_compare(left) == normalize_rows_for_compare(right)


def unordered_rows_match(left: List[List[Any]], right: List[List[Any]]) -> bool:
    return sorted(normalize_rows_for_compare(left)) == sorted(normalize_rows_for_compare(right))


def relaxed_result_match(
    *,
    pred_rows: List[List[Any]],
    gold_rows: List[List[Any]],
    pred_columns: List[str],
    gold_columns: List[str],
    max_projection_permutations: int = 20000,
) -> Dict[str, Any]:
    if not pred_rows or not gold_rows:
        return {}
    pred_width = len(pred_columns)
    gold_width = len(gold_columns)
    if pred_width <= 0 or gold_width <= 0 or pred_width < gold_width:
        return {}
    if not all(len(row) == pred_width for row in pred_rows):
        return {}
    if not all(len(row) == gold_width for row in gold_rows):
        return {}

    if pred_width == gold_width and unique_unordered_rows_match(pred_rows, gold_rows):
        return {
            "cluster": "duplicate_multiplicity_only",
            "reason": "Rows differ only by duplicate multiplicity; unique row values match.",
            "projection": {
                "projection_indices": list(range(pred_width)),
                "projection_columns": list(pred_columns),
                "projection_checked": 0,
                "match_type": "unique_unordered",
            },
        }

    checked = 0
    identity = tuple(range(gold_width))
    for indices in permutations(range(pred_width), gold_width):
        checked += 1
        if checked > max_projection_permutations:
            return {}
        if pred_width == gold_width and indices == identity:
            continue
        projected = project_rows(pred_rows, indices)
        exact = exact_rows_match(projected, gold_rows)
        unordered = unordered_rows_match(projected, gold_rows)
        unique_unordered = unique_unordered_rows_match(projected, gold_rows)
        if not exact and not unordered and not unique_unordered:
            continue
        if pred_width > gold_width:
            if unique_unordered and not exact and not unordered:
                cluster = "extra_columns_and_duplicate_multiplicity_only"
                reason = "After dropping extra columns, rows differ only by duplicate multiplicity."
            else:
                cluster = "extra_columns_only" if exact else "extra_columns_and_row_order_only"
                reason = "Predicted output contains extra columns, but a projection matches the expected rows."
        else:
            if unique_unordered and not exact and not unordered:
                cluster = "column_display_order_and_duplicate_multiplicity_only"
                reason = "After reordering columns, rows differ only by duplicate multiplicity."
            else:
                cluster = "column_display_order_only" if exact else "column_display_order_and_row_order_only"
                reason = "Predicted output columns can be reordered to match the expected rows."
        return {
            "cluster": cluster,
            "reason": reason,
            "projection": {
                "projection_indices": list(indices),
                "projection_columns": [pred_columns[idx] for idx in indices],
                "projection_checked": checked,
                "match_type": projection_match_type(exact, unordered, unique_unordered),
            },
        }
    return {}


def project_rows(rows: List[List[Any]], indices: tuple[int, ...]) -> List[List[Any]]:
    return [[row[idx] for idx in indices] for row in rows]


def normalize_rows_for_compare(rows: List[List[Any]]) -> List[str]:
    return [json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) for row in rows]


def unique_unordered_rows_match(left: List[List[Any]], right: List[List[Any]]) -> bool:
    return set(normalize_rows_for_compare(left)) == set(normalize_rows_for_compare(right))


def result_rows_all_null(rows: List[List[Any]]) -> bool:
    return bool(rows) and all(all(cell is None for cell in row) for row in rows)


def projection_match_type(exact: bool, unordered: bool, unique_unordered: bool) -> str:
    if exact:
        return "exact"
    if unordered:
        return "unordered"
    if unique_unordered:
        return "unique_unordered"
    return "none"
