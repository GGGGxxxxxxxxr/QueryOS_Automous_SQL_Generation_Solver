from __future__ import annotations

from typing import Any, Dict, List, Optional

from .state import SharedState, SQLAttempt, TableEvidence, ValidationAttempt


def snapshot_state(state: SharedState, sql_preview_rows: int = 3) -> Dict[str, Any]:
    """Build a compact, JSON-safe snapshot of the shared agent state."""
    return {
        "step": state.step,
        "max_steps": state.max_steps,
        "workflow_status": state.workflow_status.value,
        "discovered_tables": {
            table: _table_snapshot(ev)
            for table, ev in sorted(state.discovered.tables.items(), key=lambda item: item[0])
        },
        "sql_attempts": [
            _sql_attempt_snapshot(idx, attempt, preview_rows=sql_preview_rows)
            for idx, attempt in enumerate(state.sql_attempts, start=1)
        ],
        "validation_attempts": [
            _validation_attempt_snapshot(idx, attempt)
            for idx, attempt in enumerate(state.validation_attempts, start=1)
        ],
        "planner_trace": [_planner_step_snapshot(item) for item in state.planner_trace],
    }


def diff_state(
    before: Dict[str, Any],
    after: Dict[str, Any],
    *,
    writer: str = "",
) -> Dict[str, Any]:
    """Return a user-oriented diff plus the after snapshot for trace replay."""
    delta = {
        "step": _value_change(before.get("step"), after.get("step")),
        "workflow_status": _value_change(before.get("workflow_status"), after.get("workflow_status")),
        "added_tables": _added_tables(before, after),
        "removed_tables": _removed_tables(before, after),
        "updated_tables": _updated_tables(before, after),
        "added_sql_attempts": _added_sql_attempts(before, after),
        "added_validation_attempts": _added_validation_attempts(before, after),
        "added_planner_steps": _added_planner_steps(before, after),
        "updated_planner_steps": _updated_planner_steps(before, after),
    }
    delta = {k: v for k, v in delta.items() if v not in (None, [], {})}
    warnings = _state_warnings(delta, after)
    return {
        "writer": writer,
        "summary": summarize_snapshot(after),
        "delta": delta,
        "warnings": warnings,
        "snapshot": after,
    }


def summarize_delta_for_planner(delta_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Strip trace-only details and keep the dispatch-relevant state changes."""
    delta = delta_payload.get("delta") or {}
    out: Dict[str, Any] = {"writer": delta_payload.get("writer", "")}
    if delta_payload.get("warnings"):
        out["warnings"] = delta_payload.get("warnings")

    compact_delta: Dict[str, Any] = {}
    if delta.get("step"):
        compact_delta["step"] = delta.get("step")
    if delta.get("workflow_status"):
        compact_delta["workflow_status"] = delta.get("workflow_status")
    if delta.get("added_tables"):
        compact_delta["added_tables"] = [
            {
                "table": table.get("table"),
                "columns": table.get("column_names", []),
                "primary_keys": table.get("primary_keys", []),
                "foreign_keys": table.get("foreign_key_labels", []),
            }
            for table in delta.get("added_tables") or []
        ]
    if delta.get("removed_tables"):
        compact_delta["removed_tables"] = delta.get("removed_tables")
    if delta.get("updated_tables"):
        compact_delta["updated_tables"] = delta.get("updated_tables")
    if delta.get("added_sql_attempts"):
        compact_delta["added_sql_attempts"] = [
            {
                "attempt_idx": attempt.get("attempt_idx"),
                "sql": attempt.get("sql", ""),
                "status": attempt.get("status"),
                "ok": attempt.get("ok"),
                "columns": attempt.get("columns", []),
                "row_count": attempt.get("row_count"),
                "preview_rows": attempt.get("preview_rows", []),
                "warnings": attempt.get("warnings", []),
                "error": attempt.get("error", ""),
            }
            for attempt in delta.get("added_sql_attempts") or []
        ]
    if delta.get("added_validation_attempts"):
        compact_delta["added_validation_attempts"] = delta.get("added_validation_attempts")
    if delta.get("updated_planner_steps"):
        compact_delta["worker_returns"] = [
            {
                "step_idx": item.get("step_idx"),
                "action": item.get("action"),
                "agent_return": item.get("agent_return"),
            }
            for item in delta.get("updated_planner_steps") or []
        ]
    if compact_delta:
        out["delta"] = compact_delta
    return out


def summarize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    tables = snapshot.get("discovered_tables") or {}
    attempts = snapshot.get("sql_attempts") or []
    validations = snapshot.get("validation_attempts") or []
    latest_attempt = attempts[-1] if attempts else None
    latest_validation = validations[-1] if validations else None
    summary: Dict[str, Any] = {
        "step": snapshot.get("step"),
        "max_steps": snapshot.get("max_steps"),
        "workflow_status": snapshot.get("workflow_status"),
        "tables": list(tables.keys()),
        "table_count": len(tables),
        "sql_attempt_count": len(attempts),
        "validation_attempt_count": len(validations),
        "planner_step_count": len(snapshot.get("planner_trace") or []),
    }
    if latest_attempt:
        summary["latest_sql_attempt"] = {
            "attempt_idx": latest_attempt.get("attempt_idx"),
            "status": latest_attempt.get("status"),
            "row_count": latest_attempt.get("row_count"),
            "columns": latest_attempt.get("columns", []),
            "warnings": latest_attempt.get("warnings", []),
            "error": latest_attempt.get("error", ""),
            "has_null_values": latest_attempt.get("has_null_values", False),
        }
    if latest_validation:
        summary["latest_validation"] = latest_validation
    return summary


def _table_snapshot(ev: TableEvidence) -> Dict[str, Any]:
    columns = [dict(col) for col in ev.columns]
    return {
        "table": ev.table,
        "confidence": ev.confidence,
        "columns": columns,
        "column_names": [str(col.get("name", "")) for col in columns if col.get("name")],
        "primary_keys": list(ev.primary_keys),
        "foreign_keys": [dict(fk) for fk in ev.foreign_keys],
        "foreign_key_labels": [_foreign_key_label(fk) for fk in ev.foreign_keys],
    }


def _sql_attempt_snapshot(idx: int, attempt: SQLAttempt, preview_rows: int = 3) -> Dict[str, Any]:
    result = attempt.result or {}
    body = result.get("result") or {}
    rows = body.get("rows") or []
    warnings = list(body.get("warnings") or [])
    has_null = any(any(cell is None for cell in row) for row in rows)
    if rows and has_null and "result contains NULL values" not in warnings:
        warnings.append("result contains NULL values")
    if result.get("ok") and not rows and "result returned zero rows" not in warnings:
        warnings.append("result returned zero rows")
    if not result.get("ok") and result.get("error"):
        warnings.append(str(result.get("error")))
    return {
        "attempt_idx": idx,
        "sql": attempt.sql,
        "status": attempt.status,
        "ok": bool(result.get("ok")),
        "columns": body.get("columns", []),
        "row_count": len(rows),
        "preview_rows": rows if preview_rows <= 0 else rows[:preview_rows],
        "preview_rows_shown": len(rows) if preview_rows <= 0 else min(len(rows), preview_rows),
        "truncated": body.get("truncated", False),
        "warnings": warnings,
        "error": result.get("error", ""),
        "has_null_values": has_null,
    }


def _validation_attempt_snapshot(idx: int, attempt: ValidationAttempt) -> Dict[str, Any]:
    return {
        "validation_idx": idx,
        "sql_attempt_idx": attempt.sql_attempt_idx,
        "status": attempt.status,
        "issues": [dict(issue) for issue in attempt.issues],
        "feedback": attempt.feedback,
        "report": attempt.report,
        "confidence": attempt.confidence,
    }


def _planner_step_snapshot(item: Any) -> Dict[str, Any]:
    agent_return = item.agent_return
    return {
        "step_idx": item.step_idx,
        "action": item.decision.action.value,
        "guidance": item.decision.guidance,
        "agent_return": (
            {
                "agent": agent_return.agent.value,
                "ok": agent_return.ok,
                "report": agent_return.report,
            }
            if agent_return
            else None
        ),
    }


def _added_tables(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    before_tables = before.get("discovered_tables") or {}
    after_tables = after.get("discovered_tables") or {}
    return [after_tables[name] for name in after_tables.keys() if name not in before_tables]


def _removed_tables(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    before_tables = before.get("discovered_tables") or {}
    after_tables = after.get("discovered_tables") or {}
    return [name for name in before_tables.keys() if name not in after_tables]


def _updated_tables(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    before_tables = before.get("discovered_tables") or {}
    after_tables = after.get("discovered_tables") or {}
    updates = []
    for table, after_table in after_tables.items():
        before_table = before_tables.get(table)
        if not before_table:
            continue
        update: Dict[str, Any] = {"table": table}
        before_cols = set(before_table.get("column_names") or [])
        after_cols = set(after_table.get("column_names") or [])
        added_cols = sorted(after_cols - before_cols)
        removed_cols = sorted(before_cols - after_cols)
        if added_cols:
            update["added_columns"] = added_cols
        if removed_cols:
            update["removed_columns"] = removed_cols

        before_fks = set(before_table.get("foreign_key_labels") or [])
        after_fks = set(after_table.get("foreign_key_labels") or [])
        added_fks = sorted(after_fks - before_fks)
        removed_fks = sorted(before_fks - after_fks)
        if added_fks:
            update["added_foreign_keys"] = added_fks
        if removed_fks:
            update["removed_foreign_keys"] = removed_fks

        before_pks = before_table.get("primary_keys") or []
        after_pks = after_table.get("primary_keys") or []
        if before_pks != after_pks:
            update["primary_keys"] = {"from": before_pks, "to": after_pks}

        if len(update) > 1:
            updates.append(update)
    return updates


def _added_sql_attempts(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    before_count = len(before.get("sql_attempts") or [])
    return (after.get("sql_attempts") or [])[before_count:]


def _added_validation_attempts(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    before_count = len(before.get("validation_attempts") or [])
    return (after.get("validation_attempts") or [])[before_count:]


def _added_planner_steps(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    before_count = len(before.get("planner_trace") or [])
    return (after.get("planner_trace") or [])[before_count:]


def _updated_planner_steps(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    before_trace = before.get("planner_trace") or []
    after_trace = after.get("planner_trace") or []
    updates = []
    for idx, before_step in enumerate(before_trace):
        if idx >= len(after_trace):
            break
        after_step = after_trace[idx]
        if before_step != after_step:
            updates.append(after_step)
    return updates


def _value_change(before_value: Any, after_value: Any) -> Optional[Dict[str, Any]]:
    if before_value == after_value:
        return None
    return {"from": before_value, "to": after_value}


def _foreign_key_label(fk: Dict[str, Any]) -> str:
    col = str(fk.get("col") or fk.get("column") or "").strip()
    ref = str(fk.get("ref") or "").strip()
    if not ref:
        ref_table = str(fk.get("ref_table") or "").strip()
        ref_column = str(fk.get("ref_column") or "").strip()
        ref = f"{ref_table}.{ref_column}" if ref_table and ref_column else ""
    if col and ref:
        return f"{col} -> {ref}"
    return col or ref


def _state_warnings(delta: Dict[str, Any], snapshot: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    for attempt in delta.get("added_sql_attempts") or []:
        for item in attempt.get("warnings") or []:
            if item not in warnings:
                warnings.append(item)
        if attempt.get("status") == "executed_err" and "SQL execution failed" not in warnings:
            warnings.append("SQL execution failed")
    for validation in delta.get("added_validation_attempts") or []:
        if validation.get("status") == "fail":
            for issue in validation.get("issues") or []:
                issue_type = issue.get("type", "validation_issue")
                detail = issue.get("detail", "")
                text = f"{issue_type}: {detail}" if detail else str(issue_type)
                if text not in warnings:
                    warnings.append(text)
        if validation.get("status") == "error" and validation.get("report"):
            if validation.get("report") not in warnings:
                warnings.append(validation.get("report"))
    summary = summarize_snapshot(snapshot)
    if summary.get("table_count") == 0 and "no schema tables discovered" not in warnings:
        warnings.append("no schema tables discovered")
    return warnings
