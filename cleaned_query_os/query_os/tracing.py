from __future__ import annotations

import json
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


class EventTracer:
    def __init__(
        self,
        live: bool = False,
        stream: Optional[TextIO] = None,
        style: str = "pretty",
        color: str = "auto",
        max_chars: Optional[int] = None,
        state_view: str = "diff",
        result_cell_max_width: int = 32,
    ) -> None:
        self.live = live
        self.stream = stream or sys.stderr
        self.style = style if style in {"pretty", "plain"} else "pretty"
        self.color = color if color in {"auto", "always", "never"} else "auto"
        self.max_chars = max_chars if max_chars and max_chars > 0 else None
        self.state_view = state_view if state_view in {"off", "summary", "diff", "full"} else "diff"
        self.result_cell_max_width = result_cell_max_width if result_cell_max_width >= 0 else 32
        self.use_color = self._resolve_color()
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def emit(
        self,
        event_type: str,
        agent: str,
        message: str,
        *,
        global_step: Optional[int] = None,
        worker_step: Optional[int] = None,
        tool: Optional[str] = None,
        status: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            event = {
                "id": len(self.events) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "agent": agent,
                "message": message,
                "global_step": global_step,
                "worker_step": worker_step,
                "tool": tool,
                "status": status,
                "payload": payload or {},
            }
            self.events.append(event)
            if self.live:
                if self.style == "plain":
                    text = self.format_event(event)
                else:
                    text = self.format_pretty_event(event)
                if text:
                    print(text, file=self.stream, flush=True)

    def dump(self, path: str, result: Optional[Dict[str, Any]] = None) -> None:
        target = Path(path).expanduser()
        if target.parent and str(target.parent) != ".":
            target.parent.mkdir(parents=True, exist_ok=True)
        doc = {"trace_events": self.events}
        if result is not None:
            result_without_embedded_trace = dict(result)
            result_without_embedded_trace.pop("trace_events", None)
            doc["result"] = result_without_embedded_trace
        target.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def format_event(self, event: Dict[str, Any]) -> str:
        if event.get("event_type") == "llm_call":
            payload = event.get("payload") or {}
            parts = [
                f"provider={payload.get('provider')}",
                f"role={payload.get('role')}",
                f"endpoint={payload.get('endpoint')}",
                f"model={payload.get('model')}",
                f"latency_ms={payload.get('latency_ms')}",
            ]
            if payload.get("retry"):
                parts.append(f"retry={payload.get('retry')}")
            if payload.get("error"):
                parts.append(f"error={self._shorten(str(payload.get('error')), 160)}")
            return "[llm] " + ", ".join(str(part) for part in parts if part)

        if event.get("event_type") == "state_delta":
            if self.state_view == "off":
                return ""
            payload = event.get("payload") or {}
            summary = payload.get("summary") or {}
            latest = summary.get("submission_SQL") or {}
            parts = [
                f"writer={payload.get('writer')}",
                f"tables={summary.get('table_count', 0)}",
                f"sql_attempts={summary.get('sql_attempt_count', 0)}",
            ]
            if latest:
                parts.append(
                    f"submission_SQL=#{latest.get('attempt_idx')} "
                    f"{latest.get('status')} rows={latest.get('row_count')}"
                )
            warnings = payload.get("warnings") or []
            if warnings:
                parts.append(f"warnings={self._shorten(str(warnings), 160)}")
            return f"[shared_state] " + ", ".join(parts)

        parts = []
        global_step = event.get("global_step")
        worker_step = event.get("worker_step")
        if global_step is not None:
            parts.append(f"global step {global_step}")
        if worker_step is not None:
            parts.append(f"{event.get('agent')} step {worker_step}")

        prefix = "[" + "][".join(parts) + "]" if parts else f"[{event.get('agent')}]"
        tool = event.get("tool")
        status = event.get("status")
        suffix = ""
        if tool:
            suffix += f" tool={tool}"
        if status:
            suffix += f" status={status}"

        payload = event.get("payload") or {}
        payload_text = self._payload_preview(payload)
        if payload_text:
            suffix += f" | {payload_text}"

        message = self._shorten(str(event.get("message") or ""), 220)
        return f"{prefix} {message}{suffix}"

    def format_pretty_event(self, event: Dict[str, Any]) -> str:
        event_type = event.get("event_type")
        payload = event.get("payload") or {}
        global_step = event.get("global_step")
        worker_step = event.get("worker_step")
        agent = str(event.get("agent") or "")
        status = event.get("status")
        tool = event.get("tool")

        if event_type == "run_start":
            question = self._shorten(str(payload.get("question", "")), 120)
            db_path = self._shorten(str(payload.get("db_path", "")), 120)
            return "\n".join(
                [
                    self._color("+------------------------------------------------------------", "cyan"),
                    self._color("| QueryOS Agent Run", "cyan", bold=True),
                    f"| {self._color('question', 'yellow', bold=True)}: {question}",
                    f"| {self._color('database', 'yellow', bold=True)}: {db_path}",
                    self._color("+------------------------------------------------------------", "cyan"),
                ]
            )

        if event_type == "planner_decision":
            action = str(payload.get("action") or "")
            worker = self._action_worker_name(action)
            guidance = self._shorten(str(payload.get("guidance") or ""), 180)
            lines = [
                "",
                self._separator("blue"),
                self._color(f"GLOBAL STEP {global_step}", "blue", bold=True),
                f"  {self._color('Manager', 'blue', bold=True)} -> {self._color(worker, self._worker_color(worker), bold=True)}",
            ]
            if guidance:
                lines.append(f"  {self._color('guidance', 'yellow')}: {guidance}")
            if payload.get("selected_worker"):
                lines.append(f"  {self._color('selected worker', 'cyan')}: {payload.get('selected_worker')}")
            return "\n".join(lines)

        if event_type == "validation_start":
            sql_attempt_idx = payload.get("sql_attempt_idx")
            lines = [
                "",
                self._separator("cyan"),
                self._color("VALIDATION GATE", "cyan", bold=True),
                f"  {self._color('Manager', 'blue', bold=True)} -> {self._color('SQL Validator Agent', 'cyan', bold=True)}",
            ]
            if sql_attempt_idx:
                lines.append(f"  {self._color('sql attempt', 'yellow')}: #{sql_attempt_idx}")
            return "\n".join(lines)

        if event_type == "worker_start":
            raw_guidance = str(payload.get("guidance") or "")
            guidance = self._shorten(raw_guidance, 160)
            lines = [self._color(f"  {agent} worker started", self._agent_color(agent), bold=True)]
            if guidance and not self._repeats_planner_guidance(global_step, raw_guidance):
                lines.append(f"    {self._color('task', 'yellow')}: {guidance}")
            return "\n".join(lines)

        if event_type == "writer_group_start":
            workers = payload.get("workers") or []
            raw_guidance = str(payload.get("guidance") or "")
            guidance = self._shorten(raw_guidance, 180)
            lines = [self._color("  SWA writer group started", "yellow", bold=True)]
            lines.append(f"    {self._color('workers', 'cyan')}: {', '.join(str(x) for x in workers)}")
            if payload.get("timeout_seconds"):
                lines.append(f"    {self._color('timeout', 'cyan')}: {payload.get('timeout_seconds')}s")
            if guidance and not self._repeats_planner_guidance(global_step, raw_guidance):
                lines.append(f"    {self._color('task', 'yellow')}: {guidance}")
            return "\n".join(lines)

        if event_type == "schema_group_start":
            workers = payload.get("workers") or []
            raw_guidance = str(payload.get("guidance") or "")
            guidance = self._shorten(raw_guidance, 180)
            lines = [self._color("  SDA schema group started", "cyan", bold=True)]
            lines.append(f"    {self._color('workers', 'cyan')}: {', '.join(str(x) for x in workers)}")
            if payload.get("timeout_seconds"):
                lines.append(f"    {self._color('timeout', 'cyan')}: {payload.get('timeout_seconds')}s")
            if guidance and not self._repeats_planner_guidance(global_step, raw_guidance):
                lines.append(f"    {self._color('task', 'yellow')}: {guidance}")
            return "\n".join(lines)

        if event_type in {"schema_group_timeout", "writer_group_timeout"}:
            label = "SDA schema group" if event_type == "schema_group_timeout" else "SWA writer group"
            lines = [self._color(f"  [TIMEOUT] {label} watchdog fired", "yellow", bold=True)]
            if payload.get("timeout_seconds") is not None:
                lines.append(f"    {self._color('timeout', 'yellow')}: {payload.get('timeout_seconds')}s")
            workers = payload.get("workers") or []
            if workers:
                lines.append(f"    {self._color('cancelled workers', 'yellow')}: {', '.join(str(x) for x in workers)}")
            return "\n".join(lines)

        if event_type == "schema_group_merge":
            mark = "[OK]" if status == "ok" else "[ERR]"
            color = "green" if status == "ok" else "red"
            lines = [self._color(f"  {mark} SDA schema group merged", color, bold=True)]
            if payload.get("tables") is not None:
                lines.append(f"    {self._color('tables', 'cyan')}: {payload.get('tables')}")
            if payload.get("columns") is not None:
                lines.append(f"    {self._color('columns', 'cyan')}: {payload.get('columns')}")
            for item in payload.get("workers") or []:
                lines.append(
                    f"    - {item.get('worker')}: ok={item.get('ok')} "
                    f"timeout={item.get('timed_out', False)} "
                    f"tables={item.get('table_count')} columns={item.get('column_count')}"
                )
            if payload.get("timed_out_workers"):
                lines.append(
                    f"    {self._color('timed out', 'yellow')}: "
                    f"{', '.join(str(x) for x in payload.get('timed_out_workers') or [])}"
                )
            return "\n".join(lines)

        if event_type == "writer_group_round":
            lines = [self._color(f"  SWA group chat round {worker_step}", "yellow", bold=True)]
            objective = str(payload.get("objective") or "")
            if objective:
                lines.append(f"    {self._color('objective', 'cyan')}: {objective}")
            factions = payload.get("factions") or []
            for item in factions:
                sig = str(item.get("signature") or "")[:10]
                lines.append(
                    f"    - {self._color(str(item.get('representative') or ''), 'yellow')}: "
                    f"support={item.get('support_count')} sig={sig} rows={item.get('rows')}"
                )
            return "\n".join(lines)

        if event_type == "writer_group_action":
            writer = str(payload.get("writer") or agent)
            action = str(payload.get("action") or "")
            reason = str(payload.get("reason") or "")
            status_color = "red" if action == "QUIT" else "cyan"
            action_text = action
            if action == "QUIT" and reason:
                action_text = f"{action}({self._shorten(reason, 96)})"
            lines = [f"    {self._color(writer, 'yellow', bold=True)} {self._color(action_text, status_color, bold=True)}"]
            signature = str(payload.get("signature") or "")
            if signature:
                lines.append(f"      {self._color('signature', 'cyan')}: {signature[:16]}")
            convinced_by = str(payload.get("convinced_by_signature") or "")
            if convinced_by:
                lines.append(f"      {self._color('convinced by', 'cyan')}: {convinced_by[:16]}")
            if reason and action != "QUIT":
                label = "message" if action == "CHAT" else "reason"
                lines.append(f"      {self._color(label, 'yellow')}: {reason}")
            if payload.get("version") is not None:
                lines.append(f"      {self._color('version', 'cyan')}: {payload.get('version')}")
            return "\n".join(lines)

        if event_type == "writer_group_consensus":
            target = str(payload.get("target_worker") or "")
            mode = str(payload.get("mode") or "")
            lines = [self._color("  [OK] SWA writer group reached consensus", "green", bold=True)]
            if target:
                lines.append(f"    {self._color('winner', 'cyan')}: {target}")
            if mode:
                lines.append(f"    {self._color('mode', 'cyan')}: {mode}")
            if payload.get("chat_rounds") is not None:
                lines.append(f"    {self._color('chat rounds', 'cyan')}: {payload.get('chat_rounds')}")
            if payload.get("sql"):
                lines.append(f"    {self._color('SQL', 'yellow', bold=True)}")
                for sql_line in self._format_sql_block(str(payload.get("sql") or "")):
                    lines.append(f"      {self._color(sql_line, 'yellow')}")
            return "\n".join(lines)

        if event_type == "writer_group_divergence":
            if payload.get("selectable"):
                lines = [self._color("  [PENDING] SWA writer group needs manager selection", "yellow", bold=True)]
            else:
                lines = [self._color("  [ERR] SWA writer group did not reach consensus", "red", bold=True)]
            if payload.get("rounds") is not None:
                lines.append(f"    {self._color('rounds', 'yellow')}: {payload.get('rounds')}")
            if payload.get("chat_rounds") is not None:
                lines.append(f"    {self._color('chat rounds', 'yellow')}: {payload.get('chat_rounds')}")
            if payload.get("reason"):
                lines.append(f"    {self._color('reason', 'yellow')}: {payload.get('reason')}")
            candidates = payload.get("candidates") or []
            for item in candidates:
                worker = item.get("worker")
                rows = item.get("rows")
                ok = item.get("ok")
                lines.append(f"    - {worker}: ok={ok} rows={rows}")
            return "\n".join(lines)

        if event_type == "worker_step_start":
            return self._color(f"  {agent} turn {worker_step}", self._agent_color(agent), bold=True)

        if event_type == "worker_step_tools":
            tools = payload.get("tools") or []
            if tools:
                return f"    planned tools: {', '.join(str(x) for x in tools)}"
            return "    planned tools: none"

        if event_type == "worker_notice":
            notice = self._shorten(str(payload.get("notice") or ""), 220)
            lines = [self._color("    [NOTICE] worker guidance appended", "yellow", bold=True)]
            if notice:
                lines.append(f"      {self._color('notice', 'yellow')}: {notice}")
            return "\n".join(lines)

        if event_type == "llm_call":
            provider = payload.get("provider")
            endpoint = payload.get("endpoint")
            role = payload.get("role") or agent
            latency = payload.get("latency_ms")
            retry = payload.get("retry")
            mark = "[LLM]" if status == "ok" else "[LLM ERR]"
            color = "magenta" if status == "ok" else "red"
            line = (
                f"    {self._color(mark, color, bold=True)} "
                f"{role} -> {endpoint} ({provider}, {latency} ms)"
            )
            if retry:
                line += f" retry={retry}"
            if payload.get("error"):
                line += f"\n      {self._color('error', 'red')}: {self._shorten(str(payload.get('error')), 180)}"
            return line

        if event_type == "tool_result":
            if tool == "SQLITE_EXEC":
                return self._format_sqlite_exec(payload, status)
            mark = "[OK]" if status == "ok" else "[ERR]"
            color = "green" if status == "ok" else "red"
            lines = [f"    {self._color(mark, color, bold=True)} {tool or 'tool'}"]
            lines.extend(self._pretty_payload_lines(payload, indent="      "))
            return "\n".join(lines)

        if event_type == "worker_return":
            if payload.get("reason") == "finish_blocked":
                report = str(payload.get("report") or "Finish blocked by manager guard.")
                return "\n".join(
                    [
                        f"  {self._color('[BLOCKED]', 'yellow', bold=True)} Finish blocked by manager guard",
                        f"    {self._color('reason', 'yellow')}: {report}",
                    ]
                )
            mark = "[OK]" if status == "ok" else "[ERR]"
            color = "green" if status == "ok" else "red"
            return f"  {self._color(mark, color, bold=True)} {event.get('message')}"

        if event_type == "worker_finish":
            mark = "[OK]" if status == "ok" else "[ERR]"
            color = "green" if status == "ok" else "red"
            if status == "ok":
                return f"  {self._color(mark, color, bold=True)} {agent} finished"
            error = str(payload.get("error") or payload.get("report") or "")
            if error:
                return f"  {self._color(mark, color, bold=True)} {agent} finished: {error}"
            return f"  {self._color(mark, color, bold=True)} {agent} finished"

        if event_type == "run_finish":
            mark = "[DONE]" if status in {None, "ok"} else "[STOP]"
            color = "green" if status in {None, "ok"} else "red"
            lines = [
                "",
                self._separator(color),
                self._color(f"{mark} QueryOS finished", color, bold=True),
            ]
            if status not in {None, "ok"}:
                error = str(payload.get("error") or payload.get("report") or "")
                if error:
                    lines.append(f"  error: {error}")
            return "\n".join(lines)

        if event_type == "gold_start":
            sql = self._shorten(str(payload.get("sql") or ""), 240)
            return "\n".join(
                [
                    "",
                    self._separator("cyan"),
                    self._color("GOLDEN SQL CHECK", "cyan", bold=True),
                    f"  {self._color('sql', 'yellow', bold=True)}: {self._color(sql, 'yellow')}",
                ]
            )

        if event_type == "gold_result":
            mark = "[MATCH]" if payload.get("gold_match") else "[MISMATCH]"
            if status != "ok":
                mark = "[GOLD ERROR]"
            color = "green" if payload.get("gold_match") and status == "ok" else "red"
            lines = [self._color(f"{mark} Golden SQL result", color, bold=True)]
            lines.extend(self._pretty_payload_lines(payload, indent="  "))
            return "\n".join(lines)

        if event_type == "state_delta":
            return self._format_state_delta(payload)

        if event_type in {"planner_decide_start"}:
            return ""

        return self.format_event(event)

    def _pretty_payload_lines(self, payload: Dict[str, Any], indent: str) -> List[str]:
        lines: List[str] = []
        if payload.get("keywords"):
            lines.append(f"{indent}{self._color('keywords', 'yellow')}: {payload.get('keywords')}")
        if payload.get("tables_matched") is not None:
            lines.append(f"{indent}{self._color('matched tables', 'cyan')}: {payload.get('tables_matched')}")
        if payload.get("table"):
            lines.append(f"{indent}{self._color('table', 'cyan')}: {payload.get('table')}")
        if payload.get("columns"):
            columns = payload.get("columns")
            lines.append(f"{indent}{self._color('columns', 'cyan')}: {self._shorten(str(columns), 180)}")
            if payload.get("columns_available") is not None:
                lines.append(
                    f"{indent}{self._color('columns shown', 'cyan')}: "
                    f"{payload.get('columns_shown', len(columns))}/{payload.get('columns_available')}"
                )
        if payload.get("update"):
            lines.append(f"{indent}{self._color('schema update', 'green')}: {payload.get('update')}")
        if payload.get("sql"):
            sql = self._shorten(str(payload.get("sql")), 240)
            lines.append(f"{indent}{self._color('sql', 'yellow', bold=True)}: {self._color(sql, 'yellow')}")
        if payload.get("rows") is not None:
            lines.append(f"{indent}{self._color('rows returned', 'green')}: {payload.get('rows')}")
        if payload.get("preview_rows"):
            lines.append(f"{indent}{self._color('preview', 'green')}: {self._shorten(str(payload.get('preview_rows')), 220)}")
        if payload.get("warnings"):
            lines.append(f"{indent}{self._color('warnings', 'yellow')}: {payload.get('warnings')}")
        if payload.get("validation_status"):
            status = payload.get("validation_status")
            color = "green" if status == "pass" else "red"
            lines.append(f"{indent}{self._color('validation', color, bold=True)}: {status}")
        if payload.get("issues"):
            lines.append(f"{indent}{self._color('issues', 'red', bold=True)}:")
            for issue in payload.get("issues") or []:
                if isinstance(issue, dict):
                    issue_type = issue.get("type", "issue")
                    detail = issue.get("detail", "")
                    lines.append(f"{indent}  - {issue_type}: {detail}")
                else:
                    lines.append(f"{indent}  - {issue}")
        if payload.get("feedback"):
            lines.append(f"{indent}{self._color('feedback', 'yellow')}: {payload.get('feedback')}")
        if payload.get("guidance"):
            lines.append(f"{indent}{self._color('guidance', 'yellow')}: {payload.get('guidance')}")
        if payload.get("gold_match") is not None:
            lines.append(f"{indent}{self._color('gold match', 'green' if payload.get('gold_match') else 'red')}: {payload.get('gold_match')}")
        if payload.get("exact_match") is not None:
            lines.append(f"{indent}exact rows match: {payload.get('exact_match')}")
        if payload.get("unordered_match") is not None:
            lines.append(f"{indent}unordered rows match: {payload.get('unordered_match')}")
        if payload.get("predicted_preview"):
            lines.append(f"{indent}{self._color('predicted preview', 'cyan')}: {self._shorten(str(payload.get('predicted_preview')), 220)}")
        if payload.get("gold_preview"):
            lines.append(f"{indent}{self._color('gold preview', 'cyan')}: {self._shorten(str(payload.get('gold_preview')), 220)}")
        if payload.get("error"):
            lines.append(f"{indent}{self._color('error', 'red', bold=True)}: {payload.get('error')}")
        if payload.get("report"):
            lines.append(f"{indent}{self._color('report', 'green')}: {payload.get('report')}")
        return lines

    def _format_sqlite_exec(self, payload: Dict[str, Any], status: Optional[str]) -> str:
        mark = "[OK]" if status == "ok" else "[ERR]"
        color = "green" if status == "ok" else "red"
        sql = str(payload.get("sql") or "")
        columns = payload.get("columns") or []
        preview_rows = payload.get("preview_rows") or []
        row_count = payload.get("rows")

        lines = [f"    {self._color(mark, color, bold=True)} SQLite query"]
        if sql:
            lines.append(f"      {self._color('SQL', 'yellow', bold=True)}")
            for sql_line in self._format_sql_block(sql):
                lines.append(f"        {self._color(sql_line, 'yellow')}")

        if status != "ok":
            error = str(payload.get("error") or "")
            if error:
                lines.append(f"      {self._color('error', 'red', bold=True)}: {error}")
            return "\n".join(lines)

        lines.append(f"      {self._color('Result', 'green', bold=True)}")
        lines.append(f"        rows returned: {row_count}")
        if not preview_rows:
            lines.append(f"        {self._color('empty result set', 'red' if row_count == 0 else 'yellow')}")
            return "\n".join(lines)

        if self._all_preview_values_null(preview_rows):
            lines.append(f"        {self._color('preview values are all NULL', 'yellow', bold=True)}")
        lines.extend(self._format_result_table(columns, preview_rows, indent="        "))
        if payload.get("warnings"):
            lines.append(f"        {self._color('warnings', 'yellow')}: {payload.get('warnings')}")
        return "\n".join(lines)

    def _format_state_delta(self, payload: Dict[str, Any]) -> str:
        if self.state_view == "off":
            return ""
        summary = payload.get("summary") or {}
        delta = payload.get("delta") or {}
        writer = str(payload.get("writer") or "unknown")
        warnings = payload.get("warnings") or []

        lines = [
            "",
            self._separator("cyan"),
            self._color("SHARED STATE UPDATE", "cyan", bold=True),
            f"  {self._color('writer', 'yellow', bold=True)}: {writer}",
        ]
        if self.state_view == "summary":
            lines.extend(self._format_state_summary(summary, indent="  "))
            if warnings:
                lines.extend(self._format_state_warnings(warnings, indent="  "))
            return "\n".join(lines)

        lines.extend(self._format_schema_state_delta(delta))
        lines.extend(self._format_sql_state_delta(delta, include_sql=self.state_view == "full"))
        lines.extend(self._format_validation_state_delta(delta))
        lines.extend(self._format_planner_state_delta(delta))
        if warnings:
            lines.extend(self._format_state_warnings(warnings, indent="  "))
        if self.state_view == "full":
            lines.extend(self._format_state_summary(summary, indent="  "))
        return "\n".join(lines)

    def _format_state_summary(self, summary: Dict[str, Any], indent: str) -> List[str]:
        lines = [f"{indent}{self._color('Current Shared State', 'cyan', bold=True)}"]
        step = summary.get("step")
        max_steps = summary.get("max_steps")
        if step is not None and max_steps is not None:
            lines.append(f"{indent}  step: {step}/{max_steps}")
        tables = summary.get("tables") or []
        lines.append(f"{indent}  tables: {', '.join(tables) if tables else 'none'}")
        lines.append(f"{indent}  sql attempts: {summary.get('sql_attempt_count', 0)}")
        lines.append(f"{indent}  validations: {summary.get('validation_attempt_count', 0)}")
        if summary.get("workflow_status"):
            lines.append(f"{indent}  workflow: {summary.get('workflow_status')}")
        latest = summary.get("submission_SQL") or {}
        if latest:
            lines.append(
                f"{indent}  submission_SQL: #{latest.get('attempt_idx')} "
                f"{latest.get('status')} rows={latest.get('row_count')}"
            )
        latest_validation = summary.get("latest_validation") or {}
        if latest_validation:
            lines.append(
                f"{indent}  submission_SQL validation: #{latest_validation.get('validation_idx')} "
                f"{latest_validation.get('status')}"
            )
        lines.append(f"{indent}  planner steps: {summary.get('planner_step_count', 0)}")
        return lines

    def _format_schema_state_delta(self, delta: Dict[str, Any]) -> List[str]:
        lines = [f"  {self._color('Schema Memory', 'magenta', bold=True)}"]
        added = delta.get("added_tables") or []
        removed = delta.get("removed_tables") or []
        updated = delta.get("updated_tables") or []
        if not added and not removed and not updated:
            return lines + ["    no changes"]

        for table in added:
            table_name = str(table.get("table") or "")
            lines.append(f"    {self._color('+ table', 'green', bold=True)}: {table_name}")
            col_names = table.get("column_names") or []
            if col_names:
                lines.append(f"      columns: {', '.join(str(col) for col in col_names)}")
            primary_keys = table.get("primary_keys") or []
            if primary_keys:
                lines.append(f"      primary keys: {', '.join(str(pk) for pk in primary_keys)}")
            foreign_keys = table.get("foreign_key_labels") or []
            if foreign_keys:
                lines.append(f"      foreign keys: {', '.join(str(fk) for fk in foreign_keys)}")

        for table_name in removed:
            lines.append(f"    {self._color('- table', 'red', bold=True)}: {table_name}")

        for item in updated:
            lines.append(f"    {self._color('~ table', 'yellow', bold=True)}: {item.get('table')}")
            if item.get("added_columns"):
                lines.append(f"      + columns: {', '.join(str(x) for x in item.get('added_columns'))}")
            if item.get("removed_columns"):
                lines.append(f"      - columns: {', '.join(str(x) for x in item.get('removed_columns'))}")
            if item.get("added_foreign_keys"):
                lines.append(f"      + foreign keys: {', '.join(str(x) for x in item.get('added_foreign_keys'))}")
            if item.get("removed_foreign_keys"):
                lines.append(f"      - foreign keys: {', '.join(str(x) for x in item.get('removed_foreign_keys'))}")
            if item.get("primary_keys"):
                change = item.get("primary_keys") or {}
                lines.append(f"      primary keys: {change.get('from')} -> {change.get('to')}")
        return lines

    def _format_sql_state_delta(self, delta: Dict[str, Any], include_sql: bool) -> List[str]:
        lines = [f"  {self._color('SQL Memory', 'yellow', bold=True)}"]
        attempts = delta.get("added_sql_attempts") or []
        if not attempts:
            return lines + ["    no changes"]

        for attempt in attempts:
            status = str(attempt.get("status") or "unknown")
            status_color = "green" if status == "executed_ok" else "red"
            lines.append(
                f"    {self._color('+ attempt', status_color, bold=True)} "
                f"#{attempt.get('attempt_idx')}: {status}"
            )
            lines.append(f"      rows returned: {attempt.get('row_count')}")
            columns = attempt.get("columns") or []
            if columns:
                lines.append(f"      columns: {', '.join(str(col) for col in columns)}")
            attempt_warnings = attempt.get("warnings") or []
            if attempt_warnings:
                lines.append(f"      warnings: {', '.join(str(w) for w in attempt_warnings)}")
            if include_sql and attempt.get("sql"):
                lines.append("      SQL")
                for sql_line in self._format_sql_block(str(attempt.get("sql") or "")):
                    lines.append(f"        {self._color(sql_line, 'yellow')}")
            preview_rows = attempt.get("preview_rows") or []
            if preview_rows:
                lines.append("      preview")
                lines.extend(self._format_result_table(columns, preview_rows, indent="        "))
        return lines

    def _format_validation_state_delta(self, delta: Dict[str, Any]) -> List[str]:
        lines = [f"  {self._color('Validation Memory', 'cyan', bold=True)}"]
        workflow = delta.get("workflow_status")
        validations = delta.get("added_validation_attempts") or []
        if not workflow and not validations:
            return lines + ["    no changes"]
        if workflow:
            lines.append(f"    workflow: {workflow.get('from')} -> {workflow.get('to')}")
        for validation in validations:
            status = str(validation.get("status") or "unknown")
            color = "green" if status == "pass" else "red"
            lines.append(
                f"    {self._color('+ validation', color, bold=True)} "
                f"#{validation.get('validation_idx')}: {status} for SQL #{validation.get('sql_attempt_idx')}"
            )
            if validation.get("confidence"):
                lines.append(f"      confidence: {validation.get('confidence')}")
            if validation.get("issues"):
                lines.append("      issues:")
                for issue in validation.get("issues") or []:
                    issue_type = issue.get("type", "issue") if isinstance(issue, dict) else "issue"
                    detail = issue.get("detail", "") if isinstance(issue, dict) else str(issue)
                    lines.append(f"        - {issue_type}: {detail}")
            if validation.get("feedback"):
                lines.append(f"      feedback: {validation.get('feedback')}")
            if validation.get("report"):
                lines.append(f"      report: {validation.get('report')}")
        return lines

    def _format_planner_state_delta(self, delta: Dict[str, Any]) -> List[str]:
        lines = [f"  {self._color('Planner Memory', 'blue', bold=True)}"]
        added = delta.get("added_planner_steps") or []
        updated = delta.get("updated_planner_steps") or []
        if not added and not updated:
            return lines + ["    no changes"]

        for item in added:
            lines.append(f"    {self._color('+ decision', 'green', bold=True)}: {item.get('action')}")
            guidance = str(item.get("guidance") or "")
            if guidance:
                lines.append(f"      guidance: {guidance}")
            ret = item.get("agent_return") or {}
            if ret:
                lines.append(f"      worker: {ret.get('agent')} ok={ret.get('ok')}")
        for item in updated:
            ret = item.get("agent_return") or {}
            label = f"step {item.get('step_idx')}"
            lines.append(f"    {self._color('~ decision', 'yellow', bold=True)}: {label} {item.get('action')}")
            if ret:
                lines.append(f"      worker: {ret.get('agent')} ok={ret.get('ok')}")
        return lines

    def _format_state_warnings(self, warnings: List[Any], indent: str) -> List[str]:
        lines = [f"{indent}{self._color('Warnings', 'yellow', bold=True)}"]
        for warning in warnings:
            lines.append(f"{indent}  - {warning}")
        return lines

    def _format_sql_block(self, sql: str) -> List[str]:
        text = self._shorten(sql)
        text = re.sub(r"\s+", " ", text).strip()
        break_before = ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"]
        for token in break_before:
            text = re.sub(rf"\s+{token}\s+", f"\n{token} ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+(AND|OR)\s+", r"\n  \1 ", text, flags=re.IGNORECASE)
        return [line.rstrip() for line in text.splitlines() if line.strip()]

    def _format_result_table(self, columns: List[Any], rows: List[List[Any]], indent: str) -> List[str]:
        if not columns:
            return [f"{indent}{rows}"]
        str_columns = [str(c) for c in columns]
        str_rows = [[self._cell_text(cell) for cell in row] for row in rows]
        widths = []
        for idx, col in enumerate(str_columns):
            values = [row[idx] if idx < len(row) else "" for row in str_rows]
            width = max([len(col), *[len(v) for v in values]])
            if self.result_cell_max_width > 0:
                width = min(width, self.result_cell_max_width)
            widths.append(width)

        def fmt_row(values: List[str]) -> str:
            padded = []
            for idx, width in enumerate(widths):
                val = values[idx] if idx < len(values) else ""
                if len(val) > width:
                    val = val[: max(0, width - 3)] + "..."
                padded.append(val.ljust(width))
            return " | ".join(padded)

        sep = "-+-".join("-" * w for w in widths)
        lines = [
            f"{indent}{fmt_row(str_columns)}",
            f"{indent}{sep}",
        ]
        for row in str_rows:
            lines.append(f"{indent}{fmt_row(row)}")
        return lines

    @staticmethod
    def _cell_text(value: Any) -> str:
        if value is None:
            return "NULL"
        return str(value)

    @staticmethod
    def _all_preview_values_null(rows: List[List[Any]]) -> bool:
        return bool(rows) and all(all(cell is None for cell in row) for row in rows)

    @staticmethod
    def _action_worker_name(action: str) -> str:
        if action == "CALL_SCHEMA_DISCOVERY":
            return "Schema Discovery Agent"
        if action == "CALL_SQL_WRITER":
            return "SQL Writer Agent"
        if action == "SELECT_SUBMISSION_SQL":
            return "Select submission_SQL"
        if action == "FINISH":
            return "Finish"
        return action or "Worker"

    @staticmethod
    def _agent_color(agent: str) -> str:
        if agent.startswith("SDA"):
            return "magenta"
        if agent.startswith("SWA"):
            return "yellow"
        if agent.startswith("SVA"):
            return "cyan"
        if agent == "manager":
            return "blue"
        return "cyan"

    @staticmethod
    def _worker_color(worker: str) -> str:
        if "Schema" in worker:
            return "magenta"
        if "SQL" in worker:
            return "yellow"
        if "Finish" in worker:
            return "green"
        return "cyan"

    def _payload_preview(self, payload: Dict[str, Any]) -> str:
        fields = []
        for key in (
            "action",
            "guidance",
            "tools",
            "keywords",
            "table",
            "tables_matched",
            "sql",
            "rows",
            "preview_rows",
            "columns",
            "update",
            "warnings",
            "validation_status",
            "feedback",
            "gold_match",
            "writer",
            "error",
            "report",
        ):
            if key in payload and payload[key] not in (None, "", [], {}):
                fields.append(f"{key}={self._shorten(str(payload[key]), 120)}")
        return ", ".join(fields)

    def _separator(self, color: str = "cyan") -> str:
        return self._color("-" * 72, color)

    def _repeats_planner_guidance(self, global_step: Optional[int], guidance: str) -> bool:
        if global_step is None or not guidance:
            return False
        wanted = " ".join(guidance.split())
        for event in reversed(self.events[:-1]):
            if event.get("global_step") != global_step:
                continue
            if event.get("event_type") != "planner_decision":
                continue
            planner_guidance = str((event.get("payload") or {}).get("guidance") or "")
            return " ".join(planner_guidance.split()) == wanted
        return False

    def _shorten(self, text: str, max_chars: Optional[int] = None) -> str:
        text = " ".join((text or "").split())
        if self.max_chars is None:
            return text
        limit = self.max_chars
        if max_chars and max_chars > 0:
            limit = min(limit, max_chars)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _resolve_color(self) -> bool:
        if self.color == "always":
            return True
        if self.color == "never":
            return False
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def _color(self, text: str, color: str, bold: bool = False) -> str:
        if not self.use_color:
            return text
        codes = []
        if bold or color == "bold":
            codes.append("1")
        color_codes = {
            "red": "31",
            "green": "32",
            "yellow": "33",
            "blue": "34",
            "magenta": "35",
            "cyan": "36",
        }
        if color in color_codes:
            codes.append(color_codes[color])
        if not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"


NULL_TRACER = EventTracer(live=False)
