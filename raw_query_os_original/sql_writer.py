#!/usr/bin/env python3
"""
Spider2.0 SQL Writer Agent (SWA) adapted for Agent-lightning framework.

STRICT DESIGN:
- Tool-only interaction: assistant must output EXACTLY ONE tool call per turn.
- Standardized everything: SQLite execution and final report are BOTH tools.
- SQLITE_EXEC calls are auto-recorded into state.sql_attempts.
- The LAST SQLITE_EXEC before SWA_REPORT is treated as the final intended query.
- Single budget knob: max_turns (no separate "max_sqlite_calls").

TOOLS:
- SQLITE_EXEC { "sql": "..." }  (read-only)
- SWA_REPORT  { "report": "..." }  (finalize; must end the episode)

NOTES:
- SWA can probe with multiple subqueries before the final query.
- SWA should not guess categorical values; probe DISTINCT / GROUP BY first.
- SWA must not assume schema beyond discovered_schema.

Author: (you)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import openai
import agentlightning as agl

from agentlightning.litagent import DiscardTrajectory
from shared_states import SharedGlobalState, AgentReturn, AgentName, SQLAttempt

agl.setup_logging(apply_to=[__name__])
logger = logging.getLogger(__name__)


def build_sql_writer_system_prompt(db_path: str, max_turns: int) -> str:
    return f"""
You are the SQL Writer Agent (SWA) for a SQLite database.

You collaborate with:
- Schema Discovery Agent (SDA)
- PLANNER (provides hints and instructions — follow them carefully)

===============================================================================
1. DATABASE
===============================================================================
SQLite database path:
{db_path}

===============================================================================
2. TOOL USAGE (STRICT CONTRACT)
===============================================================================
You MUST follow ALL rules below:

- On every turn, output ONE OR MORE tool calls
- Do NOT output any free text
- Any non-tool output is INVALID

Allowed tools:
1) SQLITE_EXEC  {{ "sql": "..." }}
2) SWA_REPORT   {{ "report": "..." }}

Valid patterns per turn:
- one or more SQLITE_EXEC calls
- exactly one SWA_REPORT call

Constraints:
- Do NOT mix SQLITE_EXEC and SWA_REPORT in the same turn
- SWA_REPORT MUST appear alone
- SWA_REPORT is REQUIRED to finish the task

===============================================================================
3. EXECUTION MODEL
===============================================================================
Each SQLITE_EXEC call:
- Executes a read-only SQL query
- Returns results immediately
- Is automatically appended to SQL_HISTORY

You are encouraged to:
- Explore with LIMIT / DISTINCT / GROUP BY
- Inspect categorical distributions
- Test hypotheses using subqueries
- Iterate step-by-step

===============================================================================
4. FINAL QUERY REQUIREMENT (CRITICAL)
===============================================================================
Your FINAL query MUST:

- Fully answer the question inside SQL
- Perform ALL computations explicitly
- Return the final answer directly

Do NOT:
- Return intermediate rows
- Expect external reasoning
- Leave computations unfinished

Principle:
SQL must return the answer — not intermediate evidence.

===============================================================================
5. TURN TERMINATION
===============================================================================
- The LAST executed SQLITE_EXEC is treated as the final query
- SWA_REPORT must be issued in a separate final turn

Total turn budget: {max_turns}

===============================================================================
6. OUTPUT FORMAT REQUIREMENT (CRITICAL)
===============================================================================
The final result must be:

- Clean
- Minimal
- Directly aligned with the question

Rules:
- Return ONLY required fields
- Do NOT include intermediate columns (COUNT, SUM, AVG) unless explicitly requested
- If aggregation is used only for sorting/filtering:
  → keep it in ORDER BY / HAVING
  → do NOT include it in SELECT

Avoid unnecessary columns.

===============================================================================
7. SQL SAFETY RULES
===============================================================================
- READ ONLY (no INSERT / UPDATE / DELETE / DROP / ALTER / CREATE)
- Prefer explicit columns over SELECT *
- Use LIMIT for exploratory queries

===============================================================================
8. NUMERICAL & AGGREGATION RULES
===============================================================================
Division:
- Prefer REAL (decimal) results by default
- Do NOT allow integer division truncation
- Always preserve fractional values unless explicitly instructed otherwise
- Ensure REAL division via appropriate CAST

General:
- Use REAL arithmetic for:
  - ratios
  - averages
  - percentages
  - rates
- Handle NULL carefully (COALESCE, NULLIF)
- Avoid divide-by-zero (NULLIF)
- Do NOT assume numeric fields are clean (may require CAST / cleaning)

Carefully distinguish:
- difference
- ratio
- percentage
- percentage change
- average
- count

Other:
- COUNT(*) vs COUNT(column)
- Ensure output matches requested entity/value/both
- LEFT JOIN + WHERE may change semantics

===============================================================================
9. CATEGORICAL VALUE RULE (CRITICAL)
===============================================================================
NEVER guess categorical values.

Always inspect first:
- SELECT DISTINCT col ...
- SELECT col, COUNT(*) ...

===============================================================================
10. FAILURE HANDLING
===============================================================================
If schema is insufficient:
- Do NOT guess
- Use SWA_REPORT to explain missing information
"""


# =============================================================================
# SWA
# =============================================================================
class SQLwriterAgent:
    """
    SQL Writer Agent:
    - Tool-only multi-turn loop
    - SQLITE_EXEC for read-only queries (tool)
    - SWA_REPORT to finish (tool)
    - Strict validation: no text content, exactly one tool call each turn
    - Single budget: max_turns
    """

    def __init__(
        self,
        model: str,
        max_turns: int = 10,
        debug: bool = False,
        endpoint: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        verl_replacement: Dict[str, Any] | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        # sqlite execution limits
        sqlite_query_timeout_s: int = 100,
        sqlite_max_rows: int = 25,
        sqlite_max_cell_chars: int = 1000,
        sqlite_max_total_chars: int = 2000,
    ):
        self.debug = debug
        self.max_turns = max_turns
        self.endpoint = endpoint
        self.verl_replacement = verl_replacement

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # sqlite execution policies
        self.sqlite_query_timeout_s = sqlite_query_timeout_s
        self.sqlite_max_rows = sqlite_max_rows
        self.sqlite_max_cell_chars = sqlite_max_cell_chars
        self.sqlite_max_total_chars = sqlite_max_total_chars

        # -----------------------------
        # OpenAI tool schemas
        # -----------------------------
        SQLITE_TOOL = {
            "type": "function",
            "function": {
                "name": "SQLITE_EXEC",
                "description": "Execute a SQL query on the SQLite database (read-only).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL query to execute (read-only)."}
                    },
                    "required": ["sql"],
                    "additionalProperties": False,
                },
            },
        }

        REPORT_TOOL = {
            "type": "function",
            "function": {
                "name": "SWA_REPORT",
                "description": "Finish the task with a short report. Must be used to end the episode.",
                "parameters": {
                    "type": "object",
                    "properties": {"report": {"type": "string"}},
                    "required": ["report"],
                    "additionalProperties": False,
                },
            },
        }

        self.openai_tools = [SQLITE_TOOL, REPORT_TOOL]

        # Initialize OpenAI client
        self.client = openai.OpenAI(
            base_url=endpoint,
            api_key=os.getenv(api_key_env, "dummy"),
        )

    # -------------------------------------------------------------------------
    # Formatting helpers
    # -------------------------------------------------------------------------
    def _format_discovered_schema_for_swa(self, state: SharedGlobalState) -> str:
        """
        SQL Writer can ONLY see discovered_schema.

        JSON structure:
        [
          {"table":"T", "columns":[{"name":..,"type":..},...], "primary_keys":[...], "foreign_keys":[...]}
        ]
        """
        tables = []
        for tname, ev in state.discovered.tables.items():
            tables.append(
                {
                    "table": tname,
                    "columns": ev.columns or [],
                    "primary_keys": ev.primary_keys or [],
                    "foreign_keys": ev.foreign_keys or [],
                }
            )
        return json.dumps(tables, ensure_ascii=False, indent=2)

    def _compact_sql_history(self, state: SharedGlobalState, keep_last: int = 6) -> str:
        """
        Token-safe SQL history snippet for initial context.
        (Not required every turn; tool responses already provide step-by-step feedback.)
        """
        if not state.sql_attempts:
            return "[]"

        items = state.sql_attempts[-keep_last:]
        out = []

        start_idx = max(1, len(state.sql_attempts) - keep_last + 1)
        for k, a in enumerate(items, start=start_idx):
            ok = (a.status == "executed_ok") and bool((a.result or {}).get("ok"))
            err = None
            preview_rows = None

            if ok:
                res = (a.result or {}).get("result") or {}
                rows = res.get("rows") or []
                preview_rows = rows[:2]
            else:
                err = ((a.result or {}).get("error") or "")[:180]

            out.append(
                {
                    "idx": k,
                    "ok": bool(ok),
                    "sql": (a.sql or "")[:500],
                    "error": err,
                    "preview_rows": preview_rows,
                }
            )
        return json.dumps(out, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # SQLite execution (read-only) with timeout + truncation
    # -------------------------------------------------------------------------
    def execute_sqlite(self, db_path: str, sql: str) -> Dict[str, Any]:
        """
        Executes SQL on SQLite with:
        - busy_timeout for lock waits
        - hard interrupt after timeout seconds
        - result truncation to avoid huge payloads

        Returns:
          {"ok": True, "result": {...}, "error": ""}
        or
          {"ok": False, "result": {}, "error": "..."}
        """
        import sqlite3

        MAX_ROWS = self.sqlite_max_rows
        MAX_CELL_CHARS = self.sqlite_max_cell_chars
        MAX_TOTAL_CHARS = self.sqlite_max_total_chars
        QUERY_TIMEOUT = self.sqlite_query_timeout_s

        def _safe_cell(x: Any) -> Any:
            if x is None:
                return None
            s = str(x)
            if len(s) > MAX_CELL_CHARS:
                return f"[CELL_TRUNCATED len={len(s)}] " + s[:MAX_CELL_CHARS]
            return s

        try:
            conn = sqlite3.connect(db_path, timeout=QUERY_TIMEOUT)
            conn.execute(f"PRAGMA busy_timeout = {QUERY_TIMEOUT * 1000}")
            cursor = conn.cursor()

            # Hard timeout using interrupt
            timer = threading.Timer(QUERY_TIMEOUT, conn.interrupt)
            timer.start()
            try:
                cursor.execute(sql)
            finally:
                timer.cancel()

            abnormal = False
            warnings: List[str] = []

            if cursor.description:
                cols = [d[0] for d in cursor.description]

                rows = cursor.fetchmany(MAX_ROWS + 1)
                truncated_rows = len(rows) > MAX_ROWS
                if truncated_rows:
                    rows = rows[:MAX_ROWS]
                    warnings.append("ROW_LIMIT_REACHED")

                safe_rows = []
                total_chars = 0
                large_cell_detected = False

                for row in rows:
                    safe_row = []
                    for cell in row:
                        s = "" if cell is None else str(cell)
                        total_chars += len(s)
                        if len(s) > MAX_CELL_CHARS:
                            large_cell_detected = True
                        safe_row.append(_safe_cell(cell))
                    safe_rows.append(safe_row)

                if total_chars > MAX_TOTAL_CHARS:
                    abnormal = True
                    warnings.append("RESULT_TOO_LARGE")

                if large_cell_detected:
                    abnormal = True
                    warnings.append("LARGE_CELL_DETECTED")

                result = {
                    "columns": cols,
                    "rows": safe_rows,
                    "truncated": truncated_rows,
                    "max_rows": MAX_ROWS,
                    "abnormal": abnormal,
                    "warnings": warnings,
                }
            else:
                # Non-SELECT; should not happen under rules, but keep safe
                conn.commit()
                result = {
                    "columns": [],
                    "rows": [],
                    "truncated": False,
                    "max_rows": MAX_ROWS,
                    "abnormal": False,
                    "warnings": ["NON_SELECT_QUERY"],
                }

            conn.close()
            return {"ok": True, "result": result, "error": ""}

        except sqlite3.OperationalError as e:
            return {"ok": False, "result": {}, "error": f"SQLITE_TIMEOUT_OR_ERROR: {str(e)}"}
        except Exception as e:
            return {"ok": False, "result": {}, "error": str(e)}

    # -------------------------------------------------------------------------
    # LLM call
    # -------------------------------------------------------------------------
    def _call_llm(self, messages: List[Dict[str, Any]]) -> Optional[Any]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.openai_tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body={
                    "repetition_penalty": 1.05,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
            return resp.choices[0].message
        except Exception as e:
            logger.error("[SWA] LLM call failed: %s", e)
            return None

    def run(self, state: SharedGlobalState, guidance: str) -> AgentReturn:
        """
        Tool-only control loop.

        Enforced rules:
        - assistant may produce one or more tool calls per turn
        - assistant must not output free text content
        - allowed tools: SQLITE_EXEC, SWA_REPORT
        - SQLITE_EXEC and SWA_REPORT must not be mixed in the same turn
        - SWA_REPORT must appear alone to finish
        - single budget: max_turns
        """
        logger.info("[SWA] start | max_turns=%s", self.max_turns)

        system_prompt = build_sql_writer_system_prompt(
            db_path=state.db_path,
            max_turns=self.max_turns,
        )

        user_msg = (
            f"Real User Question:\n{(state.question or '').strip()}\n\n"
            f"External Knowledge:\n{(state.external_knowledge or '').strip()}\n\n"
            "PLANNER's INSTRUCTION or TIPS:\n"
            f"{(guidance or '').strip()}\n\n"
            "CURRENT discovered_schema (the ONLY schema you can use):\n"
            f"{self._format_discovered_schema_for_swa(state)}\n\n"
            "SQL_HISTORY (auto-recorded; compact):\n"
            f"{self._compact_sql_history(state)}\n"
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        last_executed_sql: Optional[str] = None
        sqlite_exec_count = 0

        for turn_idx in range(1, self.max_turns + 1):
            message = self._call_llm(messages)
            if not message:
                logger.error("[SWA] turn=%s | LLM returned None", turn_idx)
                return AgentReturn(
                    agent=AgentName.SQL_WRITER,
                    ok=False,
                    report="SWA LLM returned None",
                    payload={
                        "reason": "llm_returned_none",
                        "turn_idx": turn_idx,
                        "last_sql": last_executed_sql,
                        "sqlite_exec_count": sqlite_exec_count,
                        "sql_attempts": len(state.sql_attempts),
                    },
                )

            tool_calls = getattr(message, "tool_calls", None)
            content = (getattr(message, "content", "") or "").strip()

            logger.info(
                "[SWA] turn=%s/%s | tool_calls=%s | content_len=%s | sqlite_exec_count=%s",
                turn_idx,
                self.max_turns,
                (len(tool_calls) if tool_calls else 0),
                len(content),
                sqlite_exec_count,
            )

            # STRICT: at least one tool call, and no free text
            if not tool_calls or len(tool_calls) < 1:
                logger.error("[SWA] turn=%s | invalid: must output at least ONE tool call", turn_idx)
                return AgentReturn(
                    agent=AgentName.SQL_WRITER,
                    ok=False,
                    report="SWA invalid: must output at least one tool call",
                    payload={
                        "reason": "no_tool_call",
                        "turn_idx": turn_idx,
                        "raw_content": content[:2000],
                        "last_sql": last_executed_sql,
                        "sqlite_exec_count": sqlite_exec_count,
                        "sql_attempts": len(state.sql_attempts),
                    },
                )

            # if content:
            #     logger.error("[SWA] turn=%s | invalid: free text content is not allowed", turn_idx)
            #     return AgentReturn(
            #         agent=AgentName.SQL_WRITER,
            #         ok=False,
            #         report="SWA invalid: free text content is not allowed",
            #         payload={
            #             "reason": "free_text_not_allowed",
            #             "turn_idx": turn_idx,
            #             "raw_content": content[:2000],
            #             "last_sql": last_executed_sql,
            #             "sqlite_exec_count": sqlite_exec_count,
            #             "sql_attempts": len(state.sql_attempts),
            #         },
            #     )

            parsed_calls = []
            has_report = False
            has_sqlite = False

            for tc in tool_calls:
                if tc.type != "function":
                    logger.error("[SWA] turn=%s | invalid: tool_call type=%r", turn_idx, tc.type)
                    return AgentReturn(
                        agent=AgentName.SQL_WRITER,
                        ok=False,
                        report="SWA invalid: tool_call type is not function",
                        payload={
                            "reason": "invalid_tool_call_type",
                            "turn_idx": turn_idx,
                            "tool_call_type": tc.type,
                            "last_sql": last_executed_sql,
                            "sqlite_exec_count": sqlite_exec_count,
                            "sql_attempts": len(state.sql_attempts),
                        },
                    )

                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception as e:
                    logger.error("[SWA] turn=%s | invalid: tool arguments not JSON", turn_idx)
                    return AgentReturn(
                        agent=AgentName.SQL_WRITER,
                        ok=False,
                        report=f"SWA invalid: tool arguments not JSON ({name})",
                        payload={
                            "reason": "invalid_tool_arguments_json",
                            "turn_idx": turn_idx,
                            "tool": name,
                            "raw_arguments": tc.function.arguments,
                            "error": str(e),
                            "last_sql": last_executed_sql,
                            "sqlite_exec_count": sqlite_exec_count,
                            "sql_attempts": len(state.sql_attempts),
                        },
                    )

                if name not in {"SQLITE_EXEC", "SWA_REPORT"}:
                    logger.error("[SWA] turn=%s | invalid tool name=%r", turn_idx, name)
                    return AgentReturn(
                        agent=AgentName.SQL_WRITER,
                        ok=False,
                        report=f"SWA invalid tool name: {name}",
                        payload={
                            "reason": "invalid_tool_name",
                            "turn_idx": turn_idx,
                            "tool": name,
                            "last_sql": last_executed_sql,
                            "sqlite_exec_count": sqlite_exec_count,
                            "sql_attempts": len(state.sql_attempts),
                        },
                    )

                if name == "SQLITE_EXEC":
                    has_sqlite = True
                elif name == "SWA_REPORT":
                    has_report = True

                parsed_calls.append((tc, name, args))

            # Do not mix SQLITE_EXEC and SWA_REPORT in the same turn
            if has_report and has_sqlite:
                logger.error(
                    "[SWA] turn=%s | invalid: cannot mix SQLITE_EXEC and SWA_REPORT in the same turn",
                    turn_idx,
                )
                return AgentReturn(
                    agent=AgentName.SQL_WRITER,
                    ok=False,
                    report="SWA invalid: cannot mix SQLITE_EXEC and SWA_REPORT in the same turn",
                    payload={
                        "reason": "mixed_sqlite_exec_and_report",
                        "turn_idx": turn_idx,
                        "last_sql": last_executed_sql,
                        "sqlite_exec_count": sqlite_exec_count,
                        "sql_attempts": len(state.sql_attempts),
                    },
                )

            # SWA_REPORT must appear alone
            if has_report and len(parsed_calls) != 1:
                logger.error(
                    "[SWA] turn=%s | invalid: SWA_REPORT must appear alone in its own turn",
                    turn_idx,
                )
                return AgentReturn(
                    agent=AgentName.SQL_WRITER,
                    ok=False,
                    report="SWA invalid: SWA_REPORT must appear alone in its own turn",
                    payload={
                        "reason": "report_not_alone",
                        "turn_idx": turn_idx,
                        "num_tool_calls": len(parsed_calls),
                        "last_sql": last_executed_sql,
                        "sqlite_exec_count": sqlite_exec_count,
                        "sql_attempts": len(state.sql_attempts),
                    },
                )

            # Record assistant tool calls in messages (OpenAI tool protocol)
            assistant_tool_calls = []
            for tc, name, _args in parsed_calls:
                assistant_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": assistant_tool_calls,
                }
            )

            # -----------------------------
            # Case 1: SWA_REPORT (finalize)
            # -----------------------------
            if has_report:
                tc, name, args = parsed_calls[0]
                report = (args.get("report", "") or "").strip()
                if not report:
                    logger.error("[SWA] turn=%s | invalid SWA_REPORT: empty report", turn_idx)
                    return AgentReturn(
                        agent=AgentName.SQL_WRITER,
                        ok=False,
                        report="SWA invalid SWA_REPORT: empty report",
                        payload={
                            "reason": "empty_report",
                            "turn_idx": turn_idx,
                            "last_sql": last_executed_sql,
                            "sqlite_exec_count": sqlite_exec_count,
                            "sql_attempts": len(state.sql_attempts),
                        },
                    )

                logger.info(
                    "[SWA] DONE | sqlite_exec_count=%s | last_sql_present=%s",
                    sqlite_exec_count,
                    bool(last_executed_sql),
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"ok": True}, ensure_ascii=False),
                    }
                )

                payload = {
                    "final_sql": last_executed_sql,
                    "sqlite_exec_count": sqlite_exec_count,
                    "sql_attempts": len(state.sql_attempts),
                }

                return AgentReturn(
                    agent=AgentName.SQL_WRITER,
                    ok=True,
                    report=report,
                    payload=payload,
                )

            # -----------------------------
            # Case 2: one or more SQLITE_EXEC calls
            # -----------------------------
            for tc, name, args in parsed_calls:
                sql = (args.get("sql", "") or "").strip()
                if not sql:
                    logger.error("[SWA] turn=%s | invalid SQLITE_EXEC: empty sql", turn_idx)
                    return AgentReturn(
                        agent=AgentName.SQL_WRITER,
                        ok=False,
                        report="SWA invalid SQLITE_EXEC: empty sql",
                        payload={
                            "reason": "empty_sql",
                            "turn_idx": turn_idx,
                            "tool": name,
                            "last_sql": last_executed_sql,
                            "sqlite_exec_count": sqlite_exec_count,
                            "sql_attempts": len(state.sql_attempts),
                        },
                    )

                sqlite_exec_count += 1
                last_executed_sql = sql

                logger.info("[SWA] SQLITE_EXEC #%s | sql_head=%r", sqlite_exec_count, sql[:250])

                exec_ret = self.execute_sqlite(state.db_path, sql)

                if exec_ret.get("ok"):
                    res = exec_ret.get("result") or {}
                    cols = res.get("columns") or []
                    rows = res.get("rows") or []
                    logger.info(
                        "[SWA] SQL ok | cols=%s rows=%s truncated=%s warnings=%s",
                        len(cols),
                        len(rows),
                        bool(res.get("truncated")),
                        res.get("warnings") or [],
                    )
                    if self.debug:
                        for i, row in enumerate(rows[:5]):
                            logger.info("[SWA]   Row %s: %r", i + 1, row)
                        if len(rows) > 5:
                            logger.info("[SWA]   ... and %s more rows", len(rows) - 5)
                else:
                    logger.error("[SWA] SQL err | %s", exec_ret.get("error", ""))

                state.sql_attempts.append(
                    SQLAttempt(
                        sql=sql,
                        status="executed_ok" if exec_ret.get("ok") else "executed_err",
                        result=exec_ret,
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(exec_ret, ensure_ascii=False),
                    }
                )

            continue

        logger.error("[SWA] did not finish within max_turns=%s", self.max_turns)
        return AgentReturn(
            agent=AgentName.SQL_WRITER,
            ok=False,
            report="SWA did not finish within max_turns",
            payload={
                "reason": "max_turns_exceeded",
                "max_turns": self.max_turns,
                "last_sql": last_executed_sql,
                "sqlite_exec_count": sqlite_exec_count,
                "sql_attempts": len(state.sql_attempts),
            },
        )