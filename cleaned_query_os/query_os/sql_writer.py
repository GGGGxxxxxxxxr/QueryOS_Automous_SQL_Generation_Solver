from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import logging
import threading
from typing import Any, Dict, List, Optional

from .llm import create_chat_completion, create_llm_backend, is_fatal_llm_error, safe_llm_error
from .prompts import build_sql_writer_chat_system_prompt, build_sql_writer_system_prompt
from .sqlite_executor import SQLiteExecutor
from .state import AgentName, AgentReturn, SQLAttempt, SharedState
from .tracing import EventTracer, NULL_TRACER

logger = logging.getLogger(__name__)


SQL_WRITER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "SQLITE_EXEC",
            "description": "Execute a read-only SQLite query.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "SWA_REPORT",
            "description": "Finish after the final query has been executed.",
            "parameters": {
                "type": "object",
                "properties": {"report": {"type": "string"}},
                "required": ["report"],
                "additionalProperties": False,
            },
        },
    },
]


SQL_WRITER_CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "CHAT",
            "description": "Post a message defending your faction or challenging another faction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "QUIT",
            "description": (
                "Leave the group chat because another faction is more convincing. "
                "This is a spoken exit action: call QUIT with a natural-language reason, "
                "like QUIT(reason='My SQL is not good because ...')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "convinced_by_signature": {"type": "string"},
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class WriterCandidate:
    worker_id: str
    current_sql: str = ""
    current_result: Dict[str, Any] = field(default_factory=dict)
    report: str = ""
    version: int = 0
    ok: bool = False
    fatal: bool = False
    error: str = ""
    last_action: str = ""
    reason: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    def result_rows(self) -> List[List[Any]]:
        return ((self.current_result.get("result") or {}).get("rows") or [])

    def result_columns(self) -> List[Any]:
        return ((self.current_result.get("result") or {}).get("columns") or [])

    def result_ok(self) -> bool:
        return bool(self.current_result.get("ok"))


class SQLWriterAgent:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_turns: int = 8,
        executor: Optional[SQLiteExecutor] = None,
        trace_sql_preview_rows: int = 3,
        parallel_workers: int = 1,
        parallel_timeout_seconds: float = 0,
        chatgroup_enabled: bool = True,
        chatgroup_max_rounds: int = 2,
        consensus_require_same_columns: bool = False,
        debug: bool = False,
        tracer: Optional[EventTracer] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.client = llm_client or create_llm_backend(provider="openai", api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.executor = executor or SQLiteExecutor()
        self.trace_sql_preview_rows = trace_sql_preview_rows if trace_sql_preview_rows >= 0 else 3
        self.parallel_workers = max(1, int(parallel_workers or 1))
        self.parallel_timeout_seconds = max(0.0, float(parallel_timeout_seconds or 0))
        self.chatgroup_enabled = bool(chatgroup_enabled)
        self.chatgroup_max_rounds = max(0, int(chatgroup_max_rounds or 0))
        self.consensus_require_same_columns = bool(consensus_require_same_columns)
        self.debug = debug
        self.tracer = tracer or NULL_TRACER

    def run(self, state: SharedState, guidance: str) -> AgentReturn:
        if self.parallel_workers > 1:
            return self._run_writer_group(state, guidance)
        return self._run_single_worker(
            state,
            guidance,
            agent_label="SWA",
            worker_identity="You are the only SQL writer in this run.",
        )

    def _run_single_worker(
        self,
        state: SharedState,
        guidance: str,
        *,
        agent_label: str,
        worker_identity: str = "",
        cancel_event: Optional[threading.Event] = None,
    ) -> AgentReturn:
        global_step = state.step + 1
        self.tracer.emit(
            "worker_start",
            agent_label,
            "SQL writer worker started.",
            global_step=global_step,
            payload={"guidance": guidance},
        )
        identity_block = f"\nWORKER IDENTITY:\n{worker_identity}\n\n" if worker_identity else ""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": build_sql_writer_system_prompt(self.max_turns)},
            {
                "role": "user",
                "content": (
                    identity_block +
                    f"USER QUESTION:\n{state.question}\n\n"
                    f"EXTERNAL KNOWLEDGE:\n{state.external_knowledge}\n\n"
                    f"MANAGER GUIDANCE:\n{guidance}\n\n"
                    "CURRENT discovered_schema:\n"
                    f"{format_discovered_schema(state)}\n\n"
                    "SQL_HISTORY (writer-only diagnostic context; not part of submission_SQL):\n"
                    f"{format_sql_history(state)}\n"
                ),
            },
        ]

        last_sql = ""
        sqlite_exec_count = 0
        last_error = ""
        last_exec_signature = ""
        repeated_exec_count = 0

        for turn in range(1, self.max_turns + 1):
            if cancel_event is not None and cancel_event.is_set():
                return self._cancelled_return(agent_label, global_step, turn)
            self.tracer.emit(
                "worker_step_start",
                agent_label,
                "Requesting SQL tool call from LLM.",
                global_step=global_step,
                worker_step=turn,
            )
            try:
                message = self._call_llm(messages)
            except Exception as exc:
                last_error = f"SQL writer LLM call failed: {safe_llm_error(exc)}"
                self.tracer.emit(
                    "worker_error",
                    agent_label,
                    last_error,
                    global_step=global_step,
                    worker_step=turn,
                    status="error",
                    payload={"error": last_error},
                )
                self.tracer.emit(
                    "worker_finish",
                    agent_label,
                    "SQL writer stopped because the LLM call failed.",
                    global_step=global_step,
                    worker_step=turn,
                    status="error",
                    payload={"error": last_error},
                )
                return AgentReturn(
                    agent=AgentName.SQL_WRITER,
                    ok=False,
                    report=last_error,
                    payload={"reason": "llm_call_failed", "fatal": is_fatal_llm_error(exc)},
                )
            if cancel_event is not None and cancel_event.is_set():
                return self._cancelled_return(agent_label, global_step, turn)
            tool_calls = getattr(message, "tool_calls", None) if message is not None else None
            if not tool_calls:
                self.tracer.emit(
                    "worker_error",
                    agent_label,
                    "SQL writer produced no tool call.",
                    global_step=global_step,
                    worker_step=turn,
                    status="error",
                )
                return AgentReturn(
                    agent=AgentName.SQL_WRITER,
                    ok=False,
                    report="SQL writer produced no tool call.",
                    payload={"turn": turn},
                )

            parsed_calls = []
            has_report = False
            has_exec = False
            for tc in tool_calls:
                if getattr(tc, "type", "") != "function":
                    continue
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception as exc:
                    return AgentReturn(
                        agent=AgentName.SQL_WRITER,
                        ok=False,
                        report=f"SQL writer emitted invalid JSON for {tc.function.name}: {exc}",
                        payload={},
                    )
                if tc.function.name not in {"SQLITE_EXEC", "SWA_REPORT"}:
                    return AgentReturn(
                        agent=AgentName.SQL_WRITER,
                        ok=False,
                        report=f"SQL writer emitted unknown tool: {tc.function.name}",
                        payload={},
                    )
                has_exec = has_exec or tc.function.name == "SQLITE_EXEC"
                has_report = has_report or tc.function.name == "SWA_REPORT"
                parsed_calls.append((tc, tc.function.name, args))

            self.tracer.emit(
                "worker_step_tools",
                agent_label,
                f"LLM emitted {len(parsed_calls)} SQL writer tool call(s).",
                global_step=global_step,
                worker_step=turn,
                payload={"tools": [name for _, name, _ in parsed_calls]},
            )

            if has_exec and has_report:
                return AgentReturn(
                    agent=AgentName.SQL_WRITER,
                    ok=False,
                    report="SQL writer mixed SQLITE_EXEC and SWA_REPORT in one turn.",
                    payload={"last_sql": last_sql},
                )
            if has_report and len(parsed_calls) != 1:
                return AgentReturn(
                    agent=AgentName.SQL_WRITER,
                    ok=False,
                    report="SWA_REPORT must be the only tool call in a turn.",
                    payload={"last_sql": last_sql},
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": name, "arguments": tc.function.arguments},
                        }
                        for tc, name, _ in parsed_calls
                    ],
                }
            )

            if has_report:
                if cancel_event is not None and cancel_event.is_set():
                    return self._cancelled_return(agent_label, global_step, turn)
                tc, _, args = parsed_calls[0]
                report = str(args.get("report") or "").strip()
                self.tracer.emit(
                    "tool_result",
                    agent_label,
                    "Executed SWA_REPORT.",
                    global_step=global_step,
                    worker_step=turn,
                    tool="SWA_REPORT",
                    status="ok" if last_sql else "error",
                    payload={"report": report, "sql": last_sql},
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"ok": True}, ensure_ascii=False),
                    }
                )
                if not last_sql:
                    self.tracer.emit(
                        "worker_finish",
                        agent_label,
                        "SQL writer reported without executing SQL.",
                        global_step=global_step,
                        worker_step=turn,
                        status="error",
                        payload={"report": report},
                    )
                    return AgentReturn(
                        agent=AgentName.SQL_WRITER,
                        ok=False,
                        report=report or "SQL writer reported without executing SQL.",
                        payload={"last_sql": last_sql, "sqlite_exec_count": sqlite_exec_count},
                    )
                self.tracer.emit(
                    "worker_finish",
                    agent_label,
                    "SQL writer worker finished.",
                    global_step=global_step,
                    worker_step=turn,
                    status="ok",
                    payload={"report": report or "SQL writer finished.", "sql": last_sql},
                )
                return AgentReturn(
                    agent=AgentName.SQL_WRITER,
                    ok=True,
                    report=report or "SQL writer finished.",
                    payload={"final_sql": last_sql, "sqlite_exec_count": sqlite_exec_count},
                )

            for tc, _, args in parsed_calls:
                if cancel_event is not None and cancel_event.is_set():
                    return self._cancelled_return(agent_label, global_step, turn)
                sql = str(args.get("sql") or "").strip()
                if not sql:
                    self.tracer.emit(
                        "tool_result",
                        agent_label,
                        "SQLITE_EXEC received empty SQL.",
                        global_step=global_step,
                        worker_step=turn,
                        tool="SQLITE_EXEC",
                        status="error",
                    )
                    return AgentReturn(
                        agent=AgentName.SQL_WRITER,
                        ok=False,
                        report="SQLITE_EXEC received empty sql.",
                        payload={"last_sql": last_sql},
                    )
                sqlite_exec_count += 1
                last_sql = sql
                exec_result = self.executor.execute(state.db_path, sql)
                last_error = str(exec_result.get("error") or "")
                self.tracer.emit(
                    "tool_result",
                    agent_label,
                    "Executed SQLITE_EXEC.",
                    global_step=global_step,
                    worker_step=turn,
                    tool="SQLITE_EXEC",
                    status="ok" if exec_result.get("ok") else "error",
                    payload=sql_exec_payload(sql, exec_result, preview_rows=self.trace_sql_preview_rows),
                )
                state.sql_attempts.append(
                    SQLAttempt(
                        sql=sql,
                        status="executed_ok" if exec_result.get("ok") else "executed_err",
                        result=exec_result,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(exec_result, ensure_ascii=False),
                    }
                )
                exec_signature = duplicate_exec_signature(sql, exec_result)
                if exec_signature and exec_signature == last_exec_signature:
                    repeated_exec_count += 1
                else:
                    repeated_exec_count = 1
                    last_exec_signature = exec_signature
                if repeated_exec_count >= 2:
                    notice = (
                        "NOTICE: You have executed a duplicated or equivalent SQL result "
                        f"{repeated_exec_count} times in a row. If this SQL already answers "
                        "the full user question and follows the external evidence, call "
                        "SWA_REPORT next and finish your job. Do not execute the same or "
                        "equivalent SQL again unless the next query changes the answer logic."
                    )
                    messages.append({"role": "user", "content": notice})
                    self.tracer.emit(
                        "worker_notice",
                        agent_label,
                        "Duplicate SQL execution notice appended.",
                        global_step=global_step,
                        worker_step=turn,
                        status="ok",
                        payload={
                            "notice": notice,
                            "repeated_exec_count": repeated_exec_count,
                        },
                    )

        self.tracer.emit(
            "worker_finish",
            agent_label,
            "SQL writer did not finish within max_turns.",
            global_step=global_step,
            status="error",
            payload={"sql": last_sql, "error": last_error},
        )
        return AgentReturn(
            agent=AgentName.SQL_WRITER,
            ok=False,
            report="SQL writer did not finish within max_turns.",
            payload={
                "last_sql": last_sql,
                "sqlite_exec_count": sqlite_exec_count,
                "last_error": last_error,
            },
        )

    def _cancelled_return(self, agent_label: str, global_step: int, worker_step: int) -> AgentReturn:
        report = f"{agent_label} cancelled by parallel group watchdog."
        self.tracer.emit(
            "worker_finish",
            agent_label,
            report,
            global_step=global_step,
            worker_step=worker_step,
            status="error",
            payload={"error": report},
        )
        return AgentReturn(
            agent=AgentName.SQL_WRITER,
            ok=False,
            report=report,
            payload={"reason": "sql_worker_cancelled", "cancelled": True},
        )

    def _run_writer_group(self, state: SharedState, guidance: str) -> AgentReturn:
        global_step = state.step + 1
        worker_ids = [f"writer_{idx}" for idx in range(1, self.parallel_workers + 1)]
        self.tracer.emit(
            "writer_group_start",
            "SWA",
            "SQL writer group started.",
            global_step=global_step,
            payload={
                "workers": worker_ids,
                "guidance": guidance,
                "timeout_seconds": self.parallel_timeout_seconds,
            },
        )

        candidates = self._run_initial_group_workers(state, guidance, worker_ids)
        viable = [candidate for candidate in candidates.values() if candidate.current_sql and candidate.result_ok()]
        if not viable:
            report = "SQL writer group produced no executable SQL proposal."
            self.tracer.emit(
                "writer_group_divergence",
                "SWA",
                report,
                global_step=global_step,
                status="error",
                payload={"reason": report, "rounds": 0, "candidates": candidate_trace_summaries(candidates)},
            )
            return AgentReturn(
                agent=AgentName.SQL_WRITER,
                ok=False,
                report=report,
                payload={
                    "reason": "writer_group_no_executable_candidate",
                    "fatal": any(candidate.fatal for candidate in candidates.values()),
                    "writer_group": candidate_payloads(candidates),
                },
            )

        if len(viable) == 1:
            return self._commit_group_candidate(
                state,
                viable[0],
                candidates=candidates,
                mode="single_viable_worker",
                report=f"SQL writer group committed the only executable candidate from {viable[0].worker_id}.",
            )

        result_consensus = self._find_result_consensus(candidates)
        if result_consensus is not None:
            return self._commit_group_candidate(
                state,
                result_consensus,
                candidates=candidates,
                mode="initial_result_consensus",
                report="SQL writer group candidates produced consistent execution results.",
            )

        if not self.chatgroup_enabled or self.chatgroup_max_rounds <= 0:
            return self._writer_group_divergence(
                state=state,
                global_step=global_step,
                candidates=candidates,
                rounds=0,
                reason="Initial writer candidates disagreed and chatgroup is disabled.",
            )

        factions = build_result_factions(
            candidates,
            require_columns=self.consensus_require_same_columns,
            preview_rows=self.trace_sql_preview_rows,
        )
        active_representatives = {faction["signature"]: faction["representative_worker"] for faction in factions}
        chat_history: List[Dict[str, Any]] = []
        for round_idx in range(1, self.chatgroup_max_rounds + 1):
            active_factions = [faction for faction in factions if faction["signature"] in active_representatives]
            if len(active_factions) <= 1:
                winner = active_factions[0] if active_factions else factions[0]
                winner_candidate = candidates[str(winner["representative_worker"])]
                return self._commit_group_candidate(
                    state,
                    winner_candidate,
                    candidates=candidates,
                    mode=f"group_chat_last_faction_round_{round_idx}",
                    report=f"SQL writer group chat winner is {winner_candidate.worker_id}.",
                    chat_history=chat_history,
                )

            self.tracer.emit(
                "writer_group_round",
                "SWA",
                "SQL writer faction group chat round.",
                global_step=global_step,
                worker_step=round_idx,
                payload={
                    "objective": "Representatives speak one at a time, then chat or quit until one result faction remains.",
                    "factions": faction_trace_summaries(active_factions),
                },
            )
            round_log = {
                "round": round_idx,
                "active_factions": faction_trace_summaries(active_factions),
                "messages": [],
            }
            for faction in active_factions:
                worker_id = str(faction["representative_worker"])
                signature = str(faction["signature"])
                if signature not in active_representatives:
                    continue
                current_active_factions = [
                    item for item in factions if item["signature"] in active_representatives
                ]
                if len(current_active_factions) <= 1:
                    break
                visible_history = chat_history + [round_log]
                try:
                    action = self._call_chat_worker(
                        state=state,
                        guidance=guidance,
                        candidates=candidates,
                        factions=current_active_factions,
                        chat_history=visible_history,
                        round_idx=round_idx,
                        worker_id=worker_id,
                    )
                except Exception as exc:
                    action = {
                        "action": "CHAT",
                        "message": f"Chat action failed for {worker_id}: {safe_llm_error(exc)}",
                        "fatal": is_fatal_llm_error(exc),
                    }
                candidate = candidates[worker_id]
                action_name = str(action.get("action") or "").upper()
                reason = str(action.get("reason") or "").strip()
                signature = faction_signature_for_worker(current_active_factions, worker_id)
                message = str(action.get("message") or reason).strip()
                if action_name == "QUIT" and len(active_representatives) > 1:
                    active_representatives.pop(signature, None)
                    message = reason or "I am convinced by another faction and quit."
                elif action_name == "QUIT":
                    action_name = "CHAT"
                    message = "I am the last active representative, so I cannot quit."
                elif action_name != "CHAT":
                    action_name = "CHAT"
                    message = message or "I remain unconvinced and continue to defend my faction."

                candidate.last_action = action_name
                candidate.reason = message
                candidate.history.append(
                    {
                        "round": round_idx,
                        "action": action_name,
                        "message": message,
                        "signature": signature,
                        "convinced_by_signature": str(action.get("convinced_by_signature") or "").strip(),
                    }
                )
                self.tracer.emit(
                    "writer_group_action",
                    "SWA",
                    "SQL writer representative posted to group chat.",
                    global_step=global_step,
                    worker_step=round_idx,
                    status="ok",
                    payload={
                        "writer": worker_id,
                        "action": action_name,
                        "signature": signature,
                        "reason": message,
                        "convinced_by_signature": str(action.get("convinced_by_signature") or "").strip(),
                        "version": candidate.version,
                    },
                )
                round_log["messages"].append(
                    {
                        "worker": worker_id,
                        "signature": signature,
                        "action": action_name,
                        "message": message,
                        "convinced_by_signature": str(action.get("convinced_by_signature") or "").strip(),
                    }
                )

            chat_history.append(round_log)

            active_factions = [faction for faction in factions if faction["signature"] in active_representatives]
            if len(active_factions) == 1:
                winner_candidate = candidates[str(active_factions[0]["representative_worker"])]
                return self._commit_group_candidate(
                    state,
                    winner_candidate,
                    candidates=candidates,
                    mode=f"group_chat_winner_round_{round_idx}",
                    report=(
                        "SQL writer group chat reached one remaining result faction: "
                        f"{winner_candidate.worker_id}."
                    ),
                    chat_history=chat_history,
                )

        result_consensus = self._find_result_consensus(candidates)
        if result_consensus is not None:
            return self._commit_group_candidate(
                state,
                result_consensus,
                candidates=candidates,
                mode="post_chat_result_consensus",
                report=(
                    "SQL writer group chat did not leave one remaining faction, "
                    "but final execution results are consistent."
                ),
                chat_history=chat_history,
            )
        return self._writer_group_divergence(
            state=state,
            global_step=global_step,
            candidates=candidates,
            rounds=self.chatgroup_max_rounds,
            reason="Writer group exhausted chat rounds without consensus.",
            chat_history=chat_history,
        )

    def _run_initial_group_workers(
        self,
        state: SharedState,
        guidance: str,
        worker_ids: List[str],
    ) -> Dict[str, WriterCandidate]:
        original_attempt_count = len(state.sql_attempts)
        candidates: Dict[str, WriterCandidate] = {}
        cancel_event = threading.Event()
        pool = ThreadPoolExecutor(max_workers=len(worker_ids))
        futures = {
            pool.submit(
                self._run_initial_group_worker,
                state,
                guidance,
                worker_id,
                original_attempt_count,
                cancel_event,
            ): worker_id
            for worker_id in worker_ids
        }
        processed_futures = set()
        timeout_seconds = self.parallel_timeout_seconds if self.parallel_timeout_seconds > 0 else None
        global_step = state.step + 1
        try:
            try:
                for future in as_completed(futures, timeout=timeout_seconds):
                    processed_futures.add(future)
                    worker_id = futures[future]
                    candidates[worker_id] = self._writer_future_result(worker_id, future)
            except FuturesTimeoutError:
                timed_out = [worker_id for future, worker_id in futures.items() if not future.done()]
                cancel_event.set()
                self.tracer.emit(
                    "writer_group_timeout",
                    "SWA",
                    "SQL writer group timed out; continuing with completed workers.",
                    global_step=global_step,
                    status="error",
                    payload={"timeout_seconds": self.parallel_timeout_seconds, "workers": timed_out},
                )

            for future, worker_id in futures.items():
                if future.done() and future not in processed_futures and worker_id not in candidates:
                    candidates[worker_id] = self._writer_future_result(worker_id, future)

            for future, worker_id in futures.items():
                if worker_id in candidates:
                    continue
                future.cancel()
                candidates[worker_id] = WriterCandidate(
                    worker_id=worker_id,
                    ok=False,
                    fatal=False,
                    error=f"{worker_id} timed out before returning SQL.",
                    report=f"{worker_id} timed out before returning SQL.",
                    last_action="TIMEOUT",
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return {worker_id: candidates[worker_id] for worker_id in worker_ids}

    @staticmethod
    def _writer_future_result(worker_id: str, future: Any) -> WriterCandidate:
        try:
            return future.result()
        except Exception as exc:
            err = f"{worker_id} failed unexpectedly: {safe_llm_error(exc)}"
            return WriterCandidate(
                worker_id=worker_id,
                ok=False,
                fatal=is_fatal_llm_error(exc),
                error=err,
                report=err,
            )

    def _run_initial_group_worker(
        self,
        state: SharedState,
        guidance: str,
        worker_id: str,
        original_attempt_count: int,
        cancel_event: Optional[threading.Event] = None,
    ) -> WriterCandidate:
        worker_state = fork_shared_state_for_worker(state)
        ret = self._run_single_worker(
            worker_state,
            guidance,
            agent_label=f"SWA-{worker_id}",
            worker_identity=(
                f"YOU ARE: {worker_id}.\n"
                f"You are {worker_id} in a parallel SQL writer group. "
                "Work independently on your forked state. Your intermediate SQL executions "
                "are local to you until the group reaches consensus."
            ),
            cancel_event=cancel_event,
        )
        return candidate_from_worker_state(worker_id, worker_state, ret, original_attempt_count)

    def _call_chat_worker(
        self,
        state: SharedState,
        guidance: str,
        candidates: Dict[str, WriterCandidate],
        factions: List[Dict[str, Any]],
        chat_history: List[Dict[str, Any]],
        round_idx: int,
        worker_id: str,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": build_sql_writer_chat_system_prompt()},
            {
                "role": "user",
                "content": build_writer_group_chat_context(
                    state=state,
                    guidance=guidance,
                    candidates=candidates,
                    factions=factions,
                    chat_history=chat_history,
                    round_idx=round_idx,
                    worker_id=worker_id,
                    preview_rows=self.trace_sql_preview_rows,
                ),
            },
        ]
        message = self._call_chat_llm(messages)
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        if not tool_calls or len(tool_calls) != 1:
            return {
                "action": "CHAT",
                "message": "No clear chat action was emitted; I remain in the discussion.",
            }
        tc = tool_calls[0]
        name = str(tc.function.name or "").upper()
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception as exc:
            return {
                "action": "CHAT",
                "message": f"Invalid chat action JSON; I remain in the discussion. {exc}",
            }
        if name == "CHAT":
            return {
                "action": "CHAT",
                "message": str(args.get("message") or "").strip(),
            }
        if name == "QUIT":
            return {
                "action": "QUIT",
                "reason": str(args.get("reason") or "").strip(),
                "convinced_by_signature": str(args.get("convinced_by_signature") or "").strip(),
            }
        return {
            "action": "CHAT",
            "message": f"Unknown chat action {name}; I remain in the discussion.",
        }

    def _find_result_consensus(self, candidates: Dict[str, WriterCandidate]) -> Optional[WriterCandidate]:
        viable = [candidate for candidate in candidates.values() if candidate.current_sql and candidate.result_ok()]
        if len(viable) != len(candidates) or len(viable) < 2:
            return None
        first_signature = result_signature(
            viable[0].current_result,
            require_columns=self.consensus_require_same_columns,
        )
        if all(
            result_signature(candidate.current_result, require_columns=self.consensus_require_same_columns)
            == first_signature
            for candidate in viable[1:]
        ):
            return viable[0]
        return None

    def _commit_group_candidate(
        self,
        state: SharedState,
        candidate: WriterCandidate,
        *,
        candidates: Optional[Dict[str, WriterCandidate]] = None,
        mode: str,
        report: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentReturn:
        global_step = state.step + 1
        state.sql_attempts.append(
            SQLAttempt(
                sql=candidate.current_sql,
                status="executed_ok" if candidate.current_result.get("ok") else "executed_err",
                result=candidate.current_result,
            )
        )
        self.tracer.emit(
            "writer_group_consensus",
            "SWA",
            "SQL writer group reached consensus.",
            global_step=global_step,
            status="ok",
            payload={
                "target_worker": candidate.worker_id,
                "mode": mode,
                "sql": candidate.current_sql,
                "rows": len(candidate.result_rows()),
                "columns": candidate.result_columns(),
                "chat_rounds": len(chat_history or []),
            },
        )
        return AgentReturn(
            agent=AgentName.SQL_WRITER,
            ok=bool(candidate.current_result.get("ok")),
            report=report,
            payload={
                "final_sql": candidate.current_sql,
                "writer_group_mode": mode,
                "consensus_worker": candidate.worker_id,
                "writer_group": candidate_payloads(candidates or {candidate.worker_id: candidate}),
                "writer_group_chat_history": chat_history or [],
            },
        )

    def _writer_group_divergence(
        self,
        *,
        state: SharedState,
        global_step: int,
        candidates: Dict[str, WriterCandidate],
        rounds: int,
        reason: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentReturn:
        factions = build_result_factions(
            candidates,
            require_columns=self.consensus_require_same_columns,
            preview_rows=self.trace_sql_preview_rows,
        )
        pending_payload = {
            "reason": reason,
            "rounds": rounds,
            "chat_rounds": len(chat_history or []),
            "candidates": candidate_payloads(candidates),
            "factions": factions,
            "chat_history": chat_history or [],
        }
        state.pending_writer_group = pending_payload
        self.tracer.emit(
            "writer_group_divergence",
            "SWA",
            reason,
            global_step=global_step,
            status="ok",
            payload={
                "reason": reason,
                "rounds": rounds,
                "chat_rounds": len(chat_history or []),
                "selectable": True,
                "candidates": candidate_trace_summaries(candidates),
            },
        )
        return AgentReturn(
            agent=AgentName.SQL_WRITER,
            ok=True,
            report=reason,
            payload={
                "reason": "writer_group_divergence",
                "writer_group": pending_payload["candidates"],
                "writer_group_factions": factions,
                "writer_group_chat_history": chat_history or [],
            },
        )

    def _call_llm(self, messages: List[Dict[str, Any]]) -> Any:
        response = create_chat_completion(
            self.client,
            role="sql_writer",
            model=self.model,
            messages=messages,
            tools=SQL_WRITER_TOOLS,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message

    def _call_chat_llm(self, messages: List[Dict[str, Any]]) -> Any:
        response = create_chat_completion(
            self.client,
            role="sql_writer",
            model=self.model,
            messages=messages,
            tools=SQL_WRITER_CHAT_TOOLS,
            tool_choice="required",
            parallel_tool_calls=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message


def fork_shared_state_for_worker(state: SharedState) -> SharedState:
    return SharedState(
        question=state.question,
        db_path=state.db_path,
        db_id=state.db_id,
        external_knowledge=state.external_knowledge,
        metadata_display=state.metadata_display,
        workflow_status=state.workflow_status,
        discovered=state.discovered,
        sql_attempts=list(state.sql_attempts),
        validation_attempts=list(state.validation_attempts),
        pending_writer_group=dict(state.pending_writer_group),
        planner_trace=list(state.planner_trace),
        step=state.step,
        max_steps=state.max_steps,
    )


def candidate_from_worker_state(
    worker_id: str,
    worker_state: SharedState,
    ret: AgentReturn,
    original_attempt_count: int,
) -> WriterCandidate:
    new_attempts = worker_state.sql_attempts[original_attempt_count:]
    final_sql = str(ret.payload.get("final_sql") or ret.payload.get("last_sql") or "").strip()
    selected_attempt: Optional[SQLAttempt] = None
    if final_sql:
        for attempt in reversed(new_attempts):
            if attempt.sql.strip() == final_sql:
                selected_attempt = attempt
                break
    if selected_attempt is None and new_attempts:
        selected_attempt = new_attempts[-1]
        final_sql = selected_attempt.sql

    current_result = selected_attempt.result if selected_attempt and selected_attempt.result else {}
    candidate = WriterCandidate(
        worker_id=worker_id,
        current_sql=final_sql,
        current_result=current_result,
        report=ret.report,
        version=1 if final_sql else 0,
        ok=bool(ret.ok and current_result.get("ok")),
        fatal=bool(ret.payload.get("fatal")),
        error=str((current_result or {}).get("error") or ("" if ret.ok else ret.report)),
        last_action="INITIAL",
    )
    candidate.history.append(
        {
            "round": 0,
            "action": "INITIAL",
            "sql": final_sql,
            "result": compact_exec_result(current_result),
            "report": ret.report,
            "ok": ret.ok,
        }
    )
    return candidate


def build_writer_group_chat_context(
    *,
    state: SharedState,
    guidance: str,
    candidates: Dict[str, WriterCandidate],
    factions: List[Dict[str, Any]],
    chat_history: List[Dict[str, Any]],
    round_idx: int,
    worker_id: str,
    preview_rows: int,
) -> str:
    payload = {
        "worker_id": worker_id,
        "round": round_idx,
        "objective": (
            "Sequential quick group chat among result-faction representatives. "
            "Each representative can see earlier messages, then make one short point or quit until one faction remains."
        ),
        "question": state.question,
        "external_knowledge": state.external_knowledge,
        "manager_guidance": guidance,
        "discovered_schema": json.loads(format_discovered_schema(state)),
        "your_result_signature": faction_signature_for_worker(factions, worker_id),
        "result_factions": factions,
        "rules": [
            "You cannot revise SQL in this phase.",
            "Representatives speak one at a time; chat_history may include earlier messages from this same round.",
            "Use CHAT to defend your result or challenge another faction.",
            "Write in first person as yourself, but do not introduce yourself; do not start with 'I am writer_1' or 'As writer_2'.",
            "Keep each CHAT message to 1-3 short conversational sentences.",
            "Do not repeat an argument already made in chat_history unless you add new evidence.",
            "Do not refer to yourself in third person, and do not speak as another worker.",
            "If another faction convinces you, use QUIT instead of CHAT.",
            "QUIT is not silent: write a natural-language first-person reason, like QUIT(reason='My SQL is not good because ...').",
            "Use QUIT only when another faction has convinced you that your result should not win.",
            "The runtime declares the winner when only one representative remains.",
        ],
        "current_candidates": [
            candidate_chat_payload(candidate, preview_rows=preview_rows)
            for candidate in candidates.values()
        ],
        "chat_history": chat_history,
    }
    return (
        f"YOU ARE: {worker_id}.\n"
        f"You are speaking as {worker_id}, the representative for your current result faction.\n"
        "Use this identity internally, but do not introduce yourself in the message; the log already shows your worker id. "
        "Write like a concise coworker: 1-3 short first-person sentences, no formal debate speech, no repeated points. "
        "Do not speak as another worker. If I am convinced by another faction, I should call QUIT "
        "with a natural-language reason, e.g. QUIT(reason='My SQL is not good because ...').\n\n"
        "GROUP CHAT CONTEXT JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def candidate_chat_payload(candidate: WriterCandidate, preview_rows: int) -> Dict[str, Any]:
    result = candidate.current_result or {}
    body = result.get("result") or {}
    rows = body.get("rows") or []
    preview = rows if preview_rows <= 0 else rows[:preview_rows]
    return {
        "worker": candidate.worker_id,
        "version": candidate.version,
        "current_sql": candidate.current_sql,
        "execution": {
            "ok": bool(result.get("ok")),
            "error": result.get("error", ""),
            "columns": body.get("columns", []),
            "row_count": len(rows),
            "preview_rows": preview,
            "warnings": body.get("warnings", []),
        },
        "last_action": candidate.last_action,
        "last_reason": candidate.reason,
        "report": candidate.report,
    }


def compact_exec_result(exec_result: Dict[str, Any], preview_rows: int = 5) -> Dict[str, Any]:
    body = exec_result.get("result") or {}
    rows = body.get("rows") or []
    return {
        "ok": bool(exec_result.get("ok")),
        "error": exec_result.get("error", ""),
        "columns": body.get("columns", []),
        "row_count": len(rows),
        "preview_rows": rows[:preview_rows],
        "warnings": body.get("warnings", []),
    }


def duplicate_exec_signature(sql: str, exec_result: Dict[str, Any]) -> str:
    if exec_result.get("ok"):
        return "ok:" + result_signature(exec_result, require_columns=True)
    error = str(exec_result.get("error") or "")
    return "err:" + " ".join(str(sql or "").split()).rstrip(";").lower() + ":" + error


def candidate_payloads(candidates: Dict[str, WriterCandidate]) -> Dict[str, Any]:
    return {
        worker_id: {
            "current_sql": candidate.current_sql,
            "version": candidate.version,
            "ok": candidate.result_ok(),
            "error": candidate.error,
            "report": candidate.report,
            "last_action": candidate.last_action,
            "result": compact_exec_result(candidate.current_result),
            "exec_result": candidate.current_result,
            "history": candidate.history,
        }
        for worker_id, candidate in candidates.items()
    }


def candidate_trace_summaries(candidates: Dict[str, WriterCandidate]) -> List[Dict[str, Any]]:
    return [
        {
            "worker": candidate.worker_id,
            "version": candidate.version,
            "ok": candidate.result_ok(),
            "rows": len(candidate.result_rows()),
            "columns": candidate.result_columns(),
            "error": candidate.error,
        }
        for candidate in candidates.values()
    ]


def build_result_factions(
    candidates: Dict[str, WriterCandidate],
    *,
    require_columns: bool,
    preview_rows: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[WriterCandidate]] = {}
    for candidate in candidates.values():
        if not candidate.current_sql or not candidate.result_ok():
            continue
        signature = result_signature(candidate.current_result, require_columns=require_columns)
        grouped.setdefault(signature, []).append(candidate)

    factions = []
    for signature, members in grouped.items():
        members = sorted(members, key=lambda item: item.worker_id)
        representative = members[0]
        factions.append(
            {
                "signature": signature,
                "representative_worker": representative.worker_id,
                "supporting_workers": [member.worker_id for member in members],
                "support_count": len(members),
                "candidate": candidate_chat_payload(representative, preview_rows=preview_rows),
            }
        )
    factions.sort(key=lambda item: (-int(item["support_count"]), str(item["representative_worker"])))
    return factions


def faction_trace_summaries(factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for faction in factions:
        candidate = faction.get("candidate") or {}
        execution = candidate.get("execution") or {}
        summaries.append(
            {
                "signature": faction.get("signature"),
                "representative": faction.get("representative_worker"),
                "supporting_workers": faction.get("supporting_workers"),
                "support_count": faction.get("support_count"),
                "rows": execution.get("row_count"),
                "columns": execution.get("columns"),
            }
        )
    return summaries


def faction_signature_for_worker(factions: List[Dict[str, Any]], worker_id: str) -> str:
    for faction in factions:
        if worker_id == faction.get("representative_worker") or worker_id in (faction.get("supporting_workers") or []):
            return str(faction.get("signature") or "")
    return ""


def result_signature(exec_result: Dict[str, Any], require_columns: bool = False) -> str:
    body = exec_result.get("result") or {}
    rows = body.get("rows") or []
    normalized_rows = sorted(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) for row in rows)
    payload: Dict[str, Any] = {"rows": normalized_rows}
    if require_columns:
        payload["columns"] = body.get("columns", [])
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def format_discovered_schema(state: SharedState) -> str:
    payload = []
    for table, ev in state.discovered.tables.items():
        payload.append(
            {
                "table": table,
                "confidence": ev.confidence,
                "columns": ev.columns,
                "primary_keys": ev.primary_keys,
                "foreign_keys": ev.foreign_keys,
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def sql_exec_payload(sql: str, exec_result: Dict[str, Any], preview_rows: int = 3) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"sql": sql}
    if exec_result.get("ok"):
        result = exec_result.get("result") or {}
        payload["columns"] = result.get("columns", [])
        payload["rows"] = len(result.get("rows") or [])
        rows = result.get("rows") or []
        payload["preview_rows"] = rows if preview_rows <= 0 else rows[:preview_rows]
        payload["preview_rows_shown"] = len(payload["preview_rows"])
        if result.get("warnings"):
            payload["warnings"] = result.get("warnings")
    else:
        payload["error"] = exec_result.get("error", "")
    return payload


def format_sql_history(state: SharedState, keep_last: int = 6) -> str:
    items = []
    for attempt in state.sql_attempts[-keep_last:]:
        result = attempt.result or {}
        preview = {}
        if result.get("ok"):
            res = result.get("result") or {}
            preview = {
                "columns": res.get("columns", []),
                "rows": (res.get("rows") or [])[:3],
                "truncated": res.get("truncated", False),
                "warnings": res.get("warnings", []),
            }
        else:
            preview = {"error": result.get("error", "")}
        items.append({"sql": attempt.sql, "status": attempt.status, "preview": preview})
    return json.dumps(items, ensure_ascii=False, indent=2)
