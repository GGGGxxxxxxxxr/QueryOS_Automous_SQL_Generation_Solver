from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from .llm import create_chat_completion, create_openai_client, is_fatal_llm_error, safe_llm_error
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
            "name": "AGREE",
            "description": "Agree that a target worker's current SQL should become the group consensus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_worker": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["target_worker", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "REVISE",
            "description": "Replace your own current SQL with a revised read-only SQLite query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["sql", "reason"],
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
    agreement_target: str = ""
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
        chatgroup_enabled: bool = True,
        chatgroup_max_rounds: int = 2,
        consensus_require_same_columns: bool = False,
        debug: bool = False,
        tracer: Optional[EventTracer] = None,
    ) -> None:
        self.client = create_openai_client(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.executor = executor or SQLiteExecutor()
        self.trace_sql_preview_rows = trace_sql_preview_rows if trace_sql_preview_rows >= 0 else 3
        self.parallel_workers = max(1, int(parallel_workers or 1))
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
                    "SQL_HISTORY:\n"
                    f"{format_sql_history(state)}\n"
                ),
            },
        ]

        last_sql = ""
        sqlite_exec_count = 0
        last_error = ""

        for turn in range(1, self.max_turns + 1):
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

    def _run_writer_group(self, state: SharedState, guidance: str) -> AgentReturn:
        global_step = state.step + 1
        worker_ids = [f"writer_{idx}" for idx in range(1, self.parallel_workers + 1)]
        self.tracer.emit(
            "writer_group_start",
            "SWA",
            "SQL writer group started.",
            global_step=global_step,
            payload={"workers": worker_ids, "guidance": guidance},
        )

        candidates = self._run_initial_group_workers(state, guidance, worker_ids)
        viable = [candidate for candidate in candidates.values() if candidate.current_sql and candidate.result_ok()]
        if not viable:
            report = "SQL writer group produced no executable SQL candidate."
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
                global_step=global_step,
                candidates=candidates,
                rounds=0,
                reason="Initial writer candidates disagreed and chatgroup is disabled.",
            )

        chat_history: List[Dict[str, Any]] = []
        for round_idx in range(1, self.chatgroup_max_rounds + 1):
            self.tracer.emit(
                "writer_group_round",
                "SWA",
                "SQL writer group chat round.",
                global_step=global_step,
                worker_step=round_idx,
                payload={"objective": "Reach agreement on one worker's current SQL or revise your own SQL."},
            )
            actions = self._collect_chat_actions(
                state=state,
                guidance=guidance,
                candidates=candidates,
                chat_history=chat_history,
                round_idx=round_idx,
            )
            revised_workers = []
            round_log = {"round": round_idx, "actions": []}
            for worker_id, action in actions.items():
                candidate = candidates[worker_id]
                action_name = str(action.get("action") or "").upper()
                reason = str(action.get("reason") or "").strip()
                if action_name == "REVISE":
                    sql = str(action.get("sql") or "").strip()
                    if not sql:
                        action_name = "AGREE"
                        reason = reason or "Revision omitted SQL; treating as no agreement."
                    else:
                        revised_workers.append(worker_id)
                        exec_result = self.executor.execute(state.db_path, sql)
                        candidate.version += 1
                        candidate.current_sql = sql
                        candidate.current_result = exec_result
                        candidate.ok = bool(exec_result.get("ok"))
                        candidate.error = str(exec_result.get("error") or "")
                        candidate.report = reason
                        candidate.last_action = "REVISE"
                        candidate.agreement_target = ""
                        candidate.reason = reason
                        candidate.history.append(
                            {
                                "round": round_idx,
                                "action": "REVISE",
                                "sql": sql,
                                "result": compact_exec_result(exec_result),
                                "reason": reason,
                            }
                        )
                        self.tracer.emit(
                            "writer_group_action",
                            "SWA",
                            "SQL writer revised its candidate.",
                            global_step=global_step,
                            worker_step=round_idx,
                            status="ok" if exec_result.get("ok") else "error",
                            payload={
                                "writer": worker_id,
                                "action": "REVISE",
                                "reason": reason,
                                "version": candidate.version,
                            },
                        )
                        self.tracer.emit(
                            "tool_result",
                            f"SWA-{worker_id}",
                            "Executed revised SQL.",
                            global_step=global_step,
                            worker_step=round_idx,
                            tool="SQLITE_EXEC",
                            status="ok" if exec_result.get("ok") else "error",
                            payload=sql_exec_payload(sql, exec_result, preview_rows=self.trace_sql_preview_rows),
                        )
                        round_log["actions"].append(
                            {
                                "worker": worker_id,
                                "action": "REVISE",
                                "reason": reason,
                                "version": candidate.version,
                                "result": compact_exec_result(exec_result),
                            }
                        )
                        continue

                target_worker = str(action.get("target_worker") or "").strip()
                if target_worker not in candidates:
                    target_worker = worker_id
                    reason = reason or "Invalid agreement target; defaulted to self."
                candidate.last_action = "AGREE"
                candidate.agreement_target = target_worker
                candidate.reason = reason
                self.tracer.emit(
                    "writer_group_action",
                    "SWA",
                    "SQL writer agreed with a candidate.",
                    global_step=global_step,
                    worker_step=round_idx,
                    status="ok",
                    payload={
                        "writer": worker_id,
                        "action": "AGREE",
                        "target_worker": target_worker,
                        "reason": reason,
                        "version": candidates[target_worker].version,
                    },
                )
                round_log["actions"].append(
                    {
                        "worker": worker_id,
                        "action": "AGREE",
                        "target_worker": target_worker,
                        "target_version": candidates[target_worker].version,
                        "reason": reason,
                    }
                )

            chat_history.append(round_log)
            if revised_workers:
                # A revised current SQL invalidates all prior agreement signatures.
                for candidate in candidates.values():
                    candidate.agreement_target = ""
                continue

            agreement_consensus = self._find_agreement_consensus(candidates)
            if agreement_consensus is not None:
                return self._commit_group_candidate(
                    state,
                    agreement_consensus,
                    candidates=candidates,
                    mode=f"chatgroup_agreement_round_{round_idx}",
                    report=(
                        "SQL writer group reached agreement on "
                        f"{agreement_consensus.worker_id}'s current SQL."
                    ),
                )

        result_consensus = self._find_result_consensus(candidates)
        if result_consensus is not None:
            return self._commit_group_candidate(
                state,
                result_consensus,
                candidates=candidates,
                mode="post_chat_result_consensus",
                report="SQL writer group did not unanimously agree, but final execution results are consistent.",
            )
        return self._writer_group_divergence(
            global_step=global_step,
            candidates=candidates,
            rounds=self.chatgroup_max_rounds,
            reason="Writer group exhausted chat rounds without consensus.",
        )

    def _run_initial_group_workers(
        self,
        state: SharedState,
        guidance: str,
        worker_ids: List[str],
    ) -> Dict[str, WriterCandidate]:
        original_attempt_count = len(state.sql_attempts)
        candidates: Dict[str, WriterCandidate] = {}
        with ThreadPoolExecutor(max_workers=len(worker_ids)) as pool:
            futures = {
                pool.submit(
                    self._run_initial_group_worker,
                    state,
                    guidance,
                    worker_id,
                    original_attempt_count,
                ): worker_id
                for worker_id in worker_ids
            }
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    candidate = future.result()
                except Exception as exc:
                    err = f"{worker_id} failed unexpectedly: {safe_llm_error(exc)}"
                    candidate = WriterCandidate(
                        worker_id=worker_id,
                        ok=False,
                        fatal=is_fatal_llm_error(exc),
                        error=err,
                        report=err,
                    )
                candidates[worker_id] = candidate
        return {worker_id: candidates[worker_id] for worker_id in worker_ids}

    def _run_initial_group_worker(
        self,
        state: SharedState,
        guidance: str,
        worker_id: str,
        original_attempt_count: int,
    ) -> WriterCandidate:
        worker_state = fork_shared_state_for_worker(state)
        ret = self._run_single_worker(
            worker_state,
            guidance,
            agent_label=f"SWA-{worker_id}",
            worker_identity=(
                f"You are {worker_id} in a parallel SQL writer group. "
                "Work independently on your forked state. Your intermediate SQL executions "
                "are local to you until the group reaches consensus."
            ),
        )
        return candidate_from_worker_state(worker_id, worker_state, ret, original_attempt_count)

    def _collect_chat_actions(
        self,
        *,
        state: SharedState,
        guidance: str,
        candidates: Dict[str, WriterCandidate],
        chat_history: List[Dict[str, Any]],
        round_idx: int,
    ) -> Dict[str, Dict[str, Any]]:
        actions: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = {
                pool.submit(
                    self._call_chat_worker,
                    state,
                    guidance,
                    candidates,
                    chat_history,
                    round_idx,
                    worker_id,
                ): worker_id
                for worker_id in candidates
            }
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    actions[worker_id] = future.result()
                except Exception as exc:
                    err = f"Chat action failed for {worker_id}: {safe_llm_error(exc)}"
                    actions[worker_id] = {
                        "action": "AGREE",
                        "target_worker": worker_id,
                        "reason": err,
                        "fatal": is_fatal_llm_error(exc),
                    }
        return {worker_id: actions[worker_id] for worker_id in candidates}

    def _call_chat_worker(
        self,
        state: SharedState,
        guidance: str,
        candidates: Dict[str, WriterCandidate],
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
                "action": "AGREE",
                "target_worker": worker_id,
                "reason": "No clear chat action was emitted; keeping my current SQL.",
            }
        tc = tool_calls[0]
        name = str(tc.function.name or "").upper()
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception as exc:
            return {
                "action": "AGREE",
                "target_worker": worker_id,
                "reason": f"Invalid chat action JSON; keeping my current SQL. {exc}",
            }
        if name == "REVISE":
            return {
                "action": "REVISE",
                "sql": str(args.get("sql") or "").strip(),
                "reason": str(args.get("reason") or "").strip(),
            }
        if name == "AGREE":
            return {
                "action": "AGREE",
                "target_worker": str(args.get("target_worker") or "").strip(),
                "reason": str(args.get("reason") or "").strip(),
            }
        return {
            "action": "AGREE",
            "target_worker": worker_id,
            "reason": f"Unknown chat action {name}; keeping my current SQL.",
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

    @staticmethod
    def _find_agreement_consensus(candidates: Dict[str, WriterCandidate]) -> Optional[WriterCandidate]:
        targets = [candidate.agreement_target for candidate in candidates.values()]
        if not targets or any(not target for target in targets):
            return None
        if len(set(targets)) != 1:
            return None
        target = targets[0]
        candidate = candidates.get(target)
        if candidate and candidate.current_sql and candidate.result_ok():
            return candidate
        return None

    def _commit_group_candidate(
        self,
        state: SharedState,
        candidate: WriterCandidate,
        *,
        candidates: Optional[Dict[str, WriterCandidate]] = None,
        mode: str,
        report: str,
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
            },
        )

    def _writer_group_divergence(
        self,
        *,
        global_step: int,
        candidates: Dict[str, WriterCandidate],
        rounds: int,
        reason: str,
    ) -> AgentReturn:
        self.tracer.emit(
            "writer_group_divergence",
            "SWA",
            reason,
            global_step=global_step,
            status="error",
            payload={"reason": reason, "rounds": rounds, "candidates": candidate_trace_summaries(candidates)},
        )
        return AgentReturn(
            agent=AgentName.SQL_WRITER,
            ok=False,
            report=reason,
            payload={
                "reason": "writer_group_divergence",
                "writer_group": candidate_payloads(candidates),
            },
        )

    def _call_llm(self, messages: List[Dict[str, Any]]) -> Any:
        response = create_chat_completion(
            self.client,
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
    chat_history: List[Dict[str, Any]],
    round_idx: int,
    worker_id: str,
    preview_rows: int,
) -> str:
    payload = {
        "worker_id": worker_id,
        "round": round_idx,
        "objective": "Reach consensus on one worker's current SQL candidate.",
        "question": state.question,
        "external_knowledge": state.external_knowledge,
        "manager_guidance": guidance,
        "discovered_schema": json.loads(format_discovered_schema(state)),
        "rules": [
            "Only current SQL candidates are eligible for agreement.",
            "If any worker revises SQL, the runtime executes it and old agreements are invalidated.",
            "AGREE target_worker should be one of the current worker ids.",
        ],
        "current_candidates": [
            candidate_chat_payload(candidate, preview_rows=preview_rows)
            for candidate in candidates.values()
        ],
        "chat_history": chat_history,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


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


def candidate_payloads(candidates: Dict[str, WriterCandidate]) -> Dict[str, Any]:
    return {
        worker_id: {
            "current_sql": candidate.current_sql,
            "version": candidate.version,
            "ok": candidate.result_ok(),
            "error": candidate.error,
            "report": candidate.report,
            "last_action": candidate.last_action,
            "agreement_target": candidate.agreement_target,
            "result": compact_exec_result(candidate.current_result),
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
