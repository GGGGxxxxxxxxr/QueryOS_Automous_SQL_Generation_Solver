from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .llm import create_chat_completion, create_llm_backend, is_fatal_llm_error, safe_llm_error
from .metadata import SchemaMetadataStore
from .prompts import build_planner_system_prompt
from .schema_discovery_agent import SchemaDiscoveryAgent
from .sql_writer import SQLWriterAgent
from .sql_validator import SQLValidatorAgent
from .sqlite_executor import SQLiteExecutor
from .state import (
    AgentName,
    AgentReturn,
    PlannerAction,
    PlannerDecision,
    SQLAttempt,
    SQLGenerationResult,
    SharedState,
    TraceStep,
    ValidationAttempt,
    WorkflowStatus,
)
from .state_diff import diff_state, snapshot_state, summarize_delta_for_planner
from .tracing import EventTracer, NULL_TRACER

logger = logging.getLogger(__name__)


PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "CALL_SCHEMA_DISCOVERY",
            "description": "Ask the schema discovery worker to update discovered_schema.",
            "parameters": {
                "type": "object",
                "properties": {"guidance": {"type": "string"}},
                "required": ["guidance"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "CALL_SQL_WRITER",
            "description": "Ask the SQL writer worker to generate and execute SQL.",
            "parameters": {
                "type": "object",
                "properties": {"guidance": {"type": "string"}},
                "required": ["guidance"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "SELECT_SUBMISSION_SQL",
            "description": (
                "Select one existing SQL writer group candidate as the current submission_SQL. "
                "Use only when current_shared_state.pending_writer_group exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "worker": {
                        "type": "string",
                        "description": "The worker id to select, such as writer_1.",
                    },
                    "guidance": {
                        "type": "string",
                        "description": "Short reason why this candidate should become submission_SQL.",
                    },
                },
                "required": ["worker", "guidance"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PLANNER_FINISH",
            "description": "Finish when the current submission_SQL result answers the whole question.",
            "parameters": {
                "type": "object",
                "properties": {"guidance": {"type": "string"}},
                "required": ["guidance"],
                "additionalProperties": False,
            },
        },
    },
]


class Planner:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        context_mode: str = "dispatch",
        debug: bool = False,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.client = llm_client or create_llm_backend(provider="openai", api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_mode = context_mode if context_mode in {"compact", "dispatch"} else "dispatch"
        self.debug = debug

    def decide(self, state: SharedState) -> PlannerDecision:
        try:
            response = create_chat_completion(
                self.client,
                role="planner",
                model=self.model,
                messages=[
                    {"role": "system", "content": build_planner_system_prompt()},
                    {
                        "role": "user",
                        "content": (
                            "Current manager context:\n"
                            f"{format_state_for_planner(state, mode=self.context_mode)}\n\n"
                            "Decide the next action with exactly one tool call."
                        ),
                    },
                ],
                tools=PLANNER_TOOLS,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls or len(tool_calls) != 1:
                raise ValueError("planner did not emit exactly one tool call")
            tc = tool_calls[0]
            args = json.loads(tc.function.arguments or "{}")
            guidance = str(args.get("guidance") or "").strip()
            if not guidance:
                guidance = "Proceed with the next necessary step."
            selected_worker = str(args.get("worker") or args.get("selected_worker") or "").strip()
            return planner_tool_to_decision(tc.function.name, guidance, selected_worker=selected_worker)
        except Exception as exc:
            err = safe_llm_error(exc)
            if is_fatal_llm_error(exc):
                raise RuntimeError(err) from exc
            logger.warning("[Planner] LLM decision failed; using deterministic fallback: %s", err)
            return fallback_planner_decision(state)


class AgenticSystem:
    def __init__(
        self,
        planner: Planner,
        schema_discovery_agent: SchemaDiscoveryAgent,
        sql_writer_agent: SQLWriterAgent,
        sql_validator_agent: Optional[SQLValidatorAgent],
        metadata: SchemaMetadataStore,
        tracer: Optional[EventTracer] = None,
        validation_mode: str = "auto",
        auto_finish_on_sql: bool = False,
        state_sql_preview_rows: int = 3,
        debug: bool = False,
    ) -> None:
        self.planner = planner
        self.sda = schema_discovery_agent
        self.swa = sql_writer_agent
        self.sva = sql_validator_agent
        self.metadata = metadata
        self.tracer = tracer or NULL_TRACER
        self.validation_mode = validation_mode if validation_mode in {"off", "auto"} else "auto"
        self.auto_finish_on_sql = auto_finish_on_sql
        self.state_sql_preview_rows = state_sql_preview_rows if state_sql_preview_rows >= 0 else 3
        self.debug = debug

    def run(self, state: SharedState) -> SQLGenerationResult:
        final_report = ""
        self.tracer.emit(
            "run_start",
            "manager",
            "Starting QueryOS run.",
            payload={"question": state.question, "db_path": state.db_path},
        )
        while state.step < state.max_steps:
            global_step = state.step + 1
            self.tracer.emit(
                "planner_decide_start",
                "manager",
                "Planner deciding next action.",
                global_step=global_step,
            )
            try:
                decision = self.planner.decide(state)
            except Exception as exc:
                err = f"Planner LLM call failed: {safe_llm_error(exc)}"
                trace = TraceStep(
                    step_idx=state.step,
                    decision=PlannerDecision(PlannerAction.FINISH, "Planner LLM call failed."),
                    agent_return=AgentReturn(
                        agent=AgentName.PLANNER,
                        ok=False,
                        report=err,
                        payload={"reason": "llm_call_failed", "fatal": True},
                    ),
                )
                state.planner_trace.append(trace)
                final_report = err
                self.tracer.emit(
                    "run_finish",
                    "manager",
                    "Planner stopped because the LLM call failed.",
                    global_step=global_step,
                    status="error",
                    payload={"error": err},
                )
                break
            before_state = self._snapshot_state(state)
            before_sql_attempt_count = len(state.sql_attempts)
            trace = TraceStep(step_idx=state.step, decision=decision)
            state.planner_trace.append(trace)
            action_message = {
                PlannerAction.CALL_SCHEMA_DISCOVERY: "Call schema discovery worker.",
                PlannerAction.CALL_SQL_WRITER: "Call SQL writer worker.",
                PlannerAction.SELECT_SUBMISSION_SQL: "Select writer group submission_SQL.",
                PlannerAction.FINISH: "Finish run.",
            }.get(decision.action, f"Call {decision.action.value}.")
            self.tracer.emit(
                "planner_decision",
                "manager",
                action_message,
                global_step=global_step,
                payload={
                    "action": decision.action.value,
                    "guidance": decision.guidance,
                    "selected_worker": decision.selected_worker,
                },
            )

            validation_ret: Optional[AgentReturn] = None
            if decision.action == PlannerAction.CALL_SCHEMA_DISCOVERY:
                state.pending_writer_group = {}
                ret = self.sda.run(state, decision.guidance, self.metadata)
                if ret.ok:
                    state.workflow_status = WorkflowStatus.SCHEMA_READY
            elif decision.action == PlannerAction.CALL_SQL_WRITER:
                state.pending_writer_group = {}
                ret = self.swa.run(state, decision.guidance)
                if ret.ok and latest_successful_attempt(state):
                    state.workflow_status = WorkflowStatus.SQL_CANDIDATE_READY
                if ret.payload.get("reason") == "writer_group_divergence" and state.pending_writer_group:
                    state.workflow_status = WorkflowStatus.PENDING_SUBMISSION_SELECTION
            elif decision.action == PlannerAction.SELECT_SUBMISSION_SQL:
                ret = select_pending_submission_sql(state, decision.selected_worker, decision.guidance)
                if ret.ok:
                    state.workflow_status = WorkflowStatus.SQL_CANDIDATE_READY
            elif decision.action == PlannerAction.FINISH:
                if can_finish(state, self.validation_mode):
                    state.workflow_status = WorkflowStatus.FINISHED
                    ret = AgentReturn(
                        agent=AgentName.PLANNER,
                        ok=True,
                        report=decision.guidance or "Planner finished.",
                    )
                    trace.agent_return = ret
                    final_report = ret.report
                    self.tracer.emit(
                        "run_finish",
                        "manager",
                        "Planner finished the run.",
                        global_step=global_step,
                        status="ok",
                        payload={"report": final_report},
                    )
                    break
                ret = AgentReturn(
                    agent=AgentName.PLANNER,
                    ok=False,
                    report=finish_blocked_report(state, self.validation_mode),
                    payload={"reason": "finish_blocked"},
                )
                record_finish_guard_failure(state, ret.report)
            else:
                ret = AgentReturn(
                    agent=AgentName.PLANNER,
                    ok=False,
                    report=f"Unknown planner action: {decision.action}",
                )

            trace.agent_return = ret
            final_report = ret.report
            self.tracer.emit(
                "worker_return",
                "manager",
                f"{agent_display_name(ret.agent)} returned.",
                global_step=global_step,
                status="ok" if ret.ok else "error",
                payload=make_json_safe({"report": ret.report, **ret.payload}),
            )
            if (
                self.validation_mode == "auto"
                and ret.ok
                and decision.action in {PlannerAction.CALL_SQL_WRITER, PlannerAction.SELECT_SUBMISSION_SQL}
                and len(state.sql_attempts) > before_sql_attempt_count
                and latest_successful_attempt(state)
            ):
                validation_ret = self.sva.run(state) if self.sva else missing_validator_return(state)
            state.step += 1
            state_delta_payload = diff_state(
                before_state,
                self._snapshot_state(state),
                writer=state_writer_label(ret, validation_ret),
            )
            trace.state_delta = summarize_delta_for_planner(state_delta_payload)
            self.tracer.emit(
                "state_delta",
                "manager",
                "Shared state updated.",
                global_step=global_step,
                status="ok" if ret.ok else "error",
                payload=state_delta_payload,
            )

            # Let the planner see one worker failure and potentially recover.
            if ret.payload.get("fatal") or (validation_ret and validation_ret.payload.get("fatal")):
                break
            if not ret.ok and state.step >= state.max_steps:
                break

            if (
                self.auto_finish_on_sql
                and self.validation_mode == "off"
                and ret.ok
                and decision.action in {PlannerAction.CALL_SQL_WRITER, PlannerAction.SELECT_SUBMISSION_SQL}
            ):
                latest = latest_successful_attempt(state)
                if latest and not result_is_suspicious(latest.result or {}):
                    finish_trace = TraceStep(
                        step_idx=state.step,
                        decision=PlannerDecision(
                            action=PlannerAction.FINISH,
                            guidance="Latest SQL execution succeeded and returned a usable result.",
                        ),
                        agent_return=AgentReturn(
                            agent=AgentName.PLANNER,
                            ok=True,
                            report="Latest SQL execution succeeded and returned a usable result.",
                        ),
                    )
                    state.planner_trace.append(finish_trace)
                    final_report = finish_trace.agent_return.report
                    self.tracer.emit(
                        "run_finish",
                        "manager",
                        "Auto-finished after successful SQL execution.",
                        global_step=state.step,
                        status="ok",
                        payload={"report": final_report},
                    )
                    break

        final_attempt = latest_successful_attempt(state)
        result = final_attempt.result if final_attempt and final_attempt.result else {}
        sql_result = result.get("result") or {}
        final_sql = final_attempt.sql if final_attempt else ""
        return SQLGenerationResult(
            question=state.question,
            final_sql=final_sql,
            rows=sql_result.get("rows", []),
            columns=sql_result.get("columns", []),
            ok=bool(final_sql),
            report=final_report,
            sql_attempts=state.sql_attempts,
            validation_attempts=state.validation_attempts,
            discovered_schema=state.discovered.tables,
            planner_trace=state.planner_trace,
            workflow_status=state.workflow_status,
            trace_events=list(self.tracer.events),
        )

    def _snapshot_state(self, state: SharedState) -> Dict[str, Any]:
        return snapshot_state(state, sql_preview_rows=self.state_sql_preview_rows)


class QueryOS:
    """High-level API for the cleaned SQL generation agent."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        base_url: Optional[str] = None,
        planner_model: Optional[str] = None,
        schema_model: Optional[str] = None,
        sql_model: Optional[str] = None,
        validator_model: Optional[str] = None,
        temperature: float = 0.2,
        max_steps: int = 8,
        planner_max_tokens: int = 1024,
        schema_max_tokens: int = 4096,
        schema_max_turns: int = 6,
        schema_max_tool_calls_per_turn: int = 4,
        schema_read_table_summary_max_cols: int = 30,
        schema_trace_column_preview_limit: int = 8,
        schema_parallel_workers: int = 1,
        sql_max_tokens: int = 4096,
        sql_max_turns: int = 8,
        sql_parallel_workers: int = 1,
        sql_chatgroup_enabled: bool = True,
        sql_chatgroup_max_rounds: int = 2,
        sql_consensus_require_same_columns: bool = False,
        validator_max_tokens: int = 2048,
        debug: bool = False,
        live_trace: bool = False,
        trace_json_path: Optional[str] = None,
        trace_style: str = "pretty",
        trace_color: str = "auto",
        trace_max_chars: Optional[int] = None,
        trace_sql_preview_rows: int = 3,
        trace_gold_preview_rows: int = 3,
        trace_result_cell_max_width: int = 32,
        state_view: str = "diff",
        planner_context: str = "dispatch",
        validation_mode: str = "auto",
        auto_finish_on_sql: bool = False,
        provider: str = "openai",
        llm_router_config: Optional[Dict[str, Any]] = None,
        llm_timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider
        self.llm_router_config = llm_router_config or {}
        self.llm_timeout = float(llm_timeout) if llm_timeout else None
        self.model = model
        self.planner_model = planner_model or model
        self.schema_model = schema_model or model
        self.sql_model = sql_model or model
        self.validator_model = validator_model or model
        self.temperature = temperature
        self.max_steps = max_steps
        self.planner_max_tokens = planner_max_tokens
        self.schema_max_tokens = schema_max_tokens
        self.schema_max_turns = schema_max_turns
        self.schema_max_tool_calls_per_turn = schema_max_tool_calls_per_turn
        self.schema_read_table_summary_max_cols = schema_read_table_summary_max_cols
        self.schema_trace_column_preview_limit = schema_trace_column_preview_limit
        self.schema_parallel_workers = max(1, int(schema_parallel_workers or 1))
        self.sql_max_tokens = sql_max_tokens
        self.sql_max_turns = sql_max_turns
        self.sql_parallel_workers = max(1, int(sql_parallel_workers or 1))
        self.sql_chatgroup_enabled = bool(sql_chatgroup_enabled)
        self.sql_chatgroup_max_rounds = max(0, int(sql_chatgroup_max_rounds or 0))
        self.sql_consensus_require_same_columns = bool(sql_consensus_require_same_columns)
        self.validator_max_tokens = validator_max_tokens
        self.debug = debug
        self.live_trace = live_trace
        self.trace_json_path = trace_json_path
        self.trace_style = trace_style
        self.trace_color = trace_color
        self.trace_max_chars = trace_max_chars
        self.trace_sql_preview_rows = trace_sql_preview_rows if trace_sql_preview_rows >= 0 else 3
        self.trace_gold_preview_rows = trace_gold_preview_rows if trace_gold_preview_rows >= 0 else 3
        self.trace_result_cell_max_width = trace_result_cell_max_width
        self.state_view = state_view
        self.planner_context = planner_context if planner_context in {"compact", "dispatch"} else "dispatch"
        self.validation_mode = validation_mode if validation_mode in {"off", "auto"} else "auto"
        self.auto_finish_on_sql = auto_finish_on_sql

    def generate(
        self,
        question: str,
        db_path: str,
        schema_metadata_path: Optional[str] = None,
        external_knowledge: str = "",
        db_id: str = "",
        live_trace: Optional[bool] = None,
        trace_json_path: Optional[str] = None,
        trace_style: Optional[str] = None,
        trace_color: Optional[str] = None,
        trace_max_chars: Optional[int] = None,
        trace_sql_preview_rows: Optional[int] = None,
        trace_gold_preview_rows: Optional[int] = None,
        trace_result_cell_max_width: Optional[int] = None,
        state_view: Optional[str] = None,
        planner_context: Optional[str] = None,
        validation_mode: Optional[str] = None,
        auto_finish_on_sql: Optional[bool] = None,
        golden_sql: str = "",
    ) -> SQLGenerationResult:
        metadata = (
            SchemaMetadataStore.from_path(schema_metadata_path)
            if schema_metadata_path
            else SchemaMetadataStore.from_sqlite(db_path)
        )
        state = SharedState(
            question=question,
            db_path=db_path,
            db_id=db_id,
            external_knowledge=external_knowledge,
            metadata_display=metadata.display(),
            max_steps=self.max_steps,
        )
        executor = SQLiteExecutor()
        tracer = EventTracer(
            live=self.live_trace if live_trace is None else live_trace,
            style=trace_style or self.trace_style,
            color=trace_color or self.trace_color,
            max_chars=trace_max_chars if trace_max_chars is not None else self.trace_max_chars,
            state_view=state_view or self.state_view,
            result_cell_max_width=(
                trace_result_cell_max_width
                if trace_result_cell_max_width is not None
                else self.trace_result_cell_max_width
            ),
        )
        sql_preview_rows = trace_sql_preview_rows if trace_sql_preview_rows is not None else self.trace_sql_preview_rows
        gold_preview_rows = trace_gold_preview_rows if trace_gold_preview_rows is not None else self.trace_gold_preview_rows
        llm_backend = create_llm_backend(
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            router_config=self.llm_router_config,
            timeout=self.llm_timeout,
            tracer=tracer,
        )
        system = AgenticSystem(
            planner=Planner(
                model=self.planner_model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=min(self.temperature, 0.2),
                max_tokens=self.planner_max_tokens,
                context_mode=planner_context or self.planner_context,
                debug=self.debug,
                llm_client=llm_backend,
            ),
            schema_discovery_agent=SchemaDiscoveryAgent(
                model=self.schema_model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.schema_max_tokens,
                max_turns=self.schema_max_turns,
                max_tool_calls_per_turn=self.schema_max_tool_calls_per_turn,
                read_table_summary_max_cols=self.schema_read_table_summary_max_cols,
                trace_column_preview_limit=self.schema_trace_column_preview_limit,
                parallel_workers=self.schema_parallel_workers,
                debug=self.debug,
                tracer=tracer,
                llm_client=llm_backend,
            ),
            sql_writer_agent=SQLWriterAgent(
                model=self.sql_model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.sql_max_tokens,
                max_turns=self.sql_max_turns,
                executor=executor,
                trace_sql_preview_rows=sql_preview_rows,
                parallel_workers=self.sql_parallel_workers,
                chatgroup_enabled=self.sql_chatgroup_enabled,
                chatgroup_max_rounds=self.sql_chatgroup_max_rounds,
                consensus_require_same_columns=self.sql_consensus_require_same_columns,
                debug=self.debug,
                tracer=tracer,
                llm_client=llm_backend,
            ),
            sql_validator_agent=SQLValidatorAgent(
                model=self.validator_model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=min(self.temperature, 0.2),
                max_tokens=self.validator_max_tokens,
                debug=self.debug,
                tracer=tracer,
                llm_client=llm_backend,
            ),
            metadata=metadata,
            tracer=tracer,
            validation_mode=validation_mode or self.validation_mode,
            auto_finish_on_sql=self.auto_finish_on_sql if auto_finish_on_sql is None else auto_finish_on_sql,
            state_sql_preview_rows=sql_preview_rows,
            debug=self.debug,
        )
        result = system.run(state)
        if golden_sql.strip():
            self._run_golden_sql_check(
                result=result,
                db_path=db_path,
                golden_sql=golden_sql,
                executor=executor,
                tracer=tracer,
                preview_rows=gold_preview_rows,
            )
            result.trace_events = list(tracer.events)
        dump_path = trace_json_path if trace_json_path is not None else self.trace_json_path
        if dump_path:
            tracer.dump(dump_path, result_to_dict(result))
        return result

    def run(self, *args: Any, **kwargs: Any) -> SQLGenerationResult:
        return self.generate(*args, **kwargs)

    @staticmethod
    def _run_golden_sql_check(
        result: SQLGenerationResult,
        db_path: str,
        golden_sql: str,
        executor: SQLiteExecutor,
        tracer: EventTracer,
        preview_rows: int = 3,
    ) -> None:
        gold_sql = golden_sql.strip()
        result.gold_sql = gold_sql
        tracer.emit(
            "gold_start",
            "gold",
            "Executing golden SQL for comparison.",
            payload={"sql": gold_sql},
        )
        gold_exec = executor.execute(db_path, gold_sql)
        result.gold_result = gold_exec
        gold_rows = ((gold_exec.get("result") or {}).get("rows") or [])
        gold_preview = gold_rows if preview_rows <= 0 else gold_rows[:preview_rows]
        predicted_preview = result.rows if preview_rows <= 0 else result.rows[:preview_rows]

        predicted_result = {
            "ok": bool(result.final_sql),
            "result": {
                "columns": result.columns,
                "rows": result.rows,
            },
            "error": "",
        }
        comparison = compare_sql_execution_results(predicted_result, gold_exec)
        result.gold_comparison = comparison
        result.gold_match = bool(comparison.get("exact_rows_match") or comparison.get("unordered_rows_match"))

        tracer.emit(
            "gold_result",
            "gold",
            "Golden SQL execution completed.",
            status="ok" if gold_exec.get("ok") else "error",
            payload={
                "sql": gold_sql,
                "columns": (gold_exec.get("result") or {}).get("columns", []),
                "rows": len((gold_exec.get("result") or {}).get("rows") or []),
                "preview_rows": gold_preview,
                "error": gold_exec.get("error", ""),
                "gold_match": result.gold_match,
                "exact_match": comparison.get("exact_rows_match"),
                "unordered_match": comparison.get("unordered_rows_match"),
                "predicted_preview": predicted_preview,
                "gold_preview": gold_preview,
            },
        )


def planner_tool_to_decision(tool_name: str, guidance: str, selected_worker: str = "") -> PlannerDecision:
    if tool_name == "CALL_SCHEMA_DISCOVERY":
        return PlannerDecision(PlannerAction.CALL_SCHEMA_DISCOVERY, guidance)
    if tool_name == "CALL_SQL_WRITER":
        return PlannerDecision(PlannerAction.CALL_SQL_WRITER, guidance)
    if tool_name == "SELECT_SUBMISSION_SQL":
        return PlannerDecision(PlannerAction.SELECT_SUBMISSION_SQL, guidance, selected_worker=selected_worker)
    if tool_name == "PLANNER_FINISH":
        return PlannerDecision(PlannerAction.FINISH, guidance)
    raise ValueError(f"unknown planner tool: {tool_name}")


def select_pending_submission_sql(state: SharedState, worker: str, guidance: str) -> AgentReturn:
    worker = str(worker or "").strip()
    pending = state.pending_writer_group or {}
    candidates = pending.get("candidates") or pending.get("writer_group") or {}
    if not pending or not candidates:
        return AgentReturn(
            agent=AgentName.PLANNER,
            ok=False,
            report="Cannot select submission_SQL: no pending writer group candidates exist.",
            payload={"reason": "no_pending_writer_group"},
        )
    if not worker:
        return AgentReturn(
            agent=AgentName.PLANNER,
            ok=False,
            report="Cannot select submission_SQL: planner did not name a worker.",
            payload={"reason": "missing_selected_worker", "available_workers": sorted(candidates.keys())},
        )
    candidate = candidates.get(worker)
    if not candidate:
        return AgentReturn(
            agent=AgentName.PLANNER,
            ok=False,
            report=f"Cannot select submission_SQL: worker {worker} is not a pending candidate.",
            payload={"reason": "unknown_selected_worker", "available_workers": sorted(candidates.keys())},
        )

    sql = str(candidate.get("current_sql") or "").strip()
    exec_result = candidate.get("exec_result") or {}
    if not sql:
        return AgentReturn(
            agent=AgentName.PLANNER,
            ok=False,
            report=f"Cannot select submission_SQL from {worker}: candidate has no SQL.",
            payload={"reason": "selected_candidate_missing_sql", "selected_worker": worker},
        )
    if not exec_result.get("ok"):
        return AgentReturn(
            agent=AgentName.PLANNER,
            ok=False,
            report=f"Cannot select submission_SQL from {worker}: candidate did not execute successfully.",
            payload={
                "reason": "selected_candidate_not_executable",
                "selected_worker": worker,
                "error": exec_result.get("error", ""),
            },
        )

    state.sql_attempts.append(
        SQLAttempt(
            sql=sql,
            status="executed_ok",
            result=exec_result,
        )
    )
    state.pending_writer_group = {}
    return AgentReturn(
        agent=AgentName.PLANNER,
        ok=True,
        report=guidance or f"Manager selected {worker} as submission_SQL.",
        payload={
            "reason": "submission_sql_selected",
            "selected_worker": worker,
            "submission_SQL": sql,
        },
    )


def agent_display_name(agent: AgentName) -> str:
    if agent == AgentName.SCHEMA_DISCOVERY:
        return "Schema Discovery Agent"
    if agent == AgentName.SQL_WRITER:
        return "SQL Writer Agent"
    if agent == AgentName.PLANNER:
        return "Manager"
    return str(agent.value)


def state_writer_label(ret: AgentReturn, validation_ret: Optional[AgentReturn]) -> str:
    label = agent_display_name(ret.agent)
    if validation_ret is not None:
        return f"{label} + SQL Validator Agent"
    return label


def latest_validation_attempt(state: SharedState) -> Optional[Any]:
    return state.validation_attempts[-1] if state.validation_attempts else None


def can_finish(state: SharedState, validation_mode: str) -> bool:
    if state.pending_writer_group:
        return False
    latest = latest_successful_attempt(state)
    if not latest or result_is_suspicious(latest.result or {}):
        return False
    if validation_mode == "off":
        return True
    validation = latest_validation_attempt(state)
    return (
        state.workflow_status == WorkflowStatus.SQL_VALIDATED
        and validation is not None
        and validation.status == "pass"
        and validation.sql_attempt_idx == len(state.sql_attempts)
    )


def finish_blocked_report(state: SharedState, validation_mode: str) -> str:
    if state.pending_writer_group:
        return "Cannot finish: writer group has unresolved submission_SQL candidates."
    latest = latest_successful_attempt(state)
    if not latest:
        return "Cannot finish: no successful SQL attempt exists."
    suspicion_reasons = result_suspicion_reasons(latest.result or {})
    if suspicion_reasons:
        return f"Cannot finish: submission_SQL result is suspicious ({'; '.join(suspicion_reasons)})."
    if validation_mode != "off":
        validation = latest_validation_attempt(state)
        if validation is None:
            return "Cannot finish: submission_SQL has not been validated."
        if validation.status != "pass":
            return "Cannot finish: submission_SQL did not pass validation."
        if validation.sql_attempt_idx != len(state.sql_attempts):
            return "Cannot finish: latest validation does not correspond to the current submission_SQL."
    return "Cannot finish: workflow is not ready."


def record_finish_guard_failure(state: SharedState, report: str) -> None:
    latest = latest_successful_attempt(state)
    if latest is None:
        return
    reasons = result_suspicion_reasons(latest.result or {})
    if not reasons:
        return
    latest_idx = len(state.sql_attempts)
    latest_validation = latest_validation_attempt(state)
    if (
        latest_validation is not None
        and latest_validation.sql_attempt_idx == latest_idx
        and latest_validation.status == "fail"
        and latest_validation.report == "Manager finish guard rejected the submission_SQL."
    ):
        state.workflow_status = WorkflowStatus.VALIDATION_FAILED
        return
    state.validation_attempts.append(
        ValidationAttempt(
            sql_attempt_idx=latest_idx,
            status="fail",
            issues=[
                {
                    "type": "suspicious_result",
                    "detail": reason,
                }
                for reason in reasons
            ],
            feedback=(
                "Manager finish guard rejected the submission_SQL because "
                f"{'; '.join(reasons)}. Revise the submission_SQL instead of finishing."
            ),
            report="Manager finish guard rejected the submission_SQL.",
        )
    )
    state.workflow_status = WorkflowStatus.VALIDATION_FAILED


def missing_validator_return(state: SharedState) -> AgentReturn:
    state.validation_attempts.append(
        ValidationAttempt(
            sql_attempt_idx=len(state.sql_attempts),
            status="error",
            issues=[{"type": "other", "detail": "SQL validator is not configured."}],
            feedback="SQL validator is not configured. Planner should decide whether to retry or stop.",
            report="SQL validator is not configured.",
        )
    )
    state.workflow_status = WorkflowStatus.VALIDATION_FAILED
    return AgentReturn(
        agent=AgentName.SQL_VALIDATOR,
        ok=False,
        report="SQL validator is not configured.",
        payload={"validation_status": "error"},
    )


def fallback_planner_decision(state: SharedState) -> PlannerDecision:
    pending_worker = best_pending_submission_worker(state)
    if pending_worker:
        return PlannerDecision(
            PlannerAction.SELECT_SUBMISSION_SQL,
            (
                "Writer group did not reach one-faction consensus, but this pending candidate "
                "has the strongest available support and an executable result."
            ),
            selected_worker=pending_worker,
        )
    latest = state.sql_attempts[-1] if state.sql_attempts else None
    if latest is not None and result_is_suspicious(latest.result or {}):
        return PlannerDecision(
            PlannerAction.CALL_SQL_WRITER,
            "The submission_SQL result is suspicious or contains NULL values; revise the submission_SQL so the final answer has no NULL values.",
        )
    if state.workflow_status == WorkflowStatus.SQL_VALIDATED:
        return PlannerDecision(PlannerAction.FINISH, "submission_SQL passed validation.")
    latest_validation = latest_validation_attempt(state)
    if latest_validation and latest_validation.status in {"fail", "error"}:
        issue_types = {str(issue.get("type", "")) for issue in latest_validation.issues}
        if "schema_insufficient" in issue_types:
            return PlannerDecision(
                PlannerAction.CALL_SCHEMA_DISCOVERY,
                f"Validation feedback indicates schema is insufficient: {latest_validation.feedback}",
            )
        return PlannerDecision(
            PlannerAction.CALL_SQL_WRITER,
            f"Revise the submission_SQL using this validation feedback: {latest_validation.feedback}",
        )
    if not state.discovered.tables:
        return PlannerDecision(
            PlannerAction.CALL_SCHEMA_DISCOVERY,
            "Find the minimal tables, columns, and join keys needed for the question.",
        )
    if latest is None or latest.status != "executed_ok":
        return PlannerDecision(
            PlannerAction.CALL_SQL_WRITER,
            "Write SQL incrementally using discovered_schema, probing values before final filters.",
        )
    if result_is_suspicious(latest.result or {}):
        return PlannerDecision(
            PlannerAction.CALL_SQL_WRITER,
            "The submission_SQL result is suspicious; debug joins and filters incrementally.",
        )
    return PlannerDecision(PlannerAction.FINISH, "The submission_SQL result appears sufficient.")


def best_pending_submission_worker(state: SharedState) -> str:
    pending = state.pending_writer_group or {}
    candidates = pending.get("candidates") or {}
    if not candidates:
        return ""
    support_by_worker: Dict[str, int] = {}
    for faction in pending.get("factions") or []:
        support = int(faction.get("support_count") or 0)
        for worker in faction.get("supporting_workers") or []:
            support_by_worker[str(worker)] = support
        representative = str(faction.get("representative_worker") or "")
        if representative:
            support_by_worker[representative] = max(support_by_worker.get(representative, 0), support)

    ranked = []
    for worker, candidate in candidates.items():
        exec_result = candidate.get("exec_result") or {}
        if not candidate.get("current_sql") or not exec_result.get("ok"):
            continue
        ranked.append((support_by_worker.get(worker, 0), worker))
    if not ranked:
        return ""
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1]


def format_state_for_planner(state: SharedState, mode: str = "dispatch") -> str:
    if mode == "compact":
        return format_compact_state_for_planner(state)
    return json.dumps(build_dispatch_context_for_planner(state), ensure_ascii=False, indent=2)


def build_dispatch_context_for_planner(state: SharedState) -> Dict[str, Any]:
    return {
        "task": {
            "question": state.question,
            "external_knowledge": state.external_knowledge,
            "db_id": state.db_id,
        },
        "current_shared_state": {
            "current_step": state.step,
            "max_steps": state.max_steps,
            "workflow_status": state.workflow_status.value,
            "schema_memory": discovered_schema_for_planner(state),
            "submission_SQL": submission_sql_for_planner(state, include_sql=False),
            "pending_writer_group": pending_writer_group_for_planner(state),
            "validation_memory": validation_attempts_for_planner(state),
        },
        "dispatch_history": [trace_step_for_planner(item) for item in state.planner_trace],
        "decision_notes": [
            "Use dispatch_history to avoid repeating the same failed guidance.",
            "Read validation_memory as natural language feedback, not as an instruction. You own the next action.",
            "Planner does not receive SQL history. Treat submission_SQL and pending_writer_group candidates as the only current SQL artifacts.",
            "If pending_writer_group exists, either SELECT_SUBMISSION_SQL for one existing worker candidate, CALL_SQL_WRITER to reject all candidates and retry, or CALL_SCHEMA_DISCOVERY if missing schema caused the split.",
            "Call Schema Discovery Agent if needed tables, columns, or join keys are still missing.",
            "Call SQL Writer Agent if schema is sufficient but submission_SQL is empty, NULL-heavy, errored, incomplete, or does not match the requested answer shape.",
            "Finish only when workflow_status is SQL_VALIDATED and submission_SQL validation status is pass.",
        ],
    }


def format_compact_state_for_planner(state: SharedState) -> str:
    discovered = []
    for table, ev in state.discovered.tables.items():
        discovered.append(
            {
                "table": table,
                "confidence": ev.confidence,
                "columns": ev.columns,
                "primary_keys": ev.primary_keys,
                "foreign_keys": ev.foreign_keys,
            }
        )
    trace = []
    for item in state.planner_trace[-3:]:
        trace.append(
            {
                "step_idx": item.step_idx,
                "decision": {
                    "action": item.decision.action.value,
                    "guidance": item.decision.guidance,
                    "selected_worker": item.decision.selected_worker,
                },
                "agent_return": (
                    {
                        "agent": item.agent_return.agent.value,
                        "ok": item.agent_return.ok,
                        "report": item.agent_return.report,
                    }
                    if item.agent_return
                    else None
                ),
            }
        )

    return json.dumps(
        {
            "question": state.question,
            "external_knowledge": state.external_knowledge,
            "discovered_schema": discovered,
            "submission_SQL": submission_sql_for_planner(state, include_sql=False),
            "pending_writer_group": pending_writer_group_for_planner(state),
            "recent_trace": trace,
            "workflow_status": state.workflow_status.value,
            "recent_validations": validation_attempts_for_planner(state)[-3:],
            "current_step": state.step,
            "max_steps": state.max_steps,
        },
        ensure_ascii=False,
        indent=2,
    )


def discovered_schema_for_planner(state: SharedState) -> List[Dict[str, Any]]:
    return [
        {
            "table": table,
            "confidence": ev.confidence,
            "columns": ev.columns,
            "primary_keys": ev.primary_keys,
            "foreign_keys": ev.foreign_keys,
        }
        for table, ev in state.discovered.tables.items()
    ]


def submission_sql_for_planner(state: SharedState, include_sql: bool = False) -> Dict[str, Any]:
    if not state.sql_attempts:
        return {"exists": False}
    return sql_attempt_for_planner(len(state.sql_attempts), state.sql_attempts[-1], include_sql=include_sql)


def pending_writer_group_for_planner(state: SharedState) -> Dict[str, Any]:
    pending = state.pending_writer_group or {}
    candidates = pending.get("candidates") or {}
    if not pending or not candidates:
        return {"exists": False}

    candidate_payloads = []
    for worker, candidate in sorted(candidates.items(), key=lambda item: item[0]):
        compact_result = candidate.get("result") or {}
        candidate_payloads.append(
            {
                "worker": worker,
                "sql": candidate.get("current_sql", ""),
                "ok": candidate.get("ok", False),
                "columns": compact_result.get("columns", []),
                "row_count": compact_result.get("row_count", 0),
                "rows_preview": compact_result.get("preview_rows", []),
                "warnings": compact_result.get("warnings", []),
                "error": candidate.get("error", "") or compact_result.get("error", ""),
                "report": candidate.get("report", ""),
                "last_action": candidate.get("last_action", ""),
            }
        )

    return {
        "exists": True,
        "reason": pending.get("reason", ""),
        "rounds": pending.get("rounds", 0),
        "chat_rounds": pending.get("chat_rounds", 0),
        "factions": pending_writer_factions_for_planner(pending.get("factions") or []),
        "candidates": candidate_payloads,
        "chat_history": pending.get("chat_history") or [],
    }


def pending_writer_factions_for_planner(factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for faction in factions:
        candidate = faction.get("candidate") or {}
        execution = candidate.get("execution") or {}
        out.append(
            {
                "signature": faction.get("signature"),
                "representative_worker": faction.get("representative_worker"),
                "supporting_workers": faction.get("supporting_workers", []),
                "support_count": faction.get("support_count", 0),
                "columns": execution.get("columns", []),
                "row_count": execution.get("row_count", 0),
                "rows_preview": execution.get("preview_rows", []),
            }
        )
    return out


def sql_attempt_for_planner(idx: int, attempt: SQLAttempt, include_sql: bool = False) -> Dict[str, Any]:
    result = attempt.result or {}
    body = result.get("result") or {}
    rows = body.get("rows") or []
    payload = {
        "exists": True,
        "attempt_idx": idx,
        "status": attempt.status,
        "ok": bool(result.get("ok")),
        "error": result.get("error", ""),
        "columns": body.get("columns", []),
        "row_count": len(rows),
        "rows_preview": rows[:3],
        "truncated": body.get("truncated", False),
        "warnings": sql_attempt_warnings(result),
    }
    if include_sql:
        payload["sql"] = attempt.sql
    return payload


def sql_attempt_warnings(result: Dict[str, Any]) -> List[str]:
    body = result.get("result") or {}
    rows = body.get("rows") or []
    warnings = list(body.get("warnings") or [])
    if result.get("ok") and not rows and "result returned zero rows" not in warnings:
        warnings.append("result returned zero rows")
    if rows and any(any(cell is None for cell in row) for row in rows):
        if "result contains NULL values" not in warnings:
            warnings.append("result contains NULL values")
    if not result.get("ok") and result.get("error"):
        err = str(result.get("error"))
        if err not in warnings:
            warnings.append(err)
    return warnings


def validation_attempts_for_planner(state: SharedState) -> List[Dict[str, Any]]:
    return [
        {
            "validation_idx": idx,
            "sql_attempt_idx": item.sql_attempt_idx,
            "status": item.status,
            "issues": item.issues,
            "feedback": item.feedback,
            "report": item.report,
            "confidence": item.confidence,
        }
        for idx, item in enumerate(state.validation_attempts, start=1)
    ]


def trace_step_for_planner(item: TraceStep) -> Dict[str, Any]:
    return {
        "global_step": item.step_idx + 1,
        "manager_action": item.decision.action.value,
        "guidance_sent": item.decision.guidance,
        "selected_worker": item.decision.selected_worker,
        "worker_return": (
            {
                "worker": item.agent_return.agent.value,
                "ok": item.agent_return.ok,
                "report": item.agent_return.report,
            }
            if item.agent_return
            else None
        ),
        "state_delta": item.state_delta,
    }


def latest_successful_attempt(state: SharedState) -> Optional[Any]:
    for attempt in reversed(state.sql_attempts):
        if attempt.status == "executed_ok" and attempt.result and attempt.result.get("ok"):
            return attempt
    return None


def result_is_suspicious(result: Dict[str, Any]) -> bool:
    return bool(result_suspicion_reasons(result))


def result_suspicion_reasons(result: Dict[str, Any]) -> List[str]:
    if not result.get("ok"):
        return ["execution failed"]
    body = result.get("result") or {}
    rows = body.get("rows") or []
    if not rows:
        return ["result returned zero rows"]
    if any(any(cell is None for cell in row) for row in rows):
        return ["result contains NULL values"]
    return []


def compare_sql_execution_results(predicted: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    pred_body = predicted.get("result") or {}
    gold_body = gold.get("result") or {}
    pred_rows = pred_body.get("rows") or []
    gold_rows = gold_body.get("rows") or []
    pred_norm = normalize_rows_for_compare(pred_rows)
    gold_norm = normalize_rows_for_compare(gold_rows)
    exact = pred_norm == gold_norm
    unordered = sorted(pred_norm) == sorted(gold_norm)
    return {
        "predicted_ok": bool(predicted.get("ok")),
        "gold_ok": bool(gold.get("ok")),
        "exact_rows_match": bool(predicted.get("ok") and gold.get("ok") and exact),
        "unordered_rows_match": bool(predicted.get("ok") and gold.get("ok") and unordered),
        "predicted_row_count": len(pred_rows),
        "gold_row_count": len(gold_rows),
        "predicted_columns": pred_body.get("columns", []),
        "gold_columns": gold_body.get("columns", []),
        "gold_error": gold.get("error", ""),
    }


def normalize_rows_for_compare(rows: List[List[Any]]) -> List[str]:
    return [json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) for row in rows]


def result_to_dict(result: SQLGenerationResult) -> Dict[str, Any]:
    return make_json_safe({
        "question": result.question,
        "ok": result.ok,
        "report": result.report,
        "final_sql": result.final_sql,
        "columns": result.columns,
        "rows": result.rows,
        "workflow_status": result.workflow_status.value if hasattr(result.workflow_status, "value") else result.workflow_status,
        "sql_attempts": [asdict(item) for item in result.sql_attempts],
        "validation_attempts": [asdict(item) for item in result.validation_attempts],
        "discovered_schema": {k: asdict(v) for k, v in result.discovered_schema.items()},
        "planner_trace": [asdict(item) for item in result.planner_trace],
        "trace_events": result.trace_events,
        "gold_sql": result.gold_sql,
        "gold_result": result.gold_result,
        "gold_match": result.gold_match,
        "gold_comparison": result.gold_comparison,
    })


def make_json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return make_json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, set):
        return [make_json_safe(item) for item in sorted(value, key=str)]
    return value
