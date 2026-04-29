from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .llm import create_chat_completion, create_llm_backend, is_fatal_llm_error, safe_llm_error
from .prompts import build_sql_validator_system_prompt
from .state import AgentName, AgentReturn, SharedState, ValidationAttempt, WorkflowStatus
from .tracing import EventTracer, NULL_TRACER

logger = logging.getLogger(__name__)


SQL_VALIDATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "VALIDATION_PASS",
            "description": "Mark the latest SQL candidate as validated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "VALIDATION_FAIL",
            "description": "Reject the latest SQL candidate and explain the validation feedback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["minor", "blocking"]},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "missing_constraint",
                                        "wrong_boolean_logic",
                                        "missing_select_column",
                                        "extra_select_column",
                                        "null_ranking",
                                        "wrong_join",
                                        "schema_insufficient",
                                        "suspicious_result",
                                        "execution_error",
                                        "other",
                                    ],
                                },
                                "detail": {"type": "string"},
                            },
                            "required": ["type", "detail"],
                            "additionalProperties": False,
                        },
                    },
                    "feedback": {"type": "string"},
                },
                "required": ["severity", "issues", "feedback"],
                "additionalProperties": False,
            },
        },
    },
]


class SQLValidatorAgent:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        debug: bool = False,
        tracer: Optional[EventTracer] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.client = llm_client or create_llm_backend(provider="openai", api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug
        self.tracer = tracer or NULL_TRACER

    def run(self, state: SharedState) -> AgentReturn:
        global_step = state.step + 1
        sql_attempt_idx = len(state.sql_attempts)
        latest_attempt = state.sql_attempts[-1] if state.sql_attempts else None

        self.tracer.emit(
            "validation_start",
            "SVA",
            "Starting SQL validation gate.",
            global_step=global_step,
            payload={"sql_attempt_idx": sql_attempt_idx},
        )

        if latest_attempt is None:
            return self._record_fail(
                state,
                sql_attempt_idx=0,
                issues=[{"type": "execution_error", "detail": "No SQL attempt exists to validate."}],
                feedback="No SQL attempt exists to validate. The planner should request a SQL candidate first.",
                report="Validation failed because there is no SQL attempt.",
            )

        self.tracer.emit(
            "worker_start",
            "SVA",
            "SQL validator worker started.",
            global_step=global_step,
            payload={"guidance": "Validate the latest SQL candidate before planner can finish."},
        )
        blocking_issues = blocking_result_issues(latest_attempt.result or {})
        if blocking_issues:
            self.tracer.emit(
                "worker_step_start",
                "SVA",
                "Running deterministic validation precheck.",
                global_step=global_step,
                worker_step=1,
            )
            feedback = (
                "The latest SQL result is not acceptable: "
                + "; ".join(issue["detail"] for issue in blocking_issues)
                + ". Planner should request a revised SQL candidate."
            )
            return self._record_fail(
                state,
                sql_attempt_idx=sql_attempt_idx,
                issues=blocking_issues,
                feedback=feedback,
                report="Validation failed deterministic result precheck.",
                worker_step=1,
            )

        self.tracer.emit(
            "worker_step_start",
            "SVA",
            "Requesting validation decision from LLM.",
            global_step=global_step,
            worker_step=1,
        )

        messages = [
            {"role": "system", "content": build_sql_validator_system_prompt()},
            {
                "role": "user",
                "content": (
                    "Validate the latest SQL candidate.\n\n"
                    f"{format_validator_context(state)}\n\n"
                    "Call exactly one validation tool."
                ),
            },
        ]

        try:
            message = self._call_llm(messages)
        except Exception as exc:
            err = f"SQL validator LLM call failed: {safe_llm_error(exc)}"
            self.tracer.emit(
                "worker_error",
                "SVA",
                err,
                global_step=global_step,
                worker_step=1,
                status="error",
                payload={"error": err},
            )
            attempt = ValidationAttempt(
                sql_attempt_idx=sql_attempt_idx,
                status="error",
                issues=[{"type": "other", "detail": err}],
                feedback="Validation could not run. Planner should inspect the latest SQL candidate and decide whether to retry.",
                report=err,
            )
            state.validation_attempts.append(attempt)
            state.workflow_status = WorkflowStatus.VALIDATION_FAILED
            self.tracer.emit(
                "worker_finish",
                "SVA",
                "SQL validator stopped because the LLM call failed.",
                global_step=global_step,
                worker_step=1,
                status="error",
                payload=validation_payload(attempt),
            )
            return AgentReturn(
                agent=AgentName.SQL_VALIDATOR,
                ok=False,
                report=err,
                payload={"reason": "llm_call_failed", "fatal": is_fatal_llm_error(exc)},
            )

        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        if not tool_calls or len(tool_calls) != 1:
            content_preview = _message_content_preview(message)
            detail = "Validator did not emit exactly one validation tool call."
            if content_preview:
                detail += f" Model content preview: {content_preview}"
            return self._record_fail(
                state,
                sql_attempt_idx=sql_attempt_idx,
                issues=[{"type": "other", "detail": detail}],
                feedback="Validator did not emit a clear decision. Planner should inspect the latest SQL candidate and decide whether to retry.",
                report="Validation failed because validator emitted no clear decision.",
            )

        tc = tool_calls[0]
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception as exc:
            return self._record_fail(
                state,
                sql_attempt_idx=sql_attempt_idx,
                issues=[{"type": "other", "detail": f"Validator emitted invalid JSON: {exc}"}],
                feedback="Validator emitted invalid JSON. Planner should inspect the latest SQL candidate and decide whether to retry.",
                report="Validation failed because validator emitted invalid JSON.",
            )

        self.tracer.emit(
            "worker_step_tools",
            "SVA",
            "LLM emitted a validation tool call.",
            global_step=global_step,
            worker_step=1,
            payload={"tools": [name]},
        )

        if name == "VALIDATION_PASS":
            reason = str(args.get("reason") or "SQL candidate passed validation.").strip()
            confidence = str(args.get("confidence") or "").strip()
            attempt = ValidationAttempt(
                sql_attempt_idx=sql_attempt_idx,
                status="pass",
                report=reason,
                confidence=confidence,
            )
            state.validation_attempts.append(attempt)
            state.workflow_status = WorkflowStatus.SQL_VALIDATED
            self.tracer.emit(
                "tool_result",
                "SVA",
                "Executed VALIDATION_PASS.",
                global_step=global_step,
                worker_step=1,
                tool="VALIDATION_PASS",
                status="ok",
                payload=validation_payload(attempt),
            )
            self.tracer.emit(
                "worker_finish",
                "SVA",
                "SQL validator worker finished.",
                global_step=global_step,
                worker_step=1,
                status="ok",
                payload=validation_payload(attempt),
            )
            return AgentReturn(
                agent=AgentName.SQL_VALIDATOR,
                ok=True,
                report=reason,
                payload={"validation_status": "pass", "sql_attempt_idx": sql_attempt_idx},
            )

        if name == "VALIDATION_FAIL":
            issues = clean_issues(args.get("issues"))
            feedback = str(args.get("feedback") or "").strip()
            if not feedback:
                feedback = "The latest SQL candidate does not sufficiently satisfy the question and evidence."
            severity = str(args.get("severity") or "blocking").strip()
            report = f"Validation failed ({severity})."
            return self._record_fail(
                state,
                sql_attempt_idx=sql_attempt_idx,
                issues=issues,
                feedback=feedback,
                report=report,
                worker_step=1,
            )

        return self._record_fail(
            state,
            sql_attempt_idx=sql_attempt_idx,
            issues=[{"type": "other", "detail": f"Validator emitted unknown tool: {name}"}],
            feedback="Validator emitted an unknown tool. Planner should inspect the latest SQL candidate and decide whether to retry.",
            report="Validation failed because validator emitted an unknown tool.",
            worker_step=1,
        )

    def _record_fail(
        self,
        state: SharedState,
        *,
        sql_attempt_idx: int,
        issues: List[Dict[str, Any]],
        feedback: str,
        report: str,
        worker_step: int = 1,
    ) -> AgentReturn:
        global_step = state.step + 1
        attempt = ValidationAttempt(
            sql_attempt_idx=sql_attempt_idx,
            status="fail",
            issues=issues,
            feedback=feedback,
            report=report,
        )
        state.validation_attempts.append(attempt)
        state.workflow_status = WorkflowStatus.VALIDATION_FAILED
        self.tracer.emit(
            "tool_result",
            "SVA",
            "Executed VALIDATION_FAIL.",
            global_step=global_step,
            worker_step=worker_step,
            tool="VALIDATION_FAIL",
            status="error",
            payload=validation_payload(attempt),
        )
        self.tracer.emit(
            "worker_finish",
            "SVA",
            "SQL validator worker finished.",
            global_step=global_step,
            worker_step=worker_step,
            status="error",
            payload=validation_payload(attempt),
        )
        return AgentReturn(
            agent=AgentName.SQL_VALIDATOR,
            ok=True,
            report=report,
            payload={
                "validation_status": "fail",
                "sql_attempt_idx": sql_attempt_idx,
                "feedback": feedback,
                "issues": issues,
            },
        )

    def _call_llm(self, messages: List[Dict[str, Any]]) -> Any:
        response = create_chat_completion(
            self.client,
            role="sql_validator",
            model=self.model,
            messages=messages,
            tools=SQL_VALIDATOR_TOOLS,
            tool_choice="required",
            parallel_tool_calls=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message


def format_validator_context(state: SharedState) -> str:
    latest_attempt = state.sql_attempts[-1] if state.sql_attempts else None
    latest_body = (latest_attempt.result or {}).get("result") if latest_attempt and latest_attempt.result else {}
    latest_error = (latest_attempt.result or {}).get("error", "") if latest_attempt and latest_attempt.result else ""
    latest_rows = ((latest_body or {}).get("rows") or [])
    latest_warnings = list((latest_body or {}).get("warnings", []))
    if latest_rows and any(any(cell is None for cell in row) for row in latest_rows):
        latest_warnings.append("result contains NULL values")
    latest = {
        "attempt_idx": len(state.sql_attempts),
        "sql": latest_attempt.sql if latest_attempt else "",
        "status": latest_attempt.status if latest_attempt else "",
        "ok": bool((latest_attempt.result or {}).get("ok")) if latest_attempt and latest_attempt.result else False,
        "error": latest_error,
        "columns": (latest_body or {}).get("columns", []),
        "row_count": len(latest_rows),
        "rows_preview": latest_rows[:5],
        "warnings": latest_warnings,
    }
    return json.dumps(
        {
            "question": state.question,
            "external_knowledge": state.external_knowledge,
            "workflow_status": state.workflow_status.value,
            "discovered_schema": [
                {
                    "table": table,
                    "confidence": ev.confidence,
                    "columns": ev.columns,
                    "primary_keys": ev.primary_keys,
                    "foreign_keys": ev.foreign_keys,
                }
                for table, ev in state.discovered.tables.items()
            ],
            "latest_sql_attempt": latest,
            "previous_validation_attempts": [
                {
                    "sql_attempt_idx": item.sql_attempt_idx,
                    "status": item.status,
                    "issues": item.issues,
                    "feedback": item.feedback,
                    "report": item.report,
                }
                for item in state.validation_attempts[-3:]
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def clean_issues(value: Any) -> List[Dict[str, Any]]:
    issues = value if isinstance(value, list) else []
    cleaned = []
    for item in issues:
        if not isinstance(item, dict):
            continue
        issue_type = str(item.get("type") or "other").strip() or "other"
        detail = str(item.get("detail") or "").strip()
        cleaned.append({"type": issue_type, "detail": detail})
    if not cleaned:
        cleaned.append({"type": "other", "detail": "Validator rejected the SQL candidate."})
    return cleaned


def _message_content_preview(message: Any, limit: int = 240) -> str:
    content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    text = " ".join(str(content or "").split())
    if len(text) > limit:
        return text[: max(0, limit - 3)] + "..."
    return text


def validation_payload(attempt: ValidationAttempt) -> Dict[str, Any]:
    return {
        "sql_attempt_idx": attempt.sql_attempt_idx,
        "validation_status": attempt.status,
        "issues": attempt.issues,
        "feedback": attempt.feedback,
        "report": attempt.report,
        "confidence": attempt.confidence,
    }


def blocking_result_issues(result: Dict[str, Any]) -> List[Dict[str, str]]:
    if not result.get("ok"):
        error = str(result.get("error") or "SQL execution failed.")
        return [{"type": "execution_error", "detail": error}]
    body = result.get("result") or {}
    rows = body.get("rows") or []
    if not rows:
        return [
            {
                "type": "suspicious_result",
                "detail": "SQL returned zero rows. Empty answer sets are not accepted as final answers.",
            }
        ]
    if any(any(cell is None for cell in row) for row in rows):
        return [
            {
                "type": "suspicious_result",
                "detail": "SQL returned NULL values. NULL answer values are not accepted.",
            }
        ]
    return []
