from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PlannerAction(str, Enum):
    CALL_SCHEMA_DISCOVERY = "CALL_SCHEMA_DISCOVERY"
    CALL_SQL_WRITER = "CALL_SQL_WRITER"
    FINISH = "FINISH"


class AgentName(str, Enum):
    PLANNER = "planner"
    SCHEMA_DISCOVERY = "schema_discovery"
    SQL_WRITER = "sql_writer"
    SQL_VALIDATOR = "sql_validator"


class WorkflowStatus(str, Enum):
    NEED_SCHEMA = "NEED_SCHEMA"
    SCHEMA_READY = "SCHEMA_READY"
    SQL_CANDIDATE_READY = "SQL_CANDIDATE_READY"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    SQL_VALIDATED = "SQL_VALIDATED"
    FINISHED = "FINISHED"


@dataclass
class TableEvidence:
    table: str
    columns: List[Dict[str, str]] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class DiscoveredSchema:
    tables: Dict[str, TableEvidence] = field(default_factory=dict)


@dataclass
class SQLAttempt:
    sql: str
    status: str
    result: Optional[Dict[str, Any]] = None


@dataclass
class ValidationAttempt:
    sql_attempt_idx: int
    status: str
    issues: List[Dict[str, Any]] = field(default_factory=list)
    feedback: str = ""
    report: str = ""
    confidence: str = ""


@dataclass
class PlannerDecision:
    action: PlannerAction
    guidance: str = ""


@dataclass
class AgentReturn:
    agent: AgentName
    ok: bool = True
    report: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStep:
    step_idx: int
    decision: PlannerDecision
    agent_return: Optional[AgentReturn] = None
    state_delta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedState:
    question: str
    db_path: str
    db_id: str = ""
    external_knowledge: str = ""
    metadata_display: str = ""
    workflow_status: WorkflowStatus = WorkflowStatus.NEED_SCHEMA
    discovered: DiscoveredSchema = field(default_factory=DiscoveredSchema)
    sql_attempts: List[SQLAttempt] = field(default_factory=list)
    validation_attempts: List[ValidationAttempt] = field(default_factory=list)
    planner_trace: List[TraceStep] = field(default_factory=list)
    step: int = 0
    max_steps: int = 8


@dataclass
class SQLGenerationResult:
    question: str
    final_sql: str
    rows: List[List[Any]]
    columns: List[str]
    ok: bool
    report: str
    sql_attempts: List[SQLAttempt]
    validation_attempts: List[ValidationAttempt]
    discovered_schema: Dict[str, TableEvidence]
    planner_trace: List[TraceStep]
    workflow_status: WorkflowStatus = WorkflowStatus.NEED_SCHEMA
    trace_events: List[Dict[str, Any]] = field(default_factory=list)
    gold_sql: str = ""
    gold_result: Optional[Dict[str, Any]] = None
    gold_match: Optional[bool] = None
    gold_comparison: Dict[str, Any] = field(default_factory=dict)
