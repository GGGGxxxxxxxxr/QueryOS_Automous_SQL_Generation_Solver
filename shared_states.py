# Copyright (c) Microsoft. All rights reserved.
"""Shared types for the SQL Agentic System."""

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


@dataclass
class TableEvidence:
    table: str
    schema: Optional[str] = None
    columns: List[Dict[str, str]] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class DiscoveredSchema:
    # key: "schema.table"
    tables: Dict[str, TableEvidence] = field(default_factory=dict)


@dataclass
class SQLAttempt:
    sql: str
    status: str  # "proposed" | "executed_ok" | "executed_err"
    result: Optional[Any] = None


@dataclass
class PlannerDecision:
    action: PlannerAction
    guidance: str = ""


@dataclass
class AgentReturn:
    agent: AgentName
    ok: bool = True
    report: str = ""  # verbalized return for Planner understanding specifically
    payload: Dict[str, Any] = field(default_factory=dict)  # additional field for tracing


# -------- one step trace --------
@dataclass
class TraceStep:
    step_idx: int
    decision: PlannerDecision
    agent_return: Optional[AgentReturn] = None


@dataclass
class SharedGlobalState:
    question: str
    external_knowledge: str = ""  # [Optional] Extra info or guidance for agent
    db_id: str = ""  # db_name
    db_path: str = ""  # sqlite path for executor
    schema_metadata_path: str = ""  # file path for locally-stored table metadata
    metadata_display: str = ""  # result of `ls schema_metadata_path`

    discovered: DiscoveredSchema = field(default_factory=DiscoveredSchema)
    sql_attempts: List[SQLAttempt] = field(default_factory=list)

    step: int = 0  # current consumed steps (PLANNER specific)
    max_steps: int = 8  # Max allowed action counts (PLANNER specific)

    planner_trace: List[TraceStep] = field(default_factory=list)
