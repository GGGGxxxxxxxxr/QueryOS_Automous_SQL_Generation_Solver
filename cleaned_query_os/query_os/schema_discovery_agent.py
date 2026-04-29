from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .llm import create_chat_completion, create_llm_backend, is_fatal_llm_error, safe_llm_error
from .metadata import SchemaMetadataStore, normalize_name, parse_foreign_key
from .prompts import build_schema_discovery_system_prompt
from .state import AgentName, AgentReturn, SharedState, TableEvidence
from .tracing import EventTracer, NULL_TRACER

logger = logging.getLogger(__name__)


SCHEMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "SEARCH_METADATA",
            "description": "Search table metadata by keywords from the user question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string", "enum": ["OR", "AND"]},
                },
                "required": ["keywords"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "READ_TABLE_JSON",
            "description": "Read one table's compact metadata summary.",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string"}},
                "required": ["table"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "INTRODUCE_TABLE",
            "description": "Introduce a relevant table with only necessary columns and keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "desc": {"type": "string"},
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                    "primary_key": {"type": "array", "items": {"type": "string"}},
                    "foreign_keys": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "ref_table": {"type": "string"},
                                "ref_column": {"type": "string"},
                            },
                            "required": ["column", "ref_table", "ref_column"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["table", "columns"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ENRICH_TABLE",
            "description": "Add missing columns or foreign keys to a discovered table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "add_columns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "desc": {"type": "string"},
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                    "foreign_keys": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "ref_table": {"type": "string"},
                                "ref_column": {"type": "string"},
                            },
                            "required": ["column", "ref_table", "ref_column"],
                            "additionalProperties": False,
                        },
                    },
                    "remove_foreign_keys": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "ref_table": {"type": "string"},
                                "ref_column": {"type": "string"},
                            },
                            "required": ["column", "ref_table", "ref_column"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["table"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PRUNE_TABLE",
            "description": "Remove irrelevant columns or foreign keys from discovered_schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "remove_columns": {"type": "array", "items": {"type": "string"}},
                    "remove_foreign_keys": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "ref_table": {"type": "string"},
                                "ref_column": {"type": "string"},
                            },
                            "required": ["column", "ref_table", "ref_column"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["table"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "DROP_TABLE",
            "description": "Remove an irrelevant table from discovered_schema.",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string"}},
                "required": ["table"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "FINISH_SCHEMA_DISCOVERY",
            "description": "Finish when discovered_schema is sufficient for SQL generation.",
            "parameters": {
                "type": "object",
                "properties": {"report": {"type": "string"}},
                "required": ["report"],
                "additionalProperties": False,
            },
        },
    },
]


class SchemaDiscoveryAgent:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_turns: int = 6,
        max_tool_calls_per_turn: int = 4,
        read_table_summary_max_cols: int = 30,
        trace_column_preview_limit: int = 8,
        debug: bool = False,
        tracer: Optional[EventTracer] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.client = llm_client or create_llm_backend(provider="openai", api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.read_table_summary_max_cols = read_table_summary_max_cols
        self.trace_column_preview_limit = trace_column_preview_limit
        self.debug = debug
        self.tracer = tracer or NULL_TRACER

    def run(self, state: SharedState, guidance: str, metadata: SchemaMetadataStore) -> AgentReturn:
        global_step = state.step + 1
        self.tracer.emit(
            "worker_start",
            "SDA",
            "Schema discovery worker started.",
            global_step=global_step,
            payload={"guidance": guidance},
        )
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": build_schema_discovery_system_prompt(
                    metadata_display=state.metadata_display or metadata.display(),
                    db_name=state.db_id,
                ),
            },
            {
                "role": "user",
                "content": (
                    f"USER QUESTION:\n{state.question}\n\n"
                    f"EXTERNAL KNOWLEDGE:\n{state.external_knowledge}\n\n"
                    f"MANAGER GUIDANCE:\n{guidance}\n\n"
                    f"CURRENT discovered_schema:\n{format_discovered_schema_compact(state)}\n\n"
                    "Call a tool now."
                ),
            },
        ]

        last_error = ""
        for turn in range(1, self.max_turns + 1):
            self.tracer.emit(
                "worker_step_start",
                "SDA",
                "Requesting schema-discovery tool call from LLM.",
                global_step=global_step,
                worker_step=turn,
            )
            try:
                message = self._call_llm(messages)
            except Exception as exc:
                last_error = f"SDA LLM call failed: {safe_llm_error(exc)}"
                self.tracer.emit(
                    "worker_error",
                    "SDA",
                    last_error,
                    global_step=global_step,
                    worker_step=turn,
                    status="error",
                    payload={"error": last_error},
                )
                self.tracer.emit(
                    "worker_finish",
                    "SDA",
                    "Schema discovery stopped because the LLM call failed.",
                    global_step=global_step,
                    worker_step=turn,
                    status="error",
                    payload={"error": last_error},
                )
                return AgentReturn(
                    agent=AgentName.SCHEMA_DISCOVERY,
                    ok=False,
                    report=last_error,
                    payload={"reason": "llm_call_failed", "fatal": is_fatal_llm_error(exc)},
                )
            tool_calls = getattr(message, "tool_calls", None) if message is not None else None
            if not tool_calls:
                last_error = "SDA produced no tool call"
                logger.warning("[SDA] turn %s produced no tool call", turn)
                self.tracer.emit(
                    "worker_error",
                    "SDA",
                    last_error,
                    global_step=global_step,
                    worker_step=turn,
                    status="error",
                )
                break

            assistant_tool_calls = []
            unique_calls = []
            seen = set()
            for tc in tool_calls:
                if getattr(tc, "type", "") != "function":
                    continue
                key = (tc.function.name, tc.function.arguments)
                if key in seen:
                    continue
                seen.add(key)
                unique_calls.append(tc)

            self.tracer.emit(
                "worker_step_tools",
                "SDA",
                f"LLM emitted {len(unique_calls[: self.max_tool_calls_per_turn])} schema tool call(s).",
                global_step=global_step,
                worker_step=turn,
                payload={"tools": [tc.function.name for tc in unique_calls[: self.max_tool_calls_per_turn]]},
            )

            for tc in unique_calls[: self.max_tool_calls_per_turn]:
                assistant_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

            messages.append({"role": "assistant", "content": "", "tool_calls": assistant_tool_calls})

            schema_updates: List[str] = []
            for tc in unique_calls[: self.max_tool_calls_per_turn]:
                name = tc.function.name
                args: Dict[str, Any] = {}
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    output, update, finished = self._execute_tool(name, args, state, metadata)
                except Exception as exc:
                    output, update, finished = {"ok": False, "error": str(exc), "tool": name}, "", False

                self.tracer.emit(
                    "tool_result",
                    "SDA",
                    f"Executed {name}.",
                    global_step=global_step,
                    worker_step=turn,
                    tool=name,
                    status="ok" if output.get("ok") else "error",
                    payload=schema_tool_payload(
                        name,
                        args,
                        output,
                        update,
                        column_preview_limit=self.trace_column_preview_limit,
                    ),
                )

                if update:
                    schema_updates.append(update)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                )

                if finished:
                    report = str(output.get("report") or "schema discovery finished")
                    self.tracer.emit(
                        "worker_finish",
                        "SDA",
                        "Schema discovery worker finished.",
                        global_step=global_step,
                        worker_step=turn,
                        status="ok",
                        payload={"report": report},
                    )
                    return AgentReturn(
                        agent=AgentName.SCHEMA_DISCOVERY,
                        ok=True,
                        report=report,
                        payload={"discovered_schema": state.discovered.tables},
                    )

            follow_up = [
                "Tool results received.",
                f"Current discovered_schema:\n{format_discovered_schema_compact(state)}",
            ]
            if schema_updates:
                follow_up.append("Schema updates:\n" + "\n".join(f"- {x}" for x in schema_updates))
            follow_up.append("Continue with tools, or call FINISH_SCHEMA_DISCOVERY.")
            messages.append({"role": "user", "content": "\n\n".join(follow_up)})

        if state.discovered.tables:
            self.tracer.emit(
                "worker_finish",
                "SDA",
                "Schema discovery reached max_turns with partial schema.",
                global_step=global_step,
                status="ok",
                payload={"report": "partial discovered_schema", "last_error": last_error},
            )
            return AgentReturn(
                agent=AgentName.SCHEMA_DISCOVERY,
                ok=True,
                report="Schema discovery reached the turn limit with a partial discovered_schema.",
                payload={"last_error": last_error, "discovered_schema": state.discovered.tables},
            )
        self.tracer.emit(
            "worker_finish",
            "SDA",
            "Schema discovery failed.",
            global_step=global_step,
            status="error",
            payload={"error": last_error or "max_turns"},
        )
        return AgentReturn(
            agent=AgentName.SCHEMA_DISCOVERY,
            ok=False,
            report=last_error or "Schema discovery did not finish within max_turns.",
            payload={},
        )

    def _call_llm(self, messages: List[Dict[str, Any]]) -> Any:
        response = create_chat_completion(
            self.client,
            role="schema_discovery",
            model=self.model,
            messages=messages,
            tools=SCHEMA_TOOLS,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message

    def _execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        state: SharedState,
        metadata: SchemaMetadataStore,
    ) -> Tuple[Dict[str, Any], str, bool]:
        if name == "SEARCH_METADATA":
            keywords = args.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [keywords]
            output = metadata.search([str(k) for k in keywords], mode=str(args.get("mode") or "OR"))
            return output, "", False

        if name == "READ_TABLE_JSON":
            table = _required_str(args.get("table"), "table")
            obj = metadata.get_table(table)
            output = {
                "ok": True,
                "table": table,
                "summary": summarize_table(obj, state.question, state.external_knowledge, self.read_table_summary_max_cols),
            }
            return output, "", False

        if name == "INTRODUCE_TABLE":
            table = _required_str(args.get("table"), "table")
            columns = _list_arg(args.get("columns"))
            primary_key = [str(x).strip() for x in _list_arg(args.get("primary_key")) if str(x).strip()]
            foreign_keys = _list_arg(args.get("foreign_keys"))
            if not columns:
                raise ValueError("INTRODUCE_TABLE requires at least one column")
            metadata.verify_columns(table, columns, "columns")
            metadata.verify_primary_keys(table, primary_key)
            if foreign_keys:
                metadata.verify_foreign_keys(table, foreign_keys, "foreign_keys")
            state.discovered.tables[table] = TableEvidence(
                table=table,
                columns=hydrate_columns_from_metadata(table, columns, metadata),
                primary_keys=primary_key,
                foreign_keys=clean_foreign_keys(foreign_keys),
            )
            return {"ok": True, "action": name, "table": table}, f"introduced {table}", False

        if name == "ENRICH_TABLE":
            table = _required_str(args.get("table"), "table")
            if not metadata.table_exists(table):
                raise ValueError(f"metadata missing for table '{table}'")
            add_columns = _list_arg(args.get("add_columns"))
            foreign_keys = _list_arg(args.get("foreign_keys"))
            remove_foreign_keys = _list_arg(args.get("remove_foreign_keys"))
            if add_columns:
                metadata.verify_columns(table, add_columns, "add_columns")
            if foreign_keys:
                metadata.verify_foreign_keys(table, foreign_keys, "foreign_keys")
            ev = state.discovered.tables.setdefault(table, TableEvidence(table=table))
            add_or_replace_columns(ev, hydrate_columns_from_metadata(table, add_columns, metadata))
            add_or_replace_foreign_keys(ev, clean_foreign_keys(foreign_keys))
            remove_foreign_keys_from_table(ev, clean_foreign_keys(remove_foreign_keys))
            return {"ok": True, "action": name, "table": table}, f"enriched {table}", False

        if name == "PRUNE_TABLE":
            table = _required_str(args.get("table"), "table")
            ev = state.discovered.tables.get(table)
            if ev is None:
                return {"ok": True, "action": name, "table": table, "note": "table was not discovered"}, "", False
            remove_columns = {str(x).strip().lower() for x in _list_arg(args.get("remove_columns")) if str(x).strip()}
            remove_fks = clean_foreign_keys(_list_arg(args.get("remove_foreign_keys")))
            if remove_columns:
                ev.columns = [c for c in ev.columns if str(c.get("name", "")).strip().lower() not in remove_columns]
                ev.primary_keys = [pk for pk in ev.primary_keys if pk.strip().lower() not in remove_columns]
                ev.foreign_keys = [fk for fk in ev.foreign_keys if fk.get("col", "").strip().lower() not in remove_columns]
            remove_foreign_keys_from_table(ev, remove_fks)
            return {"ok": True, "action": name, "table": table}, f"pruned {table}", False

        if name == "DROP_TABLE":
            table = _required_str(args.get("table"), "table")
            state.discovered.tables.pop(table, None)
            return {"ok": True, "action": name, "table": table}, f"dropped {table}", False

        if name == "FINISH_SCHEMA_DISCOVERY":
            report = _required_str(args.get("report"), "report")
            return {"ok": True, "finished": True, "report": report}, "", True

        return {"ok": False, "error": f"unknown tool: {name}"}, "", False


def summarize_table(obj: Dict[str, Any], question: str, external_knowledge: str, max_cols: int) -> Dict[str, Any]:
    q = f"{question} {external_knowledge}".lower()
    pk_set = set(obj.get("primary_keys", []) or [])
    fk_cols = {parse_foreign_key(fk, strict=False)[0] for fk in obj.get("foreign_keys", []) or []}

    ranked = []
    for col in obj.get("columns", []) or []:
        if not isinstance(col, dict):
            continue
        name = str(col.get("name", "")).strip()
        desc = str(col.get("desc", "")).strip()
        score = 0
        low = name.lower()
        if low and low in q:
            score += 10
        for token in low.replace("_", " ").split():
            if token and token in q:
                score += 3
        if desc and desc.lower() in q:
            score += 2
        if name in pk_set:
            score += 4
        if name in fk_cols:
            score += 4
        if any(word in low for word in ("id", "name", "date", "time", "type", "status", "amount", "count", "score")):
            score += 1
        ranked.append((score, col))
    ranked.sort(key=lambda item: item[0], reverse=True)

    columns = []
    selected = ranked if max_cols <= 0 else ranked[:max_cols]
    for _, col in selected:
        columns.append(
            {
                "name": col.get("name", ""),
                "type": col.get("type", "UNKNOWN"),
                **({"desc": col.get("desc")} if col.get("desc") else {}),
            }
        )

    return {
        "table": obj.get("table", ""),
        "columns": columns,
        "primary_keys": obj.get("primary_keys", []) or [],
        "foreign_keys": obj.get("foreign_keys", []) or [],
        "total_columns": len(obj.get("columns", []) or []),
    }


def schema_tool_payload(
    name: str,
    args: Dict[str, Any],
    output: Dict[str, Any],
    update: str,
    column_preview_limit: int = 8,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if "keywords" in args:
        payload["keywords"] = args.get("keywords")
    if "table" in args:
        payload["table"] = args.get("table")
    if output.get("error"):
        payload["error"] = output.get("error")
    if output.get("num_tables") is not None:
        payload["tables_matched"] = output.get("num_tables")
    if output.get("summary"):
        summary = output.get("summary") or {}
        columns = [c.get("name") for c in summary.get("columns", []) if isinstance(c, dict)]
        payload["columns"] = columns if column_preview_limit <= 0 else columns[:column_preview_limit]
        payload["columns_shown"] = len(payload["columns"])
        payload["columns_available"] = len(columns)
    if update:
        payload["update"] = update
    if name == "FINISH_SCHEMA_DISCOVERY":
        payload["report"] = output.get("report")
    return payload


def format_discovered_schema_compact(state: SharedState) -> str:
    if not state.discovered.tables:
        return "[]"
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


def clean_columns(columns: List[Any]) -> List[Dict[str, str]]:
    out = []
    for col in columns:
        if not isinstance(col, dict):
            continue
        name = str(col.get("name", "")).strip()
        if not name:
            continue
        item = {"name": name, "type": str(col.get("type") or "UNKNOWN")}
        if col.get("desc"):
            item["desc"] = str(col.get("desc"))
        out.append(item)
    return out


def hydrate_columns_from_metadata(
    table: str,
    columns: List[Any],
    metadata: SchemaMetadataStore,
) -> List[Dict[str, str]]:
    """Return canonical column entries enriched with metadata type/description.

    The LLM only needs to identify the column name. Once verified, QueryOS uses
    the table metadata as source of truth so shared state consistently contains
    column descriptions for downstream workers.
    """
    table_obj = metadata.get_table(table)
    metadata_columns = {
        normalize_name(str(col.get("name", ""))): col
        for col in table_obj.get("columns", []) or []
        if isinstance(col, dict) and col.get("name")
    }
    hydrated: List[Dict[str, str]] = []
    for col in clean_columns(columns):
        key = normalize_name(str(col.get("name", "")))
        meta_col = metadata_columns.get(key) or {}
        name = str(meta_col.get("name") or col.get("name") or "").strip()
        if not name:
            continue
        item = {
            "name": name,
            "type": str(meta_col.get("type") or col.get("type") or "UNKNOWN"),
        }
        desc = str(meta_col.get("desc") or meta_col.get("description") or col.get("desc") or "").strip()
        if desc:
            item["desc"] = desc
        hydrated.append(item)
    return hydrated


def clean_foreign_keys(foreign_keys: List[Any]) -> List[Dict[str, str]]:
    out = []
    for fk in foreign_keys:
        if not isinstance(fk, dict):
            continue
        col, ref_table, ref_column = parse_foreign_key(fk, strict=False)
        if col and ref_table and ref_column:
            out.append({"col": col, "ref": f"{ref_table}.{ref_column}"})
    return out


def add_or_replace_columns(ev: TableEvidence, columns: List[Dict[str, str]]) -> None:
    merged = {c.get("name", "").strip().lower(): c for c in ev.columns if c.get("name")}
    for col in columns:
        merged[col["name"].strip().lower()] = col
    ev.columns = list(merged.values())


def add_or_replace_foreign_keys(ev: TableEvidence, foreign_keys: List[Dict[str, str]]) -> None:
    merged = {f'{fk.get("col", "").lower()}->{fk.get("ref", "").lower()}': fk for fk in ev.foreign_keys}
    for fk in foreign_keys:
        merged[f'{fk.get("col", "").lower()}->{fk.get("ref", "").lower()}'] = fk
    ev.foreign_keys = list(merged.values())


def remove_foreign_keys_from_table(ev: TableEvidence, foreign_keys: List[Dict[str, str]]) -> None:
    if not foreign_keys:
        return
    remove = {f'{fk.get("col", "").lower()}->{fk.get("ref", "").lower()}' for fk in foreign_keys}
    ev.foreign_keys = [
        fk for fk in ev.foreign_keys
        if f'{fk.get("col", "").lower()}->{fk.get("ref", "").lower()}' not in remove
    ]


def _required_str(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _list_arg(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
