from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import agentlightning as agl
import openai
from termcolor import cprint

from agentlightning.litagent import DiscardTrajectory
from shared_states import AgentName, AgentReturn, SharedGlobalState, TableEvidence
from tools import Spider2Tools

agl.setup_logging(apply_to=[__name__])
logger = logging.getLogger(__name__)


# =============================================================================
# Helpers / markers
# =============================================================================
SCHEMA_UPDATE_MARKER = "[SCHEMA_UPDATE_MESSAGE]"
TOOL_PREFIX_RE = re.compile(r"^\[TOOL:([A-Z_]+)\]\n", flags=re.MULTILINE)

METADATA_TOOL_NAMES = {
    "READ_TABLE_JSON",
    "SEARCH_METADATA",
}

SCHEMA_MUTATION_TOOL_NAMES = {
    "INTRODUCE_TABLE",
    "ENRICH_TABLE",
    "PRUNE_TABLE",
    "DROP_TABLE",
}


def _norm_col(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip().lower())


def _must_nonempty_str(x: Any, what: str) -> str:
    if x is None:
        raise ValueError(f"{what} is null")
    s = str(x).strip()
    if not s:
        raise ValueError(f"{what} is empty")
    return s


def _ensure_list(x: Any, what: str) -> List[Any]:
    if x is None:
        return []
    if not isinstance(x, list):
        raise ValueError(f"{what} must be a list")
    return x


def _strip_thinking_block(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _truncate_text(text: str, max_chars: int = 1500) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def _wrap_tool_content(tool_name: str, content: str, max_chars: int = 1500) -> str:
    body = _truncate_text(content, max_chars=max_chars)
    return f"[TOOL:{tool_name}]\n{body}"


def _extract_tool_name_from_content(content: str) -> Optional[str]:
    if not content:
        return None
    m = TOOL_PREFIX_RE.match(content)
    if not m:
        return None
    return m.group(1)


def _schema_to_json(state: SharedGlobalState, max_cols_per_table: int = 9999) -> str:
    out = []
    for tname, ev in state.discovered.tables.items():
        cols = []
        for c in (ev.columns or [])[:max_cols_per_table]:
            if isinstance(c, dict):
                item = {
                    "name": c.get("name", ""),
                    "type": c.get("type", "UNKNOWN") or "UNKNOWN",
                }
                desc = c.get("desc", "")
                if desc:
                    item["desc"] = desc
                cols.append(item)
            else:
                cols.append({"name": str(c), "type": "UNKNOWN"})
        out.append(
            {
                "table": tname,
                "columns": cols,
                "primary_keys": ev.primary_keys or [],
                "foreign_keys": ev.foreign_keys or [],
            }
        )
    return json.dumps(out, ensure_ascii=False, indent=2)


def _compact_schema_text_for_sda(
    state: SharedGlobalState,
    max_tables: int = 20,
    max_cols_per_table: int = 10,
    max_fks_per_table: int = 4,
) -> str:
    lines: List[str] = []
    items = list(state.discovered.tables.items())[:max_tables]
    for tname, ev in items:
        cols = []
        for c in (ev.columns or [])[:max_cols_per_table]:
            if isinstance(c, dict):
                cols.append(str(c.get("name", "")))
            else:
                cols.append(str(c))
        cols = [c for c in cols if c]

        pks = (ev.primary_keys or [])[:4]
        fks = []
        for fk in (ev.foreign_keys or [])[:max_fks_per_table]:
            if isinstance(fk, dict):
                col = fk.get("col", "")
                ref = fk.get("ref", "")
                if col and ref:
                    fks.append(f"{col}->{ref}")

        s_cols = ",".join(cols) if cols else "-"
        s_pks = ",".join(pks) if pks else "-"
        s_fks = ";".join(fks) if fks else "-"
        lines.append(f"- {tname} | cols: {s_cols} | pk: {s_pks} | fk: {s_fks}")
    return "\n".join(lines) if lines else "(empty discovered_schema)"


def _table_summary_for_update(ev: TableEvidence, max_cols: int = 8, max_fks: int = 3) -> str:
    cols = []
    for c in (ev.columns or [])[:max_cols]:
        if isinstance(c, dict):
            cols.append(str(c.get("name", "")))
        else:
            cols.append(str(c))
    cols = [x for x in cols if x]

    pks = (ev.primary_keys or [])[:3]
    fks = []
    for fk in (ev.foreign_keys or [])[:max_fks]:
        if isinstance(fk, dict):
            col = fk.get("col", "")
            ref = fk.get("ref", "")
            if col and ref:
                fks.append(f"{col}->{ref}")

    parts = [f"cols={','.join(cols) if cols else '-'}"]
    if pks:
        parts.append(f"pk={','.join(pks)}")
    if fks:
        parts.append(f"fk={';'.join(fks)}")
    return " | ".join(parts)


# =============================================================================
# Prompt
# =============================================================================
def build_schema_discovery_system_prompt(
    schema_metadata_path: str,
    metadata_display: str,
    db_name: str,
) -> str:
    return f"""You are the Schema Discovery Agent (SDA) for SQLite DB: {db_name}.

You are a worker. You ONLY follow the planner's instruction and update `discovered_schema`.
You do NOT write SQL.
You do NOT decide the next step globally (planner does that).

========================
TOOL-ONLY RULE (STRICT)
========================
- You MUST call a tool in EVERY turn.
- You MUST NOT output plain text as your final action.
- You MAY call multiple tools in one turn if needed.
- If unsure, call SEARCH_METADATA first.
- Keep thinking short and act quickly.

========================
METADATA
========================
Metadata directory:
{schema_metadata_path}

Metadata listing (reference only):
{metadata_display}

========================
TOOLS
========================
Metadata tools:
- SEARCH_METADATA(keywords)
  Use this FIRST when table relevance is unclear.
  Returns structured matches for relevant tables, columns, primary keys, and foreign keys.

- READ_TABLE_JSON(table)
  Use this to inspect ONE table's schema summary:
  - important columns
  - primary keys
  - foreign keys
  Do not use this to dump many irrelevant full schemas.

Schema state tools:
- INTRODUCE_TABLE(table, columns, primary_key?, foreign_keys?)
- ENRICH_TABLE(table, add_columns?, foreign_keys?, remove_foreign_keys?)
- PRUNE_TABLE(table, remove_columns?, remove_foreign_keys?)
- DROP_TABLE(table)

Finish tool:
- FINISH_SCHEMA_DISCOVERY(report)

========================
IMPORTANT: OFFLINE VERIFIED
========================
All schema update tools are offline-verified against metadata.
If a table, column, or foreign key fact does not exist in metadata, the update is rejected.

========================
MINIMALITY IS CRITICAL
========================
Your job is NOT to recover full database schema.
Your job IS to build the SMALLEST discovered_schema sufficient for SQL Writer.

Only include what is necessary for:
- output fields
- filters
- join keys
- group/order keys

Adding unnecessary tables, columns, or foreign keys is BAD behavior.

Never:
- dump full table schema
- add all columns "just in case"
- invent or guess foreign keys
- keep irrelevant tables in discovered_schema

Prefer:
- fewer tables
- fewer columns
- only necessary keys
- incremental updates

========================
DISCOVERY STRATEGY
========================
1) Find relevant tables with SEARCH_METADATA
2) Inspect only the most relevant table(s) with READ_TABLE_JSON
3) Add the minimum required columns / keys
4) Enrich only if a specific missing field is needed
5) Stop as soon as SQL Writer has enough schema

========================
STOP CONDITION
========================
When discovered_schema is sufficient for SQL generation, call FINISH_SCHEMA_DISCOVERY.
"""


# =============================================================================
# Offline verifier
# =============================================================================
class OfflineSchemaVerifier:
    """
    Strict factual verification against database_description/*.json files.

    Verified facts:
    - table exists
    - column exists
    - primary key column exists
    - foreign key fact itself exists in metadata
    """

    def __init__(self, schema_metadata_path: str, max_read_bytes: int = 250_000):
        self.md_path = Path(schema_metadata_path)
        self.max_read_bytes = max_read_bytes

        if not self.md_path.exists() or not self.md_path.is_dir():
            raise RuntimeError(f"schema_metadata_path not found: {schema_metadata_path}")

        self._available_stems: Set[str] = set(fp.stem for fp in self.md_path.glob("*.json") if fp.is_file())
        self._cols_cache: Dict[str, Set[str]] = {}
        self._fk_cache: Dict[str, Set[Tuple[str, str, str]]] = {}
        self._loaded: Set[str] = set()

    def table_exists(self, table: str) -> bool:
        return table in self._available_stems and (self.md_path / f"{table}.json").exists()

    def _read_json(self, table: str) -> Dict[str, Any]:
        fp = self.md_path / f"{table}.json"
        if not fp.exists():
            raise ValueError(f"missing <table>.json: {fp}")

        raw = fp.read_bytes()
        if len(raw) > self.max_read_bytes:
            raw = raw[: self.max_read_bytes]
        obj = json.loads(raw.decode("utf-8", errors="replace"))

        jt = str(obj.get("table", "")).strip()
        if jt and jt != table:
            raise ValueError(f"metadata mismatch: file stem='{table}' but json.table='{jt}'")
        return obj

    def _ensure_table_loaded(self, table: str) -> None:
        if table in self._loaded:
            return
        if not self.table_exists(table):
            raise ValueError(f"table metadata missing: {table}.json")

        obj = self._read_json(table)

        cols = set()
        for c in (obj.get("columns") or []):
            if isinstance(c, dict) and c.get("name"):
                cols.add(_norm_col(str(c["name"])))
        if not cols:
            raise ValueError(f"metadata has no columns for table='{table}'")

        factual_fks: Set[Tuple[str, str, str]] = set()
        for fk in (obj.get("foreign_keys") or []):
            if not isinstance(fk, dict):
                continue

            col = fk.get("column") or fk.get("col")
            ref_table = fk.get("ref_table")
            ref_column = fk.get("ref_column")

            ref = fk.get("ref")
            if (not ref_table or not ref_column) and isinstance(ref, str) and "." in ref:
                ref_table, ref_column = ref.split(".", 1)

            if col and ref_table and ref_column:
                factual_fks.add(
                    (_norm_col(str(col)), str(ref_table).strip(), _norm_col(str(ref_column)))
                )

        self._cols_cache[table] = cols
        self._fk_cache[table] = factual_fks
        self._loaded.add(table)

    def col_exists(self, table: str, col: str) -> bool:
        self._ensure_table_loaded(table)
        return _norm_col(col) in self._cols_cache[table]

    def fk_exists(self, table: str, col: str, ref_table: str, ref_column: str) -> bool:
        self._ensure_table_loaded(table)
        return (
            _norm_col(col),
            str(ref_table).strip(),
            _norm_col(ref_column),
        ) in self._fk_cache[table]

    def verify_columns(self, table: str, cols: List[Dict[str, Any]], field_name: str) -> None:
        self._ensure_table_loaded(table)
        for c in cols:
            if not isinstance(c, dict):
                raise ValueError(f"{field_name} entry must be object, got: {c}")
            name = _must_nonempty_str(c.get("name"), f"{field_name}.name")
            if not self.col_exists(table, name):
                raise ValueError(f"column '{name}' not in metadata for table '{table}'")

    def verify_pk(self, table: str, pks: List[str]) -> None:
        self._ensure_table_loaded(table)
        for pk in pks:
            pk = _must_nonempty_str(pk, "primary_key item")
            if not self.col_exists(table, pk):
                raise ValueError(f"primary_key '{pk}' not in metadata for table '{table}'")

    def verify_fks(self, table: str, fks: List[Dict[str, Any]], field_name: str) -> None:
        self._ensure_table_loaded(table)

        for fk in fks:
            if not isinstance(fk, dict):
                raise ValueError(f"{field_name} entry must be object, got: {fk}")

            col = _must_nonempty_str(fk.get("column"), f"{field_name}.column")
            rt = _must_nonempty_str(fk.get("ref_table"), f"{field_name}.ref_table")
            rc = _must_nonempty_str(fk.get("ref_column"), f"{field_name}.ref_column")

            if not self.col_exists(table, col):
                raise ValueError(f"fk column '{col}' not in table '{table}' metadata")
            if not self.table_exists(rt):
                raise ValueError(f"ref_table metadata missing: {rt}.json")

            self._ensure_table_loaded(rt)
            if not self.col_exists(rt, rc):
                raise ValueError(f"ref_column '{rc}' not in ref_table '{rt}' metadata")

            if not self.fk_exists(table, col, rt, rc):
                raise ValueError(
                    f"foreign key fact not found in metadata: {table}.{col} -> {rt}.{rc}"
                )


# =============================================================================
# Schema mutation
# =============================================================================
def _apply_introduce_table(
    state: SharedGlobalState,
    table: str,
    columns: List[Dict[str, Any]],
    primary_key: List[str],
    foreign_keys: List[Dict[str, Any]],
) -> None:
    fks_internal: List[Dict[str, str]] = []
    for fk in foreign_keys:
        col = str(fk["column"]).strip()
        rt = str(fk["ref_table"]).strip()
        rc = str(fk["ref_column"]).strip()
        fks_internal.append({"col": col, "ref": f"{rt}.{rc}"})

    cols_internal: List[Dict[str, str]] = []
    for c in columns:
        name = str(c["name"]).strip()
        item: Dict[str, str] = {"name": name}
        item["type"] = str(c.get("type", "UNKNOWN")).strip() or "UNKNOWN"
        if "desc" in c and str(c["desc"]).strip():
            item["desc"] = str(c["desc"]).strip()
        cols_internal.append(item)

    state.discovered.tables[table] = TableEvidence(
        table=table,
        schema=None,
        columns=cols_internal,
        primary_keys=[str(x).strip() for x in primary_key if str(x).strip()],
        foreign_keys=fks_internal,
    )


def _apply_enrich_table(
    state: SharedGlobalState,
    table: str,
    add_columns: List[Dict[str, Any]],
    foreign_keys: List[Dict[str, Any]],
    remove_foreign_keys: List[Dict[str, Any]],
) -> None:
    if table not in state.discovered.tables:
        state.discovered.tables[table] = TableEvidence(table=table, schema=None)
    ev = state.discovered.tables[table]

    existing = {str(c.get("name", "")).strip().lower(): c for c in (ev.columns or []) if isinstance(c, dict)}
    for c in add_columns:
        name = str(c["name"]).strip()
        key = name.lower()
        item: Dict[str, str] = {"name": name}
        item["type"] = str(c.get("type", "UNKNOWN")).strip() or "UNKNOWN"
        if "desc" in c and str(c["desc"]).strip():
            item["desc"] = str(c["desc"]).strip()
        existing[key] = item
    ev.columns = list(existing.values())

    fk_existing = {
        f"{fk.get('col','').strip().lower()}->{fk.get('ref','').strip().lower()}": fk
        for fk in (ev.foreign_keys or [])
    }

    for fk in foreign_keys:
        col = str(fk["column"]).strip()
        rt = str(fk["ref_table"]).strip()
        rc = str(fk["ref_column"]).strip()
        internal = {"col": col, "ref": f"{rt}.{rc}"}
        k = f"{col.lower()}->{internal['ref'].lower()}"
        fk_existing[k] = internal

    rm_keys = set()
    for fk in remove_foreign_keys:
        col = str(fk["column"]).strip()
        rt = str(fk["ref_table"]).strip()
        rc = str(fk["ref_column"]).strip()
        rm_keys.add(f"{col.lower()}->{rt.lower()}.{rc.lower()}")

    if rm_keys:
        fk_existing = {k: v for k, v in fk_existing.items() if k not in rm_keys}

    ev.foreign_keys = list(fk_existing.values())


def _apply_prune_table(
    state: SharedGlobalState,
    table: str,
    remove_columns: List[str],
    remove_foreign_keys: List[Dict[str, Any]],
) -> None:
    if table not in state.discovered.tables:
        return
    ev = state.discovered.tables[table]

    rm_cols = {str(c).strip().lower() for c in remove_columns if str(c).strip()}
    if rm_cols:
        ev.columns = [
            c for c in (ev.columns or [])
            if str(c.get("name", "")).strip().lower() not in rm_cols
        ]
        ev.primary_keys = [pk for pk in (ev.primary_keys or []) if str(pk).strip().lower() not in rm_cols]
        ev.foreign_keys = [fk for fk in (ev.foreign_keys or []) if str(fk.get("col", "")).strip().lower() not in rm_cols]

    rm_fk_keys = set()
    for fk in remove_foreign_keys:
        col = str(fk["column"]).strip()
        rt = str(fk["ref_table"]).strip()
        rc = str(fk["ref_column"]).strip()
        rm_fk_keys.add(f"{col.lower()}->{rt.lower()}.{rc.lower()}")

    if rm_fk_keys:
        ev.foreign_keys = [
            fk for fk in (ev.foreign_keys or [])
            if f"{str(fk.get('col','')).strip().lower()}->{str(fk.get('ref','')).strip().lower()}" not in rm_fk_keys
        ]


def _apply_drop_table(state: SharedGlobalState, table: str) -> None:
    state.discovered.tables.pop(table, None)


# =============================================================================
# SDA Agent
# =============================================================================
class SchemaDiscoveryAgent:
    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        max_turns: int = 6,
        debug: bool = False,
        api_key_env: str = "OPENAI_API_KEY",
        max_tool_calls_per_turn: int = 3,
        history_keep_last_messages: int = 8,
        history_max_chars: int = 34000,
        max_read_bytes: int = 350_000,
        tool_output_max_chars: int = 10000,
        read_table_summary_max_cols: int = 15,
        keep_recent_metadata_tool_results: int = 1,
        **_: Any,
    ):
        self.tools = Spider2Tools()
        self.client = openai.OpenAI(
            base_url=endpoint,
            api_key=os.getenv(api_key_env, "dummy"),
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.debug = debug

        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.history_keep_last_messages = history_keep_last_messages
        self.history_max_chars = history_max_chars
        self.max_read_bytes = max_read_bytes
        self.tool_output_max_chars = tool_output_max_chars
        self.read_table_summary_max_cols = read_table_summary_max_cols
        self.keep_recent_metadata_tool_results = keep_recent_metadata_tool_results

        self.table_json_cache: Dict[str, Dict[str, Any]] = {}
        self.search_cache: Dict[str, Dict[str, Any]] = {}
        self.last_schema_update_msg: str = ""

        self.openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "SEARCH_METADATA",
                    "description": (
                        "Search metadata structurally to find relevant tables, columns, primary keys, and foreign keys. "
                        "Matching is based on schema structure only (table names, column names, primary keys, and foreign key endpoints), "
                        "not free-text descriptions. "
                        "Returns matched objects with details, not raw line snippets. "
                        "Use this first when table relevance is unclear."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["keywords"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "READ_TABLE_JSON",
                    "description": (
                        "Read one table metadata file and return a compact schema summary. "
                        "Use this to inspect important columns, primary keys, and foreign keys for one relevant table."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"}
                        },
                        "required": ["table"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "INTRODUCE_TABLE",
                    "description": (
                        "Introduce a table with minimal required columns and keys only. "
                        "Do not introduce full table schema."
                    ),
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
                                        "desc": {"type": "string"}
                                    },
                                    "required": ["name"]
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
                                        "ref_column": {"type": "string"}
                                    },
                                    "required": ["column", "ref_table", "ref_column"]
                                },
                            },
                        },
                        "required": ["table", "columns"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ENRICH_TABLE",
                    "description": (
                        "Add only missing columns or foreign keys to an already discovered table. "
                        "Use incrementally and minimally."
                    ),
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
                                        "desc": {"type": "string"}
                                    },
                                    "required": ["name"]
                                },
                            },
                            "foreign_keys": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "column": {"type": "string"},
                                        "ref_table": {"type": "string"},
                                        "ref_column": {"type": "string"}
                                    },
                                    "required": ["column", "ref_table", "ref_column"]
                                },
                            },
                            "remove_foreign_keys": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "column": {"type": "string"},
                                        "ref_table": {"type": "string"},
                                        "ref_column": {"type": "string"}
                                    },
                                    "required": ["column", "ref_table", "ref_column"]
                                },
                            },
                        },
                        "required": ["table"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "PRUNE_TABLE",
                    "description": (
                        "Remove unnecessary columns or foreign keys from a discovered table "
                        "to keep discovered_schema minimal."
                    ),
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
                                        "ref_column": {"type": "string"}
                                    },
                                    "required": ["column", "ref_table", "ref_column"]
                                },
                            },
                        },
                        "required": ["table"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "DROP_TABLE",
                    "description": "Drop a table from discovered_schema if it is not needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {"table": {"type": "string"}},
                        "required": ["table"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "FINISH_SCHEMA_DISCOVERY",
                    "description": "Finish schema discovery when discovered_schema is sufficient for SQL generation.",
                    "parameters": {
                        "type": "object",
                        "properties": {"report": {"type": "string"}},
                        "required": ["report"]
                    },
                },
            },
        ]

    # -------------------------------------------------------------------------
    # LLM / history helpers
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
            logger.error("[SDA] LLM call failed: %s", e)
            return None

    def _compact_after_schema_update(self, messages: List[Dict[str, Any]]) -> None:
        if len(messages) <= 1:
            return

        keep_recent_metadata = self.keep_recent_metadata_tool_results
        metadata_tool_positions = []

        for idx, m in enumerate(messages):
            if m.get("role") != "tool":
                continue
            tool_name = _extract_tool_name_from_content(m.get("content", "") or "")
            if tool_name in METADATA_TOOL_NAMES:
                metadata_tool_positions.append(idx)

        metadata_keep_set = set(metadata_tool_positions[-keep_recent_metadata:]) if keep_recent_metadata > 0 else set()

        last_schema_update_idx = None
        for idx, m in enumerate(messages):
            if m.get("role") == "user" and SCHEMA_UPDATE_MARKER in (m.get("content", "") or ""):
                last_schema_update_idx = idx

        new_messages = []
        for idx, m in enumerate(messages):
            role = m.get("role")

            if role == "user" and SCHEMA_UPDATE_MARKER in (m.get("content", "") or ""):
                if idx != last_schema_update_idx:
                    continue

            if role == "tool":
                tool_name = _extract_tool_name_from_content(m.get("content", "") or "")
                if tool_name in METADATA_TOOL_NAMES and idx not in metadata_keep_set:
                    continue

            new_messages.append(m)

        messages[:] = new_messages

    def _trim_history(self, messages: List[Dict[str, Any]]) -> None:
        if len(messages) <= 1:
            return

        system_msg = messages[0]
        rest = messages[1:]

        schema_msg_indices = [
            i for i, m in enumerate(rest)
            if m.get("role") == "user" and SCHEMA_UPDATE_MARKER in (m.get("content", "") or "")
        ]
        if len(schema_msg_indices) > 1:
            keep_idx = schema_msg_indices[-1]
            rest = [m for i, m in enumerate(rest) if i == keep_idx or SCHEMA_UPDATE_MARKER not in (m.get("content", "") or "")]

        rest = rest[-self.history_keep_last_messages:]

        kept = []
        total = 0
        for m in reversed(rest):
            content = m.get("content", "") or ""
            msg_len = len(content)
            if total + msg_len > self.history_max_chars:
                break
            kept.append(m)
            total += msg_len

        kept.reverse()
        messages[:] = [system_msg] + kept

    def _schema_update_message(self, updates: List[str]) -> str:
        if not updates:
            return ""
        body = "\n".join(f"- {u}" for u in updates)
        return (
            f"{SCHEMA_UPDATE_MARKER}\n"
            f"Schema updates since last turn:\n"
            f"{body}\n"
            "Continue with tools, or call FINISH_SCHEMA_DISCOVERY."
        )

    # -------------------------------------------------------------------------
    # metadata summarization
    # -------------------------------------------------------------------------
    def _rank_columns_for_question(
        self,
        table_json: Dict[str, Any],
        question: str,
        evidence: str,
    ) -> List[Dict[str, Any]]:
        q = (question or "").lower()
        e = (evidence or "").lower()

        pk_set = set(table_json.get("primary_keys", []) or [])
        fk_cols = set()
        for fk in (table_json.get("foreign_keys") or []):
            if isinstance(fk, dict):
                col = fk.get("column") or fk.get("col")
                if col:
                    fk_cols.add(str(col).strip())

        ranked: List[Tuple[int, Dict[str, Any]]] = []
        for c in (table_json.get("columns") or []):
            if not isinstance(c, dict):
                continue
            name = str(c.get("name", "")).strip()
            desc = str(c.get("desc", "")).strip()
            low = name.lower()
            score = 0

            if low and low in q:
                score += 5
            if low and low in e:
                score += 4
            if desc and desc.lower() in q:
                score += 2

            for kw in [
                "id", "name", "date", "time", "city", "state", "country",
                "user", "tweet", "like", "business", "star", "rating", "location"
            ]:
                if kw in low:
                    score += 2

            if name in pk_set:
                score += 3
            if name in fk_cols:
                score += 3

            ranked.append((score, c))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked]

    def _summarize_table_json_for_llm(
        self,
        table_json: Dict[str, Any],
        question: str,
        evidence: str,
        max_cols: int,
    ) -> Dict[str, Any]:
        ranked_cols = self._rank_columns_for_question(table_json, question, evidence)
        top_cols = ranked_cols[:max_cols]

        fks = table_json.get("foreign_keys", []) or []
        compact_fks = []
        for fk in fks[:8]:
            if isinstance(fk, dict):
                compact_fks.append(
                    {
                        "column": fk.get("column", fk.get("col", "")),
                        "ref_table": fk.get("ref_table", ""),
                        "ref_column": fk.get("ref_column", ""),
                        **({"ref": fk.get("ref")} if fk.get("ref") else {}),
                    }
                )

        return {
            "table": table_json.get("table", ""),
            "columns": [
                {
                    "name": c.get("name", ""),
                    "type": c.get("type", "UNKNOWN"),
                    **({"desc": c.get("desc")} if c.get("desc") else {}),
                }
                for c in top_cols
            ],
            "primary_keys": table_json.get("primary_keys", []),
            "foreign_keys": compact_fks,
            "total_columns": len(table_json.get("columns", []) or []),
            "note": "Full metadata cached externally. Add only the minimal needed columns and factual foreign keys."
        }

    # -------------------------------------------------------------------------
    # metadata tools
    # -------------------------------------------------------------------------
    def _read_table_json(self, verifier: OfflineSchemaVerifier, table: str, state: SharedGlobalState) -> Dict[str, Any]:
        table = _must_nonempty_str(table, "table")
        fp = Path(verifier.md_path) / f"{table}.json"
        if not fp.exists():
            available = sorted(list(verifier._available_stems))[:50]
            return {
                "ok": False,
                "error": f"missing json: {fp}",
                "table": table,
                "hint": "Table must match real file stem exactly (spaces allowed).",
                "available_examples": available,
            }

        raw = fp.read_bytes()
        truncated = False
        if len(raw) > verifier.max_read_bytes:
            raw = raw[: verifier.max_read_bytes]
            truncated = True
        text = raw.decode("utf-8", errors="replace")
        try:
            obj = json.loads(text)
            self.table_json_cache[table] = obj
            summary = self._summarize_table_json_for_llm(
                table_json=obj,
                question=state.question or "",
                evidence=state.external_knowledge or "",
                max_cols=self.read_table_summary_max_cols,
            )
            return {"ok": True, "table": table, "summary": summary, "truncated": truncated}
        except Exception as e:
            return {"ok": False, "table": table, "error": f"json parse error: {e}", "excerpt": text[:1200], "truncated": truncated}

    def _match_keywords(self, text: str, kws: List[str], mode: str) -> List[str]:
        low = (text or "").lower()
        matched = [kw for kw in kws if kw in low]
        if mode == "OR":
            return matched
        return matched if len(matched) == len(kws) else []

    def _search_metadata(
        self,
        verifier: OfflineSchemaVerifier,
        keywords: List[str],
        max_tables: int = 12,
        max_matches_per_table: int = 8,
        mode: str = "OR",
    ) -> Dict[str, Any]:
        """
        Structured metadata search (STRICT version).

        Matching policy:
        - ONLY match on:
            - column names
            - primary keys
            - foreign key endpoints
        - DO NOT match on:
            - table names
            - descriptions

        Return:
        - structured matches with details
        """

        if not isinstance(keywords, list) or not keywords:
            raise ValueError("keywords must be a non-empty list of strings")

        kws = [str(k).strip().lower() for k in keywords if str(k).strip()]
        if not kws:
            raise ValueError("keywords must contain valid strings")

        mode_u = (mode or "OR").upper()
        if mode_u not in ("OR", "AND"):
            raise ValueError("mode must be 'OR' or 'AND'")

        cache_key = f"{mode_u}|||{'|||'.join(kws)}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        results = []

        for fp in sorted(verifier.md_path.glob("*.json")):
            try:
                obj = json.loads(fp.read_text("utf-8", errors="replace"))
            except Exception:
                continue

            table_name = fp.stem
            table_matches = []
            matched_keywords = set()

            # ------------------------------------------------------------
            # 1) column match (ONLY name)
            # ------------------------------------------------------------
            for c in (obj.get("columns") or []):
                if not isinstance(c, dict):
                    continue

                col_name = str(c.get("name", ""))
                hit = self._match_keywords(col_name, kws, mode_u)

                if hit:
                    matched_keywords.update(hit)
                    table_matches.append(
                        {
                            "match_type": "column",
                            "details": {
                                "name": c.get("name", ""),
                                "type": c.get("type", "UNKNOWN"),
                                **({"desc": c.get("desc")} if c.get("desc") else {}),
                            }
                        }
                    )

            # ------------------------------------------------------------
            # 2) primary key match
            # ------------------------------------------------------------
            for pk in (obj.get("primary_keys") or []):
                hit = self._match_keywords(str(pk), kws, mode_u)
                if hit:
                    matched_keywords.update(hit)
                    table_matches.append(
                        {
                            "match_type": "primary_key",
                            "details": {"name": pk},
                        }
                    )

            # ------------------------------------------------------------
            # 3) foreign key match
            # ------------------------------------------------------------
            for fk in (obj.get("foreign_keys") or []):
                if not isinstance(fk, dict):
                    continue

                col = fk.get("column") or fk.get("col") or ""
                ref_table = fk.get("ref_table") or ""
                ref_column = fk.get("ref_column") or ""
                ref = fk.get("ref") or ""

                if (not ref_table or not ref_column) and isinstance(ref, str) and "." in ref:
                    ref_table, ref_column = ref.split(".", 1)

                hay = " ".join([
                    str(col),
                    str(ref_table),
                    str(ref_column),
                    str(ref),
                ])

                hit = self._match_keywords(hay, kws, mode_u)
                if hit:
                    matched_keywords.update(hit)
                    table_matches.append(
                        {
                            "match_type": "foreign_key",
                            "details": {
                                "column": col,
                                "ref_table": ref_table,
                                "ref_column": ref_column,
                                **({"ref": ref} if ref else {}),
                            }
                        }
                    )

            # ------------------------------------------------------------
            # keep only useful tables
            # ------------------------------------------------------------
            if table_matches:
                # dedupe
                seen = set()
                deduped = []
                for m in table_matches:
                    key = json.dumps(m, sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(m)

                results.append(
                    {
                        "table": table_name,
                        "matched_keywords": sorted(matched_keywords),
                        "num_matches": len(deduped),
                        "matches": deduped[:max_matches_per_table],
                    }
                )

        # ranking
        results.sort(
            key=lambda x: (len(x["matched_keywords"]), x["num_matches"]),
            reverse=True,
        )

        results = results[:max_tables]

        out = {
            "ok": True,
            "keywords": kws,
            "mode": mode_u,
            "num_tables": len(results),
            "results": results,
        }

        self.search_cache[cache_key] = out
        return out

    # -------------------------------------------------------------------------
    # schema update tools
    # -------------------------------------------------------------------------
    def _tool_introduce_table(
        self,
        state: SharedGlobalState,
        verifier: OfflineSchemaVerifier,
        args: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        table = _must_nonempty_str(args.get("table"), "table")
        cols = _ensure_list(args.get("columns"), "columns")
        pk = _ensure_list(args.get("primary_key"), "primary_key")
        fks = _ensure_list(args.get("foreign_keys"), "foreign_keys")

        if not cols:
            raise ValueError("columns cannot be empty")

        verifier.verify_columns(table, cols, "columns")
        verifier.verify_pk(table, [str(x) for x in pk])
        #verifier.verify_fks(table, fks, "foreign_keys")

        _apply_introduce_table(state, table, cols, [str(x) for x in pk], fks)
        ev = state.discovered.tables[table]
        update = f"introduced {table} | {_table_summary_for_update(ev)}"
        return {"ok": True, "applied": True, "action": "INTRODUCE_TABLE", "table": table}, update

    def _tool_enrich_table(
        self,
        state: SharedGlobalState,
        verifier: OfflineSchemaVerifier,
        args: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        table = _must_nonempty_str(args.get("table"), "table")
        add_cols = _ensure_list(args.get("add_columns"), "add_columns")
        fks = _ensure_list(args.get("foreign_keys"), "foreign_keys")
        rm_fks = _ensure_list(args.get("remove_foreign_keys"), "remove_foreign_keys")

        if add_cols:
            verifier.verify_columns(table, add_cols, "add_columns")
        #if fks:
        #    verifier.verify_fks(table, fks, "foreign_keys")
        #if rm_fks:
        #    verifier.verify_fks(table, rm_fks, "remove_foreign_keys")

        _apply_enrich_table(state, table, add_cols, fks, rm_fks)
        ev = state.discovered.tables[table]
        update = f"enriched {table} | {_table_summary_for_update(ev)}"
        return {
            "ok": True,
            "applied": True,
            "action": "ENRICH_TABLE",
            "table": table,
            "add_columns": len(add_cols),
            "add_fks": len(fks),
            "rm_fks": len(rm_fks),
        }, update

    def _tool_prune_table(
        self,
        state: SharedGlobalState,
        verifier: OfflineSchemaVerifier,
        args: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        table = _must_nonempty_str(args.get("table"), "table")
        rm_cols = _ensure_list(args.get("remove_columns"), "remove_columns")
        rm_fks = _ensure_list(args.get("remove_foreign_keys"), "remove_foreign_keys")

        for c in rm_cols:
            c = _must_nonempty_str(c, "remove_columns item")
            if not verifier.col_exists(table, c):
                raise ValueError(f"remove_columns '{c}' not in metadata for table '{table}'")
        #if rm_fks:
        #    verifier.verify_fks(table, rm_fks, "remove_foreign_keys")

        _apply_prune_table(state, table, [str(x) for x in rm_cols], rm_fks)
        ev = state.discovered.tables.get(table)
        update = f"pruned {table} | {_table_summary_for_update(ev) if ev else 'removed/empty'}"
        return {
            "ok": True,
            "applied": True,
            "action": "PRUNE_TABLE",
            "table": table,
            "rm_columns": len(rm_cols),
            "rm_fks": len(rm_fks),
        }, update

    def _tool_drop_table(
        self,
        state: SharedGlobalState,
        verifier: OfflineSchemaVerifier,
        args: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        table = _must_nonempty_str(args.get("table"), "table")
        if not verifier.table_exists(table):
            raise ValueError(f"table metadata missing: {table}.json")
        _apply_drop_table(state, table)
        update = f"dropped {table}"
        return {"ok": True, "applied": True, "action": "DROP_TABLE", "table": table}, update

    # -------------------------------------------------------------------------
    # execute tool calls
    # -------------------------------------------------------------------------
    def _execute_tool_calls(
        self,
        message: Any,
        messages: List[Dict[str, Any]],
        state: SharedGlobalState,
        verifier: OfflineSchemaVerifier,
    ) -> Optional[AgentReturn]:
        if not getattr(message, "tool_calls", None):
            return None

        seen = set()
        unique = []
        for tc in message.tool_calls:
            if tc.type != "function":
                continue
            key = (tc.function.name, tc.function.arguments)
            if key in seen:
                continue
            seen.add(key)
            unique.append(tc)

        # NOTE:
        # SDA supports MULTIPLE tool calls in one turn.
        # We cap by max_tool_calls_per_turn.
        tool_calls = unique[: self.max_tool_calls_per_turn]

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )

        schema_updates: List[str] = []

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception as e:
                logger.error("[SDA] tool args JSON parse failed | tool=%s | raw=%r", name, tc.function.arguments)

                return AgentReturn(
                    agent=AgentName.SCHEMA_DISCOVERY,
                    ok=False,
                    report=f"SDA tool arguments invalid JSON for {name}: {e}",
                    payload={
                        "reason": "invalid_tool_arguments_json",
                        "tool": name,
                        "raw_arguments": tc.function.arguments,
                    },
                )

            try:
                if name == "SEARCH_METADATA":
                    kws = args.get("keywords", None)
                    if kws is None:
                        kw = str(args.get("keyword", "")).strip()
                        kws = [kw] if kw else []
                    elif isinstance(kws, str):
                        kws = [kws.strip()] if kws.strip() else []
                    elif isinstance(kws, list):
                        kws = [str(x).strip() for x in kws if str(x).strip()]
                    else:
                        kws = []
                    out = self._search_metadata(verifier, keywords=kws)

                elif name == "READ_TABLE_JSON":
                    out = self._read_table_json(verifier, str(args.get("table")), state)

                elif name == "INTRODUCE_TABLE":
                    out, update = self._tool_introduce_table(state, verifier, args)
                    schema_updates.append(update)
                    logger.info("[SDA] applied: INTRODUCE_TABLE(%s)", args.get("table"))

                elif name == "ENRICH_TABLE":
                    out, update = self._tool_enrich_table(state, verifier, args)
                    schema_updates.append(update)
                    logger.info("[SDA] applied: ENRICH_TABLE(%s)", args.get("table"))

                elif name == "PRUNE_TABLE":
                    out, update = self._tool_prune_table(state, verifier, args)
                    schema_updates.append(update)
                    logger.info("[SDA] applied: PRUNE_TABLE(%s)", args.get("table"))

                elif name == "DROP_TABLE":
                    out, update = self._tool_drop_table(state, verifier, args)
                    schema_updates.append(update)
                    logger.info("[SDA] applied: DROP_TABLE(%s)", args.get("table"))

                elif name == "FINISH_SCHEMA_DISCOVERY":
                    report = _must_nonempty_str(args.get("report"), "report")
                    logger.info("[SDA] FINISH called | report=%s", report)
                    out = {"ok": True, "finished": True, "report": report}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": _wrap_tool_content(
                                name,
                                json.dumps(out, ensure_ascii=False, indent=2),
                                self.tool_output_max_chars,
                            ),
                        }
                    )
                    return AgentReturn(
                        agent=AgentName.SCHEMA_DISCOVERY,
                        ok=True,
                        report=report,
                        payload={"schema": state.discovered.tables},
                    )

                else:
                    out = {"ok": False, "error": f"Unknown tool: {name}"}

            except Exception as e:
                out = {"ok": False, "applied": False, "tool": name, "error": str(e)}

            content = json.dumps(out, ensure_ascii=False, indent=2)
            if self.debug:
                cprint(f"[SDA Tool:{name}] {content[:1500]}", "green")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _wrap_tool_content(name, content, self.tool_output_max_chars),
                }
            )

        if schema_updates:
            update_msg = self._schema_update_message(schema_updates)
            self.last_schema_update_msg = update_msg
            logger.info("[SDA] schema_updates:\n%s", "\n".join(schema_updates))
            logger.info("[SDA] current_discovered_schema:\n%s", _schema_to_json(state))
            self._compact_after_schema_update(messages)

        return None

    # -------------------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------------------
    def run(self, state: SharedGlobalState, guidance: str) -> AgentReturn:
        md_path = (state.schema_metadata_path or "").strip()
        if not md_path:
            return AgentReturn(
                agent=AgentName.SCHEMA_DISCOVERY,
                ok=False,
                report="schema_metadata_path is empty; cannot inspect metadata.",
                payload={},
            )

        verifier = OfflineSchemaVerifier(md_path, max_read_bytes=self.max_read_bytes)

        system_prompt = build_schema_discovery_system_prompt(
            schema_metadata_path=md_path,
            metadata_display=state.metadata_display,
            db_name=state.db_id,
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"USER QUESTION:\n{(state.question or '').strip()}\n\n"
                    f"EXTERNAL EVIDENCE:\n{(state.external_knowledge or '').strip()}\n\n"
                    f"PLANNER INSTRUCTION:\n{(guidance or '').strip()}\n\n"
                    "Current discovered_schema:\n"
                    f"{_compact_schema_text_for_sda(state)}\n"
                    "Call a tool now."
                ),
            },
        ]

        last_text = ""

        for turn_idx in range(1, self.max_turns + 1):
            self._trim_history(messages)

            msg = self._call_llm(messages)
            if not msg:
                logger.error("[SDA] turn=%s | LLM returned None", turn_idx)
                break

            tool_calls = getattr(msg, "tool_calls", None)
            raw_text = (getattr(msg, "content", "") or "")
            visible_text = _strip_thinking_block(raw_text)
            last_text = visible_text

            has_tool = bool(tool_calls)

            logger.info(
                "[SDA] turn=%s | has_tool=%s tool_calls=%s | text_len=%s",
                turn_idx,
                has_tool,
                (len(tool_calls) if tool_calls else 0),
                len(visible_text),
            )

            if not has_tool:
                logger.error("[SDA] turn=%s | invalid: no tool call", turn_idx)

                return AgentReturn(
                    agent=AgentName.SCHEMA_DISCOVERY,
                    ok=False,
                    report="SDA invalid turn: must call a tool (schema update or FINISH)",
                    payload={
                        "reason": "no_tool_call",
                        "turn_idx": turn_idx,
                        "raw_text": visible_text[:2000],
                    },
                )

            ret = self._execute_tool_calls(
                message=msg,
                messages=messages,
                state=state,
                verifier=verifier,
            )
            if ret is not None:
                return ret

            if self.last_schema_update_msg:
                follow_up = "Tool results received.\n" + self.last_schema_update_msg
            else:
                follow_up = (
                    "Tool results received.\n"
                    "No schema updates were applied in the last turn.\n"
                    "Continue with tools, or call FINISH_SCHEMA_DISCOVERY."
                )

            messages.append({"role": "user", "content": follow_up})

        return AgentReturn(
            agent=AgentName.SCHEMA_DISCOVERY,
            ok=False,
            report="SDA did not finish within max_turns.",
            payload={"raw": last_text[:2000], "schema": state.discovered.tables},
        )