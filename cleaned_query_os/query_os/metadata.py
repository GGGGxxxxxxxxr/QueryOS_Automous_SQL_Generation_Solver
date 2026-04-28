from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def normalize_name(value: str) -> str:
    return "".join((value or "").strip().lower().split())


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class SchemaMetadataStore:
    """Table metadata used by the schema discovery worker.

    The store accepts the Spider-style `database_description/*.json` files, but
    can also introspect a SQLite database directly so callers only need a DB path.
    """

    def __init__(self, tables: Dict[str, Dict[str, Any]], source: str = "") -> None:
        self.tables = tables
        self.source = source

    @classmethod
    def from_path(cls, metadata_path: str) -> "SchemaMetadataStore":
        path = Path(metadata_path).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"metadata directory not found: {path}")

        tables: Dict[str, Dict[str, Any]] = {}
        for fp in sorted(path.glob("*.json")):
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"failed to parse {fp}: {exc}") from exc

            table = str(obj.get("table") or fp.stem).strip()
            if not table:
                continue
            tables[table] = _normalize_table_json(obj, table)

        if not tables:
            raise ValueError(f"no table json files found in {path}")
        return cls(tables=tables, source=str(path))

    @classmethod
    def from_sqlite(cls, db_path: str) -> "SchemaMetadataStore":
        path = Path(db_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"SQLite database not found: {path}")

        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute(
                """
                SELECT name, type
                FROM sqlite_master
                WHERE type IN ('table', 'view')
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()

            tables: Dict[str, Dict[str, Any]] = {}
            for table, table_type in rows:
                columns = []
                primary_keys = []
                for cid, name, col_type, notnull, default, pk in conn.execute(
                    f"PRAGMA table_info({quote_identifier(table)})"
                ).fetchall():
                    col = {
                        "name": str(name),
                        "type": str(col_type or "UNKNOWN"),
                    }
                    if table_type == "view":
                        col["desc"] = "SQLite view column"
                    columns.append(col)
                    if pk:
                        primary_keys.append(str(name))

                foreign_keys = []
                try:
                    fk_rows = conn.execute(
                        f"PRAGMA foreign_key_list({quote_identifier(table)})"
                    ).fetchall()
                except sqlite3.DatabaseError:
                    fk_rows = []

                for row in fk_rows:
                    # SQLite columns: id, seq, table, from, to, on_update, on_delete, match
                    ref_table = str(row[2])
                    from_col = str(row[3])
                    to_col = str(row[4])
                    foreign_keys.append(
                        {
                            "column": from_col,
                            "ref_table": ref_table,
                            "ref_column": to_col,
                        }
                    )

                tables[str(table)] = {
                    "table": str(table),
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys,
                }
        finally:
            conn.close()

        if not tables:
            raise ValueError(f"no user tables found in SQLite database: {path}")
        return cls(tables=tables, source=str(path))

    def list_tables(self) -> List[str]:
        return sorted(self.tables)

    def display(self) -> str:
        return "\n".join(f"- {name}" for name in self.list_tables())

    def table_exists(self, table: str) -> bool:
        return table in self.tables

    def get_table(self, table: str) -> Dict[str, Any]:
        if table not in self.tables:
            examples = ", ".join(self.list_tables()[:20])
            raise KeyError(f"unknown table '{table}'. Available examples: {examples}")
        return self.tables[table]

    def column_exists(self, table: str, column: str) -> bool:
        obj = self.get_table(table)
        wanted = normalize_name(column)
        return wanted in {
            normalize_name(str(c.get("name", "")))
            for c in obj.get("columns", [])
            if isinstance(c, dict)
        }

    def foreign_key_exists(self, table: str, column: str, ref_table: str, ref_column: str) -> bool:
        obj = self.get_table(table)
        wanted = (normalize_name(column), normalize_name(ref_table), normalize_name(ref_column))
        for fk in obj.get("foreign_keys", []) or []:
            col, parsed_ref_table, parsed_ref_column = parse_foreign_key(fk)
            parsed = (
                normalize_name(col),
                normalize_name(parsed_ref_table),
                normalize_name(parsed_ref_column),
            )
            if parsed == wanted:
                return True
        return False

    def verify_columns(self, table: str, columns: Iterable[Dict[str, Any]], field: str) -> None:
        if not self.table_exists(table):
            raise ValueError(f"metadata missing for table '{table}'")
        for column in columns:
            if not isinstance(column, dict):
                raise ValueError(f"{field} entries must be objects")
            name = str(column.get("name", "")).strip()
            if not name:
                raise ValueError(f"{field}.name is required")
            if not self.column_exists(table, name):
                raise ValueError(f"column '{name}' not found in table '{table}'")

    def verify_primary_keys(self, table: str, primary_keys: Iterable[str]) -> None:
        for pk in primary_keys:
            if not self.column_exists(table, str(pk)):
                raise ValueError(f"primary key '{pk}' not found in table '{table}'")

    def verify_foreign_keys(self, table: str, foreign_keys: Iterable[Dict[str, Any]], field: str) -> None:
        for fk in foreign_keys:
            col, ref_table, ref_column = parse_foreign_key(fk, strict=True)
            if not self.column_exists(table, col):
                raise ValueError(f"{field}: '{table}.{col}' does not exist")
            if not self.table_exists(ref_table):
                raise ValueError(f"{field}: referenced table '{ref_table}' does not exist")
            if not self.column_exists(ref_table, ref_column):
                raise ValueError(f"{field}: referenced column '{ref_table}.{ref_column}' does not exist")
            if not self.foreign_key_exists(table, col, ref_table, ref_column):
                raise ValueError(f"{field}: foreign key '{table}.{col} -> {ref_table}.{ref_column}' not in metadata")

    def search(
        self,
        keywords: List[str],
        mode: str = "OR",
        max_tables: int = 12,
        max_matches_per_table: int = 12,
    ) -> Dict[str, Any]:
        kws = [k.strip().lower() for k in keywords if str(k).strip()]
        if not kws:
            raise ValueError("keywords must contain at least one string")
        mode = (mode or "OR").upper()
        if mode not in {"OR", "AND"}:
            raise ValueError("mode must be OR or AND")

        results = []
        for table, obj in sorted(self.tables.items()):
            matches: List[Dict[str, Any]] = []
            matched_keywords: Set[str] = set()

            def maybe_match(text: str) -> List[str]:
                low = (text or "").lower()
                hits = [kw for kw in kws if kw in low]
                if mode == "AND" and len(hits) != len(kws):
                    return []
                return hits

            for hit in maybe_match(table):
                matched_keywords.add(hit)
                matches.append({"match_type": "table", "details": {"name": table}})

            for column in obj.get("columns", []) or []:
                if not isinstance(column, dict):
                    continue
                haystack = " ".join(
                    str(column.get(key, ""))
                    for key in ("name", "type", "desc", "description")
                )
                hits = maybe_match(haystack)
                if hits:
                    matched_keywords.update(hits)
                    matches.append(
                        {
                            "match_type": "column",
                            "details": {
                                "name": column.get("name", ""),
                                "type": column.get("type", "UNKNOWN"),
                                **({"desc": column.get("desc")} if column.get("desc") else {}),
                            },
                        }
                    )

            for pk in obj.get("primary_keys", []) or []:
                hits = maybe_match(str(pk))
                if hits:
                    matched_keywords.update(hits)
                    matches.append({"match_type": "primary_key", "details": {"name": pk}})

            for fk in obj.get("foreign_keys", []) or []:
                try:
                    col, ref_table, ref_column = parse_foreign_key(fk, strict=False)
                except Exception:
                    continue
                haystack = f"{col} {ref_table} {ref_column}"
                hits = maybe_match(haystack)
                if hits:
                    matched_keywords.update(hits)
                    matches.append(
                        {
                            "match_type": "foreign_key",
                            "details": {
                                "column": col,
                                "ref_table": ref_table,
                                "ref_column": ref_column,
                            },
                        }
                    )

            if matches:
                results.append(
                    {
                        "table": table,
                        "matched_keywords": sorted(matched_keywords),
                        "num_matches": len(matches),
                        "matches": matches[:max_matches_per_table],
                    }
                )

        results.sort(key=lambda r: (len(r["matched_keywords"]), r["num_matches"]), reverse=True)
        return {
            "ok": True,
            "keywords": kws,
            "mode": mode,
            "num_tables": min(len(results), max_tables),
            "results": results[:max_tables],
        }


def _normalize_table_json(obj: Dict[str, Any], fallback_table: str) -> Dict[str, Any]:
    table = str(obj.get("table") or fallback_table).strip()
    columns = []
    for col in obj.get("columns", []) or []:
        if isinstance(col, dict) and col.get("name"):
            columns.append(
                {
                    "name": str(col.get("name", "")).strip(),
                    "type": str(col.get("type") or col.get("data_type") or "UNKNOWN"),
                    **({"desc": str(col.get("desc") or col.get("description"))} if col.get("desc") or col.get("description") else {}),
                }
            )
        elif isinstance(col, str) and col.strip():
            columns.append({"name": col.strip(), "type": "UNKNOWN"})

    foreign_keys = []
    for fk in obj.get("foreign_keys", []) or []:
        try:
            col, ref_table, ref_column = parse_foreign_key(fk, strict=False)
        except Exception:
            continue
        if col and ref_table and ref_column:
            foreign_keys.append(
                {"column": col, "ref_table": ref_table, "ref_column": ref_column}
            )

    return {
        "table": table,
        "columns": columns,
        "primary_keys": [str(pk).strip() for pk in obj.get("primary_keys", []) or [] if str(pk).strip()],
        "foreign_keys": foreign_keys,
    }


def parse_foreign_key(fk: Dict[str, Any], strict: bool = False) -> Tuple[str, str, str]:
    if not isinstance(fk, dict):
        raise ValueError("foreign key must be an object")
    col = str(fk.get("column") or fk.get("col") or "").strip()
    ref_table = str(fk.get("ref_table") or "").strip()
    ref_column = str(fk.get("ref_column") or "").strip()
    ref = fk.get("ref")
    if (not ref_table or not ref_column) and isinstance(ref, str) and "." in ref:
        ref_table, ref_column = [part.strip() for part in ref.split(".", 1)]
    if strict and (not col or not ref_table or not ref_column):
        raise ValueError("foreign key requires column, ref_table, and ref_column")
    return col, ref_table, ref_column
