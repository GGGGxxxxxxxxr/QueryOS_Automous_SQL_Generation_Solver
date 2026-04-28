from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List


WRITE_ACTIONS = {
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_REINDEX,
    sqlite3.SQLITE_ANALYZE,
}


def compact_jsonable(value: Any, max_cell_chars: int) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    if len(text) > max_cell_chars:
        return f"[CELL_TRUNCATED len={len(text)}] {text[:max_cell_chars]}"
    return text


class SQLiteExecutor:
    def __init__(
        self,
        timeout_s: int = 60,
        max_rows: int = 50,
        max_cell_chars: int = 1000,
        max_total_chars: int = 8000,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self.max_cell_chars = max_cell_chars
        self.max_total_chars = max_total_chars

    def execute(self, db_path: str, sql: str) -> Dict[str, Any]:
        sql = (sql or "").strip()
        if not sql:
            return {"ok": False, "result": {}, "error": "empty SQL"}
        if not self._looks_read_only(sql):
            return {"ok": False, "result": {}, "error": "only read-only SELECT/WITH/VALUES queries are allowed"}

        path = Path(db_path).expanduser().resolve()
        if not path.exists():
            return {"ok": False, "result": {}, "error": f"SQLite database not found: {path}"}

        conn = None
        try:
            conn = sqlite3.connect(str(path), timeout=self.timeout_s)
            conn.execute(f"PRAGMA busy_timeout = {self.timeout_s * 1000}")
            conn.set_authorizer(self._authorizer)

            cursor = conn.cursor()
            timer = threading.Timer(self.timeout_s, conn.interrupt)
            timer.daemon = True
            timer.start()
            try:
                cursor.execute(sql)
            finally:
                timer.cancel()

            if cursor.description is None:
                return {
                    "ok": False,
                    "result": {},
                    "error": "query did not return rows; only read-only result queries are allowed",
                }

            columns = [desc[0] for desc in cursor.description]
            raw_rows = cursor.fetchmany(self.max_rows + 1)
            truncated = len(raw_rows) > self.max_rows
            raw_rows = raw_rows[: self.max_rows]

            rows: List[List[Any]] = []
            total_chars = 0
            for raw_row in raw_rows:
                row = []
                for cell in raw_row:
                    total_chars += len("" if cell is None else str(cell))
                    row.append(compact_jsonable(cell, self.max_cell_chars))
                rows.append(row)

            warnings = []
            if truncated:
                warnings.append("ROW_LIMIT_REACHED")
            if total_chars > self.max_total_chars:
                warnings.append("RESULT_LARGE")

            return {
                "ok": True,
                "result": {
                    "columns": columns,
                    "rows": rows,
                    "truncated": truncated,
                    "max_rows": self.max_rows,
                    "warnings": warnings,
                },
                "error": "",
            }
        except sqlite3.OperationalError as exc:
            return {"ok": False, "result": {}, "error": f"sqlite error: {exc}"}
        except Exception as exc:
            return {"ok": False, "result": {}, "error": str(exc)}
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _looks_read_only(sql: str) -> bool:
        stripped = re.sub(r"(?s)/\*.*?\*/", " ", sql).strip()
        stripped = re.sub(r"(?m)--.*?$", " ", stripped).strip()
        first = re.match(r"^[A-Za-z]+", stripped)
        if not first:
            return False
        return first.group(0).lower() in {"select", "with", "values"}

    @staticmethod
    def _authorizer(action: int, arg1: str, arg2: str, dbname: str, source: str) -> int:
        if action in WRITE_ACTIONS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

