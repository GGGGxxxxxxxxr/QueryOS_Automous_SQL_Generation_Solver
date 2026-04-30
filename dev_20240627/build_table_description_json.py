#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build per-table JSON metadata from dev_tables.json, column_meaning.json, and database_description CSV files."
    )
    parser.add_argument("--root", default=".", help="Dataset root containing dev_tables.json and dev_databases/")
    parser.add_argument(
        "--output-dir-name",
        default="database_description",
        help="Directory under each DB where table JSON files are written. Defaults to the existing database_description folder.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output JSON files.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    tables_path = root / "dev_tables.json"
    meanings_path = root / "column_meaning.json"
    db_root = root / "dev_databases"

    dev_tables = json.loads(tables_path.read_text(encoding="utf-8"))
    table_by_db = {item["db_id"]: item for item in dev_tables}
    column_meanings = json.loads(meanings_path.read_text(encoding="utf-8")) if meanings_path.exists() else {}

    summary = {
        "db_count": 0,
        "table_count": 0,
        "json_written": 0,
        "warnings": [],
    }

    for db_dir in sorted(p for p in db_root.iterdir() if p.is_dir()):
        db_id = db_dir.name
        if db_id not in table_by_db:
            summary["warnings"].append(f"skip {db_id}: missing from dev_tables.json")
            continue

        schema = table_by_db[db_id]
        csv_dir = db_dir / "database_description"
        output_dir = db_dir / args.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_by_table = load_csv_descriptions(csv_dir)
        table_docs = build_db_table_docs(db_id, schema, csv_by_table, column_meanings, summary["warnings"])

        summary["db_count"] += 1
        for table_name, doc in table_docs.items():
            summary["table_count"] += 1
            out_path = output_dir / f"{table_name}.json"
            if out_path.exists() and not args.overwrite:
                continue
            out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            summary["json_written"] += 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_db_table_docs(
    db_id: str,
    schema: Dict[str, Any],
    csv_by_table: Dict[str, Dict[str, Dict[str, str]]],
    column_meanings: Dict[str, str],
    warnings: List[str],
) -> Dict[str, Dict[str, Any]]:
    table_names = schema["table_names_original"]
    table_display_names = schema.get("table_names") or table_names
    column_names = schema["column_names_original"]
    column_display_names = schema.get("column_names") or column_names
    column_types = schema.get("column_types") or []

    columns_by_table: Dict[int, List[Tuple[int, str]]] = {i: [] for i in range(len(table_names))}
    for idx, pair in enumerate(column_names):
        table_idx, column_name = pair
        if table_idx == -1:
            continue
        columns_by_table.setdefault(table_idx, []).append((idx, column_name))

    primary_key_indices = flatten_primary_keys(schema.get("primary_keys") or [])
    primary_keys_by_table: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
    for column_idx in primary_key_indices:
        if column_idx >= len(column_names):
            warnings.append(f"{db_id}: primary key index out of range: {column_idx}")
            continue
        table_idx, column_name = column_names[column_idx]
        if table_idx != -1:
            primary_keys_by_table.setdefault(table_idx, []).append(column_name)

    foreign_keys_by_table: Dict[int, List[Dict[str, str]]] = {i: [] for i in range(len(table_names))}
    for fk in schema.get("foreign_keys") or []:
        if not isinstance(fk, list) or len(fk) != 2:
            warnings.append(f"{db_id}: invalid foreign key entry: {fk}")
            continue
        source_idx, target_idx = fk
        if source_idx >= len(column_names) or target_idx >= len(column_names):
            warnings.append(f"{db_id}: foreign key index out of range: {fk}")
            continue
        source_table_idx, source_col = column_names[source_idx]
        target_table_idx, target_col = column_names[target_idx]
        if source_table_idx == -1 or target_table_idx == -1:
            continue
        foreign_keys_by_table.setdefault(source_table_idx, []).append(
            {
                "column": source_col,
                "ref_table": table_names[target_table_idx],
                "ref_column": target_col,
            }
        )

    docs: Dict[str, Dict[str, Any]] = {}
    for table_idx, table_name in enumerate(table_names):
        table_csv = csv_by_table.get(norm_key(table_name), {})
        columns = []
        for column_idx, column_name in columns_by_table.get(table_idx, []):
            csv_desc = table_csv.get(norm_key(column_name), {})
            display_name = get_column_display_name(column_display_names, column_idx)
            meaning = clean_meaning(
                column_meanings.get(f"{db_id}|{table_name}|{column_name}", "")
            )
            csv_description = clean_text(csv_desc.get("column_description", ""))
            csv_column_name = clean_text(csv_desc.get("column_name", ""))
            data_format = clean_type(csv_desc.get("data_format", ""))
            schema_type = clean_type(column_types[column_idx]) if column_idx < len(column_types) else ""

            column_doc: Dict[str, Any] = {
                "name": column_name,
                "type": schema_type or data_format or "UNKNOWN",
            }

            desc = first_nonempty(meaning, csv_description, display_name, column_name)
            if desc:
                column_doc["desc"] = desc
            if meaning and meaning != desc:
                column_doc["column_meaning"] = meaning

            columns.append(column_doc)

        docs[table_name] = {
            "db_id": db_id,
            "table": table_name,
            "table_description": table_display_names[table_idx] if table_idx < len(table_display_names) else table_name,
            "primary_keys": primary_keys_by_table.get(table_idx, []),
            "foreign_keys": foreign_keys_by_table.get(table_idx, []),
            "columns": columns,
            "source": {
                "schema": "dev_tables.json",
                "column_descriptions": f"dev_databases/{db_id}/database_description/{table_name}.csv",
                "column_meanings": "column_meaning.json",
            },
        }

    return docs


def load_csv_descriptions(csv_dir: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    if not csv_dir.exists():
        return out
    for path in sorted(csv_dir.glob("*.csv")):
        rows = read_csv_dicts(path)
        table_key = norm_key(path.stem)
        out[table_key] = {}
        for row in rows:
            original = clean_text(row.get("original_column_name", ""))
            if not original:
                continue
            cleaned = {clean_header(k): clean_text(v) for k, v in row.items()}
            out[table_key][norm_key(original)] = cleaned
    return out


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    last_error: Optional[Exception] = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                return [dict(row) for row in reader]
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"failed to read {path}: {last_error}")


def flatten_primary_keys(primary_keys: Iterable[Any]) -> List[int]:
    out: List[int] = []
    for item in primary_keys:
        if isinstance(item, list):
            out.extend(int(x) for x in item)
        else:
            out.append(int(item))
    return out


def get_column_display_name(column_display_names: List[Any], column_idx: int) -> str:
    if column_idx >= len(column_display_names):
        return ""
    pair = column_display_names[column_idx]
    if isinstance(pair, list) and len(pair) == 2:
        return clean_text(pair[1])
    return ""


def clean_meaning(value: Any) -> str:
    text = clean_text(value)
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    text = re.sub(r"^#+\s*", "", text)
    return clean_text(text)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").strip()
    text = re.sub(r"[ \t]+", " ", text)
    return text


def clean_type(value: Any) -> str:
    return clean_text(value).lower()


def clean_header(value: Any) -> str:
    return clean_text(value).lower()


def norm_key(value: Any) -> str:
    return re.sub(r"\s+", "", clean_text(value).lower())


def first_nonempty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
