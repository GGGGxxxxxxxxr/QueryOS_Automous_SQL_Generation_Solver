from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from text_utils import safe_int, sanitize_output_obj


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield loaded


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_output_obj(obj), ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_output_obj(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_processed_question_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids = set()
    for item in load_jsonl(path):
        qid = safe_int(item.get("source_question_id"), default=-1)
        if qid >= 0:
            ids.add(qid)
    return ids


def filter_records(records: List[Dict[str, Any]], question_ids: Optional[List[int]]) -> List[Dict[str, Any]]:
    if not question_ids:
        return records
    wanted = set(question_ids)
    return [record for record in records if safe_int(record.get("question_id"), -1) in wanted]

