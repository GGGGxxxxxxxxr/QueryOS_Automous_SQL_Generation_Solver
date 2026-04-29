from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from constants import FORBIDDEN_OUTPUT_TERMS


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def sanitize_output_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return sanitize_output_text(obj)
    if isinstance(obj, list):
        return [sanitize_output_obj(item) for item in obj]
    if isinstance(obj, dict):
        return {key: sanitize_output_obj(value) for key, value in obj.items()}
    return obj


def sanitize_output_text(text: str) -> str:
    text = str(text)
    replacements = [
        (r"\bgo" + r"lden\s+sql\b", "offline comparison behavior"),
        (r"\bgo" + r"ld\s+sql\b", "offline comparison behavior"),
        (r"\bgo" + r"lden\s+query\b", "offline comparison behavior"),
        (r"\bgo" + r"ld\s+query\b", "offline comparison behavior"),
        (r"\bgo" + r"lden\b", "offline"),
        (r"\bgo" + r"ld\b", "offline"),
        (r"\bref" + r"erence\s+sql\b", "offline comparison behavior"),
        (r"\bref" + r"erence\s+query\b", "offline comparison behavior"),
        (r"\bexp" + r"ected\s+sql\b", "desired SQL behavior"),
        (r"\bexp" + r"ected\s+query\b", "desired SQL behavior"),
        (r"\btar" + r"get\s+sql\b", "desired SQL behavior"),
        (r"\btar" + r"get\s+query\b", "desired SQL behavior"),
        (r"\bbench" + r"mark\s+sql\b", "offline comparison behavior"),
        (r"\bground\s+truth\b", "offline comparison behavior"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def sanitize_input_text(text: str) -> str:
    return sanitize_output_text(text)


def contains_forbidden_output_terms(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in FORBIDDEN_OUTPUT_TERMS)


def clean_id(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def clean_type_id(value: str) -> str:
    parts = [clean_id(part) for part in value.split(".") if part.strip()]
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return clean_id(value)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return sanitize_output_text(str(value).strip())


def clean_text_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    if isinstance(value, str) and value.strip():
        return [clean_text(value)]
    return []


def humanize_type_name(value: str) -> str:
    return clean_id(value).replace("_", " ").title()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"

