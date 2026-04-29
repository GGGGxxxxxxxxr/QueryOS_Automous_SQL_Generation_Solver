from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from constants import SEED_FAMILIES
from text_utils import (
    clean_id,
    clean_text,
    clean_text_list,
    clean_type_id,
    humanize_type_name,
    safe_float,
    safe_int,
    sanitize_output_obj,
)


def normalize_atomic_items(result: Dict[str, Any], record: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_items = result.get("atomic_mistakes")
    if not isinstance(raw_items, list):
        return []
    normalized = []
    for idx, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            continue
        family = clean_id(str(raw.get("proposed_family") or "execution_strategy"))
        if family not in SEED_FAMILIES:
            family = "execution_strategy"
        mistake_type = clean_id(str(raw.get("proposed_type") or "unknown_mistake_type"))
        item = {
            "atomic_id": f"AM-{safe_int(record.get('question_id'), -1)}-{idx}",
            "source_question_id": safe_int(record.get("question_id"), -1),
            "db_id": str(record.get("db_id") or ""),
            "difficulty": str(record.get("difficulty") or ""),
            "observed_failure": clean_text(raw.get("observed_failure")),
            "risky_behavior": clean_text_list(raw.get("risky_behavior")),
            "diagnostic_signals": clean_text_list(raw.get("diagnostic_signals")),
            "repair_principle": clean_text(raw.get("repair_principle")),
            "exceptions": clean_text_list(raw.get("exceptions")),
            "risky_sql_shapes": clean_text_list(raw.get("risky_sql_shapes")),
            "proposed_family": family,
            "proposed_type": mistake_type,
            "type_definition": clean_text(raw.get("type_definition")),
            "inclusion_criteria": clean_text_list(raw.get("inclusion_criteria")),
            "exclusion_criteria": clean_text_list(raw.get("exclusion_criteria")),
            "routing": normalize_routing(raw.get("routing")),
        }
        normalized.append(sanitize_output_obj(item))
    return normalized


def normalize_routing(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    decision = str(raw.get("decision") or "PROPOSE_NEW").strip().upper()
    if decision not in {"ATTACH_ACTIVE", "VOTE_PROPOSED", "PROPOSE_NEW"}:
        decision = "PROPOSE_NEW"
    return {
        "decision": decision,
        "active_type_id": clean_type_id(str(raw.get("active_type_id") or "")),
        "proposal_id": str(raw.get("proposal_id") or "").strip(),
        "confidence": safe_float(raw.get("confidence"), 0.0),
        "reason": clean_text(raw.get("reason")),
    }


def apply_taxonomy_routing(
    *,
    taxonomy: Dict[str, Any],
    atomic: Dict[str, Any],
    promotion_threshold: int,
) -> Dict[str, Any]:
    active_types = taxonomy.setdefault("active_types", [])
    proposed_types = taxonomy.setdefault("proposed_types", [])
    routing = atomic.get("routing") or {}
    decision = routing.get("decision")
    question_id = atomic.get("source_question_id")

    active_by_id = {item.get("id"): item for item in active_types if isinstance(item, dict)}
    proposed_by_id = {item.get("proposal_id"): item for item in proposed_types if isinstance(item, dict)}

    if decision == "ATTACH_ACTIVE" and routing.get("active_type_id") in active_by_id:
        target = active_by_id[routing["active_type_id"]]
        add_support(target, atomic)
        return {"decision": "attached_active", "active_type_id": target["id"], "promoted": False}

    if decision == "VOTE_PROPOSED" and routing.get("proposal_id") in proposed_by_id:
        target = proposed_by_id[routing["proposal_id"]]
        add_support(target, atomic)
        promoted = maybe_promote_type(taxonomy, target, promotion_threshold)
        return {
            "decision": "voted_proposed",
            "proposal_id": target["proposal_id"],
            "active_type_id": promoted.get("id", ""),
            "promoted": bool(promoted),
        }

    duplicate_active = find_type_by_family_name(active_types, atomic["proposed_family"], atomic["proposed_type"])
    if duplicate_active:
        add_support(duplicate_active, atomic)
        return {"decision": "attached_active_duplicate_name", "active_type_id": duplicate_active["id"], "promoted": False}

    duplicate_proposed = find_type_by_family_name(proposed_types, atomic["proposed_family"], atomic["proposed_type"])
    if duplicate_proposed:
        add_support(duplicate_proposed, atomic)
        promoted = maybe_promote_type(taxonomy, duplicate_proposed, promotion_threshold)
        return {
            "decision": "voted_proposed_duplicate_name",
            "proposal_id": duplicate_proposed["proposal_id"],
            "active_type_id": promoted.get("id", ""),
            "promoted": bool(promoted),
        }

    proposal = make_proposed_type(taxonomy, atomic)
    proposed_types.append(proposal)
    if question_id is not None:
        add_support(proposal, atomic)
    promoted = maybe_promote_type(taxonomy, proposal, promotion_threshold)
    return {
        "decision": "created_proposed",
        "proposal_id": proposal["proposal_id"],
        "active_type_id": promoted.get("id", ""),
        "promoted": bool(promoted),
    }


def make_proposed_type(taxonomy: Dict[str, Any], atomic: Dict[str, Any]) -> Dict[str, Any]:
    next_id = int(taxonomy.get("next_proposal_idx", 1))
    taxonomy["next_proposal_idx"] = next_id + 1
    family = atomic["proposed_family"]
    mistake_type = atomic["proposed_type"]
    return {
        "proposal_id": f"PT-{next_id:06d}",
        "family": family,
        "type": mistake_type,
        "id": f"{family}.{mistake_type}",
        "name": humanize_type_name(mistake_type),
        "definition": atomic.get("type_definition") or atomic.get("observed_failure") or "",
        "inclusion_criteria": list(atomic.get("inclusion_criteria") or []),
        "exclusion_criteria": list(atomic.get("exclusion_criteria") or []),
        "risky_behavior": list(atomic.get("risky_behavior") or []),
        "diagnostic_signals": list(atomic.get("diagnostic_signals") or []),
        "repair_principles": [atomic.get("repair_principle")] if atomic.get("repair_principle") else [],
        "exceptions": list(atomic.get("exceptions") or []),
        "support_count": 0,
        "supporting_question_ids": [],
        "example_atomic_ids": [],
        "status": "proposed",
    }


def add_support(target: Dict[str, Any], atomic: Dict[str, Any]) -> None:
    target["support_count"] = int(target.get("support_count", 0) or 0) + 1
    question_id = atomic.get("source_question_id")
    ids = list(target.get("supporting_question_ids") or [])
    if question_id not in ids:
        ids.append(question_id)
    target["supporting_question_ids"] = ids
    examples = list(target.get("example_atomic_ids") or [])
    atomic_id = atomic.get("atomic_id")
    if atomic_id and atomic_id not in examples:
        examples.append(atomic_id)
    target["example_atomic_ids"] = examples[:50]
    extend_unique(target, "risky_behavior", atomic.get("risky_behavior") or [], limit=6)
    extend_unique(target, "diagnostic_signals", atomic.get("diagnostic_signals") or [], limit=6)
    extend_unique(
        target,
        "repair_principles",
        [atomic.get("repair_principle")] if atomic.get("repair_principle") else [],
        limit=3,
    )
    extend_unique(target, "exceptions", atomic.get("exceptions") or [], limit=3)
    extend_unique(target, "inclusion_criteria", atomic.get("inclusion_criteria") or [], limit=6)
    extend_unique(target, "exclusion_criteria", atomic.get("exclusion_criteria") or [], limit=6)


def maybe_promote_type(taxonomy: Dict[str, Any], proposed: Dict[str, Any], threshold: int) -> Dict[str, Any]:
    if int(proposed.get("support_count", 0) or 0) < max(1, threshold):
        return {}
    active_types = taxonomy.setdefault("active_types", [])
    proposed_types = taxonomy.setdefault("proposed_types", [])
    active_id = clean_type_id(str(proposed.get("id") or f"{proposed.get('family')}.{proposed.get('type')}"))
    existing = next((item for item in active_types if item.get("id") == active_id), None)
    if existing:
        merge_type_support(existing, proposed)
        proposed_types[:] = [item for item in proposed_types if item.get("proposal_id") != proposed.get("proposal_id")]
        return existing
    active = dict(proposed)
    active.pop("proposal_id", None)
    active["id"] = active_id
    active["status"] = "active"
    active_types.append(active)
    proposed_types[:] = [item for item in proposed_types if item.get("proposal_id") != proposed.get("proposal_id")]
    return active


def merge_type_support(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["support_count"] = int(target.get("support_count", 0) or 0) + int(source.get("support_count", 0) or 0)
    for field in (
        "supporting_question_ids",
        "example_atomic_ids",
        "risky_behavior",
        "diagnostic_signals",
        "repair_principles",
        "exceptions",
        "inclusion_criteria",
        "exclusion_criteria",
    ):
        extend_unique(target, field, source.get(field) or [], limit=compact_field_limit(field))


def build_general_mistake_set(taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    mistakes = []
    for item in taxonomy.get("active_types", []):
        if not isinstance(item, dict):
            continue
        mistakes.append(
            {
                "id": item.get("id"),
                "family": item.get("family"),
                "name": item.get("name"),
                "error_reason": compact_error_reason(item),
                "typical_error_shape": compact_typical_error_shape(item),
                "support_count": item.get("support_count", 0),
            }
        )
    mistakes.sort(key=lambda item: (str(item.get("family")), -int(item.get("support_count") or 0), str(item.get("id"))))
    return sanitize_output_obj(
        {
            "version": 1,
            "families": list((taxonomy.get("families") or SEED_FAMILIES).keys()),
            "mistakes": mistakes,
        }
    )


def compact_error_reason(item: Dict[str, Any]) -> str:
    candidates = [
        item.get("definition"),
        first_compact_line(item.get("risky_behavior", [])),
        first_compact_line(item.get("diagnostic_signals", [])),
    ]
    for candidate in candidates:
        text = compact_text(candidate)
        if text:
            return text
    return "Reusable SQL reasoning mistake."


def compact_typical_error_shape(item: Dict[str, Any]) -> str:
    candidates = [
        first_compact_line(item.get("risky_behavior", [])),
        first_compact_line(item.get("diagnostic_signals", [])),
        first_compact_line(item.get("repair_principles", [])),
    ]
    for candidate in candidates:
        text = compact_text(candidate)
        if text:
            return text
    return "The SQL shape does not match the requested answer semantics."


def load_taxonomy(path: Path) -> Dict[str, Any]:
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            loaded.setdefault("families", dict(SEED_FAMILIES))
            loaded.setdefault("active_types", [])
            loaded.setdefault("proposed_types", [])
            loaded.setdefault("next_proposal_idx", 1)
            return loaded
    return {
        "version": 1,
        "families": dict(SEED_FAMILIES),
        "active_types": [],
        "proposed_types": [],
        "next_proposal_idx": 1,
    }


def compact_types(items: Any, limit: int) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    sorted_items = sorted(
        [item for item in items if isinstance(item, dict)],
        key=lambda item: (-int(item.get("support_count", 0) or 0), str(item.get("id") or item.get("proposal_id") or "")),
    )
    compact = []
    for item in sorted_items[: max(0, limit)]:
        compact.append(
            {
                "id": item.get("id"),
                "proposal_id": item.get("proposal_id"),
                "family": item.get("family"),
                "type": item.get("type"),
                "definition": item.get("definition"),
                "inclusion_criteria": item.get("inclusion_criteria", [])[:5],
                "exclusion_criteria": item.get("exclusion_criteria", [])[:5],
                "support_count": item.get("support_count", 0),
            }
        )
    return compact


def find_type_by_family_name(items: List[Dict[str, Any]], family: str, mistake_type: str) -> Optional[Dict[str, Any]]:
    for item in items:
        if item.get("family") == family and item.get("type") == mistake_type:
            return item
    return None


def extend_unique(target: Dict[str, Any], field: str, values: List[Any], limit: int) -> None:
    cur = list(target.get(field) or [])
    for value in values:
        if value in (None, ""):
            continue
        if value not in cur:
            cur.append(value)
    target[field] = cur[:limit]


def compact_field_limit(field: str) -> int:
    if field in {"supporting_question_ids", "example_atomic_ids"}:
        return 50
    if field in {"repair_principles", "exceptions"}:
        return 3
    return 6


def first_compact_line(values: Any) -> str:
    lines = compact_lines(values, limit=1)
    return lines[0] if lines else ""


def compact_lines(values: Any, *, limit: int) -> List[str]:
    if not isinstance(values, list):
        values = [values] if values else []
    result = []
    for value in values:
        if is_too_specific_runtime_line(value):
            continue
        text = compact_text(value)
        if not text or text in result:
            continue
        result.append(text)
        if len(result) >= limit:
            break
    return result


def compact_text(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = strip_parenthetical_examples(text)
    text = re.sub(r"`[^`]+`", "schema field", text)
    text = re.sub(r"\s+", " ", text).strip()
    return shorten_sentence(text, max_words=28)


def is_too_specific_runtime_line(value: Any) -> bool:
    text = str(value or "")
    lowered = text.lower()
    return (
        "`" in text
        or "not the case here" in lowered
        or "e.g." in lowered
        or "for example" in lowered
    )


def strip_parenthetical_examples(text: str) -> str:
    text = re.sub(r"\s*\([^)]*(?:e\.g\.|for example)[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\([^)]{0,80}(?:`[^`]+`)[^)]{0,80}\)", "", text)
    return text.strip()


def shorten_sentence(text: str, *, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + "..."
