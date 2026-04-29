from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from constants import DEFAULT_FAMILY, FAMILY_ALIASES, SEED_FAMILIES
from text_utils import (
    clean_id,
    clean_text,
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
        family = normalize_family(raw.get("proposed_family"))
        mistake_type = clean_id(str(raw.get("proposed_type") or "unknown_mistake_type"))
        error = (
            raw.get("error")
            or raw.get("observed_failure")
            or raw.get("type_definition")
            or "Reusable SQL reasoning mistake."
        )
        typical_error_sql_shape = (
            raw.get("typical_error_sql_shape")
            or first_compact_line(raw.get("risky_behavior", []))
            or first_compact_line(raw.get("diagnostic_signals", []))
        )
        ideal_sql_shape = (
            raw.get("ideal_sql_shape")
            or raw.get("repair_principle")
        )
        item = {
            "atomic_id": f"AM-{safe_int(record.get('question_id'), -1)}-{idx}",
            "source_question_id": safe_int(record.get("question_id"), -1),
            "db_id": str(record.get("db_id") or ""),
            "difficulty": str(record.get("difficulty") or ""),
            "error": compact_text(error),
            "typical_error_sql_shape": compact_pattern_or_text(typical_error_sql_shape),
            "ideal_sql_shape": compact_pattern_or_text(ideal_sql_shape),
            "proposed_family": family,
            "proposed_type": mistake_type,
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
    sample_idx = current_sample_idx(taxonomy)

    active_by_id = {item.get("id"): item for item in active_types if isinstance(item, dict)}
    proposed_by_id = {item.get("proposal_id"): item for item in proposed_types if isinstance(item, dict)}

    if decision == "ATTACH_ACTIVE" and routing.get("active_type_id") in active_by_id:
        target = active_by_id[routing["active_type_id"]]
        add_support(target, atomic, sample_idx=sample_idx)
        return {"decision": "attached_active", "active_type_id": target["id"], "promoted": False}

    if decision == "VOTE_PROPOSED" and routing.get("proposal_id") in proposed_by_id:
        target = proposed_by_id[routing["proposal_id"]]
        add_support(target, atomic, sample_idx=sample_idx)
        promoted = maybe_promote_type(taxonomy, target, promotion_threshold)
        return {
            "decision": "voted_proposed",
            "proposal_id": target["proposal_id"],
            "active_type_id": promoted.get("id", ""),
            "promoted": bool(promoted),
        }

    duplicate_active = find_type_by_family_name(active_types, atomic["proposed_family"], atomic["proposed_type"])
    if duplicate_active:
        add_support(duplicate_active, atomic, sample_idx=sample_idx)
        return {"decision": "attached_active_duplicate_name", "active_type_id": duplicate_active["id"], "promoted": False}

    duplicate_proposed = find_type_by_family_name(proposed_types, atomic["proposed_family"], atomic["proposed_type"])
    if duplicate_proposed:
        add_support(duplicate_proposed, atomic, sample_idx=sample_idx)
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
        add_support(proposal, atomic, sample_idx=sample_idx)
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
        "error": compact_error_reason_from_atomic(atomic),
        "typical_error_sql_shape": compact_typical_error_sql_shape_from_atomic(atomic),
        "ideal_sql_shape": compact_ideal_sql_shape_from_atomic(atomic),
        "support_count": 0,
        "created_sample_idx": current_sample_idx(taxonomy),
        "last_vote_sample_idx": current_sample_idx(taxonomy),
        "status": "proposed",
    }


def add_support(target: Dict[str, Any], atomic: Dict[str, Any], *, sample_idx: int) -> None:
    target["support_count"] = int(target.get("support_count", 0) or 0) + 1
    target["error"] = target.get("error") or compact_error_reason_from_atomic(atomic)
    target["typical_error_sql_shape"] = target.get("typical_error_sql_shape") or compact_typical_error_sql_shape_from_atomic(atomic)
    target["ideal_sql_shape"] = target.get("ideal_sql_shape") or compact_ideal_sql_shape_from_atomic(atomic)
    if target.get("status") == "proposed":
        target.setdefault("created_sample_idx", sample_idx)
        target["last_vote_sample_idx"] = sample_idx


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
    active.pop("created_sample_idx", None)
    active.pop("last_vote_sample_idx", None)
    active["id"] = active_id
    active["status"] = "active"
    active_types.append(active)
    proposed_types[:] = [item for item in proposed_types if item.get("proposal_id") != proposed.get("proposal_id")]
    return active


def merge_type_support(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["support_count"] = int(target.get("support_count", 0) or 0) + int(source.get("support_count", 0) or 0)
    for field in ("error", "typical_error_sql_shape", "ideal_sql_shape"):
        target[field] = target.get(field) or source.get(field) or ""


def advance_taxonomy_sample(taxonomy: Dict[str, Any]) -> int:
    sample_idx = current_sample_idx(taxonomy) + 1
    taxonomy["processed_sample_count"] = sample_idx
    return sample_idx


def current_sample_idx(taxonomy: Dict[str, Any]) -> int:
    return int(taxonomy.get("processed_sample_count", 0) or 0)


def prune_stale_proposals(taxonomy: Dict[str, Any], *, stale_after: int) -> List[Dict[str, Any]]:
    if stale_after <= 0:
        return []
    sample_idx = current_sample_idx(taxonomy)
    proposed_types = taxonomy.setdefault("proposed_types", [])
    kept = []
    discarded = []
    for item in proposed_types:
        if not isinstance(item, dict):
            continue
        last_vote_idx = int(item.get("last_vote_sample_idx") or item.get("created_sample_idx") or sample_idx)
        age = sample_idx - last_vote_idx
        if age >= stale_after:
            discarded.append(
                {
                    "proposal_id": item.get("proposal_id", ""),
                    "id": item.get("id", ""),
                    "age": age,
                    "support_count": int(item.get("support_count", 0) or 0),
                }
            )
        else:
            kept.append(item)
    proposed_types[:] = kept
    return discarded


def drop_proposals_by_ids(taxonomy: Dict[str, Any], drop_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not drop_items:
        return []
    reasons = {
        str(item.get("proposal_id") or ""): clean_text(item.get("reason"))
        for item in drop_items
        if isinstance(item, dict)
    }
    drop_ids = {proposal_id for proposal_id in reasons if proposal_id}
    if not drop_ids:
        return []

    proposed_types = taxonomy.setdefault("proposed_types", [])
    kept = []
    discarded = []
    for item in proposed_types:
        if not isinstance(item, dict):
            continue
        proposal_id = str(item.get("proposal_id") or "")
        if proposal_id in drop_ids:
            discarded.append(
                {
                    "proposal_id": proposal_id,
                    "id": item.get("id", ""),
                    "support_count": int(item.get("support_count", 0) or 0),
                    "reason": reasons.get(proposal_id, ""),
                }
            )
        else:
            kept.append(item)
    proposed_types[:] = kept
    return discarded


def compact_proposed_capacity_candidates(taxonomy: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    proposed = [item for item in taxonomy.get("proposed_types", []) if isinstance(item, dict)]
    proposed.sort(
        key=lambda item: (
            int(item.get("support_count", 0) or 0),
            int(item.get("last_vote_sample_idx", 0) or 0),
            str(item.get("proposal_id") or ""),
        )
    )
    result = []
    for item in proposed[: max(0, limit)]:
        result.append(
            {
                "proposal_id": item.get("proposal_id"),
                "id": item.get("id"),
                "family": item.get("family"),
                "type": item.get("type"),
                "error": compact_error_reason(item),
                "typical_error_sql_shape": compact_typical_error_sql_shape(item),
                "ideal_sql_shape": compact_ideal_sql_shape(item),
                "support_count": int(item.get("support_count", 0) or 0),
                "created_sample_idx": int(item.get("created_sample_idx", 0) or 0),
                "last_vote_sample_idx": int(item.get("last_vote_sample_idx", 0) or 0),
            }
        )
    return result


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
                "error": compact_error_reason(item),
                "typical_error_sql_shape": compact_typical_error_sql_shape(item),
                "ideal_sql_shape": compact_ideal_sql_shape(item),
                "support_count": item.get("support_count", 0),
            }
        )
    mistakes.sort(key=lambda item: (str(item.get("family")), -int(item.get("support_count") or 0), str(item.get("id"))))
    return sanitize_output_obj(
        {
            "version": 1,
            "families": list(SEED_FAMILIES.keys()),
            "mistakes": mistakes,
        }
    )


def compact_error_reason(item: Dict[str, Any]) -> str:
    if item.get("error"):
        return compact_text(item.get("error"))
    if item.get("error_reason"):
        return compact_text(item.get("error_reason"))
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


def compact_error_reason_from_atomic(atomic: Dict[str, Any]) -> str:
    return compact_text(
        atomic.get("error")
        or atomic.get("type_definition")
        or atomic.get("observed_failure")
        or first_compact_line(atomic.get("risky_behavior", []))
        or "Reusable SQL reasoning mistake."
    )


def compact_typical_error_sql_shape(item: Dict[str, Any]) -> str:
    if item.get("typical_error_sql_shape"):
        return compact_pattern_or_text(item.get("typical_error_sql_shape"))
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


def compact_ideal_sql_shape(item: Dict[str, Any]) -> str:
    if item.get("ideal_sql_shape"):
        return compact_pattern_or_text(item.get("ideal_sql_shape"))
    return first_compact_line(item.get("repair_principles", []))


def compact_typical_error_sql_shape_from_atomic(atomic: Dict[str, Any]) -> str:
    return (
        compact_pattern_or_text(atomic.get("typical_error_sql_shape"))
        or compact_text(
            first_compact_line(atomic.get("risky_behavior", []))
            or first_compact_line(atomic.get("diagnostic_signals", []))
            or "The SQL shape does not match the requested answer semantics."
        )
    )


def compact_ideal_sql_shape_from_atomic(atomic: Dict[str, Any]) -> str:
    return (
        compact_pattern_or_text(atomic.get("ideal_sql_shape"))
        or compact_pattern_or_text(atomic.get("repair_principle"))
    )


def load_taxonomy(path: Path) -> Dict[str, Any]:
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            loaded["families"] = dict(SEED_FAMILIES)
            loaded.setdefault("active_types", [])
            loaded.setdefault("proposed_types", [])
            loaded.setdefault("next_proposal_idx", 1)
            loaded.setdefault("processed_sample_count", 0)
            loaded["active_types"] = [minimal_type_record(item, proposed=False) for item in loaded["active_types"]]
            loaded["proposed_types"] = [minimal_type_record(item, proposed=True) for item in loaded["proposed_types"]]
            sample_idx = current_sample_idx(loaded)
            for item in loaded["proposed_types"]:
                item["created_sample_idx"] = int(item.get("created_sample_idx") or sample_idx)
                item["last_vote_sample_idx"] = int(item.get("last_vote_sample_idx") or item["created_sample_idx"])
            return loaded
    return {
        "version": 1,
        "families": dict(SEED_FAMILIES),
        "active_types": [],
        "proposed_types": [],
        "next_proposal_idx": 1,
        "processed_sample_count": 0,
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
                "error": compact_error_reason(item),
                "typical_error_sql_shape": compact_typical_error_sql_shape(item),
                "ideal_sql_shape": compact_ideal_sql_shape(item),
                "support_count": item.get("support_count", 0),
            }
        )
    return compact


def find_type_by_family_name(items: List[Dict[str, Any]], family: str, mistake_type: str) -> Optional[Dict[str, Any]]:
    for item in items:
        if item.get("family") == family and item.get("type") == mistake_type:
            return item
    return None


def minimal_type_record(item: Dict[str, Any], *, proposed: bool) -> Dict[str, Any]:
    family = normalize_family(item.get("family"))
    mistake_type = str(item.get("type") or str(item.get("id") or "unknown").split(".")[-1])
    record = {
        "family": family,
        "type": mistake_type,
        "id": f"{family}.{mistake_type}",
        "name": item.get("name") or humanize_type_name(mistake_type),
        "error": compact_error_reason(item),
        "typical_error_sql_shape": compact_typical_error_sql_shape(item),
        "ideal_sql_shape": compact_ideal_sql_shape(item),
        "support_count": int(item.get("support_count", 0) or 0),
        "status": "proposed" if proposed else "active",
    }
    if proposed:
        record["proposal_id"] = item.get("proposal_id") or ""
        record["created_sample_idx"] = int(item.get("created_sample_idx") or 0)
        record["last_vote_sample_idx"] = int(item.get("last_vote_sample_idx") or record["created_sample_idx"])
    return record


def normalize_family(value: Any) -> str:
    family = clean_id(str(value or DEFAULT_FAMILY))
    family = FAMILY_ALIASES.get(family, family)
    if family not in SEED_FAMILIES:
        return DEFAULT_FAMILY
    return family


def compact_pattern_or_text(value: Any) -> str:
    if isinstance(value, list):
        return first_compact_line(value)
    return compact_text(value)


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
