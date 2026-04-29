from __future__ import annotations

from typing import Any, Dict, List

from mistake_prompts import (
    build_capacity_prune_system_prompt,
    build_capacity_prune_user_prompt,
    build_proposal_system_prompt,
    build_proposal_user_prompt,
    build_routing_system_prompt,
    build_routing_user_prompt,
)
from query_os.llm import create_chat_completion
from text_utils import parse_json_object, sanitize_output_obj


def extract_and_route_record(
    *,
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    record: Dict[str, Any],
    taxonomy: Dict[str, Any],
    active_preview_limit: int,
    proposed_preview_limit: int,
    record_max_chars: int,
) -> Dict[str, Any]:
    routing_result = call_extractor_llm(
        client=client,
        role="general_mistake_router",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=build_routing_system_prompt(),
        user_prompt=build_routing_user_prompt(
            record=record,
            taxonomy=taxonomy,
            active_preview_limit=active_preview_limit,
            proposed_preview_limit=proposed_preview_limit,
            record_max_chars=record_max_chars,
        ),
    )

    if should_skip(routing_result):
        return sanitize_output_obj(
            {
                "skip_failure_trace": routing_result.get("skip_failure_trace", {"skip": True, "reason": ""}),
                "atomic_mistakes": [],
                "routing_stage": routing_result,
                "proposal_stage": {},
            }
        )

    existing_atoms, unmatched = split_routing_decisions(routing_result, taxonomy)
    proposal_atoms: List[Dict[str, Any]] = []
    proposal_result: Dict[str, Any] = {}
    if unmatched:
        proposal_result = call_extractor_llm(
            client=client,
            role="general_mistake_proposer",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=build_proposal_system_prompt(),
            user_prompt=build_proposal_user_prompt(
                record=record,
                taxonomy=taxonomy,
                unmatched_decisions=unmatched,
                active_preview_limit=active_preview_limit,
                proposed_preview_limit=proposed_preview_limit,
                record_max_chars=record_max_chars,
            ),
        )
        proposal_atoms = normalize_proposal_atoms(proposal_result)

    return sanitize_output_obj(
        {
            "skip_failure_trace": routing_result.get("skip_failure_trace", {"skip": False, "reason": ""}),
            "atomic_mistakes": existing_atoms + proposal_atoms,
            "routing_stage": routing_result,
            "proposal_stage": proposal_result,
        }
    )


def call_extractor_llm(
    *,
    client: Any,
    role: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    response = create_chat_completion(
        client,
        role=role,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return sanitize_output_obj(parse_json_object(content))


def should_skip(result: Dict[str, Any]) -> bool:
    raw = result.get("skip_failure_trace")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, dict):
        return bool(raw.get("skip"))
    return False


def split_routing_decisions(
    result: Dict[str, Any],
    taxonomy: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_items = result.get("routing_decisions")
    if not isinstance(raw_items, list):
        raw_items = result.get("atomic_mistakes") if isinstance(result.get("atomic_mistakes"), list) else []
    active_ids = {item.get("id") for item in taxonomy.get("active_types", []) if isinstance(item, dict)}
    proposal_ids = {item.get("proposal_id") for item in taxonomy.get("proposed_types", []) if isinstance(item, dict)}

    existing_atoms: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        routing = raw.get("routing") if isinstance(raw.get("routing"), dict) else {}
        decision = str(routing.get("decision") or "").strip().upper()
        item = {
            "error": raw.get("error", ""),
            "typical_shape": raw.get("typical_shape", ""),
            "correct_pattern": raw.get("correct_pattern", ""),
            "routing": dict(routing),
        }
        if decision == "ATTACH_ACTIVE" and routing.get("active_type_id") in active_ids:
            existing_atoms.append(item)
        elif decision == "VOTE_PROPOSED" and routing.get("proposal_id") in proposal_ids:
            existing_atoms.append(item)
        else:
            item["routing"]["decision"] = "NEED_NEW_TYPE"
            unmatched.append(item)
    return existing_atoms, unmatched


def normalize_proposal_atoms(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_items = result.get("atomic_mistakes")
    if not isinstance(raw_items, list):
        return []
    atoms = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        routing = item.get("routing") if isinstance(item.get("routing"), dict) else {}
        routing = dict(routing)
        routing["decision"] = "PROPOSE_NEW"
        routing.setdefault("active_type_id", "")
        routing.setdefault("proposal_id", "")
        item["routing"] = routing
        atoms.append(item)
    return atoms


def select_unnecessary_proposals(
    *,
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    taxonomy: Dict[str, Any],
    max_proposed_types: int,
    target_drop_count: int,
    review_limit: int,
) -> List[Dict[str, Any]]:
    result = call_extractor_llm(
        client=client,
        role="general_mistake_capacity_pruner",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=build_capacity_prune_system_prompt(),
        user_prompt=build_capacity_prune_user_prompt(
            taxonomy=taxonomy,
            max_proposed_types=max_proposed_types,
            target_drop_count=target_drop_count,
            review_limit=review_limit,
        ),
    )
    return normalize_drop_proposals(result, target_drop_count=target_drop_count)


def normalize_drop_proposals(result: Dict[str, Any], *, target_drop_count: int) -> List[Dict[str, Any]]:
    raw_items = result.get("drop_proposals")
    if not isinstance(raw_items, list):
        return []
    normalized = []
    seen = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        proposal_id = str(raw.get("proposal_id") or "").strip()
        if not proposal_id or proposal_id in seen:
            continue
        seen.add(proposal_id)
        normalized.append({"proposal_id": proposal_id, "reason": str(raw.get("reason") or "").strip()})
        if len(normalized) >= max(1, target_drop_count):
            break
    return normalized
