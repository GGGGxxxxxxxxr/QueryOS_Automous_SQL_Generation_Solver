from __future__ import annotations

import json
from typing import Any, Dict

from constants import (
    EVAL_COLUMNS_KEY,
    EVAL_COMPARISON_KEY,
    EVAL_ROW_COUNT_KEY,
    EVAL_ROWS_KEY,
    EVAL_SQL_KEY,
    SEED_FAMILIES,
)
from taxonomy import compact_proposed_capacity_candidates, compact_types, normalize_pattern_tuples
from text_utils import sanitize_input_text, truncate


def build_routing_system_prompt() -> str:
    return COMMON_SYSTEM_PROMPT + """

This is stage 1: route to an existing mistake type if possible.

Mandatory stage-1 rules:
- You must inspect the displayed active_types and proposed_types before deciding.
- You are not allowed to invent a new type name in this stage.
- If an active type fits, output ATTACH_ACTIVE.
- Else if a proposed type fits, output VOTE_PROPOSED.
- Only if neither fits, output NEED_NEW_TYPE.
- NEED_NEW_TYPE is only a gate into stage 2. It is not a proposal.
- Prefer VOTE_PROPOSED for near-matches with the same root cause and correction pattern.

Return one JSON object with this exact shape:
{
  "skip_failure_trace": {
    "skip": false,
    "reason": "why this record should or should not update the general mistake taxonomy"
  },
  "routing_decisions": [
    {
      "error": "one short reusable error reason",
      "typical_error_sql_shape": "abstract risky SQL skeleton or short behavior shape",
      "ideal_sql_shape": "abstract corrected SQL skeleton or short ideal rule",
      "routing": {
        "decision": "ATTACH_ACTIVE | VOTE_PROPOSED | NEED_NEW_TYPE",
        "active_type_id": "required only for ATTACH_ACTIVE",
        "proposal_id": "required only for VOTE_PROPOSED",
        "confidence": 0.0,
        "reason": "why this routing is correct"
      }
    }
  ]
}
"""


def build_proposal_system_prompt() -> str:
    return COMMON_SYSTEM_PROMPT + """

This is stage 2: propose new mistake types only for mistakes that stage 1 marked NEED_NEW_TYPE.

Mandatory stage-2 rules:
- Propose a new type only for each provided unmatched mistake.
- Do not route to existing active or proposed types in this stage.
- Keep the proposed type compact and reusable.
- The new type must not be a near-duplicate of the active/proposed taxonomy shown in stage 1.

Return one JSON object with this exact shape:
{
  "atomic_mistakes": [
    {
      "error": "one short reusable error reason",
      "typical_error_sql_shape": "abstract risky SQL skeleton or short behavior shape",
      "ideal_sql_shape": "abstract corrected SQL skeleton or short ideal rule",
      "proposed_family": "one top-level family id",
      "proposed_type": "snake_case general type name",
      "routing": {
        "decision": "PROPOSE_NEW",
        "active_type_id": "",
        "proposal_id": "",
        "confidence": 0.0,
        "reason": "why a new type is necessary"
      }
    }
  ]
}

Type naming rules:
- snake_case
- 2 to 5 words
- no database names
- no table or column names
- no evaluator-specific language
- describe the failure mode, not a single sample
"""


def build_capacity_prune_system_prompt() -> str:
    return COMMON_SYSTEM_PROMPT + """

This is a proposed-type capacity cleanup stage.

The proposed pool is larger than the configured capacity. Your job is to identify unnecessary proposed types to drop.

Drop proposed types when they are:
- near-duplicates of an active type
- near-duplicates of another proposed type
- too sample-specific to be reusable
- vague, low-quality, or missing a useful error/typical_error_sql_shape/ideal_sql_shape
- low support and unlikely to become a stable general mistake

Prefer keeping proposed types with higher support, clearer SQL skeletons, and distinct correction patterns.
Do not drop active types.
Return enough drop candidates to reduce the proposed pool toward the configured capacity.

Return one JSON object with this exact shape:
{
  "drop_proposals": [
    {
      "proposal_id": "proposal id to remove",
      "reason": "short reason"
    }
  ]
}
"""


def build_tuple_dedupe_system_prompt() -> str:
    return COMMON_SYSTEM_PROMPT + """

This is the final intra-type pattern deduplication stage.

You are given one already-established general mistake type and many supporting pattern tuples under it.
Each tuple has:
- error
- typical_error_sql_shape
- ideal_sql_shape

Your job:
- Merge near-duplicate tuples within this type.
- Keep distinct sub-patterns when they represent materially different SQL failure shapes or repairs.
- Preserve the type boundary; do not rename the type or create new types.
- Return a compact list of representative tuple patterns.
- The tuple must remain general. Do not include dataset details.
- Prefer abstract SQL skeletons for both SQL shape fields.

Return one JSON object with this exact shape:
{
  "patterns": [
    {
      "error": "short general error reason",
      "typical_error_sql_shape": "abstract wrong SQL shape",
      "ideal_sql_shape": "abstract ideal SQL shape",
      "support_count": 1
    }
  ]
}
"""


COMMON_SYSTEM_PROMPT = """You are building a real-world SQL-agent mistake memory.

The input is a failed SQL-agent run record. It may contain evaluator-only fields.
Use all available information only to infer reusable SQL-agent mistake patterns.

Hard constraints:
- Do not mention or imply evaluator-only answer artifacts, hidden comparison SQL, or answer-key language in the output.
- Do not use evaluator-specific words anywhere in any output field.
- Do not produce database-specific lessons.
- Abstract concrete table/column names into general SQL reasoning patterns.
- Concrete SQL snippets may appear only as abstract SQL skeletons.
- Before extracting mistakes, decide whether this failed-run record is reliable enough to learn from.
- If the offline comparison behavior is clearly inconsistent with the natural-language question/evidence, mark the trace as skipped and do not extract atomic mistakes.
- Skip only when the comparison behavior is clearly wrong or self-contradictory, not merely because the failed SQL looks plausible.
- Keep reusable rule fields compact and abstract.
- Do not put concrete database names, table names, column names, place names, organization names, or literal values in error, typical_error_sql_shape, or ideal_sql_shape.
- Do not include parenthetical examples in reusable rule fields.
- Prefer one precise mistake type over broad buckets that mix unrelated causes.
- Prefer an existing near-match over creating a slightly more precise duplicate.
- Focus on three reusable facts: error reason, typical error SQL pattern, and ideal SQL pattern.
- Keep all fields minimal. Do not produce long criteria lists.
- When possible, express typical error shape as an abstract SQL skeleton.
- Generalize the mistake mechanism, not the dataset detail.
- Prefer "missing required WHERE predicate before aggregation" over "forgot to filter a specific school type".
- Prefer "wrong grouping level" over "grouped by a specific district column".
- Prefer "extra output column" over naming the concrete extra column.
- Use placeholders such as <table>, <column>, <condition>, <group_key>, <metric>, <date_col>, <value>.
- Good typical_error_sql_shape examples:
  - SELECT <group_key>, COUNT(*) FROM <table> WHERE <condition> GROUP BY <group_key>
  - WHERE DATE(<date_col>) = DATE(<literal>) when direct comparison is enough
  - SELECT <id>, <col1> UNION SELECT <id>, <col2> when paired columns should stay together
- Good ideal_sql_shape examples:
  - SELECT <group_key>, COUNT(*) FROM <table> WHERE <required_subset_filter> GROUP BY <group_key>
  - WHERE <date_col> = <date_literal>
  - SELECT <id>, <col1>, <col2> FROM <table>
- Do not use concrete table names, column names, literal values, or database-specific entities in SQL shapes.
"""


def build_routing_user_prompt(
    *,
    record: Dict[str, Any],
    taxonomy: Dict[str, Any],
    active_preview_limit: int,
    proposed_preview_limit: int,
    record_max_chars: int,
) -> str:
    safe_record = build_safe_record_for_llm(record)
    record_text = truncate(json.dumps(safe_record, ensure_ascii=False, indent=2), record_max_chars)
    taxonomy_view = {
        "families": taxonomy.get("families", SEED_FAMILIES),
        "active_types": compact_types(taxonomy.get("active_types", []), active_preview_limit),
        "proposed_types": compact_types(taxonomy.get("proposed_types", []), proposed_preview_limit),
    }
    return (
        "Current active and proposed mistake taxonomy state:\n"
        f"{json.dumps(taxonomy_view, ensure_ascii=False, indent=2)}\n\n"
        "New failed-run record for offline analysis:\n"
        f"{record_text}\n\n"
        "First decide whether this record is safe to learn from. If the offline comparison "
        "behavior is clearly inconsistent with the user question or evidence, set "
        "skip_failure_trace.skip=true and return no routing decisions. Otherwise display "
        "your decision through routing_decisions only. You must not propose any new type "
        "in this stage. If no active/proposed type fits, use NEED_NEW_TYPE. Output JSON only."
    )


def build_proposal_user_prompt(
    *,
    record: Dict[str, Any],
    taxonomy: Dict[str, Any],
    unmatched_decisions: Any,
    active_preview_limit: int,
    proposed_preview_limit: int,
    record_max_chars: int,
) -> str:
    safe_record = build_safe_record_for_llm(record)
    record_text = truncate(json.dumps(safe_record, ensure_ascii=False, indent=2), record_max_chars)
    taxonomy_view = {
        "families": taxonomy.get("families", SEED_FAMILIES),
        "active_types": compact_types(taxonomy.get("active_types", []), active_preview_limit),
        "proposed_types": compact_types(taxonomy.get("proposed_types", []), proposed_preview_limit),
    }
    unmatched_text = json.dumps(unmatched_decisions, ensure_ascii=False, indent=2)
    return (
        "Current active and proposed mistake taxonomy state from stage 1:\n"
        f"{json.dumps(taxonomy_view, ensure_ascii=False, indent=2)}\n\n"
        "Failed-run record for offline analysis:\n"
        f"{record_text}\n\n"
        "Stage-1 unmatched mistakes that require a new type:\n"
        f"{unmatched_text}\n\n"
        "Now propose new reusable mistake types only for these unmatched mistakes. "
        "Do not create near-duplicates of active_types or proposed_types. Output JSON only."
    )


def build_capacity_prune_user_prompt(
    *,
    taxonomy: Dict[str, Any],
    max_proposed_types: int,
    target_drop_count: int,
    review_limit: int,
) -> str:
    taxonomy_view = {
        "families": taxonomy.get("families", SEED_FAMILIES),
        "active_types": compact_types(taxonomy.get("active_types", []), 80),
        "proposed_drop_candidates": compact_proposed_capacity_candidates(taxonomy, review_limit),
        "current_proposed_count": len(taxonomy.get("proposed_types", [])),
        "max_proposed_types": max_proposed_types,
        "target_drop_count": target_drop_count,
    }
    return (
        "Current active types and proposed drop candidates:\n"
        f"{json.dumps(taxonomy_view, ensure_ascii=False, indent=2)}\n\n"
        "Choose only proposed IDs from proposed_drop_candidates. "
        "Prefer dropping near-duplicates, vague entries, overly narrow entries, and low-support stale entries. "
        "Output JSON only."
    )


def build_tuple_dedupe_user_prompt(
    *,
    type_item: Dict[str, Any],
    review_limit: int,
    max_patterns: int,
) -> str:
    raw_patterns = normalize_pattern_tuples(type_item.get("pattern_tuples"))
    if review_limit > 0:
        raw_patterns = raw_patterns[:review_limit]
    type_view = {
        "id": type_item.get("id"),
        "family": type_item.get("family"),
        "name": type_item.get("name"),
        "support_count": type_item.get("support_count", 0),
        "max_output_patterns": max_patterns,
        "pattern_tuples": raw_patterns,
    }
    return (
        "Deduplicate pattern tuples for this one mistake type:\n"
        f"{json.dumps(type_view, ensure_ascii=False, indent=2)}\n\n"
        "Return at most max_output_patterns representative tuple patterns. "
        "Merge only truly redundant tuples; keep different wrong SQL shapes separate. "
        "Use abstract placeholders such as <table>, <column>, <condition>, <metric>, <group_key>, <date_col>. "
        "Output JSON only."
    )


def build_safe_record_for_llm(record: Dict[str, Any]) -> Dict[str, Any]:
    comparison = record.get(EVAL_COMPARISON_KEY) if isinstance(record.get(EVAL_COMPARISON_KEY), dict) else {}
    return {
        "source_question_id": record.get("question_id"),
        "db_id": record.get("db_id"),
        "difficulty": record.get("difficulty"),
        "question": record.get("question"),
        "evidence": record.get("evidence"),
        "failed_sql": record.get("predicted_sql"),
        "comparison_sql_for_offline_analysis": record.get(EVAL_SQL_KEY),
        "failed_columns": record.get("predicted_columns"),
        "comparison_columns_for_offline_analysis": record.get(EVAL_COLUMNS_KEY),
        "failed_rows_preview": record.get("predicted_rows_preview"),
        "comparison_rows_preview_for_offline_analysis": record.get(EVAL_ROWS_KEY),
        "row_count_signal": {
            "failed_row_count": comparison.get("predicted_row_count"),
            "comparison_row_count": comparison.get(EVAL_ROW_COUNT_KEY),
        },
        "failure_reason": sanitize_input_text(str(record.get("error_reason") or "")),
    }
