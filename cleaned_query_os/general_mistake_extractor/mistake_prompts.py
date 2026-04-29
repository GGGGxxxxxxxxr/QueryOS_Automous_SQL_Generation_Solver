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
from taxonomy import compact_types
from text_utils import sanitize_input_text, truncate


def build_system_prompt() -> str:
    return """You are building a real-world SQL-agent mistake memory.

The input is a failed SQL-agent run record. It may contain evaluator-only fields.
Use all available information only to infer reusable SQL-agent mistake patterns.

Hard constraints:
- Do not mention or imply evaluator-only answer artifacts, hidden comparison SQL, or answer-key language in the output.
- Do not use evaluator-specific words anywhere in any output field.
- Do not produce database-specific lessons.
- Abstract concrete table/column names into general SQL reasoning patterns.
- Concrete SQL snippets may appear only as examples of risky SQL behavior.
- Prefer existing active or proposed mistake types when they fit.
- Create a new proposed type only when neither active nor proposed types fit.
- Before extracting mistakes, decide whether this failed-run record is reliable enough to learn from.
- If the offline comparison behavior is clearly inconsistent with the natural-language question/evidence, mark the trace as skipped and do not extract atomic mistakes.
- Skip only when the comparison behavior is clearly wrong or self-contradictory, not merely because the failed SQL looks plausible.

Return one JSON object with this exact shape:
{
  "skip_failure_trace": {
    "skip": false,
    "reason": "why this record should or should not update the general mistake taxonomy"
  },
  "atomic_mistakes": [
    {
      "observed_failure": "short real-world failure description",
      "risky_behavior": ["general risky SQL behavior"],
      "diagnostic_signals": ["how a SQL agent can notice this risk without any hidden answer"],
      "repair_principle": "general repair rule",
      "exceptions": ["anti-overgeneralization / when not to apply"],
      "risky_sql_shapes": ["optional short SQL shape examples"],
      "proposed_family": "one top-level family id",
      "proposed_type": "snake_case general type name",
      "type_definition": "one-sentence reusable type definition",
      "inclusion_criteria": ["what belongs in this type"],
      "exclusion_criteria": ["what should not be classified here"],
      "routing": {
        "decision": "ATTACH_ACTIVE | VOTE_PROPOSED | PROPOSE_NEW",
        "active_type_id": "required only for ATTACH_ACTIVE",
        "proposal_id": "required only for VOTE_PROPOSED",
        "confidence": 0.0,
        "reason": "why this routing is correct"
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


def build_user_prompt(
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
        "Current mistake taxonomy state:\n"
        f"{json.dumps(taxonomy_view, ensure_ascii=False, indent=2)}\n\n"
        "New failed-run record for offline analysis:\n"
        f"{record_text}\n\n"
        "First decide whether this record is safe to learn from. If the offline comparison "
        "behavior is clearly inconsistent with the user question or evidence, set "
        "skip_failure_trace.skip=true and return no atomic mistakes. Otherwise extract "
        "reusable general mistake patterns and route each one to an active type, an "
        "existing proposed type, or a new proposed type. Output JSON only."
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
