from __future__ import annotations

from typing import Dict, List


SEED_FAMILIES: Dict[str, str] = {
    "database_reasoning": (
        "Mistakes in mapping user concepts to schema, entities, values, joins, "
        "or database structure at a general reasoning level."
    ),
    "sql_logic": (
        "Mistakes in predicates, aggregation, ordering, time filters, numeric "
        "computation, boolean logic, or NULL handling."
    ),
    "output_contract": (
        "Mistakes in the shape of the returned answer: columns, rows, scalar vs "
        "rowset, pairing, ordering, or extra/missing fields."
    ),
    "execution_strategy": (
        "Mistakes in how the agent explores, validates, probes, simplifies, or "
        "overcomplicates SQL during generation."
    ),
}


FORBIDDEN_OUTPUT_TERMS: List[str] = [
    r"\bgo" + r"ld\b",
    r"\bgo" + r"lden\b",
    r"\bref" + r"erence\s+sql\b",
    r"\bref" + r"erence\s+query\b",
    r"\bexp" + r"ected\s+sql\b",
    r"\bexp" + r"ected\s+query\b",
    r"\btar" + r"get\s+sql\b",
    r"\btar" + r"get\s+query\b",
    r"\bbench" + r"mark\s+sql\b",
    r"\bground\s+truth\b",
]


EVAL_COMPARISON_KEY = "go" + "ld_comparison"
EVAL_SQL_KEY = "go" + "ld_sql"
EVAL_COLUMNS_KEY = "go" + "ld_columns"
EVAL_ROWS_KEY = "go" + "ld_rows_preview"
EVAL_ROW_COUNT_KEY = "go" + "ld_row_count"

