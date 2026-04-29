# General Mistake Extractor

This is an isolated offline system for mining reusable, cross-database SQL
mistake patterns from QueryOS failure-memory records.

It does not modify the QueryOS runtime pipeline. It only reads an error memory
bank JSONL file and writes general mistake artifacts to an output folder.

## Goal

Build a self-expanding general mistake set:

```text
error_bank.jsonl
   |
   v
stage 1: route against active/proposed mistake types
   |
   v
stage 2: propose a new type only for unmatched mistakes
   |
   v
active/proposed type pool
   |
   v
compact general mistake set
```

The extractor avoids database-level lessons. It should not produce rules like
"in this database, charter fields are in table X." Instead, it abstracts them
into general patterns such as "wrong source for semantic concept."

The extractor also avoids any benchmark-specific language in its outputs. The
final artifacts should be usable in real deployments where no comparison SQL is
available.

Before extracting mistake patterns, the extractor asks the model to judge
whether the failed-run record is reliable enough to learn from. If the offline
comparison behavior is clearly inconsistent with the user question or evidence,
the trace is ignored and does not update the taxonomy.

## Run

From the repository root:

```bash
python cleaned_query_os/general_mistake_extractor/build_general_mistakes.py \
  --error-bank dev_20240627-2/failure_memory/error_bank.jsonl \
  --out dev_20240627-2/general_mistakes \
  --config cleaned_query_os/queryos_vllm_config.yaml \
  --provider vllm \
  --limit 20 \
  --proposal-stale-after 50 \
  --max-proposed-types 100
```

OpenAI example:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

python cleaned_query_os/general_mistake_extractor/build_general_mistakes.py \
  --error-bank dev_20240627-2/failure_memory/error_bank.jsonl \
  --out dev_20240627-2/general_mistakes \
  --config cleaned_query_os/queryos_config.yaml
```

## Outputs

```text
general_mistakes/
  atomic_mistakes.jsonl
  taxonomy.json
  general_mistake_set.json
  ignored_traces.jsonl
  run_log.jsonl
```

`taxonomy.json` contains:

- seed top-level families
- active mistake types
- proposed mistake types

Seed families are intentionally limited to `database_reasoning`, `sql_logic`,
and `output_format`.

Mistake type records are intentionally small:

```json
{
  "proposal_id": "PT-000001",
  "family": "sql_logic",
  "type": "aggregation_scope_mismatch",
  "id": "sql_logic.aggregation_scope_mismatch",
  "name": "Aggregation Scope Mismatch",
  "error": "The SQL computes at the wrong entity or grouping level.",
  "typical_error_sql_shape": "SELECT <column>, AVG(<column>) FROM <table> GROUP BY <column>",
  "ideal_sql_shape": "SELECT <column> FROM <table> ORDER BY <column> DESC LIMIT <number>",
  "support_count": 1,
  "created_sample_idx": 12,
  "last_vote_sample_idx": 12,
  "status": "proposed"
}
```

Each new atomic mistake can:

- attach to an active type
- vote for an existing proposed type
- propose a new type

The extractor enforces this as a two-step workflow. Stage 1 displays the current
active and proposed types and can only return `ATTACH_ACTIVE`, `VOTE_PROPOSED`,
or `NEED_NEW_TYPE`. Stage 2 runs only for `NEED_NEW_TYPE` items and is the only
place where a new proposed type can be created.

When a proposed type reaches `--promotion-threshold`, it is promoted to active.
When a proposed type receives no votes for `--proposal-stale-after` later
processed samples, it is discarded. Use `--proposal-stale-after 0` to disable
this pruning.

When the proposed pool grows beyond `--max-proposed-types`, the extractor starts
a capacity-cleanup LLM pass. That pass can only return proposed IDs to discard;
it is used to remove near-duplicates, overly narrow proposals, vague proposals,
or low-support entries that are unlikely to become stable general mistakes. Use
`--max-proposed-types 0` to disable this cleanup.

`ignored_traces.jsonl` stores records that were skipped because they appeared
unsafe for taxonomy learning.

`general_mistake_set.json` is intentionally compact. It keeps only active types
and the three runtime-facing fields:

```json
{
  "id": "sql_logic.aggregation_scope_mismatch",
  "family": "sql_logic",
  "name": "Aggregation Scope Mismatch",
  "error": "The SQL computes at the wrong entity or grouping level.",
  "typical_error_sql_shape": "SELECT <group_key>, AVG(<metric>) FROM <table> GROUP BY <group_key>",
  "ideal_sql_shape": "SELECT <column> FROM <table> ORDER BY <metric> DESC LIMIT <number>",
  "support_count": 3
}
```

Each mistake type is summarized as:

- `error`: what reasoning failure happens
- `typical_error_sql_shape`: the common risky SQL shape, preferably as an abstract skeleton
- `ideal_sql_shape`: the ideal SQL shape or repair pattern, also abstract when possible

Detailed per-sample evidence stays in `atomic_mistakes.jsonl`; `taxonomy.json`
and `general_mistake_set.json` stay compact.
