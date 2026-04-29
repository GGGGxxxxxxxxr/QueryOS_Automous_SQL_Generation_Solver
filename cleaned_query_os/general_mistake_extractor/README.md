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
atomic general mistakes
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
  --limit 20
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

Each new atomic mistake can:

- attach to an active type
- vote for an existing proposed type
- propose a new type

When a proposed type reaches `--promotion-threshold`, it is promoted to active.

`ignored_traces.jsonl` stores records that were skipped because they appeared
unsafe for taxonomy learning.
