# QueryOS Autonomous SQL Generation Solver

QueryOS is a cleaned, self-contained SQL generation agent for SQLite databases.
It is designed as a small agent operating system for text-to-SQL: a manager owns
the global workflow, specialized workers run on forked state, shared memory records
what the system has learned, and validation gates prevent the planner from
finishing too early.

The cleaned implementation lives in [`cleaned_query_os/`](cleaned_query_os/).
The archived raw implementation is preserved in
[`raw_query_os_original/`](raw_query_os_original/) for reference.

## Design Goals

QueryOS is built around a few core ideas:

- **Manager as state controller**: the planner does not just write SQL. It decides
  which worker should run next, reads shared state, and owns the final finish
  decision.
- **Workers as OS-like threads**: schema discovery and SQL writing can fork
  parallel workers. Each worker works independently, then QueryOS merges the
  useful result back into global state.
- **Shared global memory**: discovered schema, SQL attempts, validation feedback,
  planner decisions, warnings, and workflow status are visible in the trace and
  saved to JSON.
- **Validator as a gate, not a dictator**: the SQL Validator Agent gives natural
  language feedback. The planner still decides whether to retry, ask another
  worker, or finish.
- **Observable execution**: the CLI prints a readable live workflow trace, SQL
  previews, shared-state updates, validation results, and optional golden SQL
  comparisons.
- **Local-model friendly**: QueryOS supports hosted OpenAI models and separate
  OpenAI-compatible vLLM backends, including routing across multiple local model
  copies.

## Workflow

```text
Question + SQLite DB + metadata
        |
        v
Manager / Planner
        |
        +--> Schema Discovery Agent
        |       - searches metadata
        |       - reads table JSON
        |       - introduces relevant tables, columns, keys
        |       - optionally runs parallel SDA workers
        |
        +--> SQL Writer Agent
        |       - writes exploratory SQL when useful
        |       - executes read-only SQLite
        |       - optionally runs parallel writer workers
        |       - optionally resolves disagreement through writer chatgroup
        |
        +--> SQL Validator Agent
        |       - checks SQL against question, evidence, schema, and result
        |       - writes natural-language validation feedback
        |
        v
Planner finish decision
        |
        v
Result + trace JSON + optional golden SQL comparison
```

## Parallel Workers

QueryOS supports parallelism inside the worker layer. The manager still sees one
logical worker call, but that worker may internally fork multiple copies.

### Parallel Schema Discovery

When `schema_discovery.parallel_workers > 1`, QueryOS runs multiple SDA workers
on forked shared state.

The merge strategy is intentionally simple:

- Take the **union** of discovered tables, columns, primary keys, and foreign keys.
- Attach numeric `confidence` based on worker agreement.
- Keep only `confidence` in the shared schema output.

Example with two SDA workers:

```json
{
  "table": "frpm",
  "confidence": 1.0,
  "columns": [
    {"name": "CDSCode", "type": "integer", "confidence": 1.0},
    {"name": "School Type", "type": "text", "confidence": 0.5}
  ]
}
```

Interpretation:

- `confidence: 1.0` means all SDA workers selected that item.
- `confidence: 0.5` means one of two workers selected that item.
- Low confidence is not automatically wrong; it tells the writer to be cautious.

### Parallel SQL Writing

When `sql_writer.parallel_workers > 1`, QueryOS forks multiple SQL writers.
Each writer gets the same shared state snapshot and produces its own SQL attempt.

If the writers agree, the consensus SQL is committed to global state. If they
disagree, the writer chatgroup can run a bounded discussion round where each
writer either agrees with an existing SQL or revises its own SQL. Revised SQL is
executed immediately before the next round.

## Repository Layout

```text
.
├── cleaned_query_os/
│   ├── query_os/                  # QueryOS package
│   ├── queryos_config.yaml        # OpenAI-oriented config
│   ├── queryos_vllm_config.yaml   # local vLLM config template
│   ├── README.md                  # package-level usage docs
│   └── pyproject.toml
├── dev_20240627-2/
│   ├── build_failure_memory.py    # batch evaluator + failure-memory builder
│   └── build_table_description_json.py
└── raw_query_os_original/         # archived original files
```

Large benchmark assets are intentionally not committed: SQLite files, CSV files,
benchmark JSON, generated traces, and failure-memory outputs are ignored by git.

## Install

From a fresh conda environment:

```bash
conda create -n queryos python=3.11 -y
conda activate queryos

cd /path/to/QueryOS_Automous_SQL_Generation_Solver/cleaned_query_os
pip install -e .
```

After installation:

```bash
query-os --help
```

## Run With OpenAI

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

query-os \
  --config cleaned_query_os/queryos_config.yaml \
  --db dev_20240627-2/dev_databases/financial/financial.sqlite \
  --metadata dev_20240627-2/dev_databases/financial/database_description \
  --question "How many accounts who have region in Prague are eligible for loans?"
```

With golden SQL comparison:

```bash
query-os \
  --config cleaned_query_os/queryos_config.yaml \
  --db dev_20240627-2/dev_databases/financial/financial.sqlite \
  --metadata dev_20240627-2/dev_databases/financial/database_description \
  --question "How many accounts who have region in Prague are eligible for loans?" \
  --gold-sql "SELECT COUNT(T1.account_id) FROM account AS T1 INNER JOIN loan AS T2 ON T1.account_id = T2.account_id INNER JOIN district AS T3 ON T1.district_id = T3.district_id WHERE T3.A3 = 'Prague'"
```

Save trace and result JSON:

```bash
query-os \
  --config cleaned_query_os/queryos_config.yaml \
  --db dev_20240627-2/dev_databases/financial/financial.sqlite \
  --metadata dev_20240627-2/dev_databases/financial/database_description \
  --question "..." \
  --trace-json dev_20240627-2/traces/example_trace.json \
  --result-json dev_20240627-2/traces/example_result.json
```

## Run With Local vLLM

Start an OpenAI-compatible vLLM server separately, then edit
[`cleaned_query_os/queryos_vllm_config.yaml`](cleaned_query_os/queryos_vllm_config.yaml):

```yaml
provider: vllm
base_url: http://localhost:8000/v1
api_key: EMPTY
model: your-served-model-name
```

Then run:

```bash
query-os \
  --config cleaned_query_os/queryos_vllm_config.yaml \
  --provider vllm \
  --db dev_20240627-2/dev_databases/financial/financial.sqlite \
  --metadata dev_20240627-2/dev_databases/financial/database_description \
  --question "The transaction of 840 USD happened in 1998/10/14, when was this account opened?"
```

QueryOS depends heavily on OpenAI-style tool/function calling. The local vLLM
deployment must support tool calls for the selected model and chat template.

### Multiple vLLM Copies

If the same model is hosted on several vLLM servers, QueryOS can route calls
across them from the client side. OpenAI and vLLM are separate backends: use
`provider: openai` for the hosted OpenAI API and `provider: vllm` for local
OpenAI-compatible vLLM routing.

```yaml
provider: vllm
model: queryos-local
api_key: EMPTY

llm_router:
  enabled: true
  strategy: least_inflight
  request_timeout_seconds: 120
  max_retries: 2
  cooldown_seconds: 30

  endpoints:
    - name: node1-copy1
      base_url: http://node1:8001/v1
      model: queryos-local
      max_inflight: 8
    - name: node1-copy2
      base_url: http://node1:8002/v1
      model: queryos-local
      max_inflight: 8
    - name: node2-copy1
      base_url: http://node2:8001/v1
      model: queryos-local
      max_inflight: 8
```

Routing strategies:

- `least_inflight`: send the next call to the least busy endpoint.
- `round_robin`: rotate through endpoints.
- `random`: weighted random routing using each endpoint's `weight`.

The router is most useful when QueryOS is already concurrent, for example with
`schema_discovery.parallel_workers: 2`, `sql_writer.parallel_workers: 2`, or
`build_failure_memory.py --workers 4`.

## Batch Failure Memory

`dev_20240627-2/build_failure_memory.py` runs QueryOS over benchmark samples,
compares each predicted result with golden SQL, and stores mismatch cases as an
error bank. Each failure receives one unified natural-language error reason from
the configured model.

```bash
cd dev_20240627-2

python build_failure_memory.py \
  --config ../cleaned_query_os/queryos_config.yaml \
  --workers 2 \
  --sleep 0.2
```

Use the vLLM config:

```bash
python build_failure_memory.py \
  --config ../cleaned_query_os/queryos_vllm_config.yaml \
  --provider vllm \
  --workers 2 \
  --sleep 0.2
```

Generated outputs are ignored by git:

```text
dev_20240627-2/failure_memory/
dev_20240627-2/traces/
```

## Config Highlights

Most runtime controls live in YAML. CLI flags override YAML values.

```yaml
workflow:
  max_steps: 20
  validation: auto

planner:
  context: dispatch

schema_discovery:
  parallel_workers: 2
  read_table_summary_max_cols: 80
  trace_column_preview_limit: 0

sql_writer:
  parallel_workers: 2
  chatgroup:
    enabled: true
    max_rounds: 2

trace:
  live: true
  style: pretty
  state_view: full
  max_chars: 0
```

## Development Notes

- SQLite execution is guarded as read-only.
- The manager owns the final workflow decision.
- SQL Validator feedback is natural language; the planner decides the next step.
- Parallel schema workers merge by union and expose only numeric `confidence`.
- Parallel writer workers operate on forked state; only the consensus result is
  committed to shared global state.
- Benchmark datasets are local artifacts and should not be committed directly.
