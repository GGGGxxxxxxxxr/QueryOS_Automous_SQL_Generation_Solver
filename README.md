# QueryOS Autonomous SQL Generation Solver

QueryOS is a cleaned, self-contained SQL generation agent for SQLite databases. It is designed like a small agent operating system: a manager/planner controls shared state, dispatches specialized workers, validates candidate SQL, and emits a live terminal trace of the workflow.

The cleaned implementation lives in [`cleaned_query_os/`](cleaned_query_os/). The previous raw implementation is preserved in [`raw_query_os_original/`](raw_query_os_original/) for reference.

## What QueryOS Does

- Discovers relevant schema with a Schema Discovery Agent.
- Writes and executes read-only SQLite SQL with a SQL Writer Agent.
- Supports parallel SQL writer workers and a bounded writer chatgroup for consensus.
- Validates SQL through an independent SQL Validator Agent before the planner can finish.
- Shows real-time CLI workflow logs, shared state updates, SQL previews, and golden SQL checks.
- Dumps full JSON traces/results for later debugging.
- Supports OpenAI API models and OpenAI-compatible local vLLM servers.

## Repository Layout

```text
.
├── cleaned_query_os/
│   ├── query_os/                  # QueryOS package
│   ├── queryos_config.yaml        # high-capacity OpenAI config
│   ├── queryos_vllm_config.yaml   # local vLLM config template
│   ├── README.md                  # package-level usage docs
│   └── pyproject.toml
├── dev_20240627-2/
│   ├── build_failure_memory.py    # batch evaluator + failure-memory builder
│   └── build_table_description_json.py
└── raw_query_os_original/         # archived original files
```

Large benchmark assets are intentionally not committed: `dev_20240627-2/dev_databases/`, traces, failure-memory outputs, SQLite files, CSV files, and JSON benchmark data are ignored by git.

## Install

From a fresh conda environment:

```bash
conda create -n queryos python=3.11 -y
conda activate queryos

cd /path/to/QueryOS_Automous_SQL_Generation_Solver/cleaned_query_os
pip install -e .
```

After installation, the CLI entrypoint is available as:

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

Start an OpenAI-compatible vLLM server separately, then edit:

```text
cleaned_query_os/queryos_vllm_config.yaml
```

Set:

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
  --db dev_20240627-2/dev_databases/financial/financial.sqlite \
  --metadata dev_20240627-2/dev_databases/financial/database_description \
  --question "The transaction of 840 USD happened in 1998/10/14, when was this account opened?"
```

QueryOS depends heavily on OpenAI-style tool/function calling. The local vLLM deployment must support tool calls for the selected model and chat template.

### Multiple vLLM Copies

If the same model is hosted on several vLLM servers, QueryOS can route calls across them from the client side. OpenAI and vLLM are separate backends: use `provider: openai` for the hosted OpenAI API, and `provider: vllm` for local OpenAI-compatible vLLM routing.

Example:

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
    - name: node2-copy2
      base_url: http://node2:8002/v1
      model: queryos-local
      max_inflight: 8
```

Routing strategies:

- `least_inflight`: send the next call to the least busy endpoint.
- `round_robin`: rotate through endpoints.
- `random`: weighted random routing using each endpoint's `weight`.

The router is most useful when QueryOS is already concurrent, for example with `sql_writer.parallel_workers: 2` or with `build_failure_memory.py --workers 4`.

Parallel schema discovery also benefits from multiple vLLM copies. SDA workers run on forked shared state, then QueryOS merges their discovered schemas by union and assigns numeric `confidence` from worker agreement. With two SDA workers, a column selected by both workers has `confidence: 1.0`; a column selected by only one worker has `confidence: 0.5`.

## Batch Failure Memory

`dev_20240627-2/build_failure_memory.py` runs QueryOS over benchmark samples, compares each predicted result with golden SQL, and stores mismatch cases as an error bank. Each failure receives a unified natural-language error reason generated by the configured model.

Example:

```bash
cd dev_20240627-2

python build_failure_memory.py \
  --config ../cleaned_query_os/queryos_config.yaml \
  --workers 2 \
  --sleep 0.2
```

Use the vLLM config instead:

```bash
python build_failure_memory.py \
  --config ../cleaned_query_os/queryos_vllm_config.yaml \
  --workers 2 \
  --sleep 0.2
```

Generated outputs are ignored by git:

```text
dev_20240627-2/failure_memory/
dev_20240627-2/traces/
```

## Config Highlights

Most runtime controls live in YAML:

```yaml
workflow:
  max_steps: 20
  validation: auto

planner:
  context: dispatch

schema_discovery:
  parallel_workers: 2

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

CLI flags override YAML values.

## Development Notes

- SQLite execution is guarded as read-only.
- The manager owns the final workflow decision.
- SQL Validator feedback is natural language; the planner decides the next step.
- Parallel writer workers operate on forked state; only the writer-group consensus result is committed to shared global state.
- Benchmark datasets are local artifacts and should not be committed directly.
