# Cleaned QueryOS

This folder contains a cleaned, self-contained version of the QueryOS SQL generation agent.

It keeps the original three-role design:

- `SchemaDiscoveryAgent`: discovers the minimal schema needed for a question.
- `SQLWriterAgent`: writes and executes read-only SQLite SQL.
- `QueryOS` manager: routes between schema discovery and SQL writing.

The training stack, AgentLightning wrapper, Spider2 helper code, reward/eval code, Snowflake code, and unrelated logs/scripts were removed.

## Install

```bash
cd cleaned_query_os
pip install -e .
```

or:

```bash
pip install -r cleaned_query_os/requirements.txt
```

## Python API

```python
from query_os import QueryOS

agent = QueryOS(
    api_key="YOUR_OPENAI_API_KEY",
    model="gpt-4.1-mini",
)

result = agent.generate(
    question="Which customer has the highest total order amount?",
    db_path="/path/to/database.sqlite",
)

print(result.final_sql)
print(result.columns)
print(result.rows)
```

If you already have Spider-style metadata JSON files, pass them explicitly:

```python
result = agent.generate(
    question="...",
    db_path="/path/to/database.sqlite",
    schema_metadata_path="/path/to/database_description",
)
```

If `schema_metadata_path` is omitted, QueryOS introspects the SQLite database schema directly.

## CLI

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

query-os \
  --db /path/to/database.sqlite \
  --question "Which customer has the highest total order amount?"
```

By default the CLI prints a real-time workflow trace to `stderr`, for example:

```text
+------------------------------------------------------------
| QueryOS Agent Run
| question: Which customer has the highest total order amount?
| database: /path/to/database.sqlite
+------------------------------------------------------------

GLOBAL STEP 1
  Manager -> Schema Discovery Agent
  guidance: Find the minimal tables, columns, and join keys needed.
  SDA worker started
    task: Find the minimal tables, columns, and join keys needed.
  SDA turn 1
    planned tools: SEARCH_METADATA
    [OK] SEARCH_METADATA
      keywords: ['customer', 'order']
      matched tables: 2

GLOBAL STEP 2
  Manager -> SQL Writer Agent
  SWA turn 1
    planned tools: SQLITE_EXEC
    [OK] SQLITE_EXEC
      sql: SELECT ...
      rows returned: 1

SHARED STATE UPDATE
  writer: SQL Writer Agent
  Schema Memory
    no changes
  SQL Memory
    + attempt #1: executed_ok
      rows returned: 1
  Planner Memory
    + decision: CALL_SQL_WRITER
```

By default, live trace is printed to the terminal and the full result JSON is not printed. To persist the full trace:

```bash
query-os \
  --db /path/to/database.sqlite \
  --question "..." \
  --trace-json traces/run_001.json
```

To save the final result JSON separately:

```bash
query-os \
  --db /path/to/database.sqlite \
  --question "..." \
  --result-json traces/run_001_result.json
```

Use `--print-result-json` to print the full result JSON to stdout.

To compare against a golden SQL after QueryOS finishes:

```bash
query-os \
  --db /path/to/database.sqlite \
  --question "..." \
  --gold-sql 'SELECT ...'
```

For long SQL, put it in a file and use `--gold-sql-file path/to/gold.sql`.

Disable live trace with `--no-live-trace`.
Use `--trace-style plain` for the compact machine-like log, or `--trace-color never` to disable colors.
Trace fields are not truncated by default. Use `--trace-max-chars 180` if you want compact logs.
Shared global state changes are shown by default with `--state-view diff`.
Use `--state-view summary` for a compact memory summary, `--state-view full` to include SQL previews in the state panel, or `--state-view off` to hide shared state panels while still keeping state events in `--trace-json`.
The planner receives dispatch-oriented context by default with `--planner-context dispatch`: task, current shared state, SQL memory, and manager dispatch history. Use `--planner-context compact` to compare against the older shorter context.
SQL validation runs automatically after each SQL writer result with `--validation auto`. The validator writes natural-language feedback into Validation Memory; the planner still decides the next action and must issue the final finish. Use `--validation off` to disable this gate. `--auto-finish-on-sql` only applies when validation is off.
When `sql_writer.parallel_workers` is greater than 1, SWA internally forks multiple SQL writers. Each writer runs on a forked state; only the writer-group consensus SQL is committed to shared global state. If candidates disagree, the writers enter a bounded chatgroup where each writer either agrees with one current SQL or revises its own SQL. Revised SQL is executed immediately before the next chat round.

### YAML Config

Agent control parameters can be kept in a YAML file:

```bash
query-os \
  --config queryos_config.yaml \
  --db /path/to/database.sqlite \
  --metadata /path/to/database_description \
  --question "..."
```

CLI flags override YAML values. Useful controls include:

```yaml
workflow:
  max_steps: 8
  validation: auto

schema_discovery:
  read_table_summary_max_cols: 30      # actual READ_TABLE_JSON columns sent to SDA
  trace_column_preview_limit: 8        # terminal preview only; 0 shows all summary columns

sql_writer:
  parallel_workers: 2
  consensus:
    require_same_columns: false
  chatgroup:
    enabled: true
    max_rounds: 2

trace:
  max_chars: 0
  sql_preview_rows: 10
  gold_preview_rows: 10
  result_cell_max_width: 64
  state_view: diff
```

Optional metadata:

```bash
query-os \
  --db /path/to/database.sqlite \
  --metadata /path/to/database_description \
  --question "..."
```

## Metadata Format

Each table JSON should look like this:

```json
{
  "table": "orders",
  "columns": [
    {"name": "id", "type": "INTEGER"},
    {"name": "customer_id", "type": "INTEGER"},
    {"name": "amount", "type": "REAL"}
  ],
  "primary_keys": ["id"],
  "foreign_keys": [
    {"column": "customer_id", "ref_table": "customers", "ref_column": "id"}
  ]
}
```

## Notes

- SQLite execution is guarded by a read-only authorizer.
- The final SQL is the latest successful `SQLITE_EXEC`.
- Result rows are capped for safety by `SQLiteExecutor`; adjust it in code if you need larger previews.
