from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import cfg_get, load_yaml_config, pick
from .sql_agent import QueryOS, result_to_dict


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the cleaned QueryOS SQL generation agent.")
    parser.add_argument("--config", help="Optional YAML config for agent control parameters.")
    parser.add_argument("--question", required=True, help="Natural-language question to answer.")
    parser.add_argument("--db", required=True, dest="db_path", help="Path to a SQLite database.")
    parser.add_argument("--metadata", dest="schema_metadata_path", help="Optional database_description directory.")
    parser.add_argument("--api-key", dest="api_key", help="OpenAI API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", dest="base_url", help="Provider base URL. For vLLM this is the OpenAI-compatible /v1 URL.")
    parser.add_argument(
        "--provider",
        choices=["openai", "vllm"],
        help="LLM provider backend. Use vllm for local OpenAI-compatible vLLM routing.",
    )
    parser.add_argument("--model", help="Model for planner/SDA/SWA/SVA.")
    parser.add_argument("--external-knowledge", default="", help="Optional extra task context.")
    parser.add_argument("--db-id", default="", help="Optional database id/name.")
    parser.add_argument("--max-steps", type=int, help="Manager step budget.")
    parser.add_argument("--gold-sql", default="", help="Optional golden SQL to execute after QueryOS finishes.")
    parser.add_argument("--gold-sql-file", help="Optional file containing golden SQL to execute after QueryOS finishes.")
    parser.add_argument(
        "--no-live-trace",
        action="store_true",
        help="Disable real-time workflow trace on stderr.",
    )
    parser.add_argument(
        "--trace-json",
        help="Write the full result and trace_events JSON to this path.",
    )
    parser.add_argument(
        "--result-json",
        help="Write the final result JSON to this path without printing it to the terminal.",
    )
    parser.add_argument(
        "--print-result-json",
        action="store_true",
        help="Print the final result JSON to stdout. By default this is off when live trace is enabled.",
    )
    parser.add_argument(
        "--trace-style",
        choices=["pretty", "plain"],
        help="Real-time trace display style.",
    )
    parser.add_argument(
        "--trace-color",
        choices=["auto", "always", "never"],
        help="Color mode for real-time trace.",
    )
    parser.add_argument(
        "--trace-max-chars",
        type=int,
        help="Maximum characters per trace field. 0 means no truncation, which is the default.",
    )
    parser.add_argument(
        "--state-view",
        choices=["off", "summary", "diff", "full"],
        help="Show shared global state changes in the live trace.",
    )
    parser.add_argument(
        "--planner-context",
        choices=["compact", "dispatch"],
        help="Context style passed to the manager planner.",
    )
    parser.add_argument(
        "--validation",
        choices=["off", "auto"],
        help="Run the SQL validator automatically after each SQL writer result.",
    )
    parser.add_argument(
        "--auto-finish-on-sql",
        action="store_true",
        help="Legacy mode: finish immediately after a non-suspicious SQL result. Only applies when --validation off.",
    )
    args = parser.parse_args(argv)
    config = load_yaml_config(args.config)
    golden_sql = args.gold_sql
    if args.gold_sql_file:
        with open(args.gold_sql_file, "r", encoding="utf-8") as handle:
            golden_sql = handle.read().strip()

    trace_max_chars = pick(args.trace_max_chars, config, "trace.max_chars", None)
    if trace_max_chars is not None:
        trace_max_chars = int(trace_max_chars)
    if trace_max_chars == 0:
        trace_max_chars = None
    live_trace = bool(cfg_get(config, "trace.live", True))
    if args.no_live_trace:
        live_trace = False
    router_config = cfg_get(config, "llm_router", {})
    if not isinstance(router_config, dict):
        router_config = {}
    provider_default = "vllm" if router_config.get("enabled") else "openai"

    agent = QueryOS(
        provider=pick(args.provider, config, "provider", provider_default),
        api_key=pick(args.api_key, config, "api_key", None),
        model=pick(args.model, config, "model", "gpt-4.1-mini"),
        base_url=pick(args.base_url, config, "base_url", None),
        llm_router_config=router_config,
        llm_timeout=cfg_get(config, "llm_timeout_seconds", None),
        planner_model=cfg_get(config, "models.planner", None),
        schema_model=cfg_get(config, "models.schema_discovery", None),
        sql_model=cfg_get(config, "models.sql_writer", None),
        validator_model=cfg_get(config, "models.sql_validator", None),
        temperature=float(cfg_get(config, "temperature", 0.2)),
        max_steps=int(pick(args.max_steps, config, "workflow.max_steps", 8)),
        planner_max_tokens=int(cfg_get(config, "planner.max_tokens", 1024)),
        schema_max_tokens=int(cfg_get(config, "schema_discovery.max_tokens", 4096)),
        schema_max_turns=int(cfg_get(config, "schema_discovery.max_turns", 6)),
        schema_max_tool_calls_per_turn=int(cfg_get(config, "schema_discovery.max_tool_calls_per_turn", 4)),
        schema_read_table_summary_max_cols=int(cfg_get(config, "schema_discovery.read_table_summary_max_cols", 30)),
        schema_trace_column_preview_limit=int(cfg_get(config, "schema_discovery.trace_column_preview_limit", 8)),
        schema_parallel_workers=int(cfg_get(config, "schema_discovery.parallel_workers", 1)),
        sql_max_tokens=int(cfg_get(config, "sql_writer.max_tokens", 4096)),
        sql_max_turns=int(cfg_get(config, "sql_writer.max_turns", 8)),
        sql_parallel_workers=int(cfg_get(config, "sql_writer.parallel_workers", 1)),
        sql_chatgroup_enabled=bool(cfg_get(config, "sql_writer.chatgroup.enabled", True)),
        sql_chatgroup_max_rounds=int(cfg_get(config, "sql_writer.chatgroup.max_rounds", 2)),
        sql_consensus_require_same_columns=bool(
            cfg_get(config, "sql_writer.consensus.require_same_columns", False)
        ),
        validator_max_tokens=int(cfg_get(config, "sql_validator.max_tokens", 2048)),
        live_trace=live_trace,
        trace_json_path=args.trace_json,
        trace_style=pick(args.trace_style, config, "trace.style", "pretty"),
        trace_color=pick(args.trace_color, config, "trace.color", "auto"),
        trace_max_chars=trace_max_chars,
        trace_sql_preview_rows=int(cfg_get(config, "trace.sql_preview_rows", 3)),
        trace_gold_preview_rows=int(cfg_get(config, "trace.gold_preview_rows", 3)),
        trace_result_cell_max_width=int(cfg_get(config, "trace.result_cell_max_width", 32)),
        state_view=pick(args.state_view, config, "trace.state_view", "diff"),
        planner_context=pick(args.planner_context, config, "planner.context", "dispatch"),
        validation_mode=pick(args.validation, config, "workflow.validation", "auto"),
        auto_finish_on_sql=args.auto_finish_on_sql or bool(cfg_get(config, "workflow.auto_finish_on_sql", False)),
    )
    result = agent.generate(
        question=args.question,
        db_path=args.db_path,
        schema_metadata_path=args.schema_metadata_path,
        external_knowledge=args.external_knowledge,
        db_id=args.db_id,
        golden_sql=golden_sql,
    )
    result_doc = result_to_dict(result)
    if args.result_json:
        result_path = Path(args.result_json).expanduser()
        if result_path.parent and str(result_path.parent) != ".":
            result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result_doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.print_result_json or args.no_live_trace:
        print(json.dumps(result_doc, ensure_ascii=False, indent=2, default=str))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
