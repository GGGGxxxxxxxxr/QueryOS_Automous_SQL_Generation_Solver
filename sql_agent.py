#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional
import agentlightning as agl
import openai
import time
import os
from agentlightning.litagent import DiscardTrajectory

# Import shared types from shared_types.py to avoid circular imports
from shared_states import (
    PlannerAction,
    AgentName,
    TableEvidence,
    DiscoveredSchema,
    SQLAttempt,
    PlannerDecision,
    AgentReturn,
    TraceStep,
    SharedGlobalState,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(processName)s | %(levelname)s | %(name)s | %(message)s",
    force=True, 
)

logger = logging.getLogger(__name__)



# =============================================================================
# Planner prompt + state formatting
# =============================================================================
def build_planner_system_prompt() -> str:
    return """You are the PLANNER in a collaborative SQL agent system.

Role:
- Decide the NEXT action only (routing + short guidance).
- You DO NOT write SQL.
- You DO NOT modify schema.
- You must rely ONLY on the provided shared state.

Workers:
- SDA (Schema Discovery Agent): updates discovered_schema only.
- SWA (SQL Writer Agent): writes SQL using ONLY discovered_schema; SQL execution results will appear in state.sql_attempts.

============================================================
TOOLS (STRICT)
============================================================
On EVERY turn, output EXACTLY ONE tool call (no free text).

Allowed tools:
1) CALL_SCHEMA_DISCOVERY {"guidance": "..."}
2) CALL_SQL_WRITER       {"guidance": "..."}
3) PLANNER_FINISH        {"guidance": "..."}

No markdown. No extra text. One tool call only.

============================================================
WHEN TO CALL SDA vs SWA vs FINISH
============================================================

CALL_SCHEMA_DISCOVERY when ANY is true:
- Missing key tables/columns needed to answer the question.
- Join path is unclear (no reliable keys between needed tables).
- SWA previously failed due to missing schema elements.

CALL_SQL_WRITER when:
- discovered_schema contains the tables/columns to answer the question,
  and at least one plausible join path exists.

PLANNER_FINISH only when ALL are true:
- Latest executed SQL exists and logically answers the question.
- Result is not empty / not all NULL (unless question expects that).
- No obvious join explosion / missing filter issue.

============================================================
SANITY CHECKS (lightweight but strict)
============================================================
Treat these results as suspicious unless the question clearly expects them:
- Empty result set / COUNT(*) = 0
- Answer column(s) are all NULL
- Sudden huge row count increase after adding a join (join explosion)
- Using categorical string filters without confirming real stored values

If suspicious, instruct SWA to debug incrementally:
- First run a small probe query (LIMIT 5) to confirm joins.
- Validate each WHERE filter separately (especially categorical fields).
- Compare counts before/after joins to detect duplication.
- For top-k: compute ranking key first, then join back for attributes.
- For ratio/percentage: compute numerator and denominator separately first.

============================================================
GUIDANCE STYLE (IMPORTANT)
============================================================
Your guidance must be short and actionable:
- Name the target tables/columns if known.
- Specify the exact sub-goal for the next step.
- If debugging: say what to validate first (counts? distinct values? join keys?).

Examples:
- "Find which table contains (customer_id, order_date) and how it joins to Orders."
- "Write SQL incrementally: first identify candidate rows, then add aggregation."
- "Result is empty. Probe DISTINCT values for status column before filtering."
"""


def format_state_for_planner(
    state: SharedGlobalState,
    max_cols_per_table: int = 9999,
) -> str:
    """
    What the PLANNER can see:
    1) Full discovered_schema
    2) SQL Writer execution history (SQL context)
    3) Last planner step + verbalized sub-agent feedback
    """

    # ------------------------------------------------------------------
    # 1) Discovered schema (FULL visibility)
    # ------------------------------------------------------------------
    discovered_tables = []
    for table_key, ev in state.discovered.tables.items():
        discovered_tables.append(
            {
                "table": table_key,
                "columns": (ev.columns or [])[:max_cols_per_table],
                "primary_keys": ev.primary_keys or [],
                "foreign_keys": ev.foreign_keys or [],
            }
        )

    # ------------------------------------------------------------------
    # 2) SQL context (SQL Writer history)
    # ------------------------------------------------------------------
    sql_context = {
        "latest_sql": None,
        "previous_sqls": [],
    }

    if state.sql_attempts:
        latest = state.sql_attempts[-1]
        sql_context["latest_sql"] = {
            "sql": latest.sql,
            "status": latest.status,
            "result": latest.result,
        }

        prev_attempts = state.sql_attempts[:-1][-2:]
        for a in reversed(prev_attempts):
            sql_context["previous_sqls"].append(
                {
                    "sql": a.sql,
                    "status": a.status,
                    "result": a.result,
                }
            )

    # ------------------------------------------------------------------
    # 3) Last planner step + verbalized sub-agent feedback
    # ------------------------------------------------------------------
    last_trace = None
    if state.planner_trace:
        last = state.planner_trace[-1]

        verbal_feedback = None
        if last.agent_return:
            verbal_feedback = {
                "from_agent": last.agent_return.agent,
                "ok": last.agent_return.ok,
                "message": last.agent_return.report,
            }

        last_trace = {
            "step_idx": last.step_idx,
            "planner_decision": {
                "action": last.decision.action,
                "guidance": last.decision.guidance,
            },
            "agent_feedback": verbal_feedback,
        }

    # ------------------------------------------------------------------
    # 4) Final payload to PLANNER
    # ------------------------------------------------------------------
    payload = {
        "question": state.question,
        "external_knowledge": (state.external_knowledge or "")[:1500],

        "discovered_schema": discovered_tables,
        "sql_context": sql_context,

        "last_action_feedback": last_trace,

        "current_step": state.step,
        "max_allowed_steps": state.max_steps,
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


# =============================================================================
# Planner (LLM-based, TOOL-ONLY)
# =============================================================================
class Planner:
    """
    Tool-only planner:
      - CALL_SCHEMA_DISCOVERY(guidance)
      - CALL_SQL_WRITER(guidance)
      - PLANNER_FINISH(guidance)

    Strict:
      - Exactly ONE tool call per turn
      - message.content must be empty
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 700,
        api_key_env: str = "OPENAI_API_KEY",
        debug: bool = False,
    ):
        self.client = openai.OpenAI(
            base_url=endpoint,
            api_key=__import__("os").getenv(api_key_env, "dummy"),
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug

        self.openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "CALL_SCHEMA_DISCOVERY",
                    "description": "Ask Schema Discovery Agent (SDA) to update discovered_schema. Provide specific guidance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "guidance": {"type": "string", "description": "Instruction to SDA"}
                        },
                        "required": ["guidance"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "CALL_SQL_WRITER",
                    "description": "Ask SQL Writer Agent (SWA) to write/execute SQL using ONLY discovered_schema. Provide specific guidance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "guidance": {"type": "string", "description": "Instruction to SWA"}
                        },
                        "required": ["guidance"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "PLANNER_FINISH",
                    "description": "Finish the trajectory when the latest SQL result is sufficient and passes sanity checks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "guidance": {"type": "string", "description": "Short finishing rationale"}
                        },
                        "required": ["guidance"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _call_llm(self, messages: List[Dict[str, Any]]) -> Optional[Any]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.openai_tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body={
                    "repetition_penalty": 1.05,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
            return resp.choices[0].message
        except Exception as e:
            logger.error("[Planner] LLM call failed: %s", e)
            return None

    @staticmethod
    def _tool_to_decision(tool_name: str, guidance: str) -> PlannerDecision:
        name = (tool_name or "").strip()
        g = (guidance or "").strip()

        if name == "CALL_SCHEMA_DISCOVERY":
            return PlannerDecision(action=PlannerAction.CALL_SCHEMA_DISCOVERY, guidance=g)
        if name == "CALL_SQL_WRITER":
            return PlannerDecision(action=PlannerAction.CALL_SQL_WRITER, guidance=g)
        if name == "PLANNER_FINISH":
            return PlannerDecision(action=PlannerAction.FINISH, guidance=g)

        raise ValueError(f"Unknown planner tool: {name}")

    def decide(self, state: SharedGlobalState) -> PlannerDecision:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": build_planner_system_prompt()},
            {
                "role": "user",
                "content": (
                    "Here is the current shared global state:\n"
                    f"{format_state_for_planner(state)}\n\n"
                    "Decide the NEXT action.\n"
                    "STRICT: output exactly ONE tool call "
                    "(CALL_SCHEMA_DISCOVERY / CALL_SQL_WRITER / PLANNER_FINISH)."
                ),
            },
        ]

        msg = self._call_llm(messages)
        if not msg:
            logger.error("[Planner] LLM returned None")
            raise DiscardTrajectory("Planner returned None")

        tool_calls = getattr(msg, "tool_calls", None)
        content = (getattr(msg, "content", "") or "").strip()

        logger.info(
            "[Planner] tool_calls=%s | content_len=%s",
            (len(tool_calls) if tool_calls else 0),
            len(content),
        )
        if self.debug and content:
            logger.info("[Planner] content_preview=%r", content[:300])

        # STRICT: exactly one tool call
        if not tool_calls or len(tool_calls) != 1:
            raise DiscardTrajectory("Planner invalid: must output exactly ONE tool call")

        tc = tool_calls[0]
        if tc.type != "function":
            raise DiscardTrajectory("Planner invalid: tool_call type must be function")

        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            raise DiscardTrajectory("Planner invalid: tool arguments not JSON-parsable")

        guidance = args.get("guidance", "")
        if not isinstance(guidance, str) or not guidance.strip():
            raise DiscardTrajectory("Planner invalid: guidance must be a non-empty string")

        try:
            decision = self._tool_to_decision(name, guidance)
        except Exception as e:
            logger.error("[Planner] invalid tool name=%r err=%s", name, e)
            raise DiscardTrajectory(f"Planner invalid tool name: {name}")

        # record trace
        state.planner_trace.append(
            TraceStep(step_idx=state.step, decision=decision, agent_return=None)
        )

        return decision

# =============================================================================
# Placeholders for other agents (to be implemented later)
# =============================================================================
from schema_discovery_agent import SchemaDiscoveryAgent
from sql_writer import SQLwriterAgent



# =============================================================================
# Orchestrator loop (treat whole system as ONE agent for RL)
# =============================================================================
class AgenticSystem:
    """
    The whole collaborative system is treated as ONE agent for RL.
    Planner = policy brain
    SDA / SWA = controlled tools with memory
    """

    def __init__(
        self,
        planner: Planner,
        schema_discovery_agent: SchemaDiscoveryAgent,
        sql_writer_agent: SQLwriterAgent,
        debug: bool = False,
    ):
        self.planner = planner
        self.sda = schema_discovery_agent
        self.swa = sql_writer_agent
        self.debug = debug

    # --------------------------------------------------
    # Core rollout loop
    # --------------------------------------------------
    def run(self, state: SharedGlobalState) -> Dict[str, Any]:
        """
        Execute one full agentic trajectory.
        This function is RL-friendly and deterministic in structure.
        """

        while state.step < state.max_steps:

            # ==================================================
            # 1. Planner decides next action
            # ==================================================
            decision = self.planner.decide(state)

            if self.debug:
                logger.info(f"[Planner] step={state.step} action={decision.action}")

            # ==================================================
            # 2. Dispatch to sub-agent
            # ==================================================
            agent_return: Optional[AgentReturn] = None

            try:
                if decision.action == PlannerAction.CALL_SCHEMA_DISCOVERY:
                    agent_return = self.sda.run(
                        state=state,
                        guidance=decision.guidance,
                    )

                elif decision.action == PlannerAction.CALL_SQL_WRITER:
                    logger.info(
                        f"[AgenticSystem] Calling SQL Writer with guidance: {decision.guidance[:100]}..."
                    )
                    agent_return = self.swa.run(
                        state=state,
                        guidance=decision.guidance,
                    )
                    logger.info(
                        f"[AgenticSystem] SQL Writer returned: ok={agent_return.ok}, "
                        f"report={agent_return.report[:100] if agent_return.report else 'None'}..."
                    )

                elif decision.action == PlannerAction.FINISH:
                    agent_return = AgentReturn(
                        agent=AgentName.PLANNER,
                        ok=True,
                        report=decision.guidance or "Planner decided to finish.",
                        payload={},
                    )
                    self._attach_agent_return(state, agent_return)
                    break

                else:
                    agent_return = AgentReturn(
                        agent=AgentName.PLANNER,
                        ok=False,
                        report="Unknown planner action. Terminating trajectory.",
                        payload={},
                    )
                    self._attach_agent_return(state, agent_return)
                    break

            except Exception as e:
                agent_return = AgentReturn(
                    agent=AgentName.PLANNER,
                    ok=False,
                    report=f"Runtime exception during agent execution: {e}",
                    payload={"reason": "runtime_exception"},
                )
                self._attach_agent_return(state, agent_return)
                break

            # ==================================================
            # 3. Attach sub-agent feedback to trace
            # ==================================================
            if agent_return is not None:
                self._attach_agent_return(state, agent_return)

                # 只要 sub-agent / planner 返回 ok=False，就整个 break
                if agent_return.ok is False:
                    break

            # ==================================================
            # 4. Step forward
            # ==================================================
            state.step += 1

        # ======================================================
        # Final output for evaluator / RL reward function
        # ======================================================
        return {
            "question": state.question,
            "steps": state.step,
            "final_sql": self._select_final_sql(state),
            "sql_attempts": [asdict(a) for a in state.sql_attempts],
            "discovered_schema": {
                k: asdict(v) for k, v in state.discovered.tables.items()
            },
            "planner_trace": [
                {
                    "step_idx": t.step_idx,
                    "decision": {
                        "action": t.decision.action,
                        "guidance": t.decision.guidance,
                    },
                    "agent_return": (
                        {
                            "agent": t.agent_return.agent,
                            "ok": t.agent_return.ok,
                            "report": t.agent_return.report,
                            "payload": t.agent_return.payload,
                        }
                        if t.agent_return
                        else None
                    ),
                }
                for t in state.planner_trace
            ],
        }

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    @staticmethod
    def _attach_agent_return(state: SharedGlobalState, ret: AgentReturn) -> None:
        """
        Attach AgentReturn to the last TraceStep.
        """
        if not state.planner_trace:
            return
        state.planner_trace[-1].agent_return = ret

    @staticmethod
    def _select_final_sql(state: SharedGlobalState) -> str:
        """
        Policy: return the most recent successfully executed SQL.
        """
        for attempt in reversed(state.sql_attempts):
            if attempt.status == "executed_ok" and attempt.sql.strip():
                return attempt.sql
        return ""


import logging
import time
from typing import Any, Dict, Optional, cast
logger = logging.getLogger(__name__)



class LitAgenticSQL(agl.LitAgent[Dict[str, Any]]):
    """
    AGL wrapper for the collaborative system:
      Planner (no tools) + SDA (CMD) + SWA (SQLITE_EXEC)
    """

    def __init__(
        self,
        trained_agents: Optional[str] = None,
        val_temperature: Optional[float] = None,
        max_steps: int = 6,
        debug: bool = True,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.max_steps = max_steps
        self.debug = debug

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:

        start_time = time.time()
        rollout_id = rollout.rollout_id

        # ----------------------------
        # Task fields (sqlite setting)
        # ----------------------------
        question             = task["question"]
        db_id                = task.get("db_id", "")
        external_knowledge   = task.get("external_knowledge", "")
        ground_truth         = task.get("expected_result", task.get("expected_sql", ""))

        # Support both Spider 1.0 and Spider 2.0 formats:
        if "db_path" in task:
            db_path = task["db_path"]
            db_root = task.get("db_root", os.path.dirname(db_path))
            schema_metadata_path = os.path.join(db_root, "database_description")
        else:
            db_root = task.get("db_root", "")
            db_path = db_root
            schema_metadata_path = os.path.join(db_root, db_id) if db_root and db_id else ""

        metadata_display = ""
        if schema_metadata_path and os.path.isdir(schema_metadata_path):
            try:
                tables = []
                for f in sorted(os.listdir(schema_metadata_path)):
                    if not f.endswith(".json"):
                        continue
                    table = os.path.splitext(f)[0]
                    tables.append(f"- {table}  ({f})")
                metadata_display = "\n".join(tables)
            except Exception:
                metadata_display = ""

        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        endpoint = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)

        sampling = dict(llm.sampling_parameters or {})
        if rollout.mode == "train":
            verl_replacement = {"model": llm.model, **sampling}
        else:
            verl_replacement = {
                "model": llm.model,
                "temperature": (
                    self.val_temperature
                    if self.val_temperature is not None
                    else sampling.get("temperature", 0.0)
                ),
            }

        if self.debug:
            logger.info(f"[Rollout {rollout_id}] Q: {question}")
            logger.info(f"[Rollout {rollout_id}] db_path={db_path}")
            logger.info(f"[Rollout {rollout_id}] schema_metadata_path={schema_metadata_path}")
            logger.info(f"[Rollout {rollout_id}] endpoint={endpoint}")
            logger.info(f"[Rollout {rollout_id}] verl_replacement={verl_replacement}")

        # ----------------------------
        # Init shared state
        # ----------------------------
        state = build_initial_state(
            question=question,
            db_id=db_id,
            db_path=db_path,
            external_knowledge=external_knowledge,
            schema_metadata_path=schema_metadata_path,
            metadata_display=metadata_display,
            max_steps=self.max_steps,
        )

        logger.info(f"[state={state}")

        # ----------------------------
        # Init 3 agents
        # ----------------------------
        planner = Planner(
            endpoint=endpoint,
            model=verl_replacement["model"],
            temperature=verl_replacement.get("temperature", 0.2),
            max_tokens=32768,
        )
        logger.info("planner init")

        sda = SchemaDiscoveryAgent(
            endpoint=endpoint,
            model=verl_replacement["model"],
            temperature=verl_replacement.get("temperature", 0.3),
            max_tokens=32768,
            max_turns=12,
            max_cmd_calls=5,
            debug=self.debug,
        )
        logger.info("sda init")

        swa = SQLwriterAgent(
            endpoint=endpoint,
            model=verl_replacement["model"],
            temperature=verl_replacement.get("temperature", 0.3),
            max_tokens=32768,
            max_turns=10,
            debug=self.debug,
        )
        logger.info("swa init")

        system = AgenticSystem(
            planner=planner,
            schema_discovery_agent=sda,
            sql_writer_agent=swa,
            debug=self.debug,
        )
        logger.info("system init")

        try:
            result = system.run(state)
        except DiscardTrajectory:
            # 保持你原来逻辑
            self.discard_trajectory("discarded trajactories")
            return None

        except Exception as e:
            logger.exception(f"[Rollout {rollout_id}] AgenticSystem failed: {e}")
            self.discard_trajectory("unknown issues, discard trajactories")
            return None

        end_time_rollout = time.time()

        if self.debug:
            logger.info(f"[Rollout {rollout_id}] steps={result.get('steps')}")
            logger.info(f"[Rollout {rollout_id}] final_sql={result.get('final_sql','')[:500]}")

        # ----------------------------
        # Reward
        # ----------------------------
        try:
            reward = self._evaluate_result(result, db_path, ground_truth)
        except Exception as e:
            logger.exception(f"[Rollout {rollout_id}] Evaluation failed: {e}")
            reward = 0.0

        end_time_eval = time.time()
        logger.info("[Rollout %s] Reward: %s", rollout_id, reward)
        logger.info("[Rollout %s] Time rollout: %.2fs", rollout_id, end_time_rollout - start_time)
        logger.info("[Rollout %s] Time eval: %.2fs", rollout_id, end_time_eval - end_time_rollout)

        return reward
    
    '''
    def _evaluate_result(self, result: Dict, db_path: str, ground_truth_sql: str) -> float:
        """
        Official EX only (binary):
        - 1.0 if execution results match (multiset rows)
        - 0.0 otherwise
        """
        import sqlite3
        from collections import Counter

        SUCCESS_PATH = "/efs/open_source_sql_agentic_rl/agent-lightning/examples/spider2_clean/dev_counter_gpt.txt"

        predicted_sql = (result.get("final_sql") or "").strip()
        if not predicted_sql:
            logger.info("[SQL EVAL] Empty SQL")
            return 0.0

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

            cur.execute(predicted_sql)
            pred_rows = cur.fetchall()

            cur.execute(ground_truth_sql)
            gt_rows = cur.fetchall()

            conn.close()
        except Exception as e:
            logger.info(f"[SQL EVAL] Execution failed | error={e}")
            return 0.0

        ex = 1.0 if set(pred_rows) == set(gt_rows) else 0.0

        if ex == 1.0:
            try:
                with open(SUCCESS_PATH, "a") as f:
                    f.write(f"1\t{ground_truth_sql}\n")
            except Exception as e:
                logger.info(f"[SQL EVAL] Write counter failed | {e}")

            print("🐱 SQL EXECUTION PERFECT! reward = 1.0")

        logger.info(
            f"[SQL EVAL] EX={int(ex)} | pred_rows={len(pred_rows)} gt_rows={len(gt_rows)}"
        )

        return ex
    '''

    def calculate_ex(self, predicted_res, ground_truth_res):
        res = 0
        if set(predicted_res) == set(ground_truth_res):
            res = 1
        return res
    
    def calculate_row_match(self, predicted_row, ground_truth_row):
        total_columns = len(ground_truth_row)

        matches = 0
        element_in_pred_only = 0
        element_in_truth_only = 0

        for pred_val in predicted_row:
            if pred_val in ground_truth_row:
                matches += 1
            else:
                element_in_pred_only += 1

        for truth_val in ground_truth_row:
            if truth_val not in predicted_row:
                element_in_truth_only += 1

        match_percentage = matches / total_columns
        pred_only_percentage = element_in_pred_only / total_columns
        truth_only_percentage = element_in_truth_only / total_columns

        return match_percentage, pred_only_percentage, truth_only_percentage
    
    def calculate_f1_score(self, predicted, ground_truth):
        """
        Official Spider Soft-F1
        """

        if not predicted and not ground_truth:
            return 1.0

        predicted_set = set(predicted) if predicted else set()
        ground_truth_set = set(ground_truth)

        predicted = list(dict.fromkeys(predicted))
        ground_truth = list(dict.fromkeys(ground_truth))

        match_scores = []
        pred_only_scores = []
        truth_only_scores = []

        for i, gt_row in enumerate(ground_truth):

            if i >= len(predicted):
                match_scores.append(0)
                truth_only_scores.append(1)
                continue

            pred_row = predicted[i]

            match_score, pred_only_score, truth_only_score = self.calculate_row_match(
                pred_row, gt_row
            )

            match_scores.append(match_score)
            pred_only_scores.append(pred_only_score)
            truth_only_scores.append(truth_only_score)

        for i in range(len(predicted) - len(ground_truth)):
            match_scores.append(0)
            pred_only_scores.append(1)
            truth_only_scores.append(0)

        tp = sum(match_scores)
        fp = sum(pred_only_scores)
        fn = sum(truth_only_scores)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        return f1_score

    def _write_sql_log(self, path, pred_sql, gt_sql, reward, ex, f1):
        try:
            with open(path, "a") as f:
                f.write("====================================\n")
                f.write(f"EX: {ex} | F1: {round(f1,4)} | reward: {round(reward,4)}\n")
                f.write("PREDICTED SQL:\n")
                f.write(pred_sql + "\n")
                f.write("GROUND TRUTH SQL:\n")
                f.write(gt_sql + "\n")
        except Exception:
            pass

    def _evaluate_result(self, result: dict, db_path: str, ground_truth_sql: str) -> float:
        import sqlite3
        import logging
        import threading
        import time
        import re

        logger = logging.getLogger(__name__)

        LOG_PATH = "/efs/open_source_sql_agentic_rl/agent-lightning/examples/spider2_clean/sql_eval_log.txt"

        QUERY_TIMEOUT_S = 200
        BUSY_TIMEOUT_MS = 10_000

        PRINT_PREVIEW_ROWS = 5
        PRINT_MAX_CELL_CHARS = 200
        PRINT_MAX_SQL_CHARS = 500

        # Extra reward for exact execution match
        EX_BONUS = 0.05

        predicted_sql = (result.get("final_sql") or "").strip()
        ground_truth_sql = (ground_truth_sql or "").strip()
        steps = int(result.get("steps", 0) or 0)

        if not predicted_sql:
            logger.info("[SQL EVAL] Empty SQL")
            return 0.0

        sql_lower = predicted_sql.lower()
        invalid_prefix = ("pragma", "explain", "describe", ".schema")
        if sql_lower.startswith(invalid_prefix):
            logger.info(f"[SQL EVAL] Ignore schema query: {predicted_sql[:PRINT_MAX_SQL_CHARS]}")
            return 0.0

        def _norm_sql(s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"\s+", " ", s)
            if s.endswith(";"):
                s = s[:-1].strip()
            return s.lower()

        if _norm_sql(predicted_sql) == _norm_sql(ground_truth_sql) and ground_truth_sql:
            reward = 1.0 + EX_BONUS
            logger.info(
                "[SQL EVAL] Fast-path exact SQL match | steps=%d reward=%.4f",
                steps,
                reward,
            )
            self._write_sql_log(
                LOG_PATH,
                predicted_sql,
                ground_truth_sql,
                reward,
                ex=1,
                f1=1.0,
            )
            return reward

        def _safe_row(row):
            out = []
            for x in row:
                if x is None:
                    out.append(None)
                    continue
                s = str(x)
                if len(s) > PRINT_MAX_CELL_CHARS:
                    s = s[:PRINT_MAX_CELL_CHARS] + "..."
                out.append(s)
            return out

        def _exec_fetchall_with_hard_timeout(
            cur: sqlite3.Cursor,
            conn: sqlite3.Connection,
            sql: str,
            tag: str,
        ):
            t0 = time.time()
            timer = threading.Timer(QUERY_TIMEOUT_S, conn.interrupt)
            timer.daemon = True
            timer.start()
            try:
                cur.execute(sql)
                rows = cur.fetchall()
            finally:
                timer.cancel()
            dt = time.time() - t0
            return rows, dt

        try:
            conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_MS / 1000.0)
            conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            cur = conn.cursor()

            try:
                pred_rows, pred_dt = _exec_fetchall_with_hard_timeout(
                    cur, conn, predicted_sql, tag="pred"
                )
            except sqlite3.OperationalError as e:
                logger.info(f"[SQL EVAL] Pred SQL failed/timeout | error={e}")
                conn.close()
                return 0.0

            try:
                gt_rows, gt_dt = _exec_fetchall_with_hard_timeout(
                    cur, conn, ground_truth_sql, tag="gt"
                )
            except sqlite3.OperationalError as e:
                logger.info(f"[SQL EVAL] GT SQL failed/timeout | error={e}")
                conn.close()
                return 0.0

            conn.close()

        except Exception as e:
            logger.info(f"[SQL EVAL] Execution failed | error={e}")
            return 0.0

        logger.info("[SQL EVAL] pred_time=%.3fs | gt_time=%.3fs", pred_dt, gt_dt)
        logger.info("[SQL EVAL] pred_rows=%d | gt_rows=%d", len(pred_rows), len(gt_rows))
        logger.info("[SQL EVAL] steps=%d", steps)
        logger.info("[SQL EVAL] predicted_sql=%s", predicted_sql[:PRINT_MAX_SQL_CHARS])
        logger.info("[SQL EVAL] ground_truth_sql=%s", ground_truth_sql[:PRINT_MAX_SQL_CHARS])

        if PRINT_PREVIEW_ROWS > 0:
            for i, r in enumerate(pred_rows[:PRINT_PREVIEW_ROWS]):
                logger.info("[SQL EVAL] pred_row[%d]=%r", i, _safe_row(r))
            if len(pred_rows) > PRINT_PREVIEW_ROWS:
                logger.info("[SQL EVAL] pred_row... (+%d more)", len(pred_rows) - PRINT_PREVIEW_ROWS)

            for i, r in enumerate(gt_rows[:PRINT_PREVIEW_ROWS]):
                logger.info("[SQL EVAL] gt_row[%d]=%r", i, _safe_row(r))
            if len(gt_rows) > PRINT_PREVIEW_ROWS:
                logger.info("[SQL EVAL] gt_row... (+%d more)", len(gt_rows) - PRINT_PREVIEW_ROWS)

        ex = self.calculate_ex(pred_rows, gt_rows)

        if ex == 1:
            reward = 1.0 + EX_BONUS
            logger.info(
                "[SQL EVAL] SQL EXECUTION PERFECT (EX=1) | steps=%d reward=%.4f",
                steps,
                reward,
            )
            self._write_sql_log(
                LOG_PATH,
                predicted_sql,
                ground_truth_sql,
                reward,
                ex=1,
                f1=1.0,
            )
            return reward

        try:
            f1 = self.calculate_f1_score(pred_rows, gt_rows)
        except Exception as e:
            logger.info(f"[SQL EVAL] F1 failed: {e}")
            f1 = 0.0

        f1 = max(0.0, min(1.0, float(f1)))
        reward = f1

        logger.info(
            "[SQL EVAL] EX=0 | F1=%.4f | reward=%.4f | steps=%d | pred_rows=%d gt_rows=%d",
            f1,
            reward,
            steps,
            len(pred_rows),
            len(gt_rows),
        )

        if reward > 0:
            self._write_sql_log(
                LOG_PATH,
                predicted_sql,
                ground_truth_sql,
                reward,
                ex=0,
                f1=f1,
            )

        return reward
    

    '''
    def _evaluate_result(self, result: Dict, db_path: str, ground_truth_sql: str) -> float:
        """
        Official EX + shaped fallback (projection row-F1).
        Adds:
        - SQL execution timeout
        - projection enumeration timeout
        """

        import sqlite3
        import time
        from collections import Counter
        from itertools import combinations, permutations

        # =============================
        # CONFIG
        # =============================
        SQL_TIMEOUT_SECONDS = 10
        MAX_ENUM_SECONDS = 10
        MAX_COL_ENUM = 8

        predicted_sql = (result.get("final_sql") or "").strip()
        if not predicted_sql:
            return 0.0

        # =============================
        # SAFE SQL EXECUTION
        # =============================
        def execute_with_timeout(conn, sql):
            start = time.time()

            def progress_handler():
                if time.time() - start > SQL_TIMEOUT_SECONDS:
                    return 1  # abort query
                return 0

            conn.set_progress_handler(progress_handler, 1000)

            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            desc = cur.description or []

            conn.set_progress_handler(None, 0)
            return rows, desc

        try:
            conn = sqlite3.connect(db_path)

            pred_rows, pred_desc = execute_with_timeout(conn, predicted_sql)
            gt_rows, gt_desc = execute_with_timeout(conn, ground_truth_sql)

            conn.close()

        except Exception as e:
            logger.info(f"[SQL EVAL] SQL execution failed or timeout | {e}")
            return 0.0

        pred_ncol = len(pred_desc)
        gt_ncol = len(gt_desc)

        # =============================
        # 1️⃣ Official EX (multiset)
        # =============================
        if Counter(pred_rows) == Counter(gt_rows):
            return 1.0

        if pred_ncol == 0 or gt_ncol == 0:
            return 0.0

        if pred_ncol < gt_ncol:
            return 0.0

        proj_gt = [r for r in gt_rows if isinstance(r, tuple) and len(r) == gt_ncol]

        def _row_f1(a_rows, b_rows) -> float:
            a = Counter(a_rows)
            b = Counter(b_rows)
            inter = sum((a & b).values())
            na = sum(a.values())
            nb = sum(b.values())
            if na == 0 and nb == 0:
                return 1.0
            if na == 0 or nb == 0:
                return 0.0
            return 2.0 * inter / (na + nb)

        def _project(rows, col_idx_tuple):
            out = []
            for r in rows:
                if isinstance(r, tuple) and len(r) == pred_ncol:
                    out.append(tuple(r[i] for i in col_idx_tuple))
            return out

        candidate_pred_cols = list(range(min(pred_ncol, MAX_COL_ENUM)))

        best = 0.0
        start_enum = time.time()

        for subset in combinations(candidate_pred_cols, gt_ncol):
            if time.time() - start_enum > MAX_ENUM_SECONDS:
                logger.info("[SQL EVAL] Projection enum timeout")
                return 0.0

            for perm in permutations(subset):
                if time.time() - start_enum > MAX_ENUM_SECONDS:
                    logger.info("[SQL EVAL] Projection enum timeout")
                    return 0.0

                proj_pred = _project(pred_rows, perm)
                s = _row_f1(proj_pred, proj_gt)
                if s > best:
                    best = s
                    if best >= 1.0:
                        break
            if best >= 1.0:
                break

        return min(0.99, 0.99 * best)
'''

# --------------------------------------------------
# Helper: build_initial_state with your newer fields
# --------------------------------------------------
def build_initial_state(
    question: str,
    db_path: str,
    schema_metadata_path: str,
    db_id: str = "",
    external_knowledge: str = "",
    metadata_display: str = "",
    max_steps: int = 8,
):
    # assuming your SharedGlobalState dataclass is in scope
    return SharedGlobalState(
        question=question,
        external_knowledge=external_knowledge,
        db_id=db_id,
        db_path=db_path,
        schema_metadata_path=schema_metadata_path,
        metadata_display=metadata_display,
        step=0,
        max_steps=max_steps,
    )



def debug_sql_agent():
    # import pandas as pd
    # df = pd.read_parquet("/efs/sunkexua/src/verl/data/train.parquet").head(3)
    # test_data = cast(List[Dict[str, Any]], df.to_dict(orient="records"))
    """Debug function for testing sql agent with Spider 1.0 (SQLite)."""
    """Load pre-built evaluation data from val_ideal.parquet."""
    import pandas as pd

    PARQUET_PATH = (
        "/efs/open_source_sql_agentic_rl/agent-lightning/"
        "examples/spider2_clean/dev_20240627/train_ideal.parquet"
    )

    df = pd.read_parquet(PARQUET_PATH)
    test_data = df.to_dict(orient="records")

    logger.info(f"Loaded {len(test_data)} samples from val_ideal.parquet")
    #for i, item in enumerate(test_data[:30]):
    #    logger.info(f"[{i}] db={item.get('db_id')} q={item.get('question','')[:50]}...")
    
    '''
    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint="http://localhost:8000/v1",
                model="Qwen/Qwen3-14B",
                sampling_parameters={"temperature": 0.7},
            )
        },
    )
    '''
    
    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=os.environ["OPENAI_API_BASE"],
                model="gpt-4o",
                sampling_parameters={"temperature": 0.7},
            )
        },
    )
    
    trainer.dev(LitAgenticSQL(max_steps=10, debug=True), test_data)


if __name__ == "__main__":
    debug_sql_agent()