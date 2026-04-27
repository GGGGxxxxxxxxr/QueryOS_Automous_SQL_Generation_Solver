# Copyright (c) Microsoft. All rights reserved.

"""Train an SQL agent on the Spider dataset using Agent-lightning.

This module provides a training script for SQL agents using different model configurations.
The script supports three different training configurations:

1. 'fast' - A lightweight configuration optimized for CI testing with reduced epochs
2. 'qwen' - Standard configuration using Qwen-2.5-Coder-1.5B-Instruct model
3. 'llama' - Configuration using LLaMA-3.2-1B-Instruct model with JSON formatting

Usage:
    python train_sql_agent.py fast    # Fast training for CI/testing
    python train_sql_agent.py qwen    # Standard Qwen model training
    python train_sql_agent.py llama   # LLaMA model training

The script uses reinforcement learning with VERL framework
to train agents on the Spider dataset for text-to-SQL generation tasks.
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
sys.path.append(os.path.dirname(__file__))
from sql_agent import LitAgenticSQL

import agentlightning as agl

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "/efs/open_source_sql_agentic_rl/agent-lightning/examples/spider2_clean/train/train_3500_verified.parquet",
        "val_files": "/efs/open_source_sql_agentic_rl/agent-lightning/examples/spider2_clean/train/val_200_verified_20260322_155343.parquet",
        "train_batch_size": 8,
        "max_prompt_length": 32768,
        "max_response_length": 32768,
        "truncation": "truncate",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 8,
            "log_prob_micro_batch_size_per_gpu": 1,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "free_cache_engine": True,
            "gpu_memory_utilization": 0.75,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "enforce_eager": True,
                    "tool_call_parser": "hermes",
                    "rope-scaling":'{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}'
                },
            },
        },
        "actor": {
            "ppo_mini_batch_size": 64,
            "use_dynamic_bsz": True,
            "ppo_epochs": 1,
            "ppo_micro_batch_size_per_gpu": 1,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "ppo_max_token_len_per_gpu": 32768,
            "kl_loss_coef": 0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "ulysses_sequence_parallel_size": 2,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 1,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen3-8B",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
            "override_config": {
                "max_position_embeddings": 65536,
            }
        },
    },
    "trainer": {
        "n_gpus_per_node": 8,
        "val_before_train": True,
        "balance_batch": False,
        "critic_warmup": 0,
        "logger": ["console", "tensorboard", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "bird_simpletir_3500_complex_newtrail_verified_123",
        "default_local_dir": '/efs/open_source_sql_agentic_rl/agent-lightning/examples/spider2_clean/train',
        "nnodes": 4,
        "test_freq": 10,
        "save_freq": 2,
        "total_epochs": 1,
    },
}


def config_train_fast() -> Dict[str, Any]:
    """A fast training run for CI testing purposes."""

    # `EXPERIMENT_NAME="spider_$(date +%Y%m%d%H%M%S)"`
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"spider_{timestamp}"

    # `PROJECT_NAME=AgentLightningCI`
    PROJECT_NAME = "AgentLightningCI"

    # Simulate writing to $GITHUB_OUTPUT if it’s set
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"project_name={PROJECT_NAME}\n")
            f.write(f"run_name={EXPERIMENT_NAME}\n")

    print("Set environment variables:")
    print(f"PROJECT_NAME={PROJECT_NAME}")
    print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    config["data"]["val_files"] = "data/test_dev.parquet"
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["experiment_name"] = EXPERIMENT_NAME
    config["trainer"]["project_name"] = PROJECT_NAME
    config["trainer"]["test_freq"] = 1
    return config


def config_train_qwen() -> Dict[str, Any]:
    """A configuration for training with Qwen-2.5B."""

    config = deepcopy(RL_TRAINING_CONFIG)
    return config


def config_train_npu() -> Dict[str, Any]:
    """A configuration for training with NPU."""

    config = deepcopy(RL_TRAINING_CONFIG)
    del config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["enable_auto_tool_choice"]
    del config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"]
    del config["trainer"]["logger"][1]
    config["actor_rollout_ref"]["actor"]["use_torch_compile"] = False
    config["trainer"]["val_before_train"] = False
    config["trainer"]["save_freq"] = 256
    config["trainer"]["device"] = "npu"
    return config


def config_train_llama() -> Dict[str, Any]:
    """A configuration for training with LLaMA-3.2-1B-Instruct.

    You will need a `HF_TOKEN` set to run with this config.
    """

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] = "llama3_json"
    config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"] = "llama3_json"
    config["actor_rollout_ref"]["model"]["path"] = "meta-llama/Llama-3.2-1B-Instruct"
    return config


def train(config: Dict[str, Any], active_agent: Optional[str]) -> None:
    """Train the SQL agent with the given configuration."""

    agent = LitAgenticSQL()
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=40, algorithm=algorithm, adapter={"agent_match": active_agent})
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore

    train_data = pd.read_parquet(config["data"]["train_files"]).to_dict(orient="records")  # type: ignore
    val_data = pd.read_parquet(config["data"]["val_files"]).to_dict(orient="records")  # type: ignore
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore


def main() -> None:
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train an SQL agent on the Spider dataset using different model configurations"
    )

    parser.add_argument(
        "config",
        choices=["fast", "qwen", "llama", "npu"],
        help="Training configuration: 'fast' (CI testing), 'qwen' (Qwen-2.5-Coder-1.5B), 'llama' (LLaMA-3.2-3B),'npu' (Train with NPU)",
    )

    parser.add_argument(
        "--active-agent", type=str, help="Override the active agent name (default: auto-generated based on config)"
    )

    args = parser.parse_args()

    # Get the appropriate configuration
    config_functions = {
        "fast": config_train_fast,
        "qwen": config_train_qwen,
        "llama": config_train_llama,
        "npu": config_train_npu,
    }
    config = config_functions[args.config]()

    # Set active agent - use provided value or default based on config choice
    active_agent = args.active_agent

    print(f"Starting training with '{args.config}' configuration...")
    print(f"Active agent: {active_agent}")

    train(config, active_agent)


if __name__ == "__main__":
    main()
