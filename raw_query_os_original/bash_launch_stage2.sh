set -x

HEAD_ADDR="http://172.31.11.9:8265"

ray job submit --address=http://172.31.11.9:8265 \
  --runtime-env=runtime_env.yaml \
  --no-wait -- \
  bash -lc '
    cd /efs/open_source_sql_agentic_rl/agent-lightning/examples/spider2_clean &&
    /efs/open_source_sql_agentic_rl/agent-lightning/.venv/bin/python train_sql_agent.py qwen
  '