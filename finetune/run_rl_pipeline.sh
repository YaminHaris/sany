#!/usr/bin/env bash
# Resumable Sandbox Exploration RL (GRPO) pipeline for SANY (qwen-pentest-v4).
# Safe to interrupt at any time with Ctrl+C or `sany-rl pause`; re-running resumes
# from the exact completed exploration mission or GRPO optimizer step.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

OUT="${OUT:-out/lora-pentest-v4}"
SOURCE_MODEL="${SOURCE_MODEL:-qwen-pentest-v3}"
TARGET_MODEL="${TARGET_MODEL:-qwen-pentest-v4}"
INIT_ADAPTER="${INIT_ADAPTER:-out/lora-pentest-v3}"
NUM_TASKS="${NUM_TASKS:-16}"
MAX_STEPS="${MAX_STEPS:-35}"
BACKEND="${BACKEND:-sim}"

mkdir -p "$OUT"
echo "=================================================================="
echo " SANY Sandbox RL (GRPO) Pipeline: $SOURCE_MODEL -> $TARGET_MODEL"
echo " Checkpoint Directory: $OUT"
echo " Tip: Run 'sany-rl ui' to open the Web Control Panel (Pause/Resume)"
echo "=================================================================="

exec "$PY" -u rl_grpo.py \
  --out "$OUT" \
  --source-model "$SOURCE_MODEL" \
  --target-model "$TARGET_MODEL" \
  --init-adapter "$INIT_ADAPTER" \
  --num-tasks "$NUM_TASKS" \
  --max-steps "$MAX_STEPS" \
  --backend "$BACKEND" 2>&1 | tee -a "$OUT/rl_pipeline.log"
