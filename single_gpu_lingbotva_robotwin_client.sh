#!/bin/bash
set -euo pipefail

SAVE_ROOT=${SAVE_ROOT:-./results/client}
TASK_NAME=${TASK_NAME:-adjust_bottle}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
POLICY_NAME=${POLICY_NAME:-ACT}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-0}
MODEL_NAME=${MODEL_NAME:-0}
SEED=${SEED:-0}
PORT=${PORT:-29056}
TEST_NUM=${TEST_NUM:-10}
VIDEO_GUIDANCE_SCALE=${VIDEO_GUIDANCE_SCALE:-5}
ACTION_GUIDANCE_SCALE=${ACTION_GUIDANCE_SCALE:-1}
SAVE_VISUALIZATION=${SAVE_VISUALIZATION:-0}

PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python -m evaluation.robotwin.eval_polict_client_openpi --config policy/${POLICY_NAME}/deploy_policy.yml \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --train_config_name "${TRAIN_CONFIG_NAME}" \
  --model_name "${MODEL_NAME}" \
  --ckpt_setting "${MODEL_NAME}" \
  --seed "${SEED}" \
  --policy_name "${POLICY_NAME}" \
  --save_root "${SAVE_ROOT}" \
  --video_guidance_scale "${VIDEO_GUIDANCE_SCALE}" \
  --action_guidance_scale "${ACTION_GUIDANCE_SCALE}" \
  --save_visualization "${SAVE_VISUALIZATION}" \
  --test_num "${TEST_NUM}" \
  --port "${PORT}"
