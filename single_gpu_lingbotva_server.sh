#!/bin/bash
set -euo pipefail

START_PORT=${START_PORT:-29056}
SAVE_ROOT=${SAVE_ROOT:-./results/server}
MODEL_PATH=${MODEL_PATH:-/nfs_global/S/heguanhua2/Models/wam/Lingbot-VA/lingbot-va-posttrain-robotwin}
CONFIG_NAME=${CONFIG_NAME:-robotwin}
ENABLE_OFFLOAD=${ENABLE_OFFLOAD:-0}
SERVER_PYTHON=${SERVER_PYTHON:-python}
MODELOPT_RESTORE_FROM=${MODELOPT_RESTORE_FROM:-}
OMEGA_QVLA_PACK=${OMEGA_QVLA_PACK:-}
OMEGA_QVLA_BRANCHES=${OMEGA_QVLA_BRANCHES:-both}
OURS_MIXED_POLICY=${OURS_MIXED_POLICY:-}

mkdir -p "$SAVE_ROOT"

"$SERVER_PYTHON" wan_va/wan_va_server.py \
  --config-name "$CONFIG_NAME" \
  --model_path "$MODEL_PATH" \
  --enable_offload "$ENABLE_OFFLOAD" \
  --port "$START_PORT" \
  --save_root "$SAVE_ROOT" \
  ${MODELOPT_RESTORE_FROM:+--modelopt_restore_from "$MODELOPT_RESTORE_FROM"} \
  ${OMEGA_QVLA_PACK:+--omega_qvla_pack "$OMEGA_QVLA_PACK"} \
  ${OMEGA_QVLA_PACK:+--omega_qvla_branches "$OMEGA_QVLA_BRANCHES"} \
  ${OURS_MIXED_POLICY:+--ours_mixed_policy "$OURS_MIXED_POLICY"}
