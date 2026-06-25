#!/bin/bash
set -euo pipefail

TASK_NAME=${TASK_NAME:-adjust_bottle}
SEED=${SEED:-0}
TEST_NUM=${TEST_NUM:-10}
PORT=${PORT:-29056}
SERVER_ENTRY=${SERVER_ENTRY:-./single_gpu_lingbotva_server.sh}
SERVER_SAVE_ROOT=${SERVER_SAVE_ROOT:-./results/server}
CLIENT_SAVE_ROOT=${CLIENT_SAVE_ROOT:-./results/client}
LOG_DIR=${LOG_DIR:-./logs}
ENABLE_OFFLOAD=${ENABLE_OFFLOAD:-0}
SAVE_VISUALIZATION=${SAVE_VISUALIZATION:-0}

mkdir -p "$SERVER_SAVE_ROOT" "$CLIENT_SAVE_ROOT" "$LOG_DIR"

timestamp=$(date +%Y%m%d_%H%M%S)
server_log="$LOG_DIR/server_${timestamp}_p${PORT}.log"
client_log="$LOG_DIR/client_${timestamp}_p${PORT}.log"
SERVER_PID=""

terminate_pid_and_group() {
  local pid=$1
  local sig=${2:-TERM}
  if [[ -z "${pid:-}" ]]; then
    return 0
  fi
  kill "-$sig" -- "-$pid" 2>/dev/null || true
  pkill "-$sig" -P "$pid" 2>/dev/null || true
  kill "-$sig" "$pid" 2>/dev/null || true
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    terminate_pid_and_group "$SERVER_PID" TERM
    sleep 2
    terminate_pid_and_group "$SERVER_PID" KILL
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

handle_interrupt() {
  echo "[single_gpu_lingbotva_benchmark] interrupted, cleaning up server process group"
  cleanup
  exit 130
}

trap cleanup EXIT
trap handle_interrupt INT TERM

echo "[single_gpu_lingbotva_benchmark] server_log=$server_log"
echo "[single_gpu_lingbotva_benchmark] client_log=$client_log"

env START_PORT="$PORT" SAVE_ROOT="$SERVER_SAVE_ROOT" MODEL_PATH="${MODEL_PATH:-}" CONFIG_NAME="${CONFIG_NAME:-robotwin}" ENABLE_OFFLOAD="$ENABLE_OFFLOAD" \
  setsid bash "$SERVER_ENTRY" >"$server_log" 2>&1 &
SERVER_PID=$!

sleep 8
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "[single_gpu_lingbotva_benchmark] server exited early"
  exit 1
fi

env SAVE_ROOT="$CLIENT_SAVE_ROOT" TASK_NAME="$TASK_NAME" SEED="$SEED" PORT="$PORT" TEST_NUM="$TEST_NUM" \
  SAVE_VISUALIZATION="$SAVE_VISUALIZATION" \
  bash ./single_gpu_lingbotva_robotwin_client.sh >"$client_log" 2>&1

cleanup
trap - EXIT INT TERM
