START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

save_root=${SAVE_ROOT:-'visualization/'}
mkdir -p "$save_root"

BENCH_ENABLE=${BENCH_ENABLE:-0}
BENCH_MODULES=${BENCH_MODULES:-all}
BENCH_STEP_ENABLE=${BENCH_STEP_ENABLE:-0}
BENCH_STEP_MAX_SAMPLES=${BENCH_STEP_MAX_SAMPLES:-0}
BENCH_KV_ENABLE=${BENCH_KV_ENABLE:-0}
ENABLE_CROSS_KV_CACHE=${ENABLE_CROSS_KV_CACHE:-0}
ATTN_WINDOW=${ATTN_WINDOW:-}
BENCH_ARGS=()
if [ "$BENCH_ENABLE" = "1" ]; then
  BENCH_ARGS+=(--bench-enable --bench-modules "$BENCH_MODULES")
  echo "[launch_server] module benchmark enabled: $BENCH_MODULES"
fi
if [ "$BENCH_STEP_ENABLE" = "1" ]; then
  BENCH_ARGS+=(--bench-step-enable --bench-step-max-samples "$BENCH_STEP_MAX_SAMPLES")
  echo "[launch_server] step benchmark enabled: max_samples=$BENCH_STEP_MAX_SAMPLES"
fi
if [ "$BENCH_KV_ENABLE" = "1" ]; then
  BENCH_ARGS+=(--bench-kv-enable)
  echo "[launch_server] kv benchmark enabled"
fi
if [ -n "$ATTN_WINDOW" ]; then
  BENCH_ARGS+=(--attn-window "$ATTN_WINDOW")
  echo "[launch_server] attn_window override: $ATTN_WINDOW"
fi
if [ "$ENABLE_CROSS_KV_CACHE" = "1" ]; then
  BENCH_ARGS+=(--enable-cross-kv-cache)
  echo "[launch_server] cross-attn KV cache enabled"
fi

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port "$START_PORT" \
    --save_root "$save_root" \
    "${BENCH_ARGS[@]}"
