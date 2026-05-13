START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
LOG_DIR='./logs'
mkdir -p $LOG_DIR

save_root='./visualization/'
mkdir -p $save_root
ENABLE_CROSS_KV_CACHE=${ENABLE_CROSS_KV_CACHE:-0}

batch_time=$(date +%Y%m%d_%H%M%S)


for i in {0..7}; do  
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))

    LOG_FILE="${LOG_DIR}/server_${i}_${batch_time}.log"
    echo "[Task ${j}] GPU: ${i} | PORT: ${CURRENT_PORT} | MASTER_PORT: ${CURRENT_MASTER_PORT} | Log: ${LOG_FILE}"

    EXTRA_ARGS=()
    if [ "$ENABLE_CROSS_KV_CACHE" = "1" ]; then
        EXTRA_ARGS+=(--enable-cross-kv-cache)
    fi

    CUDA_VISIBLE_DEVICES=$i  \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        wan_va/wan_va_server.py \
        --config-name robotwin \
        --save_root $save_root \
        --port $CURRENT_PORT \
        "${EXTRA_ARGS[@]}" > $LOG_FILE 2>&1 &
    sleep 2;
done

echo "All 8 instances have been launched in the background."
wait
