# KV Benchmark Guide (for `kv-benchmark-instrumentation`)

This document explains what was changed for KV profiling, how to run it, and how to read results.

Target audience: first-time users of this repo.

## 1. What this feature adds

The original module benchmark can measure:
- `obs_preprocess`
- `vae_to_latent`
- `text_encoder`
- `kv_prefill`
- `video_diffusion`
- `action_diffusion`
- `action_postprocess`

This KV extension adds **fine-grained KV timing + counters** during inference:
- KV prefill split:
  - `kv_prepare_input_ms`
  - `kv_prefill_latent_forward_ms`
  - `kv_prefill_action_forward_ms`
  - `kv_prefill_total_forward_ms`
- Attention-cache inner costs:
  - `kv_update_store_ms_total` (write/update cache)
  - `kv_gather_read_ms_total` (read/gather cache)
  - `kv_restore_temp_ms_total` (restore temp cache for `update_cache==0`)
- Token and eviction stats:
  - `self_attn_calls`
  - `update_cache_0_calls / 1_calls / 2_calls`
  - `avg_new_tokens`
  - `avg_valid_tokens`
  - `avg_evict_tokens`

And writes ratio summary in `latency.json`:
- `kv_strict_over_diffusion`
- `kv_with_prefill_over_diffusion`
- `kv_strict_over_e2e`
- `kv_with_prefill_over_e2e`

## 2. Files changed (KV feature)

Core implementation:
- `wan_va/modules/model.py`
  - adds WanAttention KV profiler and stage-based accumulation.
- `wan_va/wan_va_server.py`
  - adds request-level KV stats collection.
  - adds CLI flag: `--bench-kv-enable`.

Aggregation/output:
- `evaluation/robotwin/eval_polict_client_openpi.py`
  - collects `kv_timing` from server responses.
  - writes `kv_timing_ms` and `kv_ratio` in `latency.json`.

Launch wiring:
- `evaluation/robotwin/launch_server.sh`
  - forwards `BENCH_KV_ENABLE=1` to `--bench-kv-enable`.

If you use local helper scripts in this workspace, these are also wired:
- `single_gpu_server.sh`
- `single_gpu_benchmark.sh`
- `sweep_attn_window_single_gpu.sh`

## 3. How to run

## 3.1 Quick run (single task)

```bash
BENCH_ENABLE=1 BENCH_STEP_ENABLE=1 BENCH_KV_ENABLE=1 \
./single_gpu_benchmark.sh adjust_bottle \
/workspace/S/heguanhua2/wam/lingbot-va/RoboTwin/results/kv_bench_demo 10
```

## 3.2 `attn_window` sweep with KV profiling

```bash
BENCH_ENABLE=1 BENCH_STEP_ENABLE=1 BENCH_KV_ENABLE=1 \
WINDOWS="72 24 8 4" \
TASKS="put_bottles_dustbin" \
./sweep_attn_window_single_gpu.sh \
/workspace/S/heguanhua2/wam/lingbot-va/RoboTwin/results/attn_window_kv_sweep 10
```

## 3.3 If you launch server manually

Set env and run server launcher:

```bash
BENCH_KV_ENABLE=1 BENCH_ENABLE=1 BENCH_STEP_ENABLE=1 \
bash evaluation/robotwin/launch_server.sh
```

## 4. Where results are saved

Per task `latency.json` path (example):

```text
RoboTwin/results/<your_run>/stseed-10000/metrics/<task>/latency.json
```

Inside `latency.json`, KV-related sections are:
- `kv_timing_ms`
- `kv_ratio`

## 5. How to read `kv_ratio`

`kv_ratio` has two views:

1. strict KV only
- uses cache inner ops only: store + gather + restore
- keys:
  - `kv_strict_over_diffusion`
  - `kv_strict_over_e2e`

2. strict KV + prefill
- strict KV + `kv_prefill_total_forward_ms`
- keys:
  - `kv_with_prefill_over_diffusion`
  - `kv_with_prefill_over_e2e`

Suggested interpretation:
- If `kv_strict_over_diffusion` is high, optimize cache read/write path first.
- If strict is low but with-prefill is high, optimize prefill path first.

## 6. Recommended experiment matrix

For reliable conclusions:
- Tasks:
  - short: `adjust_bottle`
  - long: `put_bottles_dustbin`, `open_microwave`
- Windows: `72, 24, 8, 4`
- Repeat: `test_num=10`
- Compare both:
  - success rate
  - `kv_ratio` and `kv_timing_ms`

## 7. Common pitfalls

1. `BENCH_KV_ENABLE` not set
- symptom: no `kv_timing_ms`/`kv_ratio` in `latency.json`
- fix: set `BENCH_KV_ENABLE=1`

2. Wrong output root confusion
- always use absolute save path to avoid cwd-related ambiguity.

3. First run is slower
- warmup/JIT/cuda graph effects can inflate first samples.
- compare by mean/p50/p90, not a single episode.

## 8. Minimal checklist before analyzing data

- Confirm log shows KV benchmark enabled.
- Confirm `latency.json` contains:
  - `module_latency_ms`
  - `step_latency_ms`
  - `kv_timing_ms`
  - `kv_ratio`
- Confirm `success_rate` is not degenerate (all 0 or all 1 for all settings).


## 9. 中文术语对照与统计口径

为避免多人分析时口径不一致，建议统一使用下面定义。

术语对照：
- `strict KV`：仅统计 KV 内部缓存操作时间（store + gather + restore）。
- `with_prefill`：`strict KV` + prefill 前向时间（latent/action prefill）。
- `over_diffusion`：用 diffusion 总时间做分母。
- `over_e2e`：用 episode 端到端总时间做分母。

对应到 `latency.json -> kv_ratio`：
- `kv_strict_over_diffusion`
- `kv_with_prefill_over_diffusion`
- `kv_strict_over_e2e`
- `kv_with_prefill_over_e2e`

计算口径（单位统一为 ms 后再求比例）：
- `kv_strict_total_ms = kv_update_store_ms_total + kv_gather_read_ms_total + kv_restore_temp_ms_total`
- `kv_with_prefill_total_ms = kv_strict_total_ms + kv_prefill_total_forward_ms`
- `diffusion_total_ms = module_latency_ms.video_diffusion.total + module_latency_ms.action_diffusion.total`
- `episode_e2e_total_ms = episode_e2e_sec.total * 1000`

比例公式：
- `kv_strict_over_diffusion = kv_strict_total_ms / diffusion_total_ms`
- `kv_with_prefill_over_diffusion = kv_with_prefill_total_ms / diffusion_total_ms`
- `kv_strict_over_e2e = kv_strict_total_ms / episode_e2e_total_ms`
- `kv_with_prefill_over_e2e = kv_with_prefill_total_ms / episode_e2e_total_ms`

解读建议：
- `kv_strict_over_diffusion` 高：优先优化 cache 读写路径（layout/gather/restore）。
- `kv_strict_over_diffusion` 低但 `kv_with_prefill_over_diffusion` 高：优先优化 prefill 前向路径。
- 比例变化不明显时，优先结合 `avg_valid_tokens`、`avg_evict_tokens` 一起判断是否真正触发了窗口差异。
