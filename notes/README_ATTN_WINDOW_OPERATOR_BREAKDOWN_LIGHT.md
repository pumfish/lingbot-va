# Attn-window Operator Breakdown Light

这组实验专门用于看 **infer 内部算子的相对关系**，重点关注：

- 单次平均时间：`avg = total_ms / calls`
- 累计总时间：`total_ms`

目标不是得到“无 profiling 干预下的真实绝对时间”，而是尽量轻量地比较：

- `self` 分支
  - `self_q_proj`
  - `self_kv_proj`
  - `self_qk_norm`
  - `self_rope`
  - `self_kv_cache_io`
  - `self_attn_core`
  - `self_out_proj`
- `cross` 分支
  - `cross_q_proj`
  - `cross_kv_proj`
  - `cross_qk_norm`
  - `cross_attn_core`
  - `cross_out_proj`
- block 级
  - `ffn`
  - `aux`

## 1. 为什么这版更轻

默认配置：

- `BENCH_ENABLE=0`
- `BENCH_STEP_ENABLE=0`
- `BENCH_MEM_ENABLE=0`
- `BENCH_KV_ENABLE=1`
- `BENCH_BLOCK_ENABLE=1`
- `BENCH_FLOPS_ENABLE=0`
- `BENCH_TIMING_ENABLE=1`
- `BENCH_SELF_ATTN_MICRO_ENABLE=1`
- `BENCH_SELF_ATTN_MICRO_SAMPLE_EVERY=10`

轻量化手段：

1. 不做 per-step denoise timing
2. 不做 mem profiling
3. self/cross attention 内部时间采用抽样
4. 输出主口径为 `total_ms / calls`，而不是追求完全闭合

## 2. 如何运行

先跑 20 episode，确认趋势与字段稳定：

```bash
cd /workspace/S/heguanhua2/wam/lingbot-va
./sweep_attn_window_operator_breakdown_light_parallel.sh 20
```

若需要更稳，再跑 30 episode：

```bash
./sweep_attn_window_operator_breakdown_light_parallel.sh 30
```

默认输出目录：

- `/nfs_global/S/heguanhua2/lingbot_result/attn_window_operator_breakdown_light`

默认窗口：

- `attn_window={60,36,16,4}`

默认任务：

- `put_bottles_dustbin`

## 3. 统计口径

结果写入：

- `latency.json -> operator_breakdown_ms_rpc_infer_only`

每个算子组都有：

- `count`
  - 该算子组总调用次数
- `total`
  - 该算子组累计总时间，单位 ms
- `mean`
  - 严格按 `total / count` 计算的单次平均时间
- `sampled_count`
  - 实际被采样到的次数
- `sample_every`
  - 每多少次调用采样 1 次

说明：

- `self_q_proj` 与 `cross_q_proj` 是分开统计的，不会混在一起
- `self_kv_cache_io` 只属于 self，因为 cross 不走 KV cache
- `self_rope` 只属于 self，因为 cross 不使用 RoPE

## 4. 各算子组如何从代码映射过来

文件：

- [model.py](/workspace/S/heguanhua2/wam/lingbot-va/wan_va/modules/model.py)
- [wan_va_server.py](/workspace/S/heguanhua2/wam/lingbot-va/wan_va/wan_va_server.py)

映射关系：

- `self_q_proj`
  - `self_attn.to_q`
- `self_kv_proj`
  - `self_attn.to_k + self_attn.to_v`
- `self_qk_norm`
  - `self_attn.norm_q + self_attn.norm_k + unflat_q + unflat_k + unflat_v`
- `self_rope`
  - `self_attn.rope`
- `self_kv_cache_io`
  - `kv_update_store + kv_gather_read + kv_restore_temp`
- `self_attn_core`
  - `attn_op`，优先取 KV 路径里的 `attn_compute_ms_total`
- `self_out_proj`
  - `flatten_type + to_out0 + to_out1`

- `cross_q_proj`
  - `cross_attn.to_q`
- `cross_kv_proj`
  - `cross_attn.to_k + cross_attn.to_v`
- `cross_qk_norm`
  - `cross_attn.norm_q + cross_attn.norm_k + unflat_q + unflat_k + unflat_v`
- `cross_attn_core`
  - `cross_attn.attn_op`
- `cross_out_proj`
  - `flatten_type + to_out0 + to_out1`

- `ffn`
  - block 内 FFN 整段
- `aux`
  - `temb_mod + self/cross/ffn prep + self/cross/ffn residual`

## 5. CSV 输出

聚合脚本会额外写入：

- `summary_chunk_level.csv`
- `summary_window_level.csv`

对应列前缀：

- `op_overall_self_q_proj_*`
- `op_overall_self_kv_proj_*`
- ...

字段包括：

- `_count`
- `_total`
- `_mean`
- `_sampled_count`
- `_sample_every`

其中：

- `_total` 用来看累计时间
- `_mean` 用来看单次平均时间

## 6. 推荐画图

仓库内一键脚本（读 `summary_window_level.csv`，默认输出到 `RoboTwin/results/attn_window_operator_breakdown_light_plots/`）：

```bash
cd /workspace/S/heguanhua2/wam/lingbot-va
python3 RoboTwin/tools/plot_attn_window_operator_breakdown_light.py
```

与 **§3 的 14 个算子组** 对齐的主图：

- **`bar_operator_breakdown_14_groups_share.png`**：各 `attn_window` 下，`op_overall_<组>_total` 在 **14 组总和** 中的占比堆叠柱（组名与 §4 一致：`self_q_proj` … `aux`）；**图例在柱图右侧**；图题不写 aux 字样，图底脚注说明最后一组桶含义。

- **`bar_operator_breakdown_13_groups_share_no_aux.png`**：分母为其余 **13 组 `op_overall_*_total` 之和**，柱内占比之和为 **1**；**图例在柱图右侧**；图题不写 aux 字样，无脚注。

- **`bar_operator_breakdown_13_groups_share_no_aux_sim_zero_cross_kv_proj.png`**：在上一张相同口径下，假设 **`cross_kv_proj` 累计时间为 0**（完全优化掉），其余算子 ms 不变，再按 **新分母** 重算占比（仍 13 段堆叠，`cross_kv_proj` 柱高为 0）；**`self_kv_cache_io`** 仍带 **KV I/O 占比** 标注。数值见 `share_overall_operator_13_no_aux_sim_zero_cross_kv_proj_by_window.csv`。

与 **单次 block 前向计时边界** 对齐的辅图（**10 步**，不把 `temb_mod/各 prep/各 residual` 合并成一个大 aux）：

- **`bar_overall_block_10_step_share.png`**：`bd_overall_block_*` 的 `temb_mod → self_prep → self_attn → self_residual → cross_prep → cross_attn → cross_residual → ffn_prep → ffn → ffn_residual`，在 **10 段总和** 中的占比。

其它补充图（KV 子项、micro、video/action self 拆分）：同目录下 `bar_overall_self_attn_sub_share.png`、`bar_overall_self_attn_micro_share.png`、`bar_self_attn_video_vs_action_share.png`。

占比表：`share_overall_operator_14_by_window.csv`、`share_overall_operator_13_no_aux_by_window.csv`、`share_overall_operator_13_no_aux_sim_zero_cross_kv_proj_by_window.csv`、`share_overall_block_10_step_by_window.csv`；补充列见 `supplement_self_attn_video_action_sub_micro.csv`。

若还要对比 **mean** 或 **total** 随窗口的折线，可在上述 CSV 列基础上另画（`op_overall_*_mean` / `_total` 已在 `summary_window_level.csv`）。

## 7. 注意事项

这组实验适合回答：

- 不同 `attn_window` 下，哪个算子组相对更重
- `avg` 和 `total` 的排序是否一致
- 时间变化主要来自 self 还是 cross

这组实验不适合直接回答：

- “真实无 profiling 干预下的单次 infer 绝对时间是多少”

如果你要看真实端到端时间，仍应结合更轻量的 latency bench。 
