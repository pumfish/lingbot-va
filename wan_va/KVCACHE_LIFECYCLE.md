# Lingbot KV Cache Lifecycle (Server Inference)

本文档基于当前仓库代码，整理 Lingbot 在 server 推理路径中的 KV cache 生命周期：

- 大小如何计算
- 空间如何分配
- 推理时如何写入
- 满池淘汰策略
- `is_pred` 与真实 KV 的区别与清理
- 跨步/跨阶段 cache 行为
- 每步 token 与 cache 增量大小
- 可能优化方向

关键代码位置：

- `wan_va/wan_va_server.py`
- `wan_va/modules/model.py`
- `wan_va/configs/va_robotwin_cfg.py`
- `wan_va/configs/shared_config.py`

## 1. 生命周期总览

1. Reset 阶段：`clear_cache` 后 `create_empty_cache`，为每层 self-attn 预分配 KV 池。
2. Infer 阶段（视频/动作扩散）：每 step 先生成当前 step 的 K/V，再按 `update_cache` 决定是否持久保留。
3. KV prefill 阶段：先 `clear_pred_cache`，清除预测 KV；然后把观测+动作条件写入持久 KV（`update_cache=2`）。
4. 池满时：按最早 `id` 淘汰（近似 FIFO）。

## 2. Cache 大小计算

在 `VA_Server._reset` 中计算每 chunk 的 token 数：

- `latent_token_per_chunk = (frame_chunk_size * latent_height * latent_width) / (p1*p2*p3)`
- `action_token_per_chunk = frame_chunk_size * action_per_frame`

在 `WanTransformer3DModel.create_empty_cache` 中计算池长：

- `total_tolen = (attn_window//2)*latent_token_per_chunk + (attn_window//2)*action_token_per_chunk`

### 2.1 Robotwin 默认配置示例

配置（`va_robotwin_cfg.py` + `shared_config.py`）：

- `attn_window=72`
- `frame_chunk_size=2`
- `height=256`, `width=320`
- `patch_size=(1,2,2)`
- `action_per_frame=16`
- `env_type=robotwin_tshape`

派生：

- `latent_height=((256//16)*3)//2 = 24`
- `latent_width=320//16 = 20`
- `latent_token_per_chunk = 2*24*20/4 = 240`
- `action_token_per_chunk = 2*16 = 32`
- `total_tolen = 36*240 + 36*32 = 9792`

## 3. 如何分配空间

每个 Transformer block 的 self-attn 都有独立 cache 池，形状为：

- `k: [batch_size, total_tolen, num_heads, head_dim]`
- `v: [batch_size, total_tolen, num_heads, head_dim]`
- `id: [total_tolen]`
- `mask: [total_tolen]`
- `is_pred: [total_tolen]`

说明：

- 这是“每层一份池”，不是全模型共享一份。
- `batch_size` 取 `2`（CFG 开）或 `1`（CFG 关）。

## 4. 推理时如何写入数据

在 `WanAttention.forward`：

1. 计算当前 step 的 `query/key/value`。
2. `key_size = key.shape[1]` 决定本 step 需要的槽位数。
3. `allocate_slots` 找空槽，不足则触发淘汰。
4. 把本 step 的 K/V 写入选中槽位，同时写 `mask/id/is_pred`。
5. 注意力读取时仅使用 `mask=True` 的有效槽位。

## 5. 满池后 KV 如何处理

在 `allocate_slots`：

- 若空槽不足，取当前 `mask=True` 的已用槽位。
- 按 `id` 升序（最早）释放需要数量的槽位。
- 再写入本次 K/V。

即：默认是“按最老 id 淘汰”，不区分视频/动作来源。

## 6. `is_pred` 与真实 KV 的处理

`is_pred` 来源：

- `is_pred = (update_cache == 1)`

三种 `update_cache` 语义：

- `0`：临时写入，仅当前步注意力使用；步末会 `restore_cache` 回滚（不持久）。
- `1`：预测 KV，持久保留，`is_pred=True`。
- `2`：真实/条件 KV（prefill），持久保留，`is_pred=False`。

清理策略：

- `clear_pred_cache` 仅清理 `is_pred=True` 的槽位。
- `clear_cache` 清空整池（reset 时使用）。

## 7. 跨步与跨阶段 Cache 行为

### 7.1 扩散 step 内（视频/动作）

- 中间步通常 `update_cache=0`：写入后回滚，不污染长期池。
- 最后一步 `update_cache=1`：保留预测 KV 供后续阶段使用。

### 7.2 Infer -> KV Prefill 的切换

`_compute_kv_cache` 开头先 `clear_pred_cache`：

- 先去掉上一次 infer 留下的预测 KV。
- 再把当前观测和动作条件以 `update_cache=2` 写入真实 KV。

### 7.3 跨 episode

`reset` 时 `clear_cache + create_empty_cache`，生命周期重置。

## 8. 每一步 cache 大小（token）

### 8.1 视频 step

视频分支 token 数：

- `L_video = (F/p1)*(H/p2)*(W/p3)`

默认 robotwin 数值：

- `F=2, H=24, W=20, p=(1,2,2)`
- `L_video = 240`

### 8.2 动作 step

动作分支 token 数：

- `L_action = F * action_per_frame * 1`

默认 robotwin 数值：

- `F=2, action_per_frame=16`
- `L_action = 32`

### 8.3 单层单步 KV 显存增量（近似）

对某个分支、某个 step：

- `bytes_step_layer ~= 2(K+V) * B * L_step * num_heads * head_dim * dtype_bytes`

全模型可再乘 `num_layers` 估算上界。

### 8.4 精度与字节数（当前实现）

- KV 池 dtype 来自 `param_dtype`，robotwin 默认是 `torch.bfloat16`，即 `dtype_bytes=2`。
- `k/v` 占用是主体；`id/mask/is_pred` 元数据占比很小。

元数据（每层）：

- `id` 是 `int64`（8B）
- `mask` 是 `bool`（1B）
- `is_pred` 是 `bool`（1B）
- 合计：`10 * total_tolen` 字节（robotwin 默认约 `95.62 KiB/层`）

### 8.5 Robotwin 默认配置下的 KV 占用（精确到字节）

已知（来自当前 checkpoint `transformer/config.json` 与代码配置）：

- `num_layers=30`
- `num_heads=24`
- `head_dim=128`
- `dtype=bfloat16 (2B)`

#### 8.5.1 先推导 token 数

1. `latent_height=((height//16)*3)//2=((256//16)*3)//2=24`
2. `latent_width=width//16=320//16=20`
3. `latent_token_per_chunk=(frame_chunk_size*latent_height*latent_width)/(p1*p2*p3)`
   - `=(2*24*20)/(1*2*2)=240`
4. `action_token_per_chunk=frame_chunk_size*action_per_frame=2*16=32`
5. `total_tolen=(attn_window//2)*latent_token_per_chunk + (attn_window//2)*action_token_per_chunk`
   - `=(72//2)*240 + (72//2)*32`
   - `=36*240 + 36*32 = 9792`
6. 每步 token：
   - `L_video=(F/p1)*(H/p2)*(W/p3)=(2/1)*(24/2)*(20/2)=240`
   - `L_action=F*action_per_frame*1=2*16*1=32`

#### 8.5.2 再推导“池容量”字节数

单层持久 KV 池（K+V）公式：

- `bytes_pool_layer = 2(K+V) * B * total_tolen * num_heads * head_dim * dtype_bytes`

代入：

- `bytes_pool_layer = 2 * B * 9792 * 24 * 128 * 2`

当 `B=1`：

- `bytes_pool_layer = 120,324,096 B = 114.75 MiB`
- `bytes_pool_model = 120,324,096 * 30 = 3,609,722,880 B = 3,442.50 MiB = 3.362 GiB`

当 `B=2`：

- `bytes_pool_layer = 240,648,192 B = 229.50 MiB`
- `bytes_pool_model = 240,648,192 * 30 = 7,219,445,760 B = 6,885.00 MiB = 6.724 GiB`

#### 8.5.3 再推导“每步写入增量”字节数

单层每步增量（当前 step 的 K/V）公式：

- `bytes_step_layer = 2(K+V) * B * L_step * num_heads * head_dim * dtype_bytes`

Video step（`L_step=240`）：

- `bytes_step_layer_video = 2 * B * 240 * 24 * 128 * 2`
- `B=1`: `2,949,120 B = 2.81 MiB`，30 层 `84.38 MiB`
- `B=2`: `5,898,240 B = 5.62 MiB`，30 层 `168.75 MiB`

Action step（`L_step=32`）：

- `bytes_step_layer_action = 2 * B * 32 * 24 * 128 * 2`
- `B=1`: `393,216 B = 0.38 MiB`，30 层 `11.25 MiB`
- `B=2`: `786,432 B = 0.75 MiB`，30 层 `22.50 MiB`

#### A) 持久 KV 池总容量（K+V，不含临时张量）

- `B=1`（CFG 关）
  - 每层：`120,324,096 B = 114.75 MiB`
  - 30 层合计：`3,609,722,880 B = 3,442.50 MiB = 3.362 GiB`
- `B=2`（CFG 开）
  - 每层：`240,648,192 B = 229.50 MiB`
  - 30 层合计：`7,219,445,760 B = 6,885.00 MiB = 6.724 GiB`

元数据额外开销：

- 每层：`97,920 B = 95.62 KiB`
- 30 层：`2,937,600 B = 2.80 MiB`

> 结论：元数据可忽略，主要是 K/V 池本体。

#### B) 每步写入规模（按“当前 step 的 K/V 增量”）

`update_cache=1/2` 时可视作新增持久条目；`update_cache=0` 是临时写入后回滚。

- Video step (`L_video=240`)
  - `B=1`：每层 `2,949,120 B = 2.81 MiB`；30 层 `84.38 MiB`
  - `B=2`：每层 `5,898,240 B = 5.62 MiB`；30 层 `168.75 MiB`
- Action step (`L_action=32`)
  - `B=1`：每层 `393,216 B = 0.38 MiB`；30 层 `11.25 MiB`
  - `B=2`：每层 `786,432 B = 0.75 MiB`；30 层 `22.50 MiB`

### 8.6 关于“每一步 cache 大小”的口径说明

- **池容量**：`total_tolen` 对应的预分配总空间（固定上界，reset 后不变）。
- **步增量**：单个 step 的 `L_step` 对应 K/V 写入规模。
- **当前已用**：由 `mask=True` 的槽位数量决定，随 `update_cache` 和清理策略变化。

## 9. 为什么单步时延可能接近（即使 token 差异大）

常见原因：

- 两分支都走完整 block 栈（self-attn + cross-attn + FFN）。
- cache 管理（分配、mask 筛选、写入、回滚）有较大固定成本。
- CFG 开启时，视频/动作都可能扩成 batch=2。
- Python 调度与 scheduler 开销会抬高小 token 分支的相对耗时。

## 10. 可行优化思路

1. 将 CFG 开关拆成视频/动作独立控制，避免 `action_guidance_scale=1` 时动作仍双 batch。
2. 优化 cache 元数据路径（减少 `mask.nonzero`、排序与回滚频率）。
3. 对 `update_cache=0` 路径考虑“无池临时分支”，减少写入+恢复。
4. 视频/动作分池或分配额，降低互相挤占导致的抖动。
5. 控制 prefill 输入长度，避免一次写入过多 token。
6. 针对 kernel 与 dtype 做压测（如 attention backend、BF16/FP16）并配合单步计时开关定位瓶颈。

---

备注：如实现有变更（如 cache 淘汰策略、CFG 逻辑、`update_cache` 语义），请同步更新本文档。
