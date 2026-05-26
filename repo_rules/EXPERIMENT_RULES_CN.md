# 实验规则

本文档用于沉淀 `lingbot-va` 仓库内编写、修改和分析实验时应遵守的工程规则。

后续新增实验脚本、profiling 代码、结果聚合逻辑时，默认先对照本文件检查。

这份规则的目标很明确：

- 避免长实验中断后无法恢复
- 避免把不同口径的时间统计混在一起
- 避免 chunk 化实验重复使用相同随机种子
- 避免把结果写到错误的文件系统
- 避免 debug 代码破坏正常实验路径

## 适用范围

这些规则适用于：

- 仓库根目录下的 benchmark / sweep 脚本
- `experiments/` 下的实验脚本
- RoboTwin 评测运行
- `wan_va/` 中新增的 profiling / instrumentation 代码
- 结果汇总与画图分析代码

## 文件系统规则

### 原始实验结果

原始实验输出必须写到：

- `/nfs_global/S/heguanhua2/lingbot_result/...`

NFS 目录用于保存：

- `res.json`
- `latency.json`
- chunk 目录
- `.done` 标记
- `progress.log`
- 原始 debug dump

原因：

- 实验可能在不同节点上继续跑
- 运行周期长，结果必须能跨会话保留
- 恢复逻辑依赖共享持久存储

### 画图与分析结果

绘图、整理表格和派生分析文件应写到 workspace 仓库下，通常是：

- `/workspace/S/heguanhua2/wam/lingbot-va/RoboTwin/results/...`

workspace 目录用于保存：

- 绘图用的 csv 副本
- png/pdf 图
- markdown 报告
- 小型派生汇总结果

原因：

- 分析结果更适合和代码放在一起
- 便于本地查看和版本管理
- 原始大结果不应混入仓库工作目录

### 不要混用职责

不要：

- 把原始 chunk 结果写到 `/workspace/...`
- 把画图产物默认写到 `/nfs_global/...`

约定应始终明确：

- 原始执行结果在 NFS
- 解释性分析产物在 workspace

### 每轮实验保存一份 git diff

每个长实验或 A/B 对照实验在启动时，都应把当前仓库的 diff 保存到该实验的 NFS 结果根目录中。

推荐做法：

```bash
git diff > "$BASE_SAVE_ROOT/git_diff.patch"
git status --short > "$BASE_SAVE_ROOT/git_status.txt"
```

原因：

- 后续看结果时，必须能回溯这轮实验对应的代码改动
- diff 应该和原始实验产物一起保存，而不是只留在工作树里
- 恢复续跑时，仓库代码可能已经继续变化，结果目录里要有最初启动时的代码快照

## 恢复 / 断点续跑规则

### 默认要求

凡是预计运行时间超过简单 smoke test 的实验，都必须支持恢复。

实践上意味着：

- 使用 chunk 化执行
- 每个 chunk 写 `.done`
- 允许重复执行同一顶层脚本继续未完成部分

原因：

- GPU 作业经常被打断
- 节点可能被回收
- `SIGTERM` 在收尾阶段很常见
- 用户可能需要更换 GPU ID 后继续

### Chunk 设计

chunk 大小要小到可以控制中断损失。

推荐默认值：

- `chunk_episodes=5` 或 `10`

适合用 `5` 的情况：

- 单个 episode 很长
- 实验臂比较贵
- 用户明确要求更稳的恢复粒度

适合用 `10` 的情况：

- episode 相对较短
- chunk 调度额外开销更值得控制

### 完成判定

不能只看进程退出码来决定 chunk 是否完成。

chunk 应当在以下条件满足时判定为完成：

1. `res.json` 存在
2. `latency.json` 存在
3. `res.json.total_num == run_eps`
4. `latency.json.episode_e2e_sec.count == run_eps`

若以上成立：

- 写 `.done`
- 将该 chunk 视为有效完成

原因：

- benchmark/client 在结果已经写完后，清理阶段仍可能返回非零退出码

### GPU 可迁移恢复

恢复状态必须依赖：

- NFS 结果目录
- `.done`
- 结果文件

不能依赖：

- 原来的 GPU ID
- 原来的节点
- 原来的进程号

脚本应支持：

- 更换 `GPU_IDS` 后继续跑
- 只继续未完成 chunk

## 随机种子规则

### 核心原则

chunk 化实验不能意外重复使用同一组 seed。

这是硬规则。

### 必须满足的行为

若目标是 `N` 个 episode，拆成多个 chunk 后：

- 整个实验必须覆盖 `N` 个不同的 episode seed
- 除非脚本就是专门为 repeated-seed 压测设计的

### A/B 对照实验

对照实验中：

- baseline 和实验组必须共用同一套 episode seed 列表
- chunk `k` 在两个实验臂中应该对应同一段 seed 区间

正确示例：

- arm A chunk 0 用 seeds `[10000..10004]`
- arm B chunk 0 用 seeds `[10000..10004]`

错误示例：

- 每个 chunk 都从 `seed=0` 重新开始
- 相同 5 个 seed 被重复 10 次，却被当作 50 个 episode

### 实现建议

优先使用以下任一方式：

1. 每个 chunk 显式传不同 `seed offset`
2. 每个 chunk 显式传不同 `st_seed`
3. 在 `progress.log` 中记录 seed 区间

新增 chunk 脚本时必须验证：

- `chunk_000` 和 `chunk_001` 不会复用同样的 seed 起点
- 不会静默重复相同 seeds

## Profiling 规则

### 区分 wall time 和 operator time

不能混用：

- 完整 RPC wall time
- operator breakdown 总时间
- block timing 总时间
- KV timing 总时间

这些统计口径的分母不同。

例如：

- `rpc_infer_sec_mean` 是一次 RPC infer 的完整墙钟时间
- operator breakdown 的 share 是对“被 operator 计时覆盖到的总时间”归一化
- KV timing 只覆盖 KV 相关打点区域

不要直接推导：

- `absolute_operator_ms = rpc_infer_sec * operator_share`

除非这个 share 的分母明确就是完整 RPC wall time。

### Profiling 模式要分清

实验至少要区分以下三类：

1. 真实性能实验

用途：

- infer 延迟
- e2e 延迟
- 成功率

profiling 应尽量关闭。

2. 轻量 operator profiling

用途：
