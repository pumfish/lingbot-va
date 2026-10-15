# Control-Aware PTQ 复现说明

本文档说明 `control-aware-ptq` 分支提供的 LingBot-VA 低比特推理与
RoboTwin 评测接口。LingBot-VA 是双仓库复现的主入口；Model-Optimizer
仅作为 DuQuant 后端依赖安装。

## 方法与评测边界

| 方法 | 权重与激活路径 | 所需外部工件 |
| --- | --- | --- |
| `bf16` | 未量化的 BF16 参考 | 无 |
| `uniform_w4a4` | 均匀 ModelOpt RTN W4A4 | ModelOpt checkpoint |
| `duquant_w4a4` | 使用 Wan D5 拓扑的 DuQuant W4A4 | DuQuant checkpoint |
| `omega_qvla_w4a4` | 共享 SVD + Hadamard 旋转，以及离线尺度 W4A4 QDQ | Omega pack |
| `ours_w4a4` | Omega 旋转、在线 token-group Amax W4A4 QDQ、覆盖全部 MDP 的双分支 A4/A8 策略 | Omega pack 与 policy JSON |

本分支中的 W4A4 均采用模拟 fake Q/DQ，后续仍执行浮点 GEMM。因此这些是
精度评测，不是整数内核延迟、吞吐量或显存驻留的测量结果。

## 环境

先按 LingBot-VA 官方 README 安装官方推理依赖。参考版本为 Python 3.10.16、
PyTorch 2.9.0 + CUDA 12.6、Diffusers 0.36.0、Transformers 4.55.2。

随后在相邻目录单独安装与本分支匹配的 ModelOpt 扩展：

```bash
git clone --branch lingbotva-ptq https://github.com/pumfish/Model-Optimizer.git
cd Model-Optimizer
pip install -e .
cd ..
```

RoboTwin 2.0 使用官方仓库提交 `2eeec322` 及其评测依赖，包括
SAPIEN 3.0.0b1。评测前将 `<MODEL_PATH>/transformer/config.json` 的
`"attn_mode"` 设为 `"torch"` 或 `"flashattn"`；`"flex"` 用于训练，
不能用于推理。

## 双仓库快速开始

先克隆 LingBot-VA，再安装匹配的 ModelOpt 分支：

```bash
git clone https://github.com/pumfish/lingbot-va.git
cd lingbot-va
git checkout control-aware-ptq  # 或切换到已包含该分支提交的 main 提交。

git clone --branch lingbotva-ptq https://github.com/pumfish/Model-Optimizer.git ../Model-Optimizer
python -m pip install -e ../Model-Optimizer
```

`lingbot-va` 负责 server、RoboTwin task runner、Omega pack 与 Ours policy；
`Model-Optimizer` 负责恢复 DuQuant checkpoint 所需的自定义后端。所有对比实验
必须使用同一对仓库版本，不能以当前 Model-Optimizer `main` 替换
`lingbotva-ptq`。

## 外部工件

模型权重、checkpoint、pack、policy、数据集和评测输出均不纳入 Git。运行时
显式提供兼容工件：

```text
uniform_w4a4:      --checkpoint /path/to/uniform_w4a4.pt
duquant_w4a4:      --checkpoint /path/to/duquant_w4a4.pt
omega_qvla_w4a4:   --pack /path/to/omega_qvla_w4a4.pt
ours_w4a4:         --pack /path/to/ours_omega_pack.pt --policy /path/to/policy.json
```

Ours policy 是 all-MDP 双分支 cell 策略：action mask 的形状对应 30 个 block、
51 个 action 去噪步和 2 个 action frame；latent mask 对应 30 个 block、26 个
video 去噪步和 2 个 video frame。不得将不同模型、旋转 pack、任务协议或 mask
形状生成的 policy 混用。

## 单任务运行

`experiments/run_robotwin_task.py` 可在非 Slurm 环境使用：它启动本地 LingBot-VA
server，等待服务就绪，运行一个 RoboTwin task，并在客户端退出后终止 server。

先以不使用 GPU 的 dry run 验证路径与方法配置：

```bash
python experiments/run_robotwin_task.py \
  --method ours_w4a4 \
  --task adjust_bottle \
  --model-path /path/to/lingbot-va-robotwin \
  --output-root ./results/ours_w4a4/adjust_bottle \
  --pack /path/to/ours_omega_pack.pt \
  --policy /path/to/ours_policy.json \
  --episodes 100 \
  --dry-run
```

移除 `--dry-run` 即开始评测。需要选择单张 GPU 时使用标准环境变量：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_robotwin_task.py \
  --method bf16 \
  --task adjust_bottle \
  --model-path /path/to/lingbot-va-robotwin \
  --output-root ./results/bf16/adjust_bottle \
  --episodes 100 \
  --robotwin-root /path/to/RoboTwin
```

仅在显存不足时使用 `--enable-offload 1`；它会降低显存占用，但会改变评测速度。

## 多任务与 Slurm

在一台机器上顺序运行多个任务时，可直接给出 task 列表，或提供每行一个 task 的
文本文件：

```bash
python experiments/run_robotwin_campaign.py \
  --method omega_qvla_w4a4 \
  --tasks adjust_bottle put_bottles_dustbin \
  --model-path /path/to/lingbot-va-robotwin \
  --output-root ./results/omega_qvla_w4a4 \
  --pack /path/to/omega_qvla_w4a4.pt \
  --episodes 100
```

可选 Slurm 适配器只提交可移植的单任务 runner，不依赖特定集群布局：

```bash
bash experiments/submit_robotwin_slurm.sh \
  --partition normal --time 1-00:00:00 --gpus 1 --cpus-per-task 8 \
  --setup-script /path/to/activate_environment.sh \
  --log-dir ./slurm_logs -- \
  --method ours_w4a4 --task adjust_bottle \
  --model-path /path/to/lingbot-va-robotwin \
  --output-root ./results/ours_w4a4/adjust_bottle \
  --pack /path/to/ours_omega_pack.pt --policy /path/to/ours_policy.json \
  --episodes 100 --robotwin-root /path/to/RoboTwin
```

## 配对 episode 与断点恢复

做配对比较时，所有方法必须复用同一有序 task seed、episode ID 和冻结的指令。
通过 `--task-seeds`、`--task-episode-ids` 与 `--frozen-episodes` 传入。冻结文件为
JSONL，每行包含 `task_seed` 和 `instruction`。

每个完成 episode 会追加到 `<output-root>/client/episode_results.jsonl`。若中断位置为
完整前缀，则使用相同的 seed、episode ID 和冻结指令，并通过 `--resume-metrics`
传入该 client 目录恢复。配对状态缺失或不匹配会直接报错，runner 不会静默替换 seed
或指令。

## 定向检查

在发起 GPU campaign 前，先运行以下 CPU 定向测试：

```bash
python -m pytest -c /dev/null \
  tests/test_modelopt_checkpoint.py \
  tests/test_uniform_rtn.py \
  tests/test_omega_qvla_rotation.py \
  tests/test_omega_qvla_pack.py \
  tests/test_omega_qvla_activation.py \
  tests/test_omega_qvla_controller.py \
  tests/test_token_group_amax.py \
  tests/test_mixed_policy.py \
  tests/test_duquant_mapping.py \
  tests/test_duquant_checkpoint.py \
  tests/test_robotwin_methods.py \
  tests/test_run_robotwin_task.py \
  tests/test_run_robotwin_campaign.py \
  tests/test_submit_robotwin_slurm.py \
  tests/test_robotwin_evaluation_state.py
```

ModelOpt、上游 LingBot-VA 与 RoboTwin 的许可证分别适用于其代码、模型权重和数据集。
请从官方渠道获取上游资产并遵守相应条款。
