# Control-Aware PTQ Reproduction

This document describes the `control-aware-ptq` branch. It adds reproducible
RoboTwin evaluation interfaces for five LingBot-VA inference arms:

| Method | Weight and activation path | Required artifact |
| --- | --- | --- |
| `bf16` | Unquantized BF16 reference | None |
| `uniform_w4a4` | Uniform ModelOpt RTN W4A4 | ModelOpt checkpoint |
| `duquant_w4a4` | DuQuant W4A4 with the Wan D5 topology | DuQuant checkpoint |
| `omega_qvla_w4a4` | Shared SVD plus Hadamard rotation and offline-scale W4A4 QDQ | Omega pack |
| `ours_w4a4` | Omega rotation, online token-group Amax W4A4 QDQ, and an all-MDP dual-branch A4/A8 policy | Omega pack and policy JSON |

All W4A4 methods in this branch use simulated fake Q/DQ followed by floating
point GEMM. They are accuracy experiments, not measurements of integer-kernel
latency, throughput, or memory residency.

## Environment

Start from the official LingBot-VA inference environment. The reference
versions are Python 3.10.16, PyTorch 2.9.0 with CUDA 12.6, Diffusers 0.36.0,
and Transformers 4.55.2. Install the official dependencies from `README.md`.

Install the corresponding ModelOpt checkout separately, in editable mode:

```bash
git clone --branch lingbotva-ptq https://github.com/pumfish/Model-Optimizer.git
cd Model-Optimizer
pip install -e .
cd ..
```

For RoboTwin 2.0, use the official repository at commit `2eeec322` and its
documented evaluation dependencies, including SAPIEN 3.0.0b1. Set
`<MODEL_PATH>/transformer/config.json` to `"attn_mode": "torch"` or
`"flashattn"` before evaluation. `"flex"` is for training and is not a valid
inference setting.

## Two-Repository Quick Start

Use the two repositories together in this order:

```bash
git clone https://github.com/pumfish/lingbot-va.git
cd lingbot-va
git checkout control-aware-ptq  # Or a main commit that contains this branch.

# Install the matching ModelOpt extension before starting the server.
git clone --branch lingbotva-ptq https://github.com/pumfish/Model-Optimizer.git ../Model-Optimizer
python -m pip install -e ../Model-Optimizer
```

`lingbot-va` owns the server, task runners, Omega packs, and Ours policies.
`Model-Optimizer` supplies the DuQuant custom backend required when a DuQuant
checkpoint is restored. Use the same pair of revisions for every compared arm;
do not substitute the current Model-Optimizer `main` for `lingbotva-ptq`.

## External Artifacts

Model weights, checkpoints, packs, policies, datasets, and evaluation outputs
are deliberately excluded from Git. Supply compatible artifacts explicitly:

```text
uniform_w4a4:      --checkpoint /path/to/uniform_w4a4.pt
duquant_w4a4:      --checkpoint /path/to/duquant_w4a4.pt
omega_qvla_w4a4:   --pack /path/to/omega_qvla_w4a4.pt
ours_w4a4:         --pack /path/to/ours_omega_pack.pt --policy /path/to/policy.json
```

The Ours policy is an all-MDP dual-branch cell policy. Its action mask indexes
30 blocks, 51 action denoise steps, and two action frames. Its latent mask
indexes 30 blocks, 26 video denoise steps, and two video frames. Do not use a
policy produced for a different model, rotation pack, task protocol, or mask
shape.

## Single Task

`experiments/run_robotwin_task.py` is portable and does not require Slurm. It
starts a local LingBot-VA server, waits for it, runs one RoboTwin task, and
terminates the server when the client exits.

First validate paths and the resolved method configuration without a GPU:

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

Run the same command without `--dry-run` for evaluation. Select one GPU with
the standard environment variable when needed:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_robotwin_task.py \
  --method bf16 \
  --task adjust_bottle \
  --model-path /path/to/lingbot-va-robotwin \
  --output-root ./results/bf16/adjust_bottle \
  --episodes 100 \
  --robotwin-root /path/to/RoboTwin
```

Use `--enable-offload 1` only when GPU memory requires it. It can reduce VRAM
usage but changes evaluation speed.

## Campaigns and Slurm

Run tasks sequentially on one machine with either an inline task list or a
text file containing one task per line:

```bash
python experiments/run_robotwin_campaign.py \
  --method omega_qvla_w4a4 \
  --tasks adjust_bottle put_bottles_dustbin \
  --model-path /path/to/lingbot-va-robotwin \
  --output-root ./results/omega_qvla_w4a4 \
  --pack /path/to/omega_qvla_w4a4.pt \
  --episodes 100
```

The optional Slurm adapter only submits the portable single-task runner. It
does not require a particular cluster layout:

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

## Paired Episodes and Recovery

For paired comparisons, reuse the same ordered task seeds, episode IDs, and
frozen instructions for every method. Pass them to the runner with
`--task-seeds`, `--task-episode-ids`, and `--frozen-episodes`. The frozen
episode file is JSONL and contains one object per seed with `task_seed` and
`instruction` fields.

Each completed episode is appended to
`<output-root>/client/episode_results.jsonl`. To resume an interrupted proper
prefix, pass that client directory through `--resume-metrics` together with
the same seed, episode-ID, and frozen-instruction inputs. Mismatched or missing
paired state is a hard error; the runner never silently replaces a seed or
instruction.

## Targeted Checks

The public interfaces have focused CPU tests. Run them before a GPU campaign:

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

The ModelOpt package and the upstream LingBot-VA and RoboTwin licenses govern
their respective code, weights, and datasets. Obtain upstream assets from
their official distribution channels and comply with their terms.
