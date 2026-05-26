# RoboTwin Baseline Experiment (LingBot-VA)

This setup is optimized for **4xA100-80G / 6-hour windows** and supports **incremental, resumable baseline runs**.

Metrics produced:
- success rate (`res.json`)
- latency (`latency.json`): `episode_e2e_sec`, `rpc_infer_sec`, `rpc_reset_sec`, `rpc_kv_sec`

## 1) Start servers (recommended 4 workers)

```bash
NUM_GPUS=4 NUM_SERVER_WORKERS=4 START_PORT=29556 MASTER_PORT=29661 \
  bash evaluation/robotwin/launch_server_multigpus.sh
```

## 2) Incremental client baseline runs

Run only a small chunk each session (example: 4 tasks, 10 episodes/task):

```bash
NUM_WORKERS=4 NUM_GPUS=4 START_PORT=29556 \
  bash evaluation/robotwin/run_baseline_50tasks.sh results_baseline 10 0 4
```

Run next chunk in a later session with resume:
- keep same `SAVE_ROOT`, `TEST_NUM`, `SEED`
- script auto-skips tasks already reaching `total_num >= TEST_NUM` when `SKIP_DONE=1` (default)

```bash
NUM_WORKERS=4 NUM_GPUS=4 START_PORT=29556 SKIP_DONE=1 \
  bash evaluation/robotwin/run_baseline_50tasks.sh results_baseline 10 0 4
```

Optional controls:
- `TASK_OFFSET=K`: skip first K tasks in canonical list.
- `TASKS_OVERRIDE="task_a,task_b"`: run a custom subset.
- `SUMMARY_EACH_BATCH=1` (default): refresh summary after each batch.

## 3) Summarize current partial/final baseline

```bash
python -m evaluation.robotwin.summarize_baseline \
  --results-root RoboTwin/results_baseline
```

Outputs:
- `RoboTwin/results_baseline/summary/baseline_summary.csv`
- `RoboTwin/results_baseline/summary/baseline_summary.md`

## Practical recommendation for your cluster window

- Per 6-hour session: start with `MAX_TASKS=4` (or 8 if previous run proves stable).
- Keep parameters fixed across sessions for comparability.
- Always collect summary at the end of each session to track progress.
