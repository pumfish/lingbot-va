# Repository Guidelines

## Project Structure & Module Organization
- `wan_va/`: core model/inference server code (`wan_va_server.py`, configs, modules, dataset helpers).
- `evaluation/robotwin/`: RoboTwin evaluation entrypoints and launch scripts (`launch_server*.sh`, `launch_client*.sh`, `eval_polict_client_openpi.py`).
- `script/`: utility launchers (for example synchronized VA server startup).
- `example/`: runnable demos for `robotwin`, `franka`, and generic demo inputs.
- `RoboTwin/`: RoboTwin simulation dependency tree used by evaluation (treat as external integration code unless task requires edits).
- `assets/`, `visualization/`, `logs/`: media, generated outputs, and runtime logs.

## Cluster Rules
- For any code written from this SLURM login-node workflow, consult [SLURM_CLUSTER_AGENT_RULES.md](/workspace/S/heguanhua2/wam/lingbot-va/repo_rules/SLURM_CLUSTER_AGENT_RULES.md) first.
- Treat `SLURM_CLUSTER_AGENT_RULES.md` as the default cross-repo checklist for login-node vs compute-node behavior, NFS persistence, resume support, seed scheduling, and CUDA build constraints.

## Experiment Rules
- For any new experiment script, profiling change, result aggregation, or A/B evaluation workflow, consult [EXPERIMENT_RULES.md](/workspace/S/heguanhua2/wam/lingbot-va/repo_rules/EXPERIMENT_RULES.md) first.
- Treat `EXPERIMENT_RULES.md` as the default operational checklist for:
  - resume / recovery support,
  - NFS vs workspace result placement,
  - chunk sizing and `.done` semantics,
  - random seed scheduling across chunks and compare arms,
  - profiling denominator hygiene,
  - debug instrumentation safety.
- When launching a new experiment, also save a snapshot of the current repository diff into that experiment's result root on NFS (for example `git diff > $BASE_SAVE_ROOT/git_diff.patch`) so the run is self-describing and reproducible.

## Build, Test, and Development Commands
- Install runtime deps:
```bash
pip install -r requirements.txt
```
- Install package in editable mode:
```bash
pip install -e .
```
- Launch RoboTwin inference server:
```bash
bash evaluation/robotwin/launch_server.sh
# or multi-GPU
bash evaluation/robotwin/launch_server_multigpus.sh
```
- Launch RoboTwin client:
```bash
bash evaluation/robotwin/launch_client.sh results/ adjust_bottle
bash evaluation/robotwin/launch_client_multigpus.sh results/ 0
```
- Format code (repo-provided target):
```bash
make format
```

## Coding Style & Naming Conventions
- Python 3.10+; prefer 4-space indentation and PEP 8 naming (`snake_case` for functions/files, `PascalCase` for classes).
- Use Black-compatible style (`line-length = 88`) and `isort` with Black profile (configured in `pyproject.toml`).
- Keep shell scripts POSIX/Bash-safe and parameterized via env vars (`NUM_GPUS`, `START_PORT`, `MASTER_PORT`).

## Testing Guidelines
- Primary validation is integration-style:
  - `python -m evaluation.robotwin.test_render` (render stack sanity)
  - single-task client/server smoke run before multi-GPU runs.
- If adding unit tests, use `pytest` and name files `test_*.py` under a `tests/` folder or module-local test files.
- For evaluation changes, include reproducible command lines and expected log/output paths in PR notes.

## Commit & Pull Request Guidelines
- Follow existing commit style: short, imperative, lower-case summaries (examples: `fix bug of TRITON_MAX_BLOCK`, `add new training demo`).
- Keep commits focused; avoid bundling refactors with behavior changes.
- PRs should include:
  - what changed and why,
  - exact verification commands run,
  - linked issue/task,
  - screenshots/log snippets for evaluation or visualization changes.

## Environment & Configuration Tips
- Run evaluation on compute nodes with GPUs; login nodes may fail renderer initialization.
- Ensure `attn_mode` in model `transformer/config.json` matches phase: `flex` for training, `torch`/`flashattn` for inference.
