# Experiment Rules

This document captures the recurring engineering rules for adding or modifying experiments in `lingbot-va`.
Use it as the default checklist before writing new experiment scripts, profiling code, or result aggregation logic.

## Scope

These rules apply to:

- benchmark scripts under the repository root
- scripts under `experiments/`
- RoboTwin evaluation runs
- profiling / instrumentation code added to `wan_va/`
- result aggregation and plotting code

They exist to avoid repeating the same failures:

- long runs getting killed without resume support
- mixing incompatible timing/profiling numbers
- reusing the same random seeds by accident
- writing results into the wrong filesystem
- breaking existing experiments with debug-only code

## Filesystem Rules

### Raw experiment outputs

Raw run outputs belong under:

- `/nfs_global/S/heguanhua2/lingbot_result/...`

Use NFS for:

- `res.json`
- `latency.json`
- chunk directories
- `.done` markers
- `progress.log`
- raw debug dumps

Reason:

- jobs may run on different nodes
- runs are long-lived and need to survive session loss
- resume logic depends on persistent shared storage

### Plotting / analysis outputs

Plots, summary tables, and derived analysis files belong under the workspace repo, usually:

- `/workspace/S/heguanhua2/wam/lingbot-va/RoboTwin/results/...`

Use workspace for:

- csv copies used for plotting
- png/pdf plots
- markdown reports
- small derived summaries

Reason:

- analysis is code-adjacent
- plots should stay versionable and easy to inspect locally
- large raw experimental outputs should not be mixed into the repo tree

### Do not mix the two roles

Do not:

- write raw experiment chunks into `/workspace/...`
- write plot outputs into `/nfs_global/...` unless there is a specific reason

Keep the split explicit:

- execution data on NFS
- interpretation artifacts in workspace

### Save a git diff with the experiment

For every long-running experiment or A/B compare, save a snapshot of the repository diff into the experiment result root on NFS at launch time.

Recommended pattern:

```bash
git diff > "$BASE_SAVE_ROOT/git_diff.patch"
git status --short > "$BASE_SAVE_ROOT/git_status.txt"
```

Reason:

- later result inspection must be able to recover the exact code delta used for that run
- the diff should travel with the raw experiment artifacts, not only with the working tree
- this is especially important for resume runs, where the repo may continue changing after the first chunk

## Resume / Recovery Rules

### Default requirement

Any experiment expected to run longer than a short smoke test must support resume.

In practice, that means:

- use chunked execution
- write `.done` markers per chunk
- allow rerunning the same top-level script safely

Reason:

- GPU jobs are frequently interrupted
- nodes can be reclaimed
- `SIGTERM` during cleanup is common
- users may need to switch GPU IDs and continue

### Chunking

Default chunk size should be small enough to limit loss on interruption.

Recommended default:

- `chunk_episodes=5` or `10`

Use `5` when:

- episodes are long
- experiment arms are expensive
- the user explicitly wants stronger recovery granularity

Use `10` when:

- each episode is short enough
- overhead from chunk orchestration matters more

### Completion criteria

Do not decide completion from process exit code alone.

A chunk should be considered complete if:

1. `res.json` exists
2. `latency.json` exists
3. `res.json.total_num == run_eps`
4. `latency.json.episode_e2e_sec.count == run_eps`

If those conditions are met:

- mark `.done`
- treat the chunk as complete

Reason:

- benchmark/client cleanup can still exit non-zero after valid results were already written

### GPU portability

Resume state must depend on:

- NFS result directories
- `.done` markers
- result files

It must not depend on:

- original GPU IDs
- original node
- original process IDs

Scripts should support:

- rerunning with new `GPU_IDS`
- continuing unfinished chunks only

## Random Seed Rules

### Core principle

When running chunked experiments, do not accidentally repeat the same seed range in every chunk.

This is a critical rule.

### Required behavior

If the target is `N` episodes across multiple chunks:

- the experiment must cover `N` distinct episode seeds
- unless the script is explicitly designed for repeated-seed stress testing

### Fair A/B comparisons

For compare experiments:

- baseline and experimental arms must use the same episode seed list
- chunk `k` in both arms should correspond to the same seed range

Good:

- arm A chunk 0 uses seeds `[10000..10004]`
- arm B chunk 0 uses seeds `[10000..10004]`

Bad:

- each chunk restarts from `seed=0`
- the same 5 seeds are repeated 10 times and reported as 50 episodes

### Implementation guidance

Prefer one of these:

1. pass an explicit seed offset per chunk
2. pass an explicit `st_seed`
3. record the seed range in `progress.log`

When adding chunked scripts, verify:

- chunk 0 and chunk 1 do not produce the same `stseed-*` directory semantics
- repeated runs are not silently reusing the same seeds

## Profiling Rules

### Distinguish wall time from operator time

Do not mix:

- full RPC wall time
- operator breakdown totals
- block timing totals
- KV timing totals

These are different denominators.

Examples:

- `rpc_infer_sec_mean` is full end-to-end infer wall time for one RPC
- operator breakdown shares are normalized over operator-timed totals, not full RPC wall time
- KV timing totals only cover KV-related instrumented regions

Never derive:

- `absolute_operator_ms = rpc_infer_sec * operator_share`

unless the share is explicitly defined over full RPC wall time.

### Profiling modes

Use distinct experiment types and do not compare them blindly.

1. Real performance runs

Use when the goal is:

- infer latency
- e2e latency
- success rate

Keep profiling mostly off.

2. Light operator profiling

Use when the goal is:

- relative time share across operator groups
