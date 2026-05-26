# SLURM Cluster Agent Rules

This document defines the default engineering rules for any coding agent working
on this SLURM-based cluster from a login node.

It is intended to be repository-agnostic and reusable across projects such as:

- `lingbot-va`
- `Model-Optimizer`
- other repos developed from the same login-node workflow

---

## 1. Execution Context

### 1.1 Where the agent runs

Assume the coding agent runs on a **login node** by default.

Implications:

- the login node may have **no visible GPU**
- CUDA runtime availability on the login node is not a reliable indicator of
  what is available on compute nodes
- GPU inference, profiling, compilation of CUDA extensions, and long-running
  experiments usually must be executed on separately allocated compute nodes

Do not assume that a command succeeding on the login node implies that it will
behave the same way on a GPU compute node.

### 1.2 Where real workloads run

Assume the actual heavy jobs run on **SLURM compute nodes**.

These jobs may:

- have strict walltime limits
- be interrupted by scheduler actions
- run on different nodes across retries
- require explicit GPU allocation

Therefore, any nontrivial experiment or benchmark should be designed for
scheduler interruptions from the start.

---

## 2. Filesystem Rules

### 2.1 Use NFS for raw run outputs

Raw experiment outputs must be written to persistent shared storage, typically
under an NFS path such as:

- `/nfs_global/...`

Use shared storage for:

- raw run directories
- `res.json`
- `latency.json`
- `.done` markers
- `progress.log`
- per-chunk outputs
- raw debug dumps
- scheduler-resumable artifacts

Reason:

- compute jobs may restart on different nodes
- login-node local state is not a reliable source of truth for resuming jobs
- persistent storage is required for recovery logic

### 2.2 Use workspace repo paths for derived analysis

Derived artifacts should stay in the repo workspace, for example:

- plots
- copied summary csv files
- markdown reports
- small analysis tables

Use workspace paths such as:

- `/workspace/...`

Reason:

- these artifacts are code-adjacent
- they are easier to inspect and version with the project

### 2.3 Keep the split explicit

Do not mix the two roles.

- Raw execution outputs: NFS/shared storage
- Derived summaries/plots/docs: workspace repo

---

## 3. Environment Rules

### 3.1 Do not casually mutate the primary runtime environment

If a project has a known working runtime environment, do not casually install
new packages into it.

Prefer:

- a dedicated experimental conda environment
- a dedicated quantization environment
- a dedicated deployment environment

Reason:

- runtime environments are easy to destabilize
- experiment reproducibility depends on environment stability

### 3.2 Separate coding from heavy execution

The login node is for:

- editing code
- inspecting files
- preparing scripts
- lightweight validation

The compute node is for:

- GPU inference
- large benchmark runs
- CUDA extension compilation when required
- long-running quantization/calibration/export steps

### 3.3 CUDA extension compilation rule

If a package requires:

- `nvcc`
- CUDA toolkit
- large GPU-aware native compilation

then do not assume it can be built on the login node.

First verify:

- `which nvcc`
- `nvcc --version`
- compiler compatibility
- whether the node is actually suitable for build

If these are missing, compile on a compute node or build a wheel elsewhere and
install that wheel later.

---

## 4. Resume / Recovery Rules

### 4.1 Default requirement

Any experiment longer than a short smoke test must support resume.

This means:

- chunked execution
- `.done` markers per chunk
- rerunning the same top-level script continues unfinished work only

### 4.2 Chunk sizing

Recommended chunk sizes:

- `5` episodes for expensive or fragile runs
- `10` episodes for cheaper runs

Choose smaller chunks when:

- compute nodes have short walltime
- interruption risk is high
- each episode is long

### 4.3 Completion criteria

Do not rely on exit code alone.

A chunk should be considered complete only if its expected outputs exist and
its counts match the planned workload.

Typical required checks:

1. `res.json` exists
2. `latency.json` exists
3. recorded `total_num == run_eps`
4. recorded `episode_e2e_sec.count == run_eps`

Only then should the script write `.done`.

### 4.4 GPU portability

Recovery state must depend on result files, not device identity.

A job should remain resumable after changing:

- GPU IDs
- node allocation
- process IDs
- job IDs

---

## 5. Random Seed Rules

### 5.1 Never silently reuse the same seed block across chunks

In chunked experiments, make sure later chunks do not restart from the same
seed base unless repeated-seed testing is explicitly intended.

### 5.2 Compare arms must share the same seed schedule

For A/B or multi-arm experiments:

- chunk `k` in every arm must use the same seed group
- chunk `k+1` must use a different seed group from chunk `k`

### 5.3 Record seed behavior explicitly

Agents should make seed scheduling inspectable through one or more of:

- `SEED_BASE`
- explicit chunk seed offsets
- progress log entries
- `stseed-*` result directory names

If seed handling is unclear, the experiment should not be treated as reliable.

---

## 6. Profiling Rules

### 6.1 Distinguish measurement domains

Do not mix:

- full RPC / wall-clock latency
- operator breakdown totals
- block timing totals
- KV-only timing totals
- debug-only instrumentation output

They have different denominators and different scopes.

### 6.2 Keep profiling modes separate

Maintain separate experiment categories for:

1. real latency / success-rate evaluation
2. light profiling
3. deep profiling / debug

Do not report numbers from one category as though they belong to another.
