#!/usr/bin/env bash
set -euo pipefail

partition=""
time_limit="1-00:00:00"
gpus=1
cpus_per_task=8
job_name="lingbotva-robotwin"
log_dir=""
setup_script=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: experiments/submit_robotwin_slurm.sh [Slurm options] -- [runner options]

Slurm options:
  --partition NAME       Slurm partition to request.
  --time LIMIT           Slurm walltime, default: 1-00:00:00.
  --gpus COUNT           GPUs for one task, default: 1.
  --cpus-per-task COUNT  CPUs for one task, default: 8.
  --job-name NAME        Slurm job name.
  --log-dir PATH         Directory for Slurm stdout/stderr files.
  --setup-script PATH    Optional shell script sourced inside the allocation.
  --dry-run              Print the sbatch command without submitting it.

Arguments after -- are passed unchanged to experiments/run_robotwin_task.py.
EOF
}

while (($#)); do
  case "$1" in
    --partition) partition=$2; shift 2 ;;
    --time) time_limit=$2; shift 2 ;;
    --gpus) gpus=$2; shift 2 ;;
    --cpus-per-task) cpus_per_task=$2; shift 2 ;;
    --job-name) job_name=$2; shift 2 ;;
    --log-dir) log_dir=$2; shift 2 ;;
    --setup-script) setup_script=$2; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "unknown Slurm option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

(($#)) || { echo "missing runner options after --" >&2; usage >&2; exit 2; }
[[ "$gpus" =~ ^[1-9][0-9]*$ ]] || { echo "--gpus must be positive" >&2; exit 2; }
[[ "$cpus_per_task" =~ ^[1-9][0-9]*$ ]] || { echo "--cpus-per-task must be positive" >&2; exit 2; }
[[ -z "$setup_script" || -f "$setup_script" ]] || {
  echo "--setup-script does not exist: $setup_script" >&2
  exit 2
}

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
job_script="$repository_root/experiments/slurm_robotwin_task_job.sh"
sbatch_command=(
  sbatch
  --parsable
  --chdir="$repository_root"
  --job-name="$job_name"
  --time="$time_limit"
  --gres="gpu:${gpus}"
  --cpus-per-task="$cpus_per_task"
  --export="ALL,EXPERIMENT_SETUP_SCRIPT=${setup_script}"
)
if [[ -n "$partition" ]]; then
  sbatch_command+=(--partition="$partition")
fi
if [[ -n "$log_dir" ]]; then
  mkdir -p "$log_dir"
  sbatch_command+=(--output="$log_dir/%x-%j.out" --error="$log_dir/%x-%j.err")
fi

printf '[submit-robotwin-slurm] command='
printf ' %q' "${sbatch_command[@]}" "$job_script" "$@"
printf '\n'
if ((dry_run)); then
  exit 0
fi
"${sbatch_command[@]}" "$job_script" "$@"
