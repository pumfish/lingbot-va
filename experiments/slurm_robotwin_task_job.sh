#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n "${EXPERIMENT_SETUP_SCRIPT:-}" ]]; then
  source "$EXPERIMENT_SETUP_SCRIPT"
fi
exec "${PYTHON:-python}" "$repository_root/experiments/run_robotwin_task.py" "$@"
