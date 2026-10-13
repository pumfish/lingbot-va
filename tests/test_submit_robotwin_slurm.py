"""Tests for the optional Slurm adapter dry-run interface."""

from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).parents[1]
SUBMITTER = REPOSITORY_ROOT / "experiments/submit_robotwin_slurm.sh"


def test_dry_run_prints_single_gpu_submission_with_forwarded_runner_arguments(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--partition",
            "normal",
            "--time",
            "2-00:00:00",
            "--gpus",
            "1",
            "--cpus-per-task",
            "12",
            "--log-dir",
            str(tmp_path / "logs"),
            "--dry-run",
            "--",
            "--method",
            "bf16",
            "--task",
            "adjust_bottle",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "sbatch" in result.stdout
    assert "--partition=normal" in result.stdout
    assert "--time=2-00:00:00" in result.stdout
    assert "--gres=gpu:1" in result.stdout
    assert "--cpus-per-task=12" in result.stdout
    assert "slurm_robotwin_task_job.sh" in result.stdout
    assert "--method bf16 --task adjust_bottle" in result.stdout


def test_submitter_rejects_nonpositive_gpu_count():
    result = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--gpus",
            "0",
            "--",
            "--method",
            "bf16",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--gpus must be positive" in result.stderr
