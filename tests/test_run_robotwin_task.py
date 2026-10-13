"""Tests for the portable single-task RoboTwin runner dry-run path."""

from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).parents[1]
RUNNER = REPOSITORY_ROOT / "experiments/run_robotwin_task.py"


def test_dry_run_resolves_ours_and_prints_server_and_client_commands(tmp_path):
    pack = tmp_path / "omega.pt"
    policy = tmp_path / "policy.json"
    frozen = tmp_path / "frozen.jsonl"
    pack.touch()
    policy.touch()
    frozen.touch()

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--method",
            "ours_w4a4",
            "--task",
            "put_bottles_dustbin",
            "--model-path",
            "/models/lingbot-va",
            "--output-root",
            str(tmp_path / "results"),
            "--pack",
            str(pack),
            "--policy",
            str(policy),
            "--episodes",
            "100",
            "--task-seeds",
            "10001,10002",
            "--frozen-episodes",
            str(frozen),
            "--dry-run",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "method=ours_w4a4 task=put_bottles_dustbin episodes=100" in result.stdout
    assert "single_gpu_lingbotva_server.sh" in result.stdout
    assert "evaluation.robotwin.eval_polict_client_openpi" in result.stdout
    assert f"env OMEGA_QVLA_PACK={pack}" in result.stdout
    assert f"env OURS_MIXED_POLICY={policy}" in result.stdout
    assert "env ROBOTWIN_TASK_SEEDS=10001,10002" in result.stdout
    assert f"env ROBOTWIN_FROZEN_EPISODES_JSONL={frozen}" in result.stdout
