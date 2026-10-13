"""Tests for the portable sequential RoboTwin campaign dry-run path."""

from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).parents[1]
CAMPAIGN = REPOSITORY_ROOT / "experiments/run_robotwin_campaign.py"


def test_dry_run_executes_each_requested_task_in_sequence(tmp_path):
    checkpoint = tmp_path / "uniform.pt"
    checkpoint.touch()

    result = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN),
            "--method",
            "uniform_w4a4",
            "--model-path",
            "/models/lingbot-va",
            "--output-root",
            str(tmp_path / "results"),
            "--checkpoint",
            str(checkpoint),
            "--tasks",
            "adjust_bottle",
            "hanging_mug",
            "--dry-run",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "1/2 task=adjust_bottle" in result.stdout
    assert "2/2 task=hanging_mug" in result.stdout
    assert str(tmp_path / "results" / "adjust_bottle") in result.stdout
    assert str(tmp_path / "results" / "hanging_mug") in result.stdout


def test_tasks_file_ignores_comments_and_blank_lines(tmp_path):
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text("# representative tasks\n\nadjust_bottle\nhanging_mug # long horizon\n")

    assert "run_robotwin_campaign" in str(CAMPAIGN)
    module_namespace: dict[str, object] = {"__file__": str(CAMPAIGN)}
    exec(compile(CAMPAIGN.read_text(), str(CAMPAIGN), "exec"), module_namespace)
    assert module_namespace["_read_tasks"](tasks_file) == ["adjust_bottle", "hanging_mug"]
