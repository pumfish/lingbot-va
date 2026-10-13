"""Run a list of LingbotVA RoboTwin tasks sequentially without Slurm."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).parents[1]
TASK_RUNNER = REPOSITORY_ROOT / "experiments/run_robotwin_task.py"


def _read_tasks(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Task list does not exist: {path}")
    tasks = [line.split("#", maxsplit=1)[0].strip() for line in path.read_text().splitlines()]
    return [task for task in tasks if task]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--tasks", nargs="+")
    task_group.add_argument("--tasks-file", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--policy-name", default="ACT")
    parser.add_argument("--policy-config", default="policy/ACT/deploy_policy.yml")
    parser.add_argument("--video-guidance-scale", type=float, default=5.0)
    parser.add_argument("--action-guidance-scale", type=float, default=1.0)
    parser.add_argument("--enable-offload", choices=(0, 1), type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--pack")
    parser.add_argument("--policy")
    parser.add_argument("--duquant-rotation-block-size", type=int, default=128)
    parser.add_argument("--duquant-permutation-times", type=int, default=1)
    parser.add_argument("--robotwin-root")
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--server-start-timeout", type=int, default=420)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    return args


def _task_command(args: argparse.Namespace, task: str) -> list[str]:
    command = [
        sys.executable,
        str(TASK_RUNNER),
        "--method",
        args.method,
        "--task",
        task,
        "--model-path",
        args.model_path,
        "--output-root",
        str(Path(args.output_root) / task),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--port",
        str(args.port),
        "--task-config",
        args.task_config,
        "--policy-name",
        args.policy_name,
        "--policy-config",
        args.policy_config,
        "--video-guidance-scale",
        str(args.video_guidance_scale),
        "--action-guidance-scale",
        str(args.action_guidance_scale),
        "--enable-offload",
        str(args.enable_offload),
        "--duquant-rotation-block-size",
        str(args.duquant_rotation_block_size),
        "--duquant-permutation-times",
        str(args.duquant_permutation_times),
        "--server-python",
        args.server_python,
    ]
    for flag, value in (
        ("--checkpoint", args.checkpoint),
        ("--pack", args.pack),
        ("--policy", args.policy),
        ("--robotwin-root", args.robotwin_root),
    ):
        if value:
            command.extend((flag, str(value)))
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> int:
    args = _parse_args()
    tasks = args.tasks if args.tasks is not None else _read_tasks(args.tasks_file)
    if not tasks:
        raise ValueError("The campaign task list is empty")

    for index, task in enumerate(tasks, start=1):
        command = _task_command(args, task)
        print(f"[robotwin-campaign] {index}/{len(tasks)} task={task}")
        print(f"[robotwin-campaign] command={shlex.join(command)}")
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
        if completed.returncode:
            print(
                f"[robotwin-campaign] failed task={task} status={completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
