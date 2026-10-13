"""Run one LingbotVA RoboTwin task without requiring Slurm."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.robotwin_methods import METHODS, resolve_robotwin_method, validate_method_artifacts


def _wait_for_server(process: subprocess.Popen[bytes], port: int, timeout_seconds: int) -> None:
    """Wait for the local inference server or report its early exit."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"LingbotVA server exited early with status {process.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(1)
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(2)
    raise TimeoutError(f"LingbotVA server was not ready on port {port} within {timeout_seconds}s")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument(
        "--task",
        required=True,
        help="RoboTwin task name, for example put_bottles_dustbin",
    )
    parser.add_argument("--model-path", required=True, help="LingbotVA model directory")
    parser.add_argument("--output-root", required=True, help="Root directory for this task result")
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
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.server_start_timeout <= 0:
        parser.error("--server-start-timeout must be positive")
    return args


def _commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    output_root = Path(args.output_root)
    server_root = output_root / "server"
    client_root = output_root / "client"
    server_command = ["bash", "single_gpu_lingbotva_server.sh"]
    client_command = [
        args.server_python,
        "-m",
        "evaluation.robotwin.eval_polict_client_openpi",
        "--config",
        args.policy_config,
        "--overrides",
        "--task_name",
        args.task,
        "--task_config",
        args.task_config,
        "--train_config_name",
        "0",
        "--model_name",
        "0",
        "--ckpt_setting",
        args.method,
        "--seed",
        str(args.seed),
        "--policy_name",
        args.policy_name,
        "--save_root",
        str(client_root),
        "--video_guidance_scale",
        str(args.video_guidance_scale),
        "--action_guidance_scale",
        str(args.action_guidance_scale),
        "--test_num",
        str(args.episodes),
        "--port",
        str(args.port),
    ]
    return server_command, client_command


def main() -> int:
    args = _parse_args()
    method_environment = resolve_robotwin_method(
        args.method,
        checkpoint=args.checkpoint,
        pack=args.pack,
        policy=args.policy,
        duquant_rotation_block_size=args.duquant_rotation_block_size,
        duquant_permutation_times=args.duquant_permutation_times,
    )
    validate_method_artifacts(method_environment)
    server_command, client_command = _commands(args)
    output_root = Path(args.output_root)
    server_root = output_root / "server"
    client_root = output_root / "client"
    logs_root = output_root / "logs"

    server_environment = os.environ | method_environment | {
        "START_PORT": str(args.port),
        "SAVE_ROOT": str(server_root),
        "MODEL_PATH": args.model_path,
        "CONFIG_NAME": "robotwin",
        "ENABLE_OFFLOAD": str(args.enable_offload),
        "SERVER_PYTHON": args.server_python,
    }
    client_environment = os.environ.copy()
    if args.robotwin_root:
        client_environment["ROBOTWIN_ROOT"] = args.robotwin_root

    print(f"[robotwin-task] method={args.method} task={args.task} episodes={args.episodes}")
    print(f"[robotwin-task] server={shlex.join(server_command)}")
    print(f"[robotwin-task] client={shlex.join(client_command)}")
    if args.dry_run:
        for name in sorted(method_environment):
            print(f"[robotwin-task] env {name}={method_environment[name]}")
        return 0

    for directory in (server_root, client_root, logs_root):
        directory.mkdir(parents=True, exist_ok=True)
    with (logs_root / "server.log").open("wb") as server_log:
        server = subprocess.Popen(
            server_command,
            cwd=REPOSITORY_ROOT,
            env=server_environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_server(server, args.port, args.server_start_timeout)
            with (logs_root / "client.log").open("wb") as client_log:
                completed = subprocess.run(
                    client_command,
                    cwd=REPOSITORY_ROOT,
                    env=client_environment,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            return completed.returncode
        finally:
            _terminate_process_group(server)


if __name__ == "__main__":
    raise SystemExit(main())
