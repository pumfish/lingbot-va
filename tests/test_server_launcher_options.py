"""Regression tests for the portable LingBot-VA server launcher."""

import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
LAUNCHER = REPOSITORY_ROOT / "single_gpu_lingbotva_server.sh"


def test_server_launcher_forwards_optional_modelopt_checkpoint(tmp_path):
    checkpoint = tmp_path / "modelopt.pt"
    environment = os.environ | {
        "SERVER_PYTHON": "/bin/echo",
        "SAVE_ROOT": str(tmp_path / "results"),
        "MODEL_PATH": "/models/lingbot-va",
        "MODELOPT_RESTORE_FROM": str(checkpoint),
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wan_va/wan_va_server.py" in result.stdout
    assert f"--modelopt_restore_from {checkpoint}" in result.stdout


def test_server_launcher_omits_checkpoint_argument_when_not_configured(tmp_path):
    environment = os.environ | {
        "SERVER_PYTHON": "/bin/echo",
        "SAVE_ROOT": str(tmp_path / "results"),
        "MODEL_PATH": "/models/lingbot-va",
        "MODELOPT_RESTORE_FROM": "",
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--modelopt_restore_from" not in result.stdout


def test_server_launcher_forwards_optional_duquant_checkpoint(tmp_path):
    checkpoint = tmp_path / "duquant.pt"
    environment = os.environ | {
        "SERVER_PYTHON": "/bin/echo",
        "SAVE_ROOT": str(tmp_path / "results"),
        "MODEL_PATH": "/models/lingbot-va",
        "DUQUANT_RESTORE_FROM": str(checkpoint),
        "DUQUANT_ROTATION_BLOCK_SIZE": "64",
        "DUQUANT_PERMUTATION_TIMES": "2",
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"--duquant_restore_from {checkpoint}" in result.stdout
    assert "--duquant_rotation_block_size 64" in result.stdout
    assert "--duquant_permutation_times 2" in result.stdout


def test_server_launcher_forwards_optional_omega_pack(tmp_path):
    pack = tmp_path / "omega.pt"
    environment = os.environ | {
        "SERVER_PYTHON": "/bin/echo",
        "SAVE_ROOT": str(tmp_path / "results"),
        "MODEL_PATH": "/models/lingbot-va",
        "OMEGA_QVLA_PACK": str(pack),
        "OMEGA_QVLA_BRANCHES": "action",
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"--omega_qvla_pack {pack}" in result.stdout
    assert "--omega_qvla_branches action" in result.stdout


def test_server_launcher_forwards_optional_ours_policy(tmp_path):
    policy = tmp_path / "policy.json"
    environment = os.environ | {
        "SERVER_PYTHON": "/bin/echo",
        "SAVE_ROOT": str(tmp_path / "results"),
        "MODEL_PATH": "/models/lingbot-va",
        "OURS_MIXED_POLICY": str(policy),
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"--ours_mixed_policy {policy}" in result.stdout
