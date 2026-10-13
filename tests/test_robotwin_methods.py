"""Unit tests for portable RoboTwin quantization method configuration."""

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments/robotwin_methods.py"
SPEC = importlib.util.spec_from_file_location("robotwin_methods", MODULE_PATH)
robotwin_methods = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = robotwin_methods
SPEC.loader.exec_module(robotwin_methods)


def test_each_formal_method_resolves_to_its_required_server_environment():
    assert robotwin_methods.resolve_robotwin_method("bf16") == {}
    assert robotwin_methods.resolve_robotwin_method(
        "uniform_w4a4", checkpoint="uniform.pt"
    ) == {"MODELOPT_RESTORE_FROM": "uniform.pt"}
    assert robotwin_methods.resolve_robotwin_method(
        "duquant_w4a4",
        checkpoint="duquant.pt",
        duquant_rotation_block_size=64,
        duquant_permutation_times=2,
    ) == {
        "DUQUANT_RESTORE_FROM": "duquant.pt",
        "DUQUANT_ROTATION_BLOCK_SIZE": "64",
        "DUQUANT_PERMUTATION_TIMES": "2",
    }
    assert robotwin_methods.resolve_robotwin_method(
        "omega_qvla_w4a4", pack="omega.pt"
    ) == {"OMEGA_QVLA_PACK": "omega.pt", "OMEGA_QVLA_BRANCHES": "both"}
    assert robotwin_methods.resolve_robotwin_method(
        "ours_w4a4", pack="ours.pt", policy="policy.json"
    ) == {
        "OMEGA_QVLA_PACK": "ours.pt",
        "OMEGA_QVLA_BRANCHES": "both",
        "OURS_MIXED_POLICY": "policy.json",
    }


def test_method_resolution_rejects_missing_or_unexpected_artifacts():
    with pytest.raises(ValueError, match="requires: policy"):
        robotwin_methods.resolve_robotwin_method("ours_w4a4", pack="ours.pt")
    with pytest.raises(ValueError, match="does not use: pack"):
        robotwin_methods.resolve_robotwin_method("bf16", pack="unused.pt")
    with pytest.raises(ValueError, match="Unknown method"):
        robotwin_methods.resolve_robotwin_method("unknown")


def test_validate_method_artifacts_requires_existing_files(tmp_path):
    artifact = tmp_path / "pack.pt"
    artifact.touch()
    robotwin_methods.validate_method_artifacts({"OMEGA_QVLA_PACK": str(artifact)})

    with pytest.raises(FileNotFoundError, match="OURS_MIXED_POLICY"):
        robotwin_methods.validate_method_artifacts(
            {"OURS_MIXED_POLICY": str(tmp_path / "missing.json")}
        )
