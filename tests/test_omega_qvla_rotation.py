"""CPU checks for the Omega-QVLA block rotations."""

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn.functional as functional


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "wan_va" / "modules" / "omega_qvla_rotation.py"
SPEC = importlib.util.spec_from_file_location("omega_qvla_rotation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
omega_qvla_rotation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = omega_qvla_rotation
SPEC.loader.exec_module(omega_qvla_rotation)

build_omega_qvla_input_rotation = omega_qvla_rotation.build_omega_qvla_input_rotation
build_omega_qvla_output_rotation = omega_qvla_rotation.build_omega_qvla_output_rotation


def test_input_and_output_rotation_preserve_fp32_linear() -> None:
    torch.manual_seed(7)
    values = torch.randn(2, 3, 16)
    weight = torch.randn(16, 16)

    input_rotation = build_omega_qvla_input_rotation(weight, block_size=8)
    output_rotation = build_omega_qvla_output_rotation(weight, block_size=8)

    rotated_values = input_rotation.apply_fp32(values)
    rotated_weight = output_rotation.fold_weight_fp32(
        input_rotation.fold_weight_fp32(weight)
    )
    restored = output_rotation.apply_fp32(functional.linear(rotated_values, rotated_weight))

    torch.testing.assert_close(restored, functional.linear(values, weight), rtol=1e-4, atol=1e-4)
