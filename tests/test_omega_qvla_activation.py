"""Unit tests for the original Omega-QVLA offline activation scale policy."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).parents[1]
MODULE_ROOT = REPOSITORY_ROOT / "wan_va" / "modules"


def _load_module(name: str):
    package = sys.modules.setdefault("modules", types.ModuleType("modules"))
    package.__path__ = [str(MODULE_ROOT)]
    qualified_name = f"modules.{name}"
    spec = importlib.util.spec_from_file_location(qualified_name, MODULE_ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


_load_module("omega_qvla_pack")
activation = _load_module("omega_qvla_activation")


def _record(action_rows: int = 8) -> dict[str, object]:
    return {
        "format": "dit_svdquant_v1",
        "weight_res_q": torch.ones(8, 8),
        "lowrank_A": torch.empty(8, 0),
        "lowrank_B": torch.empty(8, 0),
        "act_scale_table": torch.stack(
            [torch.full((8,), 0.25 * (index + 1)) for index in range(action_rows)]
        ),
        "duquant_rotation_perm": torch.arange(8),
        "duquant_rotation_blocks": torch.eye(4).repeat(2, 1, 1),
        "duquant_rotation_out_blocks": torch.eye(4).repeat(2, 1, 1),
        "weight_bits": 4,
        "a_bits": 4,
        "rank": 0,
        "in_features": 8,
        "out_features": 8,
        "gptq_block_size": 128,
        "gptq_damp_percent": 0.05,
        "use_rtn": True,
        "rotation_mode": "svd_hadamard",
        "rotation_block_size": 4,
        "rotation_out_block_size": 4,
        "latent_static_scale": torch.full((8,), 0.5),
    }


def test_offline_qdq_selects_static_latent_and_bucketed_action_scales():
    controller = activation.OmegaOfflineActivationQdq(_record())

    torch.testing.assert_close(controller.scale_for("latent"), torch.full((8,), 0.5))
    torch.testing.assert_close(controller.scale_for("action", 0), torch.full((8,), 0.25))
    torch.testing.assert_close(controller.scale_for("action", 50), torch.full((8,), 2.0))

    values = torch.full((1, 8), 0.3)
    torch.testing.assert_close(
        controller.quantize(values, "latent"), torch.full((1, 8), 0.5)
    )
    torch.testing.assert_close(
        controller.quantize(values, "action", 0), torch.full((1, 8), 0.25)
    )


def test_offline_qdq_requires_valid_action_step_and_input_width():
    controller = activation.OmegaOfflineActivationQdq(_record(action_rows=51))

    with pytest.raises(ValueError, match="denoise_step"):
        controller.scale_for("action")
    with pytest.raises(ValueError, match="last dimension"):
        controller.quantize(torch.ones(1, 7), "latent")
