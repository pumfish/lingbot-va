"""CPU tests for enabling and disabling an Omega-QVLA offline pack."""

import importlib.util
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as functional


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


rotation = _load_module("omega_qvla_rotation")
_load_module("omega_qvla_pack")
_load_module("omega_qvla_activation")
controller_module = _load_module("omega_qvla_controller")


class ToyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)


def _pack(model: ToyTransformer) -> dict[str, object]:
    weight = model.linear.weight.detach().float()
    input_rotation = rotation.build_omega_qvla_input_rotation(weight, block_size=4)
    output_rotation = rotation.build_omega_qvla_output_rotation(weight, block_size=4)
    folded = output_rotation.fold_weight_fp32(input_rotation.fold_weight_fp32(weight))
    return {
        "__meta__": {"format": "lingbotva_omega_qvla_official_dit_pack_v2", "targets": ["linear"]},
        "linear": {
            "format": "dit_svdquant_v1",
            "weight_res_q": folded,
            "lowrank_A": torch.empty(8, 0),
            "lowrank_B": torch.empty(8, 0),
            "act_scale_table": torch.full((51, 8), 0.25),
            "duquant_rotation_perm": input_rotation.permutation,
            "duquant_rotation_blocks": input_rotation.blocks,
            "duquant_rotation_out_blocks": output_rotation.blocks,
            "weight_bits": 4,
            "a_bits": 4,
            "rank": 0,
            "in_features": 8,
            "out_features": 8,
            "gptq_block_size": 128,
            "gptq_damp_percent": 0.05,
            "use_rtn": True,
            "rotation_mode": input_rotation.mode,
            "rotation_block_size": 4,
            "rotation_out_block_size": 4,
            "latent_static_scale": torch.full((8,), 0.25),
        },
    }


def test_controller_activates_pack_and_restores_original_linear(tmp_path):
    torch.manual_seed(3)
    model = ToyTransformer()
    values = torch.full((1, 2, 8), 0.3)
    original_weight = model.linear.weight.detach().clone()
    original_bias = model.linear.bias.detach().clone()
    pack = _pack(model)
    pack_path = tmp_path / "omega.pt"
    torch.save(pack, pack_path)

    controller = controller_module.OmegaQVLAOfflinePackController(model, pack_path)
    controller.activate("action", 0)
    actual = model.linear(values)

    record = pack["linear"]
    transformed = rotation.OmegaQVLAInputRotation(
        record["duquant_rotation_perm"],
        record["duquant_rotation_blocks"],
        4,
        "svd_hadamard",
    ).apply(values)
    expected_input = torch.round(transformed / 0.25).clamp(-8, 7) * 0.25
    expected = rotation.OmegaQVLAOutputRotation(record["duquant_rotation_out_blocks"], 4).apply(
        functional.linear(expected_input, record["weight_res_q"])
    ) + original_bias
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    controller.deactivate()
    torch.testing.assert_close(model.linear.weight, original_weight)
    torch.testing.assert_close(model.linear.bias, original_bias)
    controller.close()


def test_controller_skips_unselected_branch(tmp_path):
    model = ToyTransformer()
    values = torch.randn(1, 2, 8)
    reference = model.linear(values)
    pack_path = tmp_path / "omega.pt"
    torch.save(_pack(model), pack_path)

    controller = controller_module.OmegaQVLAOfflinePackController(
        model, pack_path, quantized_branches="action"
    )
    controller.activate("latent", 0)
    torch.testing.assert_close(model.linear(values), reference)
    controller.close()
