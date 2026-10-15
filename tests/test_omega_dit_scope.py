"""Tests for the shared 180-target Omega DiT comparison scope."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn


MODULE_PATH = Path(__file__).parents[1] / "wan_va/modules/omega_dit_scope.py"
SPEC = importlib.util.spec_from_file_location("omega_dit_scope", MODULE_PATH)
omega_dit_scope = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(omega_dit_scope)


def test_omega_dit_target_names_match_30_block_six_projection_scope():
    targets = omega_dit_scope.omega_dit_target_names(30)

    assert len(targets) == 180
    assert targets[:6] == (
        "blocks.0.attn1.to_q",
        "blocks.0.attn1.to_k",
        "blocks.0.attn1.to_v",
        "blocks.0.attn1.to_out.0",
        "blocks.0.ffn.net.0.proj",
        "blocks.0.ffn.net.2",
    )
    assert targets[-1] == "blocks.29.ffn.net.2"


def test_scope_config_enables_only_explicit_weight_and_input_quantizers():
    targets = omega_dit_scope.omega_dit_target_names(1)
    config = omega_dit_scope.modelopt_w4a4_config(targets)

    assert config["algorithm"] == "max"
    assert config["quant_cfg"][0] == {"quantizer_name": "*", "enable": False}
    assert config["quant_cfg"][-1] == {"quantizer_name": "*output_quantizer", "enable": False}
    assert len(config["quant_cfg"]) == 2 + 2 * len(targets)
    assert config["quant_cfg"][1]["cfg"] == {"num_bits": 4, "axis": 0}
    assert config["quant_cfg"][2]["cfg"] == {"num_bits": 4, "axis": None}


@pytest.mark.parametrize("targets", [(), ("blocks.0.attn1.to_q",) * 2])
def test_scope_config_rejects_empty_or_duplicate_targets(targets):
    with pytest.raises(ValueError, match="non-empty and unique"):
        omega_dit_scope.modelopt_w4a4_config(targets)


def test_total_linear_weight_bytes_excludes_non_linear_parameters():
    model = nn.Sequential(nn.Linear(3, 2, bias=False), nn.LayerNorm(2), nn.Linear(2, 1))
    assert omega_dit_scope.total_transformer_linear_weight_bytes(model) == (6 + 2) * 4
