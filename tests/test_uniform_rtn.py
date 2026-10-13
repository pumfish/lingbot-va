"""Unit tests for the Uniform RTN W4A4 method definition."""

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "wan_va/modules/uniform_rtn.py"
SPEC = importlib.util.spec_from_file_location("uniform_rtn", MODULE_PATH)
uniform_rtn = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(uniform_rtn)


def test_uniform_w4a4_config_is_no_rotation_block_only_w4a4():
    config = uniform_rtn.build_uniform_w4a4_config()

    assert config["algorithm"] == "max"
    assert config["quant_cfg"] == [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*blocks.*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}},
        {"quantizer_name": "*blocks.*input_quantizer", "cfg": {"num_bits": 4, "axis": None}},
        {"quantizer_name": "*output_quantizer", "enable": False},
    ]


def test_uniform_w4a4_quantizer_validation_requires_block_input_and_weight():
    uniform_rtn.validate_uniform_w4a4_quantizer_names(
        [
            "blocks.0.attn1.to_k.weight_quantizer",
            "blocks.0.attn1.to_k.input_quantizer",
        ]
    )

    with pytest.raises(RuntimeError, match="weight and input"):
        uniform_rtn.validate_uniform_w4a4_quantizer_names(
            ["blocks.0.attn1.to_k.weight_quantizer"]
        )

    with pytest.raises(RuntimeError, match="output quantizers"):
        uniform_rtn.validate_uniform_w4a4_quantizer_names(
            [
                "blocks.0.attn1.to_k.weight_quantizer",
                "blocks.0.attn1.to_k.input_quantizer",
                "blocks.0.attn1.to_k.output_quantizer",
            ]
        )
