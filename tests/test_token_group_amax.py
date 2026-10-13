"""Unit tests for Ours online token-wise group-Amax activation Q/DQ."""

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
token_group_amax = _load_module("token_group_amax")


def test_token_group_amax_uses_independent_scale_per_token_and_group():
    quantizer = token_group_amax.TokenGroupAmaxQdq(group_size=2)
    values = torch.tensor(
        [[[1.0, 1.0 / 7, 4.0, 4.0 / 7], [2.0, 2.0 / 7, 8.0, 8.0 / 7]]]
    )

    quantized = quantizer.quantize(values, bits=4)

    torch.testing.assert_close(quantized, values)


def test_token_group_amax_validates_bit_width_and_channel_groups():
    quantizer = token_group_amax.TokenGroupAmaxQdq(group_size=4)

    with pytest.raises(ValueError, match="4 or 8 bits"):
        quantizer.quantize(torch.ones(1, 4), bits=6)
    with pytest.raises(ValueError, match="divisible"):
        quantizer.quantize(torch.ones(1, 6), bits=4)
