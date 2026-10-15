"""Shared DiT target scope for scope-matched W4A4 comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch.nn as nn


OMEGA_DIT_SUFFIXES = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)


def omega_dit_target_names(block_count: int) -> tuple[str, ...]:
    """Return the official Omega-QVLA DiT targets for ``block_count`` blocks."""
    if block_count <= 0:
        raise ValueError(f"block_count must be positive, got {block_count}")
    return tuple(
        f"blocks.{block_index}.{suffix}"
        for block_index in range(block_count)
        for suffix in OMEGA_DIT_SUFFIXES
    )


def modelopt_w4a4_config(target_names: Iterable[str]) -> dict[str, Any]:
    """Build exact-name W4A4 config without wildcard scope leakage."""
    targets = tuple(target_names)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target_names must be non-empty and unique")

    quant_cfg: list[dict[str, Any]] = [{"quantizer_name": "*", "enable": False}]
    for target in targets:
        quant_cfg.extend(
            (
                {
                    "quantizer_name": f"*{target}.weight_quantizer",
                    "cfg": {"num_bits": 4, "axis": 0},
                },
                {
                    "quantizer_name": f"*{target}.input_quantizer",
                    "cfg": {"num_bits": 4, "axis": None},
                },
            )
        )
    quant_cfg.append({"quantizer_name": "*output_quantizer", "enable": False})
    return {"quant_cfg": quant_cfg, "algorithm": "max"}


def verify_scope_quantizers(model: nn.Module, target_names: Iterable[str]) -> dict[str, Any]:
    """Verify enabled ModelOpt quantizers exactly match the target scope."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    targets = tuple(target_names)
    expected_weight = {f"{target}.weight_quantizer" for target in targets}
    expected_input = {f"{target}.input_quantizer" for target in targets}
    enabled = {
        name
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer) and module.is_enabled
    }
    actual_weight = {name for name in enabled if name.endswith("weight_quantizer")}
    actual_input = {name for name in enabled if name.endswith("input_quantizer")}
    if actual_weight != expected_weight or actual_input != expected_input:
        raise RuntimeError(
            "Scope-matched quantizer mismatch: "
            f"missing_weight={sorted(expected_weight - actual_weight)[:3]} "
            f"extra_weight={sorted(actual_weight - expected_weight)[:3]} "
            f"missing_input={sorted(expected_input - actual_input)[:3]} "
            f"extra_input={sorted(actual_input - expected_input)[:3]}"
        )

    weight_bytes = 0
    for target in targets:
        linear = model.get_submodule(target)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {target}, got {type(linear)}")
        weight_bytes += linear.weight.numel() * linear.weight.element_size()
    return {
        "quantized_linear_names": list(targets),
        "quantized_linear_count": len(targets),
        "enabled_quantizer_counts": {
            "total": len(enabled),
            "weight": len(actual_weight),
            "input": len(actual_input),
        },
        "quantized_weight_bytes": weight_bytes,
    }


def total_transformer_linear_weight_bytes(model: nn.Module) -> int:
    """Return the storage consumed by all two-dimensional Linear weights."""
    return sum(
        module.weight.numel() * module.weight.element_size()
        for module in model.modules()
        if isinstance(module, nn.Linear) and module.weight.ndim == 2
    )
