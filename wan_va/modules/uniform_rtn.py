"""Uniform no-rotation RTN W4A4 configuration for LingBot-VA blocks."""

from __future__ import annotations

from typing import Any, Iterable


def _is_block_quantizer_name(name: str) -> bool:
    return name.startswith("blocks.") or ".blocks." in name


def build_uniform_w4a4_config() -> dict[str, Any]:
    """Return the ModelOpt config for the Uniform RTN/max W4A4 baseline.

    Only Transformer-block Linear weights and inputs are quantized. Weight
    scales are per output channel; activation scales are per tensor. The
    baseline deliberately has no rotation, smoothing, or mixed precision.
    """
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*blocks.*weight_quantizer",
                "cfg": {"num_bits": 4, "axis": 0},
            },
            {
                "quantizer_name": "*blocks.*input_quantizer",
                "cfg": {"num_bits": 4, "axis": None},
            },
            {"quantizer_name": "*output_quantizer", "enable": False},
        ],
        "algorithm": "max",
    }


def validate_uniform_w4a4_quantizer_names(enabled_names: Iterable[str]) -> None:
    """Require restored Uniform W4A4 checkpoints to enable block W/A quantizers."""
    names = tuple(enabled_names)
    weight_count = sum(
        name.endswith("weight_quantizer") and _is_block_quantizer_name(name) for name in names
    )
    input_count = sum(
        name.endswith("input_quantizer") and _is_block_quantizer_name(name) for name in names
    )
    output_count = sum(name.endswith("output_quantizer") for name in names)

    if weight_count == 0 or input_count == 0:
        raise RuntimeError(
            "Uniform W4A4 checkpoint must enable block weight and input quantizers: "
            f"weight={weight_count}, input={input_count}."
        )
    if output_count:
        raise RuntimeError("Uniform W4A4 checkpoint must keep output quantizers disabled.")
