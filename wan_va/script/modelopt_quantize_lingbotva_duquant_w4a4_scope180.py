#!/usr/bin/env python3
"""Build a DuQuant W4A4 checkpoint over the shared Omega DiT-180 scope."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse the checkpoint-build arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--rotation-block-size", type=int, default=128)
    parser.add_argument("--max-rotation-steps", type=int, default=8)
    parser.add_argument("--permutation-times", type=int, default=1)
    parser.add_argument("--duquant-seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def build_duquant_config(args: argparse.Namespace) -> dict[str, int]:
    """Return the topology parameters required for deterministic checkpoint restore."""
    return {
        "schema_version": 2,
        "comparison_scope": "omega_qvla_dit_180",
        "block_size": args.rotation_block_size,
        "max_rotation_steps": args.max_rotation_steps,
        "permutation_times": args.permutation_times,
        "seed": args.duquant_seed,
    }


def build_manifest(
    args: argparse.Namespace,
    calibration_count: int,
    calibration_sha256: str,
    total_weight_bytes: int,
    scope: dict[str, Any],
    quant_config: dict[str, Any],
    duquant_config: dict[str, int],
    checkpoint: Path,
) -> dict[str, Any]:
    """Return the provenance record for a scope-matched DuQuant checkpoint."""
    return {
        "baseline": "D5_Wan_adapted_DuQuant_W4A4_OmegaDiT180",
        "comparison_scope": "omega_qvla_dit_180",
        "model_path": args.model_path,
        "calibration_root": args.calibration_root,
        "calibration_manifest_sha256": calibration_sha256,
        "calibration_sample_count": calibration_count,
        "dtype": args.dtype,
        "quantization": "weight_int4_per_channel_activation_int4_per_tensor",
        "duquant_config": duquant_config,
        "total_transformer_linear_weight_bytes": total_weight_bytes,
        "quant_config": quant_config,
        **scope,
        "checkpoint": str(checkpoint),
    }


def main() -> None:
    """Fold DuQuant transforms, quantize the shared scope, and save provenance."""
    repo_root = Path(__file__).resolve().parents[2]
    wan_va_root = repo_root / "wan_va"
    if str(wan_va_root) not in sys.path:
        sys.path.insert(0, str(wan_va_root))

    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    import torch
    from modules.calibration_utils import (
        calibration_forward,
        load_calibration_samples,
        resolve_dtype,
        sha256_file,
    )
    from modules.duquant_mapping import attach_wan_duquant_transforms
    from modules.omega_dit_scope import (
        modelopt_w4a4_config,
        total_transformer_linear_weight_bytes,
        verify_scope_quantizers,
    )
    from modules.utils import load_transformer

    args = parse_args()
    calibration_root = Path(args.calibration_root)
    sample_paths = load_calibration_samples(calibration_root)
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)
    transformer = load_transformer(
        os.path.join(args.model_path, "transformer"), torch_dtype=dtype, torch_device=device
    )
    total_weight_bytes = total_transformer_linear_weight_bytes(transformer)
    logical_targets = attach_wan_duquant_transforms(
        transformer,
        block_size=args.rotation_block_size,
        max_rotation_steps=args.max_rotation_steps,
        permutation_times=args.permutation_times,
        seed=args.duquant_seed,
    )
    quantizer_targets = tuple(f"{target.module_path}.linear" for target in logical_targets)
    quant_config = modelopt_w4a4_config(quantizer_targets)
    mtq.quantize(
        transformer,
        copy.deepcopy(quant_config),
        lambda model: calibration_forward(model, sample_paths, device, dtype),
    )
    scope = verify_scope_quantizers(transformer, quantizer_targets)
    scope["logical_quantized_linear_names"] = [target.module_path for target in logical_targets]
    scope["logical_quantized_linear_count"] = len(logical_targets)
    scope["quantized_weight_byte_fraction"] = scope["quantized_weight_bytes"] / total_weight_bytes
    if scope["logical_quantized_linear_count"] != 180:
        raise RuntimeError(
            "Expected 180 logical DuQuant targets, got "
            f"{scope['logical_quantized_linear_count']}"
        )

    checkpoint = save_root / "transformer_modelopt_duquant_w4a4_scope180.pt"
    duquant_config = build_duquant_config(args)
    mto.save(transformer, str(checkpoint))
    (save_root / "duquant_config.json").write_text(
        json.dumps(duquant_config, indent=2) + "\n", encoding="utf-8"
    )
    manifest = build_manifest(
        args,
        calibration_count=len(sample_paths),
        calibration_sha256=sha256_file(calibration_root / "manifest.json"),
        total_weight_bytes=total_weight_bytes,
        scope=scope,
        quant_config=quant_config,
        duquant_config=duquant_config,
        checkpoint=checkpoint,
    )
    (save_root / "quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[duquant_scope180] calibration_samples={len(sample_paths)}")
    print(f"[duquant_scope180] targets={scope['logical_quantized_linear_count']}")
    print(f"[duquant_scope180] enabled_quantizers={scope['enabled_quantizer_counts']}")
    print(f"[duquant_scope180] checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
