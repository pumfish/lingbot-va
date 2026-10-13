"""Public Omega-QVLA rank-0 RTN pack schema and fake-Q/DQ primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


PACK_FORMAT = "lingbotva_omega_qvla_official_dit_pack_v2"
RECORD_FORMAT = "dit_svdquant_v1"


def qmax(bits: int) -> int:
    """Return the positive signed code bound for a public W4/W8 recipe."""
    if bits not in (4, 8):
        raise ValueError(f"Expected 4 or 8 bits, got {bits}")
    return (1 << (bits - 1)) - 1


def symmetric_qdq(values: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """Apply symmetric fake quantization using Omega-QVLA's signed code range."""
    bound = qmax(bits)
    scale = scale.to(device=values.device, dtype=torch.float32).clamp_min(1e-8)
    integers = torch.round(values.float() / scale).clamp(-bound - 1, bound)
    return integers * scale


def validate_rank0_rtn_record(record: Mapping[str, Any]) -> None:
    """Validate one immutable rank-0 RTN DiT Linear record."""
    required = {
        "format", "weight_res_q", "lowrank_A", "lowrank_B", "act_scale_table",
        "duquant_rotation_perm", "duquant_rotation_blocks", "duquant_rotation_out_blocks",
        "weight_bits", "a_bits", "rank", "in_features", "out_features",
        "gptq_block_size", "gptq_damp_percent", "use_rtn", "latent_static_scale",
        "rotation_mode", "rotation_block_size", "rotation_out_block_size",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Omega-QVLA record is missing: {sorted(missing)}")
    if record["format"] != RECORD_FORMAT or int(record["rank"]) != 0:
        raise ValueError("Expected a rank-0 dit_svdquant_v1 record")
    if int(record["gptq_block_size"]) != 128 or float(record["gptq_damp_percent"]) != 0.05:
        raise ValueError("Expected GPTQ block=128 and damp=0.05")
    if record["use_rtn"] is not True:
        raise ValueError("Expected use_rtn=True")
    if int(record["weight_bits"]) not in (4, 8) or int(record["a_bits"]) not in (4, 8):
        raise ValueError("Public Omega-QVLA records require W/A bits in {4, 8}")

    input_features, output_features = int(record["in_features"]), int(record["out_features"])
    if tuple(record["weight_res_q"].shape) != (output_features, input_features):
        raise ValueError("weight_res_q has an invalid shape")
    if tuple(record["lowrank_A"].shape) != (output_features, 0):
        raise ValueError("rank-0 lowrank_A must have shape [out_features, 0]")
    if tuple(record["lowrank_B"].shape) != (input_features, 0):
        raise ValueError("rank-0 lowrank_B must have shape [in_features, 0]")
    if tuple(record["latent_static_scale"].shape) != (input_features,):
        raise ValueError("latent_static_scale has an invalid shape")
    action_table = record["act_scale_table"]
    valid_action_shapes = {(8, input_features), (51, input_features)}
    if action_table.ndim != 2 or action_table.shape not in valid_action_shapes:
        raise ValueError("act_scale_table must be [8|51, input_channels]")
    if tuple(record["duquant_rotation_perm"].shape) != (input_features,):
        raise ValueError("duquant_rotation_perm has an invalid shape")
    if not torch.equal(
        torch.sort(record["duquant_rotation_perm"].detach().cpu()).values,
        torch.arange(input_features),
    ):
        raise ValueError("duquant_rotation_perm must be a channel permutation")
    input_blocks = record["duquant_rotation_blocks"]
    output_blocks = record["duquant_rotation_out_blocks"]
    if (
        input_blocks.ndim != 3
        or input_blocks.shape[1] != input_blocks.shape[2]
        or input_blocks.shape[0] * input_blocks.shape[1] != input_features
    ):
        raise ValueError("duquant_rotation_blocks has an invalid shape")
    if (
        output_blocks.ndim != 3
        or output_blocks.shape[1] != output_blocks.shape[2]
        or output_blocks.shape[0] * output_blocks.shape[1] != output_features
    ):
        raise ValueError("duquant_rotation_out_blocks has an invalid shape")
    for name in ("weight_res_q", "latent_static_scale", "act_scale_table"):
        if not torch.isfinite(record[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    if torch.any(record["latent_static_scale"] <= 0) or torch.any(action_table <= 0):
        raise ValueError("offline activation scales must be positive")


def validate_rank0_rtn_pack(
    pack: Mapping[str, Any], *, expected_targets: Sequence[str] | None = None
) -> None:
    """Validate the public immutable pack and its exact target set."""
    metadata = pack.get("__meta__")
    if not isinstance(metadata, Mapping) or metadata.get("format") != PACK_FORMAT:
        raise ValueError("Unsupported Omega-QVLA pack format")
    targets = sorted(name for name in pack if name != "__meta__")
    if sorted(metadata.get("targets", [])) != targets:
        raise ValueError("Pack target list does not match record keys")
    if expected_targets is not None and targets != sorted(expected_targets):
        raise ValueError("Pack target set does not match the active transformer")
    for target in targets:
        validate_rank0_rtn_record(pack[target])


def load_rank0_rtn_pack(
    path: str | Path, *, expected_targets: Sequence[str] | None = None
) -> Mapping[str, Any]:
    """Load a trusted tensor-only Omega-QVLA pack on CPU and validate it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Omega-QVLA pack does not exist: {path}")

    pack = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(pack, Mapping):
        raise ValueError("Omega-QVLA pack must deserialize to a mapping")
    validate_rank0_rtn_pack(pack, expected_targets=expected_targets)
    return pack
