"""Explicit DuQuant projection mapping for LingbotVA's Wan blocks.

The official DuQuant repository targets decoder-only LLaMA/Mistral projections.
LingbotVA has a different block layout, so unsupported projections must be
reported rather than silently omitted during checkpoint construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from modelopt.torch.quantization.duquant import (
    DuQuantInputTransform,
    build_duquant_transform,
    fold_duquant_transform,
)


@dataclass(frozen=True)
class WanDuQuantTarget:
    """A LingbotVA Linear module matched to one official DuQuant projection role."""

    block_index: int
    official_role: str
    module_path: str
    transform_site: str


class DuQuantFoldedLinear(nn.Module):
    """Apply a DuQuant input transform before its equivalently folded Linear."""

    def __init__(self, transform: DuQuantInputTransform, linear: nn.Linear):
        super().__init__()
        if linear.in_features % transform.block_size:
            raise ValueError("DuQuant transform width does not match Linear input width")
        self.transform = transform
        self.linear = linear

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.transform(inputs))


def _require_linear(module: object, path: str) -> None:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"DuQuant expects {path} to be nn.Linear, got {type(module)}")


def resolve_wan_duquant_targets(transformer: nn.Module) -> tuple[WanDuQuantTarget, ...]:
    """Resolve all self-attention and FFN targets supported by the Wan mapping.

    Cross-attention is intentionally excluded: its K/V inputs are text features,
    whereas the official decoder-layer topology assumes a shared hidden-state
    input. Folding a decoder transform there requires a separate equivalence
    proof.
    """
    blocks = getattr(transformer, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("DuQuant expects transformer.blocks to be an nn.ModuleList")

    targets: list[WanDuQuantTarget] = []
    for block_index, block in enumerate(blocks):
        prefix = f"blocks.{block_index}"
        attn1 = getattr(block, "attn1", None)
        if not isinstance(attn1, nn.Module):
            raise TypeError(f"DuQuant expects {prefix}.attn1 to be an nn.Module")
        for official_role, name in (("q_proj", "to_q"), ("k_proj", "to_k"), ("v_proj", "to_v")):
            path = f"{prefix}.attn1.{name}"
            _require_linear(getattr(attn1, name, None), path)
            targets.append(
                WanDuQuantTarget(block_index, official_role, path, "self_attention_qkv_input")
            )

        to_out = getattr(attn1, "to_out", None)
        if not isinstance(to_out, nn.ModuleList) or not to_out:
            raise TypeError(f"DuQuant expects {prefix}.attn1.to_out to be a non-empty ModuleList")
        out_path = f"{prefix}.attn1.to_out.0"
        _require_linear(to_out[0], out_path)
        targets.append(WanDuQuantTarget(block_index, "o_proj", out_path, "self_attention_output"))

        ffn = getattr(block, "ffn", None)
        net = getattr(ffn, "net", None)
        if not isinstance(net, nn.ModuleList) or len(net) < 3:
            raise TypeError(
                f"DuQuant expects {prefix}.ffn.net to contain activation, "
                "dropout, and down projection"
            )
        up_proj = getattr(net[0], "proj", None)
        up_path = f"{prefix}.ffn.net.0.proj"
        _require_linear(up_proj, up_path)
        targets.append(WanDuQuantTarget(block_index, "up_proj", up_path, "ffn_input"))

        down_path = f"{prefix}.ffn.net.2"
        _require_linear(net[2], down_path)
        targets.append(
            WanDuQuantTarget(block_index, "down_proj", down_path, "ffn_activation_output")
        )

    return tuple(targets)


def _replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    if not parent_path:
        raise ValueError(f"DuQuant cannot replace root module {path}")
    parent = root.get_submodule(parent_path)
    if child_name.isdigit():
        if not isinstance(parent, nn.ModuleList):
            raise TypeError(f"DuQuant expected {parent_path} to be a ModuleList")
        parent[int(child_name)] = replacement
        return
    setattr(parent, child_name, replacement)


@torch.no_grad()
def attach_wan_duquant_transforms(
    transformer: nn.Module,
    *,
    block_size: int,
    max_rotation_steps: int,
    permutation_times: int,
    seed: int,
) -> tuple[WanDuQuantTarget, ...]:
    """Attach and fold DuQuant transforms for every supported Wan projection.

    The transform is applied before each Linear and folded into that Linear's
    weight. No quantizer is enabled here; callers must verify this BF16 path
    before adding quantization.
    """
    targets = resolve_wan_duquant_targets(transformer)
    for target_index, target in enumerate(targets):
        linear = transformer.get_submodule(target.module_path)
        _require_linear(linear, target.module_path)
        transform, _ = build_duquant_transform(
            linear.weight,
            block_size=block_size,
            max_rotation_steps=max_rotation_steps,
            permutation_times=permutation_times,
            seed=seed + target_index,
        )
        transform = transform.to(device=linear.weight.device, dtype=linear.weight.dtype)
        fold_duquant_transform(linear, transform)
        _replace_module(transformer, target.module_path, DuQuantFoldedLinear(transform, linear))
    return targets


@torch.no_grad()
def prepare_wan_duquant_restore_topology(
    transformer: nn.Module,
    *,
    block_size: int,
    permutation_times: int,
) -> tuple[WanDuQuantTarget, ...]:
    """Install D5 module names without rerunning the offline DuQuant search.

    ``mto.restore`` overwrites the transform buffers and folded weights from the
    checkpoint.  Recomputing the greedy rotations on every server start is both
    redundant and prohibitively slow for Wan's 180 target projections.
    """
    if permutation_times < 0:
        raise ValueError("permutation_times must be non-negative")

    targets = resolve_wan_duquant_targets(transformer)
    for target in targets:
        linear = transformer.get_submodule(target.module_path)
        _require_linear(linear, target.module_path)
        if linear.in_features % block_size:
            raise ValueError(
                f"DuQuant block size {block_size} is incompatible with {target.module_path}"
            )

        rotations = [
            torch.eye(block_size, dtype=torch.float32) for _ in range(permutation_times + 1)
        ]
        permutations = [
            torch.arange(linear.in_features, dtype=torch.long) for _ in range(permutation_times)
        ]
        transform = DuQuantInputTransform(rotations, permutations)
        _replace_module(transformer, target.module_path, DuQuantFoldedLinear(transform, linear))
    return targets
