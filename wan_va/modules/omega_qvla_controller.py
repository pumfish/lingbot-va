"""Runtime controller for an offline Omega-QVLA rank-0 RTN pack."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from .omega_qvla_activation import OmegaOfflineActivationQdq
from .omega_qvla_pack import load_rank0_rtn_pack
from .omega_qvla_rotation import OmegaQVLAInputRotation, OmegaQVLAOutputRotation

BranchName = Literal["latent", "action"]


class OmegaQVLAOfflinePackController:
    """Apply one immutable Omega-QVLA pack during selected denoising steps."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        pack_path: str | Path,
        *,
        quantized_branches: Literal["both", "latent", "action"] = "both",
    ) -> None:
        branch_sets = {
            "both": frozenset(("latent", "action")),
            "latent": frozenset(("latent",)),
            "action": frozenset(("action",)),
        }
        if quantized_branches not in branch_sets:
            raise ValueError(f"Unsupported Omega-QVLA branch selection: {quantized_branches!r}")

        pack = load_rank0_rtn_pack(pack_path)
        self._quantized_branches = branch_sets[quantized_branches]
        self._active_branch: BranchName | None = None
        self._active_step: int | None = None
        self._weights_enabled = False
        self._linears: dict[str, torch.nn.Linear] = {}
        self._weights: dict[str, torch.Tensor] = {}
        self._biases: dict[str, torch.Tensor | None] = {}
        self._quantized_weights: dict[str, torch.Tensor] = {}
        self._input_rotations: dict[str, OmegaQVLAInputRotation] = {}
        self._output_rotations: dict[str, OmegaQVLAOutputRotation] = {}
        self._activation_qdqs: dict[str, OmegaOfflineActivationQdq] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for name in pack["__meta__"]["targets"]:
            linear = transformer.get_submodule(name)
            if not isinstance(linear, torch.nn.Linear):
                raise TypeError(f"Omega-QVLA target {name} is not torch.nn.Linear: {type(linear)}")
            record: Mapping[str, object] = pack[name]
            self._linears[name] = linear
            self._weights[name] = linear.weight.detach().clone().contiguous()
            self._biases[name] = (
                None if linear.bias is None else linear.bias.detach().clone().contiguous()
            )
            self._quantized_weights[name] = record["weight_res_q"].to(
                device=linear.weight.device, dtype=linear.weight.dtype
            ).contiguous()
            self._input_rotations[name] = OmegaQVLAInputRotation(
                permutation=record["duquant_rotation_perm"].detach().long().cpu(),
                blocks=record["duquant_rotation_blocks"].detach().float().cpu(),
                block_size=int(record["rotation_block_size"]),
                mode=str(record["rotation_mode"]),
            )
            self._output_rotations[name] = OmegaQVLAOutputRotation(
                blocks=record["duquant_rotation_out_blocks"].detach().float().cpu(),
                block_size=int(record["rotation_out_block_size"]),
            )
            self._activation_qdqs[name] = OmegaOfflineActivationQdq(record)
            self._handles.append(linear.register_forward_pre_hook(self._pre_hook(name)))
            self._handles.append(linear.register_forward_hook(self._post_hook(name)))

    @property
    def target_names(self) -> tuple[str, ...]:
        """Return the exact pack targets attached to the transformer."""
        return tuple(sorted(self._linears))

    def activate(self, branch: BranchName, denoise_step: int) -> None:
        """Enable the pack for one latent or action denoising step."""
        if branch not in {"latent", "action"}:
            raise ValueError(f"Unsupported Omega-QVLA branch: {branch!r}")
        step_count = 26 if branch == "latent" else 51
        if not 0 <= denoise_step < step_count:
            raise ValueError(f"{branch} denoise_step must be in [0, {step_count - 1}]")
        if branch not in self._quantized_branches:
            self.deactivate()
            return

        self._enable_quantized_weights()
        self._active_branch = branch
        self._active_step = denoise_step

    def deactivate(self) -> None:
        """Restore BF16 Linear parameters for cache construction or bypasses."""
        self._active_branch = None
        self._active_step = None
        if not self._weights_enabled:
            return
        for name, linear in self._linears.items():
            linear.weight.data.copy_(self._weights[name].to(device=linear.weight.device))
            if linear.bias is not None:
                bias = self._biases[name]
                assert bias is not None
                linear.bias.data.copy_(bias.to(device=linear.bias.device))
        self._weights_enabled = False

    def close(self) -> None:
        """Restore original weights and remove the registered module hooks."""
        self.deactivate()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _enable_quantized_weights(self) -> None:
        if self._weights_enabled:
            return
        for name, linear in self._linears.items():
            linear.weight.data.copy_(self._quantized_weights[name])
            if linear.bias is not None:
                linear.bias.data.zero_()
        self._weights_enabled = True

    def _pre_hook(self, name: str):
        def hook(_linear: torch.nn.Module, args: tuple[object, ...]):
            if self._active_branch is None or self._active_step is None:
                return None
            if len(args) != 1 or not isinstance(args[0], torch.Tensor):
                raise RuntimeError(f"Omega-QVLA expected one Tensor input for {name}")
            values = args[0]
            transformed = self._input_rotations[name].apply(values)
            quantized = self._activation_qdqs[name].quantize(
                transformed,
                self._active_branch,
                self._active_step,
            )
            return (quantized,)

        return hook

    def _post_hook(self, name: str):
        def hook(_linear: torch.nn.Module, _args: tuple[object, ...], output: torch.Tensor):
            if self._active_branch is None:
                return output
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"Omega-QVLA expected Tensor output for {name}")
            restored = self._output_rotations[name].apply(output)
            bias = self._biases[name]
            if bias is not None:
                restored = restored + bias.to(device=restored.device, dtype=restored.dtype)
            return restored

        return hook
