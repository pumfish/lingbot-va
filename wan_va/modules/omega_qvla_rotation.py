"""Omega-QVLA A2-lite input rotations for LingbotVA Linear layers.

The implementation mirrors the released Omega-QVLA ``svd_hadamard`` path for
PyTorch Linear weights shaped ``[out_features, in_features]``.  It deliberately
keeps the rotation in block form: Wan FFN projections are too wide to materialize
their full dense input rotation matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

RotationMode = Literal["raw", "svd", "hadamard", "svd_hadamard"]


def _normalized_hadamard(size: int, device: torch.device) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a positive power of two, got {size}")
    matrix = torch.ones((1, 1), device=device, dtype=torch.float32)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / size**0.5


def _zigzag_permutation(channel_energy: torch.Tensor) -> torch.Tensor:
    """Match Omega-QVLA's descending high/low channel interleaving."""
    order = torch.argsort(channel_energy, descending=True)
    left, right = 0, len(order) - 1
    indices: list[int] = []
    take_left = True
    while left <= right:
        if take_left:
            indices.append(int(order[left]))
            left += 1
        else:
            indices.append(int(order[right]))
            right -= 1
        take_left = not take_left
    return torch.tensor(indices, dtype=torch.long, device=channel_energy.device)


@dataclass(frozen=True)
class OmegaQVLAInputRotation:
    """Permutation plus independent right rotations on input-channel blocks."""

    permutation: torch.Tensor
    blocks: torch.Tensor
    block_size: int
    mode: RotationMode

    @property
    def input_features(self) -> int:
        return int(self.permutation.numel())

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Apply ``X[:, perm] @ R_block`` to the last tensor axis."""
        if values.shape[-1] != self.input_features:
            raise ValueError(
                f"Expected last dimension {self.input_features}, got {values.shape[-1]}"
            )
        if self.mode == "raw":
            return values
        rotated = self.apply_fp32(values)
        return rotated.to(dtype=values.dtype)

    def apply_fp32(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the rotation and retain FP32 output for precision audits."""
        if values.shape[-1] != self.input_features:
            raise ValueError(
                f"Expected last dimension {self.input_features}, got {values.shape[-1]}"
            )
        permuted = values.float().index_select(-1, self.permutation.to(values.device))
        leading_shape = permuted.shape[:-1]
        block_count = self.blocks.shape[0]
        grouped = permuted.reshape(-1, block_count, self.block_size)
        # Run the rotation in FP32: BF16 einsum loses ~4e-3 relative per
        # operation, which compounds to ~3-5% across 30 blocks x 6 targets
        # and breaks the closed-loop action branch (audit 2026-08-14).
        rotated = torch.einsum(
            "nkb,kbc->nkc",
            grouped,
            self.blocks.to(device=values.device, dtype=torch.float32),
        )
        return rotated.reshape(*leading_shape, self.input_features)

    def apply_in_dtype(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the rotation entirely in ``values.dtype`` for a BF16 control."""
        if values.shape[-1] != self.input_features:
            raise ValueError(
                f"Expected last dimension {self.input_features}, got {values.shape[-1]}"
            )
        permuted = values.index_select(-1, self.permutation.to(values.device))
        leading_shape = permuted.shape[:-1]
        block_count = self.blocks.shape[0]
        grouped = permuted.reshape(-1, block_count, self.block_size)
        rotated = torch.einsum(
            "nkb,kbc->nkc", grouped, self.blocks.to(device=values.device, dtype=values.dtype)
        )
        return rotated.reshape(*leading_shape, self.input_features)

    def fold_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Return ``W[:, perm] @ R_block`` for a PyTorch Linear weight."""
        if weight.ndim != 2 or weight.shape[-1] != self.input_features:
            raise ValueError(
                "Omega-QVLA rotation expects Linear weight [out_features, "
                f"{self.input_features}], got {tuple(weight.shape)}"
            )
        return self.apply(weight)

    def fold_weight_fp32(self, weight: torch.Tensor) -> torch.Tensor:
        """Fold the input rotation into a weight while retaining FP32."""
        return self.apply_fp32(weight)

    def fold_weight_in_dtype(self, weight: torch.Tensor) -> torch.Tensor:
        """Fold the input rotation with the weight dtype as a BF16 control."""
        return self.apply_in_dtype(weight)


@dataclass(frozen=True)
class OmegaQVLAOutputRotation:
    """Official A2-lite row rotation, restored after the quantized Linear."""

    blocks: torch.Tensor
    block_size: int

    @property
    def output_features(self) -> int:
        return int(self.blocks.shape[0] * self.block_size)

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the block-diagonal output restoration ``Y @ R_out``."""
        if values.shape[-1] != self.output_features:
            raise ValueError(
                f"Expected last dimension {self.output_features}, got {values.shape[-1]}"
            )
        restored = self.apply_fp32(values)
        return restored.to(dtype=values.dtype)

    def apply_fp32(self, values: torch.Tensor) -> torch.Tensor:
        """Restore the output basis and retain FP32 output for precision audits."""
        if values.shape[-1] != self.output_features:
            raise ValueError(
                f"Expected last dimension {self.output_features}, got {values.shape[-1]}"
            )
        leading_shape = values.shape[:-1]
        grouped = values.float().reshape(-1, self.blocks.shape[0], self.block_size)
        restored = torch.einsum(
            "nkb,kbc->nkc",
            grouped,
            self.blocks.to(device=values.device, dtype=torch.float32),
        )
        return restored.reshape(*leading_shape, self.output_features)

    def apply_in_dtype(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the output rotation entirely in ``values.dtype`` for a BF16 control."""
        if values.shape[-1] != self.output_features:
            raise ValueError(
                f"Expected last dimension {self.output_features}, got {values.shape[-1]}"
            )
        leading_shape = values.shape[:-1]
        grouped = values.reshape(-1, self.blocks.shape[0], self.block_size)
        restored = torch.einsum(
            "nkb,kbc->nkc", grouped, self.blocks.to(device=values.device, dtype=values.dtype)
        )
        return restored.reshape(*leading_shape, self.output_features)

    def fold_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Return the official row-rotated weight ``R_out @ W``."""
        if weight.ndim != 2 or weight.shape[0] != self.output_features:
            raise ValueError(
                "Omega-QVLA output rotation expects Linear weight "
                f"[{self.output_features}, in_features], got {tuple(weight.shape)}"
            )
        return self.fold_weight_fp32(weight).to(dtype=weight.dtype)

    def fold_weight_fp32(self, weight: torch.Tensor) -> torch.Tensor:
        """Fold output rotation into a weight while retaining FP32."""
        if weight.ndim != 2 or weight.shape[0] != self.output_features:
            raise ValueError(
                "Omega-QVLA output rotation expects Linear weight "
                f"[{self.output_features}, in_features], got {tuple(weight.shape)}"
            )
        grouped = weight.float().reshape(self.blocks.shape[0], self.block_size, weight.shape[1])
        rotated = torch.einsum(
            "kab,kbc->kac",
            self.blocks.to(device=weight.device, dtype=torch.float32),
            grouped,
        )
        return rotated.reshape_as(weight)

    def fold_weight_in_dtype(self, weight: torch.Tensor) -> torch.Tensor:
        """Fold the output rotation with the weight dtype as a BF16 control."""
        if weight.ndim != 2 or weight.shape[0] != self.output_features:
            raise ValueError(
                "Omega-QVLA output rotation expects Linear weight "
                f"[{self.output_features}, in_features], got {tuple(weight.shape)}"
            )
        grouped = weight.reshape(self.blocks.shape[0], self.block_size, weight.shape[1])
        rotated = torch.einsum(
            "kab,kbc->kac", self.blocks.to(device=weight.device, dtype=weight.dtype), grouped
        )
        return rotated.reshape_as(weight)


@torch.no_grad()
def build_omega_qvla_input_rotation(
    weight: torch.Tensor,
    *,
    block_size: int = 64,
    mode: RotationMode = "svd_hadamard",
    enable_permutation: bool = True,
    lambda_smooth: float = 0.15,
) -> OmegaQVLAInputRotation:
    """Build the released Omega-QVLA blockwise input transform.

    ``svd`` and ``svd_hadamard`` compute the left singular vectors of each
    ``W_block.T``.  This is exactly the orientation used by the released
    ``duquant_preprocess.compute_block_rotation`` implementation.
    """
    if mode not in {"raw", "svd", "hadamard", "svd_hadamard"}:
        raise ValueError(f"Unsupported Omega-QVLA rotation mode: {mode!r}")
    if not 0.0 <= lambda_smooth <= 1.0:
        raise ValueError(f"lambda_smooth must be in [0, 1], got {lambda_smooth}")
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D Linear weight, got {tuple(weight.shape)}")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"--block-size must be a positive power of two, got {block_size}")
    in_features = int(weight.shape[-1])
    if in_features % block_size:
        raise ValueError(
            f"Input width {in_features} is not divisible by Omega block size {block_size}"
        )

    device = weight.device
    block_count = in_features // block_size
    identity = torch.eye(block_size, dtype=torch.float32, device=device)
    if mode == "raw":
        return OmegaQVLAInputRotation(
            torch.arange(in_features, dtype=torch.long, device=device),
            identity.expand(block_count, -1, -1).clone(),
            block_size,
            mode,
        )

    weight_fp32 = weight.detach().float()
    if enable_permutation:
        energy = weight_fp32.square().mean(dim=0)
        # The released pack builder applies this affine smoothing before zigzag.
        energy = (1.0 - lambda_smooth) * energy + lambda_smooth * energy.mean()
        permutation = _zigzag_permutation(energy)
    else:
        permutation = torch.arange(in_features, dtype=torch.long, device=device)
    permuted_weight = weight_fp32.index_select(1, permutation)
    hadamard = _normalized_hadamard(block_size, device)
    rotations: list[torch.Tensor] = []
    for index in range(block_count):
        start = index * block_size
        block = permuted_weight[:, start : start + block_size]
        if mode == "hadamard":
            rotation = hadamard
        else:
            left, _, _ = torch.linalg.svd(block.transpose(0, 1), full_matrices=True)
            rotation = left @ hadamard if mode == "svd_hadamard" else left
        rotations.append(rotation)
    return OmegaQVLAInputRotation(
        permutation=permutation,
        blocks=torch.stack(rotations).contiguous(),
        block_size=block_size,
        mode=mode,
    )


@torch.no_grad()
def build_omega_qvla_output_rotation(
    weight: torch.Tensor,
    *,
    block_size: int = 64,
) -> OmegaQVLAOutputRotation:
    """Build Omega-QVLA's official blockwise output-side SVD rotation.

    The released DiT pack rotates output rows before weight quantization and
    restores that basis after the Linear. This improves weight-side balance
    while preserving the full-precision function.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D Linear weight, got {tuple(weight.shape)}")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a positive power of two, got {block_size}")
    out_features = int(weight.shape[0])
    if out_features % block_size:
        raise ValueError(
            f"Output width {out_features} is not divisible by Omega block size {block_size}"
        )
    blocks = []
    weight_fp32 = weight.detach().float()
    for start in range(0, out_features, block_size):
        rows = weight_fp32[start : start + block_size]
        left, _, _ = torch.linalg.svd(rows, full_matrices=True)
        blocks.append(left)
    return OmegaQVLAOutputRotation(torch.stack(blocks).contiguous(), block_size)
