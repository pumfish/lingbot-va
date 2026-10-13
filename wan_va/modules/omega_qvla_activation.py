"""Offline per-channel activation Q/DQ used by the original Omega-QVLA recipe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch

from .omega_qvla_pack import symmetric_qdq, validate_rank0_rtn_record

BranchName = Literal["latent", "action"]


class OmegaOfflineActivationQdq:
    """Select immutable offline scales and apply symmetric activation fake-Q/DQ.

    Latent activations use one scale per input channel. Action activations use
    either one scale per action denoising step or the original 8-bucket table.
    """

    def __init__(self, record: Mapping[str, Any]) -> None:
        validate_rank0_rtn_record(record)
        self._a_bits = int(record["a_bits"])
        self._input_features = int(record["in_features"])
        self._latent_static_scale = record["latent_static_scale"].detach().float().cpu()
        self._action_scale_table = record["act_scale_table"].detach().float().cpu()

    def scale_for(self, branch: BranchName, denoise_step: int | None = None) -> torch.Tensor:
        """Return the immutable per-channel scale vector for one branch state."""
        if branch == "latent":
            return self._latent_static_scale
        if branch != "action":
            raise ValueError(f"Unsupported Omega-QVLA branch: {branch!r}")
        if denoise_step is None or not 0 <= denoise_step < 51:
            raise ValueError("Action denoise_step must be an integer in [0, 50]")

        row_count = self._action_scale_table.shape[0]
        table_index = denoise_step if row_count == 51 else denoise_step * row_count // 51
        return self._action_scale_table[table_index]

    def quantize(
        self,
        values: torch.Tensor,
        branch: BranchName,
        denoise_step: int | None = None,
    ) -> torch.Tensor:
        """Apply the selected per-channel scale along the input-channel axis."""
        if values.shape[-1] != self._input_features:
            raise ValueError(
                f"Expected activation last dimension {self._input_features}, got {values.shape[-1]}"
            )
        scale = self.scale_for(branch, denoise_step)
        scale = scale.to(device=values.device).reshape((1,) * (values.ndim - 1) + (-1,))
        return symmetric_qdq(values, scale, self._a_bits).to(dtype=values.dtype)
