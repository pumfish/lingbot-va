"""Online token-wise group-Amax activation fake quantization for Ours."""

from __future__ import annotations

import torch

from .omega_qvla_pack import symmetric_qdq


def _qmax(bits: int) -> int:
    if bits not in (4, 8):
        raise ValueError(f"Token-group Amax supports only 4 or 8 bits, got {bits}")
    return (1 << (bits - 1)) - 1


class TokenGroupAmaxQdq:
    """Quantize every token and contiguous channel group from its own Amax.

    For input ``[..., C]`` and group size ``G``, each ``[..., C/G, G]``
    group uses ``scale = amax(abs(group)) / qmax``. No calibration data or
    offline sigma table is used.
    """

    def __init__(self, *, group_size: int = 128) -> None:
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        self.group_size = group_size

    def quantize(self, values: torch.Tensor, *, bits: int) -> torch.Tensor:
        """Apply online token-wise group-Amax symmetric fake-Q/DQ."""
        _qmax(bits)
        if values.ndim < 2:
            raise ValueError(f"Expected activation [..., channels], got {tuple(values.shape)}")
        if values.shape[-1] % self.group_size:
            raise ValueError(
                "Token-group Amax requires the channel count to be divisible by "
                f"{self.group_size}, got {values.shape[-1]}"
            )

        grouped = values.reshape(*values.shape[:-1], -1, self.group_size)
        scale = grouped.detach().float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        quantized = symmetric_qdq(grouped, scale / _qmax(bits), bits)
        return quantized.reshape_as(values).to(dtype=values.dtype)
