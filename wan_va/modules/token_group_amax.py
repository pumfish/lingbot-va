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

    def quantize_mixed_by_token(self, values: torch.Tensor, token_bits: torch.Tensor) -> torch.Tensor:
        """Apply A4/A8 Q/DQ where every token receives its selected bit width."""
        if values.ndim < 3:
            raise ValueError(f"Expected activation [batch, tokens, channels], got {tuple(values.shape)}")
        if values.shape[-1] % self.group_size:
            raise ValueError(
                "Token-group Amax requires the channel count to be divisible by "
                f"{self.group_size}, got {values.shape[-1]}"
            )
        if token_bits.ndim != 1 or token_bits.numel() != values.shape[-2]:
            raise ValueError(
                "token_bits must contain one 4/8-bit value per activation token: "
                f"tokens={values.shape[-2]}, bits_shape={tuple(token_bits.shape)}"
            )
        if not torch.all((token_bits == 4) | (token_bits == 8)):
            raise ValueError("token_bits must contain only 4 or 8")

        grouped = values.reshape(*values.shape[:-1], -1, self.group_size)
        qmax = torch.where(token_bits.to(device=values.device) == 8, 127.0, 7.0)
        qmax = qmax.reshape((1,) * (values.ndim - 2) + (-1, 1, 1))
        scale = grouped.detach().float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        integers = torch.round(grouped.float() / scale)
        integers = torch.maximum(torch.minimum(integers, qmax), -qmax - 1)
        return (integers * scale).reshape_as(values).to(dtype=values.dtype)
