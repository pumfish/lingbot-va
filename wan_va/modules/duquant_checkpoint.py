"""Restore a LingbotVA DuQuant checkpoint into its explicit Wan topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .duquant_mapping import WanDuQuantTarget, prepare_wan_duquant_restore_topology
from .modelopt_checkpoint import restore_modelopt_checkpoint


@dataclass(frozen=True)
class WanDuQuantRestoreResult:
    """The prepared Wan projections and enabled ModelOpt quantizers."""

    targets: tuple[WanDuQuantTarget, ...]
    enabled_quantizers: tuple[tuple[str, Any], ...]


def restore_wan_duquant_checkpoint(
    transformer: Any,
    checkpoint_path: str,
    *,
    block_size: int = 128,
    permutation_times: int = 1,
) -> WanDuQuantRestoreResult:
    """Prepare the D5 topology, then restore its ModelOpt checkpoint.

    The topology must exist before ``mto.restore`` so checkpoint module names
    resolve to the same wrappers used when the DuQuant checkpoint was saved.
    """
    if block_size <= 0:
        raise ValueError(f"DuQuant block_size must be positive, got {block_size}")
    if permutation_times < 0:
        raise ValueError(
            f"DuQuant permutation_times must be non-negative, got {permutation_times}"
        )

    targets = prepare_wan_duquant_restore_topology(
        transformer,
        block_size=block_size,
        permutation_times=permutation_times,
    )
    enabled_quantizers = restore_modelopt_checkpoint(transformer, checkpoint_path)
    return WanDuQuantRestoreResult(targets, tuple(enabled_quantizers))
