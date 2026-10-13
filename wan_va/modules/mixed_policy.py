"""Parser for Ours dual-branch A4/A8 cell policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


POLICY_FORMAT = "lingbotva_t2_dual_branch_cell_policy_v1"


@dataclass(frozen=True)
class DualBranchMixedPolicy:
    """Immutable A8 cell masks for action and video diffusion branches."""

    action_a8_cells: torch.Tensor
    video_a8_cells: torch.Tensor
    policy_name: str

    def action_a8_frames(self, block_index: int, denoise_step: int) -> torch.Tensor:
        """Return the two action-frame A8 flags for a block and denoise step."""
        return self.action_a8_cells[block_index, denoise_step]

    def video_a8_frames(self, block_index: int, denoise_step: int) -> torch.Tensor:
        """Return the two video-frame A8 flags for a block and denoise step."""
        return self.video_a8_cells[block_index, denoise_step]


def load_dual_branch_mixed_policy(path: str | Path) -> DualBranchMixedPolicy:
    """Load and validate the all-MDP policy used by the formal Ours method."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mixed-precision policy does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != POLICY_FORMAT:
        raise ValueError(f"Expected policy format {POLICY_FORMAT!r}")
    if payload.get("mdp_scope") != "all_mdps":
        raise ValueError("Formal Ours policy must use mdp_scope='all_mdps'")
    if int(payload.get("action_frame_token_count", 0)) != 16:
        raise ValueError("Action policy must define 16 tokens per frame")
    if int(payload.get("video_frame_token_count", 0)) != 120:
        raise ValueError("Video policy must define 120 tokens per frame")

    action_cells = torch.tensor(payload.get("action_a8_cell_mask"), dtype=torch.bool)
    video_cells = torch.tensor(payload.get("video_a8_cell_mask"), dtype=torch.bool)
    if tuple(action_cells.shape) != (30, 51, 2):
        raise ValueError("action_a8_cell_mask must have shape [30, 51, 2]")
    if tuple(video_cells.shape) != (30, 26, 2):
        raise ValueError("video_a8_cell_mask must have shape [30, 26, 2]")

    return DualBranchMixedPolicy(
        action_a8_cells=action_cells,
        video_a8_cells=video_cells,
        policy_name=str(payload.get("policy_name", path.name)),
    )
