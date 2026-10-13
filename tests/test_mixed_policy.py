"""Unit tests for Ours dual-branch mixed-precision policy parser."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).parents[1] / "wan_va/modules/mixed_policy.py"
SPEC = importlib.util.spec_from_file_location("mixed_policy", MODULE_PATH)
mixed_policy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mixed_policy
SPEC.loader.exec_module(mixed_policy)


def _payload() -> dict[str, object]:
    return {
        "format": "lingbotva_t2_dual_branch_cell_policy_v1",
        "mdp_scope": "all_mdps",
        "policy_name": "mask2_video_end_p1",
        "action_frame_token_count": 16,
        "video_frame_token_count": 120,
        "action_a8_cell_mask": [[[False, True] for _ in range(51)] for _ in range(30)],
        "video_a8_cell_mask": [[[True, False] for _ in range(26)] for _ in range(30)],
    }


def test_load_dual_branch_mixed_policy_reads_expected_masks(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    policy = mixed_policy.load_dual_branch_mixed_policy(path)

    assert policy.policy_name == "mask2_video_end_p1"
    assert policy.action_a8_frames(0, 0).tolist() == [False, True]
    assert policy.video_a8_frames(0, 0).tolist() == [True, False]


def test_load_dual_branch_mixed_policy_rejects_non_formal_scope(tmp_path):
    payload = _payload()
    payload["mdp_scope"] = "mdp0_then_a4"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="all_mdps"):
        mixed_policy.load_dual_branch_mixed_policy(path)
