"""CPU tests for deterministic paired RoboTwin evaluation state."""

import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "evaluation/robotwin/evaluation_state.py"
SPEC = importlib.util.spec_from_file_location("robotwin_evaluation_state", MODULE_PATH)
evaluation_state = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = evaluation_state
SPEC.loader.exec_module(evaluation_state)


def test_seed_task_environment_reproduces_python_numpy_and_torch_randomness():
    evaluation_state.seed_task_environment(17)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    evaluation_state.seed_task_environment(17)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_parse_integer_list_rejects_missing_and_duplicate_entries():
    assert evaluation_state.parse_integer_list(
        "1, 2 3",
        name="TASK_SEEDS",
        expected_count=3,
    ) == [1, 2, 3]
    with pytest.raises(ValueError, match="expected 3"):
        evaluation_state.parse_integer_list("1 2", name="TASK_SEEDS", expected_count=3)
    with pytest.raises(ValueError, match="duplicates"):
        evaluation_state.parse_integer_list("1 1 2", name="TASK_SEEDS", expected_count=3)


def test_frozen_instructions_and_resume_rows_require_the_same_episode_prefix(tmp_path):
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text(
        "\n".join(
            [
                json.dumps({"task_seed": 11, "instruction": "pick the bottle"}),
                json.dumps({"task_seed": 12, "instruction": "put it down"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    instructions = evaluation_state.load_frozen_instructions(frozen, [11, 12])
    result_path = tmp_path / "episode_results.jsonl"
    evaluation_state.append_episode_result(
        result_path,
        {"episode_index": 0, "task_seed": 11, "instruction": "pick the bottle", "success": True},
    )

    rows = evaluation_state.load_resume_rows(
        result_path,
        expected_count=2,
        task_seeds=[11, 12],
        episode_ids=[0, 1],
        frozen_instructions=instructions,
    )

    assert rows[0]["success"] is True
