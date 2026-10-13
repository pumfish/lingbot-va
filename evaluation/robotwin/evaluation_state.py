"""Deterministic episode state helpers for paired RoboTwin evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def seed_task_environment(seed: int) -> None:
    """Reset process RNGs before constructing a RoboTwin scene."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_integer_list(raw_value: str, *, name: str, expected_count: int) -> list[int]:
    """Parse a comma- or whitespace-separated, unique list of integers."""
    try:
        values = [int(value) for value in raw_value.replace(",", " ").split()]
    except ValueError as error:
        raise ValueError(f"{name} must contain integers") from error
    if len(values) != expected_count:
        raise ValueError(f"{name} has {len(values)} entries, expected {expected_count}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def load_frozen_instructions(path: str | Path, task_seeds: list[int]) -> dict[int, str]:
    """Load one fixed language instruction for every paired task seed."""
    source = Path(path)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    instructions: dict[int, str] = {}
    for row in rows:
        task_seed = int(row["task_seed"])
        instruction = row["instruction"]
        if task_seed in instructions or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Frozen episodes require unique seeds and nonempty instructions")
        instructions[task_seed] = instruction
    missing = set(task_seeds) - instructions.keys()
    if missing:
        raise ValueError(f"Frozen episodes are missing task seeds: {sorted(missing)}")
    return instructions


def load_resume_rows(
    source: str | Path,
    *,
    expected_count: int,
    task_seeds: list[int],
    episode_ids: list[int],
    frozen_instructions: dict[int, str],
) -> list[dict[str, Any]]:
    """Load and validate a resumable prefix of paired episode results."""
    source_path = Path(source)
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len(rows) >= expected_count:
        raise ValueError(
            "Resume results must be a nonempty proper prefix of the requested episodes"
        )
    for index, row in enumerate(rows):
        if int(row["episode_index"]) != episode_ids[index]:
            raise ValueError("Resume episode indices do not match the requested prefix")
        if int(row["task_seed"]) != task_seeds[index]:
            raise ValueError("Resume task seeds do not match the requested prefix")
        if row["instruction"] != frozen_instructions[task_seeds[index]]:
            raise ValueError("Resume instructions do not match frozen instructions")
    return rows


def append_episode_result(path: str | Path, row: dict[str, Any]) -> None:
    """Append one completed episode record without overwriting previous results."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
