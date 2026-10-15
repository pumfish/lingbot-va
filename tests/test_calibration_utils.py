"""CPU tests for LingBot-VA calibration sample helpers."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "wan_va/modules/calibration_utils.py"
SPEC = importlib.util.spec_from_file_location("calibration_utils", MODULE_PATH)
calibration_utils = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(calibration_utils)


def test_resolve_dtype_and_move_tree_preserve_integer_tensor_dtype():
    moved = calibration_utils.move_tree_to_device(
        {"float": torch.ones(2, dtype=torch.float32), "integer": [torch.ones(1, dtype=torch.int64)]},
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert calibration_utils.resolve_dtype("float16") is torch.float16
    assert moved["float"].dtype is torch.bfloat16
    assert moved["integer"][0].dtype is torch.int64
    with pytest.raises(ValueError, match="Unsupported dtype"):
        calibration_utils.resolve_dtype("float32")


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"scope180")
    assert calibration_utils.sha256_file(path) == hashlib.sha256(b"scope180").hexdigest()


def test_load_calibration_samples_requires_complete_v1_manifest(tmp_path):
    samples = []
    for branch in ("latent", "action"):
        records = []
        for index in range(72):
            sample = tmp_path / f"{branch}_{index}.pt"
            sample.touch()
            samples.append(sample)
            records.append({"sample": str(sample)})
        if branch == "latent":
            latent_records = records
        else:
            action_records = records
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "lingbotva_calib_v1",
                "latent": {"records": latent_records},
                "action": {"records": action_records},
            }
        ),
        encoding="utf-8",
    )

    assert calibration_utils.load_calibration_samples(tmp_path) == samples

    action_records.pop()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "lingbotva_calib_v1",
                "latent": {"records": latent_records},
                "action": {"records": action_records},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Expected 72 action"):
        calibration_utils.load_calibration_samples(tmp_path)
