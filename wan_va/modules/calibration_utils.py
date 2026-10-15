"""Reusable helpers for LingBot-VA transformer calibration samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve a supported transformer floating-point dtype name."""
    try:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name!r}") from error


def move_tree_to_device(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Move nested tensor inputs while preserving non-floating tensor dtypes."""
    if isinstance(value, torch.Tensor):
        target_dtype = dtype if torch.is_floating_point(value) else value.dtype
        return value.to(device=device, dtype=target_dtype)
    if isinstance(value, dict):
        return {key: move_tree_to_device(item, device, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tree_to_device(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tree_to_device(item, device, dtype) for item in value)
    return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calibration_samples(root: Path) -> list[Path]:
    """Validate and return the 72 latent plus 72 action samples of calib_v1."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "lingbotva_calib_v1":
        raise ValueError(f"Expected lingbotva_calib_v1, got {manifest.get('dataset')!r}")

    samples: list[Path] = []
    for branch in ("latent", "action"):
        records = manifest.get(branch, {}).get("records", [])
        if len(records) != 72:
            raise ValueError(f"Expected 72 {branch} calibration records, found {len(records)}")
        samples.extend(Path(record["sample"]) for record in records)
    if len(samples) != 144 or not all(path.is_file() for path in samples):
        raise ValueError("calib_v1 sample manifest is incomplete")
    return samples


def calibration_forward(
    model: torch.nn.Module,
    sample_paths: list[Path],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Run all saved calibration samples through a transformer without gradients."""
    model.eval()
    with torch.inference_mode():
        for sample_path in sample_paths:
            payload = torch.load(sample_path, map_location="cpu", weights_only=False)
            _ = model(
                move_tree_to_device(payload["input_dict"], device, dtype),
                action_mode=bool(payload["action_mode"]),
                update_cache=0,
                cache_name="scope180_calibration",
            )
