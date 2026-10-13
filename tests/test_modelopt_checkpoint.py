"""Unit tests for ModelOpt checkpoint restoration without GPU dependencies."""

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "wan_va/modules/modelopt_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("modelopt_checkpoint", MODULE_PATH)
modelopt_checkpoint = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(modelopt_checkpoint)


class FakeTensorQuantizer:
    def __init__(self, is_enabled: bool):
        self.is_enabled = is_enabled


class FakeTransformer:
    def named_modules(self):
        return [
            ("enabled", FakeTensorQuantizer(True)),
            ("disabled", FakeTensorQuantizer(False)),
            ("plain", object()),
        ]


def test_restore_modelopt_checkpoint_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        modelopt_checkpoint.restore_modelopt_checkpoint(FakeTransformer(), str(tmp_path / "missing.pt"))


def test_restore_modelopt_checkpoint_returns_enabled_quantizers(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    calls = []

    def fake_restore(transformer, checkpoint_path):
        calls.append((transformer, checkpoint_path))

    transformer = FakeTransformer()
    quantizers = modelopt_checkpoint.restore_modelopt_checkpoint(
        transformer,
        str(checkpoint),
        restore_fn=fake_restore,
        tensor_quantizer_type=FakeTensorQuantizer,
    )

    assert calls == [(transformer, str(checkpoint))]
    assert [name for name, _ in quantizers] == ["enabled"]


def test_restore_modelopt_checkpoint_rejects_checkpoint_without_enabled_quantizers(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()

    class DisabledTransformer:
        def named_modules(self):
            return [("disabled", FakeTensorQuantizer(False))]

    with pytest.raises(RuntimeError, match="did not enable any TensorQuantizer"):
        modelopt_checkpoint.restore_modelopt_checkpoint(
            DisabledTransformer(),
            str(checkpoint),
            restore_fn=lambda *_: None,
            tensor_quantizer_type=FakeTensorQuantizer,
        )
