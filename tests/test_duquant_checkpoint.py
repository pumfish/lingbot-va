"""CPU tests for the LingbotVA DuQuant checkpoint restore sequence."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest


MODULE_ROOT = Path(__file__).parents[1] / "wan_va/modules"


def _load_checkpoint_module(monkeypatch):
    package = sys.modules.setdefault("modules", types.ModuleType("modules"))
    package.__path__ = [str(MODULE_ROOT)]
    mapping = types.ModuleType("modules.duquant_mapping")
    mapping.WanDuQuantTarget = object
    mapping.prepare_wan_duquant_restore_topology = None
    checkpoint = types.ModuleType("modules.modelopt_checkpoint")
    checkpoint.restore_modelopt_checkpoint = None
    monkeypatch.setitem(sys.modules, "modules.duquant_mapping", mapping)
    monkeypatch.setitem(sys.modules, "modules.modelopt_checkpoint", checkpoint)

    qualified_name = "modules.duquant_checkpoint"
    spec = importlib.util.spec_from_file_location(
        qualified_name,
        MODULE_ROOT / "duquant_checkpoint.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def test_restore_prepares_duquant_topology_before_modelopt_restore(monkeypatch):
    restore_module = _load_checkpoint_module(monkeypatch)
    transformer = object()
    calls: list[tuple[str, object]] = []
    target = object()

    def prepare(model, *, block_size, permutation_times):
        calls.append(("prepare", model, block_size, permutation_times))
        return (target,)

    def restore(model, checkpoint_path):
        calls.append(("restore", model, checkpoint_path))
        return [("blocks.0.attn1.to_q.input_quantizer", object())]

    monkeypatch.setattr(restore_module, "prepare_wan_duquant_restore_topology", prepare)
    monkeypatch.setattr(restore_module, "restore_modelopt_checkpoint", restore)

    result = restore_module.restore_wan_duquant_checkpoint(
        transformer,
        "/tmp/duquant.pt",
        block_size=64,
        permutation_times=2,
    )

    assert [call[0] for call in calls] == ["prepare", "restore"]
    assert calls[0][1:] == (transformer, 64, 2)
    assert calls[1][1:] == (transformer, "/tmp/duquant.pt")
    assert result.targets == (target,)
    assert len(result.enabled_quantizers) == 1


@pytest.mark.parametrize(
    ("block_size", "permutation_times", "message"),
    [(0, 1, "block_size"), (128, -1, "permutation_times")],
)
def test_restore_rejects_invalid_duquant_topology_arguments(
    monkeypatch, block_size, permutation_times, message
):
    restore_module = _load_checkpoint_module(monkeypatch)

    with pytest.raises(ValueError, match=message):
        restore_module.restore_wan_duquant_checkpoint(
            object(),
            "/tmp/duquant.pt",
            block_size=block_size,
            permutation_times=permutation_times,
        )
