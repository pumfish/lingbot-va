"""CPU tests for LingbotVA's explicit DuQuant projection mapping."""

import importlib.util
from pathlib import Path
import sys
import types

import torch


MODULE_PATH = Path(__file__).parents[1] / "wan_va/modules/duquant_mapping.py"


class FakeDuQuantInputTransform(torch.nn.Module):
    """Identity transform exposing the public ModelOpt DuQuant interface."""

    def __init__(self, rotations, permutations) -> None:
        super().__init__()
        self.block_size = rotations[0].shape[0]
        self.rotations = rotations
        self.permutations = permutations

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def _install_fake_duquant_api(monkeypatch):
    calls: list[dict[str, int]] = []

    def build_duquant_transform(weight, *, block_size, max_rotation_steps, permutation_times, seed):
        calls.append(
            {
                "block_size": block_size,
                "max_rotation_steps": max_rotation_steps,
                "permutation_times": permutation_times,
                "seed": seed,
            }
        )
        rotations = [torch.eye(block_size)]
        permutations = [torch.arange(weight.shape[1]) for _ in range(permutation_times)]
        return FakeDuQuantInputTransform(rotations, permutations), None

    def fold_duquant_transform(linear, transform):
        del linear, transform

    duquant = types.ModuleType("modelopt.torch.quantization.duquant")
    duquant.DuQuantInputTransform = FakeDuQuantInputTransform
    duquant.build_duquant_transform = build_duquant_transform
    duquant.fold_duquant_transform = fold_duquant_transform
    monkeypatch.setitem(sys.modules, "modelopt.torch.quantization.duquant", duquant)
    return calls


def _load_mapping_module(monkeypatch):
    _install_fake_duquant_api(monkeypatch)
    spec = importlib.util.spec_from_file_location("duquant_mapping_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ToyUpProjection(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)


class ToyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = torch.nn.Module()
        self.attn1.to_q = torch.nn.Linear(8, 8)
        self.attn1.to_k = torch.nn.Linear(8, 8)
        self.attn1.to_v = torch.nn.Linear(8, 8)
        self.attn1.to_out = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
        self.ffn = torch.nn.Module()
        self.ffn.net = torch.nn.ModuleList(
            [ToyUpProjection(), torch.nn.Identity(), torch.nn.Linear(8, 8)]
        )


class ToyWanTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([ToyBlock(), ToyBlock()])


def test_resolve_wan_duquant_targets_maps_six_supported_projections(monkeypatch):
    mapping = _load_mapping_module(monkeypatch)
    targets = mapping.resolve_wan_duquant_targets(ToyWanTransformer())

    assert [target.official_role for target in targets] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
    ] * 2
    assert [target.module_path for target in targets[:6]] == [
        "blocks.0.attn1.to_q",
        "blocks.0.attn1.to_k",
        "blocks.0.attn1.to_v",
        "blocks.0.attn1.to_out.0",
        "blocks.0.ffn.net.0.proj",
        "blocks.0.ffn.net.2",
    ]


def test_attach_wan_duquant_transforms_wraps_every_target(monkeypatch):
    calls = _install_fake_duquant_api(monkeypatch)
    spec = importlib.util.spec_from_file_location("duquant_mapping_attach", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mapping = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mapping
    spec.loader.exec_module(mapping)
    transformer = ToyWanTransformer()
    inputs = torch.randn(2, 8)
    reference = transformer.blocks[0].attn1.to_q(inputs)

    targets = mapping.attach_wan_duquant_transforms(
        transformer,
        block_size=4,
        max_rotation_steps=5,
        permutation_times=2,
        seed=11,
    )

    assert len(targets) == 12
    assert [call["seed"] for call in calls] == list(range(11, 23))
    wrapped = transformer.blocks[0].attn1.to_q
    assert isinstance(wrapped, mapping.DuQuantFoldedLinear)
    torch.testing.assert_close(wrapped(inputs), reference)


def test_prepare_restore_topology_builds_identity_wrappers(monkeypatch):
    mapping = _load_mapping_module(monkeypatch)
    transformer = ToyWanTransformer()

    targets = mapping.prepare_wan_duquant_restore_topology(
        transformer,
        block_size=4,
        permutation_times=2,
    )

    assert len(targets) == 12
    wrapped = transformer.blocks[1].ffn.net[2]
    assert isinstance(wrapped, mapping.DuQuantFoldedLinear)
    assert wrapped.transform.block_size == 4
    assert len(wrapped.transform.rotations) == 3
    assert len(wrapped.transform.permutations) == 2
