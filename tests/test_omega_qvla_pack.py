"""CPU checks for the public Omega-QVLA pack schema."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "wan_va" / "modules" / "omega_qvla_pack.py"
SPEC = importlib.util.spec_from_file_location("omega_qvla_pack", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
omega_qvla_pack = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = omega_qvla_pack
SPEC.loader.exec_module(omega_qvla_pack)


def _record() -> dict[str, object]:
    return {
        "format": "dit_svdquant_v1",
        "weight_res_q": torch.ones(8, 8),
        "lowrank_A": torch.empty(8, 0),
        "lowrank_B": torch.empty(8, 0),
        "act_scale_table": torch.ones(51, 8),
        "duquant_rotation_perm": torch.arange(8),
        "duquant_rotation_blocks": torch.eye(4).repeat(2, 1, 1),
        "duquant_rotation_out_blocks": torch.eye(4).repeat(2, 1, 1),
        "weight_bits": 4,
        "a_bits": 4,
        "rank": 0,
        "in_features": 8,
        "out_features": 8,
        "gptq_block_size": 128,
        "gptq_damp_percent": 0.05,
        "use_rtn": True,
        "rotation_mode": "svd_hadamard",
        "rotation_block_size": 4,
        "rotation_out_block_size": 4,
        "latent_static_scale": torch.ones(8),
    }


def test_validate_rank0_rtn_pack_and_qdq() -> None:
    pack = {
        "__meta__": {"format": "lingbotva_omega_qvla_official_dit_pack_v2", "targets": ["linear"]},
        "linear": _record(),
    }
    omega_qvla_pack.validate_rank0_rtn_pack(pack, expected_targets=("linear",))

    values = torch.tensor([-2.0, -0.1, 0.1, 2.0])
    result = omega_qvla_pack.symmetric_qdq(values, torch.tensor(0.25), bits=4)
    torch.testing.assert_close(result, torch.tensor([-2.0, -0.0, 0.0, 1.75]))


def test_load_rank0_rtn_pack_validates_saved_tensor_pack(tmp_path) -> None:
    pack = {
        "__meta__": {"format": "lingbotva_omega_qvla_official_dit_pack_v2", "targets": ["linear"]},
        "linear": _record(),
    }
    path = tmp_path / "omega.pt"
    torch.save(pack, path)

    loaded = omega_qvla_pack.load_rank0_rtn_pack(path, expected_targets=("linear",))

    assert loaded["__meta__"]["targets"] == ["linear"]


def test_validate_rank0_rtn_pack_rejects_non_permutation() -> None:
    record = _record()
    record["duquant_rotation_perm"] = torch.zeros(8, dtype=torch.long)
    pack = {
        "__meta__": {"format": "lingbotva_omega_qvla_official_dit_pack_v2", "targets": ["linear"]},
        "linear": record,
    }

    with pytest.raises(ValueError, match="channel permutation"):
        omega_qvla_pack.validate_rank0_rtn_pack(pack)
