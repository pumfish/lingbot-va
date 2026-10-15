"""CPU tests for the portable scope-180 DuQuant checkpoint builder."""

import argparse
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "wan_va/script/modelopt_quantize_lingbotva_duquant_w4a4_scope180.py"
)
SPEC = importlib.util.spec_from_file_location("duquant_scope_builder", MODULE_PATH)
duquant_scope_builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(duquant_scope_builder)


def test_duquant_builder_records_rotation_topology_and_shared_scope(tmp_path):
    args = argparse.Namespace(
        model_path="/models/lingbot-va",
        calibration_root="/calibration/lingbotva_calib_v1",
        dtype="bfloat16",
        rotation_block_size=128,
        max_rotation_steps=8,
        permutation_times=1,
        duquant_seed=20260716,
    )
    duquant_config = duquant_scope_builder.build_duquant_config(args)
    manifest = duquant_scope_builder.build_manifest(
        args,
        calibration_count=144,
        calibration_sha256="abc",
        total_weight_bytes=456,
        scope={"logical_quantized_linear_count": 180, "quantized_weight_bytes": 123},
        quant_config={"algorithm": "max"},
        duquant_config=duquant_config,
        checkpoint=tmp_path / "transformer_modelopt_duquant_w4a4_scope180.pt",
    )

    assert duquant_config == {
        "schema_version": 2,
        "comparison_scope": "omega_qvla_dit_180",
        "block_size": 128,
        "max_rotation_steps": 8,
        "permutation_times": 1,
        "seed": 20260716,
    }
    assert manifest["baseline"] == "D5_Wan_adapted_DuQuant_W4A4_OmegaDiT180"
    assert manifest["logical_quantized_linear_count"] == 180
    assert manifest["duquant_config"] == duquant_config
