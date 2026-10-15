"""CPU tests for the portable scope-180 Uniform checkpoint builder."""

import argparse
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "wan_va/script/modelopt_quantize_lingbotva_uniform_w4a4_scope180.py"
)
SPEC = importlib.util.spec_from_file_location("uniform_scope_builder", MODULE_PATH)
uniform_scope_builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(uniform_scope_builder)


def test_build_manifest_records_shared_scope_and_checkpoint(tmp_path):
    args = argparse.Namespace(
        model_path="/models/lingbot-va",
        calibration_root="/calibration/lingbotva_calib_v1",
        dtype="bfloat16",
    )
    checkpoint = tmp_path / "transformer_modelopt_uniform_w4a4_scope180.pt"
    scope = {"quantized_linear_count": 180, "quantized_weight_bytes": 123}
    manifest = uniform_scope_builder.build_manifest(
        args,
        calibration_count=144,
        calibration_sha256="abc",
        total_weight_bytes=456,
        scope=scope,
        quant_config={"algorithm": "max"},
        checkpoint=checkpoint,
    )

    assert manifest["baseline"] == "Uniform_RTN_W4A4_OmegaDiT180"
    assert manifest["comparison_scope"] == "omega_qvla_dit_180"
    assert manifest["calibration_sample_count"] == 144
    assert manifest["quantization"] == "weight_int4_per_channel_activation_int4_per_tensor"
    assert manifest["quantized_linear_count"] == 180
    assert manifest["checkpoint"] == str(checkpoint)
