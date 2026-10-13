"""Helpers for restoring ModelOpt checkpoints before model sharding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def restore_modelopt_checkpoint(
    transformer: Any,
    checkpoint_path: str,
    *,
    restore_fn: Callable[[Any, str], Any] | None = None,
    tensor_quantizer_type: type | None = None,
) -> list[tuple[str, Any]]:
    """Restore a ModelOpt checkpoint and return its enabled quantizers.

    The caller must invoke this before FSDP sharding so the checkpoint module
    names still match the unsharded transformer.
    """
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"ModelOpt checkpoint does not exist: {checkpoint_path}")

    if restore_fn is None or tensor_quantizer_type is None:
        from modelopt.torch.opt import restore
        import modelopt.torch.quantization  # noqa: F401  Registers quantization modes.
        from modelopt.torch.quantization.nn import TensorQuantizer

        restore_fn = restore if restore_fn is None else restore_fn
        tensor_quantizer_type = (
            TensorQuantizer if tensor_quantizer_type is None else tensor_quantizer_type
        )

    restore_fn(transformer, checkpoint_path)
    enabled_quantizers = [
        (name, module)
        for name, module in transformer.named_modules()
        if isinstance(module, tensor_quantizer_type) and module.is_enabled
    ]
    if not enabled_quantizers:
        raise RuntimeError(
            "The restored ModelOpt checkpoint did not enable any TensorQuantizer modules."
        )
    return enabled_quantizers
