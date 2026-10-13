"""Method schema for reproducible LingbotVA RoboTwin evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotWinMethod:
    """A quantization arm and the artifacts needed by its server entry."""

    name: str
    description: str
    artifact_names: tuple[str, ...]


METHODS: dict[str, RobotWinMethod] = {
    "bf16": RobotWinMethod("bf16", "Unquantized BF16 reference.", ()),
    "uniform_w4a4": RobotWinMethod(
        "uniform_w4a4",
        "Uniform ModelOpt RTN W4A4 checkpoint.",
        ("checkpoint",),
    ),
    "duquant_w4a4": RobotWinMethod(
        "duquant_w4a4",
        "DuQuant W4A4 checkpoint with the explicit Wan D5 topology.",
        ("checkpoint",),
    ),
    "omega_qvla_w4a4": RobotWinMethod(
        "omega_qvla_w4a4",
        "Omega-QVLA shared SVD+Hadamard W4A4 offline pack.",
        ("pack",),
    ),
    "ours_w4a4": RobotWinMethod(
        "ours_w4a4",
        "Omega rotation with online token-group Amax W4A4 and an all-MDP mixed policy.",
        ("pack", "policy"),
    ),
}


def resolve_robotwin_method(
    name: str,
    *,
    checkpoint: str | None = None,
    pack: str | None = None,
    policy: str | None = None,
    duquant_rotation_block_size: int = 128,
    duquant_permutation_times: int = 1,
) -> dict[str, str]:
    """Resolve a named method into portable server environment variables."""
    try:
        method = METHODS[name]
    except KeyError as error:
        choices = ", ".join(sorted(METHODS))
        raise ValueError(f"Unknown method {name!r}; choose one of: {choices}") from error

    artifacts = {"checkpoint": checkpoint, "pack": pack, "policy": policy}
    missing = [artifact for artifact in method.artifact_names if not artifacts[artifact]]
    if missing:
        raise ValueError(f"Method {name!r} requires: {', '.join(missing)}")
    unexpected = [
        artifact
        for artifact, value in artifacts.items()
        if value and artifact not in method.artifact_names
    ]
    if unexpected:
        raise ValueError(f"Method {name!r} does not use: {', '.join(unexpected)}")

    if duquant_rotation_block_size <= 0:
        raise ValueError("duquant_rotation_block_size must be positive")
    if duquant_permutation_times < 0:
        raise ValueError("duquant_permutation_times must be non-negative")

    environment: dict[str, str] = {}
    if name == "uniform_w4a4":
        environment["MODELOPT_RESTORE_FROM"] = str(checkpoint)
    elif name == "duquant_w4a4":
        environment.update(
            {
                "DUQUANT_RESTORE_FROM": str(checkpoint),
                "DUQUANT_ROTATION_BLOCK_SIZE": str(duquant_rotation_block_size),
                "DUQUANT_PERMUTATION_TIMES": str(duquant_permutation_times),
            }
        )
    elif name in {"omega_qvla_w4a4", "ours_w4a4"}:
        environment.update({"OMEGA_QVLA_PACK": str(pack), "OMEGA_QVLA_BRANCHES": "both"})
        if name == "ours_w4a4":
            environment["OURS_MIXED_POLICY"] = str(policy)
    return environment


def validate_method_artifacts(environment: dict[str, str]) -> None:
    """Fail before launching when a required checkpoint, pack, or policy is absent."""
    artifact_variables = (
        "MODELOPT_RESTORE_FROM",
        "DUQUANT_RESTORE_FROM",
        "OMEGA_QVLA_PACK",
        "OURS_MIXED_POLICY",
    )
    for variable in artifact_variables:
        value = environment.get(variable)
        if value and not Path(value).is_file():
            raise FileNotFoundError(f"{variable} does not exist: {value}")
