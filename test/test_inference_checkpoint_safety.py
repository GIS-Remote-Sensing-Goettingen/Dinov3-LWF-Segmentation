"""Inference checkpoint safety tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_head  # noqa: E402
from pipeline.context import (  # noqa: E402
    DatasetValidationConfig,
    DistContext,
    PhaseResult,
    RunContext,
    StabilityConfig,
)
from pipeline.inference_checkpoint import (  # noqa: E402
    extract_checkpoint_state_dict,
    resolve_inference_checkpoint,
    validate_checkpoint_compatibility,
)


def _make_context(run_results: list[PhaseResult] | None = None) -> RunContext:
    """Create a minimal run context for helper-function tests.

    This helper avoids initializing the full runtime stack for unit tests that
    only need run-result history and basic context metadata.

    Args:
        run_results (list[PhaseResult] | None): Optional phase results list.

    Examples:
        >>> isinstance(_make_context().run_results, list)
        True
    """

    return RunContext(
        config={},
        logger=None,
        dist_ctx=DistContext(),
        mlflow_logger=None,
        hook_manager=None,
        metrics_writer=None,
        run_dir=Path("."),
        experiment_id="test",
        run_id="test",
        start_time=0.0,
        config_path=None,
        continue_on_error=True,
        stability=StabilityConfig(),
        dataset_validation=DatasetValidationConfig(),
        run_results=run_results or [],
    )


def test_resolve_inference_checkpoint_prefers_train_artifact(
    tmp_path: Path,
) -> None:
    """Use train-phase best checkpoint when available in current run.

    The resolver should prefer the train artifact over a stale configured path.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.

    Examples:
        >>> True
        True
    """

    best_ckpt = tmp_path / "dino_dense_probe_best.pth"
    torch.save({"dummy": torch.tensor(1)}, best_ckpt)
    train_result = PhaseResult(
        name="train",
        status="success",
        start_time=0.0,
        end_time=1.0,
        duration_s=1.0,
        metrics={},
        artifacts={"best_checkpoint": str(best_ckpt)},
        error=None,
    )
    context = _make_context(run_results=[train_result])
    selected, source = resolve_inference_checkpoint(
        context, {"checkpoint": "weights/some_old_checkpoint.pth"}
    )
    assert selected == str(best_ckpt)
    assert source == "train_phase_artifact"


def test_resolve_inference_checkpoint_blocks_after_train_failure(
    tmp_path: Path,
) -> None:
    """Abort inference when training failed in the same run.

    This ensures stale checkpoints are not silently reused after a failed train
    phase in the same execution.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.

    Examples:
        >>> True
        True
    """

    configured = tmp_path / "configured.pth"
    torch.save({"dummy": torch.tensor(1)}, configured)
    train_result = PhaseResult(
        name="train",
        status="failed",
        start_time=0.0,
        end_time=1.0,
        duration_s=1.0,
        metrics={},
        artifacts={},
        error=None,
    )
    context = _make_context(run_results=[train_result])
    with pytest.raises(RuntimeError, match="Training phase failed"):
        resolve_inference_checkpoint(context, {"checkpoint": str(configured)})


def test_checkpoint_compatibility_rejects_mismatched_state_dict() -> None:
    """Reject state dicts that do not match the configured head.

    The compatibility validator should raise before strict loading when keys or
    shapes do not match.

    Examples:
        >>> True
        True
    """

    head = build_head(
        "dino_dense_probe",
        num_classes=2,
        dino_channels=32,
        model_cfg={"dense_probe": {"norm_type": "none"}},
    )
    bad_state = {"wrong.weight": torch.randn(2, 2)}
    with pytest.raises(RuntimeError, match="Checkpoint is incompatible"):
        validate_checkpoint_compatibility(head, bad_state)


def test_extract_and_validate_checkpoint_state_dict_happy_path() -> None:
    """Accept valid nested checkpoint payload and matching state dict.

    Wrapped checkpoint payloads should be unwrapped and validated without
    raising errors when they match the configured head.

    Examples:
        >>> True
        True
    """

    head = build_head(
        "dino_dense_probe",
        num_classes=2,
        dino_channels=32,
        model_cfg={"dense_probe": {"norm_type": "none"}},
    )
    wrapped = {"state_dict": head.state_dict()}
    state_dict = extract_checkpoint_state_dict(wrapped)
    validate_checkpoint_compatibility(head, state_dict)
