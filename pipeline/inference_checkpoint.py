"""Inference checkpoint selection and compatibility validation helpers."""

from __future__ import annotations

import os
from typing import Any, cast

import torch

from .context import InferenceError, RunContext


def extract_checkpoint_state_dict(loaded_object: Any) -> dict[str, torch.Tensor]:
    """Extract a state dict from a checkpoint payload.

    The loader accepts either a raw ``state_dict`` mapping or a wrapper dict
    with one of the common keys used by training scripts.

    Args:
        loaded_object (Any): Object loaded from ``torch.load``.

    Returns:
        dict[str, torch.Tensor]: State dict mapping parameter names to tensors.

    Raises:
        InferenceError: If no compatible state dict is found.

    Examples:
        >>> payload = {"state_dict": {"w": torch.zeros(1)}}
        >>> sorted(extract_checkpoint_state_dict(payload).keys())
        ['w']
    """

    if isinstance(loaded_object, dict):
        if loaded_object and all(
            isinstance(value, torch.Tensor) for value in loaded_object.values()
        ):
            return cast(dict[str, torch.Tensor], loaded_object)
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = loaded_object.get(key)
            if isinstance(candidate, dict) and candidate and all(
                isinstance(value, torch.Tensor) for value in candidate.values()
            ):
                return cast(dict[str, torch.Tensor], candidate)
    raise InferenceError(
        "Unsupported checkpoint format. Expected a state_dict mapping or a dict "
        "containing one of: state_dict/model_state_dict/model."
    )


def validate_checkpoint_compatibility(
    head: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Validate key and shape compatibility between head and checkpoint.

    The validation is explicit so inference errors remain concise and actionable
    before ``load_state_dict`` is invoked in strict mode.

    Args:
        head (torch.nn.Module): Instantiated model head for inference.
        state_dict (dict[str, torch.Tensor]): Candidate checkpoint state dict.

    Raises:
        InferenceError: If keys or tensor shapes are incompatible.

    Examples:
        >>> head = torch.nn.Conv2d(3, 2, kernel_size=1)
        >>> validate_checkpoint_compatibility(head, head.state_dict())
    """

    model_state = head.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state_dict.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    shape_mismatches: list[str] = []
    for key in sorted(model_keys & ckpt_keys):
        model_shape = tuple(model_state[key].shape)
        ckpt_shape = tuple(state_dict[key].shape)
        if model_shape != ckpt_shape:
            shape_mismatches.append(
                f"{key}: model={model_shape} checkpoint={ckpt_shape}"
            )
    if not missing and not unexpected and not shape_mismatches:
        return
    sample_n = 5
    detail_lines = [
        f"missing_keys={len(missing)} sample={missing[:sample_n]}",
        f"unexpected_keys={len(unexpected)} sample={unexpected[:sample_n]}",
        "shape_mismatches="
        f"{len(shape_mismatches)} sample={shape_mismatches[:sample_n]}",
    ]
    raise InferenceError(
        "Checkpoint is incompatible with the configured head. "
        + " | ".join(detail_lines)
    )


def latest_phase_result(context: RunContext, phase_name: str) -> Any | None:
    """Return the most recent phase result by name.

    Args:
        context (RunContext): Active run context with previous phase results.
        phase_name (str): Phase name to look up.

    Returns:
        Any | None: Matching phase result object or ``None`` when absent.

    Examples:
        >>> ctx = RunContext(  # doctest: +ELLIPSIS
        ...     config={},
        ...     logger=None,
        ...     dist_ctx=...,
        ...     mlflow_logger=None,
        ...     hook_manager=None,
        ...     metrics_writer=None,
        ...     run_dir=...,
        ...     experiment_id="e",
        ...     run_id="r",
        ...     start_time=0.0,
        ...     config_path=None,
        ...     continue_on_error=True,
        ...     stability=...,
        ...     dataset_validation=...,
        ...     run_results=[],
        ... )
        >>> latest_phase_result(ctx, "train") is None
        True
    """

    results = context.run_results or []
    for result in reversed(results):
        if getattr(result, "name", None) == phase_name:
            return result
    return None


def resolve_inference_checkpoint(
    context: RunContext,
    infer_cfg: dict[str, Any],
) -> tuple[str, str]:
    """Resolve which checkpoint inference should load.

    Selection policy:
    1) Prefer the successful train-phase artifact from the current run.
    2) Otherwise use ``inference.checkpoint`` from config.
    3) If train failed in this run, abort to avoid stale checkpoint inference.

    Args:
        context (RunContext): Active run context.
        infer_cfg (dict[str, Any]): Inference configuration section.

    Returns:
        tuple[str, str]: ``(checkpoint_path, source_tag)``.

    Raises:
        InferenceError: If no valid checkpoint can be selected.

    Examples:
        >>> isinstance(resolve_inference_checkpoint, object)
        True
    """

    train_result = latest_phase_result(context, "train")
    if train_result is not None:
        if getattr(train_result, "status", "") == "success":
            artifacts = getattr(train_result, "artifacts", {}) or {}
            train_best = str(artifacts.get("best_checkpoint", "")).strip()
            if train_best and os.path.exists(train_best):
                return train_best, "train_phase_artifact"
        if getattr(train_result, "status", "") == "failed":
            raise InferenceError(
                "Training phase failed in the current run; refusing inference with "
                "a potentially stale checkpoint."
            )
    checkpoint = str(infer_cfg.get("checkpoint", "")).strip()
    if not checkpoint:
        raise InferenceError(
            "inference.checkpoint is required when no successful train-phase "
            "checkpoint artifact is available."
        )
    if not os.path.exists(checkpoint):
        raise InferenceError(f"Inference checkpoint not found: {checkpoint}")
    return checkpoint, "config"
