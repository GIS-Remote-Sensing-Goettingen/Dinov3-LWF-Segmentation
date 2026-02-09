"""Context and result structures for pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Configuration parsing or validation error.

    This signals missing required fields or incompatible configuration values.

    Examples:
        >>> isinstance(ConfigError("x"), RuntimeError)
        True
    """


class DataError(RuntimeError):
    """Dataset preparation or validation error.

    Examples:
        >>> isinstance(DataError("x"), RuntimeError)
        True
    """


class TrainingError(RuntimeError):
    """Training execution error.

    Examples:
        >>> isinstance(TrainingError("x"), RuntimeError)
        True
    """


class InferenceError(RuntimeError):
    """Inference execution error.

    Examples:
        >>> isinstance(InferenceError("x"), RuntimeError)
        True
    """


@dataclass
class DistContext:
    """Distributed training context container.

    Attributes:
        enabled (bool): True if distributed execution is active.
        rank (int): Rank for the current process.
        world_size (int): Total number of distributed processes.
        local_rank (int): Local rank for the current process.

    Examples:
        >>> DistContext().is_main
        True
    """

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        """Return True when the context represents rank 0.

        Returns:
            bool: True when this process is the main rank.
        """

        return not self.enabled or self.rank == 0


@dataclass
class StabilityConfig:
    """Training-time numeric stability controls.

    Attributes:
        amp_enabled (str): Autocast mode (auto, on, off).
        amp_dtype (str): Preferred autocast dtype (bf16 or fp16).
        loss_fp32 (bool): Compute loss terms in fp32.
        grad_clip_norm (float): Max norm for gradient clipping.
        max_abs_logit_warn (float): Warning threshold for absolute logits.
        nonfinite_action (str): Action on non-finite values.
        nonfinite_max_consecutive_batches (int): Max consecutive non-finite batches.
        nonfinite_max_total_batches_per_epoch (int): Max total non-finite batches.
        save_bad_batch_sample (bool): Persist sample tensors on failures.
        check_params_every_steps (int): Parameter finite-check cadence.

    Examples:
        >>> StabilityConfig().nonfinite_action
        'stop_run'
    """

    amp_enabled: str = "auto"
    amp_dtype: str = "bf16"
    loss_fp32: bool = True
    grad_clip_norm: float = 1.0
    max_abs_logit_warn: float = 80.0
    nonfinite_action: str = "stop_run"
    nonfinite_max_consecutive_batches: int = 2
    nonfinite_max_total_batches_per_epoch: int = 5
    save_bad_batch_sample: bool = True
    check_params_every_steps: int = 50


@dataclass
class DatasetValidationConfig:
    """Dataset semantic validation controls for cached tiles.

    Attributes:
        enabled (bool): Whether semantic validation is active.
        allowed_labels (tuple[int, ...]): Valid class labels before mapping.
        ignore_index (int | None): Ignore index used for invalid labels.
        out_of_range_policy (str): Handling mode for invalid labels.
        require_finite_features (bool): Require finite cached features.
        require_finite_images (bool): Require finite input images.

    Examples:
        >>> DatasetValidationConfig().ignore_index
        255
    """

    enabled: bool = True
    allowed_labels: tuple[int, ...] = (0, 1)
    ignore_index: int | None = 255
    out_of_range_policy: str = "map_to_ignore"
    require_finite_features: bool = True
    require_finite_images: bool = True


@dataclass
class PhaseError:
    """Structured error metadata for a phase failure.

    Args:
        type (str): Error type name.
        message (str): Human-readable error message.
        details (str | None): Optional diagnostic details.

    Examples:
        >>> PhaseError(type="ValueError", message="bad", details=None).type
        'ValueError'
    """

    type: str
    message: str
    details: str | None


@dataclass
class PhaseOutcome:
    """Phase execution output without timing metadata.

    Args:
        metrics (dict[str, float]): Metric summary for the phase.
        artifacts (dict[str, str]): Key artifact paths produced by the phase.

    Examples:
        >>> PhaseOutcome(metrics={}, artifacts={})
        PhaseOutcome(metrics={}, artifacts={})
    """

    metrics: dict[str, float]
    artifacts: dict[str, str]


@dataclass
class PhaseResult:
    """Completed phase record with timing, metrics, and error data.

    Args:
        name (str): Phase name.
        status (str): Status of the phase (success, failed, skipped).
        start_time (float): Epoch time in seconds when the phase started.
        end_time (float): Epoch time in seconds when the phase ended.
        duration_s (float): Duration of the phase in seconds.
        metrics (dict[str, float]): Phase-level metric summary.
        artifacts (dict[str, str]): Artifact paths generated by the phase.
        error (PhaseError | None): Error metadata if the phase failed.

    Examples:
        >>> PhaseResult(
        ...     name="prepare",
        ...     status="success",
        ...     start_time=0.0,
        ...     end_time=1.0,
        ...     duration_s=1.0,
        ...     metrics={},
        ...     artifacts={},
        ...     error=None,
        ... ).status
        'success'
    """

    name: str
    status: str
    start_time: float
    end_time: float
    duration_s: float
    metrics: dict[str, float]
    artifacts: dict[str, str]
    error: PhaseError | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the phase result into a dictionary.

        Returns:
            dict[str, Any]: JSON-serializable representation of the phase.

        Examples:
            >>> PhaseResult(
            ...     name="verify",
            ...     status="skipped",
            ...     start_time=0.0,
            ...     end_time=0.0,
            ...     duration_s=0.0,
            ...     metrics={},
            ...     artifacts={},
            ...     error=None,
            ... ).to_dict()["status"]
            'skipped'
        """

        return {
            "name": self.name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.duration_s,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "error": (
                None
                if self.error is None
                else {
                    "type": self.error.type,
                    "message": self.error.message,
                    "details": self.error.details,
                }
            ),
        }


@dataclass
class RunContext:
    """Shared context for the entire pipeline run.

    Args:
        config (dict): Loaded configuration dictionary.
        logger (VerbosityLogger): Verbosity-aware logger.
        dist_ctx (DistContext): Distributed execution context.
        mlflow_logger (MlflowFileLogger | None): MLflow-compatible file logger.
        hook_manager (HookManager): Hook manager instance.
        metrics_writer (MetricsWriter | None): JSONL metrics writer.
        run_dir (Path): MLflow run directory.
        experiment_id (str): MLflow experiment ID.
        run_id (str): MLflow run ID.
        start_time (float): Run start time in seconds since epoch.
        config_path (str | None): Resolved configuration path.
        continue_on_error (bool): Whether to continue after phase failures.
        stability (StabilityConfig): Numeric stability settings.
        dataset_validation (DatasetValidationConfig): Dataset validation settings.

    Examples:
        >>> isinstance(RunContext, type)
        True
    """

    config: dict[str, Any]
    logger: Any
    dist_ctx: DistContext
    mlflow_logger: Any
    hook_manager: Any
    metrics_writer: Any
    run_dir: Path
    experiment_id: str
    run_id: str
    start_time: float
    config_path: str | None
    continue_on_error: bool
    stability: StabilityConfig
    dataset_validation: DatasetValidationConfig
    run_results: list[PhaseResult] | None = None
