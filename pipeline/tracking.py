"""MLflow-compatible tracking, hooks, and processors."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from .context import PhaseError, PhaseResult, RunContext


class MetricsWriter:
    """Append metrics to a JSONL file for lightweight visualization.

    Args:
        path (Path): File path for the JSONL metrics.

    Examples:
        >>> isinstance(MetricsWriter, type)
        True
    """

    def __init__(self, path: Path) -> None:
        """Initialize the metrics writer.

        Args:
            path (Path): Path to the JSONL file.
        """

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        phase_name: str,
        step: int,
        metrics: dict[str, float],
        timestamp_ms: int | None = None,
    ) -> None:
        """Append a metrics record to the JSONL file.

        Args:
            phase_name (str): Name of the phase producing metrics.
            step (int): Step index for the metrics.
            metrics (dict[str, float]): Metric values.
            timestamp_ms (int | None): Unix time in milliseconds.
        """

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        record = {
            "timestamp_ms": timestamp_ms,
            "phase": phase_name,
            "step": step,
            "metrics": metrics,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


class MlflowFileLogger:
    """MLflow-compatible file logger without external dependencies.

    Args:
        tracking_dir (str): Root directory for MLflow files.
        experiment_id (str): Experiment identifier.
        run_id (str): Run identifier.
        run_name (str | None): Optional display name.
        tags (dict[str, str] | None): Optional tag map.

    Examples:
        >>> isinstance(MlflowFileLogger, type)
        True
    """

    def __init__(
        self,
        tracking_dir: str,
        experiment_id: str,
        run_id: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Initialize the MLflow-compatible file logger.

        Args:
            tracking_dir (str): Root directory for MLflow runs.
            experiment_id (str): Experiment identifier.
            run_id (str): Run identifier.
            run_name (str | None): Optional run display name.
            tags (dict[str, str] | None): Optional tags to persist.
        """

        self.tracking_dir = Path(tracking_dir)
        self.experiment_id = str(experiment_id)
        self.run_id = run_id
        self.run_name = run_name
        self.tags = tags or {}
        self.experiment_dir = self.tracking_dir / self.experiment_id
        self.run_dir = self.experiment_dir / self.run_id
        self.params_dir = self.run_dir / "params"
        self.metrics_dir = self.run_dir / "metrics"
        self.tags_dir = self.run_dir / "tags"
        self.artifacts_dir = self.run_dir / "artifacts"

        self._ensure_dirs()
        self._write_experiment_meta()
        self._write_run_meta(status="RUNNING", end_time=None)
        for name, value in self.tags.items():
            self.set_tag(name, value)

    def _ensure_dirs(self) -> None:
        """Create MLflow run directories if missing.

        Returns:
            None: This method writes directories to disk.
        """

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.tags_dir.mkdir(parents=True, exist_ok=True)

    def _write_experiment_meta(self) -> None:
        """Create the experiment metadata file if needed.

        Returns:
            None: This method writes metadata to disk.
        """

        meta_path = self.experiment_dir / "meta.yaml"
        if meta_path.exists():
            return
        artifact_location = f"file://{self.experiment_dir.resolve()}"
        data = {
            "artifact_location": artifact_location,
            "experiment_id": self.experiment_id,
            "lifecycle_stage": "active",
            "name": "Default",
        }
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    def _write_run_meta(self, status: str, end_time: int | None) -> None:
        """Write the run metadata file.

        Args:
            status (str): Run status string.
            end_time (int | None): End time in milliseconds.
        """

        meta_path = self.run_dir / "meta.yaml"
        start_time = int(time.time() * 1000)
        artifact_uri = f"file://{self.artifacts_dir.resolve()}"
        data = {
            "artifact_uri": artifact_uri,
            "experiment_id": self.experiment_id,
            "lifecycle_stage": "active",
            "run_id": self.run_id,
            "run_name": self.run_name or self.run_id,
            "source_type": "LOCAL",
            "start_time": start_time,
            "status": status,
            "end_time": end_time,
            "user_id": os.environ.get("USER", "unknown"),
        }
        meta_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    def log_param(self, name: str, value: str) -> None:
        """Log a parameter value.

        Args:
            name (str): Parameter name.
            value (str): Parameter value.
        """

        path = self.params_dir / name
        path.write_text(str(value), encoding="utf-8")

    def log_params(self, params: dict[str, str]) -> None:
        """Log multiple parameters at once.

        Args:
            params (dict[str, str]): Parameter name/value pairs.
        """

        for name, value in params.items():
            self.log_param(name, value)

    def log_metric(
        self, name: str, value: float, step: int, timestamp_ms: int | None = None
    ) -> None:
        """Append a metric line to the MLflow metrics file.

        Args:
            name (str): Metric name.
            value (float): Metric value.
            step (int): Step index for the metric.
            timestamp_ms (int | None): Unix timestamp in milliseconds.
        """

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        path = self.metrics_dir / name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp_ms} {value} {step}\n")

    def log_metrics(
        self, metrics: dict[str, float], step: int, timestamp_ms: int | None = None
    ) -> None:
        """Log multiple metrics to MLflow.

        Args:
            metrics (dict[str, float]): Metric name/value pairs.
            step (int): Step index for the metrics.
            timestamp_ms (int | None): Unix timestamp in milliseconds.
        """

        for name, value in metrics.items():
            self.log_metric(name, value, step=step, timestamp_ms=timestamp_ms)

    def set_tag(self, name: str, value: str) -> None:
        """Persist a tag in the MLflow run directory.

        Args:
            name (str): Tag name.
            value (str): Tag value.
        """

        path = self.tags_dir / name
        path.write_text(str(value), encoding="utf-8")

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        """Copy a file or directory into the artifacts folder.

        Args:
            path (str): Source path to copy.
            artifact_path (str | None): Optional subdirectory under artifacts.
        """

        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        target_dir = self.artifacts_dir
        if artifact_path:
            target_dir = target_dir / artifact_path
        target_dir.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            dst = target_dir / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            return
        dst = target_dir / src.name
        if src.resolve() == dst.resolve():
            return
        shutil.copy2(src, dst)

    def close(self, status: str) -> None:
        """Finalize the MLflow run metadata.

        Args:
            status (str): Final run status string.
        """

        end_time = int(time.time() * 1000)
        self._write_run_meta(status=status, end_time=end_time)


class Hook:
    """Base hook class for pipeline lifecycle events.

    Hooks can implement any subset of the lifecycle callbacks to extend the
    pipeline without modifying core logic.

    Examples:
        >>> isinstance(Hook, type)
        True
    """

    def on_run_start(self, context: RunContext) -> None:
        """Handle the start of a pipeline run.

        Args:
            context (RunContext): Active run context.
        """

    def on_run_end(self, context: RunContext, results: list[PhaseResult]) -> None:
        """Handle the end of a pipeline run.

        Args:
            context (RunContext): Active run context.
            results (list[PhaseResult]): Phase results.
        """

    def on_phase_start(self, context: RunContext, phase_name: str) -> None:
        """Handle the start of a phase.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
        """

    def on_phase_end(self, context: RunContext, result: PhaseResult) -> None:
        """Handle the end of a phase.

        Args:
            context (RunContext): Active run context.
            result (PhaseResult): Result summary for the phase.
        """

    def on_error(self, context: RunContext, phase_name: str, error: PhaseError) -> None:
        """Handle a phase-level error.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            error (PhaseError): Structured error data.
        """

    def on_epoch_start(self, context: RunContext, phase_name: str, epoch: int) -> None:
        """Handle the start of a training epoch.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            epoch (int): Epoch index (1-based).
        """

    def on_epoch_end(
        self,
        context: RunContext,
        phase_name: str,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Handle the end of a training epoch.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            epoch (int): Epoch index (1-based).
            metrics (dict[str, float]): Epoch metrics.
        """

    def on_batch_end(
        self,
        context: RunContext,
        phase_name: str,
        batch_idx: int,
        metrics: dict[str, float],
    ) -> None:
        """Handle the end of a training batch.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            batch_idx (int): Batch index (1-based).
            metrics (dict[str, float]): Batch metrics.
        """

    def on_metrics(
        self,
        context: RunContext,
        phase_name: str,
        step: int,
        metrics: dict[str, float],
        timestamp_ms: int,
    ) -> None:
        """Handle metric logging requests.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            step (int): Step index for the metrics.
            metrics (dict[str, float]): Metric values.
            timestamp_ms (int): Unix timestamp in milliseconds.
        """

    def on_inference_tile(
        self,
        context: RunContext,
        phase_name: str,
        tile_idx: int,
        total_tiles: int,
    ) -> None:
        """Handle per-tile inference events.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            tile_idx (int): Tile index (1-based).
            total_tiles (int): Total number of tiles.
        """


class MlflowHook(Hook):
    """Hook that logs metrics and params to MLflow-compatible files.

    Args:
        mlflow_logger (MlflowFileLogger): Active MLflow file logger.
        log_batch_metrics (bool): Whether to log per-batch metrics.
        log_batch_interval (int): Batch interval for logging.

    Examples:
        >>> isinstance(MlflowHook, type)
        True
    """

    def __init__(
        self,
        mlflow_logger: MlflowFileLogger,
        log_batch_metrics: bool,
        log_batch_interval: int,
    ) -> None:
        """Initialize the MLflow hook.

        Args:
            mlflow_logger (MlflowFileLogger): MLflow file logger.
            log_batch_metrics (bool): Toggle per-batch metric logging.
            log_batch_interval (int): Logging interval for batches.
        """

        self.mlflow_logger = mlflow_logger
        self.log_batch_metrics = log_batch_metrics
        self.log_batch_interval = log_batch_interval

    def on_run_start(self, context: RunContext) -> None:
        """Log initial parameters and tags at run start.

        Args:
            context (RunContext): Active run context.
        """

        params = context.config.get("_run_params", {})
        if isinstance(params, dict):
            self.mlflow_logger.log_params({str(k): str(v) for k, v in params.items()})
        if context.config_path:
            self.mlflow_logger.set_tag("config_path", context.config_path)
        self.mlflow_logger.set_tag("run_id", context.run_id)
        self.mlflow_logger.set_tag("experiment_id", context.experiment_id)

    def on_metrics(
        self,
        context: RunContext,
        phase_name: str,
        step: int,
        metrics: dict[str, float],
        timestamp_ms: int,
    ) -> None:
        """Log metrics to MLflow-compatible files.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            step (int): Step index.
            metrics (dict[str, float]): Metric values.
            timestamp_ms (int): Unix timestamp in milliseconds.
        """

        metric_payload = {f"{phase_name}.{key}": value for key, value in metrics.items()}
        self.mlflow_logger.log_metrics(metric_payload, step=step, timestamp_ms=timestamp_ms)


class HookManager:
    """Aggregate multiple hooks into a single dispatcher.

    Args:
        hooks (list[Hook]): Hook instances.

    Examples:
        >>> HookManager([]).hooks
        []
    """

    def __init__(self, hooks: list[Hook]) -> None:
        """Initialize the hook manager.

        Args:
            hooks (list[Hook]): Hook instances to dispatch.
        """

        self.hooks = hooks

    def on_run_start(self, context: RunContext) -> None:
        """Dispatch run start events.

        Args:
            context (RunContext): Active run context.
        """

        for hook in self.hooks:
            hook.on_run_start(context)

    def on_run_end(self, context: RunContext, results: list[PhaseResult]) -> None:
        """Dispatch run end events.

        Args:
            context (RunContext): Active run context.
            results (list[PhaseResult]): Phase results.
        """

        for hook in self.hooks:
            hook.on_run_end(context, results)

    def on_phase_start(self, context: RunContext, phase_name: str) -> None:
        """Dispatch phase start events.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
        """

        for hook in self.hooks:
            hook.on_phase_start(context, phase_name)

    def on_phase_end(self, context: RunContext, result: PhaseResult) -> None:
        """Dispatch phase end events.

        Args:
            context (RunContext): Active run context.
            result (PhaseResult): Phase result.
        """

        for hook in self.hooks:
            hook.on_phase_end(context, result)

    def on_error(self, context: RunContext, phase_name: str, error: PhaseError) -> None:
        """Dispatch error events.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            error (PhaseError): Error metadata.
        """

        for hook in self.hooks:
            hook.on_error(context, phase_name, error)

    def on_epoch_start(self, context: RunContext, phase_name: str, epoch: int) -> None:
        """Dispatch epoch start events.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            epoch (int): Epoch index.
        """

        for hook in self.hooks:
            hook.on_epoch_start(context, phase_name, epoch)

    def on_epoch_end(
        self,
        context: RunContext,
        phase_name: str,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Dispatch epoch end events.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            epoch (int): Epoch index.
            metrics (dict[str, float]): Epoch metrics.
        """

        for hook in self.hooks:
            hook.on_epoch_end(context, phase_name, epoch, metrics)

    def on_batch_end(
        self,
        context: RunContext,
        phase_name: str,
        batch_idx: int,
        metrics: dict[str, float],
    ) -> None:
        """Dispatch batch end events.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            batch_idx (int): Batch index.
            metrics (dict[str, float]): Batch metrics.
        """

        for hook in self.hooks:
            hook.on_batch_end(context, phase_name, batch_idx, metrics)

    def on_metrics(
        self,
        context: RunContext,
        phase_name: str,
        step: int,
        metrics: dict[str, float],
        timestamp_ms: int | None = None,
    ) -> None:
        """Record metrics for hooks and JSONL storage.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            step (int): Step index.
            metrics (dict[str, float]): Metric values.
            timestamp_ms (int | None): Unix timestamp in milliseconds.
        """

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        if context.metrics_writer:
            context.metrics_writer.write(phase_name, step, metrics, timestamp_ms=timestamp_ms)
        for hook in self.hooks:
            hook.on_metrics(context, phase_name, step, metrics, timestamp_ms)

    def on_inference_tile(
        self,
        context: RunContext,
        phase_name: str,
        tile_idx: int,
        total_tiles: int,
    ) -> None:
        """Dispatch inference tile events.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            tile_idx (int): Tile index.
            total_tiles (int): Total tiles.
        """

        for hook in self.hooks:
            hook.on_inference_tile(context, phase_name, tile_idx, total_tiles)


class Processor:
    """Base class for pre/post phase processors.

    Processors run before or after phases to extend pipeline behavior without
    modifying the core phase logic.

    Examples:
        >>> Processor().should_run(None, "prepare", "before")
        True
    """

    def should_run(self, context: RunContext, phase_name: str, when: str) -> bool:
        """Return True if the processor should run for the given phase.

        Args:
            context (RunContext): Active run context.
            phase_name (str): Phase name.
            when (str): Either "before" or "after".

        Returns:
            bool: True when the processor should run.
        """

        _ = context
        _ = phase_name
        _ = when
        return True

    def run(self, context: RunContext, phase_result: PhaseResult, when: str) -> PhaseResult:
        """Execute processor logic for a phase.

        Args:
            context (RunContext): Active run context.
            phase_result (PhaseResult): Phase result.
            when (str): Either "before" or "after".

        Returns:
            PhaseResult: Phase result (may be unchanged).
        """

        _ = context
        _ = when
        return phase_result


class ConfigSnapshotProcessor(Processor):
    """Persist a copy of the configuration as a run artifact.

    Examples:
        >>> isinstance(ConfigSnapshotProcessor, type)
        True
    """

    def __init__(self) -> None:
        """Initialize the processor.

        Returns:
            None: This initializer has no return value.
        """

        self._ran = False

    def run(self, context: RunContext, phase_result: PhaseResult, when: str) -> PhaseResult:
        """Write the configuration snapshot to artifacts.

        Args:
            context (RunContext): Active run context.
            phase_result (PhaseResult): Phase result.
            when (str): Either "before" or "after".

        Returns:
            PhaseResult: Phase result (unchanged).
        """

        if self._ran or when != "before":
            return phase_result
        self._ran = True
        artifacts_dir = context.run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = artifacts_dir / "config_snapshot.yml"
        if context.config_path and Path(context.config_path).exists():
            shutil.copy2(context.config_path, snapshot_path)
        else:
            snapshot_path.write_text(yaml.safe_dump(context.config), encoding="utf-8")
        if context.mlflow_logger:
            context.mlflow_logger.log_artifact(str(snapshot_path))
        return phase_result


class RunSummaryProcessor(Processor):
    """Persist a run summary JSON artifact.

    Examples:
        >>> isinstance(RunSummaryProcessor, type)
        True
    """

    def run(self, context: RunContext, phase_result: PhaseResult, when: str) -> PhaseResult:
        """Write the run summary to artifacts.

        Args:
            context (RunContext): Active run context.
            phase_result (PhaseResult): Phase result.
            when (str): Either "before" or "after".

        Returns:
            PhaseResult: Phase result (unchanged).
        """

        if when != "after":
            return phase_result
        artifacts_dir = context.run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        summary_path = artifacts_dir / "run_summary.json"
        results = context.run_results or []
        status = "success"
        for result in results:
            if result.status == "failed":
                status = "failed"
                break
        payload = {
            "run_id": context.run_id,
            "experiment_id": context.experiment_id,
            "start_time": context.start_time,
            "end_time": time.time(),
            "status": status,
            "results": [result.to_dict() for result in results],
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if context.mlflow_logger:
            context.mlflow_logger.log_artifact(str(summary_path))
        return phase_result
