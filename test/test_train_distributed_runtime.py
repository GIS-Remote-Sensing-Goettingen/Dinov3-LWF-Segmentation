"""Regression tests for distributed training synchronization helpers.

Examples:
    >>> True
    True
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline.phases.train as train_module  # noqa: E402
import pipeline.phases.train_batches as train_batches_module  # noqa: E402
import pipeline.utils as pipeline_utils  # noqa: E402
from pipeline.context import DistContext  # noqa: E402


class _RecordingLogger:
    """Minimal logger stub that stores emitted info/error strings.

    This keeps the tests independent from the repo's file-backed logger while
    still allowing assertions on emitted progress messages.

    Examples:
        >>> isinstance(_RecordingLogger(), _RecordingLogger)
        True
    """

    def __init__(self) -> None:
        """Initialize empty message buffers.

        The stub mirrors only the small surface used by the training helpers.
        """

        self.info_messages: list[str] = []
        self.error_messages: list[str] = []

    def info(self, message: str) -> None:
        """Capture an info-level log line.

        Args:
            message (str): Message text to store.
        """

        self.info_messages.append(str(message))

    def error(self, message: str) -> None:
        """Capture an error-level log line.

        Args:
            message (str): Message text to store.
        """

        self.error_messages.append(str(message))


def test_setup_distributed_uses_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed init should forward the configured process-group timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    captured: dict[str, object] = {}
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(pipeline_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(
        pipeline_utils.torch.cuda,
        "set_device",
        lambda index: captured.setdefault("device", index),
    )
    monkeypatch.setattr(
        pipeline_utils.dist,
        "init_process_group",
        lambda **kwargs: captured.update(kwargs),
    )

    ctx = pipeline_utils.setup_distributed(
        {
            "distributed": True,
            "dist_backend": "nccl",
            "dist_timeout_minutes": 42,
        }
    )

    assert ctx.enabled is True
    assert ctx.rank == 1
    assert ctx.world_size == 2
    assert ctx.local_rank == 1
    assert captured["device"] == 1
    assert captured["backend"] == "nccl"
    assert captured["rank"] == 1
    assert captured["world_size"] == 2
    assert captured["timeout"] == timedelta(minutes=42)


def test_non_main_rank_uses_broadcast_validation_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-main ranks should not run validation locally under DDP.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    logger = _RecordingLogger()
    context = SimpleNamespace(
        dist_ctx=DistContext(enabled=True, rank=1, world_size=2, local_rank=1),
        logger=logger,
    )
    expected_payload = {
        "val_loss": 0.25,
        "val_metrics": {
            "miou": 0.75,
            "mdice": 0.85,
            "nonfinite_val_batches": 0.0,
            "nonfinite_val_loss_batches": 0.0,
            "max_abs_logit": 0.0,
            "loss_ce": 0.0,
            "loss_focal": 0.0,
            "loss_dice": 0.0,
            "loss_aux": 0.0,
            "loss_boundary": 0.0,
            "loss_skeleton": 0.0,
            "loss_topology": 0.0,
            "loss_total": 0.25,
        },
        "param_nonfinite_count": 0.0,
        "checkpoint_is_finite": True,
        "stop_flag": False,
        "validation_duration_s": 12.0,
    }
    monkeypatch.setattr(
        train_module,
        "evaluate",
        lambda *args, **kwargs: pytest.fail("evaluate() should not run on rank > 0"),
    )
    monkeypatch.setattr(
        train_module,
        "broadcast_main_object",
        lambda dist_ctx, payload: expected_payload,
    )

    payload, backbone, processor = train_module._resolve_epoch_validation_state(
        context=context,
        epoch=1,
        avg_train_loss=0.2,
        eval_model=torch.nn.Linear(1, 1),
        val_loader=None,
        loss_fn=SimpleNamespace(),
        device=torch.device("cpu"),
        use_amp=False,
        model_cfg={"backbone": "demo", "num_classes": 2, "layers": [1, 2]},
        cache_features=False,
        backbone=None,
        processor=None,
        ps=16,
        stability=SimpleNamespace(nonfinite_action="stop_run"),
        boundary_kernel_size=3,
        early_stopping=SimpleNamespace(),
    )

    assert payload == expected_payload
    assert backbone is None
    assert processor is None
    assert logger.info_messages == []


def test_non_main_rank_uses_broadcast_xai_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-main ranks should receive rank-0 epoch XAI metrics via broadcast.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    context = SimpleNamespace(
        dist_ctx=DistContext(enabled=True, rank=1, world_size=2, local_rank=1),
        logger=_RecordingLogger(),
    )
    plot_cfg = SimpleNamespace(enabled=True, paper_enable=False)
    expected_payload = {
        "xai_epoch_metrics": {"xai_img_importance_mean": 0.4},
        "xai_duration_s": 31.5,
    }
    monkeypatch.setattr(
        train_module,
        "collect_epoch_xai_metrics",
        lambda **kwargs: pytest.fail(
            "collect_epoch_xai_metrics() should not run on rank > 0"
        ),
    )
    monkeypatch.setattr(
        train_module,
        "broadcast_main_object",
        lambda dist_ctx, payload: expected_payload,
    )

    xai_metrics, backbone, processor = train_module._resolve_epoch_xai_state(
        context=context,
        epoch=2,
        eval_model=torch.nn.Linear(1, 1),
        val_loader=None,
        cache_features=False,
        model_cfg={"backbone": "demo", "layers": [1, 2]},
        loss_ignore_index=255,
        plot_cfg=plot_cfg,
        plot_metrics_dir="plots/metrics",
        plot_xai_dir="plots/xai",
        plot_metrics_paper_dir="plots/metrics/paper",
        plot_xai_paper_dir="plots/xai/paper",
        plot_xai_cam_layer=None,
        plot_xai_pca_layer=None,
        model_layer_ids=[1, 2],
        backbone=None,
        processor=None,
        device=torch.device("cpu"),
        ps=16,
        autocast=None,
        histories={},
    )

    assert xai_metrics == {"xai_img_importance_mean": 0.4}
    assert backbone is None
    assert processor is None


def test_wrap_model_for_training_respects_find_unused_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DDP wrapping should forward the configured unused-parameter policy.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    captured: dict[str, object] = {}

    def fake_ddp(model: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        """Capture DDP kwargs and return the unwrapped model.

        Args:
            model (torch.nn.Module): Model being wrapped.
            **kwargs (object): DDP keyword arguments.

        Returns:
            torch.nn.Module: Original model.
        """

        captured["model"] = model
        captured.update(kwargs)
        return model

    monkeypatch.setattr(train_module, "DDP", fake_ddp)
    context = SimpleNamespace(
        dist_ctx=DistContext(enabled=True, rank=0, world_size=2, local_rank=0)
    )
    model = torch.nn.Linear(2, 2)

    wrapped = train_module._wrap_model_for_training(
        model=model,
        context=context,
        resources_cfg={"ddp_find_unused_parameters": True},
    )

    assert wrapped is model
    assert captured["find_unused_parameters"] is True
    assert captured["device_ids"] == [0]
    assert captured["output_device"] == 0


def test_handle_nonfinite_batch_escalates_under_distributed() -> None:
    """Distributed mode should never continue past a local non-finite batch.

    Examples:
        >>> True
        True
    """

    context = SimpleNamespace(
        dist_ctx=DistContext(enabled=True, rank=1, world_size=2, local_rank=1),
        logger=_RecordingLogger(),
    )
    stability = SimpleNamespace(
        save_bad_batch_sample=False,
        nonfinite_max_consecutive_batches=99,
        nonfinite_max_total_batches_per_epoch=99,
        nonfinite_action="skip_batch",
    )
    epoch_health = {"nonfinite_batches": 0.0, "consecutive_nonfinite_batches": 0.0}

    action = train_batches_module._handle_nonfinite_batch(
        context=context,
        epoch=0,
        batch_idx=3,
        train_loader=[1, 2, 3, 4],
        epoch_health=epoch_health,
        stability=stability,
        weights_dir="weights",
        reason="loss",
    )

    assert action == "raise"
    assert any(
        "escalates non-finite loss" in msg for msg in context.logger.error_messages
    )


def test_log_distributed_batch_stage_logs_last_batch() -> None:
    """The DDP timing helper should log the last batch even when it is fast.

    Examples:
        >>> True
        True
    """

    context = SimpleNamespace(
        dist_ctx=DistContext(enabled=True, rank=1, world_size=2, local_rank=1),
        logger=_RecordingLogger(),
    )

    train_batches_module._log_distributed_batch_stage(
        context=context,
        epoch=0,
        batch_idx=5,
        total_batches=5,
        stage="forward_loss",
        duration_s=0.25,
        batch_shape=(4, 3, 224, 224),
        feature_count=4,
        cache_features=False,
    )

    assert len(context.logger.info_messages) == 1
    assert (
        "Rank 1 epoch 1 batch 5/5 stage=forward_loss" in context.logger.info_messages[0]
    )


def test_raise_distributed_training_error_destroys_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed fail-fast should tear down the process group before raising.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    destroyed: list[bool] = []
    monkeypatch.setattr(train_batches_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(train_batches_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        train_batches_module.dist,
        "destroy_process_group",
        lambda: destroyed.append(True),
    )
    context = SimpleNamespace(
        dist_ctx=DistContext(enabled=True, rank=0, world_size=2, local_rank=0),
        logger=_RecordingLogger(),
    )

    with pytest.raises(train_batches_module.TrainingError, match="boom"):
        train_batches_module._raise_distributed_training_error(
            context=context,
            message="boom",
        )

    assert destroyed == [True]
