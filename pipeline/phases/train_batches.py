"""Epoch batch-loop helpers for the training phase."""

from __future__ import annotations

import math
import os
import time
from typing import Any, cast

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from utils import SegmentationLoss
from utils.losses import LOSS_COMPONENT_KEYS

from ..context import RunContext, TrainingError
from ..train_utils import (
    ModelEMA,
    align_logits_to_labels,
    build_boundary_targets,
    count_nonfinite_parameters,
    extract_multiscale_features_batch,
    forward_with_optional_extras,
    move_features_to_device,
    require_patch_grid_compatible,
    should_warn_high_logit,
)

_DDP_STAGE_SLOW_SECONDS = 30.0


def _should_log_distributed_batch_stage(
    *,
    context: RunContext,
    batch_idx: int,
    total_batches: int,
    duration_s: float,
    force: bool = False,
) -> bool:
    """Return whether one batch stage should emit a detailed timing log.

    This keeps the additional DDP diagnostics high-signal by tracing the first
    batch, the last batch, slow stages, and any caller-forced events.

    Args:
        context (RunContext): Active run context.
        batch_idx (int): Current 1-based batch index.
        total_batches (int): Total number of train batches in the epoch.
        duration_s (float): Stage duration in seconds.
        force (bool): Whether to force a log regardless of thresholds.

    Returns:
        bool: True when a detailed timing log should be emitted.

    Examples:
        >>> _should_log_distributed_batch_stage(
        ...     context=cast(Any, RunContext),
        ...     batch_idx=1,
        ...     total_batches=10,
        ...     duration_s=0.1,
        ...     force=True,
        ... )
        True
    """

    if force:
        return True
    if not context.dist_ctx.enabled:
        return False
    return (
        batch_idx == 1
        or batch_idx == total_batches
        or duration_s >= _DDP_STAGE_SLOW_SECONDS
    )


def _log_distributed_batch_stage(
    *,
    context: RunContext,
    epoch: int,
    batch_idx: int,
    total_batches: int,
    stage: str,
    duration_s: float,
    batch_shape: tuple[int, ...],
    feature_count: int,
    cache_features: bool,
    force: bool = False,
) -> None:
    """Emit a rank-aware DDP batch-stage timing message when warranted.

    Args:
        context (RunContext): Active run context.
        epoch (int): Zero-based epoch index.
        batch_idx (int): Current 1-based batch index.
        total_batches (int): Total number of train batches.
        stage (str): Short stage name.
        duration_s (float): Stage duration in seconds.
        batch_shape (tuple[int, ...]): Local image batch shape.
        feature_count (int): Number of feature tensors attached to the batch.
        cache_features (bool): Whether the run reads cached features.
        force (bool): Whether to log regardless of thresholds.

    Examples:
        >>> callable(_log_distributed_batch_stage)
        True
    """

    if not _should_log_distributed_batch_stage(
        context=context,
        batch_idx=batch_idx,
        total_batches=total_batches,
        duration_s=duration_s,
        force=force,
    ):
        return
    context.logger.info(
        "Rank %s epoch %s batch %s/%s stage=%s took %.2fs "
        "img=%s feature_tensors=%s cache_features=%s"
        % (
            context.dist_ctx.rank,
            epoch + 1,
            batch_idx,
            total_batches,
            stage,
            duration_s,
            batch_shape,
            feature_count,
            cache_features,
        )
    )


def _raise_distributed_training_error(
    *,
    context: RunContext,
    message: str,
    cause: BaseException | None = None,
) -> None:
    """Raise a train-loop failure and tear down DDP first when active.

    Args:
        context (RunContext): Active run context.
        message (str): Human-readable failure message.
        cause (BaseException | None): Optional originating exception.

    Raises:
        TrainingError: Always raised with the provided message.

    Examples:
        >>> callable(_raise_distributed_training_error)
        True
    """

    if context.dist_ctx.enabled:
        context.logger.error(
            "Rank %s aborting distributed training: %s"
            % (context.dist_ctx.rank, message)
        )
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    if cause is not None:
        raise TrainingError(message) from cause
    raise TrainingError(message)


def _handle_nonfinite_batch(
    *,
    context: RunContext,
    epoch: int,
    batch_idx: int,
    train_loader: Any,
    epoch_health: dict[str, float],
    stability: Any,
    weights_dir: str,
    reason: str,
    img_tensor: torch.Tensor | None = None,
    target_tensor: torch.Tensor | None = None,
    logit_tensor: torch.Tensor | None = None,
) -> str:
    """Handle a non-finite batch event and decide control flow.

    Args:
        context (RunContext): Active run context.
        epoch (int): Zero-based epoch index.
        batch_idx (int): Current 1-based batch index.
        train_loader (Any): Training dataloader.
        epoch_health (dict[str, float]): Mutable epoch health state.
        stability (Any): Stability configuration object.
        weights_dir (str): Directory for bad-batch dumps.
        reason (str): Non-finite source label.
        img_tensor (torch.Tensor | None): Optional image snapshot.
        target_tensor (torch.Tensor | None): Optional target snapshot.
        logit_tensor (torch.Tensor | None): Optional logits snapshot.

    Returns:
        str: One of ``continue``, ``break_epoch``, or ``raise``.
    """

    epoch_health["nonfinite_batches"] += 1
    epoch_health["consecutive_nonfinite_batches"] += 1
    context.logger.error(
        "Non-finite %s at epoch %s batch %s/%s."
        % (reason, epoch + 1, batch_idx, len(train_loader))
    )
    if context.dist_ctx.is_main and stability.save_bad_batch_sample:
        bad_dir = os.path.join(weights_dir, "bad_batches")
        os.makedirs(bad_dir, exist_ok=True)
        bad_path = os.path.join(
            bad_dir, f"epoch_{epoch + 1:04d}_batch_{batch_idx:04d}.pt"
        )
        payload: dict[str, Any] = {
            "epoch": epoch + 1,
            "batch_idx": batch_idx,
            "reason": reason,
        }
        if img_tensor is not None:
            payload["image"] = img_tensor[:1].detach().cpu()
        if target_tensor is not None:
            payload["target"] = target_tensor[:1].detach().cpu()
        if logit_tensor is not None:
            payload["logits"] = logit_tensor[:1].detach().float().cpu()
        torch.save(payload, bad_path)
    too_many = (
        epoch_health["consecutive_nonfinite_batches"]
        >= stability.nonfinite_max_consecutive_batches
        or epoch_health["nonfinite_batches"]
        >= stability.nonfinite_max_total_batches_per_epoch
    )
    if too_many:
        return "raise"
    if context.dist_ctx.enabled:
        context.logger.error(
            "Distributed mode escalates non-finite %s to stop_run to keep ranks in sync."
            % reason
        )
        return "raise"
    if stability.nonfinite_action == "skip_batch":
        return "continue"
    if stability.nonfinite_action == "stop_epoch":
        return "break_epoch"
    return "raise"


def ensure_backbone_processor(
    backbone: Any,
    processor: Any,
    backbone_name: str,
    device: torch.device,
) -> tuple[Any, Any]:
    """Lazily initialize backbone and processor when needed.

    Args:
        backbone (Any): Cached backbone instance or ``None``.
        processor (Any): Cached image processor or ``None``.
        backbone_name (str): Backbone checkpoint identifier.
        device (torch.device): Target torch device.

    Returns:
        tuple[Any, Any]: Initialized ``(backbone, processor)`` pair.
    """

    if backbone is None or processor is None:
        processor = AutoImageProcessor.from_pretrained(backbone_name)
        backbone = AutoModel.from_pretrained(backbone_name).eval().to(device)
    return backbone, processor


def run_train_epoch_batches(
    *,
    context: RunContext,
    epoch: int,
    epochs: int,
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: SegmentationLoss,
    device: torch.device,
    ps: int,
    cache_features: bool,
    model_cfg: dict[str, Any],
    backbone: Any,
    processor: Any,
    autocast: Any,
    scaler: Any,
    requires_backbone_features: bool,
    require_aux_logits: bool,
    grad_accum: int,
    stability: Any,
    boundary_kernel_size: int,
    log_batch_metrics: bool,
    log_batch_interval: int,
    ema: ModelEMA | None,
    weights_dir: str,
) -> tuple[float, dict[str, float], dict[str, float], bool, Any, Any]:
    """Run all train batches for one epoch.

    Args:
        context (RunContext): Active run context.
        epoch (int): Zero-based epoch index.
        epochs (int): Total epoch count.
        model (torch.nn.Module): Trainable segmentation head (possibly wrapped).
        base_model (torch.nn.Module): Unwrapped segmentation head.
        train_loader (Any): Training dataloader.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler.
        loss_fn (SegmentationLoss): Segmentation loss module.
        device (torch.device): Target torch device.
        ps (int): Backbone patch size.
        cache_features (bool): Whether features are read from cache.
        model_cfg (dict[str, Any]): Parsed model configuration.
        backbone (Any): Cached backbone model or ``None``.
        processor (Any): Cached image processor or ``None``.
        autocast (Any): AMP autocast context manager.
        scaler (Any): Optional gradient scaler.
        requires_backbone_features (bool): Whether the head needs DINO features.
        require_aux_logits (bool): Whether aux logits must be present.
        grad_accum (int): Gradient accumulation steps.
        stability (Any): Stability configuration object.
        boundary_kernel_size (int): Boundary-target morphology kernel size.
        log_batch_metrics (bool): Whether to emit batch-level hooks.
        log_batch_interval (int): Hook emission interval.
        ema (ModelEMA | None): EMA helper.
        weights_dir (str): Directory for checkpoints and bad-batch dumps.

    Returns:
        tuple:
            - avg_train_loss
            - average loss components by key
            - epoch health metrics
            - epoch_aborted flag
            - backbone (possibly initialized)
            - processor (possibly initialized)
    """

    epoch_start = time.time()
    first_batch_logged = False
    last_log_time = epoch_start
    epoch_health: dict[str, float] = {
        "nonfinite_batches": 0.0,
        "consecutive_nonfinite_batches": 0.0,
        "skipped_optimizer_steps": 0.0,
        "optimizer_steps": 0.0,
        "scheduler_steps": 0.0,
        "max_abs_logit": 0.0,
        "grad_norm": 0.0,
        "param_nonfinite_count": 0.0,
    }
    epoch_aborted = False
    model_call = cast(Any, model)
    model_call.train()
    train_loss = 0.0
    train_loss_batches = 0
    train_loss_component_sums = {key: 0.0 for key in LOSS_COMPONENT_KEYS}
    optimizer.zero_grad()
    total_batches = len(train_loader)
    next_batch_wait_started_at = epoch_start

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{epochs} [Train]",
        leave=False,
    )
    for batch_idx, (img, features, y) in enumerate(pbar, 1):
        fetch_duration = time.time() - next_batch_wait_started_at
        batch_shape = tuple(img.shape)
        feature_count = len(features) if features is not None else 0
        if not first_batch_logged:
            first_batch_logged = True
            first_delay = time.time() - epoch_start
            context.logger.info(
                f"Epoch {epoch + 1} first batch received after {first_delay:.2f}s"
            )
            last_log_time = time.time()
        _log_distributed_batch_stage(
            context=context,
            epoch=epoch,
            batch_idx=batch_idx,
            total_batches=total_batches,
            stage="fetch",
            duration_s=fetch_duration,
            batch_shape=batch_shape,
            feature_count=feature_count,
            cache_features=cache_features,
        )
        img = img.to(device)
        y = y.to(device)
        require_patch_grid_compatible(
            img,
            ps,
            source=f"Train epoch {epoch + 1} batch {batch_idx}",
        )
        stage_name = "feature_extract"
        stage_started_at = time.time()
        try:
            if not requires_backbone_features:
                feats = []
            elif cache_features and features:
                feats = move_features_to_device(features, device)
            else:
                backbone, processor = ensure_backbone_processor(
                    backbone, processor, model_cfg["backbone"], device
                )
                feats = extract_multiscale_features_batch(
                    img,
                    backbone,
                    processor,
                    device,
                    model_cfg["layers"],
                    ps,
                )
            feature_duration = time.time() - stage_started_at
            _log_distributed_batch_stage(
                context=context,
                epoch=epoch,
                batch_idx=batch_idx,
                total_batches=total_batches,
                stage=stage_name,
                duration_s=feature_duration,
                batch_shape=batch_shape,
                feature_count=len(feats),
                cache_features=cache_features,
            )
            stage_name = "forward_loss"
            stage_started_at = time.time()
            with autocast:
                logits, aux_logits, edge_logits, skeleton_logits, _ = (
                    forward_with_optional_extras(
                        model_call,
                        img,
                        feats,
                        require_aux_logits=require_aux_logits,
                    )
                )
                logits = cast(torch.Tensor, align_logits_to_labels(logits, y))
                aux_logits = align_logits_to_labels(aux_logits, y)
                edge_logits = align_logits_to_labels(edge_logits, y)
                skeleton_logits = align_logits_to_labels(skeleton_logits, y)
                target_main = y if y.ndim == 3 else y.unsqueeze(0)
                target_aux = target_main if aux_logits is not None else None
                edge_targets, edge_mask = build_boundary_targets(
                    labels=y,
                    edge_logits=edge_logits,
                    num_classes=loss_fn.num_classes,
                    ignore_index=loss_fn.ignore_index,
                    kernel_size=boundary_kernel_size,
                )
                logits_for_loss = logits.float() if stability.loss_fp32 else logits
                aux_for_loss = (
                    aux_logits.float()
                    if aux_logits is not None and stability.loss_fp32
                    else aux_logits
                )
                edge_for_loss = (
                    edge_logits.float()
                    if edge_logits is not None and stability.loss_fp32
                    else edge_logits
                )
                loss_components = loss_fn.compute_components(
                    logits_for_loss,
                    target_main,
                    aux_logits=aux_for_loss,
                    aux_targets=target_aux,
                    edge_logits=edge_for_loss,
                    edge_targets=edge_targets,
                    edge_mask=edge_mask,
                    skeleton_logits=(
                        skeleton_logits.float()
                        if skeleton_logits is not None and stability.loss_fp32
                        else skeleton_logits
                    ),
                )
                loss = loss_components["loss_total"] / grad_accum
            forward_duration = time.time() - stage_started_at
            _log_distributed_batch_stage(
                context=context,
                epoch=epoch,
                batch_idx=batch_idx,
                total_batches=total_batches,
                stage=stage_name,
                duration_s=forward_duration,
                batch_shape=batch_shape,
                feature_count=len(feats),
                cache_features=cache_features,
            )
        except Exception as exc:
            failed_after = time.time() - stage_started_at
            context.logger.error(
                "Rank %s batch %s/%s failed at stage=%s after %.2fs; "
                "img=%s, features=%s, layers=%s, error=%s"
                % (
                    context.dist_ctx.rank,
                    batch_idx,
                    total_batches,
                    stage_name,
                    failed_after,
                    batch_shape,
                    feature_count,
                    model_cfg["layers"],
                    exc,
                )
            )
            _raise_distributed_training_error(
                context=context,
                message=(
                    "Rank %s failed at epoch %s batch %s/%s during %s"
                    % (
                        context.dist_ctx.rank,
                        epoch + 1,
                        batch_idx,
                        total_batches,
                        stage_name,
                    )
                ),
                cause=exc,
            )

        batch_max_abs_logit = float(
            torch.nan_to_num(
                logits.detach().float().abs(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            .max()
            .item()
        )
        epoch_health["max_abs_logit"] = max(
            epoch_health["max_abs_logit"], batch_max_abs_logit
        )
        if batch_idx % 10 == 0 and should_warn_high_logit(
            batch_max_abs_logit, stability.max_abs_logit_warn
        ):
            context.logger.error(
                f"High logit magnitude detected at epoch {epoch + 1} "
                f"batch {batch_idx}: "
                f"batch_max_abs_logit={batch_max_abs_logit:.2f}, "
                f"epoch_max_abs_logit={epoch_health['max_abs_logit']:.2f}"
            )
        if not torch.isfinite(loss):
            action = _handle_nonfinite_batch(
                context=context,
                epoch=epoch,
                batch_idx=batch_idx,
                train_loader=train_loader,
                epoch_health=epoch_health,
                stability=stability,
                weights_dir=weights_dir,
                reason="loss",
                img_tensor=img,
                target_tensor=target_main,
                logit_tensor=logits,
            )
            optimizer.zero_grad()
            if action == "continue":
                next_batch_wait_started_at = time.time()
                continue
            if action == "break_epoch":
                epoch_aborted = True
                break
            _raise_distributed_training_error(
                context=context,
                message=(
                    "Non-finite loss at epoch %s batch %s/%s"
                    % (epoch + 1, batch_idx, total_batches)
                ),
            )
        epoch_health["consecutive_nonfinite_batches"] = 0
        backward_started_at = time.time()
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        _log_distributed_batch_stage(
            context=context,
            epoch=epoch,
            batch_idx=batch_idx,
            total_batches=total_batches,
            stage="backward",
            duration_s=time.time() - backward_started_at,
            batch_shape=batch_shape,
            feature_count=len(feats),
            cache_features=cache_features,
        )

        if batch_idx % grad_accum == 0 or batch_idx == len(train_loader):
            grads_ok = True
            if scaler:
                scaler.unscale_(optimizer)
            grad_clip_started_at = time.time()
            if stability.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(),
                    stability.grad_clip_norm,
                )
                grad_norm_value = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else float(grad_norm)
                )
                epoch_health["grad_norm"] = float(grad_norm_value)
                if not math.isfinite(grad_norm_value):
                    grads_ok = False
            _log_distributed_batch_stage(
                context=context,
                epoch=epoch,
                batch_idx=batch_idx,
                total_batches=total_batches,
                stage="grad_clip",
                duration_s=time.time() - grad_clip_started_at,
                batch_shape=batch_shape,
                feature_count=len(feats),
                cache_features=cache_features,
            )
            if not grads_ok:
                action = _handle_nonfinite_batch(
                    context=context,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    train_loader=train_loader,
                    epoch_health=epoch_health,
                    stability=stability,
                    weights_dir=weights_dir,
                    reason="grad_norm",
                    img_tensor=img,
                    target_tensor=target_main,
                    logit_tensor=logits,
                )
                optimizer.zero_grad()
                if action == "continue":
                    next_batch_wait_started_at = time.time()
                    continue
                if action == "break_epoch":
                    epoch_aborted = True
                    break
                _raise_distributed_training_error(
                    context=context,
                    message=(
                        "Non-finite gradient norm at epoch %s batch %s/%s"
                        % (epoch + 1, batch_idx, total_batches)
                    ),
                )

            step_happened = False
            optimizer_started_at = time.time()
            if scaler:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                step_happened = scale_after >= scale_before
            else:
                optimizer.step()
                step_happened = True
            optimizer.zero_grad()
            _log_distributed_batch_stage(
                context=context,
                epoch=epoch,
                batch_idx=batch_idx,
                total_batches=total_batches,
                stage="optimizer_step",
                duration_s=time.time() - optimizer_started_at,
                batch_shape=batch_shape,
                feature_count=len(feats),
                cache_features=cache_features,
            )

            if step_happened:
                epoch_health["optimizer_steps"] += 1
                scheduler.step()
                epoch_health["scheduler_steps"] += 1
                if ema:
                    ema.update(base_model)
            else:
                epoch_health["skipped_optimizer_steps"] += 1

            if (
                epoch_health["optimizer_steps"] > 0
                and epoch_health["optimizer_steps"] % stability.check_params_every_steps
                == 0
            ):
                param_nonfinite_count = count_nonfinite_parameters(base_model)
                epoch_health["param_nonfinite_count"] = float(param_nonfinite_count)
                if param_nonfinite_count > 0:
                    action = _handle_nonfinite_batch(
                        context=context,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        train_loader=train_loader,
                        epoch_health=epoch_health,
                        stability=stability,
                        weights_dir=weights_dir,
                        reason="parameters",
                        img_tensor=img,
                        target_tensor=target_main,
                        logit_tensor=logits,
                    )
                    if action == "continue":
                        next_batch_wait_started_at = time.time()
                        continue
                    if action == "break_epoch":
                        epoch_aborted = True
                        break
                    _raise_distributed_training_error(
                        context=context,
                        message=(
                            "Detected %s non-finite parameters at epoch %s batch %s/%s"
                            % (
                                param_nonfinite_count,
                                epoch + 1,
                                batch_idx,
                                total_batches,
                            )
                        ),
                    )

        train_loss += loss.item() * grad_accum
        train_loss_batches += 1
        for key in LOSS_COMPONENT_KEYS:
            train_loss_component_sums[key] += float(
                loss_components[key].detach().item()
            )
        if log_batch_metrics and batch_idx % log_batch_interval == 0:
            batch_metrics = {
                "loss": loss.item() * grad_accum,
                "lr": scheduler.get_last_lr()[0],
            }
            context.hook_manager.on_batch_end(
                context, "train", batch_idx, batch_metrics
            )
            context.hook_manager.on_metrics(
                context,
                "train",
                batch_idx,
                {
                    "batch_loss": batch_metrics["loss"],
                    "lr": batch_metrics["lr"],
                },
            )
        if batch_idx % 10 == 0:
            now = time.time()
            avg_batch = (now - last_log_time) / 10
            context.logger.info(
                f"Epoch {epoch + 1} batch {batch_idx}/{total_batches} "
                f"avg batch time {avg_batch:.2f}s"
            )
            last_log_time = now
        next_batch_wait_started_at = time.time()

    if train_loss_batches == 0:
        avg_train_loss = float("nan")
        avg_train_loss_components = {key: float("nan") for key in LOSS_COMPONENT_KEYS}
    else:
        avg_train_loss = train_loss / train_loss_batches
        avg_train_loss_components = {
            key: train_loss_component_sums[key] / train_loss_batches
            for key in LOSS_COMPONENT_KEYS
        }
    return (
        float(avg_train_loss),
        avg_train_loss_components,
        epoch_health,
        epoch_aborted,
        backbone,
        processor,
    )
