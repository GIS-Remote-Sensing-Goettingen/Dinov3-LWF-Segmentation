"""Training phase implementation."""

from __future__ import annotations

import math
import os
import time
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR

from models import build_head
from utils import (
    EarlyStopping,
    Muon,
    SegmentationLoss,
    TimedBlock,
    resolve_cache_dir_for_train,
)
from utils.losses import LOSS_COMPONENT_KEYS

from ..constants import DEFAULT_DEVICE, DEFAULT_PROCESSED_DIR
from ..context import PhaseOutcome, RunContext, TrainingError
from ..data_splits import create_dataloaders, dataset_size
from ..phase_runner import Phase
from ..plotting import resolve_cam_layer
from ..train_config import parse_train_loss_config, parse_train_plot_config
from ..train_utils import (
    ModelEMA,
    build_autocast,
    count_nonfinite_parameters,
    evaluate,
    resolve_lr_metrics,
    split_params_for_muon,
    use_adamw_only_for_head,
)
from ..utils import get_hook_option, get_model_config, resolve_path, unwrap_model
from .train_batches import ensure_backbone_processor, run_train_epoch_batches
from .train_xai import collect_epoch_xai_metrics


def _build_optimizer(
    *,
    context: RunContext,
    head_name: str,
    base_model: torch.nn.Module,
    section: dict[str, Any],
) -> tuple[torch.optim.Optimizer, float]:
    """Build optimizer and return scheduler max-LR.

    Args:
        context (RunContext): Active run context for logging.
        head_name (str): Selected head name.
        base_model (torch.nn.Module): Unwrapped trainable model.
        section (dict[str, Any]): Train config section.

    Returns:
        tuple[torch.optim.Optimizer, float]: Optimizer instance and scheduler
        maximum LR.
    """

    if use_adamw_only_for_head(head_name):
        optimizer = torch.optim.AdamW(
            [param for param in base_model.parameters() if param.requires_grad],
            lr=section.get("adamw_lr", 1e-3),
            weight_decay=section.get("adamw_wd", 0.01),
        )
        scheduler_max_lr = section.get("adamw_lr", 1e-3)
        context.logger.info("Using AdamW-only optimizer for head '%s'." % head_name)
        return optimizer, scheduler_max_lr

    muon_params, adamw_params = split_params_for_muon(base_model)
    optimizer = Muon(
        muon_params,
        lr=section.get("muon_lr", 0.02),
        momentum=section.get("momentum", 0.95),
        adamw_params=adamw_params,
        adamw_lr=section.get("adamw_lr", 1e-3),
        adamw_wd=section.get("adamw_wd", 0.01),
        update_max_norm=section.get("optimizer_update_max_norm"),
    )
    scheduler_max_lr = section.get("muon_lr", 0.02)
    context.logger.info("Using Muon+AdamW optimizer for head '%s'." % head_name)
    return optimizer, scheduler_max_lr


def _compose_epoch_metrics(
    *,
    avg_train_loss: float,
    val_loss: float,
    val_metrics: dict[str, Any],
    lr_value: float,
    lr_muon_value: float,
    lr_adamw_value: float,
    epoch_health: dict[str, float],
    avg_train_loss_components: dict[str, float],
    xai_epoch_metrics: dict[str, float],
) -> dict[str, float]:
    """Compose per-epoch metric payload for hooks/logging.

    Args:
        avg_train_loss (float): Mean training loss across processed batches.
        val_loss (float): Validation loss value.
        val_metrics (dict[str, Any]): Validation metric dictionary.
        lr_value (float): Main scheduler learning rate.
        lr_muon_value (float): Muon learning rate component.
        lr_adamw_value (float): AdamW learning rate component.
        epoch_health (dict[str, float]): Epoch health counters and diagnostics.
        avg_train_loss_components (dict[str, float]): Mean per-loss components.
        xai_epoch_metrics (dict[str, float]): XAI metrics collected this epoch.

    Returns:
        dict[str, float]: Combined metric payload for hooks and logging.
    """

    return {
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "miou": float(val_metrics["miou"]),
        "mdice": float(val_metrics["mdice"]),
        "val_miou": float(val_metrics["miou"]),
        "val_mdice": float(val_metrics["mdice"]),
        "val_iou": float(val_metrics["miou"]),
        "val_f1": float(val_metrics["mdice"]),
        "lr": lr_value,
        "lr_muon": lr_muon_value,
        "lr_adamw": lr_adamw_value,
        "nonfinite_batches": float(epoch_health["nonfinite_batches"]),
        "skipped_optimizer_steps": float(epoch_health["skipped_optimizer_steps"]),
        "max_abs_logit": float(epoch_health["max_abs_logit"]),
        "grad_norm": float(epoch_health["grad_norm"]),
        "param_nonfinite_count": float(epoch_health["param_nonfinite_count"]),
        "optimizer_steps": float(epoch_health["optimizer_steps"]),
        "scheduler_steps": float(epoch_health["scheduler_steps"]),
        "nonfinite_val_batches": float(val_metrics.get("nonfinite_val_batches", 0.0)),
        "nonfinite_val_loss_batches": float(
            val_metrics.get("nonfinite_val_loss_batches", 0.0)
        ),
        **{key: float(avg_train_loss_components[key]) for key in LOSS_COMPONENT_KEYS},
        **{
            f"val_{key}": float(val_metrics.get(key, float("nan")))
            for key in LOSS_COMPONENT_KEYS
        },
        **xai_epoch_metrics,
    }


class TrainPhase(Phase):
    """Phase for training the segmentation head."""

    name = "train"
    config_key = "train"

    def execute(self, context: RunContext) -> PhaseOutcome:
        """Train the segmentation head on cached tiles.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Phase metrics and training artifacts.
        """

        try:
            return self._train(context)
        except Exception as exc:
            raise TrainingError(str(exc)) from exc

    def _train(self, context: RunContext) -> PhaseOutcome:
        """Execute the training loop and epoch-level validation.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Best metric values and checkpoint artifacts.
        """

        section = context.config.get(self.config_key, {})
        dataset_cfg = context.config.get("dataset", {})
        prepare_cfg = context.config.get("prepare", {})
        model_cfg = get_model_config(context.config)
        processed_dir = resolve_path(
            context.config, section, "processed_dir", DEFAULT_PROCESSED_DIR
        )
        weights_dir = section.get("weights_dir", "weights")
        os.makedirs(weights_dir, exist_ok=True)
        device = torch.device(section.get("device", DEFAULT_DEVICE))
        if context.dist_ctx.enabled:
            device = torch.device(f"cuda:{context.dist_ctx.local_rank}")
        batch_size = section.get("batch_size", 4)
        cache_features = bool(dataset_cfg.get("cache_features", True))
        tile_size = dataset_cfg.get("tile_size", prepare_cfg.get("tile_size"))
        processed_dir = resolve_cache_dir_for_train(
            processed_dir,
            tile_size,
            cache_features,
            context.logger,
        )
        max_tiles = dataset_cfg.get("max_tiles")
        context.logger.info(
            "Building dataloaders with batch_size=%s, num_workers=%s, "
            "cache_features=%s, max_tiles=%s, processed_dir=%s"
            % (
                batch_size,
                section.get("num_workers", 4),
                cache_features,
                max_tiles,
                processed_dir,
            )
        )
        train_loader, train_sampler, val_loader = create_dataloaders(
            processed_dir,
            dataset_cfg,
            section,
            batch_size,
            context.logger,
            context.dist_ctx,
        )
        context.logger.info(
            f"Dataset split: {dataset_size(train_loader.dataset)} train tiles."
        )
        if val_loader is not None:
            context.logger.info(f"Validation tiles: {dataset_size(val_loader.dataset)}")

        model = build_head(
            model_cfg["head"],
            num_classes=model_cfg["num_classes"],
            dino_channels=model_cfg["dino_channels"],
            model_cfg=context.config.get("model", {}),
        ).to(device)
        if section.get("compile", False) and hasattr(torch, "compile"):
            model = cast(torch.nn.Module, torch.compile(model))
        if context.dist_ctx.enabled:
            model = DDP(
                model,
                device_ids=[context.dist_ctx.local_rank],
                output_device=context.dist_ctx.local_rank,
                find_unused_parameters=False,
            )
        base_model = unwrap_model(cast(torch.nn.Module, model))
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(
            p.numel() for p in base_model.parameters() if p.requires_grad
        )
        non_trainable_params = total_params - trainable_params
        context.logger.info(
            f"Initialized head '{model_cfg['head']}' with {total_params:,} parameters."
        )
        if context.mlflow_logger and context.dist_ctx.is_main:
            model_size_payload = {
                "model_total_params": total_params,
                "model_trainable_params": trainable_params,
                "model_non_trainable_params": non_trainable_params,
            }
            context.mlflow_logger.log_params(
                {key: str(value) for key, value in model_size_payload.items()}
            )
            for key, value in model_size_payload.items():
                context.mlflow_logger.set_tag(key, str(value))

        head_name = str(model_cfg.get("head", ""))
        optimizer, scheduler_max_lr = _build_optimizer(
            context=context,
            head_name=head_name,
            base_model=base_model,
            section=section,
        )

        steps_per_epoch = math.ceil(
            len(train_loader) / max(1, section.get("grad_accum_steps", 1))
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=scheduler_max_lr,
            epochs=section.get("epochs", 30),
            steps_per_epoch=steps_per_epoch,
        )

        resolved_loss = parse_train_loss_config(
            section,
            (
                context.dataset_validation.ignore_index
                if context.dataset_validation.enabled
                else None
            ),
        )
        boundary_kernel_size = resolved_loss.boundary_kernel_size
        loss_ignore_index = resolved_loss.ignore_index
        loss_fn = SegmentationLoss(
            num_classes=model_cfg["num_classes"],
            ce_weight=resolved_loss.ce_weight,
            focal_weight=resolved_loss.focal_weight,
            dice_weight=resolved_loss.dice_weight,
            aux_weight=resolved_loss.aux_weight,
            class_weights=resolved_loss.class_weights,
            ignore_index=loss_ignore_index,
            label_smoothing=resolved_loss.label_smoothing,
            use_focal=False,
            focal_gamma=resolved_loss.focal_gamma,
            focal_alpha=resolved_loss.focal_alpha,
            boundary_weight=resolved_loss.boundary_weight,
            skeleton_weight=resolved_loss.skeleton_weight,
            topology_weight=resolved_loss.topology_weight,
            topology_class_index=resolved_loss.topology_class_index,
            topology_iters=resolved_loss.topology_iters,
            topology_on_aux=resolved_loss.topology_on_aux,
            topology_downsample=resolved_loss.topology_downsample,
        ).to(device)

        ps = 14 if "vitl14" in model_cfg["backbone"] else 16
        stability = context.stability
        use_amp = device.type == "cuda"
        if stability.amp_enabled == "off":
            use_amp = False
        elif stability.amp_enabled == "on" and device.type != "cuda":
            context.logger.info("AMP requested but CUDA is unavailable; using fp32.")
            use_amp = False
        use_grad_scaler = use_amp and stability.amp_dtype == "fp16"
        scaler = torch.cuda.amp.GradScaler() if use_grad_scaler else None
        autocast = build_autocast(use_amp=use_amp, amp_dtype=stability.amp_dtype)

        best_path = os.path.join(weights_dir, f"{model_cfg['head']}_best.pth")
        early_stopping = EarlyStopping(
            patience=section.get("patience", 10),
            min_delta=0.005,
            path=best_path,
            mode="max",
        )
        ema_decay = section.get("ema_decay", 0.0)
        ema = ModelEMA(base_model, ema_decay) if ema_decay > 0 else None
        epochs = section.get("epochs", 30)
        grad_accum = max(1, section.get("grad_accum_steps", 1))

        plot_cfg = parse_train_plot_config(section)
        plot_root_dir = plot_cfg.root_dir
        if context.mlflow_logger is not None:
            plot_root_dir = str(context.mlflow_logger.artifacts_dir / "plots")
        plot_metrics_dir = os.path.join(plot_root_dir, "metrics")
        plot_xai_dir = os.path.join(plot_root_dir, "xai")
        plot_xai_cam_layer = resolve_cam_layer(
            model_cfg["layers"], plot_cfg.xai_cam_layer_mode
        )
        plot_xai_pca_layer_mode = plot_cfg.xai_pca_layer_mode
        if plot_xai_pca_layer_mode.strip().lower() == "same_as_cam":
            plot_xai_pca_layer = plot_xai_cam_layer
        else:
            plot_xai_pca_layer = resolve_cam_layer(
                model_cfg["layers"], plot_xai_pca_layer_mode
            )

        model_layer_ids = [int(layer_id) for layer_id in model_cfg["layers"]]
        histories: dict[str, Any] = {
            "channel_importance_history": [],
            "branch_importance_history": [],
            "dino_layer_importance_history": [],
            "module_xai_history": {},
        }

        log_batch_metrics = get_hook_option(context.config, "log_batch_metrics", False)
        log_batch_interval = get_hook_option(context.config, "log_batch_interval", 10)
        backbone = None
        processor = None
        best_miou = 0.0
        final_val_loss = 0.0

        with TimedBlock(context.logger, "Training phase"):
            train_phase_start = time.time()
            next_progress_pct = 5
            for epoch in range(epochs):
                epoch_started_at = time.time()
                context.hook_manager.on_epoch_start(context, self.name, epoch + 1)
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                with TimedBlock(context.logger, f"Epoch {epoch + 1}"):
                    (
                        avg_train_loss,
                        avg_train_loss_components,
                        epoch_health,
                        epoch_aborted,
                        backbone,
                        processor,
                    ) = run_train_epoch_batches(
                        context=context,
                        epoch=epoch,
                        epochs=epochs,
                        model=cast(torch.nn.Module, model),
                        base_model=base_model,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        device=device,
                        ps=ps,
                        cache_features=cache_features,
                        model_cfg=model_cfg,
                        backbone=backbone,
                        processor=processor,
                        autocast=autocast,
                        scaler=scaler,
                        grad_accum=grad_accum,
                        stability=stability,
                        boundary_kernel_size=boundary_kernel_size,
                        log_batch_metrics=log_batch_metrics,
                        log_batch_interval=log_batch_interval,
                        ema=ema,
                        weights_dir=weights_dir,
                    )
                    if epoch_aborted:
                        context.logger.error(
                            f"Epoch {epoch + 1} aborted due to non-finite values."
                        )
                        continue

                    eval_model = (
                        ema.ema_model
                        if ema
                        else unwrap_model(cast(torch.nn.Module, model))
                    )
                    if not cache_features:
                        backbone, processor = ensure_backbone_processor(
                            backbone,
                            processor,
                            model_cfg["backbone"],
                            device,
                        )
                    val_loss, val_metrics = evaluate(
                        eval_model,
                        val_loader,
                        loss_fn,
                        device,
                        use_amp,
                        context.logger if context.dist_ctx.is_main else None,
                        model_cfg["num_classes"],
                        cache_features=cache_features,
                        backbone=backbone,
                        processor=processor,
                        layers=model_cfg["layers"],
                        ps=ps,
                        stability=stability,
                        boundary_kernel_size=boundary_kernel_size,
                    )
                    xai_epoch_metrics, backbone, processor = collect_epoch_xai_metrics(
                        context=context,
                        epoch=epoch,
                        eval_model=eval_model,
                        val_loader=val_loader,
                        cache_features=cache_features,
                        model_cfg=model_cfg,
                        loss_ignore_index=loss_ignore_index,
                        plot_cfg=plot_cfg,
                        plot_metrics_dir=plot_metrics_dir,
                        plot_xai_dir=plot_xai_dir,
                        plot_xai_cam_layer=plot_xai_cam_layer,
                        plot_xai_pca_layer=plot_xai_pca_layer,
                        model_layer_ids=model_layer_ids,
                        backbone=backbone,
                        processor=processor,
                        device=device,
                        ps=ps,
                        autocast=autocast,
                        histories=histories,
                    )
                    if context.dist_ctx.enabled:
                        loss_tensor = torch.tensor(
                            [val_loss, val_metrics["miou"]], device=device
                        )
                        dist.broadcast(loss_tensor, src=0)
                        val_loss = loss_tensor[0].item()
                        val_metrics["miou"] = loss_tensor[1].item()

                    context.logger.info(
                        f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val mIoU: {val_metrics['miou']:.4f}"
                    )
                    epoch_duration = time.time() - epoch_started_at
                    elapsed_total = time.time() - train_phase_start
                    completed_epochs = epoch + 1
                    avg_completed_epoch_duration = elapsed_total / float(
                        completed_epochs
                    )
                    remaining_epochs = max(0, epochs - completed_epochs)
                    epoch_eta = avg_completed_epoch_duration * float(remaining_epochs)
                    completed_pct = int(
                        math.floor((100.0 * completed_epochs) / float(max(1, epochs)))
                    )
                    while (
                        completed_pct >= next_progress_pct and next_progress_pct <= 100
                    ):
                        context.logger.info(
                            "Training progress %s%% | epoch %s/%s | last epoch %.1fs | ETA %.1fs"
                            % (
                                next_progress_pct,
                                completed_epochs,
                                epochs,
                                epoch_duration,
                                epoch_eta,
                            )
                        )
                        next_progress_pct += 5
                    epoch_health["param_nonfinite_count"] = float(
                        count_nonfinite_parameters(eval_model)
                    )
                    epoch_ckpt = os.path.join(
                        weights_dir,
                        (
                            f"{model_cfg['head']}_VALLOSS_{val_loss:.4f}_"
                            f"MIOU_{val_metrics['miou']:.4f}_EPOCH_{epoch + 1}.pth"
                        ),
                    )
                    checkpoint_is_finite = (
                        math.isfinite(avg_train_loss)
                        and math.isfinite(val_loss)
                        and epoch_health["param_nonfinite_count"] == 0
                    )
                    if context.dist_ctx.is_main and checkpoint_is_finite:
                        torch.save(eval_model.state_dict(), epoch_ckpt)
                    elif context.dist_ctx.is_main:
                        context.logger.error(
                            "Skipping checkpoint save due to non-finite training state."
                        )

                    stop_flag = False
                    if context.dist_ctx.is_main and checkpoint_is_finite:
                        early_stopping(val_metrics["miou"], eval_model)
                        stop_flag = early_stopping.early_stop
                    elif (
                        not checkpoint_is_finite
                        and stability.nonfinite_action == "stop_run"
                    ):
                        stop_flag = True
                    if context.dist_ctx.enabled:
                        flag_tensor = torch.tensor(1 if stop_flag else 0, device=device)
                        dist.broadcast(flag_tensor, src=0)
                        stop_flag = bool(flag_tensor.item())

                    lr_value, lr_muon_value, lr_adamw_value = resolve_lr_metrics(
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                    epoch_metrics = _compose_epoch_metrics(
                        avg_train_loss=avg_train_loss,
                        val_loss=val_loss,
                        val_metrics=val_metrics,
                        lr_value=lr_value,
                        lr_muon_value=lr_muon_value,
                        lr_adamw_value=lr_adamw_value,
                        epoch_health=epoch_health,
                        avg_train_loss_components=avg_train_loss_components,
                        xai_epoch_metrics=xai_epoch_metrics,
                    )
                    if (
                        epoch_metrics["optimizer_steps"]
                        != epoch_metrics["scheduler_steps"]
                    ):
                        context.logger.error(
                            "Optimizer/scheduler step mismatch at epoch %s: "
                            "optimizer_steps=%s scheduler_steps=%s"
                            % (
                                epoch + 1,
                                int(epoch_metrics["optimizer_steps"]),
                                int(epoch_metrics["scheduler_steps"]),
                            )
                        )
                    context.hook_manager.on_epoch_end(
                        context, self.name, epoch + 1, epoch_metrics
                    )
                    context.hook_manager.on_metrics(
                        context,
                        self.name,
                        epoch + 1,
                        epoch_metrics,
                    )
                    if math.isfinite(float(val_metrics["miou"])):
                        best_miou = max(best_miou, float(val_metrics["miou"]))
                    final_val_loss = val_loss
                    if stop_flag:
                        if context.dist_ctx.is_main:
                            if (
                                not checkpoint_is_finite
                                and stability.nonfinite_action == "stop_run"
                            ):
                                context.logger.info(
                                    "Stopping training due to non-finite state."
                                )
                            else:
                                context.logger.info("Early stopping triggered.")
                        break

        if context.dist_ctx.is_main:
            context.logger.info(f"Training finished. Best weights saved to {best_path}")
        artifacts = {"best_checkpoint": best_path, "weights_dir": weights_dir}
        metrics = {"best_miou": best_miou, "final_val_loss": final_val_loss}
        return PhaseOutcome(metrics=metrics, artifacts=artifacts)
