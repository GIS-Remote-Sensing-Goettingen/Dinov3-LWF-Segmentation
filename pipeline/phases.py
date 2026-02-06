"""Pipeline phase implementations for preparation, verification, training, and inference."""

from __future__ import annotations

import glob
import math
import os
import random
import time
import traceback
from contextlib import nullcontext
from typing import Any, cast

import numpy as np
import rasterio
import torch
import torch.distributed as dist
import torch.nn.functional as F
from rasterio.windows import Window
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from models import build_head
from utils import (
    EarlyStopping,
    Muon,
    SegmentationLoss,
    TimedBlock,
    extract_multiscale_features,
    prepare_data_tiles,
    resolve_cache_dir_for_prepare,
    resolve_cache_dir_for_train,
    verify_and_clean_dataset_fast,
)

from .constants import (
    DEFAULT_DEVICE,
    DEFAULT_LABEL_PATH,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_IMAGES_DIR,
)
from .context import InferenceError, PhaseOutcome, RunContext, TrainingError
from .data_splits import create_dataloaders, dataset_size
from .inference_utils import (
    build_dashboard,
    build_tta_transforms,
    compute_attention_maps,
    compute_gradcam_map,
    compute_xai_maps,
    overlay_heatmap,
    upsample_map,
)
from .phase_runner import Phase
from .train_utils import (
    ModelEMA,
    align_labels_to_logits,
    evaluate,
    extract_multiscale_features_batch,
    move_features_to_device,
    split_params_for_muon,
)
from .utils import get_hook_option, get_model_config, resolve_path, unwrap_model


class PreparePhase(Phase):
    """Phase for tiling data and caching DINO features.

    Examples:
        >>> PreparePhase().name
        'prepare'
    """

    name = "prepare"
    config_key = "prepare"

    def execute(self, context: RunContext) -> PhaseOutcome:
        """Run tiling and feature caching.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts from the phase.
        """

        section = context.config.get(self.config_key, {})
        dataset_cfg = context.config.get("dataset", {})
        model_cfg = get_model_config(context.config)
        img_dir = resolve_path(
            context.config, section, "img_dir", DEFAULT_RAW_IMAGES_DIR
        )
        label_path = resolve_path(
            context.config, section, "label_path", DEFAULT_LABEL_PATH
        )
        output_dir = resolve_path(
            context.config, section, "output_dir", DEFAULT_PROCESSED_DIR
        )
        device = torch.device(section.get("device", DEFAULT_DEVICE))
        if context.dist_ctx.enabled:
            device = torch.device(f"cuda:{context.dist_ctx.local_rank}")
        cache_features = bool(section.get("cache_features", True))
        tile_size = section.get("tile_size", 512)
        output_dir = resolve_cache_dir_for_prepare(
            output_dir,
            tile_size,
            cache_features,
            model_cfg["backbone"],
            model_cfg["layers"],
            context.logger,
        )
        before_count = len(glob.glob(os.path.join(output_dir, "*.pt")))
        max_tiles = dataset_cfg.get("max_tiles")
        with TimedBlock(context.logger, "Preparation phase"):
            prepare_data_tiles(
                img_dir=img_dir,
                label_path=label_path,
                output_dir=output_dir,
                model_name=model_cfg["backbone"],
                layers=model_cfg["layers"],
                device=device,
                tile_size=tile_size,
                cache_features=cache_features,
                workers=section.get("workers"),
                max_tiles=max_tiles,
                logger=context.logger,
            )
        after_count = len(glob.glob(os.path.join(output_dir, "*.pt")))
        metrics = {
            "tiles_total": float(after_count),
            "tiles_added": float(max(after_count - before_count, 0)),
        }
        artifacts = {"processed_dir": output_dir}
        return PhaseOutcome(metrics=metrics, artifacts=artifacts)


class VerifyPhase(Phase):
    """Phase for verifying cached tile integrity.

    Examples:
        >>> VerifyPhase().name
        'verify'
    """

    name = "verify"
    config_key = "verify"

    def execute(self, context: RunContext) -> PhaseOutcome:
        """Verify cached tiles and remove corrupted entries.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts from the phase.
        """

        section = context.config.get(self.config_key, {})
        dataset_cfg = context.config.get("dataset", {})
        prepare_cfg = context.config.get("prepare", {})
        processed_dir = resolve_path(
            context.config, section, "processed_dir", DEFAULT_PROCESSED_DIR
        )
        cache_features = dataset_cfg.get("cache_features")
        tile_size = dataset_cfg.get("tile_size", prepare_cfg.get("tile_size"))
        processed_dir = resolve_cache_dir_for_train(
            processed_dir,
            tile_size,
            cache_features if cache_features is not None else None,
            context.logger,
        )
        before_count = len(glob.glob(os.path.join(processed_dir, "*.pt")))
        with TimedBlock(context.logger, "Verification phase"):
            verify_and_clean_dataset_fast(
                processed_dir,
                num_workers=section.get("workers"),
                logger=context.logger,
            )
        after_count = len(glob.glob(os.path.join(processed_dir, "*.pt")))
        removed = max(before_count - after_count, 0)
        metrics = {
            "tiles_total": float(after_count),
            "tiles_removed": float(removed),
        }
        artifacts = {"processed_dir": processed_dir}
        return PhaseOutcome(metrics=metrics, artifacts=artifacts)


class TrainPhase(Phase):
    """Phase for training the segmentation head.

    Examples:
        >>> TrainPhase().name
        'train'
    """

    name = "train"
    config_key = "train"

    def execute(self, context: RunContext) -> PhaseOutcome:
        """Train the segmentation head on cached tiles.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts from the phase.

        Raises:
            TrainingError: If training fails unexpectedly.
        """

        try:
            return self._train(context)
        except Exception as exc:
            raise TrainingError(str(exc)) from exc

    def _train(self, context: RunContext) -> PhaseOutcome:
        """Internal training implementation.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts from the phase.
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
        loader_start = time.time()
        train_loader, train_sampler, val_loader = create_dataloaders(
            processed_dir,
            dataset_cfg,
            section,
            batch_size,
            context.logger,
            context.dist_ctx,
        )
        context.logger.info(f"Dataloaders ready in {time.time() - loader_start:.2f}s")
        context.logger.info(
            f"Dataset split: {dataset_size(train_loader.dataset)} train tiles."
        )
        if val_loader is not None:
            context.logger.info(f"Validation tiles: {dataset_size(val_loader.dataset)}")

        model = build_head(
            model_cfg["head"],
            num_classes=model_cfg["num_classes"],
            dino_channels=model_cfg["dino_channels"],
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
        context.logger.info(
            f"Initialized head '{model_cfg['head']}' with {total_params:,} parameters."
        )
        muon_params, adamw_params = split_params_for_muon(base_model)
        optimizer = Muon(
            muon_params,
            lr=section.get("muon_lr", 0.02),
            momentum=section.get("momentum", 0.95),
            adamw_params=adamw_params,
            adamw_lr=section.get("adamw_lr", 1e-3),
        )
        steps_per_epoch = math.ceil(
            len(train_loader) / max(1, section.get("grad_accum_steps", 1))
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=section.get("muon_lr", 0.02),
            epochs=section.get("epochs", 30),
            steps_per_epoch=steps_per_epoch,
        )
        loss_cfg = section.get("loss", {})
        loss_fn = SegmentationLoss(
            num_classes=model_cfg["num_classes"],
            ce_weight=loss_cfg.get("ce_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            aux_weight=loss_cfg.get("aux_weight", 0.4),
            class_weights=loss_cfg.get("class_weights"),
            ignore_index=loss_cfg.get("ignore_index"),
        ).to(device)
        backbone = None
        processor = None
        ps = 14 if "vitl14" in model_cfg["backbone"] else 16
        use_amp = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        autocast = torch.cuda.amp.autocast() if use_amp else nullcontext()
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
        log_batch_metrics = get_hook_option(context.config, "log_batch_metrics", False)
        log_batch_interval = get_hook_option(context.config, "log_batch_interval", 10)
        plot_enabled = bool(section.get("epoch_plot", False))
        plot_dir = section.get("epoch_plot_dir", os.path.join("output", "plot"))
        plot_cmap = section.get("epoch_plot_cmap", "tab20")

        context.logger.info(f"Training for up to {epochs} epochs on device {device}.")
        best_miou = 0.0
        final_val_loss = 0.0

        with TimedBlock(context.logger, "Training phase"):
            for epoch in range(epochs):
                context.hook_manager.on_epoch_start(context, self.name, epoch + 1)
                epoch_start = time.time()
                first_batch_logged = False
                last_log_time = epoch_start
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                with TimedBlock(context.logger, f"Epoch {epoch + 1}"):
                    model_call = cast(Any, model)
                    model_call.train()
                    train_loss = 0.0
                    optimizer.zero_grad()
                    pbar = tqdm(
                        train_loader,
                        desc=f"Epoch {epoch + 1}/{epochs} [Train]",
                        leave=False,
                    )
                    for batch_idx, (img, features, y) in enumerate(pbar, 1):
                        if not first_batch_logged:
                            first_batch_logged = True
                            first_delay = time.time() - epoch_start
                            context.logger.info(
                                f"Epoch {epoch + 1} first batch received after "
                                f"{first_delay:.2f}s"
                            )
                            last_log_time = time.time()
                        img = img.to(device)
                        y = y.to(device)
                        try:
                            if cache_features and features:
                                feats = move_features_to_device(features, device)
                            else:
                                if backbone is None or processor is None:
                                    processor = AutoImageProcessor.from_pretrained(
                                        model_cfg["backbone"]
                                    )
                                    backbone = (
                                        AutoModel.from_pretrained(model_cfg["backbone"])
                                        .eval()
                                        .to(device)
                                    )
                                feats = extract_multiscale_features_batch(
                                    img,
                                    backbone,
                                    processor,
                                    device,
                                    model_cfg["layers"],
                                    ps,
                                )
                            model_call = cast(Any, model)
                            with autocast:
                                if hasattr(model_call, "forward_with_aux"):
                                    logits, aux_logits = model_call.forward_with_aux(
                                        img, feats
                                    )
                                else:
                                    logits = model_call(img, feats)
                                    aux_logits = None
                                target_main = align_labels_to_logits(y, logits)
                                target_aux = (
                                    align_labels_to_logits(y, aux_logits)
                                    if aux_logits is not None
                                    else None
                                )
                                loss = loss_fn(
                                    logits,
                                    target_main,
                                    aux_logits=aux_logits,
                                    aux_targets=target_aux,
                                )
                                loss = loss / grad_accum
                        except Exception as exc:
                            context.logger.info(
                                "Batch %s failed with %s; img=%s, features=%s, layers=%s"
                                % (
                                    batch_idx,
                                    exc,
                                    tuple(img.shape),
                                    len(features) if features is not None else 0,
                                    model_cfg["layers"],
                                )
                            )
                            raise
                        if scaler:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        if batch_idx % grad_accum == 0 or batch_idx == len(
                            train_loader
                        ):
                            if scaler:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            optimizer.zero_grad()
                            scheduler.step()
                            if ema:
                                ema.update(unwrap_model(model))
                        train_loss += loss.item() * grad_accum
                        if log_batch_metrics and batch_idx % log_batch_interval == 0:
                            batch_metrics = {
                                "loss": loss.item() * grad_accum,
                                "lr": scheduler.get_last_lr()[0],
                            }
                            context.hook_manager.on_batch_end(
                                context, self.name, batch_idx, batch_metrics
                            )
                            context.hook_manager.on_metrics(
                                context,
                                self.name,
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
                                f"Epoch {epoch + 1} batch {batch_idx}/"
                                f"{len(train_loader)} avg batch time "
                                f"{avg_batch:.2f}s"
                            )
                            last_log_time = now
                    avg_train_loss = train_loss / len(train_loader)
                    eval_model = (
                        ema.ema_model
                        if ema
                        else unwrap_model(cast(torch.nn.Module, model))
                    )
                    if not cache_features and (backbone is None or processor is None):
                        processor = AutoImageProcessor.from_pretrained(
                            model_cfg["backbone"]
                        )
                        backbone = (
                            AutoModel.from_pretrained(model_cfg["backbone"])
                            .eval()
                            .to(device)
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
                    )
                    if (
                        plot_enabled
                        and val_loader is not None
                        and context.dist_ctx.is_main
                    ):
                        os.makedirs(plot_dir, exist_ok=True)
                        target_batch = random.randint(1, len(val_loader))
                        sampled = None
                        for batch_idx, (v_img, v_feats, v_y) in enumerate(
                            val_loader, 1
                        ):
                            if batch_idx == target_batch:
                                sampled = (v_img, v_feats, v_y)
                                break
                        if sampled is not None:
                            v_img, v_feats, v_y = sampled
                            v_img = v_img.to(device)
                            v_y = v_y.to(device)
                            if cache_features and v_feats:
                                v_feats = move_features_to_device(v_feats, device)
                            else:
                                if backbone is None or processor is None:
                                    processor = AutoImageProcessor.from_pretrained(
                                        model_cfg["backbone"]
                                    )
                                    backbone = (
                                        AutoModel.from_pretrained(model_cfg["backbone"])
                                        .eval()
                                        .to(device)
                                    )
                                v_feats = extract_multiscale_features_batch(
                                    v_img,
                                    backbone,
                                    processor,
                                    device,
                                    model_cfg["layers"],
                                    ps,
                                )
                            eval_call = cast(Any, eval_model)
                            with torch.no_grad(), autocast:
                                if hasattr(eval_call, "forward_with_aux"):
                                    v_logits, _ = eval_call.forward_with_aux(
                                        v_img, v_feats
                                    )
                                else:
                                    v_logits = eval_call(v_img, v_feats)
                                if v_logits.shape[-2:] != v_img.shape[-2:]:
                                    v_logits = F.interpolate(
                                        v_logits,
                                        size=v_img.shape[-2:],
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                            pred_mask = v_logits.argmax(dim=1).detach().cpu().numpy()
                            gt_mask = v_y.detach().cpu().numpy()
                            rgb = v_img.detach().cpu().numpy().transpose(0, 2, 3, 1)
                            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                            sample_idx = random.randint(0, pred_mask.shape[0] - 1)
                            out_path = os.path.join(
                                plot_dir, f"epoch_{epoch + 1:04d}.png"
                            )
                            save_epoch_plot(
                                out_path,
                                rgb[sample_idx],
                                gt_mask[sample_idx],
                                pred_mask[sample_idx],
                                plot_cmap,
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
                    epoch_ckpt = os.path.join(
                        weights_dir,
                        (
                            f"{model_cfg['head']}_VALLOSS_{val_loss:.4f}_"
                            f"MIOU_{val_metrics['miou']:.4f}_EPOCH_{epoch + 1}.pth"
                        ),
                    )
                    if context.dist_ctx.is_main:
                        torch.save(eval_model.state_dict(), epoch_ckpt)
                    stop_flag = False
                    if context.dist_ctx.is_main:
                        early_stopping(val_metrics["miou"], eval_model)
                        stop_flag = early_stopping.early_stop
                    if context.dist_ctx.enabled:
                        flag_tensor = torch.tensor(1 if stop_flag else 0, device=device)
                        dist.broadcast(flag_tensor, src=0)
                        stop_flag = bool(flag_tensor.item())
                    epoch_metrics = {
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "miou": float(val_metrics["miou"]),
                        "mdice": float(val_metrics["mdice"]),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    context.hook_manager.on_epoch_end(
                        context, self.name, epoch + 1, epoch_metrics
                    )
                    context.hook_manager.on_metrics(
                        context,
                        self.name,
                        epoch + 1,
                        epoch_metrics,
                    )
                    best_miou = max(best_miou, float(val_metrics["miou"]))
                    final_val_loss = val_loss
                    if stop_flag:
                        if context.dist_ctx.is_main:
                            context.logger.info("Early stopping triggered.")
                        break
        if context.dist_ctx.is_main:
            context.logger.info(f"Training finished. Best weights saved to {best_path}")
        artifacts = {"best_checkpoint": best_path, "weights_dir": weights_dir}
        metrics = {"best_miou": best_miou, "final_val_loss": final_val_loss}
        return PhaseOutcome(metrics=metrics, artifacts=artifacts)


class InferencePhase(Phase):
    """Phase for sliding-window inference.

    Examples:
        >>> InferencePhase().name
        'inference'
    """

    name = "inference"
    config_key = "inference"

    def is_enabled(self, context: RunContext) -> bool:
        """Return True when inference should execute on this rank.

        Args:
            context (RunContext): Active run context.

        Returns:
            bool: True when inference should run.
        """

        if not context.dist_ctx.is_main:
            return False
        infer_cfg = context.config.get("inference", context.config.get("infer", {}))
        return bool(infer_cfg and infer_cfg.get("enable", False))

    def execute(self, context: RunContext) -> PhaseOutcome:
        """Run sliding-window inference over a large raster.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts from the phase.

        Raises:
            InferenceError: If inference fails unexpectedly.
        """

        try:
            return self._infer(context)
        except Exception as exc:
            raise InferenceError(str(exc)) from exc

    def _infer(self, context: RunContext) -> PhaseOutcome:
        """Internal inference implementation.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts from the phase.
        """

        infer_cfg = context.config.get("inference", context.config.get("infer", {}))
        model_cfg = get_model_config(context.config)
        device = torch.device(infer_cfg.get("device", DEFAULT_DEVICE))
        processor = AutoImageProcessor.from_pretrained(model_cfg["backbone"])
        backbone = AutoModel.from_pretrained(model_cfg["backbone"]).eval().to(device)
        head = build_head(
            model_cfg["head"],
            num_classes=model_cfg["num_classes"],
            dino_channels=model_cfg["dino_channels"],
        ).to(device)
        checkpoint = infer_cfg["checkpoint"]
        context.logger.info(f"Loading checkpoint {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=device)
        head.load_state_dict(state_dict, strict=False)
        head.eval()
        input_dir = infer_cfg.get("input_dir")
        input_tif = infer_cfg.get("input_tif")
        output_dir = infer_cfg.get("output_dir")
        output_tif = infer_cfg.get("output_tif")
        tile_size = infer_cfg.get("tile_size", 512)
        ps = 14 if "vitl14" in model_cfg["backbone"] else 16
        overlap_cfg = infer_cfg.get("overlap", 0.0)
        overlap_px = (
            int(tile_size * overlap_cfg) if overlap_cfg < 1 else int(overlap_cfg)
        )
        stride = max(1, tile_size - overlap_px)
        tta_transforms = build_tta_transforms(infer_cfg.get("tta", {}))
        autocast = torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()
        explain_cfg = infer_cfg.get("explain", {})
        explain_enabled = bool(explain_cfg.get("enable", False))
        plots_dir = explain_cfg.get("output_dir")
        class_index = int(explain_cfg.get("class_index", 1))
        layout = explain_cfg.get("dashboard_layout", "4x3")
        plot_every_n = explain_cfg.get("plot_every_n")
        output_suffix = infer_cfg.get("output_suffix", "_pred.tif")
        glob_pattern = infer_cfg.get("glob", "*.tif")
        if input_dir:
            if not output_dir:
                raise InferenceError("output_dir is required when input_dir is set")
            os.makedirs(output_dir, exist_ok=True)
            if explain_enabled:
                plots_dir = plots_dir or os.path.join(output_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)
            tile_files = sorted(glob.glob(os.path.join(input_dir, glob_pattern)))
            if not tile_files:
                raise InferenceError(f"No input tiles found in {input_dir}")
            if plot_every_n is None:
                plot_every_n = 10 if len(tile_files) > 50 else 1

            def _infer_tile_file(path: str, index: int) -> None:
                """Run inference for a single tile file and write outputs.

                Args:
                    path (str): Input tile path.
                    index (int): 1-based tile index for plotting frequency.
                """

                with rasterio.open(path) as src:
                    profile = src.profile.copy()
                    img = src.read()
                img = np.transpose(img, (1, 2, 0))
                if img.shape[2] != 3:
                    raise InferenceError("Expected 3-band imagery.")
                orig_h, orig_w = img.shape[:2]
                probs_accum = np.zeros(
                    (model_cfg["num_classes"], orig_h, orig_w), dtype=np.float32
                )
                for transform in tta_transforms:
                    aug_img = transform.apply(img)
                    img_norm = (aug_img.astype(np.float32) / 255.0).astype(np.float32)
                    img_t = (
                        torch.from_numpy(img_norm)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device)
                    )
                    feats = extract_multiscale_features(
                        aug_img.astype(np.float32),
                        backbone,
                        processor,
                        device,
                        model_cfg["layers"],
                        ps=ps,
                    )
                    feats_batched = [f.to(device).unsqueeze(0) for f in feats]
                    with torch.no_grad(), autocast:
                        logits = head(img_t, feats_batched)
                        logits = transform.invert_logits(logits)
                        if logits.shape[-2:] != img_t.shape[-2:]:
                            logits = F.interpolate(
                                logits,
                                size=img_t.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    probs_accum += probs
                probs_accum /= len(tta_transforms)
                pred = probs_accum.argmax(axis=0).astype(np.uint8)
                profile.update(dtype=rasterio.uint8, count=1, nodata=0)
                base = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(output_dir, f"{base}{output_suffix}")
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(pred, 1)
                if explain_enabled and plot_every_n and index % plot_every_n == 0:
                    try:
                        rgb = np.clip(img, 0, 255).astype(np.uint8)
                        attn_cls, attn_rollout, had_attn = compute_attention_maps(
                            img.astype(np.float32),
                            backbone,
                            processor,
                            device,
                            ps,
                            logger=context.logger,
                        )
                        if not had_attn:
                            context.logger.info(f"Using Grad-CAM fallback for {base}.")
                            gradcam = compute_gradcam_map(
                                img.astype(np.float32),
                                backbone,
                                head,
                                processor,
                                device,
                                model_cfg["layers"],
                                ps,
                                class_index,
                                logger=context.logger,
                            )
                            attn_cls = gradcam
                            attn_rollout = gradcam
                        attn_cls = upsample_map(attn_cls, orig_h, orig_w)
                        attn_rollout = upsample_map(attn_rollout, orig_h, orig_w)
                        conf, ent, class_prob = compute_xai_maps(
                            probs_accum, class_index
                        )
                        overlay_pred = overlay_heatmap(
                            rgb,
                            pred.astype(np.float32)
                            / max(1, model_cfg["num_classes"] - 1),
                        )
                        overlay_attn = overlay_heatmap(rgb, attn_cls)
                        plot_path = os.path.join(plots_dir, f"{base}_dashboard.png")
                        build_dashboard(
                            plot_path,
                            rgb,
                            pred,
                            conf,
                            ent,
                            class_prob,
                            attn_cls,
                            attn_rollout,
                            overlay_pred,
                            overlay_attn,
                            layout=layout,
                        )
                    except Exception:
                        context.logger.error(
                            "XAI plotting failed for %s\n%s"
                            % (base, traceback.format_exc())
                        )

            for idx, tile_path in enumerate(tile_files, start=1):
                context.logger.info(f"Running tile inference {idx}/{len(tile_files)}")
                _infer_tile_file(tile_path, idx)
            metrics = {"files_total": float(len(tile_files))}
            artifacts = {"output_dir": output_dir, "checkpoint": checkpoint}
            return PhaseOutcome(metrics=metrics, artifacts=artifacts)
        if not input_tif or not output_tif:
            raise InferenceError(
                "input_tif and output_tif are required for single-file inference"
            )
        with rasterio.open(input_tif) as src:
            profile = src.profile.copy()
            height, width = src.height, src.width
            channels = src.count
        if channels != 3:
            raise InferenceError("Expected 3-band imagery.")
        prob_accum = np.zeros(
            (model_cfg["num_classes"], height, width), dtype=np.float32
        )
        count_accum = np.zeros((height, width), dtype=np.float32)
        total_tiles = math.ceil(height / stride) * math.ceil(width / stride)
        context.logger.info(
            f"Running inference on {total_tiles} tiles with stride {stride}."
        )
        if explain_enabled:
            plots_dir = plots_dir or os.path.join(os.path.dirname(output_tif), "plots")
            os.makedirs(plots_dir, exist_ok=True)
            if plot_every_n is None:
                plot_every_n = 10 if total_tiles > 50 else 1
        tile_counter = 0
        with (
            rasterio.open(input_tif) as src,
            TimedBlock(context.logger, "Inference phase"),
        ):
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    tile_counter += 1
                    y_max = min(y + tile_size, height)
                    x_max = min(x + tile_size, width)
                    window = Window.from_slices((y, y_max), (x, x_max))
                    img_tile = src.read(window=window, boundless=True)
                    img_tile = np.transpose(img_tile, (1, 2, 0))
                    if np.max(img_tile) == 0:
                        continue
                    img_tile_raw = img_tile
                    orig_h, orig_w = img_tile.shape[:2]
                    pad_h = max(0, tile_size - orig_h)
                    pad_w = max(0, tile_size - orig_w)
                    if pad_h or pad_w:
                        img_tile = np.pad(
                            img_tile,
                            ((0, pad_h), (0, pad_w), (0, 0)),
                            mode="reflect",
                        )
                    tile_probs = np.zeros(
                        (model_cfg["num_classes"], orig_h, orig_w), dtype=np.float32
                    )
                    for transform in tta_transforms:
                        aug_img = transform.apply(img_tile)
                        img_tile_norm = (aug_img.astype(np.float32) / 255.0).astype(
                            np.float32
                        )
                        img_t = (
                            torch.from_numpy(img_tile_norm)
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .to(device)
                        )
                        feats = extract_multiscale_features(
                            aug_img.astype(np.float32),
                            backbone,
                            processor,
                            device,
                            model_cfg["layers"],
                            ps=ps,
                        )
                        feats_batched = [f.to(device).unsqueeze(0) for f in feats]
                        with torch.no_grad(), autocast:
                            logits = head(img_t, feats_batched)
                            logits = transform.invert_logits(logits)
                            if logits.shape[-2:] != img_t.shape[-2:]:
                                logits = F.interpolate(
                                    logits,
                                    size=img_t.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            probs = (
                                torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                            )
                        probs = probs[:, :orig_h, :orig_w]
                        tile_probs += probs
                    tile_probs /= len(tta_transforms)
                    if (
                        explain_enabled
                        and plot_every_n
                        and tile_counter % plot_every_n == 0
                    ):
                        try:
                            rgb = np.clip(img_tile_raw, 0, 255).astype(np.uint8)
                            attn_cls, attn_rollout, had_attn = compute_attention_maps(
                                img_tile_raw.astype(np.float32),
                                backbone,
                                processor,
                                device,
                                ps,
                                logger=context.logger,
                            )
                            if not had_attn:
                                context.logger.info(
                                    f"Using Grad-CAM fallback for tile y={y} x={x}."
                                )
                                gradcam = compute_gradcam_map(
                                    img_tile_raw.astype(np.float32),
                                    backbone,
                                    head,
                                    processor,
                                    device,
                                    model_cfg["layers"],
                                    ps,
                                    class_index,
                                    logger=context.logger,
                                )
                                attn_cls = gradcam
                                attn_rollout = gradcam
                            attn_cls = upsample_map(attn_cls, orig_h, orig_w)
                            attn_rollout = upsample_map(attn_rollout, orig_h, orig_w)
                            conf, ent, class_prob = compute_xai_maps(
                                tile_probs, class_index
                            )
                            pred_tile = tile_probs.argmax(axis=0).astype(np.uint8)
                            overlay_pred = overlay_heatmap(
                                rgb,
                                pred_tile.astype(np.float32)
                                / max(1, model_cfg["num_classes"] - 1),
                            )
                            overlay_attn = overlay_heatmap(rgb, attn_cls)
                            plot_path = os.path.join(
                                plots_dir, f"tile_y{y}_x{x}_dashboard.png"
                            )
                            build_dashboard(
                                plot_path,
                                rgb,
                                pred_tile,
                                conf,
                                ent,
                                class_prob,
                                attn_cls,
                                attn_rollout,
                                overlay_pred,
                                overlay_attn,
                                layout=layout,
                            )
                        except Exception:
                            context.logger.error(
                                "XAI plotting failed for tile y=%s x=%s\n%s"
                                % (y, x, traceback.format_exc())
                            )
                    prob_accum[:, y:y_max, x:x_max] += tile_probs
                    count_accum[y:y_max, x:x_max] += 1
                    if tile_counter % 50 == 0 or tile_counter == total_tiles:
                        context.logger.info(
                            f"Inference progress: {tile_counter}/{total_tiles} tiles."
                        )
                        context.hook_manager.on_inference_tile(
                            context,
                            self.name,
                            tile_counter,
                            total_tiles,
                        )
        count_accum[count_accum == 0] = 1
        prob_accum /= count_accum
        pred_full = prob_accum.argmax(axis=0).astype(np.uint8)
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)
        os.makedirs(os.path.dirname(output_tif) or ".", exist_ok=True)
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(pred_full, 1)
        context.logger.info(f"Saved prediction to {output_tif}")
        metrics = {"tiles_total": float(total_tiles)}
        artifacts = {"output_tif": output_tif, "checkpoint": checkpoint}
        return PhaseOutcome(metrics=metrics, artifacts=artifacts)


def save_epoch_plot(
    output_path: str,
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    cmap: str,
) -> None:
    """Save a 1x3 epoch plot of RGB, ground truth, and prediction.

    Args:
        output_path (str): PNG output path.
        rgb (np.ndarray): RGB image (H, W, 3).
        gt_mask (np.ndarray): Ground-truth mask (H, W).
        pred_mask (np.ndarray): Prediction mask (H, W).
        cmap (str): Matplotlib colormap name.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    axes[1].imshow(gt_mask, cmap=cmap)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    axes[2].imshow(pred_mask, cmap=cmap)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
