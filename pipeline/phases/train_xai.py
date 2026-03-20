"""Epoch-level XAI helpers for training validation plots."""

from __future__ import annotations

import math
import os
import random
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from ..context import RunContext
from ..inference_utils import (
    compute_attention_maps,
    compute_branch_importance,
    compute_dino_layer_importance,
    compute_feature_pca_rgb,
    compute_gradcam_with_topk_channels,
    upsample_map,
    upsample_rgb_map,
)
from ..plotting import (
    _aggregate_channel_importance_samples,
    _save_branch_importance_trend_plot,
    _save_channel_importance_bar_plot,
    _save_channel_importance_heatmap,
    _save_channel_importance_trend_plot,
    _save_dino_layer_importance_trend_plot,
    _select_stable_channel_ids,
    _write_channel_importance_json,
    compute_tile_iou_f1,
    save_epoch_plot,
    save_epoch_xai_plot,
    summarize_branch_importance_epoch,
    summarize_dino_layer_importance_epoch,
)
from ..train_utils import (
    align_logits_to_labels,
    extract_multiscale_features_batch,
    move_features_to_device,
)
from ..xai.module_xai import build_module_xai_sample
from ..xai.module_xai_epoch import update_module_xai_epoch
from .train_batches import ensure_backbone_processor


def _build_plot_rgb(sample_img: torch.Tensor, sample_gt: torch.Tensor) -> np.ndarray:
    """Build a uint8 RGB preview aligned to the GT/pred label grid.

    Args:
        sample_img (torch.Tensor): Image tensor shaped ``(1, C, H, W)``.
        sample_gt (torch.Tensor): Label tensor whose spatial shape defines the
            preview grid.

    Returns:
        np.ndarray: RGB preview shaped ``(H_label, W_label, 3)``.
    """

    plot_img = sample_img.detach().float().cpu()
    target_size = (int(sample_gt.shape[-2]), int(sample_gt.shape[-1]))
    if tuple(plot_img.shape[-2:]) != target_size:
        plot_img = F.interpolate(
            plot_img,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    rgb = plot_img.permute(0, 2, 3, 1).numpy()[0]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def collect_epoch_xai_metrics(
    *,
    context: RunContext,
    epoch: int,
    eval_model: torch.nn.Module,
    val_loader: Any,
    cache_features: bool,
    model_cfg: dict[str, Any],
    loss_ignore_index: int | None,
    plot_cfg: Any,
    plot_metrics_dir: str,
    plot_xai_dir: str,
    plot_metrics_paper_dir: str,
    plot_xai_paper_dir: str,
    plot_xai_cam_layer: int | None,
    plot_xai_pca_layer: int | None,
    model_layer_ids: list[int],
    backbone: Any,
    processor: Any,
    device: torch.device,
    ps: int,
    autocast: Any,
    histories: dict[str, Any],
) -> tuple[dict[str, float], Any, Any]:
    """Generate epoch validation plots and XAI summaries.

    Args:
        context (RunContext): Active run context.
        epoch (int): Zero-based epoch index.
        eval_model (torch.nn.Module): Evaluation model for validation samples.
        val_loader (Any): Validation dataloader.
        cache_features (bool): Whether validation features are cached.
        model_cfg (dict[str, Any]): Parsed model configuration.
        loss_ignore_index (int | None): Ignore index for GT metric masking.
        plot_cfg (Any): Parsed train plot configuration.
        plot_metrics_dir (str): Directory for validation metric panels.
        plot_xai_dir (str): Directory for XAI artifacts.
        plot_metrics_paper_dir (str): Directory for curated paper metric plots.
        plot_xai_paper_dir (str): Directory for curated paper XAI plots.
        plot_xai_cam_layer (int | None): Layer index used for CAM extraction.
        plot_xai_pca_layer (int | None): Layer index used for PCA visualization.
        model_layer_ids (list[int]): Requested DINO layer ids.
        backbone (Any): Cached backbone model.
        processor (Any): Cached processor.
        device (torch.device): Target torch device.
        ps (int): Backbone patch size.
        autocast (Any): AMP autocast context.
        histories (dict[str, Any]): Mutable history buffers across epochs.

    Returns:
        tuple[dict[str, float], Any, Any]: XAI metrics plus updated backbone and
        processor handles.
    """

    xai_epoch_metrics: dict[str, float] = {}
    if not (plot_cfg.enabled and val_loader is not None and context.dist_ctx.is_main):
        return xai_epoch_metrics, backbone, processor

    os.makedirs(plot_metrics_dir, exist_ok=True)
    if plot_cfg.xai_enable:
        os.makedirs(plot_xai_dir, exist_ok=True)
    if plot_cfg.paper_enable:
        os.makedirs(plot_metrics_paper_dir, exist_ok=True)
        if plot_cfg.xai_enable:
            os.makedirs(plot_xai_paper_dir, exist_ok=True)
    val_count = len(val_loader.dataset)
    if val_count <= 0:
        context.logger.error(
            "Epoch validation plotting skipped: empty validation dataset."
        )
        return xai_epoch_metrics, backbone, processor

    desired_pairs = min(plot_cfg.pairs, val_count)
    seed_value = int(context.config.get("resources", {}).get("seed", 1337))
    rng = random.Random(seed_value + plot_cfg.seed_offset + epoch)
    selected_plot_indices = set(rng.sample(range(val_count), k=desired_pairs))
    context.logger.info(
        f"Epoch {epoch + 1} validation plot indices: {sorted(selected_plot_indices)}"
    )
    channel_tracking_target = 0
    selected_channel_indices: set[int] = set()
    if plot_cfg.xai_enable and plot_cfg.xai_channel_tracking_enable:
        channel_tracking_target = min(
            plot_cfg.xai_channel_tracking_max_samples, val_count
        )
        selected_channel_indices = set(
            rng.sample(range(val_count), k=channel_tracking_target)
        )
        context.logger.info(
            "Epoch %s channel-importance sample count: %s"
            % (epoch + 1, channel_tracking_target)
        )

    (
        sample_plots,
        branch_img_importances,
        branch_dino_importances,
        dino_layer_importance_samples,
        channel_importance_samples,
        module_xai_samples,
        backbone,
        processor,
    ) = _collect_epoch_xai_samples(
        context=context,
        epoch=epoch,
        eval_model=eval_model,
        val_loader=val_loader,
        cache_features=cache_features,
        model_cfg=model_cfg,
        plot_cfg=plot_cfg,
        selected_plot_indices=selected_plot_indices,
        selected_channel_indices=selected_channel_indices,
        channel_tracking_target=channel_tracking_target,
        desired_pairs=desired_pairs,
        plot_xai_cam_layer=plot_xai_cam_layer,
        plot_xai_pca_layer=plot_xai_pca_layer,
        loss_ignore_index=loss_ignore_index,
        backbone=backbone,
        processor=processor,
        device=device,
        ps=ps,
        autocast=autocast,
    )

    if not sample_plots:
        context.logger.error("Epoch validation plotting skipped: no samples collected.")
        return xai_epoch_metrics, backbone, processor

    out_path = os.path.join(plot_metrics_dir, f"epoch_{epoch + 1:04d}.png")
    save_epoch_plot(
        out_path,
        sample_plots,
        plot_cfg.cmap,
        class_index=plot_cfg.metric_class_index,
    )
    if plot_cfg.paper_enable:
        paper_metric_path = os.path.join(
            plot_metrics_paper_dir, f"epoch_{epoch + 1:04d}.png"
        )
        save_epoch_plot(
            paper_metric_path,
            sample_plots[: plot_cfg.paper_pairs],
            plot_cfg.cmap,
            class_index=plot_cfg.metric_class_index,
            paper_style=True,
        )
    if not plot_cfg.xai_enable:
        return xai_epoch_metrics, backbone, processor

    xai_out_path = os.path.join(plot_xai_dir, f"epoch_{epoch + 1:04d}_xai.png")
    save_epoch_xai_plot(
        xai_out_path,
        sample_plots,
        cmap=plot_cfg.cmap,
        topk_channels=plot_cfg.xai_topk_channels,
        render_rollout=plot_cfg.xai_render_rollout,
        render_pca=plot_cfg.xai_pca_enable,
        class_index=plot_cfg.xai_class_index,
    )
    if plot_cfg.paper_enable:
        paper_xai_path = os.path.join(
            plot_xai_paper_dir, f"epoch_{epoch + 1:04d}_xai.png"
        )
        save_epoch_xai_plot(
            paper_xai_path,
            sample_plots[: plot_cfg.paper_pairs],
            cmap=plot_cfg.cmap,
            topk_channels=plot_cfg.paper_xai_topk_channels,
            render_rollout=plot_cfg.paper_render_rollout,
            render_pca=plot_cfg.paper_render_pca,
            class_index=plot_cfg.xai_class_index,
            paper_style=True,
        )
    branch_summary = summarize_branch_importance_epoch(
        branch_img_importances, branch_dino_importances
    )
    if branch_summary:
        img_mean = float(branch_summary["xai_img_importance_mean"])
        dino_mean = float(branch_summary["xai_dino_importance_mean"])
        histories["branch_importance_history"].append(
            {
                "epoch": float(epoch + 1),
                "img_importance_mean": img_mean,
                "dino_importance_mean": dino_mean,
            }
        )
        branch_trend_path = os.path.join(plot_xai_dir, "branch_importance_trends.png")
        _save_branch_importance_trend_plot(
            branch_trend_path, histories["branch_importance_history"]
        )
        if plot_cfg.paper_enable:
            _save_branch_importance_trend_plot(
                os.path.join(plot_xai_paper_dir, "branch_importance_trends.png"),
                histories["branch_importance_history"],
                paper_style=True,
            )
        xai_epoch_metrics.update(branch_summary)

    layer_means, layer_metrics = summarize_dino_layer_importance_epoch(
        dino_layer_importance_samples,
        model_layer_ids,
    )
    if layer_means:
        histories["dino_layer_importance_history"].append(
            {"epoch": epoch + 1, "mean_importance": layer_means}
        )
        layer_trend_path = os.path.join(
            plot_xai_dir, "dino_layer_importance_trends.png"
        )
        _save_dino_layer_importance_trend_plot(
            layer_trend_path,
            histories["dino_layer_importance_history"],
            model_layer_ids,
        )
        if plot_cfg.paper_enable:
            _save_dino_layer_importance_trend_plot(
                os.path.join(plot_xai_paper_dir, "dino_layer_importance_trends.png"),
                histories["dino_layer_importance_history"],
                model_layer_ids,
                paper_style=True,
            )
        xai_epoch_metrics.update(layer_metrics)

    xai_epoch_metrics.update(
        update_module_xai_epoch(
            epoch + 1,
            plot_cfg.xai_module_cfg,
            model_layer_ids,
            plot_cfg.xai_class_index,
            plot_xai_dir,
            module_xai_samples,
            histories["module_xai_history"],
            context.logger,
        )
    )
    xai_epoch_metrics.update(
        _update_channel_importance_artifacts(
            epoch=epoch,
            plot_cfg=plot_cfg,
            plot_xai_dir=plot_xai_dir,
            plot_xai_paper_dir=plot_xai_paper_dir,
            channel_importance_samples=channel_importance_samples,
            channel_importance_history=histories["channel_importance_history"],
        )
    )
    return xai_epoch_metrics, backbone, processor


def _collect_epoch_xai_samples(
    *,
    context: RunContext,
    epoch: int,
    eval_model: torch.nn.Module,
    val_loader: Any,
    cache_features: bool,
    model_cfg: dict[str, Any],
    plot_cfg: Any,
    selected_plot_indices: set[int],
    selected_channel_indices: set[int],
    channel_tracking_target: int,
    desired_pairs: int,
    plot_xai_cam_layer: int | None,
    plot_xai_pca_layer: int | None,
    loss_ignore_index: int | None,
    backbone: Any,
    processor: Any,
    device: torch.device,
    ps: int,
    autocast: Any,
) -> tuple[
    list[dict[str, Any]],
    list[float],
    list[float],
    list[dict[int, float]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Any,
    Any,
]:
    """Collect validation samples used by XAI and metric plotting.

    Args:
        context (RunContext): Active run context.
        epoch (int): Zero-based epoch index.
        eval_model (torch.nn.Module): Evaluation model.
        val_loader (Any): Validation dataloader.
        cache_features (bool): Whether features are cached.
        model_cfg (dict[str, Any]): Parsed model configuration.
        plot_cfg (Any): Parsed train plot configuration.
        selected_plot_indices (set[int]): Global validation sample ids for plots.
        selected_channel_indices (set[int]): Global ids for channel tracking.
        channel_tracking_target (int): Max channel-tracking sample count.
        desired_pairs (int): Max qualitative sample count.
        plot_xai_cam_layer (int | None): CAM layer id.
        plot_xai_pca_layer (int | None): PCA layer id.
        loss_ignore_index (int | None): Ignore index for region metrics.
        backbone (Any): Cached backbone.
        processor (Any): Cached processor.
        device (torch.device): Target torch device.
        ps (int): Backbone patch size.
        autocast (Any): AMP autocast context.

    Returns:
        tuple: Collected sample lists and updated backbone/processor.
    """

    sample_plots: list[dict[str, Any]] = []
    branch_img_importances: list[float] = []
    branch_dino_importances: list[float] = []
    dino_layer_importance_samples: list[dict[int, float]] = []
    channel_importance_samples: list[dict[str, Any]] = []
    module_xai_samples: list[dict[str, Any]] = []
    eval_call = cast(Any, eval_model)
    running_idx = 0
    gradcam_topk = max(
        plot_cfg.xai_topk_channels, plot_cfg.xai_channel_top_k_per_sample
    )
    if plot_cfg.xai_enable:
        backbone, processor = ensure_backbone_processor(
            backbone,
            processor,
            model_cfg["backbone"],
            device,
        )

    for v_img, v_feats, v_y in val_loader:
        batch_size = int(v_img.shape[0])
        wanted_local: list[tuple[int, bool, bool]] = []
        for local_idx in range(batch_size):
            global_idx = running_idx + local_idx
            wants_plot = global_idx in selected_plot_indices
            wants_channel = (
                plot_cfg.xai_enable
                and plot_cfg.xai_channel_tracking_enable
                and channel_tracking_target > 0
                and len(channel_importance_samples) < channel_tracking_target
                and global_idx in selected_channel_indices
            )
            if wants_plot or wants_channel:
                wanted_local.append((local_idx, wants_plot, wants_channel))
        running_idx += batch_size
        if not wanted_local:
            if (
                len(sample_plots) >= desired_pairs
                and len(channel_importance_samples) >= channel_tracking_target
            ):
                break
            continue

        v_img = v_img.to(device)
        v_y = v_y.to(device)
        feats_device: list[torch.Tensor] = []
        if cache_features and v_feats:
            feats_device = move_features_to_device(v_feats, device)

        for local_idx, wants_plot, wants_channel in wanted_local:
            sample_img = v_img[local_idx : local_idx + 1]
            sample_gt = v_y[local_idx : local_idx + 1]
            rgb_input = sample_img.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            rgb_input = np.clip(rgb_input * 255.0, 0, 255).astype(np.uint8)
            rgb = _build_plot_rgb(sample_img, sample_gt)
            if cache_features and feats_device:
                sample_feats = [
                    feat[local_idx : local_idx + 1] for feat in feats_device
                ]
            else:
                backbone, processor = ensure_backbone_processor(
                    backbone,
                    processor,
                    model_cfg["backbone"],
                    device,
                )
                sample_feats = extract_multiscale_features_batch(
                    sample_img,
                    backbone,
                    processor,
                    device,
                    model_cfg["layers"],
                    ps,
                )

            gate_importance: float | None = None
            sample_payload: dict[str, Any] | None = None
            sample_extras: dict[str, Any] = {}
            if wants_plot:
                with torch.no_grad(), autocast:
                    if plot_cfg.xai_enable and hasattr(
                        eval_call, "forward_with_extras"
                    ):
                        extra_out = eval_call.forward_with_extras(
                            sample_img, sample_feats
                        )
                        sample_logits = extra_out["logits"]
                        sample_extras = extra_out
                    elif hasattr(eval_call, "forward_with_aux"):
                        sample_logits, _ = eval_call.forward_with_aux(
                            sample_img, sample_feats
                        )
                    else:
                        sample_logits = eval_call(sample_img, sample_feats)
                    sample_logits = cast(
                        torch.Tensor,
                        align_logits_to_labels(sample_logits, sample_gt),
                    )
                    gate_raw = sample_extras.get("gate_h4_mean")
                    if isinstance(gate_raw, torch.Tensor):
                        gate_value = float(gate_raw.detach().item())
                        if math.isfinite(gate_value):
                            gate_importance = gate_value
                pred_mask = sample_logits.argmax(dim=1).detach().cpu().numpy()[0]
                gt_mask = sample_gt.detach().cpu().numpy()[0]
                tile_iou, tile_f1 = compute_tile_iou_f1(
                    pred_mask,
                    gt_mask,
                    class_index=plot_cfg.metric_class_index,
                    ignore_index=loss_ignore_index,
                )
                sample_payload = {
                    "rgb": rgb,
                    "gt_mask": gt_mask,
                    "pred_mask": pred_mask,
                    "iou": tile_iou,
                    "f1": tile_f1,
                }
                if gate_importance is not None:
                    sample_payload["gate_importance"] = gate_importance
                if (
                    plot_cfg.xai_enable
                    and plot_cfg.xai_branch_importance_enable
                    and len(branch_img_importances)
                    < plot_cfg.xai_branch_importance_max_samples
                ):
                    branch_info = compute_branch_importance(
                        head=eval_model,
                        image=sample_img,
                        features=sample_feats,
                        class_index=plot_cfg.xai_branch_importance_class_index,
                        logger=context.logger,
                    )
                    sample_payload.update(branch_info)
                    branch_img_importances.append(float(branch_info["img_importance"]))
                    branch_dino_importances.append(
                        float(branch_info["dino_importance"])
                    )
                if (
                    plot_cfg.xai_enable
                    and plot_cfg.xai_branch_importance_enable
                    and len(dino_layer_importance_samples)
                    < plot_cfg.xai_branch_importance_max_samples
                ):
                    layer_info = compute_dino_layer_importance(
                        head=eval_model,
                        image=sample_img,
                        features=sample_feats,
                        layer_ids=model_cfg["layers"],
                        class_index=plot_cfg.xai_branch_importance_class_index,
                        logger=context.logger,
                    )
                    if layer_info:
                        dino_layer_importance_samples.append(layer_info)

            if plot_cfg.xai_enable and (wants_plot or wants_channel):
                rgb_h, rgb_w = int(rgb.shape[0]), int(rgb.shape[1])
                gradcam_result = compute_gradcam_with_topk_channels(
                    image_hw3=rgb_input.astype(np.float32),
                    backbone=backbone,
                    head=eval_model,
                    processor=processor,
                    device=device,
                    layers=model_cfg["layers"],
                    ps=ps,
                    class_index=plot_cfg.xai_class_index,
                    topk_channels=gradcam_topk,
                    cam_layer=plot_xai_cam_layer,
                    logger=context.logger,
                )
                if wants_channel:
                    channel_importance_samples.append(
                        {
                            "top_channels": [
                                int(idx)
                                for idx in gradcam_result["top_indices"][
                                    : plot_cfg.xai_channel_top_k_per_sample
                                ]
                            ],
                            "top_scores": [
                                float(score)
                                for score in gradcam_result["top_scores"][
                                    : plot_cfg.xai_channel_top_k_per_sample
                                ]
                            ],
                        }
                    )
                if wants_plot and sample_payload is not None:
                    pca_rgb_map = None
                    if plot_cfg.xai_pca_enable and sample_feats:
                        pca_feature = sample_feats[-1]
                        if (
                            plot_xai_pca_layer is not None
                            and plot_xai_pca_layer in model_cfg["layers"]
                        ):
                            pca_idx = model_cfg["layers"].index(int(plot_xai_pca_layer))
                            pca_feature = sample_feats[pca_idx]
                        pca_small = compute_feature_pca_rgb(pca_feature)
                        pca_rgb_map = upsample_rgb_map(pca_small, rgb_h, rgb_w)
                    (
                        attn_cls_map,
                        attn_rollout_map,
                        had_attn,
                    ) = compute_attention_maps(
                        rgb_input.astype(np.float32),
                        backbone,
                        processor,
                        device,
                        ps,
                        logger=context.logger,
                    )
                    if not had_attn:
                        context.logger.info(
                            "Epoch %s sample %s attention unavailable; using zero attention maps."
                            % (epoch + 1, len(sample_plots) + 1)
                        )
                    attn_cls_map = upsample_map(attn_cls_map, rgb_h, rgb_w)
                    attn_rollout_map = upsample_map(attn_rollout_map, rgb_h, rgb_w)
                    gradcam_map = upsample_map(
                        np.asarray(gradcam_result["cam_map"], dtype=np.float32),
                        rgb_h,
                        rgb_w,
                    )
                    top_maps = [
                        upsample_map(
                            np.asarray(top_map, dtype=np.float32), rgb_h, rgb_w
                        )
                        for top_map in gradcam_result["top_maps"][
                            : plot_cfg.xai_topk_channels
                        ]
                    ]
                    sample_payload.update(
                        {
                            "attn_cls": attn_cls_map,
                            "attn_rollout": attn_rollout_map,
                            "gradcam": gradcam_map,
                            "top_channels": [
                                int(idx)
                                for idx in gradcam_result["top_indices"][
                                    : plot_cfg.xai_topk_channels
                                ]
                            ],
                            "top_scores": [
                                float(score)
                                for score in gradcam_result["top_scores"][
                                    : plot_cfg.xai_topk_channels
                                ]
                            ],
                            "top_maps": top_maps,
                        }
                    )
                    if pca_rgb_map is not None:
                        sample_payload["pca_rgb"] = pca_rgb_map
                module_sample = build_module_xai_sample(sample_payload, sample_extras)
                if plot_cfg.xai_enable and module_sample is not None:
                    module_xai_samples.append(module_sample)

            if wants_plot and sample_payload is not None:
                sample_plots.append(sample_payload)
            if (
                len(sample_plots) >= desired_pairs
                and len(channel_importance_samples) >= channel_tracking_target
            ):
                break
        if (
            len(sample_plots) >= desired_pairs
            and len(channel_importance_samples) >= channel_tracking_target
        ):
            break

    return (
        sample_plots,
        branch_img_importances,
        branch_dino_importances,
        dino_layer_importance_samples,
        channel_importance_samples,
        module_xai_samples,
        backbone,
        processor,
    )


def _update_channel_importance_artifacts(
    *,
    epoch: int,
    plot_cfg: Any,
    plot_xai_dir: str,
    plot_xai_paper_dir: str,
    channel_importance_samples: list[dict[str, Any]],
    channel_importance_history: list[dict[str, Any]],
) -> dict[str, float]:
    """Write channel-importance artifacts and derive scalar metrics.

    Args:
        epoch (int): Zero-based epoch index.
        plot_cfg (Any): Parsed train plot configuration.
        plot_xai_dir (str): XAI output directory.
        plot_xai_paper_dir (str): Paper-output XAI directory.
        channel_importance_samples (list[dict[str, Any]]): Per-sample channel stats.
        channel_importance_history (list[dict[str, Any]]): Running epoch summaries.

    Returns:
        dict[str, float]: Scalar channel-importance metrics for logging.
    """

    xai_epoch_metrics: dict[str, float] = {}
    if not (plot_cfg.xai_channel_tracking_enable and channel_importance_samples):
        return xai_epoch_metrics

    epoch_channel_summary = _aggregate_channel_importance_samples(
        channel_importance_samples
    )
    if epoch_channel_summary["sample_count"] <= 0:
        return xai_epoch_metrics

    epoch_channel_summary["epoch"] = epoch + 1
    channel_importance_history.append(epoch_channel_summary)
    stable_channels = _select_stable_channel_ids(
        channel_importance_history,
        top_n=plot_cfg.xai_channel_top_n_stable,
        min_presence=plot_cfg.xai_channel_min_presence,
    )
    bar_path = os.path.join(
        plot_xai_dir,
        f"epoch_{epoch + 1:04d}_channel_importance_bar.png",
    )
    _save_channel_importance_bar_plot(
        bar_path,
        epoch=epoch + 1,
        epoch_summary=epoch_channel_summary,
        stable_channels=stable_channels,
    )
    trend_path = os.path.join(plot_xai_dir, "channel_importance_trends.png")
    _save_channel_importance_trend_plot(
        trend_path, channel_importance_history, stable_channels
    )
    heatmap_path = os.path.join(plot_xai_dir, "channel_importance_heatmap.png")
    _save_channel_importance_heatmap(
        heatmap_path, channel_importance_history, stable_channels
    )
    if plot_cfg.paper_enable:
        paper_channels = stable_channels[: plot_cfg.paper_channel_top_n_stable]
        _save_channel_importance_bar_plot(
            os.path.join(
                plot_xai_paper_dir, f"epoch_{epoch + 1:04d}_channel_importance_bar.png"
            ),
            epoch=epoch + 1,
            epoch_summary=epoch_channel_summary,
            stable_channels=paper_channels,
            paper_style=True,
        )
        _save_channel_importance_trend_plot(
            os.path.join(plot_xai_paper_dir, "channel_importance_trends.png"),
            channel_importance_history,
            paper_channels,
            paper_style=True,
        )
        if plot_cfg.paper_include_channel_heatmap:
            _save_channel_importance_heatmap(
                os.path.join(plot_xai_paper_dir, "channel_importance_heatmap.png"),
                channel_importance_history,
                paper_channels,
                paper_style=True,
            )
    if plot_cfg.xai_channel_save_json:
        channel_json_path = os.path.join(
            plot_xai_dir,
            f"epoch_{epoch + 1:04d}_channel_importance.json",
        )
        _write_channel_importance_json(
            channel_json_path,
            epoch_summary=epoch_channel_summary,
            stable_channels=stable_channels,
        )
    top_channels = epoch_channel_summary["top_channels"]
    top1_id = float(top_channels[0][0]) if top_channels else -1.0
    top1_weight = float(top_channels[0][1]) if top_channels else 0.0
    top5_mass = float(sum(float(weight) for _, weight in top_channels[:5]))
    xai_epoch_metrics.update(
        {
            "xai_channel_entropy": float(epoch_channel_summary["entropy"]),
            "xai_top1_channel_id": top1_id,
            "xai_top1_channel_importance": top1_weight,
            "xai_top5_mass": top5_mass,
            "xai_channel_samples": float(epoch_channel_summary["sample_count"]),
            "xai_channel_unique_count": float(
                len(epoch_channel_summary["mean_importance"])
            ),
        }
    )
    return xai_epoch_metrics
