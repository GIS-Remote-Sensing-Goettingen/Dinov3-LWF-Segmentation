"""Epoch-level module XAI aggregation helpers."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch

from utils.losses import soft_skeletonize

from ..inference_utils import upsample_map
from .module_xai import (
    _alpha_entropy,
    _build_region_masks,
    _count_components,
    _ensure_dir,
    _roc_ap,
    _save_alpha_region_bar,
    _save_boundary_error_panel,
    _save_gate_panel,
    _save_layermix_panel,
    _save_lora_panel,
    _save_topology_panel,
    _save_trend_plot,
    _squeeze_to_hw,
    _tensor_to_numpy,
)


def _module_dirs(plot_xai_dir: str) -> dict[str, str]:
    """Return and create module-XAI artifact directories.

    Args:
        plot_xai_dir (str): Base XAI artifact directory.

    Returns:
        dict[str, str]: Named module artifact directories.
    """

    module_root = os.path.join(plot_xai_dir, "module")
    paths = {
        "module_root": module_root,
        "layermix": os.path.join(module_root, "layermix"),
        "gate": os.path.join(module_root, "gate"),
        "lora": os.path.join(module_root, "lora"),
        "topology": os.path.join(module_root, "topology"),
        "trends": os.path.join(module_root, "trends"),
    }
    for path in paths.values():
        _ensure_dir(path)
    return paths


def _compute_layermix_metrics(
    *,
    epoch: int,
    sample_idx: int,
    sample: dict[str, Any],
    gt_mask: np.ndarray,
    boundary_mask: np.ndarray,
    interior_mask: np.ndarray,
    background_mask: np.ndarray,
    entropy_eps: float,
    save_maps_this_epoch: bool,
    layermix_dir: str,
) -> dict[str, Any]:
    """Compute layer-fusion metrics for one sample.

    Args:
        epoch (int): Current epoch index.
        sample_idx (int): 1-based sample index.
        sample (dict[str, Any]): Module sample payload.
        gt_mask (np.ndarray): Ground-truth mask for the sample.
        boundary_mask (np.ndarray): Boundary region mask.
        interior_mask (np.ndarray): Interior region mask.
        background_mask (np.ndarray): Background region mask.
        entropy_eps (float): Entropy epsilon.
        save_maps_this_epoch (bool): Whether to emit qualitative figures.
        layermix_dir (str): Output directory for layer-mix figures.

    Returns:
        dict[str, Any]: Layer-mix metrics and region vectors.
    """

    alpha_arr = _tensor_to_numpy(sample.get("layer_mix_maps"))
    if alpha_arr is None:
        return {}
    alpha_np = np.asarray(alpha_arr, dtype=np.float32)
    if alpha_np.ndim == 4:
        alpha_np = alpha_np[0]
    if alpha_np.ndim != 3 or alpha_np.shape[0] <= 0:
        return {}
    alpha_layers: list[np.ndarray] = []
    for layer_idx in range(int(alpha_np.shape[0])):
        alpha_layers.append(
            upsample_map(alpha_np[layer_idx], gt_mask.shape[0], gt_mask.shape[1])
        )
    alpha_up = np.stack(alpha_layers, axis=0).astype(np.float32)
    alpha_up = alpha_up / np.clip(np.sum(alpha_up, axis=0, keepdims=True), 1e-8, None)
    alpha_entropy = _alpha_entropy(alpha_up, eps=max(1e-12, entropy_eps))
    alpha_argmax = np.argmax(alpha_up, axis=0).astype(np.int32)
    boundary_vals = np.asarray(
        [
            (
                float(np.mean(alpha_up[idx][boundary_mask]))
                if np.any(boundary_mask)
                else float("nan")
            )
            for idx in range(alpha_up.shape[0])
        ],
        dtype=np.float32,
    )
    interior_vals = np.asarray(
        [
            (
                float(np.mean(alpha_up[idx][interior_mask]))
                if np.any(interior_mask)
                else float("nan")
            )
            for idx in range(alpha_up.shape[0])
        ],
        dtype=np.float32,
    )
    background_vals = np.asarray(
        [
            (
                float(np.mean(alpha_up[idx][background_mask]))
                if np.any(background_mask)
                else float("nan")
            )
            for idx in range(alpha_up.shape[0])
        ],
        dtype=np.float32,
    )
    shift = np.nanmean(boundary_vals - interior_vals)
    if save_maps_this_epoch:
        _save_layermix_panel(
            output_path=os.path.join(
                layermix_dir,
                f"epoch_{epoch:04d}_sample_{sample_idx:02d}_panel.png",
            ),
            rgb=np.asarray(sample["rgb"]),
            boundary_mask=boundary_mask,
            alpha_argmax=alpha_argmax,
            alpha_entropy=alpha_entropy,
        )
    return {
        "entropy_mean": float(np.mean(alpha_entropy)),
        "shift": float(shift) if math.isfinite(float(shift)) else None,
        "boundary_vals": boundary_vals,
        "interior_vals": interior_vals,
        "background_vals": background_vals,
    }


def _compute_gate_metrics(
    *,
    epoch: int,
    sample_idx: int,
    sample: dict[str, Any],
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    boundary_mask: np.ndarray,
    save_maps_this_epoch: bool,
    gate_dir: str,
) -> dict[str, Any]:
    """Compute gate and boundary-delta metrics for one sample.

    Args:
        epoch (int): Current epoch index.
        sample_idx (int): 1-based sample index.
        sample (dict[str, Any]): Module sample payload.
        gt_mask (np.ndarray): Ground-truth mask for the sample.
        pred_mask (np.ndarray): Prediction mask for the sample.
        boundary_mask (np.ndarray): Boundary region mask.
        save_maps_this_epoch (bool): Whether to emit qualitative figures.
        gate_dir (str): Output directory for gate figures.

    Returns:
        dict[str, Any]: Gate AUROC/AP and optional boundary delta.
    """

    output: dict[str, Any] = {}
    gate_map = _squeeze_to_hw(_tensor_to_numpy(sample.get("gate_map")))
    if gate_map is not None:
        gate_up = upsample_map(gate_map, gt_mask.shape[0], gt_mask.shape[1])
        roc = _roc_ap(
            scores=gate_up.reshape(-1),
            labels=boundary_mask.astype(np.int64).reshape(-1),
        )
        if roc:
            output["auroc"] = float(roc["auroc"])
            output["ap"] = float(roc["ap"])
            if save_maps_this_epoch:
                _save_gate_panel(
                    output_path=os.path.join(
                        gate_dir,
                        f"epoch_{epoch:04d}_sample_{sample_idx:02d}_gate.png",
                    ),
                    rgb=np.asarray(sample["rgb"]),
                    gate_map=gate_up,
                    boundary_mask=boundary_mask,
                    roc_info=roc,
                )
    pre_gate_logits = _tensor_to_numpy(sample.get("pre_gate_logits"))
    if (
        pre_gate_logits is None
        or pre_gate_logits.ndim != 4
        or not np.any(boundary_mask)
    ):
        return output
    pre_pred = np.argmax(np.asarray(pre_gate_logits[0], dtype=np.float32), axis=0)
    pre_pred = np.asarray(pre_pred, dtype=np.int32)
    if tuple(pre_pred.shape) != tuple(gt_mask.shape):
        pre_pred = (
            upsample_map(
                pre_pred.astype(np.float32), gt_mask.shape[0], gt_mask.shape[1]
            )
            .round()
            .astype(np.int32)
        )
    pre_error = np.logical_and(pre_pred != gt_mask, boundary_mask)
    post_error = np.logical_and(pred_mask != gt_mask, boundary_mask)
    delta_map = pre_error.astype(np.float32) - post_error.astype(np.float32)
    delta_value = float(np.mean(delta_map[boundary_mask]))
    output["boundary_delta"] = delta_value
    if save_maps_this_epoch:
        _save_boundary_error_panel(
            output_path=os.path.join(
                gate_dir,
                f"epoch_{epoch:04d}_sample_{sample_idx:02d}_boundary_delta.png",
            ),
            rgb=np.asarray(sample["rgb"]),
            boundary_mask=boundary_mask,
            error_delta_map=delta_map,
            pre_error=float(np.mean(pre_error[boundary_mask])),
            post_error=float(np.mean(post_error[boundary_mask])),
            delta_mean=delta_value,
        )
    return output


def _compute_lora_metrics(
    *,
    epoch: int,
    sample_idx: int,
    sample: dict[str, Any],
    gt_mask: np.ndarray,
    boundary_mask: np.ndarray,
    interior_mask: np.ndarray,
    background_mask: np.ndarray,
    save_maps_this_epoch: bool,
    lora_dir: str,
) -> dict[str, Any]:
    """Compute LoRA update-ratio diagnostics for one sample.

    Args:
        epoch (int): Current epoch index.
        sample_idx (int): 1-based sample index.
        sample (dict[str, Any]): Module sample payload.
        gt_mask (np.ndarray): Ground-truth mask for the sample.
        boundary_mask (np.ndarray): Boundary region mask.
        interior_mask (np.ndarray): Interior region mask.
        background_mask (np.ndarray): Background region mask.
        save_maps_this_epoch (bool): Whether to emit qualitative figures.
        lora_dir (str): Output directory for LoRA figures.

    Returns:
        dict[str, Any]: LoRA ratio aggregates for the sample.
    """

    lora_base = _squeeze_to_hw(_tensor_to_numpy(sample.get("lora_base_norm_map")))
    lora_delta = _squeeze_to_hw(_tensor_to_numpy(sample.get("lora_delta_norm_map")))
    if lora_base is None or lora_delta is None:
        return {}
    base_up = upsample_map(lora_base, gt_mask.shape[0], gt_mask.shape[1])
    delta_up = upsample_map(lora_delta, gt_mask.shape[0], gt_mask.shape[1])
    ratio = delta_up / np.clip(base_up, 1e-8, None)
    ratio_values = ratio[np.isfinite(ratio)]
    if ratio_values.size == 0:
        return {}
    boundary_mean = (
        float(np.mean(ratio[boundary_mask])) if np.any(boundary_mask) else float("nan")
    )
    interior_mean = (
        float(np.mean(ratio[interior_mask])) if np.any(interior_mask) else float("nan")
    )
    background_mean = (
        float(np.mean(ratio[background_mask]))
        if np.any(background_mask)
        else float("nan")
    )
    if save_maps_this_epoch:
        _save_lora_panel(
            output_path=os.path.join(
                lora_dir,
                f"epoch_{epoch:04d}_sample_{sample_idx:02d}_lora.png",
            ),
            rgb=np.asarray(sample["rgb"]),
            ratio_map=ratio,
            ratio_values=ratio_values,
            region_means={
                "boundary": boundary_mean,
                "interior": interior_mean,
                "background": background_mean,
            },
        )
    return {
        "ratio_mean": float(np.mean(ratio_values)),
        "boundary_mean": boundary_mean,
        "interior_mean": interior_mean,
        "background_mean": background_mean,
    }


def _compute_topology_metrics(
    *,
    epoch: int,
    sample_idx: int,
    sample: dict[str, Any],
    class_index: int,
    fg_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    gate_threshold: float,
    save_maps_this_epoch: bool,
    topology_dir: str,
) -> dict[str, float]:
    """Compute topology/skeleton diagnostics for one sample.

    Args:
        epoch (int): Current epoch index.
        sample_idx (int): 1-based sample index.
        sample (dict[str, Any]): Module sample payload.
        class_index (int): Foreground class index.
        fg_mask (np.ndarray): Foreground mask from GT.
        gt_mask (np.ndarray): Ground-truth mask.
        pred_mask (np.ndarray): Prediction mask.
        gate_threshold (float): Threshold used for predicted skeleton binarization.
        save_maps_this_epoch (bool): Whether to emit qualitative figures.
        topology_dir (str): Output directory for topology figures.

    Returns:
        dict[str, float]: Topology scalar metrics.
    """

    skeleton_logits = _squeeze_to_hw(_tensor_to_numpy(sample.get("skeleton_logits")))
    if skeleton_logits is None:
        return {}
    skel_prob = 1.0 / (1.0 + np.exp(-skeleton_logits))
    skel_up = upsample_map(
        skel_prob.astype(np.float32), gt_mask.shape[0], gt_mask.shape[1]
    )
    pred_skel = skel_up >= float(gate_threshold)
    gt_tensor = torch.from_numpy(fg_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    gt_skel = (
        soft_skeletonize(gt_tensor, iters=10)
        .squeeze(0)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        > 0.2
    )
    pred_fg = pred_mask == int(class_index)
    tprec = float(np.sum(np.logical_and(pred_skel, fg_mask))) / max(
        float(np.sum(pred_skel)), 1.0
    )
    tsens = float(np.sum(np.logical_and(gt_skel, pred_fg))) / max(
        float(np.sum(gt_skel)), 1.0
    )
    cldice_proxy = float((2.0 * tprec * tsens) / max(tprec + tsens, 1e-8))
    skel_true_pos = float(np.sum(np.logical_and(pred_skel, gt_skel)))
    skel_precision = skel_true_pos / max(float(np.sum(pred_skel)), 1.0)
    skel_recall = skel_true_pos / max(float(np.sum(gt_skel)), 1.0)
    skel_f1 = float(
        (2.0 * skel_precision * skel_recall) / max(skel_precision + skel_recall, 1e-8)
    )
    finite_prob = skel_up[np.isfinite(skel_up)]
    skel_prob_mean = (
        float(np.mean(finite_prob)) if finite_prob.size > 0 else float("nan")
    )
    skel_prob_p95 = (
        float(np.percentile(finite_prob, 95.0))
        if finite_prob.size > 0
        else float("nan")
    )
    skel_pred_pos_rate = float(np.mean(pred_skel.astype(np.float32)))
    pred_components = float(_count_components(pred_skel))
    gt_components = float(_count_components(gt_skel))
    component_delta = pred_components - gt_components
    if save_maps_this_epoch:
        _save_topology_panel(
            output_path=os.path.join(
                topology_dir,
                f"epoch_{epoch:04d}_sample_{sample_idx:02d}_topology.png",
            ),
            rgb=np.asarray(sample["rgb"]),
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            pred_skel_prob=skel_up,
            pred_skel=pred_skel,
            gt_skel=gt_skel,
            threshold=float(gate_threshold),
            metrics={
                "cldice_proxy": cldice_proxy,
                "skel_precision": skel_precision,
                "skel_recall": skel_recall,
                "skel_f1": skel_f1,
                "skel_prob_mean": skel_prob_mean,
                "skel_prob_p95": skel_prob_p95,
                "skel_pred_pos_rate": skel_pred_pos_rate,
                "pred_components": pred_components,
                "gt_components": gt_components,
                "component_delta": component_delta,
            },
        )
    return {
        "cldice_proxy": cldice_proxy,
        "skel_precision": skel_precision,
        "skel_recall": skel_recall,
        "skel_f1": skel_f1,
        "skel_prob_mean": skel_prob_mean,
        "skel_prob_p95": skel_prob_p95,
        "skel_pred_pos_rate": skel_pred_pos_rate,
        "pred_components": pred_components,
        "gt_components": gt_components,
        "component_delta": component_delta,
    }


def _collect_sample_metrics(
    *,
    epoch: int,
    selected_samples: list[dict[str, Any]],
    class_index: int,
    gate_threshold: float,
    entropy_eps: float,
    enable_lora: bool,
    enable_topology: bool,
    save_maps_this_epoch: bool,
    dirs: dict[str, str],
) -> dict[str, list[Any]]:
    """Collect sample-wise module-XAI metrics.

    Args:
        epoch (int): Current epoch index.
        selected_samples (list[dict[str, Any]]): Trimmed sample list for this epoch.
        class_index (int): Foreground class index.
        gate_threshold (float): Threshold for gate/skeleton maps.
        entropy_eps (float): Entropy epsilon.
        enable_lora (bool): Whether LoRA diagnostics are enabled.
        enable_topology (bool): Whether topology diagnostics are enabled.
        save_maps_this_epoch (bool): Whether to save qualitative maps.
        dirs (dict[str, str]): Module-XAI output directories.

    Returns:
        dict[str, list[Any]]: Collected metric vectors.
    """

    collected: dict[str, list[Any]] = {
        "layermix_entropy_values": [],
        "layermix_shift_values": [],
        "gate_aurocs": [],
        "gate_aps": [],
        "boundary_delta_values": [],
        "lora_ratio_values": [],
        "lora_region_boundary": [],
        "lora_region_interior": [],
        "lora_region_background": [],
        "topology_cldice_values": [],
        "topology_skel_precision_values": [],
        "topology_skel_recall_values": [],
        "topology_skel_f1_values": [],
        "topology_skel_prob_mean_values": [],
        "topology_skel_prob_p95_values": [],
        "topology_skel_pred_pos_rate_values": [],
        "topology_pred_components": [],
        "topology_gt_components": [],
        "topology_delta_components": [],
        "layer_region_boundary_accum": [],
        "layer_region_interior_accum": [],
        "layer_region_background_accum": [],
    }

    for sample_idx, sample in enumerate(selected_samples, start=1):
        gt_mask = np.asarray(sample["gt_mask"])
        pred_mask = np.asarray(sample["pred_mask"])
        fg_mask, boundary_mask, interior_mask, background_mask = _build_region_masks(
            gt_mask=gt_mask,
            class_index=class_index,
            boundary_band_px=3,
        )
        layermix = _compute_layermix_metrics(
            epoch=epoch,
            sample_idx=sample_idx,
            sample=sample,
            gt_mask=gt_mask,
            boundary_mask=boundary_mask,
            interior_mask=interior_mask,
            background_mask=background_mask,
            entropy_eps=entropy_eps,
            save_maps_this_epoch=save_maps_this_epoch,
            layermix_dir=dirs["layermix"],
        )
        if layermix:
            collected["layermix_entropy_values"].append(layermix["entropy_mean"])
            if layermix.get("shift") is not None:
                collected["layermix_shift_values"].append(layermix["shift"])
            collected["layer_region_boundary_accum"].append(layermix["boundary_vals"])
            collected["layer_region_interior_accum"].append(layermix["interior_vals"])
            collected["layer_region_background_accum"].append(
                layermix["background_vals"]
            )

        gate = _compute_gate_metrics(
            epoch=epoch,
            sample_idx=sample_idx,
            sample=sample,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            boundary_mask=boundary_mask,
            save_maps_this_epoch=save_maps_this_epoch,
            gate_dir=dirs["gate"],
        )
        if "auroc" in gate and math.isfinite(float(gate["auroc"])):
            collected["gate_aurocs"].append(float(gate["auroc"]))
        if "ap" in gate and math.isfinite(float(gate["ap"])):
            collected["gate_aps"].append(float(gate["ap"]))
        if "boundary_delta" in gate and math.isfinite(float(gate["boundary_delta"])):
            collected["boundary_delta_values"].append(float(gate["boundary_delta"]))

        if enable_lora:
            lora = _compute_lora_metrics(
                epoch=epoch,
                sample_idx=sample_idx,
                sample=sample,
                gt_mask=gt_mask,
                boundary_mask=boundary_mask,
                interior_mask=interior_mask,
                background_mask=background_mask,
                save_maps_this_epoch=save_maps_this_epoch,
                lora_dir=dirs["lora"],
            )
            if lora:
                collected["lora_ratio_values"].append(float(lora["ratio_mean"]))
                if math.isfinite(float(lora["boundary_mean"])):
                    collected["lora_region_boundary"].append(
                        float(lora["boundary_mean"])
                    )
                if math.isfinite(float(lora["interior_mean"])):
                    collected["lora_region_interior"].append(
                        float(lora["interior_mean"])
                    )
                if math.isfinite(float(lora["background_mean"])):
                    collected["lora_region_background"].append(
                        float(lora["background_mean"])
                    )

        if enable_topology:
            topology = _compute_topology_metrics(
                epoch=epoch,
                sample_idx=sample_idx,
                sample=sample,
                class_index=class_index,
                fg_mask=fg_mask,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                gate_threshold=gate_threshold,
                save_maps_this_epoch=save_maps_this_epoch,
                topology_dir=dirs["topology"],
            )
            if topology:
                collected["topology_cldice_values"].append(topology["cldice_proxy"])
                collected["topology_skel_precision_values"].append(
                    topology["skel_precision"]
                )
                collected["topology_skel_recall_values"].append(topology["skel_recall"])
                collected["topology_skel_f1_values"].append(topology["skel_f1"])
                if math.isfinite(float(topology["skel_prob_mean"])):
                    collected["topology_skel_prob_mean_values"].append(
                        topology["skel_prob_mean"]
                    )
                if math.isfinite(float(topology["skel_prob_p95"])):
                    collected["topology_skel_prob_p95_values"].append(
                        topology["skel_prob_p95"]
                    )
                if math.isfinite(float(topology["skel_pred_pos_rate"])):
                    collected["topology_skel_pred_pos_rate_values"].append(
                        topology["skel_pred_pos_rate"]
                    )
                collected["topology_pred_components"].append(
                    topology["pred_components"]
                )
                collected["topology_gt_components"].append(topology["gt_components"])
                collected["topology_delta_components"].append(
                    topology["component_delta"]
                )

    return collected


def _metrics_from_collected(collected: dict[str, list[Any]]) -> dict[str, float]:
    """Aggregate collected vectors into scalar metrics.

    Args:
        collected (dict[str, list[Any]]): Sample-wise collected vectors.

    Returns:
        dict[str, float]: Scalar metrics for MLflow logging.
    """

    metrics: dict[str, float] = {}
    scalar_specs = {
        "xai_layermix_entropy_mean": "layermix_entropy_values",
        "xai_layermix_boundary_shift_mean": "layermix_shift_values",
        "xai_gate_boundary_auroc": "gate_aurocs",
        "xai_gate_boundary_ap": "gate_aps",
        "xai_boundary_error_reduction_mean": "boundary_delta_values",
        "xai_lora_ratio_mean": "lora_ratio_values",
        "xai_lora_ratio_boundary_mean": "lora_region_boundary",
        "xai_lora_ratio_interior_mean": "lora_region_interior",
        "xai_lora_ratio_background_mean": "lora_region_background",
        "xai_topology_cldice_proxy": "topology_cldice_values",
        "xai_topology_skel_precision": "topology_skel_precision_values",
        "xai_topology_skel_recall": "topology_skel_recall_values",
        "xai_topology_skel_f1": "topology_skel_f1_values",
        "xai_topology_skel_prob_mean": "topology_skel_prob_mean_values",
        "xai_topology_skel_prob_p95": "topology_skel_prob_p95_values",
        "xai_topology_skel_pred_pos_rate": "topology_skel_pred_pos_rate_values",
        "xai_topology_skel_components_pred": "topology_pred_components",
        "xai_topology_skel_components_gt": "topology_gt_components",
        "xai_topology_skel_component_delta": "topology_delta_components",
    }
    for metric_name, key in scalar_specs.items():
        values = collected.get(key, [])
        if values:
            metrics[metric_name] = float(np.mean(values))
    return metrics


def _append_layer_region_metrics(
    *,
    epoch: int,
    layer_ids: list[int],
    collected: dict[str, list[Any]],
    metrics: dict[str, float],
    save_maps_this_epoch: bool,
    layermix_dir: str,
) -> None:
    """Append layer-by-region alpha means to metrics.

    Args:
        epoch (int): Current epoch index.
        layer_ids (list[int]): Configured DINO layer ids.
        collected (dict[str, list[Any]]): Sample-wise vectors.
        metrics (dict[str, float]): Mutable scalar metrics dictionary.
        save_maps_this_epoch (bool): Whether to save grouped region bar figure.
        layermix_dir (str): Layer-mix output directory.
    """

    region_boundary = collected.get("layer_region_boundary_accum", [])
    if not region_boundary:
        return
    boundary_mean = np.nanmean(np.stack(region_boundary, axis=0), axis=0)
    interior_mean = np.nanmean(
        np.stack(collected["layer_region_interior_accum"], axis=0), axis=0
    )
    background_mean = np.nanmean(
        np.stack(collected["layer_region_background_accum"], axis=0), axis=0
    )
    if save_maps_this_epoch:
        _save_alpha_region_bar(
            output_path=os.path.join(
                layermix_dir, f"epoch_{epoch:04d}_alpha_region_bar.png"
            ),
            layer_ids=list(layer_ids),
            boundary_means=boundary_mean,
            interior_means=interior_mean,
            background_means=background_mean,
        )
    for idx in range(int(boundary_mean.size)):
        layer_id = int(layer_ids[idx]) if idx < len(layer_ids) else int(idx)
        if math.isfinite(float(boundary_mean[idx])):
            metrics[f"xai_layermix_layer_{layer_id}_boundary_mean"] = float(
                boundary_mean[idx]
            )
        if math.isfinite(float(interior_mean[idx])):
            metrics[f"xai_layermix_layer_{layer_id}_interior_mean"] = float(
                interior_mean[idx]
            )
        if math.isfinite(float(background_mean[idx])):
            metrics[f"xai_layermix_layer_{layer_id}_background_mean"] = float(
                background_mean[idx]
            )


def _update_trend_history_and_artifacts(
    *,
    epoch: int,
    metrics: dict[str, float],
    history: dict[str, Any],
    trends_dir: str,
) -> None:
    """Update trend history and write trend plots.

    Args:
        epoch (int): Current epoch index.
        metrics (dict[str, float]): Scalar metrics for this epoch.
        history (dict[str, Any]): Mutable trend history.
        trends_dir (str): Directory for trend figures.
    """

    epochs_hist = history.setdefault("epochs", [])
    epochs_hist.append(int(epoch))
    for key, value in metrics.items():
        history.setdefault(key, []).append(float(value))
    trend_specs = {
        "module_layermix_trends.png": (
            {
                "entropy": history.get("xai_layermix_entropy_mean", []),
                "boundary_shift": history.get("xai_layermix_boundary_shift_mean", []),
            },
            "Layer-fusion module trends",
            "Value",
        ),
        "module_gate_trends.png": (
            {
                "gate_auroc": history.get("xai_gate_boundary_auroc", []),
                "gate_ap": history.get("xai_gate_boundary_ap", []),
            },
            "Boundary gate trends",
            "Score",
        ),
        "module_lora_trends.png": (
            {
                "rho_boundary": history.get("xai_lora_ratio_boundary_mean", []),
                "rho_interior": history.get("xai_lora_ratio_interior_mean", []),
            },
            "LoRA ratio trends",
            "Ratio",
        ),
        "module_topology_trends.png": (
            {
                "mask_support_proxy": history.get("xai_topology_cldice_proxy", []),
                "skeleton_f1": history.get("xai_topology_skel_f1", []),
            },
            "Topology trends",
            "Score",
        ),
        "module_topology_branch_health.png": (
            {
                "pred_pos_rate": history.get("xai_topology_skel_pred_pos_rate", []),
                "prob_mean": history.get("xai_topology_skel_prob_mean", []),
                "prob_p95": history.get("xai_topology_skel_prob_p95", []),
            },
            "Skeleton branch health",
            "Value",
        ),
        "module_topology_components.png": (
            {
                "pred_components": history.get("xai_topology_skel_components_pred", []),
                "gt_components": history.get("xai_topology_skel_components_gt", []),
                "component_delta": history.get("xai_topology_skel_component_delta", []),
            },
            "Skeleton component trends",
            "Count",
        ),
    }
    for filename, (series, title, ylabel) in trend_specs.items():
        non_empty = {
            label: list(values)
            for label, values in series.items()
            if isinstance(values, list)
            and len(values) == len(epochs_hist)
            and len(values) > 0
        }
        if non_empty:
            _save_trend_plot(
                output_path=os.path.join(trends_dir, filename),
                epochs=list(epochs_hist),
                series=non_empty,
                title=title,
                ylabel=ylabel,
            )


def update_module_xai_epoch(
    epoch: int,
    module_cfg: dict[str, Any] | None,
    layer_ids: list[int],
    class_index: int,
    plot_xai_dir: str,
    samples: list[dict[str, Any]],
    history: dict[str, Any],
    logger: Any | None = None,
) -> dict[str, float]:
    """Compute module-XAI metrics and write optional artifacts for one epoch.

    Args:
        epoch (int): 1-based epoch index.
        module_cfg (dict[str, Any] | None): ``train.plots.xai.module`` configuration.
        layer_ids (list[int]): DINO layer ids used by the model.
        class_index (int): Foreground class index for module diagnostics.
        plot_xai_dir (str): Base XAI directory for artifacts.
        samples (list[dict[str, Any]]): Collected module-XAI samples.
        history (dict[str, Any]): Mutable trend history storage.
        logger (Any | None): Optional logger.

    Returns:
        dict[str, float]: MLflow-friendly scalar metrics.
    """

    module_cfg = module_cfg if isinstance(module_cfg, dict) else {}
    enabled = bool(module_cfg.get("enable", True))
    if not enabled:
        return {}
    strict = bool(module_cfg.get("strict", False))
    max_samples = max(1, int(module_cfg.get("max_samples", 8)))
    every_n = max(1, int(module_cfg.get("every_n_epochs", 5)))
    save_maps_this_epoch = bool(module_cfg.get("save_maps", True)) and (
        int(epoch) % every_n == 0
    )
    gate_threshold = float(module_cfg.get("gate_threshold", 0.5))
    entropy_eps = float(module_cfg.get("entropy_eps", 1e-8))
    enable_lora = bool(module_cfg.get("enable_lora_ratio", True))
    enable_topology = bool(module_cfg.get("enable_topology_panels", True))

    selected_samples = list(samples[:max_samples])
    if not selected_samples:
        if strict:
            raise RuntimeError(
                "Module XAI is enabled but no compatible validation samples were collected."
            )
        return {}

    dirs = _module_dirs(plot_xai_dir)
    collected = _collect_sample_metrics(
        epoch=epoch,
        selected_samples=selected_samples,
        class_index=class_index,
        gate_threshold=gate_threshold,
        entropy_eps=entropy_eps,
        enable_lora=enable_lora,
        enable_topology=enable_topology,
        save_maps_this_epoch=save_maps_this_epoch,
        dirs=dirs,
    )
    metrics = _metrics_from_collected(collected)
    _append_layer_region_metrics(
        epoch=epoch,
        layer_ids=layer_ids,
        collected=collected,
        metrics=metrics,
        save_maps_this_epoch=save_maps_this_epoch,
        layermix_dir=dirs["layermix"],
    )
    if not metrics:
        if strict:
            raise RuntimeError(
                "Module XAI is enabled but no module-specific metrics were produced."
            )
        return {}

    _update_trend_history_and_artifacts(
        epoch=epoch,
        metrics=metrics,
        history=history,
        trends_dir=dirs["trends"],
    )
    if logger:
        logger.info(
            "Module XAI epoch %s metrics: %s"
            % (
                int(epoch),
                ", ".join(
                    f"{key}={value:.4f}" for key, value in sorted(metrics.items())
                ),
            )
        )
    return metrics
