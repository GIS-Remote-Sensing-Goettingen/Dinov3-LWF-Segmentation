"""Module-specific XAI metrics and artifact helpers for validation epochs."""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..inference_utils import overlay_heatmap


def _tensor_to_numpy(value: Any) -> np.ndarray | None:
    """Convert an optional value to a numpy array.

    Args:
        value (Any): Tensor/array-like value.

    Returns:
        np.ndarray | None: Converted numpy array, or ``None`` when conversion is
        not possible.
    """

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return None


def _squeeze_to_hw(value: np.ndarray | None) -> np.ndarray | None:
    """Reduce a map to ``(H, W)`` when possible.

    Args:
        value (np.ndarray | None): Optional array.

    Returns:
        np.ndarray | None: Squeezed 2D array, or ``None`` when incompatible.
    """

    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    return arr


def _boundary_band(mask_bool: np.ndarray, radius: int) -> np.ndarray:
    """Compute a boundary band from a binary mask.

    Args:
        mask_bool (np.ndarray): Boolean foreground mask.
        radius (int): Boundary radius in pixels.

    Returns:
        np.ndarray: Boolean boundary-band mask.
    """

    radius = max(1, int(radius))
    mask_t = torch.from_numpy(mask_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(mask_t, kernel_size=kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-mask_t, kernel_size=kernel, stride=1, padding=radius)
    band = (dilated - eroded).squeeze(0).squeeze(0).cpu().numpy() > 0.0
    return band


def _build_region_masks(
    gt_mask: np.ndarray,
    class_index: int,
    boundary_band_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create foreground/boundary/interior/background masks.

    Args:
        gt_mask (np.ndarray): Ground-truth segmentation mask.
        class_index (int): Foreground class index.
        boundary_band_px (int): Boundary radius in pixels.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Foreground,
        boundary, interior, and background masks.
    """

    fg = np.asarray(gt_mask == int(class_index), dtype=bool)
    boundary = _boundary_band(fg, radius=max(1, int(boundary_band_px)))
    interior = np.logical_and(fg, np.logical_not(boundary))
    background = np.logical_not(fg)
    return fg, boundary, interior, background


def _alpha_entropy(alpha_map: np.ndarray, eps: float) -> np.ndarray:
    """Compute entropy over layer weights.

    Args:
        alpha_map (np.ndarray): Layer-weight map with shape ``(L, H, W)``.
        eps (float): Numerical stability epsilon.

    Returns:
        np.ndarray: Entropy map with shape ``(H, W)``.
    """

    probs = np.clip(np.asarray(alpha_map, dtype=np.float32), eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=0)


def _roc_ap(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Compute ROC and AP values without external dependencies.

    Args:
        scores (np.ndarray): Continuous scores.
        labels (np.ndarray): Binary labels.

    Returns:
        dict[str, Any]: Dictionary containing AUROC/AP and ROC arrays. Empty
        when labels are degenerate.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    if scores.size == 0:
        return {}
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    if positives <= 0.0 or negatives <= 0.0:
        return {}

    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted == 1, dtype=np.float64)
    fp = np.cumsum(labels_sorted == 0, dtype=np.float64)
    tpr = tp / positives
    fpr = fp / negatives
    auroc = float(
        np.trapz(
            np.concatenate(([0.0], tpr, [1.0])), np.concatenate(([0.0], fpr, [1.0]))
        )
    )
    precision = tp / np.clip(tp + fp, 1.0, None)
    recall = tpr
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - recall_prev) * precision))
    return {
        "auroc": auroc,
        "ap": ap,
        "fpr": fpr.astype(np.float32),
        "tpr": tpr.astype(np.float32),
    }


def _count_components(mask_bool: np.ndarray) -> int:
    """Count 8-connected components in a binary mask.

    Args:
        mask_bool (np.ndarray): Binary mask.

    Returns:
        int: Number of connected components.
    """

    mask = np.asarray(mask_bool, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = 0
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            components += 1
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width:
                        continue
                    if visited[nr, nc] or not mask[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))
    return int(components)


def _ensure_dir(path: str) -> None:
    """Create a directory if missing.

    Args:
        path (str): Directory path.
    """

    os.makedirs(path, exist_ok=True)


def _save_layermix_panel(
    output_path: str,
    rgb: np.ndarray,
    boundary_mask: np.ndarray,
    alpha_argmax: np.ndarray,
    alpha_entropy: np.ndarray,
) -> None:
    """Save RGB/boundary/argmax/entropy panel.

    Args:
        output_path (str): Output PNG path.
        rgb (np.ndarray): RGB image.
        boundary_mask (np.ndarray): Boundary mask.
        alpha_argmax (np.ndarray): Layer argmax map.
        alpha_entropy (np.ndarray): Entropy map.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.8))
    ax = axes.ravel()
    ax[0].imshow(rgb)
    ax[0].set_title("RGB")
    ax[0].axis("off")
    ax[1].imshow(rgb)
    boundary_overlay = np.ma.masked_where(
        ~boundary_mask, boundary_mask.astype(np.float32)
    )
    ax[1].imshow(boundary_overlay, cmap="Reds", alpha=0.45)
    ax[1].set_title("GT boundary band")
    ax[1].axis("off")
    ax[2].imshow(alpha_argmax, cmap="tab20")
    ax[2].set_title("Layer argmax map")
    ax[2].axis("off")
    ax[3].imshow(overlay_heatmap(rgb, alpha_entropy, cmap="viridis", alpha=0.5))
    ax[3].set_title("Layer entropy")
    ax[3].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_alpha_region_bar(
    output_path: str,
    layer_ids: list[int],
    boundary_means: np.ndarray,
    interior_means: np.ndarray,
    background_means: np.ndarray,
) -> None:
    """Save grouped alpha-by-region bars.

    Args:
        output_path (str): Output PNG path.
        layer_ids (list[int]): Layer identifiers.
        boundary_means (np.ndarray): Boundary alpha means.
        interior_means (np.ndarray): Interior alpha means.
        background_means (np.ndarray): Background alpha means.
    """

    import matplotlib.pyplot as plt

    if boundary_means.size == 0:
        return
    x = np.arange(boundary_means.size)
    width = 0.26
    labels = [
        str(int(layer_ids[idx])) if idx < len(layer_ids) else str(idx)
        for idx in range(boundary_means.size)
    ]
    fig, ax = plt.subplots(figsize=(max(8.0, boundary_means.size * 1.1), 4.8))
    ax.bar(x - width, boundary_means, width, label="boundary", color="tab:red")
    ax.bar(x, interior_means, width, label="interior", color="tab:green")
    ax.bar(x + width, background_means, width, label="background", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean alpha")
    ax.set_xlabel("Layer id")
    ax.set_title("Layer-fusion alpha by region")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_gate_panel(
    output_path: str,
    rgb: np.ndarray,
    gate_map: np.ndarray,
    boundary_mask: np.ndarray,
    roc_info: dict[str, Any],
) -> None:
    """Save gate overlay and boundary ROC panel.

    Args:
        output_path (str): Output PNG path.
        rgb (np.ndarray): RGB image.
        gate_map (np.ndarray): Gate map.
        boundary_mask (np.ndarray): Boundary mask.
        roc_info (dict[str, Any]): ROC payload with curves and AUROC.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].imshow(overlay_heatmap(rgb, gate_map, cmap="magma", alpha=0.5))
    boundary_overlay = np.ma.masked_where(
        ~boundary_mask, boundary_mask.astype(np.float32)
    )
    axes[0].imshow(boundary_overlay, cmap="Greens", alpha=0.25)
    axes[0].set_title("Gate map + GT boundary")
    axes[0].axis("off")

    fpr = np.asarray(roc_info.get("fpr", []), dtype=np.float32)
    tpr = np.asarray(roc_info.get("tpr", []), dtype=np.float32)
    auroc = float(roc_info.get("auroc", float("nan")))
    if fpr.size > 0 and tpr.size > 0:
        axes[1].plot(fpr, tpr, color="tab:purple", linewidth=2.0)
    axes[1].plot([0.0, 1.0], [0.0, 1.0], "--", color="gray", linewidth=1.0)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    title = "Gate boundary ROC"
    if math.isfinite(auroc):
        title += f" (AUROC={auroc:.3f})"
    axes[1].set_title(title)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_boundary_error_panel(
    output_path: str,
    rgb: np.ndarray,
    boundary_mask: np.ndarray,
    error_delta_map: np.ndarray,
    pre_error: float,
    post_error: float,
    delta_mean: float,
) -> None:
    """Save boundary error-reduction panel.

    Args:
        output_path (str): Output PNG path.
        rgb (np.ndarray): RGB image.
        boundary_mask (np.ndarray): Boundary mask.
        error_delta_map (np.ndarray): Pre-vs-post boundary error delta map.
        pre_error (float): Mean pre-gate boundary error.
        post_error (float): Mean post-gate boundary error.
        delta_mean (float): Mean boundary error delta.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].imshow(rgb)
    overlay = np.ma.masked_where(~boundary_mask, boundary_mask.astype(np.float32))
    axes[0].imshow(overlay, cmap="Reds", alpha=0.35)
    axes[0].set_title("Boundary band")
    axes[0].axis("off")

    vmax = max(float(np.max(np.abs(error_delta_map))), 1e-6)
    axes[1].imshow(error_delta_map, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[1].set_title(
        f"Boundary error delta\npre={pre_error:.3f} post={post_error:.3f} mean={delta_mean:.3f}"
    )
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_lora_panel(
    output_path: str,
    rgb: np.ndarray,
    ratio_map: np.ndarray,
    ratio_values: np.ndarray,
    region_means: dict[str, float],
) -> None:
    """Save LoRA ratio map, histogram, and region bars.

    Args:
        output_path (str): Output PNG path.
        rgb (np.ndarray): RGB image.
        ratio_map (np.ndarray): LoRA ratio map.
        ratio_values (np.ndarray): Flattened ratio samples.
        region_means (dict[str, float]): Region-wise mean ratios.
    """

    import matplotlib.pyplot as plt

    finite_ratio_values = np.asarray(ratio_values, dtype=np.float32)
    finite_ratio_values = finite_ratio_values[np.isfinite(finite_ratio_values)]
    vmin: float | None = None
    vmax: float | None = None
    if finite_ratio_values.size > 0:
        vmin = float(np.percentile(finite_ratio_values, 5.0))
        vmax = float(np.percentile(finite_ratio_values, 95.0))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            vmin = None
            vmax = None

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    axes[0].imshow(
        overlay_heatmap(
            rgb,
            ratio_map,
            cmap="viridis",
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
        )
    )
    axes[0].set_title("LoRA ratio map")
    axes[0].axis("off")

    axes[1].hist(
        np.asarray(ratio_values, dtype=np.float32),
        bins=30,
        color="tab:blue",
        alpha=0.8,
    )
    axes[1].set_title("LoRA ratio histogram")
    axes[1].set_xlabel("||Δy|| / (||y0|| + eps)")
    axes[1].set_ylabel("Pixels")
    axes[1].grid(alpha=0.25)

    labels = ["boundary", "interior", "background"]
    values = [float(region_means.get(name, float("nan"))) for name in labels]
    values = [0.0 if not math.isfinite(value) else value for value in values]
    axes[2].bar(labels, values, color=["tab:red", "tab:green", "tab:gray"])
    axes[2].set_title("LoRA ratio by region")
    axes[2].set_ylabel("Mean ratio")
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_topology_panel(
    output_path: str,
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    pred_skel_prob: np.ndarray,
    pred_skel: np.ndarray,
    gt_skel: np.ndarray,
    threshold: float,
    metrics: dict[str, float],
) -> None:
    """Save topology panel with skeleton overlays and metrics.

    Args:
        output_path (str): Output PNG path.
        rgb (np.ndarray): RGB image.
        gt_mask (np.ndarray): Ground-truth mask.
        pred_mask (np.ndarray): Predicted mask.
        pred_skel_prob (np.ndarray): Predicted skeleton probability map.
        pred_skel (np.ndarray): Predicted skeleton mask.
        gt_skel (np.ndarray): Ground-truth skeleton mask.
        threshold (float): Threshold used to binarize the predicted skeleton.
        metrics (dict[str, float]): Topology metrics to show in the panel.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.0))
    ax = axes.ravel()
    ax[0].imshow(rgb)
    ax[0].set_title("RGB")
    ax[0].axis("off")
    ax[1].imshow(gt_mask, cmap="tab20")
    ax[1].set_title("GT mask")
    ax[1].axis("off")
    ax[2].imshow(pred_mask, cmap="tab20")
    ax[2].set_title("Pred mask")
    ax[2].axis("off")
    ax[3].axis("off")
    table_text = "\n".join(
        [
            (
                "Mask-support clDice proxy: "
                f"{float(metrics.get('cldice_proxy', float('nan'))):.3f}"
            ),
            f"Explicit skel F1: {float(metrics.get('skel_f1', float('nan'))):.3f}",
            (
                "Explicit skel recall: "
                f"{float(metrics.get('skel_recall', float('nan'))):.3f}"
            ),
            (
                "Explicit skel precision: "
                f"{float(metrics.get('skel_precision', float('nan'))):.3f}"
            ),
            (
                "Pred skel prob mean: "
                f"{float(metrics.get('skel_prob_mean', float('nan'))):.3f}"
            ),
            (
                "Pred skel prob p95: "
                f"{float(metrics.get('skel_prob_p95', float('nan'))):.3f}"
            ),
            (
                "Pred skel pos rate: "
                f"{float(metrics.get('skel_pred_pos_rate', float('nan'))):.4f}"
            ),
            f"Pred comps: {int(metrics.get('pred_components', 0))}",
            f"GT comps: {int(metrics.get('gt_components', 0))}",
            f"Delta comps: {float(metrics.get('component_delta', float('nan'))):.2f}",
        ]
    )
    ax[3].text(0.02, 0.98, table_text, va="top", ha="left", fontsize=10)
    ax[3].set_title("Topology summary")
    ax[4].imshow(pred_skel_prob, cmap="magma", vmin=0.0, vmax=1.0)
    ax[4].set_title("Pred skel prob")
    ax[4].axis("off")
    ax[5].imshow(pred_skel, cmap="gray")
    ax[5].set_title(f"Pred skel @ {threshold:.2f}")
    ax[5].axis("off")
    ax[6].imshow(gt_skel, cmap="gray")
    ax[6].set_title("GT skeleton")
    ax[6].axis("off")
    overlap = np.zeros((*pred_skel.shape, 3), dtype=np.uint8)
    true_pos = np.logical_and(pred_skel, gt_skel)
    false_pos = np.logical_and(pred_skel, np.logical_not(gt_skel))
    false_neg = np.logical_and(np.logical_not(pred_skel), gt_skel)
    overlap[true_pos] = np.array([255, 255, 255], dtype=np.uint8)
    overlap[false_pos] = np.array([220, 70, 70], dtype=np.uint8)
    overlap[false_neg] = np.array([70, 200, 255], dtype=np.uint8)
    ax[7].imshow(overlap)
    ax[7].set_title("Skel overlap TP/FP/FN")
    ax[7].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_trend_plot(
    output_path: str,
    epochs: list[int],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
) -> None:
    """Save a generic trend plot.

    Args:
        output_path (str): Output PNG path.
        epochs (list[int]): Epoch indices.
        series (dict[str, list[float]]): Series mapping label to values.
        title (str): Plot title.
        ylabel (str): Y-axis label.
    """

    import matplotlib.pyplot as plt

    if not epochs or not series:
        return
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for label, values in series.items():
        if len(values) != len(epochs):
            continue
        ax.plot(epochs, values, marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_module_xai_sample(
    sample_payload: dict[str, Any] | None,
    sample_extras: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build one module-XAI sample payload.

    Args:
        sample_payload (dict[str, Any] | None): Existing sample payload with
            RGB/GT/pred.
        sample_extras (dict[str, Any] | None): Extra tensors from head forward
            pass.

    Returns:
        dict[str, Any] | None: Module-XAI payload or ``None`` when mandatory
        sample fields are missing.
    """

    if not isinstance(sample_payload, dict):
        return None
    extras = sample_extras if isinstance(sample_extras, dict) else {}
    rgb = sample_payload.get("rgb")
    gt_mask = sample_payload.get("gt_mask")
    pred_mask = sample_payload.get("pred_mask")
    if rgb is None or gt_mask is None or pred_mask is None:
        return None
    output: dict[str, Any] = {
        "rgb": np.asarray(rgb),
        "gt_mask": np.asarray(gt_mask),
        "pred_mask": np.asarray(pred_mask),
    }
    for key in (
        "layer_mix_maps",
        "gate_map",
        "pre_gate_logits",
        "lora_base_norm_map",
        "lora_delta_norm_map",
        "skeleton_logits",
    ):
        output[key] = _tensor_to_numpy(extras.get(key))
    return output
