"""Module-specific XAI metrics and artifact helpers for validation epochs."""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from utils.losses import soft_skeletonize

from .inference_utils import overlay_heatmap, upsample_map


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
        np.trapz(np.concatenate(([0.0], tpr, [1.0])), np.concatenate(([0.0], fpr, [1.0])))
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
    boundary_overlay = np.ma.masked_where(~boundary_mask, boundary_mask.astype(np.float32))
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
    boundary_overlay = np.ma.masked_where(~boundary_mask, boundary_mask.astype(np.float32))
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

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    axes[0].imshow(overlay_heatmap(rgb, ratio_map, cmap="plasma", alpha=0.45))
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
    pred_skel: np.ndarray,
    gt_skel: np.ndarray,
    metrics: dict[str, float],
) -> None:
    """Save topology panel with skeleton overlays and metrics.

    Args:
        output_path (str): Output PNG path.
        rgb (np.ndarray): RGB image.
        gt_mask (np.ndarray): Ground-truth mask.
        pred_mask (np.ndarray): Predicted mask.
        pred_skel (np.ndarray): Predicted skeleton mask.
        gt_skel (np.ndarray): Ground-truth skeleton mask.
        metrics (dict[str, float]): Topology metrics to show in the panel.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
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
    ax[3].imshow(pred_skel, cmap="gray")
    ax[3].set_title("Pred skeleton")
    ax[3].axis("off")
    ax[4].imshow(gt_skel, cmap="gray")
    ax[4].set_title("GT skeleton")
    ax[4].axis("off")
    ax[5].axis("off")
    table_text = "\n".join(
        [
            f"clDice proxy: {float(metrics.get('cldice_proxy', float('nan'))):.3f}",
            f"pred comps: {int(metrics.get('pred_components', 0))}",
            f"gt comps: {int(metrics.get('gt_components', 0))}",
            f"delta comps: {float(metrics.get('component_delta', float('nan'))):.2f}",
        ]
    )
    ax[5].text(0.02, 0.95, table_text, va="top", ha="left", fontsize=11)
    ax[5].set_title("Topology summary")
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
    sample_payload: dict[str, Any],
    sample_extras: dict[str, Any],
) -> dict[str, Any] | None:
    """Build one module-XAI sample payload.

    Args:
        sample_payload (dict[str, Any]): Existing sample payload with RGB/GT/pred.
        sample_extras (dict[str, Any]): Extra tensors from head forward pass.

    Returns:
        dict[str, Any] | None: Module-XAI payload or ``None`` when mandatory
        sample fields are missing.
    """

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
        output[key] = _tensor_to_numpy(sample_extras.get(key))
    return output


def update_module_xai_epoch(
    epoch: int,
    module_cfg: dict[str, Any],
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
        module_cfg (dict[str, Any]): `train.plots.xai.module` configuration.
        layer_ids (list[int]): DINO layer ids used by the model.
        class_index (int): Foreground class index for module diagnostics.
        plot_xai_dir (str): Base XAI directory for artifacts.
        samples (list[dict[str, Any]]): Collected module-XAI samples.
        history (dict[str, Any]): Mutable trend history storage.
        logger (Any | None): Optional logger.

    Returns:
        dict[str, float]: MLflow-friendly scalar metrics.
    """

    enabled = bool(module_cfg.get("enable", True))
    if not enabled:
        return {}

    strict = bool(module_cfg.get("strict", False))
    max_samples = max(1, int(module_cfg.get("max_samples", 8)))
    every_n = max(1, int(module_cfg.get("every_n_epochs", 5)))
    save_maps = bool(module_cfg.get("save_maps", True))
    boundary_band_px = max(1, int(module_cfg.get("boundary_band_px", 3)))
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

    module_root = os.path.join(plot_xai_dir, "module")
    layermix_dir = os.path.join(module_root, "layermix")
    gate_dir = os.path.join(module_root, "gate")
    lora_dir = os.path.join(module_root, "lora")
    topology_dir = os.path.join(module_root, "topology")
    trends_dir = os.path.join(module_root, "trends")
    for path in (layermix_dir, gate_dir, lora_dir, topology_dir, trends_dir):
        _ensure_dir(path)

    save_maps_this_epoch = save_maps and (int(epoch) % every_n == 0)
    layermix_entropy_values: list[float] = []
    layermix_shift_values: list[float] = []
    gate_aurocs: list[float] = []
    gate_aps: list[float] = []
    boundary_delta_values: list[float] = []
    lora_ratio_values: list[float] = []
    lora_region_boundary: list[float] = []
    lora_region_interior: list[float] = []
    lora_region_background: list[float] = []
    topology_cldice_values: list[float] = []
    topology_pred_components: list[float] = []
    topology_gt_components: list[float] = []
    topology_delta_components: list[float] = []
    layer_region_boundary_accum: list[np.ndarray] = []
    layer_region_interior_accum: list[np.ndarray] = []
    layer_region_background_accum: list[np.ndarray] = []

    for sample_idx, sample in enumerate(selected_samples, start=1):
        rgb = np.asarray(sample["rgb"])
        gt_mask = np.asarray(sample["gt_mask"])
        pred_mask = np.asarray(sample["pred_mask"])
        fg_mask, boundary_mask, interior_mask, background_mask = _build_region_masks(
            gt_mask=gt_mask,
            class_index=class_index,
            boundary_band_px=boundary_band_px,
        )

        alpha_arr = _tensor_to_numpy(sample.get("layer_mix_maps"))
        if alpha_arr is not None:
            alpha_np = np.asarray(alpha_arr, dtype=np.float32)
            if alpha_np.ndim == 4:
                alpha_np = alpha_np[0]
            if alpha_np.ndim == 3 and alpha_np.shape[0] > 0:
                alpha_layers: list[np.ndarray] = []
                for layer_idx in range(int(alpha_np.shape[0])):
                    alpha_layers.append(
                        upsample_map(alpha_np[layer_idx], gt_mask.shape[0], gt_mask.shape[1])
                    )
                alpha_up = np.stack(alpha_layers, axis=0).astype(np.float32)
                alpha_up = alpha_up / np.clip(
                    np.sum(alpha_up, axis=0, keepdims=True),
                    1e-8,
                    None,
                )
                alpha_entropy = _alpha_entropy(alpha_up, eps=max(1e-12, entropy_eps))
                alpha_argmax = np.argmax(alpha_up, axis=0).astype(np.int32)
                layermix_entropy_values.append(float(np.mean(alpha_entropy)))

                boundary_vals = np.asarray(
                    [
                        float(np.mean(alpha_up[idx][boundary_mask]))
                        if np.any(boundary_mask)
                        else float("nan")
                        for idx in range(alpha_up.shape[0])
                    ],
                    dtype=np.float32,
                )
                interior_vals = np.asarray(
                    [
                        float(np.mean(alpha_up[idx][interior_mask]))
                        if np.any(interior_mask)
                        else float("nan")
                        for idx in range(alpha_up.shape[0])
                    ],
                    dtype=np.float32,
                )
                background_vals = np.asarray(
                    [
                        float(np.mean(alpha_up[idx][background_mask]))
                        if np.any(background_mask)
                        else float("nan")
                        for idx in range(alpha_up.shape[0])
                    ],
                    dtype=np.float32,
                )
                layer_region_boundary_accum.append(boundary_vals)
                layer_region_interior_accum.append(interior_vals)
                layer_region_background_accum.append(background_vals)
                shift = np.nanmean(boundary_vals - interior_vals)
                if math.isfinite(float(shift)):
                    layermix_shift_values.append(float(shift))

                if save_maps_this_epoch:
                    _save_layermix_panel(
                        output_path=os.path.join(
                            layermix_dir,
                            f"epoch_{epoch:04d}_sample_{sample_idx:02d}_panel.png",
                        ),
                        rgb=rgb,
                        boundary_mask=boundary_mask,
                        alpha_argmax=alpha_argmax,
                        alpha_entropy=alpha_entropy,
                    )

        gate_map = _squeeze_to_hw(_tensor_to_numpy(sample.get("gate_map")))
        if gate_map is not None:
            gate_up = upsample_map(gate_map, gt_mask.shape[0], gt_mask.shape[1])
            roc = _roc_ap(
                scores=gate_up.reshape(-1),
                labels=boundary_mask.astype(np.int64).reshape(-1),
            )
            if roc:
                auroc = float(roc["auroc"])
                ap = float(roc["ap"])
                if math.isfinite(auroc):
                    gate_aurocs.append(auroc)
                if math.isfinite(ap):
                    gate_aps.append(ap)
                if save_maps_this_epoch:
                    _save_gate_panel(
                        output_path=os.path.join(
                            gate_dir,
                            f"epoch_{epoch:04d}_sample_{sample_idx:02d}_gate.png",
                        ),
                        rgb=rgb,
                        gate_map=gate_up,
                        boundary_mask=boundary_mask,
                        roc_info=roc,
                    )

        pre_gate_logits = _tensor_to_numpy(sample.get("pre_gate_logits"))
        if pre_gate_logits is not None and pre_gate_logits.ndim == 4 and np.any(boundary_mask):
            pre_pred = np.argmax(np.asarray(pre_gate_logits[0], dtype=np.float32), axis=0)
            pre_pred = np.asarray(pre_pred, dtype=np.int32)
            if tuple(pre_pred.shape) != tuple(gt_mask.shape):
                pre_pred = upsample_map(
                    pre_pred.astype(np.float32), gt_mask.shape[0], gt_mask.shape[1]
                ).round().astype(np.int32)
            pre_error = np.logical_and(pre_pred != gt_mask, boundary_mask)
            post_error = np.logical_and(pred_mask != gt_mask, boundary_mask)
            delta_map = pre_error.astype(np.float32) - post_error.astype(np.float32)
            delta_value = float(np.mean(delta_map[boundary_mask]))
            boundary_delta_values.append(delta_value)
            if save_maps_this_epoch:
                _save_boundary_error_panel(
                    output_path=os.path.join(
                        gate_dir,
                        f"epoch_{epoch:04d}_sample_{sample_idx:02d}_boundary_delta.png",
                    ),
                    rgb=rgb,
                    boundary_mask=boundary_mask,
                    error_delta_map=delta_map,
                    pre_error=float(np.mean(pre_error[boundary_mask])),
                    post_error=float(np.mean(post_error[boundary_mask])),
                    delta_mean=delta_value,
                )

        if enable_lora:
            lora_base = _squeeze_to_hw(_tensor_to_numpy(sample.get("lora_base_norm_map")))
            lora_delta = _squeeze_to_hw(
                _tensor_to_numpy(sample.get("lora_delta_norm_map"))
            )
            if lora_base is not None and lora_delta is not None:
                base_up = upsample_map(lora_base, gt_mask.shape[0], gt_mask.shape[1])
                delta_up = upsample_map(lora_delta, gt_mask.shape[0], gt_mask.shape[1])
                ratio = delta_up / np.clip(base_up, 1e-8, None)
                ratio_values = ratio[np.isfinite(ratio)]
                if ratio_values.size > 0:
                    lora_ratio_values.append(float(np.mean(ratio_values)))
                    if np.any(boundary_mask):
                        lora_region_boundary.append(float(np.mean(ratio[boundary_mask])))
                    if np.any(interior_mask):
                        lora_region_interior.append(float(np.mean(ratio[interior_mask])))
                    if np.any(background_mask):
                        lora_region_background.append(
                            float(np.mean(ratio[background_mask]))
                        )
                    if save_maps_this_epoch:
                        _save_lora_panel(
                            output_path=os.path.join(
                                lora_dir,
                                f"epoch_{epoch:04d}_sample_{sample_idx:02d}_lora.png",
                            ),
                            rgb=rgb,
                            ratio_map=ratio,
                            ratio_values=ratio_values,
                            region_means={
                                "boundary": float(np.mean(ratio[boundary_mask]))
                                if np.any(boundary_mask)
                                else float("nan"),
                                "interior": float(np.mean(ratio[interior_mask]))
                                if np.any(interior_mask)
                                else float("nan"),
                                "background": float(np.mean(ratio[background_mask]))
                                if np.any(background_mask)
                                else float("nan"),
                            },
                        )

        if enable_topology:
            skeleton_logits = _squeeze_to_hw(_tensor_to_numpy(sample.get("skeleton_logits")))
            if skeleton_logits is not None:
                skel_prob = 1.0 / (1.0 + np.exp(-skeleton_logits))
                skel_up = upsample_map(skel_prob.astype(np.float32), gt_mask.shape[0], gt_mask.shape[1])
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
                tprec = float(np.sum(np.logical_and(pred_skel, fg_mask))) / max(float(np.sum(pred_skel)), 1.0)
                tsens = float(np.sum(np.logical_and(gt_skel, pred_fg))) / max(float(np.sum(gt_skel)), 1.0)
                cldice_proxy = float((2.0 * tprec * tsens) / max(tprec + tsens, 1e-8))
                pred_components = float(_count_components(pred_skel))
                gt_components = float(_count_components(gt_skel))
                component_delta = pred_components - gt_components
                topology_cldice_values.append(cldice_proxy)
                topology_pred_components.append(pred_components)
                topology_gt_components.append(gt_components)
                topology_delta_components.append(component_delta)
                if save_maps_this_epoch:
                    _save_topology_panel(
                        output_path=os.path.join(
                            topology_dir,
                            f"epoch_{epoch:04d}_sample_{sample_idx:02d}_topology.png",
                        ),
                        rgb=rgb,
                        gt_mask=gt_mask,
                        pred_mask=pred_mask,
                        pred_skel=pred_skel,
                        gt_skel=gt_skel,
                        metrics={
                            "cldice_proxy": cldice_proxy,
                            "pred_components": pred_components,
                            "gt_components": gt_components,
                            "component_delta": component_delta,
                        },
                    )

    metrics: dict[str, float] = {}
    if layermix_entropy_values:
        metrics["xai_layermix_entropy_mean"] = float(np.mean(layermix_entropy_values))
    if layermix_shift_values:
        metrics["xai_layermix_boundary_shift_mean"] = float(np.mean(layermix_shift_values))
    if gate_aurocs:
        metrics["xai_gate_boundary_auroc"] = float(np.mean(gate_aurocs))
    if gate_aps:
        metrics["xai_gate_boundary_ap"] = float(np.mean(gate_aps))
    if boundary_delta_values:
        metrics["xai_boundary_error_reduction_mean"] = float(np.mean(boundary_delta_values))
    if lora_ratio_values:
        metrics["xai_lora_ratio_mean"] = float(np.mean(lora_ratio_values))
    if lora_region_boundary:
        metrics["xai_lora_ratio_boundary_mean"] = float(np.mean(lora_region_boundary))
    if lora_region_interior:
        metrics["xai_lora_ratio_interior_mean"] = float(np.mean(lora_region_interior))
    if lora_region_background:
        metrics["xai_lora_ratio_background_mean"] = float(np.mean(lora_region_background))
    if topology_cldice_values:
        metrics["xai_topology_cldice_proxy"] = float(np.mean(topology_cldice_values))
    if topology_pred_components:
        metrics["xai_topology_skel_components_pred"] = float(np.mean(topology_pred_components))
    if topology_gt_components:
        metrics["xai_topology_skel_components_gt"] = float(np.mean(topology_gt_components))
    if topology_delta_components:
        metrics["xai_topology_skel_component_delta"] = float(np.mean(topology_delta_components))

    if layer_region_boundary_accum:
        boundary_mean = np.nanmean(np.stack(layer_region_boundary_accum, axis=0), axis=0)
        interior_mean = np.nanmean(np.stack(layer_region_interior_accum, axis=0), axis=0)
        background_mean = np.nanmean(np.stack(layer_region_background_accum, axis=0), axis=0)
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

    if not metrics:
        if strict:
            raise RuntimeError(
                "Module XAI is enabled but no module-specific metrics were produced."
            )
        return {}

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
                "rho_mean": history.get("xai_lora_ratio_mean", []),
                "rho_boundary": history.get("xai_lora_ratio_boundary_mean", []),
                "rho_interior": history.get("xai_lora_ratio_interior_mean", []),
            },
            "LoRA ratio trends",
            "Ratio",
        ),
        "module_topology_trends.png": (
            {
                "cldice_proxy": history.get("xai_topology_cldice_proxy", []),
                "component_delta": history.get("xai_topology_skel_component_delta", []),
            },
            "Topology trends",
            "Value",
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

    if logger and metrics:
        logger.info(
            "Module XAI epoch %s metrics: %s"
            % (
                int(epoch),
                ", ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items())),
            )
        )
    return metrics
