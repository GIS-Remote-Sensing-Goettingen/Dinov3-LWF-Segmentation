"""Plotting and XAI summary helpers used by training and validation phases."""

from __future__ import annotations

import json
import math
from typing import Any, cast

import numpy as np

from .inference_utils import overlay_heatmap


def save_epoch_plot(
    output_path: str,
    samples: list[dict[str, Any]],
    cmap: str,
    gt_overlay_alpha: float = 0.35,
) -> None:
    """Save a per-epoch validation comparison grid.

    Args:
        output_path (str): PNG output path.
        samples (list[dict[str, Any]]): Validation sample payloads.
        cmap (str): Matplotlib colormap used for segmentation masks.
        gt_overlay_alpha (float): Alpha for the GT overlay on RGB.
    """

    import matplotlib.pyplot as plt

    rows = len(samples)
    if rows == 0:
        return
    fig, axes = plt.subplots(rows, 2, figsize=(10, rows * 3.6))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    axes_arr = np.asarray(axes, dtype=object)
    for row_idx, sample in enumerate(samples):
        rgb = sample["rgb"]
        gt_mask = sample["gt_mask"]
        pred_mask = sample["pred_mask"]
        iou = float(sample["iou"])
        f1 = float(sample["f1"])
        left_ax = axes_arr[row_idx, 0]
        right_ax = axes_arr[row_idx, 1]
        left_ax.imshow(rgb)
        gt_overlay = np.ma.masked_where(gt_mask == 0, gt_mask)
        left_ax.imshow(gt_overlay, cmap=cmap, alpha=gt_overlay_alpha)
        left_ax.set_title(f"Tile {row_idx + 1} | GT overlay")
        left_ax.axis("off")
        right_ax.imshow(pred_mask, cmap=cmap)
        right_ax.set_title(f"Tile {row_idx + 1} | Pred IoU={iou:.3f} F1={f1:.3f}")
        right_ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_epoch_xai_plot(
    output_path: str,
    samples: list[dict[str, Any]],
    cmap: str,
    topk_channels: int = 5,
    render_rollout: bool = True,
    render_pca: bool = True,
    gt_overlay_alpha: float = 0.35,
) -> None:
    """Save a per-epoch explainability grid for sampled validation tiles.

    Args:
        output_path (str): PNG output path.
        samples (list[dict[str, Any]]): Samples with rgb/gt/pred and XAI maps.
        cmap (str): Matplotlib colormap used for segmentation masks.
        topk_channels (int): Number of top channel maps to render per sample.
        render_rollout (bool): Whether to include attention rollout panel.
        render_pca (bool): Whether to include a PCA feature panel.
        gt_overlay_alpha (float): Alpha for GT overlay on RGB.
    """

    import matplotlib.pyplot as plt

    rows = len(samples)
    if rows == 0:
        return
    topk = max(1, int(topk_channels))
    base_cols = 5 if render_rollout else 4
    if render_pca:
        base_cols += 1
    cols = base_cols + topk
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.1))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    axes_arr = np.asarray(axes, dtype=object)
    for row_idx, sample in enumerate(samples):
        rgb = sample["rgb"]
        gt_mask = sample["gt_mask"]
        pred_mask = sample["pred_mask"]
        iou = float(sample["iou"])
        f1 = float(sample["f1"])
        img_importance = sample.get("img_importance")
        dino_importance = sample.get("dino_importance")
        gate_importance = sample.get("gate_importance")
        zero_map = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        attn_cls = np.asarray(sample.get("attn_cls", zero_map), dtype=np.float32)
        attn_rollout = np.asarray(
            sample.get("attn_rollout", zero_map), dtype=np.float32
        )
        gradcam = np.asarray(sample.get("gradcam", zero_map), dtype=np.float32)
        zero_rgb = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.float32)
        pca_rgb = np.asarray(sample.get("pca_rgb", zero_rgb), dtype=np.float32)
        if pca_rgb.ndim != 3 or pca_rgb.shape[2] != 3:
            pca_rgb = zero_rgb
        top_maps = [
            np.asarray(map_data, dtype=np.float32)
            for map_data in sample.get("top_maps", [])
        ]
        top_indices = [int(idx) for idx in sample.get("top_channels", [])]
        top_scores = [float(score) for score in sample.get("top_scores", [])]

        col_idx = 0
        gt_ax = axes_arr[row_idx, col_idx]
        gt_ax.imshow(rgb)
        gt_overlay = np.ma.masked_where(gt_mask == 0, gt_mask)
        gt_ax.imshow(gt_overlay, cmap=cmap, alpha=gt_overlay_alpha)
        gt_ax.set_title(f"Tile {row_idx + 1} | GT overlay")
        gt_ax.axis("off")
        col_idx += 1

        pred_ax = axes_arr[row_idx, col_idx]
        pred_ax.imshow(pred_mask, cmap=cmap)
        pred_title = f"Pred IoU={iou:.3f} F1={f1:.3f}"
        if img_importance is not None and dino_importance is not None:
            pred_title += (
                f"\nImp I/D={float(img_importance):.2f}/{float(dino_importance):.2f}"
            )
        pred_ax.set_title(pred_title)
        pred_ax.axis("off")
        col_idx += 1

        cls_ax = axes_arr[row_idx, col_idx]
        cls_ax.imshow(overlay_heatmap(rgb, attn_cls, cmap="viridis", alpha=0.45))
        cls_ax.set_title("DINO CLS focus")
        cls_ax.axis("off")
        col_idx += 1

        if render_rollout:
            rollout_ax = axes_arr[row_idx, col_idx]
            rollout_ax.imshow(
                overlay_heatmap(rgb, attn_rollout, cmap="viridis", alpha=0.45)
            )
            rollout_ax.set_title("DINO rollout")
            rollout_ax.axis("off")
            col_idx += 1

        cam_ax = axes_arr[row_idx, col_idx]
        cam_ax.imshow(overlay_heatmap(rgb, gradcam, cmap="magma", alpha=0.5))
        cam_title = "Grad-CAM"
        if gate_importance is not None and math.isfinite(float(gate_importance)):
            cam_title += f"\nGate={float(gate_importance):.2f}"
        cam_ax.set_title(cam_title)
        cam_ax.axis("off")
        col_idx += 1

        if render_pca:
            pca_ax = axes_arr[row_idx, col_idx]
            pca_ax.imshow(np.clip(pca_rgb, 0.0, 1.0))
            pca_ax.set_title("DINO PCA (PC1-3)")
            pca_ax.axis("off")
            col_idx += 1

        for top_idx in range(topk):
            top_ax = axes_arr[row_idx, col_idx + top_idx]
            if top_idx < len(top_maps):
                map_data = top_maps[top_idx]
                channel_id = (
                    str(top_indices[top_idx]) if top_idx < len(top_indices) else "?"
                )
                score = top_scores[top_idx] if top_idx < len(top_scores) else 0.0
                top_ax.imshow(overlay_heatmap(rgb, map_data, cmap="plasma", alpha=0.45))
                top_ax.set_title(f"Top {top_idx + 1} ch={channel_id} w={score:.3g}")
            else:
                top_ax.imshow(rgb)
                top_ax.set_title(f"Top {top_idx + 1} unavailable")
            top_ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _normalized_channel_pairs(sample: dict[str, Any]) -> list[tuple[int, float]]:
    """Normalize one sample's top-channel scores.

    Args:
        sample (dict[str, Any]): Payload with `top_channels` and `top_scores`.

    Returns:
        list[tuple[int, float]]: Channel ids with normalized weights.
    """

    channels_raw = sample.get("top_channels", [])
    scores_raw = sample.get("top_scores", [])
    if not isinstance(channels_raw, list) or not isinstance(scores_raw, list):
        return []
    channel_scores: dict[int, float] = {}
    for idx, ch_value in enumerate(channels_raw):
        if idx >= len(scores_raw):
            break
        try:
            ch = int(ch_value)
            score = float(scores_raw[idx])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score) or score <= 0.0:
            continue
        channel_scores[ch] = channel_scores.get(ch, 0.0) + score
    if not channel_scores:
        return []
    total = float(sum(channel_scores.values()))
    if total <= 0.0:
        return []
    return [(ch, score / total) for ch, score in channel_scores.items()]


def _aggregate_channel_importance_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate normalized DINO channel importance across samples.

    Args:
        samples (list[dict[str, Any]]): Validation sample channel payloads.

    Returns:
        dict[str, Any]: Epoch channel summary with mean/presence and entropy.
    """

    importance_sum: dict[int, float] = {}
    presence_count: dict[int, int] = {}
    sample_count = 0
    for sample in samples:
        pairs = _normalized_channel_pairs(sample)
        if not pairs:
            continue
        sample_count += 1
        seen: set[int] = set()
        for ch, weight in pairs:
            importance_sum[ch] = importance_sum.get(ch, 0.0) + float(weight)
            if ch not in seen:
                presence_count[ch] = presence_count.get(ch, 0) + 1
                seen.add(ch)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "mean_importance": {},
            "presence_ratio": {},
            "top_channels": [],
            "entropy": 0.0,
        }
    mean_importance = {
        ch: float(total / float(sample_count)) for ch, total in importance_sum.items()
    }
    presence_ratio = {
        ch: float(count / float(sample_count)) for ch, count in presence_count.items()
    }
    top_channels = sorted(
        mean_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    probs = np.asarray([weight for _, weight in top_channels], dtype=np.float64)
    if probs.size > 0:
        probs = probs / np.clip(probs.sum(), 1e-12, None)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    else:
        entropy = 0.0
    return {
        "sample_count": int(sample_count),
        "mean_importance": mean_importance,
        "presence_ratio": presence_ratio,
        "top_channels": [(int(ch), float(weight)) for ch, weight in top_channels],
        "entropy": entropy,
    }


def _select_stable_channel_ids(
    history: list[dict[str, Any]],
    top_n: int,
    min_presence: float,
) -> list[int]:
    """Select stable channels across epochs using sample-weighted means.

    Args:
        history (list[dict[str, Any]]): Per-epoch channel summaries.
        top_n (int): Number of stable channels to keep.
        min_presence (float): Minimum sample-presence ratio threshold.

    Returns:
        list[int]: Stable channel ids sorted by global importance.
    """

    total_samples = float(
        sum(int(epoch_data.get("sample_count", 0)) for epoch_data in history)
    )
    if total_samples <= 0.0:
        return []
    importance_sum: dict[int, float] = {}
    presence_sum: dict[int, float] = {}
    for epoch_data in history:
        sample_count = float(int(epoch_data.get("sample_count", 0)))
        if sample_count <= 0.0:
            continue
        mean_importance = cast(dict[int, float], epoch_data.get("mean_importance", {}))
        presence_ratio = cast(dict[int, float], epoch_data.get("presence_ratio", {}))
        for ch, value in mean_importance.items():
            weight = float(value)
            if not math.isfinite(weight) or weight < 0.0:
                continue
            importance_sum[int(ch)] = importance_sum.get(int(ch), 0.0) + (
                weight * sample_count
            )
        for ch, value in presence_ratio.items():
            ratio = float(value)
            if not math.isfinite(ratio) or ratio < 0.0:
                continue
            presence_sum[int(ch)] = presence_sum.get(int(ch), 0.0) + (
                ratio * sample_count
            )
    if not importance_sum:
        return []
    global_importance = {
        ch: float(total / total_samples) for ch, total in importance_sum.items()
    }
    global_presence = {
        ch: float(total / total_samples) for ch, total in presence_sum.items()
    }
    sorted_channels = sorted(
        global_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    stable: list[int] = [
        int(ch)
        for ch, _ in sorted_channels
        if global_presence.get(int(ch), 0.0) >= min_presence
    ]
    if len(stable) < top_n:
        for ch, _ in sorted_channels:
            ch_int = int(ch)
            if ch_int not in stable:
                stable.append(ch_int)
            if len(stable) >= top_n:
                break
    return stable[:top_n]


def _grouped_importance_values(
    epoch_summary: dict[str, Any],
    stable_channels: list[int],
) -> tuple[list[str], list[float]]:
    """Build grouped values for stable channels plus OTHER.

    Args:
        epoch_summary (dict[str, Any]): Per-epoch channel summary.
        stable_channels (list[int]): Stable channel ids to track explicitly.

    Returns:
        tuple[list[str], list[float]]: Labels and grouped mean importance values.
    """

    mean_importance = cast(dict[int, float], epoch_summary.get("mean_importance", {}))
    labels = [f"ch{int(ch)}" for ch in stable_channels]
    values = [float(mean_importance.get(int(ch), 0.0)) for ch in stable_channels]
    other_mass = max(0.0, 1.0 - float(sum(values)))
    labels.append("OTHER")
    values.append(other_mass)
    return labels, values


def _save_channel_importance_bar_plot(
    output_path: str,
    epoch: int,
    epoch_summary: dict[str, Any],
    stable_channels: list[int],
) -> None:
    """Save a grouped bar chart for one epoch.

    Args:
        output_path (str): Destination PNG path.
        epoch (int): 1-based epoch index.
        epoch_summary (dict[str, Any]): Per-epoch channel summary.
        stable_channels (list[int]): Stable channel ids for grouping.
    """

    import matplotlib.pyplot as plt

    labels, values = _grouped_importance_values(epoch_summary, stable_channels)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.85), 4.2))
    x = np.arange(len(labels))
    ax.bar(x, values, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean normalized importance")
    ax.set_title(f"Epoch {epoch} DINO channel importance (grouped)")
    ax.set_ylim(0.0, max(0.1, float(max(values, default=0.0)) * 1.2))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_channel_importance_trend_plot(
    output_path: str,
    history: list[dict[str, Any]],
    stable_channels: list[int],
) -> None:
    """Save per-epoch line trends for stable channels and OTHER.

    Args:
        output_path (str): Destination PNG path.
        history (list[dict[str, Any]]): Ordered per-epoch summaries.
        stable_channels (list[int]): Stable channel ids for trend lines.
    """

    import matplotlib.pyplot as plt

    if not history:
        return
    epochs = [
        int(epoch_data.get("epoch", idx + 1)) for idx, epoch_data in enumerate(history)
    ]
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    for channel_id in stable_channels:
        values = [
            float(
                cast(dict[int, float], epoch_data.get("mean_importance", {})).get(
                    int(channel_id), 0.0
                )
            )
            for epoch_data in history
        ]
        ax.plot(epochs, values, marker="o", linewidth=1.8, label=f"ch{channel_id}")
    other_values: list[float] = []
    for epoch_data in history:
        mean_importance = cast(dict[int, float], epoch_data.get("mean_importance", {}))
        stable_mass = float(
            sum(
                float(mean_importance.get(int(channel_id), 0.0))
                for channel_id in stable_channels
            )
        )
        other_values.append(max(0.0, 1.0 - stable_mass))
    ax.plot(
        epochs,
        other_values,
        marker="o",
        linewidth=1.8,
        linestyle="--",
        color="black",
        label="OTHER",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean normalized importance")
    ax.set_title("DINO channel-importance evolution")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_channel_importance_heatmap(
    output_path: str,
    history: list[dict[str, Any]],
    stable_channels: list[int],
) -> None:
    """Save an epoch-by-channel importance heatmap.

    Args:
        output_path (str): Destination PNG path.
        history (list[dict[str, Any]]): Ordered per-epoch summaries.
        stable_channels (list[int]): Stable channel ids for heatmap columns.
    """

    import matplotlib.pyplot as plt

    if not history or not stable_channels:
        return
    matrix = np.asarray(
        [
            [
                float(
                    cast(dict[int, float], epoch_data.get("mean_importance", {})).get(
                        int(channel_id), 0.0
                    )
                )
                for channel_id in stable_channels
            ]
            for epoch_data in history
        ],
        dtype=np.float32,
    )
    fig, ax = plt.subplots(
        figsize=(max(6.0, len(stable_channels) * 0.85), max(4.0, len(history) * 0.35))
    )
    image = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(stable_channels)))
    ax.set_xticklabels(
        [f"ch{channel_id}" for channel_id in stable_channels],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(history)))
    ax.set_yticklabels(
        [
            str(int(epoch_data.get("epoch", idx + 1)))
            for idx, epoch_data in enumerate(history)
        ]
    )
    ax.set_xlabel("Stable channels")
    ax.set_ylabel("Epoch")
    ax.set_title("Validation mean DINO channel importance")
    fig.colorbar(image, ax=ax, label="Mean normalized importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_channel_importance_json(
    output_path: str,
    epoch_summary: dict[str, Any],
    stable_channels: list[int],
) -> None:
    """Write one epoch channel summary to a JSON artifact.

    Args:
        output_path (str): Destination JSON path.
        epoch_summary (dict[str, Any]): Per-epoch channel summary.
        stable_channels (list[int]): Stable channel ids for grouped plots.
    """

    mean_importance = cast(dict[int, float], epoch_summary.get("mean_importance", {}))
    presence_ratio = cast(dict[int, float], epoch_summary.get("presence_ratio", {}))
    top_channels = cast(list[tuple[int, float]], epoch_summary.get("top_channels", []))
    payload = {
        "epoch": int(epoch_summary.get("epoch", 0)),
        "sample_count": int(epoch_summary.get("sample_count", 0)),
        "entropy": float(epoch_summary.get("entropy", 0.0)),
        "stable_channels": [int(channel_id) for channel_id in stable_channels],
        "top_channels": [
            {"channel_id": int(channel_id), "importance": float(weight)}
            for channel_id, weight in top_channels
        ],
        "mean_importance": {
            str(int(channel_id)): float(value)
            for channel_id, value in mean_importance.items()
        },
        "presence_ratio": {
            str(int(channel_id)): float(value)
            for channel_id, value in presence_ratio.items()
        },
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def resolve_cam_layer(layers: list[int], mode: str) -> int | None:
    """Resolve the DINO layer index used for CAM extraction.

    Args:
        layers (list[int]): Configured backbone layer indices.
        mode (str): Selection mode (`last_requested_layer` or `first_requested_layer`).

    Returns:
        int | None: Selected layer index, or None when no layers are configured.
    """

    if not layers:
        return None
    normalized = str(mode).strip().lower()
    if normalized == "first_requested_layer":
        return int(layers[0])
    return int(layers[-1])


def compute_tile_iou_f1(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_index: int = 1,
    ignore_index: int | None = None,
) -> tuple[float, float]:
    """Compute binary IoU and F1 for one validation tile.

    Args:
        pred_mask (np.ndarray): Predicted class mask.
        gt_mask (np.ndarray): Ground-truth class mask.
        class_index (int): Positive class index for binary metrics.
        ignore_index (int | None): Optional label index ignored in metric counts.

    Returns:
        tuple[float, float]: IoU and F1 values for the selected class.
    """

    pred = pred_mask.astype(np.int64)
    gt = gt_mask.astype(np.int64)
    valid = np.ones_like(gt, dtype=bool)
    if ignore_index is not None:
        valid &= gt != int(ignore_index)
    if not np.any(valid):
        return 0.0, 0.0
    pred_pos = pred[valid] == int(class_index)
    gt_pos = gt[valid] == int(class_index)
    tp = int(np.logical_and(pred_pos, gt_pos).sum())
    fp = int(np.logical_and(pred_pos, np.logical_not(gt_pos)).sum())
    fn = int(np.logical_and(np.logical_not(pred_pos), gt_pos).sum())
    iou_denom = tp + fp + fn
    iou = float(tp / iou_denom) if iou_denom > 0 else 0.0
    f1_denom = (2 * tp) + fp + fn
    f1 = float((2 * tp) / f1_denom) if f1_denom > 0 else 0.0
    return iou, f1
