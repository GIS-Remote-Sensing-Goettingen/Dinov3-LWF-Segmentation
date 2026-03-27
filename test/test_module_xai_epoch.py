"""Regression coverage for module-XAI topology diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.xai.module_xai import _save_topology_panel  # noqa: E402
from pipeline.xai.module_xai_epoch import _compute_topology_metrics  # noqa: E402


def test_topology_metrics_distinguish_mask_proxy_from_explicit_skeleton() -> None:
    """Proxy clDice can stay high even when explicit skeleton F1 is low.

    Examples:
        >>> True
        True
    """

    gt_mask = np.zeros((16, 16), dtype=np.int64)
    gt_mask[8, 2:14] = 1
    pred_mask = np.zeros((16, 16), dtype=np.int64)
    pred_mask[7:10, 1:15] = 1
    skeleton_logits = np.full((16, 16), -12.0, dtype=np.float32)
    skeleton_logits[8, 4] = 12.0
    skeleton_logits[8, 11] = 12.0

    metrics = _compute_topology_metrics(
        epoch=1,
        sample_idx=1,
        sample={"skeleton_logits": skeleton_logits},
        class_index=1,
        fg_mask=gt_mask == 1,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        gate_threshold=0.5,
        save_maps_this_epoch=False,
        topology_dir="unused",
    )

    assert metrics["cldice_proxy"] > 0.95
    assert metrics["skel_f1"] < 0.5
    assert metrics["skel_precision"] > metrics["skel_recall"]


def test_save_topology_panel_writes_probability_and_binary_views(
    tmp_path: Path,
) -> None:
    """Topology panel should render with probability and thresholded skeleton maps.

    Examples:
        >>> True
        True

    Args:
        tmp_path (Path): Temporary directory used for the rendered panel.
    """

    out_path = tmp_path / "topology.png"
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    gt_mask = np.zeros((8, 8), dtype=np.int64)
    pred_mask = np.zeros((8, 8), dtype=np.int64)
    pred_skel_prob = np.linspace(0.0, 1.0, num=64, dtype=np.float32).reshape(8, 8)
    pred_skel = pred_skel_prob >= 0.5
    gt_skel = np.zeros((8, 8), dtype=bool)
    gt_skel[4, 2:6] = True

    _save_topology_panel(
        output_path=str(out_path),
        rgb=rgb,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        pred_skel_prob=pred_skel_prob,
        pred_skel=pred_skel,
        gt_skel=gt_skel,
        threshold=0.5,
        metrics={
            "cldice_proxy": 0.9,
            "skel_f1": 0.2,
            "skel_recall": 0.1,
            "skel_precision": 1.0,
            "skel_prob_mean": 0.2,
            "skel_prob_p95": 0.8,
            "skel_pred_pos_rate": 0.25,
            "pred_components": 2.0,
            "gt_components": 1.0,
            "component_delta": 1.0,
        },
    )

    assert out_path.is_file()
    assert out_path.stat().st_size > 0
