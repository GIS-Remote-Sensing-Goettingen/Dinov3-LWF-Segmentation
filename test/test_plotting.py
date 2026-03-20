"""Regression coverage for training plot artifact helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.plotting import (  # noqa: E402
    _save_branch_importance_trend_plot,
    save_epoch_plot,
    save_epoch_xai_plot,
    save_training_summary_plot,
)


def _sample_payload() -> dict[str, object]:
    """Build one small plotting payload for regression checks.

    The payload mimics one validation/XAI sample closely enough for the plotting
    helpers to render without needing a full training run.
    """

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 1] = 80
    gt_mask = np.zeros((8, 8), dtype=np.int64)
    gt_mask[2:6, 2:6] = 1
    pred_mask = np.zeros((8, 8), dtype=np.int64)
    pred_mask[3:7, 3:7] = 1
    return {
        "rgb": rgb,
        "gt_mask": gt_mask,
        "pred_mask": pred_mask,
        "iou": 0.5,
        "f1": 0.67,
        "img_importance": 0.4,
        "dino_importance": 0.6,
        "attn_cls": np.full((8, 8), 0.2, dtype=np.float32),
        "attn_rollout": np.full((8, 8), 0.35, dtype=np.float32),
        "gradcam": np.full((8, 8), 0.8, dtype=np.float32),
        "pca_rgb": np.full((8, 8, 3), 0.5, dtype=np.float32),
        "top_channels": [7, 11],
        "top_scores": [0.8, 0.2],
        "top_maps": [
            np.full((8, 8), 0.6, dtype=np.float32),
            np.full((8, 8), 0.4, dtype=np.float32),
        ],
    }


def test_save_epoch_plot_supports_paper_style(tmp_path: Path) -> None:
    """Paper-style metric grids should render successfully.

    Examples:
        >>> True
        True

    Args:
        tmp_path (Path): Temporary output directory.
    """

    out_path = tmp_path / "epoch_plot.png"
    save_epoch_plot(
        str(out_path),
        [_sample_payload()],
        cmap="tab20",
        class_index=1,
        paper_style=True,
    )
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_save_epoch_xai_plot_supports_paper_style(tmp_path: Path) -> None:
    """Paper-style XAI grids should render successfully.

    Examples:
        >>> True
        True

    Args:
        tmp_path (Path): Temporary output directory.
    """

    out_path = tmp_path / "epoch_xai.png"
    save_epoch_xai_plot(
        str(out_path),
        [_sample_payload()],
        cmap="tab20",
        topk_channels=2,
        render_rollout=False,
        render_pca=False,
        class_index=1,
        paper_style=True,
    )
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_save_training_summary_plot_writes_png(tmp_path: Path) -> None:
    """Cross-epoch summary plot should render in paper style.

    Examples:
        >>> True
        True

    Args:
        tmp_path (Path): Temporary output directory.
    """

    out_path = tmp_path / "training_summary.png"
    save_training_summary_plot(
        str(out_path),
        [
            {
                "epoch": 1.0,
                "train_loss": 0.9,
                "val_loss": 0.7,
                "miou": 0.35,
                "f1": 0.52,
            },
            {
                "epoch": 2.0,
                "train_loss": 0.6,
                "val_loss": 0.5,
                "miou": 0.48,
                "f1": 0.64,
            },
        ],
        paper_style=True,
    )
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_branch_contribution_plot_renders_two_lines_with_updated_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Branch trend plot should use two contribution lines and updated wording.

    Examples:
        >>> True
        True

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
        tmp_path (Path): Temporary output directory.
    """

    captured: dict[str, object] = {}
    original_savefig = Figure.savefig

    def _capture_savefig(self: Figure, *args: object, **kwargs: object) -> None:
        """Capture branch plot metadata before writing the file.

        This keeps the regression focused on figure labels without changing how
        the plot is actually written to disk.

        Args:
            self (Figure): Figure being saved.
            args (object): Positional savefig arguments.
            kwargs (object): Keyword savefig arguments.
        """

        ax = self.axes[0]
        captured["title"] = ax.get_title()
        captured["ylabel"] = ax.get_ylabel()
        captured["line_count"] = len(ax.lines)
        _, labels = ax.get_legend_handles_labels()
        captured["legend_labels"] = labels
        original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _capture_savefig)

    out_path = tmp_path / "branch_contribution.png"
    _save_branch_importance_trend_plot(
        str(out_path),
        [
            {"epoch": 1.0, "img_importance_mean": 0.4, "dino_importance_mean": 0.6},
            {"epoch": 2.0, "img_importance_mean": 0.5, "dino_importance_mean": 0.5},
        ],
        paper_style=True,
    )

    assert out_path.is_file()
    assert captured["title"] == "Validation branch contribution over epochs"
    assert captured["ylabel"] == "Mean normalized contribution"
    assert captured["line_count"] == 2
    assert captured["legend_labels"] == [
        "Image contribution",
        "DINO contribution",
    ]
