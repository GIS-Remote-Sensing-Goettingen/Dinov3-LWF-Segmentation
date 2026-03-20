"""Safety helper tests for training utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.unet_nano_fapm import DinoUNetNanoFAPMHead  # noqa: E402
from pipeline.phases.train_xai import _build_plot_rgb  # noqa: E402
from pipeline.train_utils import (  # noqa: E402
    NormalizedForwardAdapter,
    align_logits_to_labels,
    forward_with_optional_extras,
    resolve_lr_metrics,
    should_warn_high_logit,
    use_adamw_only_for_head,
)
from utils.optim import Muon  # noqa: E402


def test_use_adamw_only_for_baseline_heads() -> None:
    """Ensure baseline lightweight heads route to AdamW-only optimization.

    This verifies the stability-oriented optimizer routing helper.

    Examples:
        >>> True
        True
    """

    assert use_adamw_only_for_head("dino_dense_probe")
    assert use_adamw_only_for_head("dino_segdino_light")
    assert not use_adamw_only_for_head("unet_lite")


def test_align_logits_to_labels_downsamples_to_native_label_grid() -> None:
    """Supervised logits should be resized down to the label grid.

    Examples:
        >>> True
        True
    """

    logits = torch.randn(2, 3, 20, 20)
    labels = torch.zeros(2, 4, 4, dtype=torch.long)

    aligned = align_logits_to_labels(logits, labels)

    assert aligned is not None
    assert aligned.shape == (2, 3, 4, 4)


def test_build_plot_rgb_matches_native_label_grid_shape() -> None:
    """XAI preview RGB should be rendered on the GT/pred label grid.

    Examples:
        >>> True
        True
    """

    sample_img = torch.linspace(0.0, 1.0, steps=3 * 8 * 8).reshape(1, 3, 8, 8)
    sample_gt = torch.zeros(1, 2, 2, dtype=torch.long)

    rgb = _build_plot_rgb(sample_img, sample_gt)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8


def test_should_warn_high_logit_uses_batch_value() -> None:
    """Ensure high-logit warning helper is batch-local and finite-safe.

    The helper must ignore non-finite values and warn only on threshold breach.

    Examples:
        >>> True
        True
    """

    assert should_warn_high_logit(120.0, 80.0)
    assert not should_warn_high_logit(40.0, 80.0)
    assert not should_warn_high_logit(float("nan"), 80.0)


def test_resolve_lr_metrics_handles_adamw_and_muon() -> None:
    """Ensure LR metric extraction works for both optimizer paths.

    Examples:
        >>> True
        True
    """

    p_adamw = torch.nn.Parameter(torch.ones(1))
    opt_adamw = torch.optim.AdamW([p_adamw], lr=1e-3)
    sch_adamw = torch.optim.lr_scheduler.OneCycleLR(
        opt_adamw, max_lr=1e-3, epochs=1, steps_per_epoch=1
    )
    lr, lr_muon, lr_adamw = resolve_lr_metrics(opt_adamw, sch_adamw)
    assert lr >= 0.0
    assert lr_muon == 0.0
    assert lr_adamw > 0.0

    p_muon = torch.nn.Parameter(torch.ones(2, 2))
    p_aux = torch.nn.Parameter(torch.ones(2))
    opt_muon = Muon(
        [p_muon],
        lr=0.02,
        adamw_params=[p_aux],
        adamw_lr=1e-3,
        adamw_wd=0.01,
    )
    sch_muon = torch.optim.lr_scheduler.OneCycleLR(
        opt_muon, max_lr=0.02, epochs=1, steps_per_epoch=1
    )
    lr2, lr_muon2, lr_adamw2 = resolve_lr_metrics(opt_muon, sch_muon)
    assert lr2 >= 0.0
    assert lr_muon2 == lr2
    assert lr_adamw2 > 0.0


def test_forward_adapter_preserves_aux_and_edge_outputs() -> None:
    """Ensure wrapper-based forward paths keep optional outputs accessible.

    This covers the normalized adapter path used by DDP-wrapped training.

    Examples:
        >>> True
        True
    """

    head = DinoUNetNanoFAPMHead(num_classes=2, dino_channels=8)
    adapter = NormalizedForwardAdapter(head)
    image = torch.randn(1, 3, 256, 256)
    features = [
        torch.randn(1, 8, 32, 32),
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 8, 8, 8),
        torch.randn(1, 8, 4, 4),
    ]

    logits, aux_logits, edge_logits, skeleton_logits, payload = (
        forward_with_optional_extras(
            adapter,
            image,
            features,
            require_aux_logits=True,
        )
    )

    assert logits.shape == (1, 2, 256, 256)
    assert aux_logits is not None
    assert aux_logits.shape == (1, 2, 32, 32)
    assert edge_logits is not None
    assert edge_logits.shape == (1, 1, 256, 256)
    assert skeleton_logits is None
    assert "decoder_h8" in payload


def test_forward_with_optional_extras_requires_aux_when_configured() -> None:
    """Fail fast when aux supervision is enabled but aux logits are missing.

    This prevents silent DDP runs where the auxiliary head never receives
    gradients.

    Examples:
        >>> True
        True
    """

    class ForwardOnly(torch.nn.Module):
        """Minimal forward-only module for aux-missing regression coverage."""

        def __init__(self) -> None:
            """Build the tiny forward-only probe head.

            The module intentionally omits any aux-returning helper.
            """

            super().__init__()
            self.head = torch.nn.Conv2d(8, 2, kernel_size=1)

        def forward(
            self,
            image: torch.Tensor,
            features: list[torch.Tensor],
        ) -> torch.Tensor:
            """Return only main logits for the first feature map.

            Args:
                image (torch.Tensor): Input image tensor.
                features (list[torch.Tensor]): Feature tensors.

            Returns:
                torch.Tensor: Main logits without any auxiliary payload.
            """

            _ = image
            return self.head(features[0])

    with pytest.raises(RuntimeError, match="Auxiliary supervision is enabled"):
        forward_with_optional_extras(
            ForwardOnly(),
            torch.randn(1, 3, 4, 4),
            [torch.randn(1, 8, 4, 4)],
            require_aux_logits=True,
        )


def test_forward_adapter_restores_ds_head_gradients() -> None:
    """Ensure aux-head parameters receive gradients through the adapter path.

    This exercises the same path that previously dropped `ds_head` gradients
    under DDP.

    Examples:
        >>> True
        True
    """

    head = DinoUNetNanoFAPMHead(num_classes=2, dino_channels=8)
    adapter = NormalizedForwardAdapter(head)
    image = torch.randn(1, 3, 256, 256)
    features = [
        torch.randn(1, 8, 32, 32),
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 8, 8, 8),
        torch.randn(1, 8, 4, 4),
    ]

    logits, aux_logits, _, _, _ = forward_with_optional_extras(
        adapter,
        image,
        features,
        require_aux_logits=True,
    )
    assert aux_logits is not None

    loss = logits.mean() + aux_logits.mean()
    loss.backward()

    assert head.ds_head.weight.grad is not None
    assert head.ds_head.bias.grad is not None
