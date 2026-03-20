"""Safety helper tests for training utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.unet import DinoUNetHead  # noqa: E402
from models.unet_nano import DinoUNetNanoHead  # noqa: E402
from models.unet_nano_fapm import DinoUNetNanoFAPMHead  # noqa: E402
from pipeline.phases.train_xai import _build_plot_rgb  # noqa: E402
from pipeline.train_utils import (  # noqa: E402
    NormalizedForwardAdapter,
    align_logits_to_labels,
    evaluate,
    forward_with_optional_extras,
    head_supports_aux_logits,
    head_uses_backbone_features,
    resolve_lr_metrics,
    resolve_model_patch_size,
    should_warn_high_logit,
    use_adamw_only_for_head,
)
from utils import SegmentationLoss  # noqa: E402
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


def test_standard_unet_is_image_only_and_has_no_aux_logits() -> None:
    """Standard `unet` should bypass DINO features and aux supervision.

    Examples:
        >>> True
        True
    """

    assert not head_uses_backbone_features("unet")
    assert not head_supports_aux_logits("unet")
    assert (
        resolve_model_patch_size("facebook/dinov3-vitl16-pretrain-sat493m", "unet") == 1
    )


def test_standard_unet_forward_ignores_feature_list() -> None:
    """The standard U-Net head should segment directly from the image tensor.

    Examples:
        >>> True
        True
    """

    head = DinoUNetHead(num_classes=2, dino_channels=1024)
    image = torch.randn(2, 3, 128, 128, requires_grad=True)
    logits = head(
        image,
        [torch.randn(2, 1024, 8, 8), torch.randn(2, 1024, 4, 4)],
    )

    assert logits.shape == (2, 2, 128, 128)
    logits.mean().backward()
    assert image.grad is not None


def test_evaluate_supports_image_only_unet_without_backbone() -> None:
    """Validation should run for the standard U-Net with empty feature lists.

    Examples:
        >>> True
        True
    """

    head = DinoUNetHead(num_classes=2, dino_channels=1024)
    loader = [
        (
            torch.randn(1, 3, 64, 64),
            [],
            torch.zeros(1, 64, 64, dtype=torch.long),
        )
    ]
    loss_fn = SegmentationLoss(num_classes=2, aux_weight=0.4)

    loss, metrics = evaluate(
        head,
        loader,
        loss_fn,
        torch.device("cpu"),
        use_amp=False,
        num_classes=2,
        cache_features=False,
        requires_backbone_features=False,
        require_aux_logits=False,
        ps=1,
    )

    assert isinstance(loss, float)
    assert "miou" in metrics


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


def _build_nano_feature_pyramid(
    dino_channels: int,
    feature_count: int,
) -> list[torch.Tensor]:
    """Build a shallow-to-deep feature pyramid for Nano head tests.

    Args:
        dino_channels (int): Channel count for each synthetic feature map.
        feature_count (int): Number of DINO feature maps to synthesize.
    """

    spatial_sizes = [32, 16, 8, 4]
    if feature_count < 0 or feature_count > 5:
        raise ValueError(f"Unsupported test feature count: {feature_count}")
    if feature_count == 0:
        selected_sizes: list[int] = []
    elif feature_count <= 4:
        selected_sizes = spatial_sizes[-feature_count:]
    else:
        selected_sizes = [64] + spatial_sizes
    return [torch.randn(1, dino_channels, size, size) for size in selected_sizes]


@pytest.mark.parametrize("feature_count", [1, 2, 3, 4])
def test_unet_nano_accepts_one_to_four_feature_maps(feature_count: int) -> None:
    """`unet_nano` should adapt to 1-4 DINO layers without losing aux outputs.

    This verifies the new optional-skip behavior while preserving the expected
    full-resolution main logits and H/8 auxiliary logits.

    Args:
        feature_count (int): Number of DINO feature maps passed to the head.

    Examples:
        >>> True
        True
    """

    head = DinoUNetNanoHead(num_classes=2, dino_channels=8)
    image = torch.randn(1, 3, 256, 256)
    features = _build_nano_feature_pyramid(8, feature_count)

    logits, aux_logits = head.forward_with_aux(image, features)
    payload = head.forward_with_extras(image, features)

    assert logits.shape == (1, 2, 256, 256)
    assert aux_logits.shape == (1, 2, 32, 32)
    assert payload["logits"].shape == logits.shape
    assert payload["aux_logits"].shape == aux_logits.shape
    assert "bottleneck_features" in payload
    assert "decoder_h8" in payload
    assert "decoder_h4" in payload
    assert "decoder_h2" in payload


@pytest.mark.parametrize("feature_count", [1, 2, 3, 4])
def test_unet_nano_backward_works_with_partial_feature_pyramids(
    feature_count: int,
) -> None:
    """`unet_nano` gradients should flow for every supported feature count.

    This ensures the adaptive skip-dropping path still propagates gradients to
    both the auxiliary and final prediction heads.

    Args:
        feature_count (int): Number of DINO feature maps passed to the head.

    Examples:
        >>> True
        True
    """

    head = DinoUNetNanoHead(num_classes=2, dino_channels=8)
    image = torch.randn(1, 3, 256, 256)
    features = _build_nano_feature_pyramid(8, feature_count)

    logits, aux_logits = head.forward_with_aux(image, features)
    loss = logits.mean() + aux_logits.mean()
    loss.backward()

    assert head.ds_head.weight.grad is not None
    assert head.final_conv.weight.grad is not None


@pytest.mark.parametrize("feature_count", [0, 5])
def test_unet_nano_rejects_invalid_feature_counts(feature_count: int) -> None:
    """`unet_nano` should fail fast outside the supported 1-4 layer range.

    This guards against misconfigured `model.layers` lists that would otherwise
    produce ambiguous decoder wiring.

    Args:
        feature_count (int): Number of DINO feature maps passed to the head.

    Examples:
        >>> True
        True
    """

    head = DinoUNetNanoHead(num_classes=2, dino_channels=8)
    image = torch.randn(1, 3, 256, 256)
    features = _build_nano_feature_pyramid(8, feature_count)

    with pytest.raises(ValueError, match="requires 1 to 4 DINO feature maps"):
        head.forward_with_aux(image, features)
