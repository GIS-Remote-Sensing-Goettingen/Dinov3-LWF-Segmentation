"""Tests for lightweight DINO baseline heads."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import available_heads, build_head  # noqa: E402


def test_registry_contains_new_baseline_heads() -> None:
    """Ensure new baseline heads are registered.

    This guards model-registry wiring for both lightweight baselines.

    Examples:
        >>> True
        True
    """

    names = set(available_heads().keys())
    assert "dino_dense_probe" in names
    assert "dino_segdino_light" in names


def test_dense_probe_build_and_forward_shape() -> None:
    """Validate dense probe forward output shape.

    The logits must match the input image spatial resolution.

    Examples:
        >>> True
        True
    """

    model = build_head(
        "dino_dense_probe",
        num_classes=2,
        dino_channels=64,
        model_cfg={
            "dense_probe": {"norm_type": "groupnorm", "groupnorm_groups": 8},
        },
    )
    image = torch.randn(2, 3, 128, 128)
    features = [torch.randn(2, 64, 8, 8), torch.randn(2, 64, 8, 8)]
    logits = model(image, features)
    assert tuple(logits.shape) == (2, 2, 128, 128)


def test_segdino_light_build_and_forward_shape() -> None:
    """Validate the paper-like SegDINO-light forward output shape.

    The decoded logits must be resized to the input image grid.

    Examples:
        >>> True
        True
    """

    model = build_head(
        "dino_segdino_light",
        num_classes=2,
        dino_channels=64,
        model_cfg={"layers": [5, 11, 17, 23]},
    )
    image = torch.randn(2, 3, 128, 128)
    features = [torch.randn(2, 64, 8, 8) for _ in range(4)]
    logits = model(image, features)
    assert tuple(logits.shape) == (2, 2, 128, 128)


def test_segdino_light_feature_count_mismatch_raises() -> None:
    """Validate exact feature-count enforcement for SegDINO-light.

    Mismatch between extracted features and configured layers should raise.

    Examples:
        >>> True
        True
    """

    model = build_head(
        "dino_segdino_light",
        num_classes=2,
        dino_channels=64,
        model_cfg={"layers": [5, 11, 17, 23]},
    )
    image = torch.randn(1, 3, 64, 64)
    bad_features = [torch.randn(1, 64, 4, 4) for _ in range(3)]
    with pytest.raises(ValueError, match="expected 4 features"):
        _ = model(image, bad_features)


def test_segdino_light_legacy_config_block_raises() -> None:
    """Validate removal of the legacy SegDINO-light config block.

    Old YAMLs should fail clearly instead of silently using ignored settings.

    Examples:
        >>> True
        True
    """

    with pytest.raises(ValueError, match="model.segdino_light is no longer supported"):
        _ = build_head(
            "dino_segdino_light",
            num_classes=2,
            dino_channels=64,
            model_cfg={
                "layers": [5, 11, 17, 23],
                "segdino_light": {"proj_dim": 32},
            },
        )


def test_segdino_light_requires_model_layers() -> None:
    """Validate that SegDINO-light requires explicit selected DINO layers.

    The fixed head should not guess a layer count when `model.layers` is absent.

    Examples:
        >>> True
        True
    """

    with pytest.raises(ValueError, match="requires a non-empty model.layers list"):
        _ = build_head(
            "dino_segdino_light",
            num_classes=2,
            dino_channels=64,
            model_cfg={},
        )


def test_existing_head_build_path_still_works() -> None:
    """Regression check for existing head build path.

    Existing decoder names must remain buildable after adding new baselines.

    Examples:
        >>> True
        True
    """

    head = build_head("unet_lite", num_classes=2, dino_channels=64)
    assert hasattr(head, "forward")
