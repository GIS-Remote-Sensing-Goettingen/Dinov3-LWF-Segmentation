"""Tests for lightweight DINO baseline heads."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models.deeplabv3 as deeplabv3_module  # noqa: E402
import models.mask2former_semantic as mask2former_module  # noqa: E402
from models import available_heads, build_head  # noqa: E402


def test_registry_contains_new_baseline_heads() -> None:
    """Ensure new baseline heads are registered.

    This guards model-registry wiring for both lightweight baselines.

    Examples:
        >>> True
        True
    """

    names = set(available_heads().keys())
    assert "deeplabv3" in names
    assert "dino_dense_probe" in names
    assert "dino_segdino_light" in names
    assert "mask2former_semantic" in names


def test_deeplabv3_build_and_forward_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate the torchvision DeepLabV3 adapter payload shape.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to stub the torchvision
            factory and avoid network-bound weight downloads during tests.

    Examples:
        >>> True
        True
    """

    class DummyDeepLab(torch.nn.Module):
        def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            """Return torchvision-like DeepLab outputs for one RGB batch.

            Args:
                image (torch.Tensor): Input image tensor.

            Returns:
                dict[str, torch.Tensor]: Main and auxiliary logits keyed like
                    torchvision segmentation models.
            """

            batch, _, height, width = image.shape
            logits = torch.randn(batch, 2, height, width)
            aux_logits = torch.randn(batch, 2, height, width)
            return {"out": logits, "aux": aux_logits}

    def fake_factory(**_: object) -> torch.nn.Module:
        """Build a deterministic no-download DeepLab stub for tests.

        Args:
            **_ (object): Ignored torchvision factory keyword arguments.

        Returns:
            torch.nn.Module: Stub DeepLab module with torchvision-like outputs.
        """

        return DummyDeepLab()

    monkeypatch.setattr(deeplabv3_module, "deeplabv3_resnet50", fake_factory)
    model = build_head("deeplabv3", num_classes=2, dino_channels=64)
    payload = model(torch.randn(2, 3, 128, 128), [])

    assert tuple(payload["logits"].shape) == (2, 2, 128, 128)
    assert tuple(payload["aux_logits"].shape) == (2, 2, 128, 128)


def test_mask2former_semantic_build_and_native_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate the HF Mask2Former semantic adapter without downloads.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to stub local-HF loading.

    Examples:
        >>> True
        True
    """

    class DummyProcessor:
        """Small processor stub exposing only normalization statistics."""

        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.25, 0.25, 0.25]

    class DummyOutputs:
        """Container mirroring the HF return object attributes."""

        def __init__(
            self,
            *,
            class_queries_logits: torch.Tensor,
            masks_queries_logits: torch.Tensor,
            loss: torch.Tensor | None,
        ) -> None:
            """Store one fake Mask2Former output bundle.

            Args:
                class_queries_logits (torch.Tensor): Query-class logits.
                masks_queries_logits (torch.Tensor): Query-mask logits.
                loss (torch.Tensor | None): Optional native training loss.
            """

            self.class_queries_logits = class_queries_logits
            self.masks_queries_logits = masks_queries_logits
            self.loss = loss

    class DummyMask2Former(torch.nn.Module):
        """Return deterministic query logits and one optional training loss."""

        def forward(
            self,
            pixel_values: torch.Tensor,
            mask_labels: list[torch.Tensor] | None = None,
            class_labels: list[torch.Tensor] | None = None,
            pixel_mask: torch.Tensor | None = None,
            **_: object,
        ) -> DummyOutputs:
            """Return fake query logits for one RGB batch.

            Args:
                pixel_values (torch.Tensor): Normalized image batch.
                mask_labels (list[torch.Tensor] | None): Optional binary masks.
                class_labels (list[torch.Tensor] | None): Optional class ids.
                pixel_mask (torch.Tensor | None): Optional pixel-validity mask.
                **_ (object): Ignored HF keyword arguments.

            Returns:
                DummyOutputs: Query logits and an optional native loss.
            """

            batch, _, height, width = pixel_values.shape
            _ = pixel_mask
            class_queries = torch.randn(
                batch,
                4,
                3,
                device=pixel_values.device,
            )
            mask_queries = torch.randn(
                batch,
                4,
                max(1, height // 4),
                max(1, width // 4),
                device=pixel_values.device,
            )
            loss = None
            if mask_labels is not None and class_labels is not None:
                loss = pixel_values.mean().abs()
            return DummyOutputs(
                class_queries_logits=class_queries,
                masks_queries_logits=mask_queries,
                loss=loss,
            )

    def fake_processor_from_pretrained(
        path: str,
        *,
        local_files_only: bool = True,
    ) -> DummyProcessor:
        """Return a no-download processor stub for one local path.

        Args:
            path (str): Requested local processor path.
            local_files_only (bool): Whether HF local-only loading is enforced.

        Returns:
            DummyProcessor: Stub processor with normalization stats.
        """

        assert local_files_only
        assert path == "/tmp/hf-mask2former"
        return DummyProcessor()

    def fake_model_from_pretrained(
        path: str,
        *,
        local_files_only: bool = True,
        num_labels: int,
        ignore_mismatched_sizes: bool = True,
    ) -> DummyMask2Former:
        """Return a no-download Mask2Former stub for one local path.

        Args:
            path (str): Requested local checkpoint path.
            local_files_only (bool): Whether HF local-only loading is enforced.
            num_labels (int): Requested semantic class count.
            ignore_mismatched_sizes (bool): Whether the classifier head may be
                resized during loading.

        Returns:
            DummyMask2Former: Stub semantic model.
        """

        assert local_files_only
        assert ignore_mismatched_sizes
        assert path == "/tmp/hf-mask2former"
        assert num_labels == 2
        return DummyMask2Former()

    monkeypatch.setattr(
        mask2former_module.AutoImageProcessor,
        "from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        mask2former_module.Mask2FormerForUniversalSegmentation,
        "from_pretrained",
        fake_model_from_pretrained,
    )
    model = build_head(
        "mask2former_semantic",
        num_classes=2,
        dino_channels=64,
        model_cfg={
            "mask2former": {
                "model_name_or_path": "/tmp/hf-mask2former",
                "preprocessor_name_or_path": "/tmp/hf-mask2former",
                "use_pretrained": True,
            }
        },
    )

    image = torch.rand(2, 3, 96, 80)
    labels = torch.tensor(
        [
            [[0, 1], [1, 255]],
            [[0, 0], [1, 1]],
        ],
        dtype=torch.long,
    )
    payload = model(image, [])
    native_payload = model.forward_with_native_loss(
        image,
        [],
        labels,
        ignore_index=255,
    )

    assert tuple(payload["logits"].shape) == (2, 2, 96, 80)
    assert tuple(native_payload["logits"].shape) == (2, 2, 96, 80)
    assert native_payload["native_loss"] is not None


def test_mask2former_target_builder_handles_ignore_and_empty_masks() -> None:
    """Mask2Former targets should skip ignored pixels but keep class masks.

    Examples:
        >>> True
        True
    """

    labels = torch.tensor(
        [
            [[0, 1], [1, 255]],
            [[255, 255], [255, 255]],
        ],
        dtype=torch.long,
    )

    mask_labels, class_labels = mask2former_module.build_mask2former_targets(
        labels,
        num_classes=2,
        ignore_index=255,
        image_size=(4, 4),
    )

    assert tuple(mask_labels[0].shape) == (2, 4, 4)
    assert class_labels[0].tolist() == [0, 1]
    assert tuple(mask_labels[1].shape) == (0, 4, 4)
    assert class_labels[1].numel() == 0


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
