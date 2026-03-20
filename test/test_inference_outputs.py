"""Inference output helper tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fiona
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.inference_utils import (  # noqa: E402
    append_prediction_shapefile,
    build_blend_weight_mask,
    compute_gradcam_with_topk_channels,
    extract_prediction_features,
    overlay_binary_mask,
)


class _DummyBatch(dict):
    """Minimal processor batch payload with `.to()` support."""

    def to(self, device: torch.device) -> "_DummyBatch":
        """Move tensor values to the requested device.

        Args:
            device (torch.device): Target device.

        Returns:
            _DummyBatch: Device-moved batch payload.
        """

        return _DummyBatch({key: value.to(device) for key, value in self.items()})


class _DummyProcessor:
    """Processor stub that mirrors the backbone helper contract."""

    def __call__(
        self,
        *,
        images: np.ndarray,
        return_tensors: str,
        do_resize: bool,
        do_center_crop: bool,
    ) -> _DummyBatch:
        """Build a BatchFeature-like payload for one HWC image.

        Args:
            images (np.ndarray): Input image in HWC format.
            return_tensors (str): Unused tensor format selector.
            do_resize (bool): Unused resize flag.
            do_center_crop (bool): Unused crop flag.

        Returns:
            _DummyBatch: Payload containing `pixel_values`.
        """

        _ = return_tensors, do_resize, do_center_crop
        pixel_values = (
            torch.from_numpy(images.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            / 255.0
        )
        return _DummyBatch({"pixel_values": pixel_values})


class _DummyBackbone(nn.Module):
    """Small backbone stub that exposes one hidden-state layer."""

    def __init__(self) -> None:
        """Initialize the fake backbone config.

        This keeps the test backbone interface aligned with the real backbone.
        """

        super().__init__()
        self.config = SimpleNamespace(num_register_tokens=0)

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool) -> Any:
        """Return one patch-token hidden state derived from the input image.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            output_hidden_states (bool): Unused hidden-state flag.

        Returns:
            Any: Namespace with `hidden_states`.
        """

        _ = output_hidden_states
        pixel_values = pixel_values.requires_grad_(True)
        pooled = F.adaptive_avg_pool2d(pixel_values, (2, 2))
        patch_tokens = pooled.permute(0, 2, 3, 1).reshape(1, 4, 3)
        cls_token = patch_tokens.mean(dim=1, keepdim=True)
        hidden_state = torch.cat([cls_token, patch_tokens], dim=1)
        return SimpleNamespace(hidden_states=[hidden_state, hidden_state])


class _PayloadHead(nn.Module):
    """Head stub that returns a payload dict instead of plain logits."""

    def forward(
        self, image: torch.Tensor, features: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Project one feature map into two-class logits.

        Args:
            image (torch.Tensor): Unused image tensor.
            features (list[torch.Tensor]): Feature tensors.

        Returns:
            dict[str, torch.Tensor]: Payload containing `logits`.
        """

        _ = image
        feat = features[0]
        logits = torch.stack([feat[:, 0], feat[:, 1]], dim=1)
        return {"logits": logits}


class _ExplodingHead(nn.Module):
    """Head stub that raises to exercise Grad-CAM exception reporting."""

    def forward(
        self, image: torch.Tensor, features: list[torch.Tensor]
    ) -> torch.Tensor:
        """Raise one deterministic runtime error.

        Args:
            image (torch.Tensor): Unused image tensor.
            features (list[torch.Tensor]): Unused feature tensors.

        Returns:
            torch.Tensor: Never returns.
        """

        _ = image, features
        raise RuntimeError("boom")


class _RecordingLogger:
    """Minimal logger stub used by Grad-CAM regression tests."""

    def __init__(self) -> None:
        """Initialize the captured info messages.

        The tests inspect these messages to validate Grad-CAM failures.
        """

        self.info_messages: list[str] = []

    def info(self, message: str) -> None:
        """Record one info-level message.

        Args:
            message (str): Message text.
        """

        self.info_messages.append(str(message))


def test_center_weight_mask_emphasizes_tile_center() -> None:
    """Center-weighted merge masks should favor central pixels.

    This guards the seam-reduction weighting used during scene assembly.
    """

    mask = build_blend_weight_mask(5, 5, mode="center_weighted")
    assert mask.shape == (5, 5)
    assert float(mask[2, 2]) > float(mask[0, 0])
    assert float(mask.min()) > 0.0


def test_overlay_binary_mask_tints_only_foreground() -> None:
    """Prediction overlay should tint foreground pixels and keep background.

    This keeps the unified inference figure readable while preserving context.
    """

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    overlay = overlay_binary_mask(rgb, mask, color=(120, 190, 255), alpha=0.5)
    assert overlay[0, 0].tolist() == [0, 0, 0]
    assert overlay[0, 1].tolist() == [60, 95, 127]


def test_gradcam_supports_payload_head_outputs() -> None:
    """Grad-CAM should handle heads that return payload dicts with logits.

    This covers heads that return richer forward payloads instead of raw logits.
    """

    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    result = compute_gradcam_with_topk_channels(
        image_hw3=image,
        backbone=_DummyBackbone(),
        head=_PayloadHead(),
        processor=_DummyProcessor(),
        device=torch.device("cpu"),
        layers=[1],
        ps=16,
        class_index=1,
        topk_channels=2,
    )

    assert result["success"] is True
    assert result["failure_reason"] is None
    assert result["selected_layer"] == 1
    assert result["cam_map"].shape == (2, 2)
    assert len(result["top_indices"]) == 2


def test_gradcam_reports_missing_selected_layer() -> None:
    """Grad-CAM should report when the requested CAM layer is not configured.

    This keeps layer-selection failures explicit instead of silently falling back.
    """

    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    result = compute_gradcam_with_topk_channels(
        image_hw3=image,
        backbone=_DummyBackbone(),
        head=_PayloadHead(),
        processor=_DummyProcessor(),
        device=torch.device("cpu"),
        layers=[1],
        ps=16,
        class_index=1,
        topk_channels=2,
        cam_layer=23,
    )

    assert result["success"] is False
    assert result["failure_stage"] == "selected_layer_missing"
    assert "23" in str(result["failure_reason"])


def test_gradcam_reports_exception_details() -> None:
    """Grad-CAM should preserve the underlying exception in failure logs.

    This keeps debugging output actionable when the head forward path fails.
    """

    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    logger = _RecordingLogger()
    result = compute_gradcam_with_topk_channels(
        image_hw3=image,
        backbone=_DummyBackbone(),
        head=_ExplodingHead(),
        processor=_DummyProcessor(),
        device=torch.device("cpu"),
        layers=[1],
        ps=16,
        class_index=1,
        topk_channels=2,
        logger=logger,
    )

    assert result["success"] is False
    assert result["failure_stage"] == "exception"
    assert result["failure_reason"] == "RuntimeError: boom"
    assert any("RuntimeError: boom" in message for message in logger.info_messages)


def test_prediction_shapefile_append_uses_epsg4326(tmp_path: Path) -> None:
    """Vector export should append features into one EPSG:4326 shapefile.

    The append path should preserve the cumulative dataset and CRS metadata.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    output_path = tmp_path / "predictions_4326.shp"
    transform = from_origin(12.0, 46.0, 0.01, 0.01)
    mask_a = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    mask_b = np.array([[0, 0], [0, 1]], dtype=np.uint8)

    features_a = extract_prediction_features(
        mask_a,
        transform,
        "EPSG:4326",
        source_id="scene_a",
        run_id="run_1",
    )
    features_b = extract_prediction_features(
        mask_b,
        transform,
        "EPSG:4326",
        source_id="scene_b",
        run_id="run_1",
    )

    append_prediction_shapefile(str(output_path), features_a, target_epsg=4326)
    append_prediction_shapefile(str(output_path), features_b, target_epsg=4326)

    with fiona.open(output_path) as src:
        records = list(src)
        assert len(records) == 2
        assert str(src.crs).lower().find("4326") != -1
        assert {record["properties"]["source_id"] for record in records} == {
            "scene_a",
            "scene_b",
        }
