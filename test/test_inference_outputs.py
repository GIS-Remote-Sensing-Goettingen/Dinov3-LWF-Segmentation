"""Inference output helper tests."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fiona
import numpy as np
import pytest
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.transform import from_origin
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline.phases.inference as inference_module  # noqa: E402
from pipeline.context import (  # noqa: E402
    DatasetValidationConfig,
    DistContext,
    InferenceError,
    RunContext,
    StabilityConfig,
)
from pipeline.inference_utils import (  # noqa: E402
    append_prediction_shapefile,
    backup_prediction_raster,
    build_blend_weight_mask,
    build_cumulative_raster_backup_path,
    compute_gradcam_with_topk_channels,
    ensure_cumulative_prediction_raster,
    extract_prediction_features,
    overlay_binary_mask,
    write_prediction_to_cumulative_raster,
)
from pipeline.phases.inference import InferencePhase  # noqa: E402
from pipeline.tracking import HookManager  # noqa: E402


def _write_test_geotiff(
    path: Path,
    data: np.ndarray,
    *,
    transform,
    crs: str = "EPSG:25832",
) -> None:
    """Write one small GeoTIFF fixture.

    Args:
        path (Path): Output path.
        data (np.ndarray): Raster data, either ``(H, W)`` or ``(H, W, C)``.
        transform: Raster affine transform.
        crs (str): CRS identifier.

    Examples:
        >>> callable(_write_test_geotiff)
        True
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        count = 1
        write_data = data[np.newaxis, ...]
    else:
        count = int(data.shape[2])
        write_data = np.transpose(data, (2, 0, 1))
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=int(data.shape[0]),
        width=int(data.shape[1]),
        count=count,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(write_data)


class _DummyBatch(dict):
    """Minimal processor batch payload with `.to()` support."""

    def to(self, device: torch.device) -> "_DummyBatch":
        """Move tensor values to the requested device.

        Args:
            device (torch.device): Target device.

        Returns:
            _DummyBatch: Device-moved batch payload.

        Examples:
            >>> batch = _DummyBatch({"x": torch.ones(1)})
            >>> isinstance(batch.to(torch.device("cpu")), _DummyBatch)
            True
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

        Examples:
            >>> processor = _DummyProcessor()
            >>> payload = processor(
            ...     images=np.zeros((2, 2, 3), dtype=np.uint8),
            ...     return_tensors="pt",
            ...     do_resize=False,
            ...     do_center_crop=False,
            ... )
            >>> tuple(payload["pixel_values"].shape)
            (1, 3, 2, 2)
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

        Examples:
            >>> backbone = _DummyBackbone()
            >>> outputs = backbone(torch.ones(1, 3, 4, 4), output_hidden_states=True)
            >>> len(outputs.hidden_states)
            2
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

        Examples:
            >>> head = _PayloadHead()
            >>> logits = head(
            ...     torch.ones(1, 3, 2, 2),
            ...     [torch.ones(1, 3, 2, 2)],
            ... )["logits"]
            >>> tuple(logits.shape)
            (1, 2, 2, 2)
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

        Examples:
            >>> logger = _RecordingLogger()
            >>> logger.info("hi")
            >>> logger.info_messages[-1]
            'hi'
        """

        self.info_messages.append(str(message))

    def warning(self, message: str) -> None:
        """Record one warning-level message.

        Args:
            message (str): Message text.
        """

        self.info_messages.append(str(message))

    def error(self, message: str) -> None:
        """Record one error-level message.

        Args:
            message (str): Message text.
        """

        self.info_messages.append(str(message))


class _DeterministicHead(nn.Module):
    """Tiny inference head that predicts class 1 on bright pixels."""

    def forward(
        self, image: torch.Tensor, features: list[torch.Tensor]
    ) -> torch.Tensor:
        """Return two-class logits aligned to the input image grid.

        Args:
            image (torch.Tensor): Input image tensor.
            features (list[torch.Tensor]): Unused feature list.

        Returns:
            torch.Tensor: Two-class logits.

        Examples:
            >>> head = _DeterministicHead()
            >>> logits = head(torch.ones(1, 3, 2, 2), [])
            >>> tuple(logits.shape)
            (1, 2, 2, 2)
        """

        _ = features
        foreground = image.mean(dim=1, keepdim=True)
        background = 1.0 - foreground
        return torch.cat([background, foreground], dim=1)


class _PayloadDeterministicHead(nn.Module):
    """Tiny inference head that returns logits inside a payload dict."""

    def forward(
        self, image: torch.Tensor, features: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return two-class logits wrapped in a payload mapping.

        Args:
            image (torch.Tensor): Input image tensor.
            features (list[torch.Tensor]): Unused feature list.

        Returns:
            dict[str, torch.Tensor]: Payload containing `logits`.

        Examples:
            >>> head = _PayloadDeterministicHead()
            >>> payload = head(torch.ones(1, 3, 2, 2), [])
            >>> sorted(payload)
            ['logits']
        """

        _ = features
        foreground = image.mean(dim=1, keepdim=True)
        background = 1.0 - foreground
        return {"logits": torch.cat([background, foreground], dim=1)}


def _make_inference_context(tmp_path: Path, config: dict[str, Any]) -> RunContext:
    """Build the minimal runtime context needed by ``InferencePhase`` tests.

    Args:
        tmp_path (Path): Pytest temporary directory.
        config (dict[str, Any]): Runtime configuration mapping.

    Returns:
        RunContext: Minimal phase-compatible context.

    Examples:
        >>> ctx = _make_inference_context(Path("."), {"inference": {"enable": True}})
        >>> ctx.run_id
        'testrun'
    """

    run_dir = tmp_path / "mlruns" / "0" / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        config=config,
        logger=_RecordingLogger(),
        dist_ctx=DistContext(),
        mlflow_logger=None,
        hook_manager=HookManager([]),
        metrics_writer=None,
        run_dir=run_dir,
        experiment_id="0",
        run_id="testrun",
        start_time=time.time(),
        config_path=None,
        continue_on_error=False,
        stability=StabilityConfig(),
        dataset_validation=DatasetValidationConfig(),
        run_results=[],
    )


def _patch_inference_phase_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
    *,
    payload_head: bool = False,
) -> None:
    """Replace heavyweight inference dependencies with deterministic stubs.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
        checkpoint_path (Path): Stub checkpoint path returned by the resolver.
        payload_head (bool): Whether to return a payload-style head.

    Examples:
        >>> callable(_patch_inference_phase_dependencies)
        True
    """

    monkeypatch.setattr(
        inference_module,
        "build_head",
        (
            (lambda *args, **kwargs: _PayloadDeterministicHead())
            if payload_head
            else (lambda *args, **kwargs: _DeterministicHead())
        ),
    )
    monkeypatch.setattr(
        inference_module,
        "resolve_inference_checkpoint",
        lambda context, infer_cfg: (str(checkpoint_path), "configured"),
    )
    monkeypatch.setattr(
        inference_module,
        "extract_checkpoint_state_dict",
        lambda payload: {},
    )
    monkeypatch.setattr(
        inference_module,
        "validate_checkpoint_compatibility",
        lambda head, state_dict: None,
    )


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


def test_cumulative_prediction_raster_uses_template_grid(tmp_path: Path) -> None:
    """Shared prediction rasters should inherit the label-template grid.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    template_path = tmp_path / "labels.tif"
    output_path = tmp_path / "predictions.tif"
    template = np.zeros((4, 4), dtype=np.uint8)
    transform = from_origin(100.0, 200.0, 1.0, 1.0)
    _write_test_geotiff(template_path, template, transform=transform)

    created = ensure_cumulative_prediction_raster(
        str(output_path),
        str(template_path),
    )

    assert created is True
    with rasterio.open(output_path) as src:
        assert src.transform == transform
        assert str(src.crs) == "EPSG:25832"
        assert src.width == 4
        assert src.height == 4
        assert src.read(1).tolist() == template.tolist()


def test_cumulative_prediction_raster_supports_template_window(
    tmp_path: Path,
) -> None:
    """Shared prediction rasters should support a template-aligned subwindow.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    template_path = tmp_path / "labels.tif"
    output_path = tmp_path / "predictions.tif"
    _write_test_geotiff(
        template_path,
        np.zeros((4, 4), dtype=np.uint8),
        transform=from_origin(100.0, 200.0, 1.0, 1.0),
    )

    created = ensure_cumulative_prediction_raster(
        str(output_path),
        str(template_path),
        template_window=Window(col_off=-2, row_off=-1, width=6, height=5),
    )

    assert created is True
    with rasterio.open(output_path) as src:
        assert src.transform == from_origin(98.0, 201.0, 1.0, 1.0)
        assert src.width == 6
        assert src.height == 5


def test_cumulative_prediction_backup_and_overwrite_are_deterministic(
    tmp_path: Path,
) -> None:
    """Later cumulative writes should overwrite earlier overlapping pixels.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    template_path = tmp_path / "labels.tif"
    output_path = tmp_path / "predictions.tif"
    template = np.zeros((4, 4), dtype=np.uint8)
    _write_test_geotiff(
        template_path,
        template,
        transform=from_origin(0.0, 4.0, 1.0, 1.0),
    )
    ensure_cumulative_prediction_raster(str(output_path), str(template_path))

    first_pred = np.ones((2, 2), dtype=np.uint8)
    second_pred = np.full((2, 2), 2, dtype=np.uint8)
    write_prediction_to_cumulative_raster(
        str(output_path),
        first_pred,
        from_origin(0.0, 4.0, 1.0, 1.0),
        "EPSG:25832",
    )
    backup_path = build_cumulative_raster_backup_path(str(output_path), "run1")
    backup_prediction_raster(str(output_path), backup_path)
    write_prediction_to_cumulative_raster(
        str(output_path),
        second_pred,
        from_origin(1.0, 4.0, 1.0, 1.0),
        "EPSG:25832",
    )

    with rasterio.open(output_path) as current:
        current_data = current.read(1)
    with rasterio.open(backup_path) as backup:
        backup_data = backup.read(1)

    assert backup_data.tolist() == [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert current_data.tolist() == [
        [1, 2, 2, 0],
        [1, 2, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]


def test_directory_inference_writes_one_shared_output_tif(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory inference should update one shared GeoTIFF, not per-image TIFFs.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """

    image_dir = tmp_path / "images"
    image_a = image_dir / "scene_a.tif"
    image_b = image_dir / "scene_b.tif"
    label_path = tmp_path / "labels.tif"
    checkpoint_path = tmp_path / "checkpoint.pth"
    output_tif = tmp_path / "shared_predictions.tif"

    _write_test_geotiff(
        label_path,
        np.zeros((4, 4), dtype=np.uint8),
        transform=from_origin(100.0, 104.0, 1.0, 1.0),
    )
    _write_test_geotiff(
        image_a,
        np.full((2, 2, 3), 255, dtype=np.uint8),
        transform=from_origin(0.0, 4.0, 1.0, 1.0),
    )
    _write_test_geotiff(
        image_b,
        np.full((2, 2, 3), 255, dtype=np.uint8),
        transform=from_origin(2.0, 4.0, 1.0, 1.0),
    )
    torch.save({}, checkpoint_path)
    _patch_inference_phase_dependencies(monkeypatch, checkpoint_path)

    context = _make_inference_context(
        tmp_path,
        {
            "model": {
                "head": "unet",
                "num_classes": 2,
                "dino_channels": 1,
                "backbone": "stub",
                "layers": [1],
            },
            "paths": {"label_path": str(label_path)},
            "inference": {
                "enable": True,
                "device": "cpu",
                "input_dir": str(image_dir),
                "input_tif": "",
                "output_tif": str(output_tif),
                "output_dir": "",
                "glob": "*.tif",
                "checkpoint": str(checkpoint_path),
                "tile_size": 8,
                "overlap": 0.0,
                "merge": {"mode": "uniform"},
                "tta": {
                    "horizontal_flip": False,
                    "vertical_flip": False,
                },
                "explain": {"enable": False},
                "vector": {"enable": False},
            },
        },
    )

    outcome = InferencePhase().execute(context)

    with rasterio.open(output_tif) as src:
        data = src.read(1)
        assert src.transform == from_origin(0.0, 4.0, 1.0, 1.0)
        assert src.width == 4
        assert src.height == 2
    assert data.tolist() == [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    assert outcome.metrics["files_total"] == 2.0
    assert outcome.metrics["cumulative_updates"] == 2.0
    assert outcome.artifacts["output_tif"] == str(output_tif)
    assert not list(tmp_path.glob("*_pred.tif"))


def test_directory_inference_covers_all_tiles_of_large_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory inference should process every tile window of a large scene.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """

    image_dir = tmp_path / "images"
    image_a = image_dir / "scene_a.tif"
    label_path = tmp_path / "labels.tif"
    checkpoint_path = tmp_path / "checkpoint.pth"
    output_tif = tmp_path / "shared_predictions.tif"

    _write_test_geotiff(
        label_path,
        np.zeros((4, 4), dtype=np.uint8),
        transform=from_origin(100.0, 104.0, 1.0, 1.0),
    )
    _write_test_geotiff(
        image_a,
        np.full((4, 4, 3), 255, dtype=np.uint8),
        transform=from_origin(0.0, 4.0, 1.0, 1.0),
    )
    torch.save({}, checkpoint_path)
    _patch_inference_phase_dependencies(monkeypatch, checkpoint_path)

    context = _make_inference_context(
        tmp_path,
        {
            "model": {
                "head": "unet",
                "num_classes": 2,
                "dino_channels": 1,
                "backbone": "stub",
                "layers": [1],
            },
            "paths": {"label_path": str(label_path)},
            "inference": {
                "enable": True,
                "device": "cpu",
                "input_dir": str(image_dir),
                "input_tif": "",
                "output_tif": str(output_tif),
                "output_dir": "",
                "glob": "*.tif",
                "checkpoint": str(checkpoint_path),
                "tile_size": 2,
                "overlap": 0.0,
                "merge": {"mode": "uniform"},
                "tta": {
                    "horizontal_flip": False,
                    "vertical_flip": False,
                },
                "explain": {"enable": False},
                "vector": {"enable": False},
            },
        },
    )

    outcome = InferencePhase().execute(context)

    with rasterio.open(output_tif) as src:
        data = src.read(1)
    assert data.shape == (4, 4)
    assert np.all(data == 1)
    assert outcome.metrics["files_skipped_overlap"] == 0.0
    assert outcome.metrics["cumulative_updates"] == 1.0


def test_directory_inference_honors_input_paths_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory inference should process only the manifest-listed files.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    image_dir = tmp_path / "images"
    image_a = image_dir / "scene_a.tif"
    image_b = image_dir / "scene_b.tif"
    manifest_path = tmp_path / "batch_inputs.txt"
    label_path = tmp_path / "labels.tif"
    checkpoint_path = tmp_path / "checkpoint.pth"
    output_tif = tmp_path / "shared_predictions.tif"

    _write_test_geotiff(
        label_path,
        np.zeros((4, 4), dtype=np.uint8),
        transform=from_origin(0.0, 4.0, 1.0, 1.0),
    )
    _write_test_geotiff(
        image_a,
        np.full((2, 2, 3), 255, dtype=np.uint8),
        transform=from_origin(0.0, 4.0, 1.0, 1.0),
    )
    _write_test_geotiff(
        image_b,
        np.zeros((2, 2, 3), dtype=np.uint8),
        transform=from_origin(2.0, 4.0, 1.0, 1.0),
    )
    manifest_path.write_text(str(image_a) + "\n", encoding="utf-8")
    torch.save({}, checkpoint_path)
    _patch_inference_phase_dependencies(monkeypatch, checkpoint_path)

    context = _make_inference_context(
        tmp_path,
        {
            "model": {
                "head": "unet",
                "num_classes": 2,
                "dino_channels": 1,
                "backbone": "stub",
                "layers": [1],
            },
            "paths": {"label_path": str(label_path)},
            "inference": {
                "enable": True,
                "device": "cpu",
                "input_dir": str(image_dir),
                "input_paths_file": str(manifest_path),
                "input_tif": "",
                "output_tif": str(output_tif),
                "output_dir": "",
                "glob": "*.tif",
                "checkpoint": str(checkpoint_path),
                "tile_size": 8,
                "overlap": 0.0,
                "merge": {"mode": "uniform"},
                "tta": {
                    "horizontal_flip": False,
                    "vertical_flip": False,
                },
                "explain": {"enable": False},
                "vector": {"enable": False},
            },
        },
    )

    outcome = InferencePhase().execute(context)

    with rasterio.open(output_tif) as src:
        data = src.read(1)
        assert src.width == 2
        assert src.height == 2
    assert data.tolist() == [[1, 1], [1, 1]]
    assert outcome.metrics["files_total"] == 1.0
    assert outcome.metrics["cumulative_updates"] == 1.0


def test_directory_inference_overwrites_overlapping_scene_footprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory inference should allow overlapping scenes without skipping.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """

    image_dir = tmp_path / "images"
    image_a = image_dir / "scene_a.tif"
    image_b = image_dir / "scene_b.tif"
    label_path = tmp_path / "labels.tif"
    checkpoint_path = tmp_path / "checkpoint.pth"
    output_tif = tmp_path / "shared_predictions.tif"

    _write_test_geotiff(
        label_path,
        np.zeros((8, 8), dtype=np.uint8),
        transform=from_origin(0.0, 8.0, 1.0, 1.0),
    )
    _write_test_geotiff(
        image_a,
        np.full((6, 6, 3), 255, dtype=np.uint8),
        transform=from_origin(0.0, 8.0, 1.0, 1.0),
    )
    _write_test_geotiff(
        image_b,
        np.full((6, 6, 3), 255, dtype=np.uint8),
        transform=from_origin(0.0, 8.0, 1.0, 1.0),
    )
    torch.save({}, checkpoint_path)
    _patch_inference_phase_dependencies(monkeypatch, checkpoint_path)

    context = _make_inference_context(
        tmp_path,
        {
            "model": {
                "head": "unet",
                "num_classes": 2,
                "dino_channels": 1,
                "backbone": "stub",
                "layers": [1],
            },
            "paths": {"label_path": str(label_path)},
            "inference": {
                "enable": True,
                "device": "cpu",
                "input_dir": str(image_dir),
                "input_tif": "",
                "output_tif": str(output_tif),
                "output_dir": "",
                "glob": "*.tif",
                "checkpoint": str(checkpoint_path),
                "tile_size": 8,
                "overlap": 0.0,
                "merge": {"mode": "uniform"},
                "tta": {
                    "horizontal_flip": False,
                    "vertical_flip": False,
                },
                "explain": {"enable": False},
                "vector": {"enable": False},
            },
        },
    )

    outcome = InferencePhase().execute(context)

    with rasterio.open(output_tif) as src:
        data = src.read(1)
        assert src.width == 6
        assert src.height == 6
    assert data.tolist() == [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ]
    assert outcome.metrics["files_skipped_overlap"] == 0.0
    assert outcome.metrics["cumulative_updates"] == 2.0


def test_directory_inference_accepts_payload_style_heads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory inference should normalize payload-style head outputs.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """

    image_dir = tmp_path / "images"
    image_path = image_dir / "scene_a.tif"
    label_path = tmp_path / "labels.tif"
    checkpoint_path = tmp_path / "checkpoint.pth"
    output_tif = tmp_path / "shared_predictions.tif"

    _write_test_geotiff(
        label_path,
        np.zeros((2, 2), dtype=np.uint8),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
    )
    _write_test_geotiff(
        image_path,
        np.full((2, 2, 3), 255, dtype=np.uint8),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
    )
    torch.save({}, checkpoint_path)
    _patch_inference_phase_dependencies(
        monkeypatch,
        checkpoint_path,
        payload_head=True,
    )

    context = _make_inference_context(
        tmp_path,
        {
            "model": {
                "head": "unet",
                "num_classes": 2,
                "dino_channels": 1,
                "backbone": "stub",
                "layers": [1],
            },
            "paths": {"label_path": str(label_path)},
            "inference": {
                "enable": True,
                "device": "cpu",
                "input_dir": str(image_dir),
                "input_tif": "",
                "output_tif": str(output_tif),
                "output_dir": "",
                "glob": "*.tif",
                "checkpoint": str(checkpoint_path),
                "tile_size": 8,
                "overlap": 0.0,
                "merge": {"mode": "uniform"},
                "tta": {
                    "horizontal_flip": False,
                    "vertical_flip": False,
                },
                "explain": {"enable": False},
                "vector": {"enable": False},
            },
        },
    )

    outcome = InferencePhase().execute(context)

    with rasterio.open(output_tif) as src:
        assert src.read(1).tolist() == [[1, 1], [1, 1]]
    assert outcome.metrics["cumulative_updates"] == 1.0


def test_directory_inference_requires_label_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory inference should fail fast without a usable label raster.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """

    image_dir = tmp_path / "images"
    image_path = image_dir / "scene.tif"
    checkpoint_path = tmp_path / "checkpoint.pth"
    _write_test_geotiff(
        image_path,
        np.full((2, 2, 3), 255, dtype=np.uint8),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
    )
    torch.save({}, checkpoint_path)
    _patch_inference_phase_dependencies(monkeypatch, checkpoint_path)

    context = _make_inference_context(
        tmp_path,
        {
            "model": {
                "head": "unet",
                "num_classes": 2,
                "dino_channels": 1,
                "backbone": "stub",
                "layers": [1],
            },
            "paths": {},
            "inference": {
                "enable": True,
                "device": "cpu",
                "input_dir": str(image_dir),
                "input_tif": "",
                "output_tif": str(tmp_path / "shared_predictions.tif"),
                "output_dir": "",
                "glob": "*.tif",
                "checkpoint": str(checkpoint_path),
                "tile_size": 8,
                "overlap": 0.0,
                "merge": {"mode": "uniform"},
                "tta": {
                    "horizontal_flip": False,
                    "vertical_flip": False,
                },
                "explain": {"enable": False},
                "vector": {"enable": False},
            },
        },
    )

    with pytest.raises(InferenceError, match="label_path"):
        InferencePhase().execute(context)


def test_directory_inference_skips_alignment_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory inference should skip scenes that cannot align to label grid.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """

    image_dir = tmp_path / "images"
    image_a = image_dir / "scene_a.tif"
    image_b = image_dir / "scene_b.tif"
    label_path = tmp_path / "labels.tif"
    checkpoint_path = tmp_path / "checkpoint.pth"
    output_tif = tmp_path / "shared_predictions.tif"

    _write_test_geotiff(
        label_path,
        np.zeros((2, 2), dtype=np.uint8),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        crs="EPSG:25832",
    )
    _write_test_geotiff(
        image_a,
        np.full((2, 2, 3), 255, dtype=np.uint8),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        crs="EPSG:25832",
    )
    _write_test_geotiff(
        image_b,
        np.full((2, 2, 3), 255, dtype=np.uint8),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        crs="EPSG:4326",
    )
    torch.save({}, checkpoint_path)
    _patch_inference_phase_dependencies(monkeypatch, checkpoint_path)

    context = _make_inference_context(
        tmp_path,
        {
            "model": {
                "head": "unet",
                "num_classes": 2,
                "dino_channels": 1,
                "backbone": "stub",
                "layers": [1],
            },
            "paths": {"label_path": str(label_path)},
            "inference": {
                "enable": True,
                "device": "cpu",
                "input_dir": str(image_dir),
                "input_tif": "",
                "output_tif": str(output_tif),
                "output_dir": "",
                "glob": "*.tif",
                "checkpoint": str(checkpoint_path),
                "tile_size": 8,
                "overlap": 0.0,
                "merge": {"mode": "uniform"},
                "tta": {
                    "horizontal_flip": False,
                    "vertical_flip": False,
                },
                "explain": {"enable": False},
                "vector": {"enable": False},
            },
        },
    )

    outcome = InferencePhase().execute(context)

    with rasterio.open(output_tif) as src:
        assert src.read(1).tolist() == [[1, 1], [1, 1]]
    assert outcome.metrics["files_skipped_alignment"] == 1.0
