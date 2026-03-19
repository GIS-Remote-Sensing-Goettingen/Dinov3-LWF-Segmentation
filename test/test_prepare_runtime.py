"""Regression tests for distributed prepare/runtime behavior.

Examples:
    >>> isinstance(_RecordingLogger(), _RecordingLogger)
    True
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module  # noqa: E402
import pipeline.phases.prepare as prepare_module  # noqa: E402
from pipeline.context import DistContext, PhaseResult  # noqa: E402
from pipeline.phases.prepare import PreparePhase  # noqa: E402
from utils.data.core import (  # noqa: E402
    _write_tile_payload_atomic,
    build_tile_grid_layout,
    process_image_tiles_no_features,
    read_label_window_for_image_bounds,
)


class _RecordingLogger:
    """Minimal logger stub that captures emitted messages."""

    def __init__(self) -> None:
        """Initialize the captured message buffers.

        This keeps the stub compatible with the repository logger interface
        while still making assertions easy inside the tests.
        """

        self.info_messages: list[str] = []
        self.error_messages: list[str] = []
        self.debug_messages: list[str] = []

    def info(self, message: str) -> None:
        """Record one info-level message.

        Args:
            message (str): Message text to capture.
        """

        self.info_messages.append(str(message))

    def error(self, message: str) -> None:
        """Record one error-level message.

        Args:
            message (str): Message text to capture.
        """

        self.error_messages.append(str(message))

    def debug(self, message: str) -> None:
        """Record one debug-level message.

        Args:
            message (str): Message text to capture.
        """

        self.debug_messages.append(str(message))


def _write_test_geotiff(
    path: Path,
    data: np.ndarray,
    *,
    transform,
    crs: str = "EPSG:25832",
) -> None:
    """Write one small GeoTIFF fixture for prepare/runtime tests.

    Args:
        path (Path): Output path.
        data (np.ndarray): Raster data, either ``(H, W)`` or ``(H, W, C)``.
        transform: Raster affine transform.
        crs (str): CRS identifier.
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


def _prepare_context(tmp_path: Path, dist_ctx: DistContext) -> SimpleNamespace:
    """Build the minimal context object needed by ``PreparePhase`` tests.

    Args:
        tmp_path (Path): Temporary test directory.
        dist_ctx (DistContext): Distributed context to inject into the phase.

    Returns:
        SimpleNamespace: Context stub compatible with ``PreparePhase``.
    """

    return SimpleNamespace(
        config={
            "paths": {
                "raw_images_dir": str(tmp_path / "images"),
                "label_path": str(tmp_path / "labels.tif"),
                "processed_dir": str(tmp_path / "cache"),
            },
            "prepare": {
                "enable": True,
                "cache_features": False,
                "tile_size": 32,
                "workers": 1,
            },
            "dataset": {},
        },
        dist_ctx=dist_ctx,
        logger=_RecordingLogger(),
    )


def test_prepare_phase_runs_only_on_main_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distributed prepare should only execute the worker on rank 0.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    phase = PreparePhase()
    ctx = _prepare_context(
        tmp_path, DistContext(enabled=True, rank=0, world_size=2, local_rank=0)
    )
    called: list[str] = []

    def fake_prepare_data_tiles(**kwargs: object) -> None:
        """Create one cached tile so prepare metrics can be asserted.

        Args:
            **kwargs (object): Prepare call arguments passed by the phase.
        """

        called.append(str(kwargs["img_dir"]))
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"image": torch.zeros(1)}, output_dir / "tile.pt")

    monkeypatch.setattr(prepare_module, "prepare_data_tiles", fake_prepare_data_tiles)
    monkeypatch.setattr(
        prepare_module,
        "resolve_cache_dir_for_prepare",
        lambda output_dir, *args: output_dir,
    )
    monkeypatch.setattr(
        prepare_module, "broadcast_main_object", lambda dist_ctx, payload: payload
    )

    outcome = phase.execute(ctx)

    assert called == [str(tmp_path / "images")]
    assert outcome.metrics["tiles_total"] == 1.0
    assert outcome.metrics["tiles_added"] == 1.0
    assert outcome.artifacts["processed_dir"] == str(tmp_path / "cache")


def test_prepare_phase_non_main_rank_uses_broadcast_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-main ranks should wait for rank 0 and not run tiling themselves.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    phase = PreparePhase()
    ctx = _prepare_context(
        tmp_path, DistContext(enabled=True, rank=1, world_size=2, local_rank=1)
    )
    called: list[str] = []

    monkeypatch.setattr(
        prepare_module,
        "prepare_data_tiles",
        lambda **kwargs: called.append("unexpected"),
    )
    monkeypatch.setattr(
        prepare_module,
        "broadcast_main_object",
        lambda dist_ctx, payload: {
            "error": None,
            "outcome": {
                "metrics": {"tiles_total": 7.0, "tiles_added": 2.0},
                "artifacts": {"processed_dir": "/shared/cache"},
            },
        },
    )

    outcome = phase.execute(ctx)

    assert called == []
    assert outcome.metrics == {"tiles_total": 7.0, "tiles_added": 2.0}
    assert outcome.artifacts == {"processed_dir": "/shared/cache"}


def test_prepare_phase_propagates_rank_zero_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distributed prepare failures should surface clearly on other ranks.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    phase = PreparePhase()
    ctx = _prepare_context(
        tmp_path, DistContext(enabled=True, rank=1, world_size=2, local_rank=1)
    )

    monkeypatch.setattr(
        prepare_module,
        "broadcast_main_object",
        lambda dist_ctx, payload: {
            "error": {"type": "RuntimeError", "message": "disk full"},
            "outcome": None,
        },
    )

    with pytest.raises(RuntimeError, match="Prepare phase failed on rank 0: disk full"):
        phase.execute(ctx)


def test_atomic_tile_writer_preserves_existing_cache(tmp_path: Path) -> None:
    """A second writer should treat an existing tile as a cache hit.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    save_path = tmp_path / "tile.pt"
    first_payload = {"value": torch.tensor([1])}
    second_payload = {"value": torch.tensor([2])}

    assert _write_tile_payload_atomic(first_payload, str(save_path)) is True
    assert _write_tile_payload_atomic(second_payload, str(save_path)) is False

    stored = torch.load(save_path, weights_only=False, map_location="cpu")
    assert stored["value"].item() == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_tile_writer_cleans_temp_files_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Temp files should be removed if the payload write itself fails.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Monkeypatch fixture.

    Examples:
        >>> True
        True
    """


def test_build_tile_grid_layout_derives_native_label_supervision_sizes(
    tmp_path: Path,
) -> None:
    """Native label-grid tiling should derive compatible image/label sizes.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "labels.tif"
    _write_test_geotiff(
        image_path,
        np.zeros((512, 512, 3), dtype=np.uint8),
        transform=from_origin(0.0, 102.4, 0.2, 0.2),
    )
    _write_test_geotiff(
        label_path,
        np.zeros((128, 128), dtype=np.uint8),
        transform=from_origin(0.0, 128.0, 1.0, 1.0),
    )

    layout = build_tile_grid_layout(
        str(image_path),
        str(label_path),
        requested_tile_size=512,
        patch_size=16,
    )

    assert layout.image_tile_size == 480
    assert layout.label_tile_size == 96
    assert layout.scale_factor == 5


def test_read_label_window_for_image_bounds_keeps_native_label_grid(
    tmp_path: Path,
) -> None:
    """Image-footprint label reads should preserve native label resolution.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "labels.tif"
    _write_test_geotiff(
        image_path,
        np.zeros((20, 20, 3), dtype=np.uint8),
        transform=from_origin(0.0, 4.0, 0.2, 0.2),
    )
    label_data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    _write_test_geotiff(
        label_path,
        label_data,
        transform=from_origin(0.0, 4.0, 1.0, 1.0),
    )

    labels, meta = read_label_window_for_image_bounds(str(image_path), str(label_path))

    assert labels.shape == (4, 4)
    assert labels.tolist() == label_data.tolist()
    assert meta["width"] == 4
    assert meta["height"] == 4
    assert meta["transform"] == from_origin(0.0, 4.0, 1.0, 1.0)


def test_process_image_tiles_no_features_writes_smaller_label_grid_tiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached tiles should keep image and label grids at their native scales.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Monkeypatch fixture.
    """

    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "labels.tif"
    output_dir = tmp_path / "cache"
    image = np.full((32, 32, 3), 10, dtype=np.uint8)
    labels = np.ones((8, 8), dtype=np.uint8)
    _write_test_geotiff(
        image_path,
        image,
        transform=from_origin(0.0, 32.0, 1.0, 1.0),
    )
    _write_test_geotiff(
        label_path,
        labels,
        transform=from_origin(0.0, 32.0, 4.0, 4.0),
    )

    result = process_image_tiles_no_features(
        str(image_path),
        str(label_path),
        str(output_dir),
        tile_size=32,
        patch_size=4,
    )

    assert result["status"] == "ok"
    saved = sorted(output_dir.glob("*.pt"))
    assert len(saved) == 1
    payload = torch.load(saved[0], weights_only=False, map_location="cpu")
    assert tuple(payload["image"].shape) == (32, 32, 3)
    assert np.asarray(payload["label"]).shape == (8, 8)

    save_path = tmp_path / "tile.pt"

    def fake_save(payload: object, path: str) -> None:
        """Write a partial temp file, then simulate a storage failure.

        Args:
            payload (object): Unused payload placeholder.
            path (str): Temp file path chosen by the atomic writer.
        """

        Path(path).write_text("partial", encoding="utf-8")
        raise RuntimeError("disk full")

    monkeypatch.setattr("utils.data.core.torch.save", fake_save)

    with pytest.raises(RuntimeError, match="disk full"):
        _write_tile_payload_atomic({"value": torch.tensor([1])}, str(save_path))

    assert not save_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_main_logs_failed_phase_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI should log failures instead of an unconditional success line.

    Args:
        monkeypatch (pytest.MonkeyPatch): Monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    logger = _RecordingLogger()
    cleanup_called: list[bool] = []

    class _FakeRunner:
        def __init__(self, phases: list[object], processors: list[object]) -> None:
            """Store constructor inputs to mirror the real runner interface.

            Args:
                phases (list[object]): Placeholder phase instances.
                processors (list[object]): Placeholder processor instances.
            """

            self.phases = phases
            self.processors = processors

        def run(self, context: object) -> list[PhaseResult]:
            """Return one failed phase result for the logger regression test.

            Args:
                context (object): Unused run context placeholder.

            Returns:
                list[PhaseResult]: One failed train-phase result.
            """

            return [
                PhaseResult(
                    name="train",
                    status="failed",
                    start_time=0.0,
                    end_time=1.0,
                    duration_s=1.0,
                    metrics={},
                    artifacts={},
                    error=None,
                )
            ]

    monkeypatch.setattr(main_module, "load_config", lambda path: {"_config_path": path})
    monkeypatch.setattr(main_module, "apply_resource_config", lambda config: None)
    monkeypatch.setattr(main_module, "setup_distributed", lambda cfg: DistContext())
    monkeypatch.setattr(
        main_module, "build_logger", lambda config, run_id, enabled: logger
    )
    monkeypatch.setattr(
        main_module,
        "build_run_context",
        lambda config, logger, dist_ctx, run_id: SimpleNamespace(mlflow_logger=None),
    )
    monkeypatch.setattr(main_module, "build_processors", lambda config: [])
    monkeypatch.setattr(main_module, "PhaseRunner", _FakeRunner)
    monkeypatch.setattr(
        main_module, "cleanup_distributed", lambda dist_ctx: cleanup_called.append(True)
    )

    main_module.main("configs/config_hpc.yml")

    assert cleanup_called == [True]
    assert any("failed phases: train" in msg.lower() for msg in logger.error_messages)
    assert not any(
        "All enabled phases completed" in msg for msg in logger.info_messages
    )
