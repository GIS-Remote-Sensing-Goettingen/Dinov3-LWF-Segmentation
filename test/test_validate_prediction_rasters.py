"""Tests for the prediction-raster validation utility."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_validate_module():
    """Load the prediction-raster validator module from disk.

    This mirrors the existing script-wrapper test pattern so the test does not
    depend on `utility/` being installed as a package.

    Returns:
        object: Imported module object.
    """

    module_path = REPO_ROOT / "scripts" / "validate_prediction_rasters.py"
    spec = importlib.util.spec_from_file_location(
        "validate_prediction_rasters", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_raster(path: Path, array: np.ndarray, *, transform) -> None:
    """Write one small single-band GeoTIFF.

    Args:
        path (Path): Output GeoTIFF path.
        array (np.ndarray): Single-band raster values.
        transform: Raster transform used for the fixture.
    """

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=int(array.shape[0]),
        width=int(array.shape[1]),
        count=1,
        dtype=array.dtype,
        crs="EPSG:25832",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(array, 1)


def test_validate_prediction_rasters_scores_overlap_only(tmp_path: Path) -> None:
    """Validation should score only the overlapping label window.

    Args:
        tmp_path (Path): Temporary directory used for raster fixtures.
    """

    module = _load_validate_module()

    prediction = np.zeros((8, 8), dtype=np.uint8)
    prediction[1:5, 2:6] = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    label = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 1],
        ],
        dtype=np.uint8,
    )

    pred_path = tmp_path / "predictions_a.tif"
    label_path = tmp_path / "labels.tif"
    _write_raster(pred_path, prediction, transform=from_origin(0, 8, 1, 1))
    _write_raster(label_path, label, transform=from_origin(2, 7, 1, 1))

    report = module.validate_prediction_rasters(label_path, [str(pred_path)])
    result = report["results"][0]

    assert result["status"] == "ok"
    assert result["overlap_pixels"] == 16
    assert result["counts"]["tp"] == 5
    assert result["counts"]["fp"] == 0
    assert result["counts"]["fn"] == 2
    assert result["counts"]["tn"] == 9
    assert result["metrics"]["iou"] == 5 / 7
    assert result["metrics"]["dice"] == 10 / 12


def test_validate_prediction_rasters_reports_no_overlap(tmp_path: Path) -> None:
    """Validation should return a no-overlap status when rasters do not intersect.

    Args:
        tmp_path (Path): Temporary directory used for raster fixtures.
    """

    module = _load_validate_module()

    pred_path = tmp_path / "predictions_a.tif"
    label_path = tmp_path / "labels.tif"
    _write_raster(
        pred_path,
        np.zeros((4, 4), dtype=np.uint8),
        transform=from_origin(0, 4, 1, 1),
    )
    _write_raster(
        label_path,
        np.ones((4, 4), dtype=np.uint8),
        transform=from_origin(10, 14, 1, 1),
    )

    report = module.validate_prediction_rasters(label_path, [str(pred_path)])
    result = report["results"][0]

    assert result["status"] == "no_overlap"
    assert result["overlap_pixels"] == 0
    assert result["metrics"]["iou"] == 0.0


def test_resolve_prediction_paths_sorts_matches(tmp_path: Path) -> None:
    """Glob resolution should return sorted prediction files once each.

    Args:
        tmp_path (Path): Temporary directory used for glob fixtures.
    """

    module = _load_validate_module()

    for name in ("predictions_2.tif", "predictions_1.tif", "predictions_3.tif"):
        (tmp_path / name).write_bytes(b"x")

    resolved = module.resolve_prediction_paths([str(tmp_path / "predictions_*.tif")])

    assert [path.name for path in resolved] == [
        "predictions_1.tif",
        "predictions_2.tif",
        "predictions_3.tif",
    ]
