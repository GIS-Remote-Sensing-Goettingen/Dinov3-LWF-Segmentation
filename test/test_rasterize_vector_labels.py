"""Tests for vector-label rasterization helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rasterize_vector_labels import (  # noqa: E402
    derive_output_path,
    rasterize_reference_labels,
)


def _write_test_vector(path: Path) -> None:
    """Write a small EPSG:25832 polygon shapefile for tests."""

    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    with fiona.open(
        path,
        "w",
        driver="ESRI Shapefile",
        schema=schema,
        crs="EPSG:25832",
    ) as dst:
        dst.write(
            {
                "geometry": mapping(box(2.0, 2.0, 6.0, 6.0)),
                "properties": {"id": 1},
            }
        )


def _write_test_reference(path: Path) -> None:
    """Write a simple 4x4 reference GeoTIFF for tests."""

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="uint8",
        crs="EPSG:25832",
        transform=from_origin(0.0, 8.0, 2.0, 2.0),
    ) as dst:
        dst.write(np.zeros((4, 4), dtype=np.uint8), 1)


def test_derive_output_path_replaces_prediction_suffix() -> None:
    """Prediction-stem outputs should normalize to `_labels.tif`."""

    ref = Path("dop20_596000_5973000_1km_20cm_pred.tif")
    output = derive_output_path(ref, Path("labels"))
    assert output.name == "dop20_596000_5973000_1km_20cm_labels.tif"


def test_rasterize_reference_labels_burns_binary_mask(tmp_path: Path) -> None:
    """Rasterization should align polygon coverage to the reference grid."""

    vector_path = tmp_path / "union.shp"
    reference_path = tmp_path / "scene.tif"
    output_path = tmp_path / "scene_labels.tif"
    _write_test_vector(vector_path)
    _write_test_reference(reference_path)

    feature_count = rasterize_reference_labels(
        vector_path=vector_path,
        reference_path=reference_path,
        output_path=output_path,
    )

    with rasterio.open(output_path) as src:
        data = src.read(1)
        assert src.crs == rasterio.CRS.from_epsg(25832)
        assert src.transform == from_origin(0.0, 8.0, 2.0, 2.0)
        assert data.dtype == np.uint8

    assert feature_count == 1
    assert data.tolist() == [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
