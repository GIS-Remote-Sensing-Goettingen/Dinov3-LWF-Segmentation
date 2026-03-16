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
    """Write a small EPSG:25832 polygon shapefile for tests.

    The geometry is centered on the reference raster so rasterization can be
    checked against a small expected binary mask.

    Args:
        path (Path): Shapefile path to create.
    """

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
    """Write a simple 4x4 reference GeoTIFF for tests.

    The raster uses 2-meter pixels in `EPSG:25832` so the polygon test fixture
    lands on the middle 2x2 block.

    Args:
        path (Path): GeoTIFF path to create.
    """

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
    """Prediction stems should normalize to `_labels.tif`.

    This keeps generated mask names aligned with the repository naming
    convention used by existing test labels.
    """

    ref = Path("dop20_596000_5973000_1km_20cm_pred.tif")
    output = derive_output_path(ref, Path("labels"))
    assert output.name == "dop20_596000_5973000_1km_20cm_labels.tif"


def test_derive_output_path_preserves_existing_labels_suffix() -> None:
    """Existing label stems should not get a duplicate `_labels` suffix.

    This keeps replacement outputs aligned with the current scene naming
    convention when old label TIFFs are used as references.
    """

    ref = Path("dop20_592000_5975000_1km_20cm_labels.tif")
    output = derive_output_path(ref, Path("labels"))
    assert output.name == "dop20_592000_5975000_1km_20cm_labels.tif"


def test_rasterize_reference_labels_burns_binary_mask(tmp_path: Path) -> None:
    """Rasterization should align polygon coverage to the reference grid.

    The output should preserve CRS and transform metadata while burning the
    polygon into the expected binary mask footprint.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

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


def test_rasterize_reference_labels_supports_windowed_output(tmp_path: Path) -> None:
    """Windowed rasterization should match the expected binary mask.

    This exercises the low-memory streaming path used for large reference
    rasters.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    vector_path = tmp_path / "union.shp"
    reference_path = tmp_path / "scene.tif"
    output_path = tmp_path / "scene_labels_windowed.tif"
    _write_test_vector(vector_path)
    _write_test_reference(reference_path)

    feature_count = rasterize_reference_labels(
        vector_path=vector_path,
        reference_path=reference_path,
        output_path=output_path,
        window_size=2,
    )

    with rasterio.open(output_path) as src:
        data = src.read(1)

    assert feature_count == 1
    assert data.tolist() == [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]


def test_rasterize_reference_labels_supports_threaded_windows(
    tmp_path: Path,
) -> None:
    """Threaded window rasterization should preserve the expected mask.

    This exercises the concurrent window path used to speed up large
    rasterization jobs.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    vector_path = tmp_path / "union.shp"
    reference_path = tmp_path / "scene.tif"
    output_path = tmp_path / "scene_labels_threaded.tif"
    _write_test_vector(vector_path)
    _write_test_reference(reference_path)

    feature_count = rasterize_reference_labels(
        vector_path=vector_path,
        reference_path=reference_path,
        output_path=output_path,
        window_size=2,
        workers=2,
    )

    with rasterio.open(output_path) as src:
        data = src.read(1)

    assert feature_count == 1
    assert data.tolist() == [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
