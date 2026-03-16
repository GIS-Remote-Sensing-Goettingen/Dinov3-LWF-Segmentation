"""Tests for vector-label rasterization helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import fiona
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rasterize_vector_labels import (  # noqa: E402
    collect_vector_paths,
    derive_output_path,
    derive_vector_output_path,
    rasterize_reference_labels,
    rasterize_vector_directory,
)


def _write_box_vector(
    path: Path,
    bounds: tuple[float, float, float, float],
) -> None:
    """Write one small EPSG:25832 polygon shapefile for tests.

    Args:
        path (Path): Shapefile path to create.
        bounds (tuple[float, float, float, float]): Polygon bounds in
            `(minx, miny, maxx, maxy)` order.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = bounds
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
                "geometry": mapping(box(minx, miny, maxx, maxy)),
                "properties": {"id": 1},
            }
        )


def _write_test_vector(path: Path) -> None:
    """Write a centered EPSG:25832 polygon shapefile for tests.

    The geometry is centered on the reference raster so rasterization can be
    checked against a small expected binary mask.

    Args:
        path (Path): Shapefile path to create.
    """

    _write_box_vector(path, bounds=(2.0, 2.0, 6.0, 6.0))


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


def test_collect_vector_paths_recurses_into_nested_directories(
    tmp_path: Path,
) -> None:
    """Recursive vector discovery should include nested shapefiles.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    first = tmp_path / "first.shp"
    second = tmp_path / "nested" / "second.shp"
    _write_box_vector(first, bounds=(0.0, 0.0, 1.0, 1.0))
    _write_box_vector(second, bounds=(1.0, 1.0, 2.0, 2.0))

    matches = collect_vector_paths(tmp_path, "*.shp")

    assert matches == [first, second]


def test_derive_vector_output_path_preserves_relative_subdirectories() -> None:
    """Per-vector outputs should mirror the input directory structure.

    Examples:
        >>> True
        True
    """

    merged = Path("labels/scene_labels.tif")
    vector_root = Path("vectors")
    vector_path = vector_root / "nested" / "part.shp"

    output = derive_vector_output_path(merged, vector_path, vector_root)

    assert output.as_posix() == "labels/scene_labels_parts/nested/part_labels.tif"


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


def test_rasterize_reference_labels_supports_higher_resolution_output(
    tmp_path: Path,
) -> None:
    """Higher resolution output should preserve extent while densifying pixels.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    vector_path = tmp_path / "union.shp"
    reference_path = tmp_path / "scene.tif"
    output_path = tmp_path / "scene_labels_2x.tif"
    _write_test_vector(vector_path)
    _write_test_reference(reference_path)

    feature_count = rasterize_reference_labels(
        vector_path=vector_path,
        reference_path=reference_path,
        output_path=output_path,
        resolution_factor=2,
    )

    with rasterio.open(reference_path) as reference, rasterio.open(output_path) as src:
        data = src.read(1)
        assert src.crs == reference.crs
        assert src.bounds == reference.bounds
        assert src.width == reference.width * 2
        assert src.height == reference.height * 2
        assert src.transform == from_origin(0.0, 8.0, 1.0, 1.0)
        assert data.dtype == np.uint8

    assert feature_count == 1
    assert data.tolist() == [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]


def test_rasterize_reference_labels_windowed_high_resolution_matches_full(
    tmp_path: Path,
) -> None:
    """Windowed rasterization should match single-pass higher resolution output.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    vector_path = tmp_path / "union.shp"
    reference_path = tmp_path / "scene.tif"
    full_output_path = tmp_path / "scene_labels_full_2x.tif"
    windowed_output_path = tmp_path / "scene_labels_windowed_2x.tif"
    _write_test_vector(vector_path)
    _write_test_reference(reference_path)

    rasterize_reference_labels(
        vector_path=vector_path,
        reference_path=reference_path,
        output_path=full_output_path,
        resolution_factor=2,
    )
    rasterize_reference_labels(
        vector_path=vector_path,
        reference_path=reference_path,
        output_path=windowed_output_path,
        window_size=3,
        resolution_factor=2,
    )

    with rasterio.open(full_output_path) as full_src:
        full_data = full_src.read(1)
        full_bounds = full_src.bounds
        full_transform = full_src.transform
    with rasterio.open(windowed_output_path) as windowed_src:
        windowed_data = windowed_src.read(1)
        assert windowed_src.bounds == full_bounds
        assert windowed_src.transform == full_transform

    assert windowed_data.tolist() == full_data.tolist()


def test_rasterize_reference_labels_rejects_invalid_resolution_factor(
    tmp_path: Path,
) -> None:
    """Resolution factor must be a positive integer.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    vector_path = tmp_path / "union.shp"
    reference_path = tmp_path / "scene.tif"
    output_path = tmp_path / "scene_labels_invalid.tif"
    _write_test_vector(vector_path)
    _write_test_reference(reference_path)

    with pytest.raises(ValueError, match="resolution_factor must be >= 1"):
        rasterize_reference_labels(
            vector_path=vector_path,
            reference_path=reference_path,
            output_path=output_path,
            resolution_factor=0,
        )


def test_rasterize_vector_directory_merges_nested_shapefiles(
    tmp_path: Path,
) -> None:
    """Directory rasterization should emit per-shape TIFFs and one merged TIFF.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    vector_dir = tmp_path / "vectors"
    first_vector = vector_dir / "left.shp"
    second_vector = vector_dir / "nested" / "right.shp"
    reference_path = tmp_path / "scene.tif"
    output_dir = tmp_path / "labels"
    _write_box_vector(first_vector, bounds=(0.0, 4.0, 2.0, 6.0))
    _write_box_vector(second_vector, bounds=(6.0, 0.0, 8.0, 2.0))
    _write_test_reference(reference_path)

    merged_output_path, individual_paths = rasterize_vector_directory(
        vector_dir=vector_dir,
        reference_path=reference_path,
        output_path=output_dir,
        vector_workers=2,
    )

    assert merged_output_path == output_dir / "scene_labels.tif"
    assert individual_paths == [
        output_dir / "scene_labels_parts" / "left_labels.tif",
        output_dir / "scene_labels_parts" / "nested" / "right_labels.tif",
    ]
    for path in individual_paths:
        assert path.exists()

    with rasterio.open(individual_paths[0]) as first_src:
        first_data = first_src.read(1)
    with rasterio.open(individual_paths[1]) as second_src:
        second_data = second_src.read(1)
    with rasterio.open(reference_path) as reference_src:
        reference_bounds = reference_src.bounds
    with rasterio.open(merged_output_path) as merged_src:
        merged_data = merged_src.read(1)
        assert merged_src.bounds == reference_bounds

    assert first_data.tolist() == [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert second_data.tolist() == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]
    assert merged_data.tolist() == [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]
