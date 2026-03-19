"""Tests for vector-label rasterization helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import fiona
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rasterize_vector_labels import (  # noqa: E402
    build_grid_spec_from_verify,
    collect_vector_paths,
    derive_output_path,
    derive_raster_output_path,
    derive_vector_output_path,
    measure_planet_coverage,
    rasterize_reference_labels,
    rasterize_vector_directory,
    run_configured_raster_merge,
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


def _write_test_raster(
    path: Path,
    data: np.ndarray,
    *,
    crs: str,
    transform,
) -> None:
    """Write one small single-band uint8 GeoTIFF for tests.

    Args:
        path (Path): Output GeoTIFF path.
        data (np.ndarray): Raster array written as band 1.
        crs (str): Raster CRS string.
        transform: Raster affine transform.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=int(data.shape[0]),
        width=int(data.shape[1]),
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data.astype(np.uint8), 1)


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


def test_derive_raster_output_path_preserves_relative_subdirectories() -> None:
    """Aligned raster outputs should mirror the input directory structure.

    Examples:
        >>> True
        True
    """

    merged = Path("labels/final_labels.tif")
    raster_root = Path("rasters")
    raster_path = raster_root / "nested" / "part.tif"

    output = derive_raster_output_path(merged, raster_path, raster_root)

    assert output.as_posix() == "labels/final_labels_rasters/nested/part_aligned.tif"


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


def test_build_grid_spec_from_verify_reprojects_and_snaps_bounds(
    tmp_path: Path,
) -> None:
    """Verification bounds should become a snapped 1 m target grid in UTM.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    verify_path = tmp_path / "verify.tif"
    original_bounds = (500000.0, 6000000.0, 500010.0, 6000010.0)
    verify_bounds = transform_bounds(
        "EPSG:25832",
        "EPSG:3857",
        *original_bounds,
        densify_pts=21,
    )
    verify_transform = from_origin(
        verify_bounds[0],
        verify_bounds[3],
        (verify_bounds[2] - verify_bounds[0]) / 5.0,
        (verify_bounds[3] - verify_bounds[1]) / 5.0,
    )
    _write_test_raster(
        verify_path,
        np.zeros((5, 5), dtype=np.uint8),
        crs="EPSG:3857",
        transform=verify_transform,
    )

    grid_spec = build_grid_spec_from_verify(verify_path, "EPSG:25832", 1.0)

    assert grid_spec.crs == rasterio.CRS.from_epsg(25832)
    assert grid_spec.transform.a == 1.0
    assert grid_spec.transform.e == -1.0
    assert grid_spec.bounds.left <= original_bounds[0]
    assert grid_spec.bounds.bottom <= original_bounds[1]
    assert grid_spec.bounds.right >= original_bounds[2]
    assert grid_spec.bounds.top >= original_bounds[3]
    assert grid_spec.bounds.left == round(grid_spec.bounds.left)
    assert grid_spec.bounds.bottom == round(grid_spec.bounds.bottom)
    assert grid_spec.bounds.right == round(grid_spec.bounds.right)
    assert grid_spec.bounds.top == round(grid_spec.bounds.top)


def test_run_configured_raster_merge_merges_rasters_and_vectors(
    tmp_path: Path,
) -> None:
    """Configured workflow should merge raster and shapefile labels on one grid.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    vector_dir = tmp_path / "vectors"
    raster_dir = tmp_path / "rasters"
    verify_path = tmp_path / "verify.tif"
    output_path = tmp_path / "final_labels.tif"
    config_path = tmp_path / "rasterize_labels.yml"

    _write_box_vector(vector_dir / "union.shp", bounds=(0.0, 0.0, 4.0, 8.0))
    _write_box_vector(
        vector_dir / "folder_2" / "union.shp", bounds=(4.0, 0.0, 8.0, 4.0)
    )
    _write_test_raster(
        verify_path,
        np.ones((4, 4), dtype=np.uint8),
        crs="EPSG:25832",
        transform=from_origin(0.0, 8.0, 2.0, 2.0),
    )

    raster_top_right = np.zeros((16, 16), dtype=np.uint8)
    raster_top_right[:8, 8:] = 1
    raster_extra = np.zeros((16, 16), dtype=np.uint8)
    raster_extra[:4, 12:] = 1
    _write_test_raster(
        raster_dir / "union_folder3.tif",
        raster_top_right,
        crs="EPSG:25832",
        transform=from_origin(0.0, 8.0, 0.5, 0.5),
    )
    _write_test_raster(
        raster_dir / "union_folder4.tif",
        raster_extra,
        crs="EPSG:25832",
        transform=from_origin(0.0, 8.0, 0.5, 0.5),
    )

    config_path.write_text(
        yaml.safe_dump(
            {
                "logging": {"level": "INFO"},
                "workflow": {
                    "vector_dir": str(vector_dir),
                    "vector_glob": "*.shp",
                    "raster_dir": str(raster_dir),
                    "merge_raster_glob": "union_folder*.tif",
                    "verify_path": str(verify_path),
                    "output_path": str(output_path),
                    "target_crs": "EPSG:25832",
                    "target_resolution": 1.0,
                    "min_planet_coverage": 0.8,
                    "overwrite": True,
                    "vector_workers": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_configured_raster_merge(yaml.safe_load(config_path.read_text()))

    assert result["output_path"] == output_path
    assert len(result["vector_parts"]) == 2
    assert len(result["raster_parts"]) == 2
    assert result["vector_merged_path"].exists()
    assert result["raster_merged_path"].exists()
    assert output_path.exists()

    with rasterio.open(output_path) as src:
        data = src.read(1)
        assert src.crs == rasterio.CRS.from_epsg(25832)
        assert src.transform == from_origin(0.0, 8.0, 1.0, 1.0)
        assert data.tolist() == np.ones((8, 8), dtype=np.uint8).tolist()

    metrics = measure_planet_coverage(output_path, verify_path)
    assert metrics["coverage"] == pytest.approx(1.0)
    assert result["metrics"]["coverage"] == pytest.approx(1.0)


def test_run_configured_raster_merge_fails_below_planet_threshold(
    tmp_path: Path,
) -> None:
    """Configured workflow should fail when Planet coverage is too low.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    vector_dir = tmp_path / "vectors"
    raster_dir = tmp_path / "rasters"
    verify_path = tmp_path / "verify.tif"
    output_path = tmp_path / "final_labels.tif"

    _write_box_vector(vector_dir / "union.shp", bounds=(0.0, 0.0, 2.0, 2.0))
    _write_test_raster(
        verify_path,
        np.ones((4, 4), dtype=np.uint8),
        crs="EPSG:25832",
        transform=from_origin(0.0, 8.0, 2.0, 2.0),
    )
    _write_test_raster(
        raster_dir / "union_folder3.tif",
        np.zeros((16, 16), dtype=np.uint8),
        crs="EPSG:25832",
        transform=from_origin(0.0, 8.0, 0.5, 0.5),
    )
    _write_test_raster(
        raster_dir / "union_folder4.tif",
        np.zeros((16, 16), dtype=np.uint8),
        crs="EPSG:25832",
        transform=from_origin(0.0, 8.0, 0.5, 0.5),
    )

    with pytest.raises(ValueError, match="minimum Planet coverage"):
        run_configured_raster_merge(
            {
                "workflow": {
                    "vector_dir": str(vector_dir),
                    "vector_glob": "*.shp",
                    "raster_dir": str(raster_dir),
                    "merge_raster_glob": "union_folder*.tif",
                    "verify_path": str(verify_path),
                    "output_path": str(output_path),
                    "target_crs": "EPSG:25832",
                    "target_resolution": 1.0,
                    "min_planet_coverage": 0.8,
                    "overwrite": True,
                }
            }
        )
