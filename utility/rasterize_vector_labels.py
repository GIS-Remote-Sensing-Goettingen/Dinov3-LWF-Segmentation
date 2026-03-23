"""Rasterize vector labels onto reference GeoTIFF grids.

This utility is meant for cases where labels arrive as polygons (for example a
shapefile) but the pipeline expects raster labels aligned to image GeoTIFFs.
It can rasterize one reference TIFF or every TIFF in a directory.
"""

from __future__ import annotations

import argparse
import logging
import math
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import fiona
import numpy as np
import rasterio
import yaml
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds, transform_geom
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

LOGGER = logging.getLogger(__name__)
DEFAULT_RASTERIZE_CONFIG_PATH = Path("configs/rasterize_labels.yml")


@dataclass(frozen=True)
class GridSpec:
    """Explicit raster grid definition used for aligned label outputs.

    Attributes:
        crs (CRS): Output CRS.
        transform (Affine): Affine transform of the output grid.
        width (int): Output width in pixels.
        height (int): Output height in pixels.
    """

    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def bounds(self) -> BoundingBox:
        """Return the geographic bounds of the grid.

        Returns:
            BoundingBox: Bounds in `(left, bottom, right, top)` order.
        """

        left = float(self.transform.c)
        top = float(self.transform.f)
        right = left + float(self.width) * float(self.transform.a)
        bottom = top + float(self.height) * float(self.transform.e)
        return BoundingBox(left=left, bottom=bottom, right=right, top=top)


def collect_reference_paths(reference_path: Path, glob_pattern: str) -> list[Path]:
    """Return one or more reference TIFF paths.

    Args:
        reference_path (Path): Reference TIFF path or directory.
        glob_pattern (str): Glob applied when `reference_path` is a directory.

    Returns:
        list[Path]: Sorted TIFF paths.

    Raises:
        FileNotFoundError: If no matching references are found.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     root = Path(d)
        ...     ref = root / "scene.tif"
        ...     _ = ref.write_bytes(b"")
        ...     [p.name for p in collect_reference_paths(root, "*.tif")]
        ['scene.tif']
    """

    if reference_path.is_file():
        return [reference_path]
    matches = sorted(
        path for path in reference_path.glob(glob_pattern) if path.is_file()
    )
    if matches:
        return matches
    raise FileNotFoundError(
        f"no reference GeoTIFFs found under {reference_path} with glob {glob_pattern!r}"
    )


def collect_vector_paths(vector_path: Path, glob_pattern: str) -> list[Path]:
    """Return one or more vector label paths.

    When `vector_path` is a directory, shapefiles are discovered recursively so
    nested label folders can be rasterized in one batch.

    Args:
        vector_path (Path): Vector label path or directory of vector files.
        glob_pattern (str): Recursive glob applied when `vector_path` is a
            directory.

    Returns:
        list[Path]: Sorted vector paths.

    Raises:
        FileNotFoundError: If no matching vector files are found.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     root = Path(d)
        ...     nested = root / "nested"
        ...     nested.mkdir()
        ...     shp = nested / "scene.shp"
        ...     _ = shp.write_bytes(b"")
        ...     [p.name for p in collect_vector_paths(root, "*.shp")]
        ['scene.shp']
    """

    if vector_path.is_file():
        return [vector_path]
    matches = sorted(path for path in vector_path.rglob(glob_pattern) if path.is_file())
    if matches:
        return matches
    raise FileNotFoundError(
        f"no vector label files found under {vector_path} with glob {glob_pattern!r}"
    )


def collect_raster_paths(raster_path: Path, glob_pattern: str) -> list[Path]:
    """Return one or more raster label TIFF paths.

    Args:
        raster_path (Path): Raster label path or directory.
        glob_pattern (str): Recursive glob applied when `raster_path` is a
            directory.

    Returns:
        list[Path]: Sorted raster TIFF paths.

    Raises:
        FileNotFoundError: If no matching raster files are found.
    """

    if raster_path.is_file():
        return [raster_path]
    matches = sorted(path for path in raster_path.rglob(glob_pattern) if path.is_file())
    if matches:
        return matches
    raise FileNotFoundError(
        f"no raster label TIFFs found under {raster_path} with glob {glob_pattern!r}"
    )


def derive_output_path(reference_path: Path, output_path: Path) -> Path:
    """Return the label TIFF path for a reference raster.

    When `output_path` is a directory, this function derives a label filename
    from the reference stem. Prediction/image suffixes are normalized to
    `_labels`, while an existing `_labels` suffix is preserved.

    Args:
        reference_path (Path): Source raster used as the alignment grid.
        output_path (Path): Output file or directory path.

    Returns:
        Path: Target label TIFF path.

    Examples:
        >>> ref = Path("dop20_596000_5973000_1km_20cm_pred.tif")
        >>> derive_output_path(ref, Path("labels")).name
        'dop20_596000_5973000_1km_20cm_labels.tif'
        >>> ref = Path("dop20_592000_5975000_1km_20cm_labels.tif")
        >>> derive_output_path(ref, Path("labels")).name
        'dop20_592000_5975000_1km_20cm_labels.tif'
        >>> derive_output_path(Path("scene.tif"), Path("labels")).name
        'scene_labels.tif'
    """

    if output_path.suffix.lower() in {".tif", ".tiff"}:
        return output_path
    stem = reference_path.stem
    for suffix in ("_pred", "_image", "_img", "_labels"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return output_path / f"{stem}_labels.tif"


def derive_vector_output_path(
    merged_output_path: Path,
    vector_path: Path,
    vector_root: Path,
) -> Path:
    """Return the per-vector raster output path under the merged output folder.

    Args:
        merged_output_path (Path): Final merged TIFF path.
        vector_path (Path): Source vector file path.
        vector_root (Path): Root directory used for recursive discovery.

    Returns:
        Path: Per-vector TIFF path inside the sibling `_parts` folder.

    Examples:
        >>> merged = Path("labels/scene_labels.tif")
        >>> root = Path("vectors")
        >>> shp = root / "nested" / "part.shp"
        >>> derive_vector_output_path(merged, shp, root).as_posix()
        'labels/scene_labels_parts/nested/part_labels.tif'
    """

    relative_path = vector_path.relative_to(vector_root)
    parts_root = merged_output_path.parent / f"{merged_output_path.stem}_parts"
    return parts_root / relative_path.parent / f"{vector_path.stem}_labels.tif"


def derive_raster_output_path(
    merged_output_path: Path,
    raster_path: Path,
    raster_root: Path,
) -> Path:
    """Return the aligned raster output path under the merged output folder.

    Args:
        merged_output_path (Path): Final merged TIFF path.
        raster_path (Path): Source raster label TIFF path.
        raster_root (Path): Root directory used for recursive discovery.

    Returns:
        Path: Per-raster aligned TIFF path inside the sibling `_rasters` folder.
    """

    relative_path = raster_path.relative_to(raster_root)
    parts_root = merged_output_path.parent / f"{merged_output_path.stem}_rasters"
    return parts_root / relative_path.parent / f"{raster_path.stem}_aligned.tif"


def _normalize_crs(crs_value: str | CRS | dict[str, str] | None) -> CRS | None:
    """Convert CRS-like input into a rasterio CRS.

    This accepts the CRS forms commonly returned by Fiona and normalizes them
    into one `rasterio.crs.CRS` instance for comparisons and reprojection.

    Args:
        crs_value (str | CRS | dict[str, str] | None): CRS-like input value.

    Returns:
        CRS | None: Normalized CRS object when input is present.

    Examples:
        >>> str(_normalize_crs("EPSG:25832"))
        'EPSG:25832'
        >>> _normalize_crs(None) is None
        True
    """

    if not crs_value:
        return None
    return CRS.from_user_input(crs_value)


def _resolve_vector_crs(
    src: fiona.Collection,
    vector_crs_override: str,
) -> CRS | None:
    """Resolve the effective CRS for the input vector layer.

    This prefers an explicit CLI override and otherwise falls back to Fiona's
    CRS metadata representations.

    Args:
        src (fiona.Collection): Open vector dataset.
        vector_crs_override (str): Optional CLI CRS override.

    Returns:
        CRS | None: Effective vector CRS, if it can be determined.
    """

    vector_crs = _normalize_crs(vector_crs_override) or _normalize_crs(src.crs_wkt)
    if vector_crs is None:
        vector_crs = _normalize_crs(src.crs)
    return vector_crs


def load_rasterize_config(config_path: Path) -> dict[str, Any]:
    """Load one YAML config file for the raster merge workflow.

    Args:
        config_path (Path): YAML configuration path.

    Returns:
        dict[str, Any]: Parsed configuration mapping.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML file does not contain a mapping.
    """

    resolved_path = Path(config_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"rasterize config not found: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"rasterize config must define a mapping at the top level: {resolved_path}"
        )
    return data


def _config_value(
    config: dict[str, Any],
    key: str,
    default: Any = None,
    *,
    required: bool = False,
) -> Any:
    """Read one workflow key from the config mapping.

    Args:
        config (dict[str, Any]): Workflow config mapping.
        key (str): Key name.
        default (Any): Default value when key is absent.
        required (bool): Whether the key must be present.

    Returns:
        Any: Resolved config value.

    Raises:
        ValueError: If the key is required but missing.
    """

    workflow = config.get("workflow", {})
    if key in workflow:
        return workflow[key]
    if key in config:
        return config[key]
    if required:
        raise ValueError(f"missing required rasterize config value: {key}")
    return default


def build_grid_spec_from_verify(
    verify_path: Path,
    target_crs: str | CRS,
    target_resolution: float,
) -> GridSpec:
    """Build a snapped output grid from the verification raster footprint.

    Args:
        verify_path (Path): Verification raster used to define the footprint.
        target_crs (str | CRS): Requested output CRS.
        target_resolution (float): Requested output pixel size.

    Returns:
        GridSpec: Canonical output grid.
    """

    resolution = float(target_resolution)
    if resolution <= 0:
        raise ValueError("target_resolution must be > 0")
    target_crs_obj = CRS.from_user_input(target_crs)
    with rasterio.open(verify_path) as verify:
        verify_crs = verify.crs
        if verify_crs is None:
            raise ValueError(f"verification raster has no CRS: {verify_path}")
        left, bottom, right, top = verify.bounds
        if verify_crs != target_crs_obj:
            left, bottom, right, top = transform_bounds(
                verify_crs,
                target_crs_obj,
                left,
                bottom,
                right,
                top,
                densify_pts=21,
            )
    snapped_left = math.floor(left / resolution) * resolution
    snapped_bottom = math.floor(bottom / resolution) * resolution
    snapped_right = math.ceil(right / resolution) * resolution
    snapped_top = math.ceil(top / resolution) * resolution
    width = int(round((snapped_right - snapped_left) / resolution))
    height = int(round((snapped_top - snapped_bottom) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError("target grid must have positive width and height")
    return GridSpec(
        crs=target_crs_obj,
        transform=from_origin(snapped_left, snapped_top, resolution, resolution),
        width=width,
        height=height,
    )


def _build_grid_profile(
    grid_spec: GridSpec,
    dtype: str,
    fill_value: int,
    compress: str,
) -> dict[str, object]:
    """Build a GeoTIFF profile for one explicit output grid.

    Args:
        grid_spec (GridSpec): Output grid definition.
        dtype (str): Output raster dtype.
        fill_value (int): Background fill value.
        compress (str): GeoTIFF compression codec.

    Returns:
        dict[str, object]: Profile used to create the output raster.
    """

    return {
        "driver": "GTiff",
        "width": int(grid_spec.width),
        "height": int(grid_spec.height),
        "count": 1,
        "dtype": dtype,
        "crs": grid_spec.crs,
        "transform": grid_spec.transform,
        "nodata": fill_value,
        "compress": compress,
    }


def _iter_burn_shapes(
    src: fiona.Collection,
    reference_bounds: tuple[float, float, float, float],
    reference_crs: CRS | None,
    vector_crs: CRS | None,
    burn_value: int,
    seen_feature_ids: set[str] | None = None,
) -> tuple[Iterator[tuple[dict[str, object], int]], list[int]]:
    """Yield rasterize-ready shapes for the reference extent.

    Args:
        src (fiona.Collection): Open vector label dataset.
        reference_bounds (tuple[float, float, float, float]): Target bounds in
            reference CRS order `(left, bottom, right, top)`.
        reference_crs (CRS | None): Reference raster CRS.
        vector_crs (CRS | None): Vector CRS.
        burn_value (int): Label value written for each polygon.
        seen_feature_ids (set[str] | None): Optional set used to count unique
            features across multiple windows while still yielding duplicates as
            needed for rasterization.

    Returns:
        tuple[Iterator[tuple[dict[str, object], int]], list[int]]: Shape
        iterator and mutable one-item list storing the yielded feature count.
    """

    count = [0]

    def iterator() -> Iterator[tuple[dict[str, object], int]]:
        """Yield geometries intersecting the current reference window.

        The iterator keeps the Fiona dataset open only while rasterization
        consumes shapes, which avoids loading intersecting polygons into memory.
        """

        query_bounds = reference_bounds
        if reference_crs and vector_crs and reference_crs != vector_crs:
            query_bounds = transform_bounds(
                reference_crs,
                vector_crs,
                *reference_bounds,
                densify_pts=21,
            )
        features = src.filter(bbox=query_bounds)
        for feature in features:
            geometry = feature.get("geometry")
            if not geometry:
                continue
            feature_id = str(feature.get("id", ""))
            if seen_feature_ids is None or feature_id not in seen_feature_ids:
                count[0] += 1
                if seen_feature_ids is not None and feature_id:
                    seen_feature_ids.add(feature_id)
            if reference_crs and vector_crs and reference_crs != vector_crs:
                geometry = transform_geom(
                    vector_crs,
                    reference_crs,
                    geometry,
                )
            yield geometry, burn_value

    return iterator(), count


def _iter_windows(height: int, width: int, window_size: int) -> Iterator[Window]:
    """Yield raster windows covering the reference raster.

    Args:
        height (int): Raster height in pixels.
        width (int): Raster width in pixels.
        window_size (int): Maximum window edge length in pixels.

    Returns:
        Iterator[Window]: Raster windows in row-major order.
    """

    for row_off in range(0, height, window_size):
        row_size = min(window_size, height - row_off)
        for col_off in range(0, width, window_size):
            col_size = min(window_size, width - col_off)
            yield Window(
                col_off=col_off, row_off=row_off, width=col_size, height=row_size
            )


def _build_output_profile(
    reference: rasterio.DatasetReader,
    dtype: str,
    fill_value: int,
    compress: str,
    width: int | None = None,
    height: int | None = None,
    transform: Affine | None = None,
) -> dict[str, object]:
    """Build the output GeoTIFF profile for rasterized labels.

    Args:
        reference (rasterio.DatasetReader): Reference raster dataset.
        dtype (str): Output raster dtype.
        fill_value (int): Background fill value.
        compress (str): GeoTIFF compression codec.
        width (int | None): Optional output raster width in pixels.
        height (int | None): Optional output raster height in pixels.
        transform (Affine | None): Optional output affine transform.

    Returns:
        dict[str, object]: Profile used to create the output label raster.
    """

    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=fill_value,
        compress=compress,
    )
    if width is not None:
        profile["width"] = int(width)
    if height is not None:
        profile["height"] = int(height)
    if transform is not None:
        profile["transform"] = transform
    return profile


def _resolve_output_grid(
    reference: rasterio.DatasetReader,
    resolution_factor: int,
) -> tuple[tuple[int, int], Affine]:
    """Return the raster shape and transform for the output label grid.

    Args:
        reference (rasterio.DatasetReader): Reference raster dataset.
        resolution_factor (int): Integer multiplier for raster density.

    Returns:
        tuple[tuple[int, int], Affine]: Output shape as `(height, width)` and
        output transform preserving the reference extent.
    """

    factor = int(resolution_factor)
    if factor < 1:
        raise ValueError("resolution_factor must be >= 1")
    output_shape = (int(reference.height) * factor, int(reference.width) * factor)
    output_transform = reference.transform * Affine.scale(1 / factor, 1 / factor)
    return output_shape, output_transform


def _rasterize_full_reference(
    src: fiona.Collection,
    reference: rasterio.DatasetReader,
    output_path: Path,
    vector_crs: CRS | None,
    burn_value: int,
    fill_value: int,
    dtype: str,
    all_touched: bool,
    compress: str,
    output_shape: tuple[int, int],
    output_transform: Affine,
) -> int:
    """Rasterize one full reference raster in a single pass.

    Args:
        src (fiona.Collection): Open vector label dataset.
        reference (rasterio.DatasetReader): Reference raster dataset.
        output_path (Path): Output GeoTIFF path.
        vector_crs (CRS | None): Vector CRS.
        burn_value (int): Value burned into polygon pixels.
        fill_value (int): Background fill value.
        dtype (str): Output raster dtype.
        all_touched (bool): Whether to burn every touched pixel.
        compress (str): GeoTIFF compression codec.
        output_shape (tuple[int, int]): Output raster shape `(height, width)`.
        output_transform (Affine): Output affine transform.

    Returns:
        int: Number of intersecting vector features.
    """

    LOGGER.info(
        "Rasterizing %s in single-pass mode (%dx%d pixels)",
        reference.name,
        reference.width,
        reference.height,
    )
    shapes, feature_count = _iter_burn_shapes(
        src=src,
        reference_bounds=reference.bounds,
        reference_crs=reference.crs,
        vector_crs=vector_crs,
        burn_value=burn_value,
    )
    label_array = rasterize(
        shapes=shapes,
        out_shape=output_shape,
        transform=output_transform,
        fill=fill_value,
        dtype=dtype,
        all_touched=all_touched,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        **_build_output_profile(
            reference,
            dtype,
            fill_value,
            compress,
            width=output_shape[1],
            height=output_shape[0],
            transform=output_transform,
        ),
    ) as dst:
        dst.write(label_array, 1)
    LOGGER.info(
        "Finished %s with %d intersecting features",
        output_path,
        feature_count[0],
    )
    return feature_count[0]


def _rasterize_windowed_reference(
    src: fiona.Collection,
    reference: rasterio.DatasetReader,
    output_path: Path,
    vector_crs: CRS | None,
    burn_value: int,
    fill_value: int,
    dtype: str,
    all_touched: bool,
    compress: str,
    window_size: int,
    vector_path: Path,
    workers: int,
    output_shape: tuple[int, int],
    output_transform: Affine,
) -> int:
    """Rasterize a reference raster incrementally by windows.

    This path trades some repeated bbox queries for much lower peak memory and
    is meant for very large reference rasters.

    Args:
        src (fiona.Collection): Open vector label dataset.
        reference (rasterio.DatasetReader): Reference raster dataset.
        output_path (Path): Output GeoTIFF path.
        vector_crs (CRS | None): Vector CRS.
        burn_value (int): Value burned into polygon pixels.
        fill_value (int): Background fill value.
        dtype (str): Output raster dtype.
        all_touched (bool): Whether to burn every touched pixel.
        compress (str): GeoTIFF compression codec.
        window_size (int): Window edge length in pixels.
        vector_path (Path): Vector label path used by worker threads.
        workers (int): Number of worker threads for window rasterization.
        output_shape (tuple[int, int]): Output raster shape `(height, width)`.
        output_transform (Affine): Output affine transform.

    Returns:
        int: Number of unique intersecting vector features.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_feature_ids: set[str] = set()
    output_height, output_width = output_shape
    total_windows = ((output_height + window_size - 1) // window_size) * (
        (output_width + window_size - 1) // window_size
    )
    progress_step = max(1, total_windows // 20)
    LOGGER.info(
        "Rasterizing %s in windowed mode (%dx%d pixels, %d windows of %d px)",
        reference.name,
        reference.width,
        reference.height,
        total_windows,
        window_size,
    )
    with rasterio.open(
        output_path,
        "w",
        **_build_output_profile(
            reference,
            dtype,
            fill_value,
            compress,
            width=output_width,
            height=output_height,
            transform=output_transform,
        ),
    ) as dst:
        if workers <= 1:
            for index, window in enumerate(
                _iter_windows(output_height, output_width, window_size),
                start=1,
            ):
                bounds = window_bounds(window, output_transform)
                shapes, _ = _iter_burn_shapes(
                    src=src,
                    reference_bounds=bounds,
                    reference_crs=reference.crs,
                    vector_crs=vector_crs,
                    burn_value=burn_value,
                    seen_feature_ids=seen_feature_ids,
                )
                window_array = rasterize(
                    shapes=shapes,
                    out_shape=(int(window.height), int(window.width)),
                    transform=window_transform(window, output_transform),
                    fill=fill_value,
                    dtype=dtype,
                    all_touched=all_touched,
                )
                dst.write(window_array, 1, window=window)
                if index == 1 or index == total_windows or index % progress_step == 0:
                    LOGGER.info(
                        "Window progress for %s: %d/%d (%.1f%%)",
                        output_path,
                        index,
                        total_windows,
                        100.0 * index / total_windows,
                    )
        else:
            LOGGER.info("Using %d worker threads for window rasterization", workers)
            windows_iter = iter(_iter_windows(output_height, output_width, window_size))
            completed = 0
            max_inflight = max(workers * 2, 1)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                inflight: dict[Future[tuple[Window, np.ndarray, set[str]]], Window] = {}

                def submit_next() -> bool:
                    """Submit one more window task if work remains.

                    Returns:
                        bool: True when a task was submitted.
                    """

                    try:
                        next_window = next(windows_iter)
                    except StopIteration:
                        return False
                    future = executor.submit(
                        _rasterize_window_task,
                        vector_path=vector_path,
                        window=next_window,
                        output_transform=output_transform,
                        reference_crs=reference.crs,
                        vector_crs=vector_crs,
                        burn_value=burn_value,
                        fill_value=fill_value,
                        dtype=dtype,
                        all_touched=all_touched,
                    )
                    inflight[future] = next_window
                    return True

                while len(inflight) < max_inflight and submit_next():
                    pass
                while inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for future in done:
                        window, window_array, feature_ids = future.result()
                        inflight.pop(future)
                        seen_feature_ids.update(feature_ids)
                        dst.write(window_array, 1, window=window)
                        completed += 1
                        if (
                            completed == 1
                            or completed == total_windows
                            or completed % progress_step == 0
                        ):
                            LOGGER.info(
                                "Window progress for %s: %d/%d (%.1f%%)",
                                output_path,
                                completed,
                                total_windows,
                                100.0 * completed / total_windows,
                            )
                        while len(inflight) < max_inflight and submit_next():
                            pass
    LOGGER.info(
        "Finished %s with %d unique intersecting features",
        output_path,
        len(seen_feature_ids),
    )
    return len(seen_feature_ids)


def _rasterize_window_task(
    vector_path: Path,
    window: Window,
    output_transform: Affine,
    reference_crs: CRS | None,
    vector_crs: CRS | None,
    burn_value: int,
    fill_value: int,
    dtype: str,
    all_touched: bool,
) -> tuple[Window, np.ndarray, set[str]]:
    """Rasterize one reference window in isolation.

    Each worker opens the vector layer independently to avoid sharing Fiona
    handles across threads.

    Args:
        vector_path (Path): Vector label path.
        window (Window): Reference raster window.
        output_transform (Affine): Full output affine transform.
        reference_crs (CRS | None): Reference raster CRS.
        vector_crs (CRS | None): Vector CRS.
        burn_value (int): Value burned into polygon pixels.
        fill_value (int): Background fill value.
        dtype (str): Output raster dtype.
        all_touched (bool): Whether to burn every touched pixel.

    Returns:
        tuple[Window, np.ndarray, set[str]]: Window descriptor, rasterized window
        array, and unique feature ids intersecting this window.
    """

    feature_ids: set[str] = set()
    bounds = window_bounds(window, output_transform)
    with fiona.open(vector_path) as src:
        query_bounds = bounds
        if reference_crs and vector_crs and reference_crs != vector_crs:
            query_bounds = transform_bounds(
                reference_crs,
                vector_crs,
                *bounds,
                densify_pts=21,
            )

        def shapes_iter() -> Iterator[tuple[dict[str, object], int]]:
            """Yield window-local shapes for rasterization.

            The worker keeps feature iteration lazy so each thread only holds
            geometry objects long enough for one `rasterize()` call.
            """

            for feature in src.filter(bbox=query_bounds):
                geometry = feature.get("geometry")
                if not geometry:
                    continue
                feature_id = str(feature.get("id", ""))
                if feature_id:
                    feature_ids.add(feature_id)
                if reference_crs and vector_crs and reference_crs != vector_crs:
                    geometry = transform_geom(
                        vector_crs,
                        reference_crs,
                        geometry,
                    )
                yield geometry, burn_value

        window_array = rasterize(
            shapes=shapes_iter(),
            out_shape=(int(window.height), int(window.width)),
            transform=window_transform(window, output_transform),
            fill=fill_value,
            dtype=dtype,
            all_touched=all_touched,
        )
    return window, window_array, feature_ids


def rasterize_reference_labels(
    vector_path: Path,
    reference_path: Path,
    output_path: Path,
    burn_value: int = 1,
    fill_value: int = 0,
    dtype: str = "uint8",
    all_touched: bool = False,
    compress: str = "deflate",
    vector_crs_override: str = "",
    window_size: int = 0,
    stream_threshold_pixels: int = 50_000_000,
    workers: int = 1,
    resolution_factor: int = 1,
) -> int:
    """Rasterize labels onto a single reference grid.

    Args:
        vector_path (Path): Input vector label path.
        reference_path (Path): Reference GeoTIFF defining bounds and transform.
        output_path (Path): Output label GeoTIFF path.
        burn_value (int): Value burned into polygon pixels.
        fill_value (int): Background value.
        dtype (str): Output raster dtype.
        all_touched (bool): Whether to burn every touched pixel.
        compress (str): GeoTIFF compression.
        vector_crs_override (str): Optional CRS override for the vector layer.
        window_size (int): Optional fixed streaming window size in pixels.
        stream_threshold_pixels (int): Auto-switch threshold for windowed
            rasterization based on total reference pixels.
        workers (int): Worker threads used for windowed rasterization.
        resolution_factor (int): Integer multiplier applied to output width and
            height while preserving CRS and bounds.

    Returns:
        int: Number of vector features burned into the output raster.
    """

    with fiona.open(vector_path) as src, rasterio.open(reference_path) as reference:
        if int(resolution_factor) < 1:
            raise ValueError("resolution_factor must be >= 1")
        vector_crs = _resolve_vector_crs(src, vector_crs_override)
        output_shape, output_transform = _resolve_output_grid(
            reference,
            resolution_factor,
        )
        total_pixels = int(output_shape[0]) * int(output_shape[1])
        resolved_window_size = int(window_size)
        if resolved_window_size <= 0 and total_pixels >= int(stream_threshold_pixels):
            resolved_window_size = 4096
        if resolved_window_size > 0:
            return _rasterize_windowed_reference(
                src=src,
                reference=reference,
                output_path=output_path,
                vector_crs=vector_crs,
                burn_value=burn_value,
                fill_value=fill_value,
                dtype=dtype,
                all_touched=all_touched,
                compress=compress,
                window_size=resolved_window_size,
                vector_path=vector_path,
                workers=max(1, int(workers)),
                output_shape=output_shape,
                output_transform=output_transform,
            )
        return _rasterize_full_reference(
            src=src,
            reference=reference,
            output_path=output_path,
            vector_crs=vector_crs,
            burn_value=burn_value,
            fill_value=fill_value,
            dtype=dtype,
            all_touched=all_touched,
            compress=compress,
            output_shape=output_shape,
            output_transform=output_transform,
        )


def _rasterize_full_grid(
    src: fiona.Collection,
    grid_spec: GridSpec,
    output_path: Path,
    vector_crs: CRS | None,
    burn_value: int,
    fill_value: int,
    dtype: str,
    all_touched: bool,
    compress: str,
) -> int:
    """Rasterize one vector layer onto an explicit output grid.

    Args:
        src (fiona.Collection): Open vector label dataset.
        grid_spec (GridSpec): Explicit output grid definition.
        output_path (Path): Output GeoTIFF path.
        vector_crs (CRS | None): Vector CRS.
        burn_value (int): Value burned into polygon pixels.
        fill_value (int): Background fill value.
        dtype (str): Output raster dtype.
        all_touched (bool): Whether to burn every touched pixel.
        compress (str): GeoTIFF compression codec.

    Returns:
        int: Number of intersecting vector features.
    """

    LOGGER.info(
        "Rasterizing %s on target grid (%dx%d pixels)",
        output_path.name,
        grid_spec.width,
        grid_spec.height,
    )
    shapes, feature_count = _iter_burn_shapes(
        src=src,
        reference_bounds=tuple(grid_spec.bounds),
        reference_crs=grid_spec.crs,
        vector_crs=vector_crs,
        burn_value=burn_value,
    )
    label_array = rasterize(
        shapes=shapes,
        out_shape=(grid_spec.height, grid_spec.width),
        transform=grid_spec.transform,
        fill=fill_value,
        dtype=dtype,
        all_touched=all_touched,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        **_build_grid_profile(grid_spec, dtype, fill_value, compress),
    ) as dst:
        dst.write(label_array, 1)
    LOGGER.info(
        "Finished %s with %d intersecting features",
        output_path,
        feature_count[0],
    )
    return feature_count[0]


def _rasterize_windowed_grid(
    src: fiona.Collection,
    grid_spec: GridSpec,
    output_path: Path,
    vector_crs: CRS | None,
    burn_value: int,
    fill_value: int,
    dtype: str,
    all_touched: bool,
    compress: str,
    window_size: int,
    vector_path: Path,
    workers: int,
) -> int:
    """Rasterize a vector layer onto an explicit output grid by windows.

    Args:
        src (fiona.Collection): Open vector label dataset.
        grid_spec (GridSpec): Explicit output grid definition.
        output_path (Path): Output GeoTIFF path.
        vector_crs (CRS | None): Vector CRS.
        burn_value (int): Value burned into polygon pixels.
        fill_value (int): Background fill value.
        dtype (str): Output raster dtype.
        all_touched (bool): Whether to burn every touched pixel.
        compress (str): GeoTIFF compression codec.
        window_size (int): Window edge length in pixels.
        vector_path (Path): Vector label path used by worker threads.
        workers (int): Number of worker threads for window rasterization.

    Returns:
        int: Number of unique intersecting vector features.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_feature_ids: set[str] = set()
    output_height = int(grid_spec.height)
    output_width = int(grid_spec.width)
    total_windows = ((output_height + window_size - 1) // window_size) * (
        (output_width + window_size - 1) // window_size
    )
    progress_step = max(1, total_windows // 20)
    LOGGER.info(
        "Rasterizing %s in windowed grid mode (%dx%d pixels, %d windows of %d px)",
        output_path.name,
        output_width,
        output_height,
        total_windows,
        window_size,
    )
    with rasterio.open(
        output_path,
        "w",
        **_build_grid_profile(grid_spec, dtype, fill_value, compress),
    ) as dst:
        if workers <= 1:
            for index, window in enumerate(
                _iter_windows(output_height, output_width, window_size),
                start=1,
            ):
                bounds = window_bounds(window, grid_spec.transform)
                shapes, _ = _iter_burn_shapes(
                    src=src,
                    reference_bounds=bounds,
                    reference_crs=grid_spec.crs,
                    vector_crs=vector_crs,
                    burn_value=burn_value,
                    seen_feature_ids=seen_feature_ids,
                )
                window_array = rasterize(
                    shapes=shapes,
                    out_shape=(int(window.height), int(window.width)),
                    transform=window_transform(window, grid_spec.transform),
                    fill=fill_value,
                    dtype=dtype,
                    all_touched=all_touched,
                )
                dst.write(window_array, 1, window=window)
                if index == 1 or index == total_windows or index % progress_step == 0:
                    LOGGER.info(
                        "Window progress for %s: %d/%d (%.1f%%)",
                        output_path,
                        index,
                        total_windows,
                        100.0 * index / total_windows,
                    )
        else:
            LOGGER.info("Using %d worker threads for window rasterization", workers)
            windows_iter = iter(_iter_windows(output_height, output_width, window_size))
            completed = 0
            max_inflight = max(workers * 2, 1)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                inflight: dict[Future[tuple[Window, np.ndarray, set[str]]], Window] = {}

                def submit_next() -> bool:
                    """Submit one more window task if work remains.

                    Returns:
                        bool: True when a task was submitted.
                    """

                    try:
                        next_window = next(windows_iter)
                    except StopIteration:
                        return False
                    future = executor.submit(
                        _rasterize_window_task,
                        vector_path=vector_path,
                        window=next_window,
                        output_transform=grid_spec.transform,
                        reference_crs=grid_spec.crs,
                        vector_crs=vector_crs,
                        burn_value=burn_value,
                        fill_value=fill_value,
                        dtype=dtype,
                        all_touched=all_touched,
                    )
                    inflight[future] = next_window
                    return True

                while len(inflight) < max_inflight and submit_next():
                    pass
                while inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for future in done:
                        window, window_array, feature_ids = future.result()
                        inflight.pop(future)
                        seen_feature_ids.update(feature_ids)
                        dst.write(window_array, 1, window=window)
                        completed += 1
                        if (
                            completed == 1
                            or completed == total_windows
                            or completed % progress_step == 0
                        ):
                            LOGGER.info(
                                "Window progress for %s: %d/%d (%.1f%%)",
                                output_path,
                                completed,
                                total_windows,
                                100.0 * completed / total_windows,
                            )
                        while len(inflight) < max_inflight and submit_next():
                            pass
    LOGGER.info(
        "Finished %s with %d unique intersecting features",
        output_path,
        len(seen_feature_ids),
    )
    return len(seen_feature_ids)


def rasterize_labels_to_grid(
    vector_path: Path,
    output_path: Path,
    grid_spec: GridSpec,
    burn_value: int = 1,
    fill_value: int = 0,
    dtype: str = "uint8",
    all_touched: bool = False,
    compress: str = "deflate",
    vector_crs_override: str = "",
    window_size: int = 0,
    stream_threshold_pixels: int = 50_000_000,
    workers: int = 1,
) -> int:
    """Rasterize labels onto one explicit target grid.

    Args:
        vector_path (Path): Input vector label path.
        output_path (Path): Output label GeoTIFF path.
        grid_spec (GridSpec): Explicit output grid definition.
        burn_value (int): Value burned into polygon pixels.
        fill_value (int): Background value.
        dtype (str): Output raster dtype.
        all_touched (bool): Whether to burn every touched pixel.
        compress (str): GeoTIFF compression codec.
        vector_crs_override (str): Optional CRS override for the vector layer.
        window_size (int): Optional fixed streaming window size in pixels.
        stream_threshold_pixels (int): Auto-switch threshold for windowed
            rasterization based on total output pixels.
        workers (int): Worker threads used for windowed rasterization.

    Returns:
        int: Number of vector features burned into the output raster.

    Examples:
        >>> callable(rasterize_labels_to_grid)
        True
    """

    with fiona.open(vector_path) as src:
        vector_crs = _resolve_vector_crs(src, vector_crs_override)
        total_pixels = int(grid_spec.height) * int(grid_spec.width)
        resolved_window_size = int(window_size)
        if resolved_window_size <= 0 and total_pixels >= int(stream_threshold_pixels):
            resolved_window_size = 4096
        if resolved_window_size > 0:
            return _rasterize_windowed_grid(
                src=src,
                grid_spec=grid_spec,
                output_path=output_path,
                vector_crs=vector_crs,
                burn_value=burn_value,
                fill_value=fill_value,
                dtype=dtype,
                all_touched=all_touched,
                compress=compress,
                window_size=resolved_window_size,
                vector_path=vector_path,
                workers=max(1, int(workers)),
            )
        return _rasterize_full_grid(
            src=src,
            grid_spec=grid_spec,
            output_path=output_path,
            vector_crs=vector_crs,
            burn_value=burn_value,
            fill_value=fill_value,
            dtype=dtype,
            all_touched=all_touched,
            compress=compress,
        )


def rasterize_vector_directory_to_grid(
    vector_dir: Path,
    output_path: Path,
    grid_spec: GridSpec,
    *,
    vector_glob: str = "*.shp",
    burn_value: int = 1,
    fill_value: int = 0,
    dtype: str = "uint8",
    all_touched: bool = False,
    compress: str = "deflate",
    vector_crs_override: str = "",
    window_size: int = 0,
    stream_threshold_pixels: int = 50_000_000,
    workers: int = 1,
    vector_workers: int = 1,
    overwrite: bool = False,
) -> tuple[Path, list[Path]]:
    """Rasterize all shapefiles in a directory tree onto one explicit grid.

    Args:
        vector_dir (Path): Directory containing one or more shapefiles.
        output_path (Path): Merged output TIFF path.
        grid_spec (GridSpec): Explicit output grid definition.
        vector_glob (str): Recursive glob used to discover shapefiles.
        burn_value (int): Value burned into polygon pixels.
        fill_value (int): Background value.
        dtype (str): Output raster dtype.
        all_touched (bool): Whether to burn every touched pixel.
        compress (str): GeoTIFF compression codec.
        vector_crs_override (str): Optional CRS override for the vector layer.
        window_size (int): Optional fixed streaming window size in pixels.
        stream_threshold_pixels (int): Auto-switch threshold for windowed
            rasterization based on total output pixels.
        workers (int): Worker threads used inside one rasterization job.
        vector_workers (int): Parallel worker count across shapefiles.
        overwrite (bool): Whether to overwrite existing outputs.

    Returns:
        tuple[Path, list[Path]]: Merged TIFF path and per-shapefile TIFF paths.

    Examples:
        >>> callable(rasterize_vector_directory_to_grid)
        True
    """

    vector_paths = collect_vector_paths(vector_dir, vector_glob)
    merged_output_path = output_path
    if merged_output_path.exists() and not overwrite:
        LOGGER.info("Skipping existing merged vector output %s", merged_output_path)
        individual_paths = [
            derive_vector_output_path(merged_output_path, path, vector_dir)
            for path in vector_paths
        ]
        return merged_output_path, individual_paths

    def rasterize_one(vector_file: Path) -> Path:
        """Rasterize one discovered shapefile onto the configured grid.

        Args:
            vector_file (Path): Shapefile discovered under `vector_dir`.

        Returns:
            Path: Rasterized TIFF path for the one shapefile.
        """

        individual_output_path = derive_vector_output_path(
            merged_output_path,
            vector_file,
            vector_dir,
        )
        if individual_output_path.exists() and not overwrite:
            LOGGER.info("Skipping existing %s", individual_output_path)
            return individual_output_path
        feature_count = rasterize_labels_to_grid(
            vector_path=vector_file,
            output_path=individual_output_path,
            grid_spec=grid_spec,
            burn_value=burn_value,
            fill_value=fill_value,
            dtype=dtype,
            all_touched=all_touched,
            compress=compress,
            vector_crs_override=vector_crs_override,
            window_size=window_size,
            stream_threshold_pixels=stream_threshold_pixels,
            workers=workers,
        )
        LOGGER.info(
            "Wrote %s from %s with %d intersecting features",
            individual_output_path,
            vector_file,
            feature_count,
        )
        return individual_output_path

    if int(vector_workers) <= 1:
        individual_paths = [rasterize_one(path) for path in vector_paths]
    else:
        LOGGER.info(
            "Rasterizing %d shapefiles with %d parallel vector workers",
            len(vector_paths),
            vector_workers,
        )
        with ThreadPoolExecutor(max_workers=max(1, int(vector_workers))) as executor:
            futures = [executor.submit(rasterize_one, path) for path in vector_paths]
            individual_paths = [future.result() for future in futures]
    merged_path = _merge_label_rasters(individual_paths, merged_output_path, compress)
    return merged_path, sorted(individual_paths)


def align_label_raster_to_grid(
    raster_path: Path,
    output_path: Path,
    grid_spec: GridSpec,
    *,
    fill_value: int = 0,
    dtype: str = "uint8",
    compress: str = "deflate",
) -> Path:
    """Reproject one label TIFF onto the explicit target grid.

    Args:
        raster_path (Path): Input label TIFF path.
        output_path (Path): Aligned output TIFF path.
        grid_spec (GridSpec): Explicit output grid definition.
        fill_value (int): Background value.
        dtype (str): Output raster dtype.
        compress (str): GeoTIFF compression codec.

    Returns:
        Path: Output TIFF path aligned to the target grid.

    Examples:
        >>> callable(align_label_raster_to_grid)
        True
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        rasterio.open(raster_path) as src,
        rasterio.open(
            output_path,
            "w",
            **_build_grid_profile(grid_spec, dtype, fill_value, compress),
        ) as dst,
    ):
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid_spec.transform,
            dst_crs=grid_spec.crs,
            dst_nodata=fill_value,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )
    LOGGER.info("Aligned %s onto %s", raster_path, output_path)
    return output_path


def align_raster_directory_to_grid(
    raster_dir: Path,
    output_path: Path,
    grid_spec: GridSpec,
    *,
    raster_glob: str = "*.tif",
    fill_value: int = 0,
    dtype: str = "uint8",
    compress: str = "deflate",
    overwrite: bool = False,
) -> tuple[Path, list[Path]]:
    """Align all raster label TIFFs in a directory tree to one explicit grid.

    Args:
        raster_dir (Path): Directory containing one or more label TIFFs.
        output_path (Path): Merged output TIFF path.
        grid_spec (GridSpec): Explicit output grid definition.
        raster_glob (str): Recursive glob used to discover label TIFFs.
        fill_value (int): Background value.
        dtype (str): Output raster dtype.
        compress (str): GeoTIFF compression codec.
        overwrite (bool): Whether to overwrite existing outputs.

    Returns:
        tuple[Path, list[Path]]: Merged TIFF path and aligned per-raster TIFFs.

    Examples:
        >>> callable(align_raster_directory_to_grid)
        True
    """

    raster_paths = collect_raster_paths(raster_dir, raster_glob)
    merged_output_path = output_path
    if merged_output_path.exists() and not overwrite:
        LOGGER.info("Skipping existing merged raster output %s", merged_output_path)
        individual_paths = [
            derive_raster_output_path(merged_output_path, path, raster_dir)
            for path in raster_paths
        ]
        return merged_output_path, individual_paths

    individual_paths: list[Path] = []
    for raster_path in raster_paths:
        aligned_output_path = derive_raster_output_path(
            merged_output_path,
            raster_path,
            raster_dir,
        )
        if aligned_output_path.exists() and not overwrite:
            LOGGER.info("Skipping existing %s", aligned_output_path)
            individual_paths.append(aligned_output_path)
            continue
        individual_paths.append(
            align_label_raster_to_grid(
                raster_path=raster_path,
                output_path=aligned_output_path,
                grid_spec=grid_spec,
                fill_value=fill_value,
                dtype=dtype,
                compress=compress,
            )
        )
    merged_path = _merge_label_rasters(individual_paths, merged_output_path, compress)
    return merged_path, sorted(individual_paths)


def measure_planet_coverage(
    final_output_path: Path,
    verify_path: Path,
) -> dict[str, float]:
    """Measure final-mask coverage against the verification raster.

    Args:
        final_output_path (Path): Final merged output label TIFF.
        verify_path (Path): Verification raster used for QA.

    Returns:
        dict[str, float]: Coverage, containment, IoU, and pixel-count metrics.

    Examples:
        >>> callable(measure_planet_coverage)
        True
    """

    with (
        rasterio.open(final_output_path) as final_src,
        rasterio.open(verify_path) as verify_src,
    ):
        final_on_verify = np.zeros(
            (verify_src.height, verify_src.width),
            dtype=np.uint8,
        )
        reproject(
            source=rasterio.band(final_src, 1),
            destination=final_on_verify,
            src_transform=final_src.transform,
            src_crs=final_src.crs,
            src_nodata=final_src.nodata,
            dst_transform=verify_src.transform,
            dst_crs=verify_src.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )
        verify_data = verify_src.read(1)
    final_mask = final_on_verify > 0
    verify_mask = verify_data > 0
    intersection = int(np.count_nonzero(final_mask & verify_mask))
    verify_positive = int(np.count_nonzero(verify_mask))
    final_positive = int(np.count_nonzero(final_mask))
    union = int(np.count_nonzero(final_mask | verify_mask))
    coverage = intersection / verify_positive if verify_positive else 0.0
    output_inside = intersection / final_positive if final_positive else 0.0
    iou = intersection / union if union else 0.0
    return {
        "coverage": float(coverage),
        "output_inside": float(output_inside),
        "iou": float(iou),
        "intersection_pixels": float(intersection),
        "verify_positive_pixels": float(verify_positive),
        "final_positive_pixels": float(final_positive),
    }


def run_configured_raster_merge(config: dict[str, Any]) -> dict[str, Any]:
    """Run the config-driven merge workflow for raster and vector labels.

    Args:
        config (dict[str, Any]): Workflow config mapping loaded from YAML.

    Returns:
        dict[str, Any]: Paths, grid metadata, and QA metrics for the run.

    Examples:
        >>> callable(run_configured_raster_merge)
        True
    """

    vector_dir = Path(_config_value(config, "vector_dir", required=True))
    raster_dir = Path(_config_value(config, "raster_dir", required=True))
    verify_path = Path(_config_value(config, "verify_path", required=True))
    output_path = Path(_config_value(config, "output_path", required=True))
    target_crs = _config_value(config, "target_crs", "EPSG:25832")
    target_resolution = float(_config_value(config, "target_resolution", 1.0))
    burn_value = int(_config_value(config, "burn_value", 1))
    fill_value = int(_config_value(config, "fill_value", 0))
    dtype = str(_config_value(config, "dtype", "uint8"))
    all_touched = bool(_config_value(config, "all_touched", False))
    compress = str(_config_value(config, "compress", "deflate"))
    vector_crs_override = str(_config_value(config, "vector_crs", ""))
    window_size = int(_config_value(config, "window_size", 0))
    stream_threshold_pixels = int(
        _config_value(config, "stream_threshold_pixels", 50_000_000)
    )
    workers = int(_config_value(config, "workers", 1))
    vector_workers = int(_config_value(config, "vector_workers", 1))
    overwrite = bool(_config_value(config, "overwrite", False))
    vector_glob = str(_config_value(config, "vector_glob", "*.shp"))
    raster_glob = str(_config_value(config, "merge_raster_glob", "*.tif"))
    min_planet_coverage = float(_config_value(config, "min_planet_coverage", 0.8))

    grid_spec = build_grid_spec_from_verify(
        verify_path=verify_path,
        target_crs=target_crs,
        target_resolution=target_resolution,
    )
    vector_merged_path = output_path.with_name(
        f"{output_path.stem}_vectors{output_path.suffix}"
    )
    raster_merged_path = output_path.with_name(
        f"{output_path.stem}_rasters{output_path.suffix}"
    )
    vector_merged_path, vector_parts = rasterize_vector_directory_to_grid(
        vector_dir=vector_dir,
        output_path=vector_merged_path,
        grid_spec=grid_spec,
        vector_glob=vector_glob,
        burn_value=burn_value,
        fill_value=fill_value,
        dtype=dtype,
        all_touched=all_touched,
        compress=compress,
        vector_crs_override=vector_crs_override,
        window_size=window_size,
        stream_threshold_pixels=stream_threshold_pixels,
        workers=workers,
        vector_workers=vector_workers,
        overwrite=overwrite,
    )
    raster_merged_path, raster_parts = align_raster_directory_to_grid(
        raster_dir=raster_dir,
        output_path=raster_merged_path,
        grid_spec=grid_spec,
        raster_glob=raster_glob,
        fill_value=fill_value,
        dtype=dtype,
        compress=compress,
        overwrite=overwrite,
    )
    if output_path.exists() and not overwrite:
        LOGGER.info("Skipping existing final output %s", output_path)
    else:
        _merge_label_rasters(
            [raster_merged_path, vector_merged_path],
            output_path,
            compress,
        )
    metrics = measure_planet_coverage(output_path, verify_path)
    LOGGER.info(
        "Planet coverage for %s: coverage=%.4f output_inside=%.4f iou=%.4f",
        output_path,
        metrics["coverage"],
        metrics["output_inside"],
        metrics["iou"],
    )
    if metrics["coverage"] < min_planet_coverage:
        raise ValueError(
            "final merged output does not meet minimum Planet coverage: "
            f"{metrics['coverage']:.4f} < {min_planet_coverage:.4f}"
        )
    return {
        "output_path": output_path,
        "vector_merged_path": vector_merged_path,
        "raster_merged_path": raster_merged_path,
        "vector_parts": vector_parts,
        "raster_parts": raster_parts,
        "grid_spec": grid_spec,
        "metrics": metrics,
    }


def _merge_label_rasters(
    input_paths: list[Path],
    output_path: Path,
    compress: str,
) -> Path:
    """Merge multiple rasterized label TIFFs into one output TIFF.

    The merge uses a pixelwise `max` so any positive label in the source stack
    is preserved in the final raster.

    Args:
        input_paths (list[Path]): Rasterized label TIFFs sharing one grid.
        output_path (Path): Merged output TIFF path.
        compress (str): GeoTIFF compression codec.

    Returns:
        Path: Merged output TIFF path.
    """

    if not input_paths:
        raise ValueError("input_paths must not be empty when merging label rasters")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in input_paths]
        first = sources[0]
        for src in sources[1:]:
            if (
                src.width != first.width
                or src.height != first.height
                or src.crs != first.crs
                or src.transform != first.transform
                or src.dtypes[0] != first.dtypes[0]
            ):
                raise ValueError(
                    "all input rasters must share shape, CRS, transform, and dtype"
                )
        profile = first.profile.copy()
        profile.update(driver="GTiff", count=1, compress=compress)
        with rasterio.open(output_path, "w", **profile) as dst:
            for window in _iter_windows(first.height, first.width, 4096):
                merged = sources[0].read(1, window=window)
                for src in sources[1:]:
                    merged = np.maximum(merged, src.read(1, window=window))
                dst.write(merged, 1, window=window)
    LOGGER.info(
        "Merged %d rasterized label TIFFs into %s", len(input_paths), output_path
    )
    return output_path


def rasterize_vector_directory(
    vector_dir: Path,
    reference_path: Path,
    output_path: Path,
    *,
    vector_glob: str = "*.shp",
    burn_value: int = 1,
    fill_value: int = 0,
    dtype: str = "uint8",
    all_touched: bool = False,
    compress: str = "deflate",
    vector_crs_override: str = "",
    window_size: int = 0,
    stream_threshold_pixels: int = 50_000_000,
    workers: int = 1,
    resolution_factor: int = 1,
    vector_workers: int = 1,
    overwrite: bool = False,
) -> tuple[Path, list[Path]]:
    """Rasterize all shapefiles in a directory tree and merge them.

    Args:
        vector_dir (Path): Directory containing one or more shapefiles.
        reference_path (Path): Reference GeoTIFF defining bounds and transform.
        output_path (Path): Merged output TIFF path or output directory.
        vector_glob (str): Recursive glob used to discover shapefiles.
        burn_value (int): Value burned into polygon pixels.
        fill_value (int): Background value.
        dtype (str): Output raster dtype.
        all_touched (bool): Whether to burn every touched pixel.
        compress (str): GeoTIFF compression codec.
        vector_crs_override (str): Optional CRS override for the vector layer.
        window_size (int): Optional fixed streaming window size in pixels.
        stream_threshold_pixels (int): Auto-switch threshold for windowed
            rasterization based on total output pixels.
        workers (int): Worker threads used inside one rasterization job.
        resolution_factor (int): Integer multiplier for output width and height.
        vector_workers (int): Parallel worker count across shapefiles.
        overwrite (bool): Whether to overwrite existing per-shape and merged
            TIFFs.

    Returns:
        tuple[Path, list[Path]]: Merged TIFF path and per-shapefile TIFF paths.

    Examples:
        >>> callable(rasterize_vector_directory)
        True
    """

    vector_paths = collect_vector_paths(vector_dir, vector_glob)
    merged_output_path = derive_output_path(reference_path, output_path)
    if merged_output_path.exists() and not overwrite:
        LOGGER.info("Skipping existing merged output %s", merged_output_path)
        individual_paths = [
            derive_vector_output_path(merged_output_path, path, vector_dir)
            for path in vector_paths
        ]
        return merged_output_path, individual_paths

    def rasterize_one(vector_file: Path) -> Path:
        """Rasterize one discovered shapefile into its individual TIFF.

        Args:
            vector_file (Path): Shapefile discovered under `vector_dir`.

        Returns:
            Path: Rasterized TIFF path for the one shapefile.
        """

        individual_output_path = derive_vector_output_path(
            merged_output_path,
            vector_file,
            vector_dir,
        )
        if individual_output_path.exists() and not overwrite:
            LOGGER.info("Skipping existing %s", individual_output_path)
            return individual_output_path
        feature_count = rasterize_reference_labels(
            vector_path=vector_file,
            reference_path=reference_path,
            output_path=individual_output_path,
            burn_value=burn_value,
            fill_value=fill_value,
            dtype=dtype,
            all_touched=all_touched,
            compress=compress,
            vector_crs_override=vector_crs_override,
            window_size=window_size,
            stream_threshold_pixels=stream_threshold_pixels,
            workers=workers,
            resolution_factor=resolution_factor,
        )
        LOGGER.info(
            "Wrote %s from %s with %d intersecting features",
            individual_output_path,
            vector_file,
            feature_count,
        )
        return individual_output_path

    if int(vector_workers) <= 1:
        individual_paths = [rasterize_one(path) for path in vector_paths]
    else:
        LOGGER.info(
            "Rasterizing %d shapefiles with %d parallel vector workers",
            len(vector_paths),
            vector_workers,
        )
        with ThreadPoolExecutor(max_workers=max(1, int(vector_workers))) as executor:
            futures = [executor.submit(rasterize_one, path) for path in vector_paths]
            individual_paths = [future.result() for future in futures]
    merged_path = _merge_label_rasters(individual_paths, merged_output_path, compress)
    return merged_path, sorted(individual_paths)


def _configure_logging(level: str) -> None:
    """Configure CLI logging for the rasterization script.

    Args:
        level (str): Requested logging level name.
    """

    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the config-driven raster merge workflow.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_path",
        nargs="?",
        default=DEFAULT_RASTERIZE_CONFIG_PATH,
        type=Path,
        help="Path to the rasterize-labels YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the config-driven raster and vector label merge CLI.

    Returns:
        int: Exit code.

    Examples:
        >>> callable(main)
        True
    """

    args = _parse_args()
    config = load_rasterize_config(args.config_path)
    logging_config = config.get("logging", {})
    _configure_logging(str(logging_config.get("level", "INFO")))
    result = run_configured_raster_merge(config)
    LOGGER.info("Wrote final merged labels to %s", result["output_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
