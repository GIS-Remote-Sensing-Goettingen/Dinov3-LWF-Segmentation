"""Rasterize vector labels onto reference GeoTIFF grids.

This utility is meant for cases where labels arrive as polygons (for example a
shapefile) but the pipeline expects raster labels aligned to image GeoTIFFs.
It can rasterize one reference TIFF or every TIFF in a directory.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator

import fiona
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

LOGGER = logging.getLogger(__name__)


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
    """Parse CLI arguments for vector-to-label raster conversion.

    The parser accepts a vector layer, one reference TIFF or directory of TIFFs,
    and either one output TIFF path or an output directory for batch mode.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "vector_path",
        type=Path,
        help="Vector label path or directory of shapefiles.",
    )
    parser.add_argument(
        "reference_path",
        type=Path,
        help="Reference GeoTIFF or directory of reference GeoTIFFs.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output TIFF path for a single reference or output directory.",
    )
    parser.add_argument(
        "--glob",
        default="*.tif",
        help="Glob for reference TIFF discovery when reference_path is a directory.",
    )
    parser.add_argument(
        "--vector-glob",
        default="*.shp",
        help="Recursive glob for shapefile discovery when vector_path is a directory.",
    )
    parser.add_argument(
        "--burn-value",
        type=int,
        default=1,
        help="Value written inside polygons.",
    )
    parser.add_argument(
        "--fill-value",
        type=int,
        default=0,
        help="Background value written outside polygons.",
    )
    parser.add_argument(
        "--dtype",
        default="uint8",
        help="Output raster dtype, e.g. uint8 or uint16.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Burn every pixel touched by a polygon edge.",
    )
    parser.add_argument(
        "--compress",
        default="deflate",
        help="GeoTIFF compression codec.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output TIFFs if they already exist.",
    )
    parser.add_argument(
        "--vector-crs",
        default="",
        help="Optional CRS override, e.g. EPSG:25832.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="Optional streaming window size in pixels for large references.",
    )
    parser.add_argument(
        "--stream-threshold-pixels",
        type=int,
        default=50_000_000,
        help="Auto-enable windowed rasterization above this pixel count.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level, e.g. INFO or DEBUG.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker threads for windowed rasterization.",
    )
    parser.add_argument(
        "--vector-workers",
        type=int,
        default=1,
        help="Parallel worker threads across shapefiles in a vector directory.",
    )
    parser.add_argument(
        "--resolution-factor",
        type=int,
        default=1,
        help=(
            "Integer multiplier for output width and height while preserving "
            "the reference CRS and geographic extent."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the vector label rasterization CLI.

    Returns:
        int: Exit code.

    Examples:
        >>> callable(main)
        True
    """

    args = _parse_args()
    _configure_logging(args.log_level)
    reference_paths = collect_reference_paths(args.reference_path, args.glob)
    if len(reference_paths) > 1 and args.output_path.suffix.lower() in {
        ".tif",
        ".tiff",
    }:
        raise ValueError(
            "output_path must be a directory when reference_path resolves to "
            "multiple GeoTIFFs"
        )
    for reference_path in reference_paths:
        if args.vector_path.is_dir():
            merged_path, individual_paths = rasterize_vector_directory(
                vector_dir=args.vector_path,
                reference_path=reference_path,
                output_path=args.output_path,
                vector_glob=args.vector_glob,
                burn_value=args.burn_value,
                fill_value=args.fill_value,
                dtype=args.dtype,
                all_touched=args.all_touched,
                compress=args.compress,
                vector_crs_override=args.vector_crs,
                window_size=args.window_size,
                stream_threshold_pixels=args.stream_threshold_pixels,
                workers=args.workers,
                resolution_factor=args.resolution_factor,
                vector_workers=args.vector_workers,
                overwrite=args.overwrite,
            )
            LOGGER.info(
                "Wrote merged %s from %d shapefiles for %s",
                merged_path,
                len(individual_paths),
                reference_path,
            )
            continue
        target_path = derive_output_path(reference_path, args.output_path)
        LOGGER.info("Using reference %s -> %s", reference_path, target_path)
        if target_path.exists() and not args.overwrite:
            LOGGER.info("Skipping existing %s", target_path)
            continue
        feature_count = rasterize_reference_labels(
            vector_path=args.vector_path,
            reference_path=reference_path,
            output_path=target_path,
            burn_value=args.burn_value,
            fill_value=args.fill_value,
            dtype=args.dtype,
            all_touched=args.all_touched,
            compress=args.compress,
            vector_crs_override=args.vector_crs,
            window_size=args.window_size,
            stream_threshold_pixels=args.stream_threshold_pixels,
            workers=args.workers,
            resolution_factor=args.resolution_factor,
        )
        LOGGER.info(
            "Wrote %s from %s with %d intersecting features",
            target_path,
            reference_path,
            feature_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
