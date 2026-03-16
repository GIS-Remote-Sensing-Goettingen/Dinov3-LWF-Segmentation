"""Rasterize vector labels onto reference GeoTIFF grids.

This utility is meant for cases where labels arrive as polygons (for example a
shapefile) but the pipeline expects raster labels aligned to image GeoTIFFs.
It can rasterize one reference TIFF or every TIFF in a directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import fiona
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, transform_geom


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


def derive_output_path(reference_path: Path, output_path: Path) -> Path:
    """Return the label TIFF path for a reference raster.

    When `output_path` is a directory, this function derives a label filename
    from the reference stem. `_pred` and `_image` suffixes are normalized to
    `_labels`.

    Args:
        reference_path (Path): Source raster used as the alignment grid.
        output_path (Path): Output file or directory path.

    Returns:
        Path: Target label TIFF path.

    Examples:
        >>> ref = Path("dop20_596000_5973000_1km_20cm_pred.tif")
        >>> derive_output_path(ref, Path("labels")).name
        'dop20_596000_5973000_1km_20cm_labels.tif'
        >>> derive_output_path(Path("scene.tif"), Path("labels")).name
        'scene_labels.tif'
    """

    if output_path.suffix.lower() in {".tif", ".tiff"}:
        return output_path
    stem = reference_path.stem
    for suffix in ("_pred", "_image", "_img"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return output_path / f"{stem}_labels.tif"


def _normalize_crs(crs_value: str | CRS | dict[str, str] | None) -> CRS | None:
    """Convert CRS-like input into a rasterio CRS."""

    if not crs_value:
        return None
    return CRS.from_user_input(crs_value)


def _iter_burn_shapes(
    vector_path: Path,
    reference_bounds: tuple[float, float, float, float],
    reference_crs: CRS | None,
    vector_crs: CRS | None,
    burn_value: int,
) -> tuple[Iterator[tuple[dict[str, object], int]], list[int]]:
    """Yield rasterize-ready shapes for the reference extent.

    Args:
        vector_path (Path): Input vector label file.
        reference_bounds (tuple[float, float, float, float]): Target bounds in
            reference CRS order `(left, bottom, right, top)`.
        reference_crs (CRS | None): Reference raster CRS.
        vector_crs (CRS | None): Vector CRS.
        burn_value (int): Label value written for each polygon.

    Returns:
        tuple[Iterator[tuple[dict[str, object], int]], list[int]]: Shape
        iterator and mutable one-item list storing the yielded feature count.
    """

    count = [0]

    def iterator() -> Iterator[tuple[dict[str, object], int]]:
        with fiona.open(vector_path) as src:
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
                if reference_crs and vector_crs and reference_crs != vector_crs:
                    geometry = transform_geom(
                        vector_crs,
                        reference_crs,
                        geometry,
                    )
                count[0] += 1
                yield geometry, burn_value

    return iterator(), count


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

    Returns:
        int: Number of vector features burned into the output raster.
    """

    with fiona.open(vector_path) as src:
        vector_crs = _normalize_crs(vector_crs_override) or _normalize_crs(src.crs_wkt)
        if vector_crs is None:
            vector_crs = _normalize_crs(src.crs)

    with rasterio.open(reference_path) as reference:
        shapes, feature_count = _iter_burn_shapes(
            vector_path=vector_path,
            reference_bounds=reference.bounds,
            reference_crs=reference.crs,
            vector_crs=vector_crs,
            burn_value=burn_value,
        )
        label_array = rasterize(
            shapes=shapes,
            out_shape=reference.shape,
            transform=reference.transform,
            fill=fill_value,
            dtype=dtype,
            all_touched=all_touched,
        )
        profile = reference.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype=dtype,
            nodata=fill_value,
            compress=compress,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.asarray(label_array), 1)
    return feature_count[0]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for vector-to-label raster conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "vector_path",
        type=Path,
        help="Vector label path, e.g. /path/to/union.shp.",
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
        target_path = derive_output_path(reference_path, args.output_path)
        if target_path.exists() and not args.overwrite:
            print(f"skip existing {target_path}")
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
        )
        print(
            f"wrote {target_path} from {reference_path} "
            f"with {feature_count} intersecting features"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
