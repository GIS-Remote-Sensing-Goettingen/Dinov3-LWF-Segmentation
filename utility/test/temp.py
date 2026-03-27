"""Plot folder tile coverage and overlay Desktop prediction-labelled tiles.

Examples:
    >>> callable(main)
    True
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.vrt import WarpedVRT  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

DEFAULT_TILES_DIR = Path("/home/mak/cluster_hdd/patches_mt/folder_1")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "folder_1_coverage"
DEFAULT_PREDICTION_DIR = Path("/home/mak/Desktop")
DEFAULT_PLANET_LABELS_TIF = Path(
    "/run/media/mak/Partition of 1TB disk/SH_dataset/planet_labels_2022.tif"
)
TILE_NAME_RE = re.compile(r"^dop20_(\d+)_(\d+)_1km_20cm\.tif$")
TILE_SIZE_M = 1000
LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure simple progress logging for the coverage helper.

    This keeps long raster scans observable without requiring extra flags.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        force=True,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the coverage plotting script.

    Returns:
        argparse.ArgumentParser: Configured parser.

    Examples:
        >>> isinstance(_build_arg_parser().prog, str)
        True
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiles-dir",
        default=str(DEFAULT_TILES_DIR),
        help="Directory containing the DOP20 TIFF tiles to map.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the coverage PNG and summary JSON are written.",
    )
    parser.add_argument(
        "--prediction-dir",
        default=str(DEFAULT_PREDICTION_DIR),
        help="Directory containing prediction rasters such as predictions.tif.",
    )
    parser.add_argument(
        "--prediction-glob",
        default="predictions*.tif",
        help="Glob used to discover prediction rasters in the prediction directory.",
    )
    parser.add_argument(
        "--planet-labels-tif",
        default=str(DEFAULT_PLANET_LABELS_TIF),
        help="Planet label raster used for the violet >60%% tile threshold.",
    )
    parser.add_argument(
        "--row-chunk-pixels",
        type=int,
        default=TILE_SIZE_M,
        help="Retained for compatibility; coarse 1 km max-resampling is used now.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Retained for compatibility; prediction rasters are scanned sequentially.",
    )
    parser.add_argument(
        "--gdal-cache-mb",
        type=int,
        default=64,
        help="Per-worker GDAL cache limit in MB used during coarse warp reads.",
    )
    return parser


def parse_tile_origin_from_name(name: str) -> tuple[float, float] | None:
    """Parse one canonical DOP20 filename into its tile origin.

    Args:
        name (str): Tile filename.

    Returns:
        tuple[float, float] | None: Parsed `(x0, y0)` tile origin or None when
            the file name does not match the expected pattern.

    Examples:
        >>> parse_tile_origin_from_name('dop20_453000_6066000_1km_20cm.tif')
        (453000.0, 6066000.0)
        >>> parse_tile_origin_from_name('notes.txt') is None
        True
    """

    match = TILE_NAME_RE.match(name)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def collect_tile_origins(tiles_dir: Path) -> list[tuple[float, float]]:
    """Collect sorted tile origins from one folder of TIFF filenames.

    Args:
        tiles_dir (Path): Directory containing DOP20 TIFF tiles.

    Returns:
        list[tuple[float, float]]: Sorted unique tile origins.

    Examples:
        >>> callable(collect_tile_origins)
        True
    """

    iterator = (
        tqdm(tiles_dir.iterdir(), desc="Folder tiles", unit="file")
        if tqdm is not None
        else tiles_dir.iterdir()
    )
    origins = {
        origin
        for path in iterator
        if path.is_file()
        and (origin := parse_tile_origin_from_name(path.name)) is not None
    }
    return sorted(origins)


def infer_tile_grid_crs(tiles_dir: Path) -> rasterio.crs.CRS:
    """Infer the shared folder tile CRS from the first matching TIFF.

    Args:
        tiles_dir (Path): Directory containing folder tile TIFFs.

    Returns:
        rasterio.crs.CRS: Shared tile-grid CRS.

    Examples:
        >>> callable(infer_tile_grid_crs)
        True
    """

    for path in sorted(tiles_dir.iterdir()):
        if not path.is_file() or parse_tile_origin_from_name(path.name) is None:
            continue
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"folder tile has no CRS: {path}")
            return src.crs
    raise ValueError(f"no DOP20 TIFF filenames found under {tiles_dir}")


def build_coverage_grid(
    tile_origins: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one occupancy grid from tile origins.

    Args:
        tile_origins (list[tuple[float, float]]): Tile origins in meters.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Sorted x origins, sorted y
            origins, and a `(len(ys), len(xs))` occupancy array.

    Examples:
        >>> xs, ys, grid = build_coverage_grid([(0.0, 0.0), (1000.0, 0.0)])
        >>> xs.tolist(), ys.tolist(), grid.tolist()
        ([0.0, 1000.0], [0.0], [[1, 1]])
    """

    if not tile_origins:
        raise ValueError("tile_origins must not be empty")
    xs = np.array(sorted({x0 for x0, _ in tile_origins}), dtype=float)
    ys = np.array(sorted({y0 for _, y0 in tile_origins}), dtype=float)
    x_index = {float(x0): idx for idx, x0 in enumerate(xs)}
    y_index = {float(y0): idx for idx, y0 in enumerate(ys)}
    grid = np.zeros((len(ys), len(xs)), dtype=np.uint8)
    for x0, y0 in tile_origins:
        grid[y_index[float(y0)], x_index[float(x0)]] = 1
    return xs, ys, grid


def discover_prediction_paths(
    prediction_dir: Path,
    *,
    prediction_glob: str = "predictions*.tif",
) -> list[Path]:
    """Discover Desktop prediction rasters while skipping the merged mosaic.

    Args:
        prediction_dir (Path): Directory containing prediction rasters.
        prediction_glob (str): Filename glob used for discovery.

    Returns:
        list[Path]: Sorted prediction raster paths excluding `*_merged.tif`.

    Examples:
        >>> callable(discover_prediction_paths)
        True
    """

    return sorted(
        path
        for path in prediction_dir.glob(prediction_glob)
        if path.name != "predictions_merged.tif"
    )


def format_tile_name(x0: float, y0: float) -> str:
    """Format one tile origin back into the canonical DOP20 filename.

    Args:
        x0 (float): Tile-origin easting in meters.
        y0 (float): Tile-origin northing in meters.

    Returns:
        str: Canonical DOP20 TIFF filename.

    Examples:
        >>> format_tile_name(453000.0, 6066000.0)
        'dop20_453000_6066000_1km_20cm.tif'
    """

    return f"dop20_{int(x0)}_{int(y0)}_1km_20cm.tif"


def build_tile_transform(xs: np.ndarray, ys: np.ndarray):
    """Build the shared 1 km transform covering the folder tile grid.

    Args:
        xs (np.ndarray): Sorted tile-origin eastings.
        ys (np.ndarray): Sorted tile-origin northings.

    Returns:
        Affine: Shared folder tile transform.

    Examples:
        >>> transform = build_tile_transform(np.array([0.0, 1000.0]), np.array([0.0, 1000.0]))
        >>> float(transform.c), float(transform.f)
        (0.0, 2000.0)
    """

    return from_origin(
        float(xs.min()),
        float(ys.max()) + TILE_SIZE_M,
        TILE_SIZE_M,
        TILE_SIZE_M,
    )


def _cell_pixels_per_tile(src: rasterio.io.DatasetReader) -> int:
    """Return the approximate number of source pixels inside one 1 km tile.

    Args:
        src (rasterio.io.DatasetReader): Open raster source.

    Returns:
        int: Source-pixel count per 1 km tile.
    """

    res_x = abs(float(src.transform.a))
    res_y = abs(float(src.transform.e))
    if res_x <= 0.0 or res_y <= 0.0:
        raise ValueError(f"invalid raster resolution for count aggregation: {src.name}")
    return max(1, int(round((TILE_SIZE_M / res_x) * (TILE_SIZE_M / res_y))))


def _read_binary_count_grid(
    raster_path: Path,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    target_crs,
    gdal_cache_mb: int = 64,
) -> tuple[Path, np.ndarray]:
    """Read one binary raster as per-tile positive-pixel counts on the folder grid.

    Args:
        raster_path (Path): Binary raster path.
        xs (np.ndarray): Sorted folder tile-origin eastings.
        ys (np.ndarray): Sorted folder tile-origin northings.
        target_crs: CRS of the shared folder tile grid.
        gdal_cache_mb (int): GDAL cache size in MB for the warp.

    Returns:
        tuple[Path, np.ndarray]: Raster path plus a `(len(ys), len(xs))`
            integer count grid aligned to the folder tile grid.
    """

    grid_transform = build_tile_transform(xs, ys)
    with rasterio.Env(GDAL_CACHEMAX=gdal_cache_mb):
        with rasterio.open(raster_path) as src:
            LOGGER.info(
                "Scanning binary count raster: %s (%sx%s, GDAL cache=%s MB)",
                raster_path.name,
                src.width,
                src.height,
                gdal_cache_mb,
            )
            if src.count < 1:
                return raster_path, np.zeros((len(ys), len(xs)), dtype=np.int64)
            pixels_per_tile = _cell_pixels_per_tile(src)
            with WarpedVRT(
                src,
                crs=target_crs,
                transform=grid_transform,
                width=len(xs),
                height=len(ys),
                resampling=Resampling.average,
            ) as vrt:
                coarse = vrt.read(1, out_dtype="float32")
    if not np.isfinite(coarse).all():
        raise ValueError(f"non-finite coarse grid encountered for {raster_path}")
    if float(coarse.min()) < -1e-6 or float(coarse.max()) > 1.0 + 1e-6:
        raise ValueError(
            f"Expected binary 0/1 raster values in {raster_path}, got coarse range "
            f"[{float(coarse.min()):.4f}, {float(coarse.max()):.4f}]"
        )
    coarse = np.clip(coarse, 0.0, 1.0)
    count_grid = np.rint(np.flipud(coarse) * float(pixels_per_tile)).astype(np.int64)
    return raster_path, count_grid


def collect_prediction_count_grid(
    prediction_paths: list[Path],
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    target_crs,
    row_chunk_pixels: int = TILE_SIZE_M,
    max_workers: int = 1,
    gdal_cache_mb: int = 64,
) -> np.ndarray:
    """Collect combined per-tile positive-pixel counts from prediction rasters.

    Args:
        prediction_paths (list[Path]): Prediction rasters to scan.
        xs (np.ndarray): Sorted folder tile-origin eastings.
        ys (np.ndarray): Sorted folder tile-origin northings.
        target_crs: CRS of the shared folder tile grid.
        row_chunk_pixels (int): Retained for CLI compatibility; ignored by the
            coarse resampling implementation.
        max_workers (int): Retained for CLI compatibility; rasters are scanned
            sequentially.
        gdal_cache_mb (int): GDAL cache size in MB used during each warp.

    Returns:
        np.ndarray: Combined positive-pixel count grid aligned to the folder grid.

    Examples:
        >>> callable(collect_prediction_count_grid)
        True
    """

    if row_chunk_pixels <= 0:
        raise ValueError("row_chunk_pixels must be > 0")
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0")
    if gdal_cache_mb <= 0:
        raise ValueError("gdal_cache_mb must be > 0")
    prediction_count_grid = np.zeros((len(ys), len(xs)), dtype=np.int64)
    LOGGER.info(
        (
            "Scanning %s prediction rasters sequentially with %s MB GDAL cache. "
            "--max-workers is retained only for CLI compatibility."
        ),
        len(prediction_paths),
        gdal_cache_mb,
    )
    iterator = (
        tqdm(
            prediction_paths,
            total=len(prediction_paths),
            desc="Prediction rasters",
            unit="raster",
        )
        if tqdm is not None
        else prediction_paths
    )
    for raster_idx, prediction_path in enumerate(iterator, start=1):
        completed_path, count_grid = _read_binary_count_grid(
            prediction_path,
            xs=xs,
            ys=ys,
            target_crs=target_crs,
            gdal_cache_mb=gdal_cache_mb,
        )
        prediction_count_grid += count_grid
        LOGGER.info(
            "Completed prediction raster %s/%s: %s",
            raster_idx,
            len(prediction_paths),
            prediction_path.name,
        )
        LOGGER.info(
            "Finished %s: prediction_positive_tiles=%s cumulative_positive_tiles=%s",
            completed_path.name,
            int(np.count_nonzero(count_grid > 0)),
            int(np.count_nonzero(prediction_count_grid > 0)),
        )
    return prediction_count_grid


def read_planet_count_grid(
    planet_labels_tif: Path,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    target_crs,
    gdal_cache_mb: int = 64,
) -> np.ndarray:
    """Read per-tile positive-pixel counts from the Planet label raster.

    Args:
        planet_labels_tif (Path): Planet label raster path.
        xs (np.ndarray): Sorted folder tile-origin eastings.
        ys (np.ndarray): Sorted folder tile-origin northings.
        target_crs: CRS of the shared folder tile grid.
        gdal_cache_mb (int): GDAL cache size in MB used during the warp.

    Returns:
        np.ndarray: Planet positive-pixel counts aligned to the folder tile grid.

    Examples:
        >>> callable(read_planet_count_grid)
        True
    """

    if not planet_labels_tif.exists():
        raise ValueError(f"planet label raster not found: {planet_labels_tif}")
    _, planet_count_grid = _read_binary_count_grid(
        planet_labels_tif,
        xs=xs,
        ys=ys,
        target_crs=target_crs,
        gdal_cache_mb=gdal_cache_mb,
    )
    return planet_count_grid


def classify_tile_masks(
    *,
    coverage_grid: np.ndarray,
    prediction_count_grid: np.ndarray,
    planet_count_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify folder tiles into blue-only, orange, and violet masks.

    Args:
        coverage_grid (np.ndarray): Folder tile occupancy grid.
        prediction_count_grid (np.ndarray): Combined prediction-positive counts.
        planet_count_grid (np.ndarray): Planet-positive counts.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Blue-only, orange, and
            violet boolean masks.

    Examples:
        >>> blue, orange, violet = classify_tile_masks(
        ...     coverage_grid=np.array([[1, 1, 1]], dtype=np.uint8),
        ...     prediction_count_grid=np.array([[0, 3600, 3601]]),
        ...     planet_count_grid=np.array([[6000, 6000, 6000]]),
        ... )
        >>> blue.tolist(), orange.tolist(), violet.tolist()
        ([[True, False, False]], [[False, True, False]], [[False, False, True]])
    """

    if coverage_grid.shape != prediction_count_grid.shape:
        raise ValueError("prediction_count_grid shape must match coverage_grid")
    if coverage_grid.shape != planet_count_grid.shape:
        raise ValueError("planet_count_grid shape must match coverage_grid")
    coverage_mask = coverage_grid.astype(bool)
    violet_mask = (
        coverage_mask
        & (planet_count_grid > 0)
        & (prediction_count_grid > (planet_count_grid.astype(np.float64) * 0.6))
    )
    orange_mask = coverage_mask & (prediction_count_grid > 0) & ~violet_mask
    blue_only_mask = coverage_mask & (prediction_count_grid <= 0)
    return blue_only_mask, orange_mask, violet_mask


def compute_prediction_planet_ratio_percentages(
    *,
    coverage_grid: np.ndarray,
    prediction_count_grid: np.ndarray,
    planet_count_grid: np.ndarray,
) -> np.ndarray:
    """Compute per-tile prediction-vs-Planet positive-pixel ratios in percent.

    Args:
        coverage_grid (np.ndarray): Folder tile occupancy grid.
        prediction_count_grid (np.ndarray): Combined prediction-positive counts.
        planet_count_grid (np.ndarray): Planet-positive counts.

    Returns:
        np.ndarray: Ratio percentages for covered tiles with Planet positives.

    Examples:
        >>> compute_prediction_planet_ratio_percentages(
        ...     coverage_grid=np.array([[1, 1, 1]], dtype=np.uint8),
        ...     prediction_count_grid=np.array([[0, 3600, 7200]], dtype=np.int64),
        ...     planet_count_grid=np.array([[0, 6000, 6000]], dtype=np.int64),
        ... )
        array([ 60., 120.])
    """

    if coverage_grid.shape != prediction_count_grid.shape:
        raise ValueError("prediction_count_grid shape must match coverage_grid")
    if coverage_grid.shape != planet_count_grid.shape:
        raise ValueError("planet_count_grid shape must match coverage_grid")
    eligible_mask = coverage_grid.astype(bool) & (planet_count_grid > 0)
    if not np.any(eligible_mask):
        return np.array([], dtype=np.float64)
    return (
        prediction_count_grid[eligible_mask].astype(np.float64)
        / planet_count_grid[eligible_mask].astype(np.float64)
        * 100.0
    )


def tile_origins_from_mask(
    mask: np.ndarray,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
) -> list[tuple[float, float]]:
    """Convert one boolean grid mask into sorted tile origins.

    Args:
        mask (np.ndarray): Boolean mask aligned to the folder tile grid.
        xs (np.ndarray): Sorted folder tile-origin eastings.
        ys (np.ndarray): Sorted folder tile-origin northings.

    Returns:
        list[tuple[float, float]]: Sorted tile origins where the mask is true.

    Examples:
        >>> tile_origins_from_mask(
        ...     np.array([[True, False], [False, True]]),
        ...     xs=np.array([0.0, 1000.0]),
        ...     ys=np.array([2000.0, 3000.0]),
        ... )
        [(0.0, 2000.0), (1000.0, 3000.0)]
    """

    if mask.shape != (len(ys), len(xs)):
        raise ValueError("mask shape must match the x/y origins")
    return [
        (float(xs[x_idx]), float(ys[y_idx]))
        for y_idx in range(mask.shape[0])
        for x_idx in range(mask.shape[1])
        if bool(mask[y_idx, x_idx])
    ]


def compute_uncovered_tile_origins(
    tile_origins: list[tuple[float, float]],
    covered_tile_origins: set[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return folder tiles that have no positive prediction coverage.

    Args:
        tile_origins (list[tuple[float, float]]): All folder tile origins.
        covered_tile_origins (set[tuple[float, float]]): Covered tile origins.

    Returns:
        list[tuple[float, float]]: Sorted tile origins with no positive labels.

    Examples:
        >>> compute_uncovered_tile_origins([(0.0, 0.0), (1000.0, 0.0)], {(1000.0, 0.0)})
        [(0.0, 0.0)]
    """

    return sorted(set(tile_origins) - covered_tile_origins)


def write_tile_csv(path: Path, tile_origins: list[tuple[float, float]]) -> None:
    """Write one CSV listing canonical tile names and origins.

    Args:
        path (Path): Output CSV path.
        tile_origins (list[tuple[float, float]]): Tile origins to serialize.

    Examples:
        >>> callable(write_tile_csv)
        True
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tile_name", "x", "y"])
        for x0, y0 in tile_origins:
            writer.writerow([format_tile_name(x0, y0), int(x0), int(y0)])


def _write_summary(
    path: Path,
    *,
    tile_origins: list[tuple[float, float]],
    prediction_positive_tile_origins: set[tuple[float, float]],
    violet_tile_origins: set[tuple[float, float]],
    planet_positive_tile_origins: set[tuple[float, float]],
    ratio_percentages: np.ndarray,
    prediction_paths: list[Path],
    planet_labels_tif: Path,
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    """Write one JSON summary of the plotted coverage.

    Args:
        path (Path): Output JSON path.
        tile_origins (list[tuple[float, float]]): Tile origins used in the map.
        prediction_positive_tile_origins (set[tuple[float, float]]): Tile
            origins with at least one positive prediction pixel.
        violet_tile_origins (set[tuple[float, float]]): Tile origins above the
            Planet-vs-prediction threshold.
        planet_positive_tile_origins (set[tuple[float, float]]): Tile origins
            with at least one Planet-positive pixel.
        ratio_percentages (np.ndarray): Per-tile prediction-vs-Planet ratios.
        prediction_paths (list[Path]): Prediction rasters scanned for labels.
        planet_labels_tif (Path): Planet label raster path.
        xs (np.ndarray): Sorted x origins.
        ys (np.ndarray): Sorted y origins.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    orange_count = len(prediction_positive_tile_origins) - len(violet_tile_origins)
    blue_only_count = len(tile_origins) - len(prediction_positive_tile_origins)
    payload = {
        "tile_count": len(tile_origins),
        "labeled_tile_count": len(prediction_positive_tile_origins),
        "prediction_positive_tile_count": len(prediction_positive_tile_origins),
        "planet_positive_tile_count": len(planet_positive_tile_origins),
        "violet_tile_count": len(violet_tile_origins),
        "orange_tile_count": orange_count,
        "blue_only_tile_count": blue_only_count,
        "uncovered_tile_count": blue_only_count,
        "max_prediction_vs_planet_ratio_percent": (
            None if ratio_percentages.size == 0 else float(np.max(ratio_percentages))
        ),
        "min_x": int(xs.min()),
        "max_x": int(xs.max()),
        "min_y": int(ys.min()),
        "max_y": int(ys.max()),
        "prediction_rasters": [str(path_item) for path_item in prediction_paths],
        "planet_labels_tif": str(planet_labels_tif),
        "bounds": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + TILE_SIZE_M,
            int(ys.max()) + TILE_SIZE_M,
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def plot_coverage_map(
    *,
    coverage_grid: np.ndarray,
    blue_only_mask: np.ndarray,
    orange_mask: np.ndarray,
    violet_mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the folder coverage map with blue, orange, and violet tile classes.

    Args:
        coverage_grid (np.ndarray): Folder tile occupancy grid.
        blue_only_mask (np.ndarray): Tiles with no positive prediction pixels.
        orange_mask (np.ndarray): Tiles with prediction positives below the
            Planet threshold.
        violet_mask (np.ndarray): Tiles above the Planet threshold.
        xs (np.ndarray): Sorted x origins.
        ys (np.ndarray): Sorted y origins.
        output_path (Path): Output PNG path.

    Examples:
        >>> callable(plot_coverage_map)
        True
    """

    if coverage_grid.shape != blue_only_mask.shape:
        raise ValueError("blue_only_mask shape must match coverage_grid")
    if coverage_grid.shape != orange_mask.shape:
        raise ValueError("orange_mask shape must match coverage_grid")
    if coverage_grid.shape != violet_mask.shape:
        raise ValueError("violet_mask shape must match coverage_grid")
    render_grid = np.zeros_like(coverage_grid, dtype=np.uint8)
    render_grid = np.where(coverage_grid > 0, 1, render_grid)
    render_grid = np.where(orange_mask, 2, render_grid)
    render_grid = np.where(violet_mask, 3, render_grid)
    extent = [
        float(xs.min()),
        float(xs.max()) + TILE_SIZE_M,
        float(ys.min()),
        float(ys.max()) + TILE_SIZE_M,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 14))
    cmap = ListedColormap(["#f8fbff", "#2563eb", "#f97316", "#7c3aed"])
    ax.imshow(
        render_grid,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        extent=extent,
        vmin=0,
        vmax=3,
    )
    ax.set_aspect("equal")
    ax.set_title("Folder Coverage With Prediction vs Planet Threshold Overlay")
    ax.set_xlabel("Easting (m, EPSG:25832)")
    ax.set_ylabel("Northing (m, EPSG:25832)")
    ax.legend(
        handles=[
            Patch(facecolor="#f8fbff", edgecolor="black", label="Background"),
            Patch(facecolor="#2563eb", edgecolor="black", label="Folder tile only"),
            Patch(
                facecolor="#f97316",
                edgecolor="black",
                label="Prediction-positive, below Planet threshold",
            ),
            Patch(
                facecolor="#7c3aed",
                edgecolor="black",
                label="Prediction-positive > 60% of Planet positives",
            ),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ratio_distribution(
    ratio_percentages: np.ndarray,
    *,
    output_path: Path,
) -> None:
    """Plot the per-tile prediction-vs-Planet ratio distribution.

    Args:
        ratio_percentages (np.ndarray): Ratio percentages to plot.
        output_path (Path): Output PNG path.

    Examples:
        >>> callable(plot_ratio_distribution)
        True
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    if ratio_percentages.size == 0:
        ax.text(
            0.5,
            0.5,
            "No covered tiles with Planet-positive pixels were found.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        max_ratio = float(np.max(ratio_percentages))
        upper_bound = max(100.0, max_ratio)
        bins = min(60, max(10, int(np.sqrt(ratio_percentages.size))))
        ax.hist(
            ratio_percentages,
            bins=bins,
            range=(0.0, upper_bound),
            color="#7c3aed",
            edgecolor="white",
            alpha=0.9,
        )
        ax.axvline(
            60.0,
            color="#f97316",
            linestyle="--",
            linewidth=2.0,
            label="Violet threshold (60%)",
        )
        ax.set_xlim(0.0, upper_bound)
        ax.set_xlabel("Prediction-positive pixels as % of Planet-positive pixels")
        ax.set_ylabel("Number of tiles")
        ax.legend(loc="upper right", frameon=True)
    ax.set_title("Prediction vs Planet Positive-Pixel Ratio Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Generate the folder coverage map with Planet-vs-prediction overlays.

    Args:
        argv (list[str] | None): Optional CLI argument list.

    Examples:
        >>> callable(main)
        True
    """

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    tiles_dir = Path(args.tiles_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = Path(args.prediction_dir)
    planet_labels_tif = Path(args.planet_labels_tif)

    _configure_logging()
    LOGGER.info("Tiles dir: %s", tiles_dir)
    LOGGER.info("Prediction dir: %s", prediction_dir)
    LOGGER.info("Planet labels tif: %s", planet_labels_tif)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Collecting tile origins from folder filenames.")
    tile_origins = collect_tile_origins(tiles_dir)
    if not tile_origins:
        raise ValueError(f"no DOP20 TIFF filenames found under {tiles_dir}")
    tile_grid_crs = infer_tile_grid_crs(tiles_dir)
    LOGGER.info("Collected %s folder tiles.", len(tile_origins))
    LOGGER.info("Folder tile grid CRS: %s", tile_grid_crs)
    prediction_paths = discover_prediction_paths(
        prediction_dir,
        prediction_glob=str(args.prediction_glob),
    )
    if not prediction_paths:
        raise ValueError(
            f"no prediction rasters matching {args.prediction_glob!r} found under {prediction_dir}"
        )
    LOGGER.info(
        "Discovered %s folder tiles and %s prediction rasters.",
        len(tile_origins),
        len(prediction_paths),
    )

    xs, ys, coverage_grid = build_coverage_grid(tile_origins)
    planet_count_grid = read_planet_count_grid(
        planet_labels_tif,
        xs=xs,
        ys=ys,
        target_crs=tile_grid_crs,
        gdal_cache_mb=int(args.gdal_cache_mb),
    )
    prediction_count_grid = collect_prediction_count_grid(
        prediction_paths,
        xs=xs,
        ys=ys,
        target_crs=tile_grid_crs,
        row_chunk_pixels=int(args.row_chunk_pixels),
        max_workers=int(args.max_workers),
        gdal_cache_mb=int(args.gdal_cache_mb),
    )
    blue_only_mask, orange_mask, violet_mask = classify_tile_masks(
        coverage_grid=coverage_grid,
        prediction_count_grid=prediction_count_grid,
        planet_count_grid=planet_count_grid,
    )
    ratio_percentages = compute_prediction_planet_ratio_percentages(
        coverage_grid=coverage_grid,
        prediction_count_grid=prediction_count_grid,
        planet_count_grid=planet_count_grid,
    )
    prediction_positive_tile_origins = set(
        tile_origins_from_mask(
            orange_mask | violet_mask,
            xs=xs,
            ys=ys,
        )
    )
    violet_tile_origins = set(
        tile_origins_from_mask(
            violet_mask,
            xs=xs,
            ys=ys,
        )
    )
    planet_positive_tile_origins = set(
        tile_origins_from_mask(
            (coverage_grid.astype(bool) & (planet_count_grid > 0)),
            xs=xs,
            ys=ys,
        )
    )
    uncovered_tile_origins = compute_uncovered_tile_origins(
        tile_origins,
        prediction_positive_tile_origins,
    )
    LOGGER.info(
        "Found %s prediction-positive folder tiles.",
        len(prediction_positive_tile_origins),
    )
    LOGGER.info(
        "Found %s violet tiles above the Planet threshold.",
        len(violet_tile_origins),
    )
    LOGGER.info(
        "Found %s orange tiles below the Planet threshold.",
        int(np.count_nonzero(orange_mask)),
    )
    LOGGER.info(
        "Found %s folder tiles without any positive prediction label.",
        len(uncovered_tile_origins),
    )
    LOGGER.info(
        "Computed %s prediction-vs-Planet tile ratios (max=%.2f%%).",
        int(ratio_percentages.size),
        0.0 if ratio_percentages.size == 0 else float(np.max(ratio_percentages)),
    )
    output_path = output_dir / "folder_1_prediction_overlay.png"
    ratio_output_path = (
        output_dir / "folder_1_prediction_vs_planet_ratio_distribution.png"
    )
    summary_path = output_dir / "folder_1_filename_coverage_summary.json"
    covered_csv_path = output_dir / "folder_1_tiles_with_labels.csv"
    uncovered_csv_path = output_dir / "folder_1_tiles_without_labels.csv"
    violet_csv_path = output_dir / "folder_1_tiles_violet.csv"
    plot_coverage_map(
        coverage_grid=coverage_grid,
        blue_only_mask=blue_only_mask,
        orange_mask=orange_mask,
        violet_mask=violet_mask,
        xs=xs,
        ys=ys,
        output_path=output_path,
    )
    plot_ratio_distribution(
        ratio_percentages,
        output_path=ratio_output_path,
    )
    _write_summary(
        summary_path,
        tile_origins=tile_origins,
        prediction_positive_tile_origins=prediction_positive_tile_origins,
        violet_tile_origins=violet_tile_origins,
        planet_positive_tile_origins=planet_positive_tile_origins,
        ratio_percentages=ratio_percentages,
        prediction_paths=prediction_paths,
        planet_labels_tif=planet_labels_tif,
        xs=xs,
        ys=ys,
    )
    write_tile_csv(covered_csv_path, sorted(prediction_positive_tile_origins))
    write_tile_csv(uncovered_csv_path, uncovered_tile_origins)
    write_tile_csv(violet_csv_path, sorted(violet_tile_origins))
    print(f"wrote coverage map -> {output_path}")
    print(f"wrote ratio distribution -> {ratio_output_path}")
    print(f"wrote coverage summary -> {summary_path}")
    print(f"wrote covered-tile CSV -> {covered_csv_path}")
    print(f"wrote uncovered-tile CSV -> {uncovered_csv_path}")
    print(f"wrote violet-tile CSV -> {violet_csv_path}")


if __name__ == "__main__":
    main()
