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
from rasterio import Affine  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.vrt import WarpedVRT  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

DEFAULT_TILES_DIR = Path("/home/mak/cluster_hdd/patches_mt/folder_1")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "folder_1_coverage"
DEFAULT_PREDICTION_DIR = Path("/home/mak/Desktop")
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


def _tile_origins_from_resampled_grid(
    *,
    coarse_grid: np.ndarray,
    x_origins: np.ndarray,
    y_origins: np.ndarray,
) -> set[tuple[float, float]]:
    """Extract 1 km tile origins with any positive label from a coarse grid.

    Args:
        coarse_grid (np.ndarray): Coarse 1 km grid after max-resampling.
        x_origins (np.ndarray): Tile-origin x coordinates for the grid columns.
        y_origins (np.ndarray): Tile-origin y coordinates for the grid rows.

    Returns:
        set[tuple[float, float]]: Tile origins with any positive pixels.

    Examples:
        >>> grid = np.array([[1, 0], [0, 2]], dtype=np.uint8)
        >>> sorted(
        ...     _tile_origins_from_resampled_grid(
        ...         coarse_grid=grid,
        ...         x_origins=np.array([0.0, 1000.0]),
        ...         y_origins=np.array([2000.0, 3000.0]),
        ...     )
        ... )
        [(0.0, 2000.0), (1000.0, 3000.0)]
    """

    if coarse_grid.ndim != 2:
        raise ValueError("coarse_grid must be 2D")
    if coarse_grid.shape != (len(y_origins), len(x_origins)):
        raise ValueError("coarse_grid shape must match the x/y origins")
    return {
        (float(x_origins[x_idx]), float(y_origins[y_idx]))
        for y_idx in range(coarse_grid.shape[0])
        for x_idx in range(coarse_grid.shape[1])
        if int(coarse_grid[y_idx, x_idx]) > 0
    }


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


def _scan_prediction_raster(
    prediction_path: Path,
    *,
    valid_tile_origins: set[tuple[float, float]] | None = None,
    gdal_cache_mb: int = 64,
) -> tuple[Path, set[tuple[float, float]], int, int]:
    """Scan one prediction raster and return covered tile origins.

    Args:
        prediction_path (Path): Prediction raster to scan.
        valid_tile_origins (set[tuple[float, float]] | None): Optional allowed
            tile-origin filter.
        gdal_cache_mb (int): GDAL cache size in MB for the warp.

    Returns:
        tuple[Path, set[tuple[float, float]], int, int]: Raster path, covered
            tile origins, coarse width, and coarse height.
    """

    with rasterio.Env(GDAL_CACHEMAX=gdal_cache_mb):
        with rasterio.open(prediction_path) as src:
            LOGGER.info(
                "Scanning prediction raster: %s (%sx%s, GDAL cache=%s MB)",
                prediction_path.name,
                src.width,
                src.height,
                gdal_cache_mb,
            )
            if src.count < 1:
                return prediction_path, set(), 0, 0
            if src.width % TILE_SIZE_M != 0 or src.height % TILE_SIZE_M != 0:
                raise ValueError(
                    "prediction raster dimensions must align to 1 km tiles: "
                    f"{prediction_path}"
                )
            coarse_width = src.width // TILE_SIZE_M
            coarse_height = src.height // TILE_SIZE_M
            x_origins = np.arange(
                float(src.bounds.left),
                float(src.bounds.right),
                TILE_SIZE_M,
                dtype=float,
            )
            y_origins = np.arange(
                float(src.bounds.bottom),
                float(src.bounds.top),
                TILE_SIZE_M,
                dtype=float,
            )
            coarse_transform = src.transform * Affine.scale(TILE_SIZE_M, TILE_SIZE_M)
            with WarpedVRT(
                src,
                crs=src.crs,
                transform=coarse_transform,
                width=coarse_width,
                height=coarse_height,
                resampling=Resampling.max,
            ) as vrt:
                coarse = vrt.read(1)
    chunk_origins = _tile_origins_from_resampled_grid(
        coarse_grid=np.flipud(coarse),
        x_origins=x_origins,
        y_origins=y_origins,
    )
    if valid_tile_origins is not None:
        chunk_origins &= valid_tile_origins
    return prediction_path, chunk_origins, coarse_width, coarse_height


def collect_labeled_tile_origins(
    prediction_paths: list[Path],
    *,
    valid_tile_origins: set[tuple[float, float]] | None = None,
    row_chunk_pixels: int = TILE_SIZE_M,
    max_workers: int = 1,
    gdal_cache_mb: int = 64,
) -> set[tuple[float, float]]:
    """Collect 1 km tile origins containing any positive prediction labels.

    Args:
        prediction_paths (list[Path]): Prediction rasters to scan.
        valid_tile_origins (set[tuple[float, float]] | None): Optional set of
            tile origins to keep in the final overlay.
        row_chunk_pixels (int): Retained for CLI compatibility; ignored by the
            coarse resampling implementation.
        max_workers (int): Retained for CLI compatibility; rasters are scanned
            sequentially.
        gdal_cache_mb (int): GDAL cache size in MB used during each warp.

    Returns:
        set[tuple[float, float]]: Tile origins with at least one positive label.

    Examples:
        >>> callable(collect_labeled_tile_origins)
        True
    """

    if row_chunk_pixels <= 0:
        raise ValueError("row_chunk_pixels must be > 0")
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0")
    if gdal_cache_mb <= 0:
        raise ValueError("gdal_cache_mb must be > 0")
    labeled_tile_origins: set[tuple[float, float]] = set()
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
        completed_path, chunk_origins, coarse_width, coarse_height = (
            _scan_prediction_raster(
                prediction_path,
                valid_tile_origins=valid_tile_origins,
                gdal_cache_mb=gdal_cache_mb,
            )
        )
        LOGGER.info(
            "Completed prediction raster %s/%s: %s",
            raster_idx,
            len(prediction_paths),
            prediction_path.name,
        )
        labeled_tile_origins.update(chunk_origins)
        LOGGER.info(
            "Finished %s: coarse grid=%sx%s labeled_tiles=%s cumulative=%s",
            completed_path.name,
            coarse_width,
            coarse_height,
            len(chunk_origins),
            len(labeled_tile_origins),
        )
    return labeled_tile_origins


def compute_uncovered_tile_origins(
    tile_origins: list[tuple[float, float]],
    labeled_tile_origins: set[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return folder tiles that have no positive prediction coverage.

    Args:
        tile_origins (list[tuple[float, float]]): All folder tile origins.
        labeled_tile_origins (set[tuple[float, float]]): Covered tile origins.

    Returns:
        list[tuple[float, float]]: Sorted tile origins with no positive labels.

    Examples:
        >>> compute_uncovered_tile_origins([(0.0, 0.0), (1000.0, 0.0)], {(1000.0, 0.0)})
        [(0.0, 0.0)]
    """

    return sorted(set(tile_origins) - labeled_tile_origins)


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
    labeled_tile_origins: set[tuple[float, float]],
    prediction_paths: list[Path],
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    """Write one JSON summary of the plotted coverage.

    Args:
        path (Path): Output JSON path.
        tile_origins (list[tuple[float, float]]): Tile origins used in the map.
        labeled_tile_origins (set[tuple[float, float]]): Tile origins with at
            least one positive prediction label.
        prediction_paths (list[Path]): Prediction rasters scanned for labels.
        xs (np.ndarray): Sorted x origins.
        ys (np.ndarray): Sorted y origins.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tile_count": len(tile_origins),
        "labeled_tile_count": len(labeled_tile_origins),
        "uncovered_tile_count": len(tile_origins) - len(labeled_tile_origins),
        "min_x": int(xs.min()),
        "max_x": int(xs.max()),
        "min_y": int(ys.min()),
        "max_y": int(ys.max()),
        "prediction_rasters": [str(path_item) for path_item in prediction_paths],
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
    tile_origins: list[tuple[float, float]],
    labeled_tile_origins: set[tuple[float, float]],
    output_path: Path,
) -> None:
    """Plot one blue occupancy map from tile origins.

    Args:
        tile_origins (list[tuple[float, float]]): Tile origins used in the map.
        labeled_tile_origins (set[tuple[float, float]]): Tile origins with at
            least one positive prediction label.
        output_path (Path): Output PNG path.

    Examples:
        >>> callable(plot_coverage_map)
        True
    """

    xs, ys, grid = build_coverage_grid(tile_origins)
    labeled_grid = np.zeros_like(grid)
    x_index = {float(x0): idx for idx, x0 in enumerate(xs)}
    y_index = {float(y0): idx for idx, y0 in enumerate(ys)}
    for x0, y0 in labeled_tile_origins:
        if float(x0) not in x_index or float(y0) not in y_index:
            continue
        labeled_grid[y_index[float(y0)], x_index[float(x0)]] = 1
    render_grid = np.where(labeled_grid == 1, 2, grid)
    extent = [
        float(xs.min()),
        float(xs.max()) + TILE_SIZE_M,
        float(ys.min()),
        float(ys.max()) + TILE_SIZE_M,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = ListedColormap(["#f8fbff", "#2563eb", "#f97316"])
    ax.imshow(
        render_grid,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        extent=extent,
        vmin=0,
        vmax=2,
    )
    ax.set_aspect("equal")
    ax.set_title("Folder Coverage With Prediction-Label Overlay")
    ax.set_xlabel("Easting (m, EPSG:25832)")
    ax.set_ylabel("Northing (m, EPSG:25832)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Generate the blue coverage map for one folder of DOP20 tiles.

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

    _configure_logging()
    LOGGER.info("Tiles dir: %s", tiles_dir)
    LOGGER.info("Prediction dir: %s", prediction_dir)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Collecting tile origins from folder filenames.")
    tile_origins = collect_tile_origins(tiles_dir)
    if not tile_origins:
        raise ValueError(f"no DOP20 TIFF filenames found under {tiles_dir}")
    LOGGER.info("Collected %s folder tiles.", len(tile_origins))
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

    xs, ys, _ = build_coverage_grid(tile_origins)
    labeled_tile_origins = collect_labeled_tile_origins(
        prediction_paths,
        valid_tile_origins=set(tile_origins),
        row_chunk_pixels=int(args.row_chunk_pixels),
        max_workers=int(args.max_workers),
        gdal_cache_mb=int(args.gdal_cache_mb),
    )
    uncovered_tile_origins = compute_uncovered_tile_origins(
        tile_origins,
        labeled_tile_origins,
    )
    LOGGER.info(
        "Found %s covered tiles with at least one positive prediction label.",
        len(labeled_tile_origins),
    )
    LOGGER.info(
        "Found %s folder tiles without any positive prediction label.",
        len(uncovered_tile_origins),
    )
    output_path = output_dir / "folder_1_prediction_overlay.png"
    summary_path = output_dir / "folder_1_filename_coverage_summary.json"
    covered_csv_path = output_dir / "folder_1_tiles_with_labels.csv"
    uncovered_csv_path = output_dir / "folder_1_tiles_without_labels.csv"
    plot_coverage_map(
        tile_origins=tile_origins,
        labeled_tile_origins=labeled_tile_origins,
        output_path=output_path,
    )
    _write_summary(
        summary_path,
        tile_origins=tile_origins,
        labeled_tile_origins=labeled_tile_origins,
        prediction_paths=prediction_paths,
        xs=xs,
        ys=ys,
    )
    write_tile_csv(covered_csv_path, sorted(labeled_tile_origins))
    write_tile_csv(uncovered_csv_path, uncovered_tile_origins)
    print(f"wrote coverage map -> {output_path}")
    print(f"wrote coverage summary -> {summary_path}")
    print(f"wrote covered-tile CSV -> {covered_csv_path}")
    print(f"wrote uncovered-tile CSV -> {uncovered_csv_path}")


if __name__ == "__main__":
    main()
