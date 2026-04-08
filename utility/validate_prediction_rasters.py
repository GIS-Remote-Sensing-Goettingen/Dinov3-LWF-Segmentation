"""Score prediction GeoTIFFs against a label raster on the overlapping area.

The helper is intended for ad hoc validation of previously exported prediction
rasters against one gold-label raster without rebuilding a full cached dataset.
It reprojects each prediction window onto the label grid with nearest-neighbor
sampling and accumulates binary confusion counts over the overlap only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds


def resolve_prediction_paths(patterns: Sequence[str]) -> list[Path]:
    """Resolve one or more literal paths or glob patterns into files.

    Args:
        patterns (Sequence[str]): Literal paths or shell-style glob patterns.

    Returns:
        list[Path]: Sorted, deduplicated prediction raster paths.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     root = Path(d)
        ...     _ = (root / "predictions_a.tif").write_bytes(b"a")
        ...     _ = (root / "predictions_b.tif").write_bytes(b"b")
        ...     paths = resolve_prediction_paths([str(root / "predictions_*.tif")])
        ...     [p.name for p in paths]
        ['predictions_a.tif', 'predictions_b.tif']
    """

    resolved: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = [Path(match).expanduser().resolve() for match in glob.glob(pattern)]
        if not matches:
            candidate = Path(pattern).expanduser()
            if candidate.exists():
                matches = [candidate.resolve()]
        for path in sorted(matches):
            if path in seen:
                continue
            seen.add(path)
            resolved.append(path)
    return resolved


def intersect_bounds(
    left_a: float,
    bottom_a: float,
    right_a: float,
    top_a: float,
    left_b: float,
    bottom_b: float,
    right_b: float,
    top_b: float,
) -> tuple[float, float, float, float] | None:
    """Return the intersection of two bounds tuples.

    Args:
        left_a (float): First bounds left edge.
        bottom_a (float): First bounds bottom edge.
        right_a (float): First bounds right edge.
        top_a (float): First bounds top edge.
        left_b (float): Second bounds left edge.
        bottom_b (float): Second bounds bottom edge.
        right_b (float): Second bounds right edge.
        top_b (float): Second bounds top edge.

    Returns:
        tuple[float, float, float, float] | None: Overlap bounds or ``None``.

    Examples:
        >>> intersect_bounds(0, 0, 4, 4, 2, 2, 6, 6)
        (2.0, 2.0, 4.0, 4.0)
        >>> intersect_bounds(0, 0, 1, 1, 2, 2, 3, 3) is None
        True
    """

    left = max(float(left_a), float(left_b))
    bottom = max(float(bottom_a), float(bottom_b))
    right = min(float(right_a), float(right_b))
    top = min(float(top_a), float(top_b))
    if left >= right or bottom >= top:
        return None
    return (left, bottom, right, top)


def normalize_window(window: Window, width: int, height: int) -> Window:
    """Clamp a floating-point window to valid integer dataset bounds.

    Args:
        window (Window): Floating-point source window.
        width (int): Dataset width in pixels.
        height (int): Dataset height in pixels.

    Returns:
        Window: Integer-clamped window.
    """

    col_off = max(0, int(math.floor(float(window.col_off))))
    row_off = max(0, int(math.floor(float(window.row_off))))
    col_end = min(width, int(math.ceil(float(window.col_off + window.width))))
    row_end = min(height, int(math.ceil(float(window.row_off + window.height))))
    return Window(
        col_off=col_off,
        row_off=row_off,
        width=max(0, col_end - col_off),
        height=max(0, row_end - row_off),
    )


def iter_windows(window: Window, chunk_size: int) -> Iterable[Window]:
    """Yield chunked subwindows covering one parent window.

    Args:
        window (Window): Parent window.
        chunk_size (int): Max chunk edge length in pixels.

    Yields:
        Window: Chunk window.
    """

    row_stop = int(window.row_off + window.height)
    col_stop = int(window.col_off + window.width)
    for row_off in range(int(window.row_off), row_stop, int(chunk_size)):
        chunk_height = min(int(chunk_size), row_stop - row_off)
        for col_off in range(int(window.col_off), col_stop, int(chunk_size)):
            chunk_width = min(int(chunk_size), col_stop - col_off)
            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=chunk_width,
                height=chunk_height,
            )


def update_binary_counts(
    counts: dict[str, int],
    pred_chunk: np.ndarray,
    label_chunk: np.ndarray,
    positive_prediction: int,
    positive_label: int,
    ignore_label_values: set[int],
) -> None:
    """Update binary confusion counts for one aligned raster chunk.

    Args:
        counts (dict[str, int]): Mutable confusion-count accumulator.
        pred_chunk (np.ndarray): Prediction chunk on the label grid.
        label_chunk (np.ndarray): Label chunk on the label grid.
        positive_prediction (int): Value treated as predicted foreground.
        positive_label (int): Value treated as label foreground.
        ignore_label_values (set[int]): Label values excluded from scoring.
    """

    label_int = np.asarray(label_chunk).astype(np.int64, copy=False)
    pred_int = np.asarray(pred_chunk).astype(np.int64, copy=False)
    valid_mask = np.ones(label_int.shape, dtype=bool)
    if ignore_label_values:
        valid_mask &= ~np.isin(label_int, list(ignore_label_values))
    if not valid_mask.any():
        return

    label_pos = label_int == int(positive_label)
    pred_pos = pred_int == int(positive_prediction)

    tp = int(np.count_nonzero(valid_mask & pred_pos & label_pos))
    tn = int(np.count_nonzero(valid_mask & ~pred_pos & ~label_pos))
    fp = int(np.count_nonzero(valid_mask & pred_pos & ~label_pos))
    fn = int(np.count_nonzero(valid_mask & ~pred_pos & label_pos))

    counts["tp"] += tp
    counts["tn"] += tn
    counts["fp"] += fp
    counts["fn"] += fn
    counts["valid_pixels"] += int(np.count_nonzero(valid_mask))
    counts["label_positive_pixels"] += int(np.count_nonzero(valid_mask & label_pos))
    counts["prediction_positive_pixels"] += int(np.count_nonzero(valid_mask & pred_pos))


def compute_binary_metrics(counts: dict[str, int]) -> dict[str, float]:
    """Compute scalar binary metrics from confusion counts.

    Args:
        counts (dict[str, int]): Confusion counts.

    Returns:
        dict[str, float]: Scalar metrics.

    Examples:
        >>> compute_binary_metrics(
        ...     {
        ...         "tp": 4,
        ...         "tn": 3,
        ...         "fp": 1,
        ...         "fn": 2,
        ...         "valid_pixels": 10,
        ...         "label_positive_pixels": 6,
        ...         "prediction_positive_pixels": 5,
        ...     }
        ... )["iou"]
        0.5714285714285714
    """

    tp = float(counts["tp"])
    tn = float(counts["tn"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])
    valid = float(counts["valid_pixels"])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / valid if valid > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = (2.0 * tp) / ((2.0 * tp) + fp + fn) if ((2.0 * tp) + fp + fn) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "iou": iou,
        "dice": dice,
        "f1": dice,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "label_positive_rate": (
            float(counts["label_positive_pixels"]) / valid if valid > 0 else 0.0
        ),
        "prediction_positive_rate": (
            float(counts["prediction_positive_pixels"]) / valid if valid > 0 else 0.0
        ),
    }


def score_prediction_raster(
    prediction_path: Path,
    label_path: Path,
    *,
    positive_prediction: int = 1,
    positive_label: int = 1,
    ignore_label_values: Sequence[int] = (),
    chunk_size: int = 2048,
) -> dict[str, Any]:
    """Score one prediction raster against the overlapping label area.

    Args:
        prediction_path (Path): Prediction raster path.
        label_path (Path): Label raster path.
        positive_prediction (int): Foreground prediction value.
        positive_label (int): Foreground label value.
        ignore_label_values (Sequence[int]): Label values excluded from scoring.
        chunk_size (int): Chunk edge length used for windowed reads.

    Returns:
        dict[str, Any]: Per-file scoring summary.
    """

    counts = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "valid_pixels": 0,
        "label_positive_pixels": 0,
        "prediction_positive_pixels": 0,
    }
    ignore_set = {int(value) for value in ignore_label_values}
    with (
        rasterio.open(label_path) as label_src,
        rasterio.open(prediction_path) as pred_src,
    ):
        if label_src.crs is None or pred_src.crs is None:
            raise ValueError("Both prediction and label rasters must define a CRS.")
        pred_bounds_in_label = (
            pred_src.bounds
            if pred_src.crs == label_src.crs
            else transform_bounds(
                pred_src.crs,
                label_src.crs,
                *pred_src.bounds,
                densify_pts=21,
            )
        )
        overlap_bounds = intersect_bounds(
            label_src.bounds.left,
            label_src.bounds.bottom,
            label_src.bounds.right,
            label_src.bounds.top,
            pred_bounds_in_label[0],
            pred_bounds_in_label[1],
            pred_bounds_in_label[2],
            pred_bounds_in_label[3],
        )
        result: dict[str, Any] = {
            "prediction_path": str(prediction_path),
            "label_path": str(label_path),
            "status": "ok",
            "prediction_crs": pred_src.crs.to_string(),
            "label_crs": label_src.crs.to_string(),
            "prediction_resolution": tuple(float(v) for v in pred_src.res),
            "label_resolution": tuple(float(v) for v in label_src.res),
        }
        if overlap_bounds is None:
            result.update(
                {
                    "status": "no_overlap",
                    "counts": counts,
                    "metrics": compute_binary_metrics(counts),
                    "overlap_bounds": None,
                    "overlap_pixels": 0,
                    "label_coverage_fraction": 0.0,
                }
            )
            return result

        label_window = normalize_window(
            from_bounds(*overlap_bounds, transform=label_src.transform),
            width=label_src.width,
            height=label_src.height,
        )
        if int(label_window.width) <= 0 or int(label_window.height) <= 0:
            result.update(
                {
                    "status": "no_overlap",
                    "counts": counts,
                    "metrics": compute_binary_metrics(counts),
                    "overlap_bounds": None,
                    "overlap_pixels": 0,
                    "label_coverage_fraction": 0.0,
                }
            )
            return result

        overlap_bounds_exact = window_bounds(label_window, label_src.transform)
        vrt_options = {
            "crs": label_src.crs,
            "transform": label_src.transform,
            "width": label_src.width,
            "height": label_src.height,
            "resampling": Resampling.nearest,
        }
        with WarpedVRT(pred_src, **vrt_options) as pred_vrt:
            for chunk_window in iter_windows(label_window, chunk_size):
                label_chunk = label_src.read(1, window=chunk_window, boundless=False)
                pred_chunk = pred_vrt.read(1, window=chunk_window, boundless=False)
                update_binary_counts(
                    counts,
                    pred_chunk=pred_chunk,
                    label_chunk=label_chunk,
                    positive_prediction=positive_prediction,
                    positive_label=positive_label,
                    ignore_label_values=ignore_set,
                )

        overlap_pixel_count = int(label_window.width) * int(label_window.height)
        result.update(
            {
                "counts": counts,
                "metrics": compute_binary_metrics(counts),
                "overlap_bounds": tuple(float(v) for v in overlap_bounds_exact),
                "overlap_pixels": overlap_pixel_count,
                "label_coverage_fraction": overlap_pixel_count
                / float(label_src.width * label_src.height),
            }
        )
        return result


def validate_prediction_rasters(
    label_path: Path,
    prediction_patterns: Sequence[str],
    *,
    positive_prediction: int = 1,
    positive_label: int = 1,
    ignore_label_values: Sequence[int] = (),
    chunk_size: int = 2048,
) -> dict[str, Any]:
    """Validate prediction rasters against one label raster.

    Args:
        label_path (Path): Label raster path.
        prediction_patterns (Sequence[str]): Prediction file paths or globs.
        positive_prediction (int): Foreground prediction value.
        positive_label (int): Foreground label value.
        ignore_label_values (Sequence[int]): Label values excluded from scoring.
        chunk_size (int): Chunk edge length used for windowed reads.

    Returns:
        dict[str, Any]: Run summary with one result per prediction raster.
    """

    prediction_paths = resolve_prediction_paths(prediction_patterns)
    if not prediction_paths:
        raise FileNotFoundError(
            "No prediction rasters matched the provided paths/patterns."
        )
    label_path = label_path.expanduser().resolve()
    results = [
        score_prediction_raster(
            prediction_path=path,
            label_path=label_path,
            positive_prediction=positive_prediction,
            positive_label=positive_label,
            ignore_label_values=ignore_label_values,
            chunk_size=chunk_size,
        )
        for path in prediction_paths
    ]
    return {
        "label_path": str(label_path),
        "prediction_count": len(results),
        "results": results,
    }


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    """Write the validation report to a JSON file.

    Args:
        report (dict[str, Any]): Validation summary.
        output_path (Path): JSON output path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def print_report(report: dict[str, Any]) -> None:
    """Print a compact text report to stdout.

    Args:
        report (dict[str, Any]): Validation summary.
    """

    print(f"Label raster: {report['label_path']}")
    print(f"Prediction rasters: {int(report['prediction_count'])}")
    for item in report["results"]:
        name = Path(str(item["prediction_path"])).name
        print("")
        print(name)
        print(
            "  status={status} overlap_pixels={pixels} label_coverage={coverage:.4f}".format(
                status=item["status"],
                pixels=int(item["overlap_pixels"]),
                coverage=float(item["label_coverage_fraction"]),
            )
        )
        metrics = item["metrics"]
        print(
            "  iou={iou:.6f} dice={dice:.6f} precision={precision:.6f} recall={recall:.6f}".format(
                iou=float(metrics["iou"]),
                dice=float(metrics["dice"]),
                precision=float(metrics["precision"]),
                recall=float(metrics["recall"]),
            )
        )
        counts = item["counts"]
        print(
            "  tp={tp} fp={fp} fn={fn} tn={tn} pred_pos={pred_pos} label_pos={label_pos}".format(
                tp=int(counts["tp"]),
                fp=int(counts["fp"]),
                fn=int(counts["fn"]),
                tn=int(counts["tn"]),
                pred_pos=int(counts["prediction_positive_pixels"]),
                label_pos=int(counts["label_positive_pixels"]),
            )
        )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-raster",
        type=Path,
        required=True,
        help="Ground-truth label raster used for scoring.",
    )
    parser.add_argument(
        "predictions",
        nargs="+",
        help="Prediction raster paths or glob patterns.",
    )
    parser.add_argument(
        "--positive-prediction",
        type=int,
        default=1,
        help="Prediction value treated as foreground.",
    )
    parser.add_argument(
        "--positive-label",
        type=int,
        default=1,
        help="Label value treated as foreground.",
    )
    parser.add_argument(
        "--ignore-label-value",
        type=int,
        action="append",
        default=[],
        help="Optional label value to exclude from scoring. Repeat as needed.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk edge length used for windowed reads.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON path for the full validation report.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the prediction-raster validation CLI.

    Returns:
        int: Process exit code.

    Examples:
        >>> callable(main)
        True
    """

    args = _parse_args()
    report = validate_prediction_rasters(
        label_path=args.label_raster,
        prediction_patterns=args.predictions,
        positive_prediction=int(args.positive_prediction),
        positive_label=int(args.positive_label),
        ignore_label_values=[int(value) for value in args.ignore_label_value],
        chunk_size=int(args.chunk_size),
    )
    print_report(report)
    if args.output_json is not None:
        write_json_report(report, args.output_json.expanduser().resolve())
        print(f"\nWrote JSON report to {args.output_json.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
