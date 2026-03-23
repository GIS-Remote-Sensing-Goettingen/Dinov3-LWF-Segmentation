"""Sample MD DOP acquisition dates on a coarse 10 km grid and plot them.

Examples:
    >>> callable(main)
    True
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "md_dop_date_distribution"
DEFAULT_SPACING_M = 10_000


def _load_get_data_api_module():
    """Load the downloader helper module from disk.

    Returns:
        object: Imported module object.
    """

    module_path = REPO_ROOT / "utility" / "get_data_api.py"
    spec = importlib.util.spec_from_file_location("get_data_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GET_DATA_API = _load_get_data_api_module()


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the metadata sampling script.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the CSV and PNG outputs will be written.",
    )
    parser.add_argument(
        "--spacing-m",
        type=int,
        default=DEFAULT_SPACING_M,
        help="Sampling spacing in meters across the snapped AOI grid.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=20,
        help="Per-request timeout in seconds for metadata queries.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of sampled grid points.",
    )
    return parser


def _build_sampling_origins(
    *,
    spacing_m: int,
    max_samples: int | None = None,
) -> list[tuple[float, float]]:
    """Build snapped tile origins at the requested coarse spacing.

    Args:
        spacing_m (int): Sampling spacing in meters.
        max_samples (int | None): Optional maximum number of origins.

    Returns:
        list[tuple[float, float]]: Ordered `(x0, y0)` tile origins.
    """

    if spacing_m <= 0:
        raise ValueError("spacing_m must be > 0")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided")

    _, _, _, _, gx0, gy0, gx1, gy1 = GET_DATA_API.project_and_snap_bbox(
        bbox_ll=GET_DATA_API.BBOX_LL
    )
    origins = [
        (float(x0), float(y0))
        for x0 in range(int(gx0), int(gx1), spacing_m)
        for y0 in range(int(gy0), int(gy1), spacing_m)
    ]
    return origins if max_samples is None else origins[:max_samples]


def _write_sample_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write sampled metadata rows to CSV.

    Args:
        path (Path): Output CSV path.
        rows (list[dict[str, str]]): Metadata rows.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "x0",
        "y0",
        "status_code",
        "content_type",
        "metadata_error",
        "acquisition_date",
        "season_ok",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_date_distribution(path: Path, rows: list[dict[str, str]]) -> None:
    """Plot one month-level distribution of sampled acquisition dates.

    Args:
        path (Path): Output PNG path.
        rows (list[dict[str, str]]): Sampled metadata rows.
    """

    month_counts = Counter()
    for row in rows:
        acquisition_date = row.get("acquisition_date") or ""
        if not acquisition_date:
            continue
        month_counts[acquisition_date[:7]] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    if month_counts:
        labels = sorted(month_counts)
        values = [month_counts[label] for label in labels]
        ax.bar(labels, values, color="#2d6a4f")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "No acquisition dates found", ha="center", va="center")
        ax.set_xticks([])
    ax.set_title("MD DOP Acquisition-Date Distribution")
    ax.set_ylabel("Sample count")
    ax.set_xlabel("Acquisition month")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Run the 10 km metadata sampling script.

    Args:
        argv (list[str] | None): Optional CLI argument list.

    Examples:
        >>> callable(main)
        True
    """

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    origins = _build_sampling_origins(
        spacing_m=int(args.spacing_m),
        max_samples=args.max_samples,
    )
    session = GET_DATA_API.make_session()
    rows: list[dict[str, str]] = []
    for x0, y0 in origins:
        result = GET_DATA_API.fetch_tile_metadata(
            session,
            x0,
            y0,
            timeout_s=int(args.timeout_s),
        )
        rows.append(
            {
                "x0": str(int(x0)),
                "y0": str(int(y0)),
                "status_code": str(result["status_code"]),
                "content_type": str(result["content_type"]),
                "metadata_error": str(result["metadata_error"] or ""),
                "acquisition_date": str(result["acquisition_date"] or ""),
                "season_ok": str(bool(result["season_ok"])),
            }
        )

    csv_path = output_dir / "md_dop_date_samples.csv"
    plot_path = output_dir / "md_dop_date_distribution.png"
    _write_sample_csv(csv_path, rows)
    _plot_date_distribution(plot_path, rows)
    print(f"wrote {len(rows)} samples -> {csv_path}")
    print(f"wrote plot -> {plot_path}")


if __name__ == "__main__":
    main()
