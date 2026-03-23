"""Recover missing DOP20 tiles into resumable staging shards.

Examples:
    >>> tile_name_from_origin(453000.0, 6066000.0)
    'dop20_453000_6066000_1km_20cm.tif'
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility import get_data_api  # noqa: E402

logger = logging.getLogger(__name__)
TILE_NAME_RE = re.compile(r"^dop20_(\d+)_(\d+)_1km_20cm\.tif$")
FINAL_STATUSES = frozenset(
    {"DONE", "SKIP", "SKIP_SEASON", "FAIL_METADATA", "FAIL_BLANK"}
)
DEFAULT_PRESENT_DIR = Path(
    "/mnt/ceph-hdd/projects/mthesis_davide_mattioli/patches_mt/folder_1"
)
DEFAULT_STAGING_PARENT = Path(
    "/mnt/ceph-hdd/projects/mthesis_davide_mattioli/patches_mt"
)
DEFAULT_BATCH_SIZE = 2000
DEFAULT_MAX_WORKERS = 4
DEFAULT_COVERAGE_THRESHOLD = 0.99


def tile_name_from_origin(x0: float, y0: float) -> str:
    """Return the canonical tile filename for one 1 km origin.

    Args:
        x0 (float): Tile origin x coordinate.
        y0 (float): Tile origin y coordinate.

    Returns:
        str: Canonical tile filename.

    Examples:
        >>> tile_name_from_origin(1.2, 3.9)
        'dop20_1_3_1km_20cm.tif'
    """

    return f"dop20_{int(x0)}_{int(y0)}_1km_20cm.tif"


def parse_tile_name(name: str) -> tuple[float, float] | None:
    """Parse one canonical tile filename into its origin coordinates.

    Args:
        name (str): Tile filename.

    Returns:
        tuple[float, float] | None: Tile origin or None when not a tile file.

    Examples:
        >>> parse_tile_name('dop20_453000_6066000_1km_20cm.tif')
        (453000.0, 6066000.0)
        >>> parse_tile_name('notes.txt') is None
        True
    """

    match = TILE_NAME_RE.match(name)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def _load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load one JSON file or return a fallback mapping.

    Args:
        path (Path): JSON file path.
        default (dict[str, Any] | None): Optional fallback mapping.

    Returns:
        dict[str, Any]: Parsed or fallback mapping.
    """

    if not path.exists():
        return {} if default is None else dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON mapping with stable formatting.

    Args:
        path (Path): Output JSON path.
        payload (dict[str, Any]): Mapping payload.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _configure_logging(log_file: Path) -> None:
    """Configure console and file logging for one recovery run.

    Args:
        log_file (Path): Output log file path.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a"), logging.StreamHandler()],
        force=True,
    )


def discover_present_tiles(present_dir: Path) -> dict[tuple[float, float], Path]:
    """Discover canonical present tiles from a directory.

    Args:
        present_dir (Path): Directory scanned for downloaded tiles.

    Returns:
        dict[tuple[float, float], Path]: Mapping from tile origin to file path.

    Examples:
        >>> callable(discover_present_tiles)
        True
    """

    tiles: dict[tuple[float, float], Path] = {}
    if not present_dir.exists():
        return tiles
    for path in sorted(present_dir.glob("dop20_*_1km_20cm.tif")):
        origin = parse_tile_name(path.name)
        if origin is None:
            continue
        tiles[origin] = path
    return tiles


def chunk_tile_origins(
    tile_origins: list[tuple[float, float]],
    batch_size: int,
) -> list[list[tuple[float, float]]]:
    """Split tile origins into contiguous fixed-size shards.

    Args:
        tile_origins (list[tuple[float, float]]): Ordered tile origins.
        batch_size (int): Maximum tiles per shard.

    Returns:
        list[list[tuple[float, float]]]: Contiguous origin chunks.

    Examples:
        >>> chunk_tile_origins([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], 2)
        [[(0.0, 0.0), (1.0, 1.0)], [(2.0, 2.0)]]
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [
        tile_origins[idx : idx + batch_size]
        for idx in range(0, len(tile_origins), batch_size)
    ]


def _write_tile_csv(
    path: Path,
    tile_origins: list[tuple[float, float]],
    *,
    include_name: bool = True,
) -> None:
    """Write one tile-origin CSV file.

    Args:
        path (Path): Output CSV path.
        tile_origins (list[tuple[float, float]]): Tile-origin rows.
        include_name (bool): Whether to include the canonical tile name column.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        headers = ["x", "y"]
        if include_name:
            headers.append("tile_name")
        writer.writerow(headers)
        for x0, y0 in tile_origins:
            row: list[object] = [int(x0), int(y0)]
            if include_name:
                row.append(tile_name_from_origin(x0, y0))
            writer.writerow(row)


def _write_tile_manifest(path: Path, tile_origins: list[tuple[float, float]]) -> None:
    """Write one newline-delimited tile-origin manifest.

    Args:
        path (Path): Output manifest path.
        tile_origins (list[tuple[float, float]]): Ordered tile origins.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(x0)},{int(y0)}" for x0, y0 in tile_origins]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_tile_manifest(path: Path) -> list[tuple[float, float]]:
    """Read one newline-delimited tile-origin manifest.

    Args:
        path (Path): Manifest path.

    Returns:
        list[tuple[float, float]]: Ordered tile origins.
    """

    return [
        get_data_api.parse_tile_origin(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _recovery_batch_dir(staging_root: Path, batch_idx: int) -> Path:
    """Return one batch staging directory.

    Args:
        staging_root (Path): Recovery root path.
        batch_idx (int): Zero-based batch index.

    Returns:
        Path: Batch directory.
    """

    return staging_root / "batches" / f"batch_{batch_idx:03d}"


def _results_path(batch_dir: Path) -> Path:
    """Return one batch results-jsonl path.

    Args:
        batch_dir (Path): Batch directory.

    Returns:
        Path: Results path.
    """

    return batch_dir / "results.jsonl"


def _summary_path(batch_dir: Path) -> Path:
    """Return one batch summary path.

    Args:
        batch_dir (Path): Batch directory.

    Returns:
        Path: Summary path.
    """

    return batch_dir / "summary.json"


def _tiles_dir(batch_dir: Path) -> Path:
    """Return one batch tile-output directory.

    Args:
        batch_dir (Path): Batch directory.

    Returns:
        Path: Tile output directory.
    """

    return batch_dir / "tiles"


def _batch_manifest_path(batch_dir: Path) -> Path:
    """Return one batch tile manifest path.

    Args:
        batch_dir (Path): Batch directory.

    Returns:
        Path: Manifest path.
    """

    return batch_dir / "tiles.txt"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON line to a results log.

    Args:
        path (Path): Output JSONL path.
        payload (dict[str, Any]): Row payload.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _message_status(message: str) -> str:
    """Extract the leading status token from one downloader message.

    Args:
        message (str): Downloader status message.

    Returns:
        str: Leading status token.

    Examples:
        >>> _message_status('DONE example.tif (0.1s)')
        'DONE'
    """

    return message.split(" ", 1)[0]


def _parse_message_metadata(message: str) -> dict[str, str]:
    """Extract optional reason/date fields from one downloader message.

    Args:
        message (str): Downloader status message.

    Returns:
        dict[str, str]: Parsed optional message fields.
    """

    payload: dict[str, str] = {}
    reason_match = re.search(r"reason=([^\s]+)", message)
    if reason_match is not None:
        payload["reason"] = reason_match.group(1)
    date_match = re.search(r"date=([0-9-]+)", message)
    if date_match is not None:
        payload["acquisition_date"] = date_match.group(1)
    return payload


def _record_from_message(
    x0: float,
    y0: float,
    message: str,
) -> dict[str, Any]:
    """Convert one downloader message into a structured result record.

    Args:
        x0 (float): Tile origin x coordinate.
        y0 (float): Tile origin y coordinate.
        message (str): Downloader status message.

    Returns:
        dict[str, Any]: Structured result record.
    """

    record: dict[str, Any] = {
        "x": int(x0),
        "y": int(y0),
        "tile_name": tile_name_from_origin(x0, y0),
        "status": _message_status(message),
        "message": message,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    record.update(_parse_message_metadata(message))
    return record


def _load_result_records(path: Path) -> list[dict[str, Any]]:
    """Load one results JSONL file.

    Args:
        path (Path): Results JSONL path.

    Returns:
        list[dict[str, Any]]: Parsed result rows.
    """

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _latest_records_by_origin(path: Path) -> dict[tuple[float, float], dict[str, Any]]:
    """Return the most recent result record per tile origin.

    Args:
        path (Path): Results JSONL path.

    Returns:
        dict[tuple[float, float], dict[str, Any]]: Latest records by origin.
    """

    latest: dict[tuple[float, float], dict[str, Any]] = {}
    for record in _load_result_records(path):
        origin = (float(record["x"]), float(record["y"]))
        latest[origin] = record
    return latest


def _existing_batch_tiles(batch_dir: Path) -> dict[tuple[float, float], Path]:
    """Discover already written TIFFs inside one batch staging directory.

    Args:
        batch_dir (Path): Batch directory.

    Returns:
        dict[tuple[float, float], Path]: Existing tile files by origin.
    """

    return discover_present_tiles(_tiles_dir(batch_dir))


def build_batch_summary(batch_dir: Path) -> dict[str, Any]:
    """Build a structured summary for one recovery shard.

    Args:
        batch_dir (Path): Batch directory.

    Returns:
        dict[str, Any]: Batch summary mapping.

    Examples:
        >>> callable(build_batch_summary)
        True
    """

    manifest_path = _batch_manifest_path(batch_dir)
    tile_origins = _read_tile_manifest(manifest_path)
    existing = _existing_batch_tiles(batch_dir)
    latest = _latest_records_by_origin(_results_path(batch_dir))
    resolved = set(existing)
    counts = {
        "DONE": len(existing),
        "SKIP": 0,
        "SKIP_SEASON": 0,
        "FAIL_METADATA": 0,
        "FAIL_BLANK": 0,
        "FAIL": 0,
    }
    for origin, record in latest.items():
        status = str(record.get("status", ""))
        if origin in existing and status == "DONE":
            continue
        counts[status] = counts.get(status, 0) + 1
        if status in FINAL_STATUSES:
            resolved.add(origin)
    remaining = [origin for origin in tile_origins if origin not in resolved]
    return {
        "batch_id": int(batch_dir.name.split("_")[-1]),
        "complete": len(remaining) == 0,
        "expected_tiles": len(tile_origins),
        "resolved_tiles": len(resolved),
        "remaining_tiles": len(remaining),
        "done_tiles": counts.get("DONE", 0) + counts.get("SKIP", 0),
        "status_counts": counts,
        "remaining_tile_origins": [[int(x0), int(y0)] for x0, y0 in remaining],
        "tiles_dir": str(_tiles_dir(batch_dir)),
        "results_path": str(_results_path(batch_dir)),
    }


def _write_batch_summary(batch_dir: Path) -> dict[str, Any]:
    """Recompute and persist one batch summary.

    Args:
        batch_dir (Path): Batch directory.

    Returns:
        dict[str, Any]: Written summary.
    """

    summary = build_batch_summary(batch_dir)
    _write_json(_summary_path(batch_dir), summary)
    return summary


def write_recovery_plan(
    *,
    present_dir: Path,
    staging_root: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Write one missing-tile recovery plan under a staging root.

    Args:
        present_dir (Path): Canonical tile directory.
        staging_root (Path): Recovery root.
        batch_size (int): Maximum tiles per shard.

    Returns:
        dict[str, Any]: Recovery manifest payload.

    Examples:
        >>> callable(write_recovery_plan)
        True
    """

    expected = get_data_api.build_tile_origins()
    present = discover_present_tiles(present_dir)
    missing = [origin for origin in expected if origin not in present]
    batches = chunk_tile_origins(missing, batch_size=batch_size)
    manifests: list[str] = []
    for idx, batch_origins in enumerate(batches):
        batch_dir = _recovery_batch_dir(staging_root, idx)
        _write_tile_manifest(_batch_manifest_path(batch_dir), batch_origins)
        manifests.append(str(_batch_manifest_path(batch_dir)))
    _write_tile_csv(staging_root / "expected_tiles.csv", expected)
    _write_tile_csv(staging_root / "present_tiles.csv", list(present))
    _write_tile_csv(staging_root / "missing_tiles.csv", missing)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "present_dir": str(present_dir),
        "expected_tiles": len(expected),
        "present_tiles": len(present),
        "missing_tiles": len(missing),
        "batch_size": int(batch_size),
        "num_batches": len(batches),
        "batch_manifests": manifests,
        "coverage_threshold": DEFAULT_COVERAGE_THRESHOLD,
    }
    _write_json(staging_root / "manifest.json", manifest)
    return manifest


def _load_or_create_manifest(
    *,
    present_dir: Path,
    staging_root: Path,
    batch_size: int,
    resume: bool,
) -> dict[str, Any]:
    """Load an existing recovery manifest or create a new one.

    Args:
        present_dir (Path): Canonical tile directory.
        staging_root (Path): Recovery root.
        batch_size (int): Maximum tiles per batch.
        resume (bool): Whether an existing manifest may be reused.

    Returns:
        dict[str, Any]: Recovery manifest payload.
    """

    manifest_path = staging_root / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError(f"recovery manifest already exists: {manifest_path}")
        return _load_json(manifest_path)
    return write_recovery_plan(
        present_dir=present_dir,
        staging_root=staging_root,
        batch_size=batch_size,
    )


def run_batch(
    *,
    batch_dir: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    """Run or resume one recovery shard.

    Args:
        batch_dir (Path): Batch staging directory.
        max_workers (int): Concurrent tile workers.

    Returns:
        dict[str, Any]: Final batch summary.

    Examples:
        >>> callable(run_batch)
        True
    """

    manifest_path = _batch_manifest_path(batch_dir)
    tile_origins = _read_tile_manifest(manifest_path)
    latest = _latest_records_by_origin(_results_path(batch_dir))
    existing = _existing_batch_tiles(batch_dir)
    remaining: list[tuple[float, float]] = []
    for origin in tile_origins:
        if origin in existing:
            continue
        record = latest.get(origin)
        if record is not None and str(record.get("status", "")) in FINAL_STATUSES:
            continue
        remaining.append(origin)
    if not remaining:
        logger.info("Batch %s already complete; skipping.", batch_dir.name)
        return _write_batch_summary(batch_dir)

    logger.info(
        "Running %s with %s remaining tile(s) and %s worker(s).",
        batch_dir.name,
        len(remaining),
        max_workers,
    )
    _tiles_dir(batch_dir).mkdir(parents=True, exist_ok=True)
    image_session = get_data_api.make_session()
    metadata_session = get_data_api.make_session()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                get_data_api.fetch_and_write_tile,
                image_session,
                metadata_session,
                x0,
                y0,
                out_dir=_tiles_dir(batch_dir),
            ): (x0, y0)
            for x0, y0 in remaining
        }
        for future in as_completed(futures):
            x0, y0 = futures[future]
            message = future.result()
            record = _record_from_message(x0, y0, message)
            _append_jsonl(_results_path(batch_dir), record)
            if record["status"] in FINAL_STATUSES:
                logger.info("%s", message)
            else:
                logger.warning("%s", message)
    return _write_batch_summary(batch_dir)


def run_recovery(
    *,
    present_dir: Path,
    staging_root: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    resume: bool = True,
) -> dict[str, Any]:
    """Run or resume all recovery shards sequentially.

    Args:
        present_dir (Path): Canonical tile directory.
        staging_root (Path): Recovery root.
        batch_size (int): Maximum tiles per batch.
        max_workers (int): Concurrent tile workers within each batch.
        resume (bool): Whether existing manifests and partial batches may be reused.

    Returns:
        dict[str, Any]: Final audit summary.
    """

    manifest = _load_or_create_manifest(
        present_dir=present_dir,
        staging_root=staging_root,
        batch_size=batch_size,
        resume=resume,
    )
    for batch_idx in range(int(manifest["num_batches"])):
        run_batch(
            batch_dir=_recovery_batch_dir(staging_root, batch_idx),
            max_workers=max_workers,
        )
    return audit_recovery(staging_root=staging_root, present_dir=present_dir)


def audit_recovery(
    *,
    staging_root: Path,
    present_dir: Path,
) -> dict[str, Any]:
    """Aggregate all shard results into one recovery audit summary.

    Args:
        staging_root (Path): Recovery root.
        present_dir (Path): Canonical tile directory.

    Returns:
        dict[str, Any]: Global audit summary.

    Examples:
        >>> callable(audit_recovery)
        True
    """

    manifest = _load_json(staging_root / "manifest.json")
    present = discover_present_tiles(present_dir)
    staged: dict[tuple[float, float], Path] = {}
    aggregate_counts = {
        "DONE": 0,
        "SKIP": 0,
        "SKIP_SEASON": 0,
        "FAIL_METADATA": 0,
        "FAIL_BLANK": 0,
        "FAIL": 0,
    }
    incomplete_batches: list[int] = []
    for batch_idx in range(int(manifest.get("num_batches", 0))):
        batch_dir = _recovery_batch_dir(staging_root, batch_idx)
        summary = build_batch_summary(batch_dir)
        _write_json(_summary_path(batch_dir), summary)
        if not bool(summary["complete"]):
            incomplete_batches.append(int(summary["batch_id"]))
        for key, value in summary["status_counts"].items():
            aggregate_counts[key] = aggregate_counts.get(key, 0) + int(value)
        staged.update(_existing_batch_tiles(batch_dir))
    expected_tiles = int(manifest.get("expected_tiles", 0))
    accepted_tiles = len(set(present) | set(staged))
    remaining_tiles = expected_tiles - accepted_tiles
    coverage_ratio = 0.0 if expected_tiles == 0 else accepted_tiles / expected_tiles
    audit = {
        "staging_root": str(staging_root),
        "present_dir": str(present_dir),
        "expected_tiles": expected_tiles,
        "present_tiles": len(present),
        "staged_tiles": len(staged),
        "accepted_tiles": accepted_tiles,
        "remaining_tiles": remaining_tiles,
        "coverage_ratio": coverage_ratio,
        "coverage_percent": coverage_ratio * 100.0,
        "status_counts": aggregate_counts,
        "num_batches": int(manifest.get("num_batches", 0)),
        "incomplete_batches": incomplete_batches,
        "complete": len(incomplete_batches) == 0,
    }
    _write_json(staging_root / "audit.json", audit)
    return audit


def promote_staged_tiles(
    *,
    staging_root: Path,
    present_dir: Path,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    """Promote staged accepted tiles into the canonical directory.

    Args:
        staging_root (Path): Recovery root.
        present_dir (Path): Canonical tile directory.
        coverage_threshold (float): Minimum accepted coverage required.

    Returns:
        dict[str, Any]: Promotion summary.

    Examples:
        >>> callable(promote_staged_tiles)
        True
    """

    audit = audit_recovery(staging_root=staging_root, present_dir=present_dir)
    if not bool(audit["complete"]):
        raise ValueError(
            "cannot promote tiles while some recovery batches remain incomplete"
        )
    if float(audit["coverage_ratio"]) < float(coverage_threshold):
        raise ValueError(
            f"coverage below threshold: {audit['coverage_ratio']:.4f} < {coverage_threshold:.4f}"
        )
    present_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    skipped_existing = 0
    for batch_idx in range(int(audit["num_batches"])):
        for origin, source_path in _existing_batch_tiles(
            _recovery_batch_dir(staging_root, batch_idx)
        ).items():
            destination = present_dir / tile_name_from_origin(*origin)
            if destination.exists():
                skipped_existing += 1
                continue
            shutil.move(str(source_path), str(destination))
            moved += 1
    summary = {
        "moved_tiles": moved,
        "skipped_existing": skipped_existing,
        "present_dir": str(present_dir),
        "coverage_ratio": audit["coverage_ratio"],
    }
    _write_json(staging_root / "promotion.json", summary)
    return summary


def _default_staging_root() -> Path:
    """Return a timestamped default staging root.

    Returns:
        Path: Default recovery staging path.
    """

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_STAGING_PARENT / f"recovery_{timestamp}"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the recovery wrapper.

    Returns:
        argparse.ArgumentParser: Configured parser.

    Examples:
        >>> isinstance(_build_arg_parser().prog, str)
        True
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    parser.add_argument("--staging-root", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing manifest and partial shard state.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write manifests and coverage inventory without downloading.",
    )
    parser.add_argument(
        "--run", action="store_true", help="Run or resume all missing-tile shards."
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote staged accepted tiles into the canonical folder after audit.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the missing-tile recovery wrapper.

    Args:
        argv (list[str] | None): Optional CLI arguments.

    Examples:
        >>> callable(main)
        True
    """

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    staging_root = (
        Path(args.staging_root) if args.staging_root else _default_staging_root()
    )
    present_dir = Path(args.present_dir)
    _configure_logging(staging_root / "recovery.log")
    logger.info("Present dir: %s", present_dir)
    logger.info("Staging root: %s", staging_root)
    if not any((args.plan_only, args.run, args.promote)):
        args.plan_only = True

    manifest = _load_or_create_manifest(
        present_dir=present_dir,
        staging_root=staging_root,
        batch_size=int(args.batch_size),
        resume=bool(args.resume or args.run or args.promote),
    )
    logger.info(
        "Recovery plan: expected=%s present=%s missing=%s batches=%s batch_size=%s",
        int(manifest["expected_tiles"]),
        int(manifest["present_tiles"]),
        int(manifest["missing_tiles"]),
        int(manifest["num_batches"]),
        int(manifest["batch_size"]),
    )
    if args.plan_only and not args.run and not args.promote:
        return
    if args.run:
        audit = run_recovery(
            present_dir=present_dir,
            staging_root=staging_root,
            batch_size=int(manifest["batch_size"]),
            max_workers=int(args.max_workers),
            resume=True,
        )
        logger.info(
            "Recovery audit: accepted=%s expected=%s coverage=%.2f%% incomplete_batches=%s",
            int(audit["accepted_tiles"]),
            int(audit["expected_tiles"]),
            float(audit["coverage_percent"]),
            audit["incomplete_batches"],
        )
        if not bool(audit["complete"]):
            raise SystemExit("recovery batches remain incomplete")
        if float(audit["coverage_ratio"]) < float(args.coverage_threshold):
            raise SystemExit(
                f"coverage {audit['coverage_ratio']:.4f} below threshold {float(args.coverage_threshold):.4f}"
            )
    if args.promote:
        summary = promote_staged_tiles(
            staging_root=staging_root,
            present_dir=present_dir,
            coverage_threshold=float(args.coverage_threshold),
        )
        logger.info(
            "Promotion summary: moved=%s skipped_existing=%s",
            int(summary["moved_tiles"]),
            int(summary["skipped_existing"]),
        )


if __name__ == "__main__":
    main()
