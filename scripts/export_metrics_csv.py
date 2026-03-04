"""Export JSONL metrics artifacts to a flat CSV table.

The input format matches `artifacts/metrics.jsonl` records written by the
pipeline metrics writer:
`{"timestamp_ms": ..., "phase": ..., "step": ..., "metrics": {...}}`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL metric records from disk.

    Args:
        path (Path): Path to the JSONL file.

    Returns:
        list[dict[str, Any]]: Parsed records.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "m.jsonl"
        ...     _ = p.write_text(
        ...         '{"timestamp_ms":1,"phase":"train","step":1,"metrics":{"miou":0.5}}\\n',
        ...         encoding="utf-8",
        ...     )
        ...     rows = load_records(p)
        ...     len(rows), rows[0]["phase"]
        (1, 'train')
    """

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            records.append(raw)
    return records


def collect_metric_columns(records: list[dict[str, Any]]) -> list[str]:
    """Collect sorted metric-key columns from records.

    Args:
        records (list[dict[str, Any]]): Metric records.

    Returns:
        list[str]: Sorted metric column names.

    Examples:
        >>> cols = collect_metric_columns(
        ...     [{"metrics": {"a": 1.0, "b": 2.0}}, {"metrics": {"c": 3.0}}]
        ... )
        >>> cols
        ['a', 'b', 'c']
    """

    keys: set[str] = set()
    for record in records:
        metrics = record.get("metrics", {})
        if isinstance(metrics, dict):
            keys.update(str(k) for k in metrics.keys())
    return sorted(keys)


def write_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    """Write flat CSV with one row per metrics record.

    Args:
        records (list[dict[str, Any]]): Metric records.
        out_path (Path): Destination CSV file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     out = Path(d) / "out.csv"
        ...     write_csv(
        ...         [{"timestamp_ms": 1, "phase": "train", "step": 1, "metrics": {"miou": 0.5}}],
        ...         out,
        ...     )
        ...     out.exists()
        True
    """

    metric_cols = collect_metric_columns(records)
    columns = ["timestamp_ms", "phase", "step"] + metric_cols
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                "timestamp_ms": record.get("timestamp_ms"),
                "phase": record.get("phase"),
                "step": record.get("step"),
            }
            metrics = record.get("metrics", {})
            if isinstance(metrics, dict):
                for key in metric_cols:
                    row[key] = metrics.get(key)
            writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to metrics JSONL file (typically artifacts/metrics.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="",
        help="Optional phase filter (e.g., 'train', 'verify', 'inference').",
    )
    return parser.parse_args()


def main() -> int:
    """Run metrics export CLI.

    Returns:
        int: Exit code.

    Examples:
        >>> callable(main)
        True
    """

    args = _parse_args()
    records = load_records(args.input)
    phase_filter = str(args.phase).strip()
    if phase_filter:
        records = [r for r in records if str(r.get("phase", "")) == phase_filter]
    records.sort(
        key=lambda r: (
            str(r.get("phase", "")),
            int(r.get("step", 0) or 0),
            int(r.get("timestamp_ms", 0) or 0),
        )
    )
    write_csv(records, args.output)
    print(f"wrote {len(records)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
