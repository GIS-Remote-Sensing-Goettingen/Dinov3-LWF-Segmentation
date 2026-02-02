"""Check file length thresholds and enforce hard limits.

Rules:
- Warn if a file exceeds 1000 lines.
- Fail if a file exceeds 2000 lines.

Examples:
    >>> is_too_long(2001)
    True
    >>> is_warning(1500)
    True
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class FileLength:
    """File length metadata."""

    path: Path
    lines: int


def is_warning(lines: int) -> bool:
    """Return True if line count triggers a warning.

    Args:
        lines (int): Line count to check.

    Examples:
        >>> is_warning(1001)
        True
        >>> is_warning(1000)
        False
    """
    return lines > 1000


def is_too_long(lines: int) -> bool:
    """Return True if line count exceeds the hard limit.

    Args:
        lines (int): Line count to check.

    Examples:
        >>> is_too_long(2000)
        False
        >>> is_too_long(2001)
        True
    """
    return lines > 2000


def iter_repo_files(root: Path) -> Iterable[Path]:
    """Yield repository files, excluding common build artifacts.

    Args:
        root (Path): Repository root to scan.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     sample = root / "sample.txt"
        ...     _ = sample.write_text("hi")
        ...     sample in list(iter_repo_files(root))
        True
    """
    excluded = {".git", ".venv", "build", "dist", "__pycache__"}
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in excluded for part in path.parts):
            continue
        yield path


def count_lines(path: Path) -> int:
    """Count lines in a text file, returning 0 on read errors.

    Args:
        path (Path): File to read.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "sample.txt"
        ...     _ = sample.write_text("a\\n")
        ...     count_lines(sample)
        2
    """
    try:
        return path.read_text(encoding="utf-8").count("\n") + 1
    except OSError:
        return 0


def collect_lengths(root: Path) -> List[FileLength]:
    """Collect line counts for repo files.

    Args:
        root (Path): Repository root to scan.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     sample = root / "sample.py"
        ...     _ = sample.write_text("x\\n")
        ...     lengths = collect_lengths(root)
        ...     any(entry.path == sample for entry in lengths)
        True
    """
    results: List[FileLength] = []
    for path in iter_repo_files(root):
        results.append(FileLength(path=path, lines=count_lines(path)))
    return results


def format_warning(file_len: FileLength) -> str:
    """Format a warning line.

    Args:
        file_len (FileLength): File length metadata.

    Examples:
        >>> format_warning(FileLength(Path("x.py"), 1001))
        'WARN: x.py has 1001 lines'
    """
    return f"WARN: {file_len.path} has {file_len.lines} lines"


def format_error(file_len: FileLength) -> str:
    """Format an error line.

    Args:
        file_len (FileLength): File length metadata.

    Examples:
        >>> format_error(FileLength(Path("x.py"), 2001))
        'ERROR: x.py has 2001 lines (limit 2000)'
    """
    return f"ERROR: {file_len.path} has {file_len.lines} lines (limit 2000)"


def main() -> int:
    """Run the file length check.

    Examples:
        >>> callable(main)
        True
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root to scan.",
    )
    args = parser.parse_args()

    errors = 0
    for file_len in collect_lengths(args.root):
        if is_too_long(file_len.lines):
            print(format_error(file_len))
            errors += 1
        elif is_warning(file_len.lines):
            print(format_warning(file_len))

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
