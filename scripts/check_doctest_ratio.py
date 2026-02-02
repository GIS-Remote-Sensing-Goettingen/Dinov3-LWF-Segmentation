"""Check doctest coverage ratio for public symbols."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def _main() -> int:
    """Run the doctest ratio check.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=float, required=True)
    args = parser.parse_args()

    total = 0
    with_doctest = 0
    for path in _iter_python_files(_repo_root()):
        counts = _count_doctests(path)
        total += counts[0]
        with_doctest += counts[1]

    ratio = 1.0 if total == 0 else with_doctest / total
    if ratio < args.min:
        print(f"doctest ratio {ratio:.2f} below minimum {args.min:.2f}")
        return 1
    print(f"doctest ratio {ratio:.2f} (min {args.min:.2f})")
    return 0


def _repo_root() -> Path:
    """Return the repository root.

    Returns:
        Path: Repository root path.
    """
    return Path(__file__).resolve().parents[1]


def _iter_python_files(root: Path) -> list[Path]:
    """Collect Python files under the repository.

    Args:
        root (Path): Root directory to scan.

    Returns:
        list[Path]: Python files to analyze.
    """
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if _should_skip(path):
            continue
        files.append(path)
    return files


def _should_skip(path: Path) -> bool:
    """Return True if a path should be skipped.

    Args:
        path (Path): Candidate path.

    Returns:
        bool: True if the path is excluded.
    """
    parts = set(path.parts)
    return any(
        name in parts
        for name in (
            ".git",
            ".venv",
            "__pycache__",
            "build",
            "dist",
            "tests",
            "scripts",
        )
    )


def _count_doctests(path: Path) -> tuple[int, int]:
    """Count total public symbols and doctested ones.

    Args:
        path (Path): Python file to inspect.

    Returns:
        tuple[int, int]: Total public symbols and doctested count.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    total = 0
    with_doctest = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            total += 1
            docstring = ast.get_docstring(node) or ""
            if ">>>" in docstring:
                with_doctest += 1
    return total, with_doctest


if __name__ == "__main__":
    raise SystemExit(_main())
