"""Tests for the coarse MD DOP date-distribution sampler."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_temp_module():
    """Load the metadata sampling helper from disk.

    Returns:
        object: Imported module object.

    Examples:
        >>> callable(_load_temp_module)
        True
    """

    module_path = REPO_ROOT / "utility" / "test" / "temp.py"
    spec = importlib.util.spec_from_file_location("md_dop_temp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_sampling_origins_uses_requested_spacing() -> None:
    """Sampling origins should advance on the requested coarse grid.

    This verifies that the helper walks the AOI in 10 km steps rather than
    every 1 km tile.

    Examples:
        >>> True
        True
    """

    module = _load_temp_module()
    origins = module._build_sampling_origins(spacing_m=10_000, max_samples=3)

    assert len(origins) == 3
    assert origins[0] == (439000.0, 5903000.0)
    assert origins[1] == (439000.0, 5913000.0)
    assert origins[2] == (439000.0, 5923000.0)
