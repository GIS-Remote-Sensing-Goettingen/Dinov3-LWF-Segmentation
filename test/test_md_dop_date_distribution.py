"""Tests for the filename-derived folder coverage helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_temp_module():
    """Load the temporary coverage helper from disk.

    Returns:
        object: Imported module object.

    Examples:
        >>> callable(_load_temp_module)
        True
    """

    module_path = REPO_ROOT / "utility" / "test" / "temp.py"
    spec = importlib.util.spec_from_file_location("folder_coverage_temp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_tile_origins_parses_and_sorts_folder_tiles(tmp_path: Path) -> None:
    """Folder coverage helper should parse canonical DOP20 filenames.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    module = _load_temp_module()
    (tmp_path / "dop20_454000_6066000_1km_20cm.tif").touch()
    (tmp_path / "dop20_453000_6066000_1km_20cm.tif").touch()
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    origins = module.collect_tile_origins(tmp_path)

    assert origins == [(453000.0, 6066000.0), (454000.0, 6066000.0)]


def test_build_coverage_grid_marks_present_tiles() -> None:
    """Coverage grid should mark the occupied 1 km cells in blue.

    Examples:
        >>> True
        True
    """

    module = _load_temp_module()
    xs, ys, grid = module.build_coverage_grid(
        [
            (453000.0, 6066000.0),
            (454000.0, 6066000.0),
            (454000.0, 6067000.0),
        ]
    )

    assert xs.tolist() == [453000.0, 454000.0]
    assert ys.tolist() == [6066000.0, 6067000.0]
    assert grid.tolist() == [[1, 1], [0, 1]]


def test_discover_prediction_paths_skips_merged_output(tmp_path: Path) -> None:
    """Prediction discovery should ignore the merged Desktop mosaic.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    module = _load_temp_module()
    (tmp_path / "predictions.tif").touch()
    (tmp_path / "predictions_2.tif").touch()
    (tmp_path / "predictions_merged.tif").touch()

    paths = module.discover_prediction_paths(tmp_path)

    assert [path.name for path in paths] == ["predictions.tif", "predictions_2.tif"]


def test_tile_origins_from_resampled_grid_marks_positive_tiles() -> None:
    """Positive coarse-grid values should map to the touched 1 km tile origins.

    Examples:
        >>> True
        True
    """

    module = _load_temp_module()
    grid = np.array([[1, 0], [0, 2]], dtype=np.uint8)

    origins = module._tile_origins_from_resampled_grid(
        coarse_grid=grid,
        x_origins=np.array([453000.0, 454000.0], dtype=float),
        y_origins=np.array([6066000.0, 6067000.0], dtype=float),
    )

    assert origins == {(453000.0, 6066000.0), (454000.0, 6067000.0)}


def test_compute_uncovered_tile_origins_returns_missing_label_tiles() -> None:
    """Uncovered helper should return folder tiles without positive labels.

    Examples:
        >>> True
        True
    """

    module = _load_temp_module()

    uncovered = module.compute_uncovered_tile_origins(
        [
            (453000.0, 6066000.0),
            (454000.0, 6066000.0),
            (454000.0, 6067000.0),
        ],
        {(454000.0, 6066000.0)},
    )

    assert uncovered == [
        (453000.0, 6066000.0),
        (454000.0, 6067000.0),
    ]


def test_write_tile_csv_serializes_tile_names_and_origins(tmp_path: Path) -> None:
    """CSV export should include the canonical file name and snapped origins.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    module = _load_temp_module()
    csv_path = tmp_path / "tiles.csv"

    module.write_tile_csv(
        csv_path,
        [(453000.0, 6066000.0), (454000.0, 6067000.0)],
    )

    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "tile_name,x,y",
        "dop20_453000_6066000_1km_20cm.tif,453000,6066000",
        "dop20_454000_6067000_1km_20cm.tif,454000,6067000",
    ]
