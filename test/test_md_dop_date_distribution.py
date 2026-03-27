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


def test_build_tile_transform_uses_folder_grid_bounds() -> None:
    """Shared tile transform should start at min x and one tile above max y.

    Examples:
        >>> True
        True
    """

    module = _load_temp_module()

    transform = module.build_tile_transform(
        np.array([453000.0, 454000.0], dtype=float),
        np.array([6066000.0, 6067000.0], dtype=float),
    )

    assert float(transform.c) == 453000.0
    assert float(transform.f) == 6068000.0


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


def test_classify_tile_masks_uses_strict_violet_threshold() -> None:
    """Tiles should be violet only when prediction count is strictly above 60%.

    Examples:
        >>> True
        True
    """

    module = _load_temp_module()
    blue_only, orange, violet = module.classify_tile_masks(
        coverage_grid=np.array([[1, 1, 1, 1]], dtype=np.uint8),
        prediction_count_grid=np.array([[0, 1, 3600, 3601]], dtype=np.int64),
        planet_count_grid=np.array([[6000, 0, 6000, 6000]], dtype=np.int64),
    )

    assert blue_only.tolist() == [[True, False, False, False]]
    assert orange.tolist() == [[False, True, True, False]]
    assert violet.tolist() == [[False, False, False, True]]


def test_tile_origins_from_mask_serializes_true_cells_in_grid_order() -> None:
    """Mask conversion should return sorted tile origins for true cells.

    Examples:
        >>> True
        True
    """

    module = _load_temp_module()

    origins = module.tile_origins_from_mask(
        np.array([[True, False], [False, True]]),
        xs=np.array([453000.0, 454000.0], dtype=float),
        ys=np.array([6066000.0, 6067000.0], dtype=float),
    )

    assert origins == [
        (453000.0, 6066000.0),
        (454000.0, 6067000.0),
    ]


def test_compute_prediction_planet_ratio_percentages_uses_only_planet_positive_tiles() -> (
    None
):
    """Ratio helper should ignore zero-Planet tiles and return percentages.

    Examples:
        >>> True
        True
    """

    module = _load_temp_module()

    ratios = module.compute_prediction_planet_ratio_percentages(
        coverage_grid=np.array([[1, 1, 1]], dtype=np.uint8),
        prediction_count_grid=np.array([[0, 3600, 7200]], dtype=np.int64),
        planet_count_grid=np.array([[0, 6000, 6000]], dtype=np.int64),
    )

    assert np.allclose(ratios, np.array([60.0, 120.0]))


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
