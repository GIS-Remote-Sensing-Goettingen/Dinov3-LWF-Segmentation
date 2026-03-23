"""Recovery-wrapper tests for missing DOP20 tiles."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_recovery_module():
    """Load the recovery wrapper from disk.

    Returns:
        object: Imported module object.

    Examples:
        >>> callable(_load_recovery_module)
        True
    """

    module_path = REPO_ROOT / "utility" / "recover_missing_tiles.py"
    spec = importlib.util.spec_from_file_location("recover_missing_tiles", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch_tile(path: Path) -> None:
    """Create one empty tile placeholder file.

    Args:
        path (Path): Tile path.

    Examples:
        >>> callable(_touch_tile)
        True
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_write_recovery_plan_inventories_missing_tiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Recovery planning should derive the missing set from filename inventory.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch: Pytest monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    module = _load_recovery_module()
    present_dir = tmp_path / "folder_1"
    _touch_tile(present_dir / "dop20_0_0_1km_20cm.tif")
    _touch_tile(present_dir / "dop20_2000_0_1km_20cm.tif")
    monkeypatch.setattr(
        module.get_data_api,
        "build_tile_origins",
        lambda: [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)],
    )

    manifest = module.write_recovery_plan(
        present_dir=present_dir,
        staging_root=tmp_path / "recovery",
        batch_size=2,
    )

    assert manifest["expected_tiles"] == 3
    assert manifest["present_tiles"] == 2
    assert manifest["missing_tiles"] == 1
    assert manifest["num_batches"] == 1
    assert (tmp_path / "recovery" / "missing_tiles.csv").exists()
    manifest_lines = (
        (tmp_path / "recovery" / "batches" / "batch_000" / "tiles.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert manifest_lines == ["1000,0"]


def test_run_batch_resume_retries_only_remaining_tiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Shard resume should skip final statuses and retry transient failures only.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch: Pytest monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    module = _load_recovery_module()
    batch_dir = tmp_path / "recovery" / "batches" / "batch_000"
    module._write_tile_manifest(
        batch_dir / "tiles.txt",
        [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)],
    )
    monkeypatch.setattr(module.get_data_api, "make_session", lambda: object())
    call_counts: dict[tuple[float, float], int] = {}

    def _fake_fetch(_image_session, _metadata_session, x0, y0, *, out_dir, **_kwargs):
        """Return deterministic downloader outcomes for one resume test.

        Args:
            _image_session: Ignored fake imagery session.
            _metadata_session: Ignored fake metadata session.
            x0 (float): Tile origin x coordinate.
            y0 (float): Tile origin y coordinate.
            out_dir (Path): Batch output directory where fake TIFFs are touched.
            **_kwargs: Ignored keyword arguments forwarded by the wrapper.
        """

        origin = (x0, y0)
        call_counts[origin] = call_counts.get(origin, 0) + 1
        if origin == (0.0, 0.0):
            _touch_tile(out_dir / module.tile_name_from_origin(x0, y0))
            return f"DONE {module.tile_name_from_origin(x0, y0)} (0.1s)"
        if origin == (1000.0, 0.0):
            return (
                f"SKIP_SEASON {module.tile_name_from_origin(x0, y0)} "
                "reason=season_rejected date=2025-04-03 (0.1s)"
            )
        if call_counts[origin] == 1:
            return "FAIL 2000,0 [503] ct=text/plain (0.1s) temporary"
        _touch_tile(out_dir / module.tile_name_from_origin(x0, y0))
        return f"DONE {module.tile_name_from_origin(x0, y0)} (0.1s)"

    monkeypatch.setattr(module.get_data_api, "fetch_and_write_tile", _fake_fetch)

    summary_first = module.run_batch(batch_dir=batch_dir, max_workers=2)
    summary_second = module.run_batch(batch_dir=batch_dir, max_workers=2)

    assert summary_first["complete"] is False
    assert summary_first["remaining_tiles"] == 1
    assert summary_second["complete"] is True
    assert summary_second["remaining_tiles"] == 0
    assert call_counts[(0.0, 0.0)] == 1
    assert call_counts[(1000.0, 0.0)] == 1
    assert call_counts[(2000.0, 0.0)] == 2


def test_audit_and_promote_move_only_staged_accepted_tiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Audit and promotion should aggregate staged tiles and move them safely.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch: Pytest monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    module = _load_recovery_module()
    present_dir = tmp_path / "folder_1"
    _touch_tile(present_dir / "dop20_0_0_1km_20cm.tif")
    monkeypatch.setattr(
        module.get_data_api,
        "build_tile_origins",
        lambda: [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)],
    )
    staging_root = tmp_path / "recovery"
    module.write_recovery_plan(
        present_dir=present_dir,
        staging_root=staging_root,
        batch_size=1,
    )
    batch_000 = staging_root / "batches" / "batch_000"
    batch_001 = staging_root / "batches" / "batch_001"
    _touch_tile(batch_000 / "tiles" / "dop20_1000_0_1km_20cm.tif")
    module._append_jsonl(
        batch_001 / "results.jsonl",
        {
            "x": 2000,
            "y": 0,
            "tile_name": "dop20_2000_0_1km_20cm.tif",
            "status": "SKIP_SEASON",
            "message": "SKIP_SEASON dop20_2000_0_1km_20cm.tif reason=season_rejected date=2025-04-03 (0.1s)",
        },
    )

    audit = module.audit_recovery(staging_root=staging_root, present_dir=present_dir)
    promotion = module.promote_staged_tiles(
        staging_root=staging_root,
        present_dir=present_dir,
        coverage_threshold=0.5,
    )

    assert audit["complete"] is True
    assert audit["accepted_tiles"] == 2
    assert audit["expected_tiles"] == 3
    assert round(float(audit["coverage_ratio"]), 4) == round(2 / 3, 4)
    assert promotion["moved_tiles"] == 1
    assert (present_dir / "dop20_1000_0_1km_20cm.tif").exists()
