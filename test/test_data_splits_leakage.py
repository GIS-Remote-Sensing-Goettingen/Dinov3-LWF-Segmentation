"""Leakage guards for dataset split resolution."""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data_splits import resolve_dataset_splits  # noqa: E402

_TILE_SUFFIX_RE = re.compile(r"_y-?\d+_x-?\d+$")


class _NoopLogger:
    """Minimal logger stub for split resolution tests.

    This keeps unit tests focused on split logic without requiring the full
    runtime logger implementation.
    """

    def info(self, *_args, **_kwargs) -> None:
        """Accept an info log call and intentionally do nothing.

        Args:
            *_args: Positional log arguments.
            **_kwargs: Keyword log arguments.
        """

        return None


def _group_name(path: str) -> str:
    """Approximate source-group extraction used for leakage checks.

    This mirrors the split grouping logic used by production code for
    test assertions.

    Args:
        path (str): Cached tile path.

    Returns:
        str: Group identifier used in assertions.
    """

    stem = Path(path).stem
    stem = _TILE_SUFFIX_RE.sub("", stem)
    for suffix in (
        "_orig",
        "_flip_lr",
        "_flip_ud",
        "_rot90",
        "_rot180",
        "_rot270",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def test_explicit_lists_reject_exact_tile_overlap(tmp_path: Path) -> None:
    """Ensure explicit split lists reject exact tile overlap.

    Args:
        tmp_path (Path): Temporary cache directory.

    Examples:
        >>> True
        True
    """

    (tmp_path / "scene_a_y0_x0.pt").touch()
    (tmp_path / "scene_b_y0_x0.pt").touch()
    train_list = tmp_path / "train.txt"
    val_list = tmp_path / "val.txt"
    train_list.write_text("scene_a_y0_x0\n", encoding="utf-8")
    val_list.write_text("scene_a_y0_x0\nscene_b_y0_x0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        _ = resolve_dataset_splits(
            processed_dir=str(tmp_path),
            split_cfg={"train_list": str(train_list), "val_list": str(val_list)},
            val_fraction=0.2,
            max_tiles=None,
            logger=_NoopLogger(),
        )


def test_explicit_lists_reject_source_group_overlap(tmp_path: Path) -> None:
    """Ensure explicit split lists reject same-source group overlap.

    Args:
        tmp_path (Path): Temporary cache directory.

    Examples:
        >>> True
        True
    """

    (tmp_path / "scene_a_y0_x0.pt").touch()
    (tmp_path / "scene_a_y512_x0.pt").touch()
    (tmp_path / "scene_b_y0_x0.pt").touch()
    train_list = tmp_path / "train.txt"
    val_list = tmp_path / "val.txt"
    train_list.write_text("scene_a_y0_x0\n", encoding="utf-8")
    val_list.write_text("scene_a_y512_x0\nscene_b_y0_x0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source groups overlap"):
        _ = resolve_dataset_splits(
            processed_dir=str(tmp_path),
            split_cfg={"train_list": str(train_list), "val_list": str(val_list)},
            val_fraction=0.2,
            max_tiles=None,
            logger=_NoopLogger(),
        )


def test_random_split_is_source_group_disjoint(tmp_path: Path) -> None:
    """Ensure random split partitions by source groups, not individual tiles.

    Args:
        tmp_path (Path): Temporary cache directory.

    Examples:
        >>> True
        True
    """

    for name in (
        "scene_a_y0_x0.pt",
        "scene_a_y512_x0.pt",
        "scene_b_y0_x0.pt",
        "scene_b_y512_x0.pt",
        "scene_c_y0_x0.pt",
    ):
        (tmp_path / name).touch()
    random.seed(7)
    train_files, val_files = resolve_dataset_splits(
        processed_dir=str(tmp_path),
        split_cfg={},
        val_fraction=0.4,
        max_tiles=None,
        logger=_NoopLogger(),
    )
    train_groups = {_group_name(path) for path in train_files}
    val_groups = {_group_name(path) for path in val_files}
    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)


def test_random_split_requires_two_source_groups(tmp_path: Path) -> None:
    """Ensure leakage-safe random split fails with only one source group.

    Args:
        tmp_path (Path): Temporary cache directory.

    Examples:
        >>> True
        True
    """

    (tmp_path / "scene_a_y0_x0.pt").touch()
    (tmp_path / "scene_a_y512_x0.pt").touch()
    with pytest.raises(ValueError, match="At least two disjoint source groups"):
        _ = resolve_dataset_splits(
            processed_dir=str(tmp_path),
            split_cfg={},
            val_fraction=0.2,
            max_tiles=None,
            logger=_NoopLogger(),
        )
