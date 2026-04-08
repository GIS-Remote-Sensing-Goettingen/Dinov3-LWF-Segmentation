"""Leakage guards for dataset split resolution."""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline.data_splits as data_splits_module  # noqa: E402
from pipeline.context import DistContext  # noqa: E402
from pipeline.data_splits import (  # noqa: E402
    _collate_variable_tiles,
    resolve_dataset_splits,
)

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


def test_explicit_scene_lists_expand_to_all_matching_tiles(tmp_path: Path) -> None:
    """Scene-level manifest entries should match every tile from that scene.

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
        "scene_c_y0_x0.pt",
    ):
        (tmp_path / name).touch()
    train_list = tmp_path / "train.txt"
    val_list = tmp_path / "val.txt"
    train_list.write_text("scene_a\n", encoding="utf-8")
    val_list.write_text("scene_b\nscene_c\n", encoding="utf-8")
    train_files, val_files = resolve_dataset_splits(
        processed_dir=str(tmp_path),
        split_cfg={"train_list": str(train_list), "val_list": str(val_list)},
        val_fraction=0.2,
        max_tiles=None,
        logger=_NoopLogger(),
    )

    assert sorted(Path(path).stem for path in train_files) == [
        "scene_a_y0_x0",
        "scene_a_y512_x0",
    ]
    assert sorted(Path(path).stem for path in val_files) == [
        "scene_b_y0_x0",
        "scene_c_y0_x0",
    ]


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


def test_resolve_rank_consistent_splits_uses_rank0_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed split resolution should reuse rank-0 file lists on workers.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    expected = {
        "train_files": ["/tmp/train_a.pt", "/tmp/train_b.pt"],
        "val_files": ["/tmp/val_a.pt"],
    }
    monkeypatch.setattr(
        data_splits_module,
        "resolve_dataset_splits",
        lambda *args, **kwargs: pytest.fail(
            "resolve_dataset_splits() should not run on non-main ranks"
        ),
    )
    monkeypatch.setattr(
        data_splits_module,
        "broadcast_main_object",
        lambda dist_ctx, payload: expected,
    )

    train_files, val_files = data_splits_module._resolve_rank_consistent_splits(
        processed_dir="/tmp/cache",
        split_cfg={},
        val_fraction=0.2,
        max_tiles=100,
        logger=_NoopLogger(),
        dist_ctx=DistContext(enabled=True, rank=1, world_size=2, local_rank=1),
    )

    assert train_files == expected["train_files"]
    assert val_files == expected["val_files"]


def test_validate_distributed_train_loader_shape_rejects_rank_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed dataloader sanity check should fail on length mismatches.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Examples:
        >>> True
        True
    """

    monkeypatch.setattr(
        data_splits_module.dist,
        "all_gather_object",
        lambda gathered, local: gathered.__setitem__(
            slice(None),
            [
                {"rank": 0, "dataset_len": 3685, "loader_len": 461},
                {"rank": 1, "dataset_len": 3740, "loader_len": 468},
            ],
        ),
    )

    with pytest.raises(ValueError, match="Distributed train split mismatch"):
        data_splits_module._validate_distributed_train_loader_shape(
            train_dataset=[0] * 3685,
            train_loader=[0] * 461,
            logger=_NoopLogger(),
            dist_ctx=DistContext(enabled=True, rank=0, world_size=2, local_rank=0),
        )


def test_collate_variable_tiles_pads_images_labels_and_features() -> None:
    """Batch collation should pad mixed-size cached tiles safely.

    Examples:
        >>> True
        True
    """

    batch = [
        (
            torch.ones(3, 4, 5),
            [torch.ones(2, 2, 3)],
            torch.zeros(2, 3, dtype=torch.long),
        ),
        (
            torch.ones(3, 6, 4),
            [torch.ones(2, 3, 2) * 2],
            torch.ones(3, 2, dtype=torch.long),
        ),
    ]

    images, features, labels = _collate_variable_tiles(batch, label_ignore_index=255)

    assert images.shape == (2, 3, 6, 5)
    assert len(features) == 1
    assert features[0].shape == (2, 2, 3, 3)
    assert labels.shape == (2, 3, 3)
    assert labels[0, -1, -1].item() == 255
