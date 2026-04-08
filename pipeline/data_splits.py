"""Dataset split helpers and dataloader builders."""

from __future__ import annotations

import glob
import os
import random
import re
from collections import defaultdict
from functools import partial
from typing import Optional, Sized, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import PrecomputedDataset, VerbosityLogger

from .context import DistContext
from .train_utils import head_uses_backbone_features, resolve_model_patch_size
from .utils import broadcast_main_object

_TILE_SUFFIX_RE = re.compile(r"_y-?\d+_x-?\d+$")
_AUGMENT_SUFFIXES = (
    "_orig",
    "_flip_lr",
    "_flip_ud",
    "_rot90",
    "_rot180",
    "_rot270",
)


def _pad_spatial_tensor(
    tensor: torch.Tensor,
    target_hw: tuple[int, int],
    fill_value: float | int = 0,
) -> torch.Tensor:
    """Pad one tensor on the bottom/right edges to a target spatial size.

    Args:
        tensor (torch.Tensor): Tensor with trailing ``(H, W)`` dimensions.
        target_hw (tuple[int, int]): Target ``(height, width)``.
        fill_value (float | int): Constant pad value.

    Returns:
        torch.Tensor: Tensor padded to ``target_hw``.
    """

    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    pad_h = max(0, target_h - int(tensor.shape[-2]))
    pad_w = max(0, target_w - int(tensor.shape[-1]))
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=float(fill_value))


def _collate_variable_tiles(
    batch: list[tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]],
    *,
    label_ignore_index: int,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Collate one batch of cached tiles with bottom/right padding.

    This keeps batching valid when native label-grid tiling yields different
    image or label shapes across scenes.

    Args:
        batch: Sequence of ``(image, features, label)`` samples.
        label_ignore_index: Fill value for padded label regions.

    Returns:
        tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
            Padded image batch, padded feature batches per layer, and padded
            label batch.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    images, feature_lists, labels = zip(*batch)
    image_target = (
        max(int(image.shape[-2]) for image in images),
        max(int(image.shape[-1]) for image in images),
    )
    label_target = (
        max(int(label.shape[-2]) for label in labels),
        max(int(label.shape[-1]) for label in labels),
    )
    padded_images = torch.stack(
        [_pad_spatial_tensor(image, image_target, fill_value=0.0) for image in images],
        dim=0,
    )
    padded_labels = torch.stack(
        [
            _pad_spatial_tensor(label, label_target, fill_value=label_ignore_index)
            for label in labels
        ],
        dim=0,
    )

    feature_count = len(feature_lists[0])
    if any(len(features) != feature_count for features in feature_lists):
        raise ValueError("Feature-list length mismatch inside one batch.")
    padded_feature_batches: list[torch.Tensor] = []
    for layer_idx in range(feature_count):
        layer_tensors = [features[layer_idx] for features in feature_lists]
        layer_target = (
            max(int(feat.shape[-2]) for feat in layer_tensors),
            max(int(feat.shape[-1]) for feat in layer_tensors),
        )
        padded_feature_batches.append(
            torch.stack(
                [
                    _pad_spatial_tensor(feat, layer_target, fill_value=0.0)
                    for feat in layer_tensors
                ],
                dim=0,
            )
        )
    return padded_images, padded_feature_batches, padded_labels


def _file_stem(path: str) -> str:
    """Return the file stem for a path string.

    Args:
        path (str): File path.

    Returns:
        str: File stem without extension.

    Examples:
        >>> _file_stem("/tmp/sample.pt")
        'sample'
    """

    return os.path.splitext(os.path.basename(path))[0]


def _normalize_name_entry(entry: str) -> str:
    """Normalize a split-list entry to the canonical split-entry stem.

    Args:
        entry (str): Raw list entry (stem, filename, or path).

    Returns:
        str: Canonical stem value used for matching cached tiles or scenes.

    Examples:
        >>> _normalize_name_entry("tile_a.pt")
        'tile_a'
    """

    value = entry.strip()
    if not value:
        return value
    return _file_stem(value)


def _source_group(path: str) -> str:
    """Return a source-group key for a cached tile path.

    The key removes tile-coordinate suffixes and common augmentation suffixes
    so train/validation partitions can be checked for source-scene leakage.

    Args:
        path (str): Cached tile path.

    Returns:
        str: Source-scene group key.

    Examples:
        >>> _source_group("/tmp/scene_a_flip_lr_y0_x512.pt")
        'scene_a'
    """

    stem = _file_stem(path)
    stem = _TILE_SUFFIX_RE.sub("", stem)
    for suffix in _AUGMENT_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or _file_stem(path)


def _matches_split_entry(file_path: str, entries: set[str]) -> bool:
    """Return true when one cached tile matches one explicit split entry.

    Entries may be exact cached tile stems or source-scene stems such as
    ``dop20_592000_5982000_1km_20cm``. Scene-level entries expand to every
    cached tile belonging to that source group.

    Args:
        file_path (str): Cached tile path.
        entries (set[str]): Normalized explicit split entries.

    Returns:
        bool: True when the tile stem or its source group matches an entry.

    Examples:
        >>> _matches_split_entry("/tmp/scene_a_y0_x0.pt", {"scene_a_y0_x0"})
        True
        >>> _matches_split_entry("/tmp/scene_a_y0_x0.pt", {"scene_a"})
        True
        >>> _matches_split_entry("/tmp/scene_a_y0_x0.pt", {"scene_b"})
        False
    """

    tile_stem = _file_stem(file_path)
    return tile_stem in entries or _source_group(file_path) in entries


def _assert_split_disjoint(train_files: list[str], val_files: list[str]) -> None:
    """Fail fast when train/validation files or source groups overlap.

    Args:
        train_files (list[str]): Train split file paths.
        val_files (list[str]): Validation split file paths.

    Raises:
        ValueError: If exact tile overlap or source-group overlap is detected.

    Examples:
        >>> _assert_split_disjoint(['/tmp/a_y0_x0.pt'], ['/tmp/b_y0_x0.pt'])
    """

    train_set = set(train_files)
    val_set = set(val_files)
    overlap_files = sorted(train_set & val_set)
    if overlap_files:
        sample = ", ".join(_file_stem(path) for path in overlap_files[:3])
        raise ValueError(
            "Train/validation split overlap detected for cached tiles: "
            f"{sample}. Update split lists to be disjoint."
        )
    train_groups = {_source_group(path) for path in train_files}
    val_groups = {_source_group(path) for path in val_files}
    overlap_groups = sorted(train_groups & val_groups)
    if overlap_groups:
        sample = ", ".join(overlap_groups[:3])
        raise ValueError(
            "Train/validation source groups overlap (spatial leakage risk): "
            f"{sample}. Use disjoint scenes for train and validation."
        )


def _read_name_list(path: str) -> list[str]:
    """Read a list of names from a text or YAML/JSON file.

    Args:
        path (str): Path to the list file.

    Returns:
        list[str]: Cleaned list of names.

    Raises:
        FileNotFoundError: If the list file is missing.

    Examples:
        >>> from tempfile import NamedTemporaryFile
        >>> tmp = NamedTemporaryFile(delete=False, suffix=".txt")
        >>> _ = tmp.write(b"a\\n\\nB\\n")
        >>> tmp.close()
        >>> _read_name_list(tmp.name)
        ['a', 'B']
    """

    def _flatten_name_entries(value: object) -> list[str]:
        """Flatten nested YAML/JSON list payloads into one string list.

        Args:
            value (object): Raw decoded YAML/JSON value.

        Returns:
            list[str]: Flattened non-empty string entries.
        """

        if isinstance(value, list):
            flattened: list[str] = []
            for item in value:
                flattened.extend(_flatten_name_entries(item))
            return flattened
        text_value = str(value).strip()
        return [text_value] if text_value else []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Split list not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    ext = os.path.splitext(path)[1].lower()
    if ext in {".yml", ".yaml", ".json"}:
        import yaml

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            combined = []
            for value in data.values():
                combined.extend(_flatten_name_entries(value))
            return combined
        if isinstance(data, list):
            return _flatten_name_entries(data)
    return [line.strip() for line in text.splitlines() if line.strip()]


def resolve_dataset_splits(
    processed_dir: str,
    split_cfg: dict,
    val_fraction: float,
    max_tiles: int | None,
    logger: VerbosityLogger,
) -> tuple[list[str], list[str]]:
    """Resolve train/validation file lists for cached tiles.

    Args:
        processed_dir (str): Directory containing cached tiles.
        split_cfg (dict): Split configuration block.
        val_fraction (float): Fraction of tiles reserved for validation.
        max_tiles (int | None): Optional cap on the number of tiles to use.
        logger (VerbosityLogger): Logger for split details.

    Returns:
        tuple[list[str], list[str]]: Train and validation file paths.

    Raises:
        ValueError: If no cached tiles are found or splits are empty.

    Examples:
        >>> callable(resolve_dataset_splits)
        True
    """

    all_files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
    if not all_files:
        raise ValueError(f"No cached tiles found in {processed_dir}")
    if not split_cfg.get("train_list") and max_tiles and max_tiles > 0:
        if len(all_files) > max_tiles:
            logger.info(
                f"Sampling {max_tiles} tiles from {len(all_files)} total cached tiles."
            )
            all_files = random.sample(all_files, k=max_tiles)
    if split_cfg.get("train_list"):
        train_names = {
            _normalize_name_entry(name)
            for name in _read_name_list(split_cfg["train_list"])
        }
        train_files = [f for f in all_files if _matches_split_entry(f, train_names)]
        if split_cfg.get("val_list"):
            val_names = {
                _normalize_name_entry(name)
                for name in _read_name_list(split_cfg["val_list"])
            }
            val_files = [f for f in all_files if _matches_split_entry(f, val_names)]
        else:
            val_files = [f for f in all_files if f not in train_files]
        if not train_files or not val_files:
            raise ValueError("Split lists produced empty train/val subsets.")
        _assert_split_disjoint(train_files, val_files)
        return train_files, val_files
    group_to_files: dict[str, list[str]] = defaultdict(list)
    for file_path in all_files:
        group_to_files[_source_group(file_path)].append(file_path)
    groups = sorted(group_to_files.keys())
    if len(groups) < 2:
        raise ValueError(
            "At least two disjoint source groups are required to build leakage-safe "
            "train/validation splits."
        )
    random.shuffle(groups)
    split_idx = max(1, int(len(groups) * (1 - val_fraction)))
    split_idx = min(split_idx, len(groups) - 1)
    train_groups = groups[:split_idx]
    val_groups = groups[split_idx:]
    train_files = [path for group in train_groups for path in group_to_files[group]]
    val_files = [path for group in val_groups for path in group_to_files[group]]
    _assert_split_disjoint(train_files, val_files)
    logger.info(
        "Using leakage-safe random split with "
        f"{len(train_files)} train tiles ({len(train_groups)} source groups) and "
        f"{len(val_files)} validation tiles ({len(val_groups)} source groups)."
    )
    return train_files, val_files


def create_dataloaders(
    processed_dir: str,
    dataset_cfg: dict,
    train_cfg: dict,
    model_cfg: dict,
    batch_size: int,
    logger: VerbosityLogger,
    dist_ctx: DistContext,
) -> tuple[DataLoader, Optional[DistributedSampler], Optional[DataLoader]]:
    """Build training and validation dataloaders.

    Args:
        processed_dir (str): Cached tile directory.
        dataset_cfg (dict): Dataset configuration block.
        train_cfg (dict): Training configuration block.
        model_cfg (dict): Model configuration block.
        batch_size (int): Batch size for loaders.
        logger (VerbosityLogger): Logger for split information.
        dist_ctx (DistContext): Distributed execution context.

    Returns:
        tuple[DataLoader, Optional[DistributedSampler], Optional[DataLoader]]: Train loader,
        train sampler, and validation loader.

    Examples:
        >>> callable(create_dataloaders)
        True
    """

    augment_cfg = dataset_cfg.get("augmentations", {})
    validation_cfg = dataset_cfg.get("validation", {})
    raw_ignore_index = validation_cfg.get("ignore_index", 255)
    label_ignore_index = 255 if raw_ignore_index is None else int(raw_ignore_index)
    split_cfg = dataset_cfg.get("splits", {})
    cache_features = bool(dataset_cfg.get("cache_features", False))
    allow_feature_mismatch = bool(augment_cfg.get("allow_feature_mismatch", False))
    has_image_only_aug = any(
        bool(augment_cfg.get(block, {}).get("enable", False))
        for block in ("color_jitter", "cutout", "gridmask")
    )
    if (
        cache_features
        and augment_cfg.get("enable", False)
        and has_image_only_aug
        and not allow_feature_mismatch
    ):
        logger.info(
            "Image-only augmentations (color_jitter/cutout/gridmask) are "
            "disabled because dataset.cache_features=true and "
            "dataset.augmentations.allow_feature_mismatch=false."
        )
    val_fraction = train_cfg.get("val_fraction", 0.2)
    max_tiles = dataset_cfg.get("max_tiles")
    requested_layers = (
        model_cfg.get("layers")
        if head_uses_backbone_features(str(model_cfg.get("head", "")))
        else []
    )
    expected_patch_size = resolve_model_patch_size(
        str(model_cfg.get("backbone", "")),
        str(model_cfg.get("head", "")),
    )
    train_files, val_files = _resolve_rank_consistent_splits(
        processed_dir=processed_dir,
        split_cfg=split_cfg,
        val_fraction=val_fraction,
        max_tiles=max_tiles,
        logger=logger,
        dist_ctx=dist_ctx,
    )
    train_dataset = PrecomputedDataset(
        processed_dir,
        augmentation_cfg=augment_cfg,
        file_subset=train_files,
        validation_cfg=validation_cfg,
        requested_layers=requested_layers,
        expected_patch_size=expected_patch_size,
    )
    train_sampler = None
    if dist_ctx.enabled:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=False,
        )
    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=partial(
            _collate_variable_tiles,
            label_ignore_index=label_ignore_index,
        ),
    )
    _validate_distributed_train_loader_shape(
        train_dataset=train_dataset,
        train_loader=train_loader,
        logger=logger,
        dist_ctx=dist_ctx,
    )
    val_loader = None
    if (not dist_ctx.enabled) or dist_ctx.is_main:
        val_dataset = PrecomputedDataset(
            processed_dir,
            augmentation_cfg={"enable": False},
            file_subset=val_files,
            validation_cfg=validation_cfg,
            requested_layers=requested_layers,
            expected_patch_size=expected_patch_size,
        )
        val_workers = train_cfg.get("val_workers", max(1, num_workers // 2))
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True,
            persistent_workers=val_workers > 0,
            collate_fn=partial(
                _collate_variable_tiles,
                label_ignore_index=label_ignore_index,
            ),
        )
    return train_loader, train_sampler, val_loader


def _resolve_rank_consistent_splits(
    *,
    processed_dir: str,
    split_cfg: dict,
    val_fraction: float,
    max_tiles: int | None,
    logger: VerbosityLogger,
    dist_ctx: DistContext,
) -> tuple[list[str], list[str]]:
    """Resolve one train/validation split and share it across DDP ranks.

    Args:
        processed_dir (str): Cached tile directory.
        split_cfg (dict): Dataset split configuration block.
        val_fraction (float): Fraction reserved for validation.
        max_tiles (int | None): Optional cap on total tiles.
        logger (VerbosityLogger): Logger for split diagnostics.
        dist_ctx (DistContext): Distributed execution context.

    Returns:
        tuple[list[str], list[str]]: Train and validation file lists shared by
        every rank.

    Examples:
        >>> callable(_resolve_rank_consistent_splits)
        True
    """

    payload: dict[str, list[str]] | None = None
    if (not dist_ctx.enabled) or dist_ctx.is_main:
        train_files, val_files = resolve_dataset_splits(
            processed_dir,
            split_cfg,
            val_fraction,
            max_tiles,
            logger,
        )
        payload = {
            "train_files": [str(path) for path in train_files],
            "val_files": [str(path) for path in val_files],
        }
    if dist_ctx.enabled:
        payload = cast(dict[str, list[str]], broadcast_main_object(dist_ctx, payload))
    assert payload is not None
    return list(payload["train_files"]), list(payload["val_files"])


def _validate_distributed_train_loader_shape(
    *,
    train_dataset: Sized,
    train_loader: Sized,
    logger: VerbosityLogger,
    dist_ctx: DistContext,
) -> None:
    """Fail fast when distributed ranks build different train loader sizes.

    Args:
        train_dataset (Sized): Training dataset for the current rank.
        train_loader (Sized): Training loader for the current rank.
        logger (VerbosityLogger): Logger for optional summary output.
        dist_ctx (DistContext): Distributed execution context.

    Raises:
        ValueError: If dataset or loader sizes differ across ranks.

    Examples:
        >>> callable(_validate_distributed_train_loader_shape)
        True
    """

    if not dist_ctx.enabled:
        return
    local_state = {
        "rank": int(dist_ctx.rank),
        "dataset_len": int(len(train_dataset)),
        "loader_len": int(len(train_loader)),
    }
    gathered_states: list[dict[str, int] | None] = [None] * dist_ctx.world_size
    dist.all_gather_object(gathered_states, local_state)
    states = [
        cast(dict[str, int], state) for state in gathered_states if state is not None
    ]
    dataset_lengths = {state["dataset_len"] for state in states}
    loader_lengths = {state["loader_len"] for state in states}
    if len(dataset_lengths) != 1 or len(loader_lengths) != 1:
        raise ValueError(
            "Distributed train split mismatch across ranks: "
            f"{states}. Ensure train/validation file lists are broadcast once "
            "before creating per-rank datasets and samplers."
        )
    if dist_ctx.is_main:
        logger.info(
            "Distributed train loader shape verified across %s ranks: "
            "dataset_len=%s loader_len=%s"
            % (
                dist_ctx.world_size,
                next(iter(dataset_lengths)),
                next(iter(loader_lengths)),
            )
        )


def dataset_size(dataset: object) -> int:
    """Return the dataset size if available.

    Args:
        dataset (object): Dataset instance.

    Returns:
        int: Dataset length if available, else 0.

    Examples:
        >>> dataset_size([1, 2, 3])
        3
    """

    if isinstance(dataset, Sized):
        return len(cast(Sized, dataset))
    return 0
