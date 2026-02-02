"""Dataset split helpers and dataloader builders."""

from __future__ import annotations

import glob
import os
import random
from typing import Optional, Sized, cast

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import PrecomputedDataset, VerbosityLogger

from .context import DistContext


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
                if isinstance(value, list):
                    combined.extend(value)
            return [str(item).strip() for item in combined if str(item).strip()]
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def resolve_dataset_splits(
    processed_dir: str,
    split_cfg: dict,
    val_fraction: float,
    logger: VerbosityLogger,
) -> tuple[list[str], list[str]]:
    """Resolve train/validation file lists for cached tiles.

    Args:
        processed_dir (str): Directory containing cached tiles.
        split_cfg (dict): Split configuration block.
        val_fraction (float): Fraction of tiles reserved for validation.
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
    if split_cfg.get("train_list"):
        train_names = set(_read_name_list(split_cfg["train_list"]))
        train_files = [f for f in all_files if _file_stem(f) in train_names]
        if split_cfg.get("val_list"):
            val_names = set(_read_name_list(split_cfg["val_list"]))
            val_files = [f for f in all_files if _file_stem(f) in val_names]
        else:
            val_files = [f for f in all_files if f not in train_files]
        if not train_files or not val_files:
            raise ValueError("Split lists produced empty train/val subsets.")
        return train_files, val_files
    files = all_files.copy()
    random.shuffle(files)
    split_idx = max(1, int(len(files) * (1 - val_fraction)))
    train_files = files[:split_idx]
    val_files = files[split_idx:] or files[-1:]
    logger.info(
        f"Using random split with {len(train_files)} train and {len(val_files)} validation tiles."
    )
    return train_files, val_files


def create_dataloaders(
    processed_dir: str,
    dataset_cfg: dict,
    train_cfg: dict,
    batch_size: int,
    logger: VerbosityLogger,
    dist_ctx: DistContext,
) -> tuple[DataLoader, Optional[DistributedSampler], Optional[DataLoader]]:
    """Build training and validation dataloaders.

    Args:
        processed_dir (str): Cached tile directory.
        dataset_cfg (dict): Dataset configuration block.
        train_cfg (dict): Training configuration block.
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
    split_cfg = dataset_cfg.get("splits", {})
    val_fraction = train_cfg.get("val_fraction", 0.2)
    train_files, val_files = resolve_dataset_splits(processed_dir, split_cfg, val_fraction, logger)
    train_dataset = PrecomputedDataset(
        processed_dir,
        augmentation_cfg=augment_cfg,
        file_subset=train_files,
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
    )
    val_loader = None
    if (not dist_ctx.enabled) or dist_ctx.is_main:
        val_dataset = PrecomputedDataset(
            processed_dir,
            augmentation_cfg={"enable": False},
            file_subset=val_files,
        )
        val_workers = train_cfg.get("val_workers", max(1, num_workers // 2))
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True,
            persistent_workers=val_workers > 0,
        )
    return train_loader, train_sampler, val_loader


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
