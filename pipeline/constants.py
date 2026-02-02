"""Shared defaults and environment setup for the pipeline.

This module centralizes default paths and settings used by pipeline phases.
"""

from __future__ import annotations

import os

import torch

DEFAULT_RAW_IMAGES_DIR = (
    "//mnt/ceph-hdd/projects/mthesis_davide_mattioli/patches_mt/folder_1/"
)
DEFAULT_LABEL_PATH = (
    "/run/media/mak/Partition of 1TB disk/SH_dataset/planet_labels_2022.tif"
)
DEFAULT_PROCESSED_DIR = (
    "/mnt/ceph-hdd/projects/mthesis_davide_mattioli/processed/folder_1/"
)
DEFAULT_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-sat493m"
DEFAULT_LAYERS = [5, 11, 17, 23]
DEFAULT_HEAD = "unet"
DEFAULT_NUM_CLASSES = 2
DEFAULT_DINO_CHANNELS = 1024
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TRACKING_DIR = "mlruns"
DEFAULT_EXPERIMENT_ID = "0"


def ensure_env_defaults() -> None:
    """Apply safe environment defaults for CPU threads and CUDA alloc.

    This avoids oversubscription and reduces CUDA allocator fragmentation in
    research workflows.

    Examples:
        >>> ensure_env_defaults()
        >>> os.environ.get("OMP_NUM_THREADS") is not None
        True
    """

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
