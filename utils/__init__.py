"""
Utility helpers for data preparation, optimization, and training glue.
"""

from config import load_config

from .data import (
    PrecomputedDataset,
    extract_multiscale_features,
    prepare_data_tiles,
    subset_label_to_image_bounds,
    verify_and_clean_dataset_fast,
)
from .logging import TimedBlock, VerbosityLogger
from .losses import SegmentationLoss
from .metrics import SegmentationMetrics
from .optim import EarlyStopping, Muon, zeropower_via_newtonschulz5

__all__ = [
    "load_config",
    "TimedBlock",
    "VerbosityLogger",
    "PrecomputedDataset",
    "extract_multiscale_features",
    "prepare_data_tiles",
    "subset_label_to_image_bounds",
    "verify_and_clean_dataset_fast",
    "EarlyStopping",
    "Muon",
    "zeropower_via_newtonschulz5",
    "SegmentationLoss",
    "SegmentationMetrics",
]
