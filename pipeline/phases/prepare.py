"""Preparation phase implementation."""

from __future__ import annotations

import glob
import os

import torch

from utils import prepare_data_tiles, resolve_cache_dir_for_prepare

from ..constants import (
    DEFAULT_DEVICE,
    DEFAULT_LABEL_PATH,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_IMAGES_DIR,
)
from ..context import PhaseOutcome, RunContext
from ..phase_runner import Phase
from ..utils import get_model_config, resolve_path


class PreparePhase(Phase):
    """Phase for tiling data and caching DINO features."""

    name = "prepare"
    config_key = "prepare"

    def execute(self, context: RunContext) -> PhaseOutcome:
        """Run tiling and feature caching.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts from the phase.
        """

        section = context.config.get(self.config_key, {})
        dataset_cfg = context.config.get("dataset", {})
        model_cfg = get_model_config(context.config)
        img_dir = resolve_path(
            context.config, section, "img_dir", DEFAULT_RAW_IMAGES_DIR
        )
        label_path = resolve_path(
            context.config, section, "label_path", DEFAULT_LABEL_PATH
        )
        output_dir = resolve_path(
            context.config, section, "output_dir", DEFAULT_PROCESSED_DIR
        )
        device = torch.device(section.get("device", DEFAULT_DEVICE))
        if context.dist_ctx.enabled:
            device = torch.device(f"cuda:{context.dist_ctx.local_rank}")
        cache_features = bool(section.get("cache_features", True))
        tile_size = section.get("tile_size", 512)
        output_dir = resolve_cache_dir_for_prepare(
            output_dir,
            tile_size,
            cache_features,
            model_cfg["backbone"],
            model_cfg["layers"],
            context.logger,
        )
        before_count = len(glob.glob(os.path.join(output_dir, "*.pt")))
        max_tiles = dataset_cfg.get("max_tiles")
        prepare_data_tiles(
            img_dir=img_dir,
            label_path=label_path,
            output_dir=output_dir,
            model_name=model_cfg["backbone"],
            layers=model_cfg["layers"],
            device=device,
            tile_size=tile_size,
            cache_features=cache_features,
            tile_filter_cfg=dataset_cfg.get("tile_filter"),
            workers=section.get("workers"),
            max_tiles=max_tiles,
            logger=context.logger,
        )
        after_count = len(glob.glob(os.path.join(output_dir, "*.pt")))
        metrics = {
            "tiles_total": float(after_count),
            "tiles_added": float(max(after_count - before_count, 0)),
        }
        artifacts = {"processed_dir": output_dir}
        return PhaseOutcome(metrics=metrics, artifacts=artifacts)
