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
from ..train_utils import head_uses_backbone_features, resolve_model_patch_size
from ..utils import broadcast_main_object, get_model_config, resolve_path


class PreparePhase(Phase):
    """Phase for tiling data and caching DINO features."""

    name = "prepare"
    config_key = "prepare"

    def _execute_main_rank(self, context: RunContext) -> PhaseOutcome:
        """Run prepare work once on rank 0 or in non-distributed mode.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Prepared cache metrics and artifact paths.
        """

        section = context.config.get(self.config_key, {})
        dataset_cfg = context.config.get("dataset", {})
        model_cfg = get_model_config(context.config)
        img_dir = resolve_path(
            context.config,
            section,
            "raw_images_dir",
            DEFAULT_RAW_IMAGES_DIR,
            legacy_keys=("img_dir",),
        )
        label_path = resolve_path(
            context.config, section, "label_path", DEFAULT_LABEL_PATH
        )
        output_dir = resolve_path(
            context.config,
            section,
            "processed_dir",
            DEFAULT_PROCESSED_DIR,
            legacy_keys=("output_dir",),
        )
        device = torch.device(section.get("device", DEFAULT_DEVICE))
        if context.dist_ctx.enabled:
            device = torch.device(f"cuda:{context.dist_ctx.local_rank}")
        requested_cache_features = bool(section.get("cache_features", True))
        uses_backbone_features = head_uses_backbone_features(model_cfg["head"])
        cache_features = requested_cache_features and uses_backbone_features
        if requested_cache_features and not uses_backbone_features:
            context.logger.info(
                "Head '%s' is image-only; prepare will cache image/label tiles "
                "without DINO features." % model_cfg["head"]
            )
        tile_size = section.get("tile_size", 512)
        patch_size = resolve_model_patch_size(model_cfg["backbone"], model_cfg["head"])
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
            patch_size=patch_size,
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

    def execute(self, context: RunContext) -> PhaseOutcome:
        """Run tiling and feature caching.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts from the phase.
        """

        local_error: Exception | None = None
        sync_payload: dict[str, object | None] = {"error": None, "outcome": None}
        if (not context.dist_ctx.enabled) or context.dist_ctx.is_main:
            try:
                outcome = self._execute_main_rank(context)
                sync_payload["outcome"] = {
                    "metrics": outcome.metrics,
                    "artifacts": outcome.artifacts,
                }
            except Exception as exc:
                local_error = exc
                sync_payload["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        sync_payload = broadcast_main_object(context.dist_ctx, sync_payload)
        error = sync_payload.get("error")
        if error is not None:
            if local_error is not None:
                raise local_error
            error_map = error if isinstance(error, dict) else {}
            message = str(error_map.get("message", "prepare failed on rank 0"))
            raise RuntimeError(f"Prepare phase failed on rank 0: {message}")
        outcome_map = sync_payload.get("outcome")
        if not isinstance(outcome_map, dict):
            raise RuntimeError("Prepare phase did not produce a distributed outcome.")
        metrics = outcome_map.get("metrics")
        artifacts = outcome_map.get("artifacts")
        if not isinstance(metrics, dict) or not isinstance(artifacts, dict):
            raise RuntimeError("Prepare phase returned an invalid distributed outcome.")
        return PhaseOutcome(metrics=metrics, artifacts=artifacts)
