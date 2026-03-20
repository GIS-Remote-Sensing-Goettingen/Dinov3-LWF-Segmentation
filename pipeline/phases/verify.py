"""Verification phase implementation."""

from __future__ import annotations

import glob
import os

from utils import resolve_cache_dir_for_train, verify_and_clean_dataset_fast

from ..constants import DEFAULT_PROCESSED_DIR
from ..context import PhaseOutcome, RunContext
from ..phase_runner import Phase
from ..train_utils import head_uses_backbone_features, resolve_model_patch_size
from ..utils import get_model_config, resolve_path


class VerifyPhase(Phase):
    """Phase for verifying cached tile integrity."""

    name = "verify"
    config_key = "verify"

    def execute(self, context: RunContext) -> PhaseOutcome:
        """Verify cached tiles and remove corrupted entries.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts from the phase.
        """

        section = context.config.get(self.config_key, {})
        dataset_cfg = context.config.get("dataset", {})
        prepare_cfg = context.config.get("prepare", {})
        model_cfg = get_model_config(context.config)
        processed_dir = resolve_path(
            context.config, section, "processed_dir", DEFAULT_PROCESSED_DIR
        )
        requested_cache_features = dataset_cfg.get("cache_features")
        requires_backbone_features = head_uses_backbone_features(model_cfg["head"])
        cache_features = (
            bool(requested_cache_features) and requires_backbone_features
            if requested_cache_features is not None
            else None
        )
        tile_size = dataset_cfg.get("tile_size", prepare_cfg.get("tile_size"))
        patch_size = resolve_model_patch_size(model_cfg["backbone"], model_cfg["head"])
        processed_dir = resolve_cache_dir_for_train(
            processed_dir,
            tile_size,
            cache_features,
            patch_size=patch_size,
            edge_policy="drop_partial",
            logger=context.logger,
        )
        before_count = len(glob.glob(os.path.join(processed_dir, "*.pt")))
        verify_summary = verify_and_clean_dataset_fast(
            processed_dir,
            num_workers=section.get("workers"),
            logger=context.logger,
            validation_cfg=context.config.get("dataset", {}).get("validation"),
        )
        after_count = len(glob.glob(os.path.join(processed_dir, "*.pt")))
        removed = max(before_count - after_count, 0)
        metrics = {
            "tiles_total": float(after_count),
            "tiles_removed": float(removed),
            "tiles_corrupt": float(verify_summary.get("tiles_corrupt", 0)),
            "tiles_nonfinite": float(verify_summary.get("tiles_nonfinite", 0)),
            "tiles_bad_labels": float(verify_summary.get("tiles_bad_labels", 0)),
        }
        artifacts = {"processed_dir": processed_dir}
        return PhaseOutcome(metrics=metrics, artifacts=artifacts)
