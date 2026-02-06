# Changelog

## [Unreleased]
- Rule: Every completed task must create a new version entry and move changes from Unreleased into that release.

## [0.1.0] - 2026-02-06
- Description: Refactored the pipeline orchestration with phases, hooks, and MLflow-compatible file logging.
- file touched: main.py, utils/*.py, models/*.py, config.py
- reason: Improve maintainability, extensibility, and tracking for research workflows.
- problems fixed: Centralized run artifacts, clarified docstrings, and added structured run summaries.
- Description: Ensure log file directories are created automatically.
- file touched: pipeline/utils.py
- reason: Prevent crashes when log file paths point to missing directories.
- problems fixed: FileNotFoundError during logger initialization.
- Description: Add image-processing ETA logs and a max_tiles cap for sampling training tiles.
- file touched: utils/data.py, pipeline/data_splits.py, config_*.yml
- reason: Improve progress visibility and allow smaller training subsets.
- problems fixed: Long-running tiling runs without ETA; unbounded tile counts.
- Description: Create a new log file per run with timestamp and run ID.
- file touched: pipeline/utils.py, main.py, config_*.yml
- reason: Prevent interleaved logs and improve run traceability.
- problems fixed: Mixed log output across runs in shared log files.
- Description: Add prepare-phase multiprocessing and per-image ETA logging.
- file touched: utils/data.py, pipeline/phases.py, config_*.yml
- reason: Speed up tile generation and improve progress visibility.
- problems fixed: Slow single-thread tiling and unclear progress tracking.
- Description: Adjust HPC training defaults for batch_size and num_workers.
- file touched: config_hpc.yml
- reason: Avoid invalid batch_size=0 and keep training single-threaded.
- problems fixed: Training stalls from invalid batch_size.
- Description: Compute validation features on the fly when cache_features is disabled.
- file touched: pipeline/phases.py, pipeline/train_utils.py
- reason: Keep validation compatible with zero-cache mode.
- problems fixed: list index out of range during evaluation.
- Description: Add folder inference and XAI dashboards with attention, confidence, and entropy.
- file touched: pipeline/phases.py, pipeline/inference_utils.py, config_*.yml
- reason: Support batch inference and richer explainability outputs.
- problems fixed: Single-file inference only and missing visualization outputs.
- Description: Add per-epoch validation tile plots with ground truth and prediction.
- file touched: pipeline/phases.py, config_*.yml
- reason: Provide visual training feedback each epoch.
- problems fixed: Missing qualitative monitoring of segmentation outputs.
- Description: Add Grad-CAM fallback for inference attention maps.
- file touched: pipeline/inference_utils.py, pipeline/phases.py
- reason: Ensure explainability plots remain available when attentions are missing.
- problems fixed: Attention maps returning zeros for DINO backbones without attentions.

EXAMPLE
## [0.0.1]
- Description:
- file touched:
- reason:
- problems fixed:
