# Changelog

## Rules
- Every completed task must create a new version entry and move changes from Unreleased into that release.

## [Unreleased]

## [0.1.0] - 2026-02-06
### Added
- Phase-based orchestration with MLflow-compatible logging to improve maintainability and tracking (main.py, utils/*.py, models/*.py, config.py).
- Image-processing ETA logs and `max_tiles` sampling for smaller training subsets (utils/data.py, pipeline/data_splits.py, config_*.yml).
- Per-run log files with timestamp and run ID to avoid interleaved output (pipeline/utils.py, main.py, config_*.yml).
- Prepare-phase multiprocessing to speed up tiling (utils/data.py, pipeline/phases.py, config_*.yml).
- Folder inference with XAI dashboards (attention, confidence, entropy) and plot outputs (pipeline/phases.py, pipeline/inference_utils.py, config_*.yml).
- Per-epoch validation tile plots for qualitative monitoring (pipeline/phases.py, config_*.yml).

### Changed
- HPC defaults for `batch_size` and `num_workers` to prevent invalid settings (config_hpc.yml).
- Validation now computes features on the fly when cache_features is disabled (pipeline/phases.py, pipeline/train_utils.py).

### Fixed
- Create log directories automatically to prevent FileNotFoundError on logging (pipeline/utils.py).
- Grad-CAM fallback for attention maps when the backbone provides no attentions (pipeline/inference_utils.py, pipeline/phases.py).

EXAMPLE
## [0.0.1]
- Description:
- file touched:
- reason:
- problems fixed:
