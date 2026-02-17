# Changelog

## Rules
- Every completed task must create a new version entry and move changes from Unreleased into that release.

## [Unreleased]
### Changed
- Add prepare-time foreground label filtering (`dataset.tile_filter`) so tile caching can keep
  only tiles containing configured target labels, including multiprocessing support and skip-count
  logging (`utils/data.py`, `pipeline/phases.py`, `config_*.yml`, `README.md`, `ARCHITECTURE.md`).
- Add epoch-level branch-importance explainability metrics (gradient sensitivity of image vs DINO
  features) plus Lite+ H/4 gate-importance summaries in XAI plots and MLflow metrics
  (`pipeline/inference_utils.py`, `pipeline/phases.py`, `models/unet_lite_plus.py`,
  `config_*.yml`, `README.md`, `ARCHITECTURE.md`).
- Add epoch-wise validation DINO channel-importance tracking with grouped stable-channel bars,
  evolution trends, heatmaps, JSON artifacts, and MLflow summary metrics for interpretability
  over training (`pipeline/phases.py`, `config_*.yml`, `README.md`, `ARCHITECTURE.md`).
- Unify inference execution so `input_dir` now reuses the same sliding-window tiled engine as
  `input_tif` (with merged outputs per file), removing duplicate full-image folder logic and
  reducing OOM risk on large rasters (`pipeline/phases.py`, `README.md`, `ARCHITECTURE.md`).
- Add new `unet_nano` decoder head: an aggressively compact DINO-only U-Net variant with
  GroupNorm + GELU + Dropout2d blocks, deep supervision compatibility, and registry/docs
  integration (`models/unet_nano.py`, `models/__init__.py`, `README.md`, `ARCHITECTURE.md`).
- Update `unet_nano` to include Lite-style late RGB prior fusion at H/4 and H/2 so boundary
  details can be recovered without widening the deep decoder path
  (`models/unet_nano.py`, `README.md`, `ARCHITECTURE.md`).

## [0.1.5] - 2026-02-16
### Changed
- Replace single validation epoch tile plot with deterministic multi-tile grids (4 tile pairs / 8 subplots by default), showing GT overlays and prediction tiles with per-tile IoU/F1 titles (pipeline/phases.py, config_*.yml, README.md).
- Persist MLflow run status as numeric enum codes in run `meta.yaml` to match MLflow file-store expectations and prevent UI/API 500 errors on run search (pipeline/tracking.py).
- Add epoch-level validation XAI plots with DINO CLS/rollout focus maps, decoder Grad-CAM overlays, and top-k influential DINO feature channel visualizations (pipeline/phases.py, pipeline/inference_utils.py, config_*.yml, README.md, ARCHITECTURE.md).
- Expand MLflow epoch traces with explicit validation aliases (`val_miou`, `val_iou`, `val_f1`), full train/validation loss decomposition (`loss_*`, `val_loss_*`), and model parameter counts logged to run settings as params/tags (pipeline/phases.py, pipeline/train_utils.py, utils/losses.py, README.md, ARCHITECTURE.md).
- Fix `DinoUNetLiteHead` H/4 alignment by replacing conditional extra transposed convolution with deterministic bilinear interpolation to target SPM spatial size, preventing accidental over-upsampling on odd dimensions (models/UnetLite.py).
- Add non-breaking decoder upgrades: fix `DinoUNetV2Head` odd-size H/4 alignment with interpolation, add `forward_with_extras` intermediates for Lite explainability hooks, and introduce opt-in `unet_lite_plus` (interpolate+conv upsampling, GN+GELU residual blocks, gated H/4 fusion) while preserving existing head defaults (models/unet_v2.py, models/UnetLite.py, models/unet_lite_plus.py, models/__init__.py, README.md, ARCHITECTURE.md).
- Improve module-level architecture descriptions across decoder files to make head internals and fusion strategy easier to understand (`models/unet.py`, `models/unet_v2.py`, `models/UnetLite.py`, `models/unet_lite_plus.py`, `models/maskformer.py`).
- Make prepare-phase multiprocessing stop-on-`max_tiles` responsive by switching to bounded in-flight scheduling, adding shared stop signaling, canceling queued futures, and emitting compact drain/shutdown timing summaries to explain post-stop wait time (`utils/data.py`).
- Make DINO CLS/rollout explainability maps robust when transformer attentions are unavailable by retrying with eager attention backend, ignoring `None`/invalid attention placeholders, and falling back to hidden-state proxy focus maps (requested only on fallback) instead of returning zeros (`pipeline/inference_utils.py`).
- Add per-sample DINO PCA visualization (PC1-3) to epoch XAI plots and inference dashboards, with configurable layer selection and opt-in flags in train/inference config (`pipeline/inference_utils.py`, `pipeline/phases.py`, `config_*.yml`, `README.md`, `ARCHITECTURE.md`).
- Expose configurable AdamW weight decay for the Muon optimizer path via `train.adamw_wd`, replacing a hardcoded default with config-driven control (`pipeline/phases.py`, `config_*.yml`, `README.md`).
- Add cache-safe image-only regularization augmentations (color jitter, cutout, gridmask) and expose CE label smoothing (`train.loss.label_smoothing`) for main+aux branches while preserving geometric feature/label alignment (`utils/data.py`, `pipeline/data_splits.py`, `utils/losses.py`, `pipeline/phases.py`, `config_*.yml`, `README.md`, `ARCHITECTURE.md`).

## [0.1.4] - 2026-02-09
### Changed
- Add systemic numeric-stability controls (`train.stability`) and dataset semantic validation controls (`dataset.validation`) across config, runtime context, and docs.
- Harden training/validation loops with fp32 loss under AMP, non-finite detection, gradient clipping, step accounting, and checkpoint gating when model state is non-finite.
- Extend cache verification to remove unreadable, non-finite, or label-invalid tiles and report detailed verification counters.
- Harden Muon optimizer and Newton-Schulz orthogonalization against non-finite gradients/updates while exposing per-step skip/update stats.

## [0.1.3] - 2026-02-06
### Changed
- Partition cached tiles by tile size/feature mode and auto-select the matching cache directory for training and verification (utils/data.py, pipeline/phases.py).

## [0.1.2] - 2026-02-06
### Changed
- Restore HPC training defaults while keeping 1024 tile size for preparation (config_hpc.yml).

## [0.1.1] - 2026-02-06
### Added
- Respect `dataset.max_tiles` during preparation to avoid tiling the full dataset when sampling (utils/data.py, pipeline/phases.py).

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
