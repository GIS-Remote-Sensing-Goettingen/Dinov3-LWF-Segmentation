# Changelog

## Rules
- Every completed task must create a new version entry and move changes from Unreleased into that release.

## [Unreleased]
### Changed
- Rework `scripts/rasterize_vector_labels.py` into a config-driven merge
  workflow that builds one snapped `EPSG:25832` 1 m grid from the verification
  raster footprint, rasterizes all matching shapefiles onto that grid, aligns
  matching pre-rasterized label TIFFs onto the same grid, merges the raster
  and vector stacks in separate stages before a final merge, and hard-fails
  when Planet-style verification coverage drops below the configured threshold;
  also add a small committed config file plus wrapper/script regression tests
  for the new workflow
  (`scripts/rasterize_vector_labels.py`, `configs/rasterize_labels.yml`,
  `rasterize_labels.sh`, `test/test_rasterize_vector_labels.py`,
  `docs/ARCHITECTURE.md`).
- Extend `scripts/rasterize_vector_labels.py` with a
  `--resolution-factor` option so label GeoTIFFs can be generated on a
  denser pixel grid while preserving the reference CRS and geographic extent,
  and add recursive shapefile-directory processing with parallel per-shape TIFF
  outputs plus a merged final TIFF, with regression coverage for scaled
  transforms, bounds preservation, higher-resolution windowed parity, and
  nested shapefile batch merging; also expose the new vector-glob,
  vector-worker, and resolution-factor controls through the Slurm wrapper
  (`scripts/rasterize_vector_labels.py`, `rasterize_labels.sh`,
  `test/test_rasterize_vector_labels.py`).
- Fix distributed prepare correctness by making tile generation a main-rank-only
  phase under DDP, broadcasting the rank-0 outcome to other ranks, hardening
  cache writes with unique atomic temp files, and replacing misleading
  "corrupted image" rename-race logs with read/label/write-specific diagnostics;
  also make the CLI report failed phase summaries instead of always logging a
  success footer
  (`pipeline/phases/prepare.py`, `pipeline/utils.py`, `utils/data/core.py`,
  `utils/data/pipeline.py`, `main.py`, `test/test_prepare_runtime.py`,
  `docs/ARCHITECTURE.md`).
- Unify prepare path resolution with the documented config schema so the shared
  `paths.raw_images_dir`, `paths.label_path`, and `paths.processed_dir` keys are
  now the single source of truth in the shipped YAML profiles, while legacy
  `img_dir` / `output_dir` aliases remain supported in code for backward
  compatibility
  (`pipeline/phases/prepare.py`, `pipeline/utils.py`,
  `configs/config.example.yml`, `configs/config_local.yml`,
  `configs/config_hpc.yml`, `test/test_config_integrity.py`).
- Rewrite the shipped YAML profiles so every user-facing config entry now has a
  direct explanatory comment, with especially explicit cache/label/processed-dir
  semantics to make prepare-vs-train behavior easier to navigate; also update
  the README config reference to mirror the clarified cache rules
  (`configs/config.example.yml`, `configs/config_local.yml`,
  `configs/config_hpc.yml`, `README.md`).
- Fix distributed training for auxiliary-output heads by routing train-time
  forwards through a normalized adapter that preserves aux/boundary/skeleton
  payloads under DDP, adds a clear aux-required runtime guard, and restores
  `unet_nano_fapm` deep-supervision gradients; also add regression tests for
  wrapped optional outputs and `ds_head` gradient flow
  (`pipeline/phases/train.py`, `pipeline/phases/train_batches.py`,
  `pipeline/train_utils.py`, `test/test_train_utils_safety.py`,
  `docs/ARCHITECTURE.md`).
- Align the Muon optimizer with the repo's scalable paper-inspired defaults by
  adding decoupled Muon weight decay, shape-aware update scaling, safer
  non-finite handling, and embedding-aware Muon/AdamW parameter routing, plus
  new config/logging knobs for the Muon-specific settings
  (`utils/optim.py`, `pipeline/train_utils.py`, `pipeline/phases/train.py`,
  `pipeline/utils.py`, `configs/config_*.yml`, `README.md`,
  `test/test_muon_optimizer.py`, `docs/ARCHITECTURE.md`).
- Update `segmentation.sh` to request 2 GPUs by default and launch single-node
  training through
  `torchrun`, make the GPU count/config path overridable via environment
  variables, and align `configs/config_hpc.yml` with distributed training by
  default while fixing the cluster label path; also resolve the repo root and
  config path from `SLURM_SUBMIT_DIR`/absolute paths so `torchrun` works from
  Slurm's spool directory
  (`segmentation.sh`, `configs/config_hpc.yml`, `README.md`).
- Add `scripts/rasterize_vector_labels.py` to convert polygon label sources
  such as `crf/union.shp` into binary GeoTIFF masks aligned to one reference
  TIFF or a directory of reference TIFFs, including auto-windowed
  low-memory rasterization for large references, CLI progress logging, and
  optional threaded window concurrency, plus a Slurm wrapper defaulting to the
  cluster repo path under `/user/davide.mattioli/u20330/Dinov3-LWF-Segmentation`
  and the single-raster `planet_labels_2022.tif -> crf/new_labels.tif`
  workflow by default, with regression coverage for naming and rasterization behavior
  (`scripts/rasterize_vector_labels.py`,
  `test/test_rasterize_vector_labels.py`, `rasterize_labels.sh`,
  `docs/ARCHITECTURE.md`).
- Add 5%-interval training progress logs with epoch counts and ETA so long HPC
  runs expose coarse-grained status in plain logs
  (`pipeline/phases/train.py`).
- Rework inference outputs to use one scene-level explainability figure per
  input image (light-blue prediction overlay, Grad-CAM, class-probability
  panel), switch overlap merging to center-weighted probability blending, and
  add cumulative foreground-mask shapefile export reprojected to `EPSG:4326`
  with append-per-scene behavior
  (`pipeline/phases/inference.py`, `pipeline/inference_utils.py`,
  `configs/config_*.yml`, `README.md`, `docs/ARCHITECTURE.md`,
  `test/test_inference_outputs.py`).
- Refactor oversized pipeline modules into focused components to keep
  maintainability limits enforceable: split monolithic phase implementations
  into dedicated packages: `pipeline/phases/` (phase logic),
  `pipeline/xai/` (module-XAI internals), and `utils/data/`
  (data core/pipeline internals), and remove transitional thin wrapper modules
  (`pipeline/phase_*.py`, `pipeline/module_xai*.py`, `utils/data_core.py`,
  `utils/data_pipeline.py`, `utils/data.py`) to reduce file clutter.
- Harden inference checkpoint safety: inference now auto-selects the current
  run's successful train artifact when available, aborts after same-run train
  failures to avoid stale-weight inference, and enforces strict checkpoint/head
  compatibility checks (missing/unexpected/shape mismatches) before loading;
  also align HPC default inference checkpoint with the active head
  (`pipeline/phases.py`, `configs/config_hpc.yml`,
  `test/test_inference_checkpoint_safety.py`).
- Add a model-KB parameter snapshot table for all registered heads under
  `docs/MODELS.md` (total/trainable/frozen), computed with the standard binary
  setup (`num_classes=2`, `dino_channels=1024`, `layers=[5,11,17,23]`) so
  architecture capacity comparisons are documented in one place (`docs/MODELS.md`).
- Harden training stability and split integrity: `dino_dense_probe` and
  `dino_segdino_light` now use an AdamW-only optimizer path by default in
  training, high-logit warnings report batch-triggered events (with both batch
  and epoch maxima), `dino_segdino_light` adds internal GroupNorm + conservative
  output initialization to reduce early saturation, and dataset splitting now
  hard-fails on train/val overlap while enforcing source-group disjoint splits
  to prevent leakage (`pipeline/phases.py`, `pipeline/train_utils.py`,
  `models/dino_segdino_light.py`, `pipeline/data_splits.py`,
  `test/test_data_splits_leakage.py`, `test/test_train_utils_safety.py`,
  `test/test_dino_baselines.py`).
- Fix training-phase crash for AdamW-only baseline heads by making LR metric
  logging optimizer-type aware (safe `lr/lr_muon/lr_adamw` extraction for both
  Muon and plain AdamW paths) and add regression coverage
  (`pipeline/train_utils.py`, `pipeline/phases.py`,
  `test/test_train_utils_safety.py`).
- Add config-integrity tests that verify shipped YAML profiles parse correctly,
  stay key-synchronized (`configs/config.example.yml`, `configs/config_hpc.yml`,
  `configs/config_local.yml`), and remain viable for model/train parser wiring
  (`test/test_config_integrity.py`).
- Synchronize config schema surface across `configs/config.example.yml`,
  `configs/config_hpc.yml`, and `configs/config_local.yml` by adding missing distributed
  resource keys (`resources.distributed`, `resources.dist_backend`) to the
  HPC/local configs for maintainability and parity.
- Add two lightweight DINOv3 baseline heads under `models/`: `dino_dense_probe`
  (dense linear probe on last-layer tokens) and `dino_segdino_light`
  (SegDINO-style multi-layer light fusion), wire them into the model registry,
  and add strict configured-layer count checks for the SegDINO baseline
  (`models/dino_dense_probe.py`, `models/dino_segdino_light.py`,
  `models/__init__.py`, `test/test_dino_baselines.py`).
- Add grouped model config knobs for the new baselines
  (`model.dense_probe.*`, `model.segdino_light.*`) in shipped configs
  (`configs/config_*.yml`).
- Add `scripts/export_metrics_csv.py` to convert `artifacts/metrics.jsonl` into
  thesis-ready CSV tables.
- Add explanatory inline comments across `configs/config_hpc.yml`, `configs/config_local.yml`,
  and `configs/config.example.yml` so training/loss/topology/XAI options are easier to
  understand without reading code.
- Reorganize model config for readability by grouping topology-fusion controls
  into `model.fusion`, `model.lora`, and `model.boundary_gate`, and update
  head construction to accept both grouped keys and legacy flat keys
  (`models/__init__.py`, `configs/config_*.yml`, `README.md`, `docs/ARCHITECTURE.md`).
- Add `docs/MODELS.md` with formula-level documentation for DINO hidden-state extraction,
  layer-to-head mapping, and both classic/Nano FAPM modulation equations; link it
  from architecture docs for discoverability (`docs/MODELS.md`, `docs/ARCHITECTURE.md`).
- Add `unet_topo_fusion` head with learned DINO layer fusion, LoRA-style
  projection adapters, boundary-gated refinement, and an auxiliary skeleton
  stream, plus topology-aware loss terms (soft-clDice + skeleton BCE), new
  model/loss config knobs, and MLflow traces for gate and layer-mix statistics
  (`models/unet_topo_fusion.py`, `models/__init__.py`, `utils/losses.py`,
  `pipeline/train_utils.py`, `pipeline/phases.py`, `configs/config_*.yml`, `README.md`,
  `docs/ARCHITECTURE.md`, `docs/MODELS.md`).
- Fix `unet_topo_fusion` layer mixing to use joint multi-layer scoring (instead
  of per-layer independent scoring), wire config ablation toggles
  (`enable_layer_fusion`, `enable_lora`, `enable_boundary_gate`) into runtime
  behavior, and add one-time patch-grid crop warnings plus aux-resolution
  assertions to fail fast on feature/label misalignment
  (`models/unet_topo_fusion.py`, `models/__init__.py`, `pipeline/train_utils.py`).
- Improve `unet_topo_fusion` robustness by masking padded layer scores before
  softmax, deriving aux-resolution checks from fused-feature grid semantics,
  emitting gate stats as scalar floats for tracking backends, and exposing
  `model.layer_fusion_hidden` to decouple mixer capacity from projection width
  (`models/unet_topo_fusion.py`, `models/__init__.py`, `configs/config_*.yml`,
  `README.md`).
- Tune training defaults across `configs/config_hpc.yml`, `configs/config_local.yml`, and
  `configs/config.example.yml` for boundary/topology stability by reducing Cutout
  probability (`0.10`), disabling GridMask, lowering label smoothing
  (`0.08`), and enabling gradual topology supervision
  (`skeleton_weight=0.05`, `topology_weight=0.15`).
- Add module-specific XAI diagnostics for compatible heads with per-epoch
  metrics and optional sampled map panels: layer-fusion alpha argmax/entropy +
  region bars, gate-vs-boundary ROC, boundary error reduction (pre/post gate),
  LoRA update ratio maps/histograms, and topology skeleton/connectivity
  summaries with trend plots under `plots/xai/module`
  (`pipeline/module_xai.py`, `pipeline/phases.py`, `models/unet_topo_fusion.py`,
  `configs/config_*.yml`, `README.md`, `docs/ARCHITECTURE.md`).
- Reorganize training config for readability: move plot options under
  `train.plots`, split losses into `train.loss.main/focal/boundary`, move
  topology controls to `train.topology`, and add inline comments in all shipped
  configs; parser now supports both new nested keys and legacy flat keys.
  Also switch focal control to a weight-based setting (`focal.weight`) while
  preserving legacy `use_focal` behavior for backward compatibility
  (`pipeline/train_config.py`, `pipeline/phases.py`, `utils/losses.py`,
  `configs/config_*.yml`, `README.md`, `docs/ARCHITECTURE.md`).
- Fix train-phase crash in module-XAI collection when a sampled item has no
  plot payload (channel-tracking-only path) by making module sample/config
  handling null-safe (`pipeline/module_xai.py`).

## [0.1.6] - 2026-02-17
### Changed
- Add prepare-time foreground label filtering (`dataset.tile_filter`) so tile caching can keep
  only tiles containing configured target labels, including multiprocessing support and skip-count
  logging (`utils/data.py`, `pipeline/phases.py`, `configs/config_*.yml`, `README.md`, `docs/ARCHITECTURE.md`).
- Add epoch-level branch-importance explainability metrics (gradient sensitivity of image vs DINO
  features) plus Lite+ H/4 gate-importance summaries in XAI plots and MLflow metrics
  (`pipeline/inference_utils.py`, `pipeline/phases.py`, `models/unet_lite_plus.py`,
  `configs/config_*.yml`, `README.md`, `docs/ARCHITECTURE.md`).
- Add validation epoch trend plotting for mean branch importance (`image` vs `dino`) as
  `branch_importance_trends.png`, logged to XAI artifacts and updated each epoch
  (`pipeline/phases.py`, `pipeline/plotting.py`, `README.md`).
- Add per-layer DINO connection-importance tracking (using configured backbone
  layers) with epoch trend plotting and MLflow metrics, and reorganize plot
  artifacts under per-run subfolders `plots/{metrics,xai,inference}` to keep
  MLflow runs uncluttered (`pipeline/inference_utils.py`, `pipeline/phases.py`,
  `pipeline/plotting.py`, `configs/config_*.yml`, `README.md`, `docs/ARCHITECTURE.md`).
- Export split optimizer learning rates to MLflow epoch metrics (`lr_muon`,
  `lr_adamw`) while preserving the existing `lr` alias for compatibility
  (`pipeline/phases.py`, `README.md`).
- Route training and inference plot outputs directly into the active MLflow run
  artifact subfolders (`artifacts/plots/{metrics,xai,inference}`) when MLflow
  is enabled, keeping local output directories as fallback-only behavior
  (`pipeline/phases.py`, `README.md`, `docs/ARCHITECTURE.md`).
- Add `unet_nano_fapm` head with low-rank split-and-modulate DINO projections
  (NanoFAPM), late RGB fusion, and a lightweight boundary branch fused into
  final logits (`models/unet_nano_fapm.py`, `models/__init__.py`, `README.md`,
  `docs/ARCHITECTURE.md`).
- Extend segmentation loss with optional focal classification term and boundary
  BCE supervision, including boundary-target generation and train/eval wiring
  for heads exposing `edge_logits` (`utils/losses.py`, `pipeline/train_utils.py`,
  `pipeline/phases.py`, `configs/config_*.yml`, `README.md`).
- Add epoch-wise validation DINO channel-importance tracking with grouped stable-channel bars,
  evolution trends, heatmaps, JSON artifacts, and MLflow summary metrics for interpretability
  over training (`pipeline/phases.py`, `configs/config_*.yml`, `README.md`, `docs/ARCHITECTURE.md`).
- Unify inference execution so `input_dir` now reuses the same sliding-window tiled engine as
  `input_tif` (with merged outputs per file), removing duplicate full-image folder logic and
  reducing OOM risk on large rasters (`pipeline/phases.py`, `README.md`, `docs/ARCHITECTURE.md`).
- Add new `unet_nano` decoder head: an aggressively compact DINO-only U-Net variant with
  GroupNorm + GELU + Dropout2d blocks, deep supervision compatibility, and registry/docs
  integration (`models/unet_nano.py`, `models/__init__.py`, `README.md`, `docs/ARCHITECTURE.md`).
- Update `unet_nano` to include Lite-style late RGB prior fusion at H/4 and H/2 so boundary
  details can be recovered without widening the deep decoder path
  (`models/unet_nano.py`, `README.md`, `docs/ARCHITECTURE.md`).

## [0.1.5] - 2026-02-16
### Changed
- Replace single validation epoch tile plot with deterministic multi-tile grids (4 tile pairs / 8 subplots by default), showing GT overlays and prediction tiles with per-tile IoU/F1 titles (pipeline/phases.py, configs/config_*.yml, README.md).
- Persist MLflow run status as numeric enum codes in run `meta.yaml` to match MLflow file-store expectations and prevent UI/API 500 errors on run search (pipeline/tracking.py).
- Add epoch-level validation XAI plots with DINO CLS/rollout focus maps, decoder Grad-CAM overlays, and top-k influential DINO feature channel visualizations (pipeline/phases.py, pipeline/inference_utils.py, configs/config_*.yml, README.md, docs/ARCHITECTURE.md).
- Expand MLflow epoch traces with explicit validation aliases (`val_miou`, `val_iou`, `val_f1`), full train/validation loss decomposition (`loss_*`, `val_loss_*`), and model parameter counts logged to run settings as params/tags (pipeline/phases.py, pipeline/train_utils.py, utils/losses.py, README.md, docs/ARCHITECTURE.md).
- Fix `DinoUNetLiteHead` H/4 alignment by replacing conditional extra transposed convolution with deterministic bilinear interpolation to target SPM spatial size, preventing accidental over-upsampling on odd dimensions (models/UnetLite.py).
- Add non-breaking decoder upgrades: fix `DinoUNetV2Head` odd-size H/4 alignment with interpolation, add `forward_with_extras` intermediates for Lite explainability hooks, and introduce opt-in `unet_lite_plus` (interpolate+conv upsampling, GN+GELU residual blocks, gated H/4 fusion) while preserving existing head defaults (models/unet_v2.py, models/UnetLite.py, models/unet_lite_plus.py, models/__init__.py, README.md, docs/ARCHITECTURE.md).
- Improve module-level architecture descriptions across decoder files to make head internals and fusion strategy easier to understand (`models/unet.py`, `models/unet_v2.py`, `models/UnetLite.py`, `models/unet_lite_plus.py`, `models/maskformer.py`).
- Make prepare-phase multiprocessing stop-on-`max_tiles` responsive by switching to bounded in-flight scheduling, adding shared stop signaling, canceling queued futures, and emitting compact drain/shutdown timing summaries to explain post-stop wait time (`utils/data.py`).
- Make DINO CLS/rollout explainability maps robust when transformer attentions are unavailable by retrying with eager attention backend, ignoring `None`/invalid attention placeholders, and falling back to hidden-state proxy focus maps (requested only on fallback) instead of returning zeros (`pipeline/inference_utils.py`).
- Add per-sample DINO PCA visualization (PC1-3) to epoch XAI plots and inference dashboards, with configurable layer selection and opt-in flags in train/inference config (`pipeline/inference_utils.py`, `pipeline/phases.py`, `configs/config_*.yml`, `README.md`, `docs/ARCHITECTURE.md`).
- Expose configurable AdamW weight decay for the Muon optimizer path via `train.adamw_wd`, replacing a hardcoded default with config-driven control (`pipeline/phases.py`, `configs/config_*.yml`, `README.md`).
- Add cache-safe image-only regularization augmentations (color jitter, cutout, gridmask) and expose CE label smoothing (`train.loss.label_smoothing`) for main+aux branches while preserving geometric feature/label alignment (`utils/data.py`, `pipeline/data_splits.py`, `utils/losses.py`, `pipeline/phases.py`, `configs/config_*.yml`, `README.md`, `docs/ARCHITECTURE.md`).

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
- Restore HPC training defaults while keeping 1024 tile size for preparation (configs/config_hpc.yml).

## [0.1.1] - 2026-02-06
### Added
- Respect `dataset.max_tiles` during preparation to avoid tiling the full dataset when sampling (utils/data.py, pipeline/phases.py).

## [0.1.0] - 2026-02-06
### Added
- Phase-based orchestration with MLflow-compatible logging to improve maintainability and tracking (main.py, utils/*.py, models/*.py, config.py).
- Image-processing ETA logs and `max_tiles` sampling for smaller training subsets (utils/data.py, pipeline/data_splits.py, configs/config_*.yml).
- Per-run log files with timestamp and run ID to avoid interleaved output (pipeline/utils.py, main.py, configs/config_*.yml).
- Prepare-phase multiprocessing to speed up tiling (utils/data.py, pipeline/phases.py, configs/config_*.yml).
- Folder inference with XAI dashboards (attention, confidence, entropy) and plot outputs (pipeline/phases.py, pipeline/inference_utils.py, configs/config_*.yml).
- Per-epoch validation tile plots for qualitative monitoring (pipeline/phases.py, configs/config_*.yml).

### Changed
- HPC defaults for `batch_size` and `num_workers` to prevent invalid settings (configs/config_hpc.yml).
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
