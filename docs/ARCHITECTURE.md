# ARCHITECTURE

## Goal
Provide a config-driven segmentation pipeline with pluggable heads, reproducible runs, and
MLflow-compatible artifacts for research workflows.

## Folder Structure
- `main.py`: Thin CLI entry point for running the pipeline.
- `merge_all_folders.sh`: Ready-made Slurm wrapper for merging the default
  folder-level inference mosaics into one final raster on the cluster.
- `configs/`: Shipped YAML profiles for example, local, and HPC runs.
- `configs/thesis_runs/`: Standalone thesis experiment YAMLs for the clean
  split, multi-seed, and targeted-ablation rerun campaign.
- `splits/thesis_geo_v1/`: Thesis manifest staging area with example files, a
  deferred `holdout.txt` placeholder, and instructions for generating
  leakage-safe train/validation manifests from the cluster-mounted cached-tile
  inventory.
- `docs/`: Supplemental documentation such as architecture notes, changelog,
  model notes, and style guidance.
- `utility/`: Operational helper scripts that sit beside the main pipeline,
  including data acquisition, login-node recovery workflows, batch inference
  orchestration, and folder-level raster merges. `get_data_api.py` downloads
  DOP20 imagery from the original image WMS after first validating acquisition
  dates against the official `MD DOP` metadata service, while
  `recover_missing_tiles.py` inventories the canonical `folder_1` tile store,
  computes missing AOI cells, shards them into resumable recovery batches,
  audits final coverage, and promotes accepted staged tiles only after the
  coverage threshold is met.
- `pipeline/`: Phase runner, hooks, processors, and tracking utilities.
  Concrete phases are grouped in the `pipeline/phases/` package
  (`prepare.py`, `verify.py`, `train.py`, `inference.py`) with helper modules
  (`train_batches.py`, `train_xai.py`). Module-XAI internals are grouped under
  `pipeline/xai/`.
- `models/`: Segmentation heads (U-Net variants, MaskFormer-style head).
- `scripts/`: Small one-off utilities for repository workflows and data
  conversions (for example metrics export or rasterizing vector labels onto
  reference TIFF grids). The rasterize-label workflow now also supports a
  config-driven merge path that builds one canonical 1 m output grid from a
  verification raster footprint, rasterizes multiple shapefiles onto that
  grid, aligns existing label TIFFs to the same grid, merges both stacks, and
  validates coverage against the verification raster. Batch folder inference
  orchestration also lives here via `launch_batched_inference.py`, which
  derives batch-local configs from the HPC template, submits one Slurm worker
  per manifest chunk, and merges the batch prediction TIFFs after the
  controller stage finishes. A companion `merge_folder_prediction_tifs.py`
  utility can merge several already-completed folder-level `merged/predictions.tif`
  outputs into one final raster mosaic. The merge path is windowed and
  block-driven so giant GeoTIFFs are copied chunk-by-chunk instead of being
  materialized fully in RAM, and optional multicore source-window reads can
  overlap I/O for faster final mosaics.
- `utils/`: Data preparation, losses, metrics, optimization helpers, logging.
  Data internals are grouped under the `utils/data/` package (`core.py`,
  `pipeline.py`) with `utils/data/__init__.py` as the public data facade.
- `config.py`: YAML configuration loader.
- `docs/MODELS.md`: Formula-level notes on DINO layer extraction and FAPM
  projections.

## Phase Orchestration
- **Phase base class:** Standardizes enable checks, timing, and error handling.
- **PhaseRunner:** Executes phases in order and coordinates hooks/processors.
- **Hooks:** Lifecycle callbacks (run/phase/epoch/batch/tile) for extensibility.
- **Processors:** Pre/post phase modules for snapshotting and summaries.
- **Train loop decomposition:** `TrainPhase` orchestrates setup/checkpointing,
  while `phase_train_batches.py` handles per-batch optimization and
  `phase_train_xai.py` handles epoch validation/XAI artifact aggregation.
- **Stability policy:** `train.stability` controls AMP mode/dtype, fp32 loss, gradient clipping,
  non-finite handling, and checkpoint safety gates.
- **Training config schema:** `train.plots` now groups all epoch/XAI plotting options,
  `train.loss` groups main/focal/boundary loss terms, and `train.topology` groups
  skeleton/clDice settings so class indices and weights are not mixed in one block.
  The topology block also carries `skeleton_pos_weight`, which upweights the
  sparse positive skeleton pixels inside the skeleton BCE term without changing
  the main mask supervision.
  A nested `train.plots.paper` profile can additionally emit curated
  publication-oriented copies of the training/XAI figures without disabling
  the full diagnostic artifacts.
- **Model config schema:** topology-fusion controls are grouped under
  `model.fusion`, `model.lora`, and `model.boundary_gate` (with legacy flat-key
  compatibility for existing configs).
- **Dataset validation policy:** `dataset.validation` defines finite checks and allowed label
  values for both dataloading and cache verification.
- **Tile intake policy:** `dataset.tile_filter` can keep only tiles containing foreground labels
  during prepare, reducing background-only training samples.
- **Leakage-safe splitting:** train/validation partitioning now enforces both
  tile-level and source-group disjointness (derived from cached tile stems),
  including hard-fail checks for explicit split-list overlap. Explicit
  manifests may list exact cached tile stems or whole source-scene stems,
  allowing geographically defined scene-level split files to expand to all
  matching cached tiles.
- **Baseline optimizer policy:** lightweight/image-only baselines
  (`deeplabv3`, `dino_dense_probe`, `dino_segdino_light`, `unet`) use an
  AdamW-only optimization path by default, while heavier decoder heads keep
  the Muon+AdamW split path. In the split path, embeddings and 1D parameters
  stay on AdamW, while Muon applies decoupled weight decay plus paper-style
  shape-aware update scaling to matrix-like parameters.
- **Distributed forward policy:** train-time forwards wrap the selected head in
  a small normalized adapter before DDP so custom aux/boundary/skeleton outputs
  remain visible to the loss code even when the wrapper only exposes
  `forward()`.
- **Distributed epoch-boundary policy:** under DDP, rank 0 remains the source
  of truth for validation metrics, early-stopping decisions, checkpoint
  eligibility, and epoch-level XAI/plot artifacts. Validation/XAI summaries are
  broadcast back to the other ranks, and an explicit barrier is used before the
  next epoch begins so one rank cannot enter the next DDP forward while rank 0
  is still finishing epoch-end diagnostics. The process-group timeout is also
  configurable through `resources.dist_timeout_minutes` for slower HPC runs.
- **Distributed train-loop safety policy:** the train batch loop emits rank-aware
  timing logs for first/last/slow stages, escalates local non-finite recovery
  to a hard failure under DDP so ranks do not diverge, and exposes
  `resources.ddp_find_unused_parameters` so distributed runs can trade some
  performance for safer gradient synchronization while debugging intermittent
  graph/bucket mismatches.
- **Distributed split policy:** under DDP, rank 0 resolves any random
  `max_tiles` sampling and leakage-safe train/validation split once, broadcasts
  the exact file lists to the other ranks before dataset construction, and
  verifies that every rank sees the same train dataset and dataloader length
  before epoch 1 begins.
- **Distributed prepare policy:** when `resources.distributed` is enabled,
  `PreparePhase` runs tiling/cache writes only on rank 0, then broadcasts the
  resulting metrics/artifacts or failure to the other ranks before later phases
  continue. Cached tile writes use unique temp files plus an atomic final claim
  so concurrent jobs do not misclassify rename races as corrupted imagery. When
  a compatible cache directory already exists and already satisfies
  `dataset.max_tiles`, prepare short-circuits instead of rescanning source
  imagery; existing tiles also count toward any remaining top-up budget.
  DINO-oriented no-feature caches now use a hard `drop_partial` edge policy:
  prepare writes only fully fitting native-label-grid tiles, never zero-fills
  or shifts partial edge tiles inward, records the patch-size/edge-policy
  contract in cache metadata, and stores per-tile geometry metadata inside each
  payload so legacy `1020`-style caches fail closed instead of being reused.
- **Training XAI padding policy:** epoch validation/XAI plots treat
  `ignore_index`-only bottom/right label regions as batch padding, crop RGB/GT/
  prediction/XAI maps back to the real tile footprint before saving artifacts,
  and therefore avoid showing black padding strips in training-time plots.
- **DINO geometry safety policy:** train/validation no longer repair cached
  DINO inputs by cropping them down to the nearest patch multiple at runtime.
  Cached DINO tiles must already be patch-compatible and match their declared
  per-tile geometry metadata, or the job fails immediately with a cache
  geometry error.
- **Inference XAI failure policy:** scene/tile predictions remain the source of
  truth for inference success, while Grad-CAM stays best-effort. Shared
  Grad-CAM helpers now return structured failure metadata so training and
  inference can report the concrete fallback reason instead of only logging a
  generic extraction failure.
- **Native label-grid supervision policy:** prepare caches paired tensors on two
  grids: image tiles stay on the finer image resolution while label masks stay
  on the native label raster grid. Training/evaluation treat the label grid as
  the supervision source of truth by resizing logits down to label space before
  losses/metrics, inference writes prediction rasters on the same label
  grid when `label_path` is available, and train/validation dataloaders pad
  mixed-size cached tensors within a batch so scenes with different native
  scale factors remain batchable.
- **Inference border policy:** scene inference keeps full coverage with
  sliding-window starts computed from the effective DINO-compatible tile size
  and requested stride/overlap, but the final window in each row/column shifts
  inward so it lands exactly on the right/bottom border. Ordinary edge tiles
  are therefore full-size by construction instead of relying on partial-window
  reflect padding.
- **Cache compatibility policy:** label-grid-supervised tiles live in
  cache directories suffixed with `_labelgrid`, and cache metadata records the
  supervision-grid mode so older image-grid caches are rejected instead of
  being mixed into new runs.
- **Image-only baseline policy:** `model.head: unet` and
  `model.head: deeplabv3` are RGB-only baselines. They ignore `model.layers`,
  bypass cached/on-the-fly DINO feature extraction, and train directly on
  image/label tiles; `deeplabv3` still exposes auxiliary logits through the
  shared normalized-forward payload so the standard aux-supervision path stays
  available.

## Tracking & Artifacts
- MLflow-compatible file layout under `mlruns/<experiment_id>/<run_id>/`.
- `artifacts/metrics.jsonl` for lightweight visualization.
- `artifacts/run_summary.json` for run metadata and phase outputs.
- Epoch logging emits explicit validation aliases (mIoU/IoU/F1) plus decomposed
  train/validation loss components for richer MLflow dashboards.
- Epoch XAI logging can emit branch-contribution metrics (image-vs-DINO gradient
  sensitivity), per-layer DINO connection importance trends, Lite+ gate
  importance summaries, and epoch-wise DINO channel importance evolution
  artifacts (bar/trend/heatmap + JSON summaries). A module-specific XAI bundle
  can additionally log layer-fusion alpha diagnostics, boundary-gate ROC/effect
  maps, LoRA ratio distributions, and topology/skeleton connectivity summaries
  under `artifacts/plots/xai/module`. The topology diagnostics now distinguish
  the existing mask-support `clDice` proxy from explicit skeleton-branch
  precision, recall, F1, probability, and positive-rate metrics, and the
  qualitative topology panel shows both the raw skeleton probability map and
  the thresholded skeleton mask.
- Plot artifacts are grouped per run under `artifacts/plots/metrics`,
  `artifacts/plots/xai`, and `artifacts/plots/inference` to reduce clutter.
- When `train.plots.paper.enable` is on, training also emits a compact
  `artifacts/plots/metrics/training_summary.png` and curated paper copies under
  `artifacts/plots/metrics/paper` and `artifacts/plots/xai/paper`. These
  variants use lower-density layouts, contour overlays, error maps, and calmer
  trend styling so paper-candidate figures are separated from exhaustive debug
  dashboards.
- When MLflow is active, plots are written directly into these run subfolders;
  local plot directories are used only as a fallback when MLflow logging is disabled.
- Decoder family includes lightweight baselines and compact variants
  (`deeplabv3`, `dino_dense_probe`, `dino_segdino_light`, `unet`,
  `unet_lite`, `unet_lite_plus`, `unet_nano`, `unet_nano_fapm`,
  `unet_topo_fusion`) so users can trade compute for quality without changing
  pipeline wiring. `unet` is the standard image-only U-Net baseline with a
  64/128/256/512-contracting path, 1024-channel bottleneck, and symmetric
  transpose-convolution decoder, while `deeplabv3` is the official
  torchvision DeepLabV3-ResNet50 baseline with ImageNet-pretrained backbone
  weights and an auxiliary classifier enabled for standard DeepLab-style
  supervision.
  `dino_dense_probe` is the minimal dense-probe head over last-layer DINO
  tokens, and `dino_segdino_light` is a fixed paper-like lightweight SegDINO
  decoder that reforms selected DINO layers, concatenates them on one token
  grid, and predicts logits with a minimal per-pixel MLP.
  `unet_nano` keeps the deep path tiny, adds RGB priors only in late decoder
  stages (H/4, H/2), and can run with 1-4 selected DINO layers by dropping
  missing shallow skip connections, while `unet_nano_fapm` adds low-rank
  split-and-modulate DINO projections and a lightweight boundary branch fused
  into final logits.
  `unet_topo_fusion` adds learned DINO layer mixing, LoRA-style projection
  adapters, boundary feature gating, and an auxiliary skeleton stream for
  topology-aware training.

## Design Principles
- **Modularity:** Small, focused modules with explicit contracts.
- **Documentation:** Docstrings + doctests for public symbols.
- **Minimal diffs:** Avoid structural churn unless necessary.

## Workflow
1. Prepare tiles and features (optional)
   Foreground-aware filtering can be applied here (`dataset.tile_filter`) so only tiles with
   configured target labels are cached. Under the default native label-grid
   policy, cached image tensors stay on the image grid while cached labels stay
   on the coarser native label raster grid for the same map footprint. For the
   standard `unet` and `deeplabv3` baselines, this phase caches only
   image/label pairs and skips DINO feature generation entirely.
2. Verify cached tiles (readability + semantic checks) (optional)
3. Train segmentation head with per-epoch validation visualization panels (optional),
   including optional XAI dashboards (DINO attention, Grad-CAM, PCA feature maps,
   top-k feature channels, and channel-importance evolution plots). Training
   augmentation includes geometric transforms and
   optional image-only regularizers (color jitter/cutout/gridmask), which are
   cache-safe by default when precomputed features are used. Image/label tensors
   are patch-aligned to backbone patch size during train/eval on-the-fly feature
   extraction to avoid DINO token-grid misalignment, but supervision itself is
   computed on the native label grid by aligning logits down to label space.
4. Run inference (optional)
   Both `input_tif` and `input_dir` modes use the same sliding-window tiled inference
   and merge path to keep behavior consistent and memory-bounded on large inputs.
   Scene outputs now use center-weighted overlap blending, emit one compact
   explainability figure per input image, and when `input_dir` is used they now
   update one cumulative GeoTIFF (with one safety backup if the file already
   exists) on a folder-wide extent derived from the union of the input-image
   footprints, snapped to the `label_path` grid, instead of emitting per-image
5. Recover missing source imagery (optional, login node)
   `utility/recover_missing_tiles.py` is a resumable operational wrapper around
   `utility/get_data_api.py`: it inventories the canonical tile directory,
   computes missing AOI cells from filename coverage, downloads only those
   gaps into staged 2k-tile batches after passing the metadata season gate,
   writes per-batch JSONL/summary artifacts for resume, audits global coverage,
   and promotes accepted staged TIFFs into `folder_1` only when the configured
   coverage threshold is satisfied.
   prediction TIFFs. The only secondary raster artifact is that literal
   pre-update backup; there is no separate coverage raster. Directory mode still uses `label_path` for CRS,
   resolution, and grid alignment, but no longer clips output to the label
   raster extent. Overlapping scene writes now follow deterministic sorted-file
   overwrite order rather than being rejected. The prediction raster is still
   written on the label-grid resolution rather than the finer source-image
   grid. Directory inference can also consume an explicit
   `inference.input_paths_file` manifest so batch orchestrators can hand one
   fixed file list to each worker instead of rescanning the full folder. That
   manifest may be either a literal newline-delimited file list or a YAML/JSON
   scene-stem manifest such as `splits/thesis_geo_v1/val.yml`, in which case
   the listed stems are resolved back to raw GeoTIFFs via `inference.input_dir`.
   Likewise, when `prepare.enable: true` and explicit split manifests are
   configured, prepare restricts new cache creation to the union of those
   split-listed source scenes instead of tiling unrelated raw images.
5. Launch batched folder inference on Slurm (optional)
   `scripts/launch_batched_inference.py` treats `configs/config_hpc.yml` as a
   read-only template, scans `inference.input_dir`, splits the discovered files
   into fixed-size batches, writes one copied YAML config per batch with
   `tracking.mlflow.enable: false`, runs each worker from its own batch-local
   directory under `output/batches/<job_name>/runs/`, retries incomplete
   batches through a dependent controller job, and finally merges each batch
   `predictions.tif` into one final `output/batches/<job_name>/merged/predictions.tif`.
   For ad hoc post-hoc evaluation outside the main phase runner,
   `scripts/validate_prediction_rasters.py` can then score one or more
   exported prediction GeoTIFFs against a gold-label raster over the exact
   overlapping area on the label grid.
