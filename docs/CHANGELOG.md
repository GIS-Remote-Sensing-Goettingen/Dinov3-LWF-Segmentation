# Changelog

## Rules
- Every completed task must create a new version entry and move changes from Unreleased into that release.

## [Unreleased]
### Changed
- Add an official torchvision `deeplabv3` RGB-only baseline head around
  `deeplabv3_resnet50` with ImageNet-pretrained backbone weights, aux-logit
  support, AdamW-only optimizer routing, registry/test coverage, and a thesis
  rerun config `R12_deeplabv3_split_s1337.yml` so the experiment pack can
  compare the existing image-only U-Net baseline against a widely recognized
  DeepLabV3 baseline without adding a new third-party dependency
  (`models/deeplabv3.py`, `models/__init__.py`, `pipeline/train_utils.py`,
  `test/test_dino_baselines.py`, `test/test_train_utils_safety.py`,
  `configs/thesis_runs/R12_deeplabv3_split_s1337.yml`,
  `configs/thesis_runs/README.md`, `README.md`, `docs/ARCHITECTURE.md`,
  `configs/config*.yml`).
- Add one thesis-oriented coarse-vs-refined supervision baseline around the
  existing topology-fusion head: directory inference manifests now accept the
  same scene-stem YAML/JSON files used by split resolution, prepare can
  restrict new cache creation to the union of explicit split-listed source
  scenes, and `configs/thesis_runs/` now includes one coarse-label training
  config plus refined/coarse validation-scene inference configs and runbook
  commands for external scoring against `crf/final_labels_1m.tif`
  (`pipeline/phases/inference.py`, `pipeline/phases/prepare.py`,
  `utils/data/pipeline.py`, `test/test_inference_outputs.py`,
  `test/test_prepare_runtime.py`, `configs/thesis_runs/*.yml`,
  `configs/thesis_runs/README.md`, `README.md`, `docs/ARCHITECTURE.md`).
- Add a thesis-focused HPC experiment pack under `configs/thesis_runs/` plus
  `splits/thesis_geo_v1/` manifest templates so the master-thesis comparison
  campaign can be rerun with explicit train/validation manifests, isolated
  per-run weights/logs, multi-seed top-model configs, targeted
  boundary/topology ablations, exact `CONFIG_PATH=... sbatch segmentation.sh`
  submission commands, and a deferred empty `holdout.txt` placeholder for the
  later geographic-holdout phase (`configs/thesis_runs/*.yml`,
  `configs/thesis_runs/README.md`, `splits/thesis_geo_v1/README.md`,
  `splits/thesis_geo_v1/*.example.txt`, `splits/thesis_geo_v1/holdout.txt`,
  `docs/ARCHITECTURE.md`).
- Allow explicit train/validation split manifests to list either exact cached
  tile stems or whole source-scene stems, so geographically defined split
  files based on `dop20_<x>_<y>` scene coordinates expand automatically to all
  matching cached tiles (`pipeline/data_splits.py`,
  `test/test_data_splits_leakage.py`).
- Align the thesis rerun configs with the cluster's `tiles_1024_*` cache
  layout and keep image-only U-Net training compatible with the existing
  legacy `tiles_1024_nofeat_labelgrid` cache when the old cache metadata does
  not yet record the later `drop_partial` edge-policy suffix
  (`configs/thesis_runs/*.yml`, `utils/data/core.py`,
  `test/test_prepare_runtime.py`).
- Flatten nested YAML scene batches in explicit split manifests so the
  thesis-run `train.yml` and `val.yml` files expand into individual
  `dop20_<x>_<y>_1km_20cm` scene stems instead of being parsed as stringified
  list literals, which previously produced empty train/validation subsets at
  runtime (`pipeline/data_splits.py`, `test/test_data_splits_leakage.py`).
- Enforce `dataset.max_tiles` after split resolution as well, so explicit
  train/validation manifests can still run a smaller thesis rerun subset
  without editing the manifest files themselves; the cap now samples train and
  validation subsets proportionally while keeping both non-empty
  (`pipeline/data_splits.py`, `test/test_data_splits_leakage.py`).
- Add a small prediction-raster validation utility that scores one or more
  exported GeoTIFF predictions against a gold-label raster over their
  overlapping area using windowed reads and nearest-neighbor reprojection onto
  the label grid, plus a thin `scripts/` compatibility wrapper and regression
  coverage for overlap-only scoring and no-overlap handling
  (`utility/validate_prediction_rasters.py`,
  `scripts/validate_prediction_rasters.py`,
  `test/test_validate_prediction_rasters.py`, `README.md`,
  `docs/ARCHITECTURE.md`).
- Point the shipped HPC profile at the handmade binary 1 m gold-label raster
  `utility/test/golden_labels_gt_1m_mosaic_25832.tif` so directory inference
  can align its shared prediction GeoTIFF to that label grid during manual
  checkpoint testing (`configs/config_hpc.yml`).
- Exclude the aggregate `rho_mean` series from the LoRA module-XAI trend plot
  so `module_lora_trends.png` focuses on the boundary/interior ratios that are
  actually compared in the report, while keeping the underlying scalar metric
  logging unchanged (`pipeline/xai/module_xai_epoch.py`).
- Increase the paper readability of the LoRA module-XAI panel by stretching
  the ratio-map overlay contrast with robust percentile bounds and switching to
  a stronger `viridis` overlay so track/background differences are easier to
  inspect in exported figures (`pipeline/inference_utils.py`,
  `pipeline/xai/module_xai.py`, `test/test_inference_outputs.py`).
- Add sparse-positive skeleton weighting to the topology loss config and
  update topology XAI to separate the existing mask-support clDice proxy from
  explicit skeleton-branch precision/recall/F1, probability, and positive-rate
  diagnostics; topology panels now show the raw skeleton probability map
  alongside the thresholded skeleton view, and the trend artifacts split branch
  health from connected-component counts (`configs/config*.yml`,
  `pipeline/train_config.py`, `pipeline/phases/train.py`, `utils/losses.py`,
  `pipeline/xai/module_xai.py`, `pipeline/xai/module_xai_epoch.py`,
  `test/test_config_integrity.py`, `test/test_train_utils_safety.py`,
  `test/test_module_xai_epoch.py`).
- Add `scripts/render_unet_topo_fusion_schema.py`, a small matplotlib-backed
  CLI that renders a thesis-ready schematic of the `unet_topo_fusion` decoder
  head to SVG/PDF/PNG so the architecture figure can be regenerated and
  iterated from code while matching the hand-authored visual style used in the
  example plots and the actual decoder structure of the topology-fusion head;
  refine the layout through render-review iterations so the RGB prior labels,
  supervision branches, the image-to-prior routing, and the final refinement
  stage read more cleanly at paper scale.
- Ignore local downloader smoke-test outputs and temporary checkpoint files so
  ad hoc recovery/download experiments under `utility/test/download_*`,
  `utility/test/md_dop_date_distribution/`, and local `tmp*.pth` files stop
  showing up as untracked git noise (`.gitignore`).
- Stop tracking the moved TIFF and `.aux.xml` files under `utility/test/` so
  these local raster artifacts no longer appear in the commit while remaining
  available in the working tree for ad hoc inspection (`.gitignore`).
- Add a resumable login-node recovery wrapper for missing DOP20 source tiles:
  `utility/recover_missing_tiles.py` now inventories the canonical `folder_1`
  tile store by filename, computes missing AOI cells from the downloader's
  snapped grid, writes deterministic 2k-tile recovery shard manifests, resumes
  incomplete shards from per-batch JSONL/summary state, audits final accepted
  coverage across staged downloads plus existing tiles, and promotes staged
  TIFFs into `folder_1` only after the configured coverage threshold is met;
  add regression coverage for missing-tile planning, shard resume behavior, and
  safe promotion, plus lightweight `scripts/` compatibility wrappers that
  forward to the moved `utility/` helpers so existing tests and entrypoint
  paths keep working (`utility/recover_missing_tiles.py`,
  `test/test_recover_missing_tiles.py`, `scripts/launch_batched_inference.py`,
  `scripts/merge_folder_prediction_tifs.py`,
  `scripts/rasterize_vector_labels.py`, `scripts/export_metrics_csv.py`,
  `docs/ARCHITECTURE.md`).
- Refactor `utility/get_data_api.py` into import-safe helper functions so the
  downloader can build `GetFeatureInfo` metadata requests, extract acquisition
  dates, and evaluate the preferred spring/summer season in unit-tested code;
  also add a capped CLI smoke-test mode plus a small wrapper under
  `utility/test/` that can download only 1-2 tiles into a utility-local test
  directory, reject all-white blank WMS tiles instead of saving them as valid
  outputs, and optionally target explicit `x,y` tile origins for known-good
  smoke checks, alongside regression tests for metadata parameter building,
  nested date parsing, blank-tile rejection, October rejection, and explicit
  WMS `ServiceExceptionReport` handling for non-queryable metadata layers; the
  metadata request path now also uses the smaller `1000 x 1000` virtual query
  canvas accepted by the official `MD DOP` service instead of the image
  downloader's `5000 x 5000` raster size, and each tile download is now gated
  by a metadata pre-check so only spring/summer `A_DATUM` tiles are fetched
  from the original image WMS while October tiles are skipped before any image
  request; the season gate now opens on April 20 instead of waiting until May.
  Replace the ad hoc `utility/test/temp.py` probe with a filename-based folder
  coverage mapper that scans `folder_1`, extracts tile origins from TIFF
  names, streams the Desktop `predictions*.tif` rasters (excluding the merged
  mosaic) in logged row chunks, and overlays the tiles containing positive
  prediction labels on top of the blue folder coverage map, with regression
  coverage for filename parsing, prediction discovery, and chunk-to-tile
  aggregation. The smoke-download wrapper now can also expand a lower-left
  `START_X, START_Y` plus `BLOCK_SIZE_KM` into a full rectangular block of
  explicit tile origins for quick 5 km x 5 km tests
  (`utility/get_data_api.py`, `test/test_get_data_api_metadata.py`,
  `utility/test/run_get_data_api_smoke.sh`, `utility/test/temp.py`,
  `test/test_md_dop_date_distribution.py`). Fix a regression where the `MD DOP`
  metadata query canvas size accidentally leaked into the image `GetMap`
  request, shrinking downloaded imagery to `1000 x 1000`; image downloads now
  stay at the original `5000 x 5000` while metadata queries remain
  `1000 x 1000`
  (`utility/get_data_api.py`, `test/test_get_data_api_metadata.py`,
  `utility/test/run_get_data_api_smoke.sh`, `utility/test/temp.py`,
  `test/test_md_dop_date_distribution.py`).
- Add `scripts/merge_folder_prediction_tifs.py` so multiple completed
  folder-level `merged/predictions.tif` outputs can be combined into one final
  GeoTIFF using the same grid-compatible overwrite merge helper as the batch
  orchestrator; also rework that merge helper so it now copies rasters
  window-by-window, creates tiled BigTIFF-ready outputs, and can prefetch
  source windows with multiple read workers instead of loading whole giant
  TIFFs into RAM (`scripts/launch_batched_inference.py`,
  `pipeline/inference_utils.py`, `scripts/merge_folder_prediction_tifs.py`,
  `test/test_launch_batched_inference.py`,
  `test/test_merge_folder_prediction_tifs.py`, `README.md`,
  `docs/ARCHITECTURE.md`).
- Add `merge_all_folders.sh` as a ready-made Slurm wrapper that submits the
  default four-folder final merge on the cluster without requiring a fragile
  long `sbatch --wrap` command (`merge_all_folders.sh`, `README.md`,
  `docs/ARCHITECTURE.md`).
- Extend the batch inference launcher so it can take an explicit
  `--input-paths-file` selection file, including the uncovered-tile CSV format
  emitted by `utility/test/temp.py` via its `tile_name,x,y` rows, making it
  possible to launch a targeted `missing_tiles` re-inference campaign without
  rescanning the entire inference directory (`utility/launch_batched_inference.py`,
  `test/test_launch_batched_inference.py`).
- Make the local file-length pre-commit hook respect git-ignored paths so
  generated coverage CSVs/PNGs under ignored utility output folders do not fail
  repository checks (`scripts/check_file_length.py`).
- Make the thin `scripts/*` compatibility wrappers prepend the repository root
  to `sys.path` before importing from `utility/`, so direct absolute-path
  execution in the cluster checkout works instead of failing with
  `ModuleNotFoundError: No module named 'utility'`
  (`scripts/launch_batched_inference.py`, `scripts/merge_folder_prediction_tifs.py`,
  `scripts/rasterize_vector_labels.py`, `scripts/export_metrics_csv.py`).
- Fix the temporary folder-coverage overlay helper so its prediction scan uses
  a count-based `WarpedVRT` path on the shared 1 km folder grid so it can
  compare per-tile positive-pixel counts from the combined non-merged
  `predictions*.tif` rasters against `planet_labels_2022.tif`, color tiles
  with the requested `violet > orange > blue` priority, apply the strict
  `prediction_count > 0.6 * planet_count` violet threshold, emit a larger map
  with legend, write covered/uncovered/violet CSV exports plus expanded
  summary metrics, and add a second histogram-style output showing the
  prediction-vs-Planet positive-pixel ratio distribution from `0%` up to the
  observed maximum. Keep the large Desktop prediction TIFF scan sequential and
  expose a GDAL cache limit so the helper stays memory-safe instead of trying
  to process multiple rasters at once. Fix the Planet-label comparison path so
  rasters such as the `EPSG:3857` Planet label mosaic are reprojected into the
  shared `folder_1` tile CRS before per-tile counts are computed. Ignore the generated
  `utility/test/folder_1_coverage/` output directory so local map renders do
  not appear as untracked files (`utility/test/temp.py`,
  `test/test_md_dop_date_distribution.py`, `.gitignore`).
- Point the shipped HPC inference template at `patches_mt/folder_4` so the
  default batch-launch workflow now scans `folder_4`
  (`configs/config_hpc.yml`).
- Add manifest-driven folder inference selection through
  `inference.input_paths_file`, plus a new
  `scripts/launch_batched_inference.py` Slurm orchestrator that reuses
  `configs/config_hpc.yml` as a read-only template, splits directory inference
  into fixed-size batch jobs, disables MLflow in the generated batch configs so
  outputs/logs stay under one orchestration root instead of `mlruns`, retries
  incomplete batches through a controller stage, and merges the per-batch
  prediction TIFFs into one final GeoTIFF
  (`pipeline/phases/inference.py`, `scripts/launch_batched_inference.py`,
  `test/test_inference_outputs.py`, `test/test_launch_batched_inference.py`,
  `configs/config_*.yml`, `README.md`, `docs/ARCHITECTURE.md`).
- Rework directory-mode inference so it now updates one shared GeoTIFF on the
  native label grid (taking one pre-update backup when that file already
  exists) instead of writing per-image prediction TIFFs or relying on the old
  cumulative shapefile workflow; also clarify the reused `output_tif` /
  `output_dir` config semantics, normalize payload-style head outputs during
  inference, skip only raster-masked nodata tiles instead of all-zero imagery,
  use disk-backed scene accumulators for lower-memory large-scene runs, build
  the cumulative directory-mode GeoTIFF from the union of the input image
  footprints while reusing `label_path` only for CRS/resolution/grid
  alignment, keep only the literal pre-update backup raster instead of any
  separate coverage TIFF, overwrite overlapping scene writes in sorted order
  instead of skipping them, and fix a tiled-inference loop bug that previously
  processed only the last x-window in each row; also add regression coverage for
  cumulative raster creation, backup, overwrite, folder-union grid creation,
  large-scene tile coverage, payload heads, and label-path validation
  (`pipeline/phases/inference.py`, `pipeline/inference_utils.py`,
  `test/test_inference_outputs.py`, `configs/config_*.yml`, `README.md`,
  `docs/ARCHITECTURE.md`).
- Redesign DINO tiling so prepare drops partial edge tiles instead of shifting
  or zero-filling them, inference uses border-aligned sliding windows instead
  of partial reflected edge pads, cached tiles now carry per-tile geometry
  metadata, and DINO train/validation paths fail fast on legacy or
  patch-incompatible caches instead of cropping inputs at runtime
  (`utils/data/core.py`, `utils/data/pipeline.py`, `pipeline/data_splits.py`,
  `pipeline/phases/inference.py`, `pipeline/phases/train.py`,
  `pipeline/phases/train_batches.py`, `pipeline/phases/prepare.py`,
  `pipeline/phases/verify.py`, `pipeline/train_utils.py`,
  `test/test_prepare_runtime.py`, `test/test_train_utils_safety.py`,
  `docs/ARCHITECTURE.md`).
- Make no-feature cache compatibility patch-size aware so DINO heads no longer
  silently reuse old image-only cache geometry, and crop training XAI/epoch
  plots to the real non-padded tile footprint instead of showing black
  bottom/right padding artifacts
  (`utils/data/core.py`, `pipeline/phases/train_xai.py`,
  `test/test_prepare_runtime.py`, `test/test_train_utils_safety.py`,
  `docs/ARCHITECTURE.md`).
- Fix baseline-head aux-supervision capability detection so
  `dino_dense_probe` and `dino_segdino_light` no longer claim auxiliary logits
  they do not return; training now ignores `aux_weight` for those heads
  instead of failing on the first forward pass
  (`pipeline/train_utils.py`, `test/test_train_utils_safety.py`).
- Rebuild `dino_segdino_light` as a fixed paper-like lightweight SegDINO
  decoder (`reform -> align -> concat -> MLP`), remove the old
  `model.segdino_light.*` config block from shipped examples, and make older
  configs fail fast with a targeted error instead of silently using ignored
  per-head knobs (`models/dino_segdino_light.py`, `models/__init__.py`,
  `configs/config_*.yml`, `README.md`, `test/test_dino_baselines.py`,
  `docs/ARCHITECTURE.md`).
- Make prepare reuse compatible no-feature caches when they already satisfy
  `dataset.max_tiles`, and count existing tiles toward the remaining top-up
  budget instead of rescanning imagery from zero on every run
  (`utils/data/pipeline.py`, `test/test_prepare_runtime.py`,
  `docs/ARCHITECTURE.md`).
- Simplify the branch XAI trend figure by dropping the third `|image-dino|`
  balance curve and renaming the visible plot wording from branch
  `importance` to branch `contribution` so the figure reads more cleanly in
  paper-oriented outputs (`pipeline/plotting.py`, `test/test_plotting.py`,
  `README.md`, `docs/ARCHITECTURE.md`).
- Add `warning()` support to `VerbosityLogger` so train/inference warning sites
  no longer crash phase execution when they emit crop or fallback diagnostics
  (`utils/logging.py`).
- Replace `head: unet` with a standard image-only U-Net (Ronneberger-style
  symmetric encoder/decoder with double 3x3 conv blocks, max-pooling, and
  transpose-convolution upsampling), and update prepare/train/inference/XAI so
  this head ignores `model.layers`, skips DINO feature caching/extraction, and
  no longer requires auxiliary logits. Train/validation dataloaders now also
  pad mixed-size cached image/label/feature tensors per batch so native
  label-grid caches from scenes with different scale factors can still be
  batched safely. The standard `unet` baseline now also uses the AdamW-only
  optimizer path instead of the Muon+AdamW split to avoid unstable high-logit
  divergence under the previous Muon learning-rate defaults
  (`models/unet.py`, `pipeline/train_utils.py`, `pipeline/data_splits.py`,
  `pipeline/phases/prepare.py`, `pipeline/phases/train.py`,
  `pipeline/phases/train_batches.py`, `pipeline/phases/train_xai.py`,
  `pipeline/phases/inference.py`, `utils/data/pipeline.py`,
  `test/test_train_utils_safety.py`, `test/test_prepare_runtime.py`,
  `test/test_data_splits_leakage.py`,
  `README.md`, `docs/ARCHITECTURE.md`).
- Add a `train.plots.paper` profile that emits curated paper-oriented copies of
  epoch metric/XAI figures under `artifacts/plots/*/paper`, adds a compact
  `training_summary.png` cross-epoch plot, and restyles the paper variants with
  contour overlays, error maps, fewer qualitative panels, and calmer
  branch/layer/channel summaries while keeping the existing full diagnostic
  artifacts (`pipeline/plotting.py`, `pipeline/phases/train.py`,
  `pipeline/phases/train_xai.py`, `pipeline/train_config.py`,
  `configs/config_*.yml`, `test/test_plotting.py`,
  `test/test_config_integrity.py`, `test/test_train_distributed_runtime.py`,
  `docs/ARCHITECTURE.md`).
- Allow cached feature directories prepared with a superset of DINO layers to
  be reused by narrower runs (for example cached `[5, 11, 17, 23]` with
  requested `[23]`) by accepting subset layer metadata during prepare and
  selecting only the requested cached feature tensors at dataset load time
  (`utils/data/core.py`, `utils/data/pipeline.py`,
  `pipeline/data_splits.py`, `pipeline/phases/train.py`,
  `test/test_prepare_runtime.py`).
- Ignore backbone model/layer metadata for `prepare` caches when
  `cache_features: false`, so no-feature tile caches are reusable across layer
  config changes and new no-feature cache metadata no longer persists unused
  model/layer fields (`utils/data/core.py`, `test/test_prepare_runtime.py`).
- Harden shared Grad-CAM extraction so training and inference now preserve the
  real failure reason, support payload-style head outputs, and report
  per-scene inference fallback counts instead of repeating opaque
  `Grad-CAM extraction failed.` messages (`pipeline/inference_utils.py`,
  `pipeline/phases/inference.py`, `test/test_inference_outputs.py`).
- Make `unet_nano` accept 1-4 selected DINO layers by keeping the deepest
  available feature as the bottleneck input and dropping missing shallow DINO
  skip concatenations, while preserving the existing 4-layer path; add
  regression coverage for supported and invalid feature-count cases
  (`models/unet_nano.py`, `test/test_train_utils_safety.py`, `README.md`,
  `docs/ARCHITECTURE.md`).
- Expand the inline comments in all shipped YAML profiles so selector-style
  config fields now spell out their accepted values directly in the config
  files (for example head names, normalization/activation modes, AMP options,
  inference merge/layout choices, booleans, and blank-vs-path fields)
  (`configs/config.example.yml`, `configs/config_local.yml`,
  `configs/config_hpc.yml`).
- Fix distributed train/validation split construction so rank 0 now resolves
  the random `max_tiles` sample plus leakage-safe split once and broadcasts the
  exact file lists to all other ranks before `DistributedSampler` is built;
  also add a fail-fast cross-rank dataset/loader length check so mismatched
  per-rank batch counts abort immediately instead of surfacing later as epoch-end
  NCCL `BROADCAST` or train-step `ALLREDUCE` timeouts
  (`pipeline/data_splits.py`, `test/test_data_splits_leakage.py`,
  `docs/ARCHITECTURE.md`).
- Add a second distributed-training hardening pass for last-batch NCCL
  allreduce hangs by exposing `resources.ddp_find_unused_parameters`, adding
  rank-aware batch-stage timing logs plus optional Slurm debug env toggles, and
  escalating local non-finite/batch failures to immediate distributed
  stop-run behavior so ranks do not silently diverge
  (`pipeline/phases/train.py`, `pipeline/phases/train_batches.py`,
  `configs/config_*.yml`, `segmentation.sh`, `README.md`,
  `test/test_train_distributed_runtime.py`, `docs/ARCHITECTURE.md`).
- Harden distributed training against epoch-boundary NCCL timeouts by making
  validation and epoch-level XAI rank-0-only work that is synchronized back to
  the other ranks via explicit DDP summary broadcasts plus an end-of-epoch
  barrier, adding a configurable `resources.dist_timeout_minutes` process-group
  timeout, avoiding the unsupported-attention warning path for rollout maps by
  switching to eager attention before requesting attentions when needed, and
  adding regression coverage for the new distributed helpers
  (`pipeline/phases/train.py`, `pipeline/utils.py`,
  `pipeline/inference_utils.py`, `configs/config_*.yml`,
  `test/test_train_distributed_runtime.py`, `docs/ARCHITECTURE.md`).
- Fix the train-time XAI dashboard GT panel after the native label-grid
  supervision change by rendering the preview RGB on the same label grid as
  GT/pred masks while still computing Grad-CAM and attention maps from the
  full-resolution image tile, plus add a regression test for the preview-grid
  helper (`pipeline/phases/train_xai.py`,
  `test/test_train_utils_safety.py`).
- Switch the default supervision/output policy to the native label grid:
  prepare now caches image tiles on the image grid paired with smaller native
  label-grid masks, training/evaluation resize logits down to label space
  instead of upsampling labels, inference writes scene predictions on the
  label-grid transform when a label raster is configured, and cache metadata
  now records the label-grid mode to avoid reusing older image-grid caches
  (`utils/data/core.py`, `utils/data/pipeline.py`,
  `pipeline/train_utils.py`, `pipeline/phases/train_batches.py`,
  `pipeline/phases/train_xai.py`, `pipeline/phases/inference.py`,
  `test/test_prepare_runtime.py`, `test/test_train_utils_safety.py`,
  `docs/ARCHITECTURE.md`).
- Make `rasterize_labels.sh` default `OMP_NUM_THREADS` to
  `SLURM_CPUS_PER_TASK` instead of `1` so the Slurm rasterization job can use
  the CPU allocation it already requests unless the user explicitly overrides
  the thread count (`rasterize_labels.sh`).
- Replace the placeholder introduction/purpose text in `AGENTS.md` with a
  concise summary of the repo's DINOv3 segmentation pipeline, configs, and
  supporting documentation (`AGENTS.md`).
- Expand `docs/STYLE.MD` with explicit logging guidance, a concrete
  module-level logger example, and clearer wording around config-file-driven
  script execution and docstring/doctest expectations (`docs/STYLE.MD`).
- Extend `docs/STYLE.MD` with performance guidance on choosing efficient
  libraries and preferring vectorized operations for array-heavy work, with a
  concrete `zip()`-loop versus NumPy example (`docs/STYLE.MD`).
- Clarify in `docs/STYLE.MD` that numeric bulk operations should prefer
  `numpy.ndarray` over Python lists so array code stays vectorized and
  efficient (`docs/STYLE.MD`).
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
