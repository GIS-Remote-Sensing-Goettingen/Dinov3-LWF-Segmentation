# DINOv3Seg Experiments

This repository provides a research-grade segmentation pipeline that keeps a frozen **DINOv3** backbone and swaps different decoder heads (classic U-Net, SPM/FAPM-enhanced U-Net, and a MaskFormer-style transformer). The pipeline now runs entirely from a YAML configuration file, supports structured verbosity with timestamps, and records durations for every major phase.

## Quick Start

1. **Copy the example config** and tailor it to your environment:
   ```bash
   cp configs/config.example.yml config.yml
   ```
   Update the paths (raw imagery, labels, cache directory), toggle phases (`prepare`, `verify`, `train`, `inference`), and adjust hyperparameters or decoder selection under the `model` section.

2. **Run the pipeline**:
   ```bash
   python main.py config.yml
   ```
   For multi-GPU training set `resources.distributed: true` and launch with e.g.
   ```bash
   torchrun --standalone --nproc_per_node=4 main.py config.yml
   ```
   On Slurm, `segmentation.sh` now defaults to a single-node 2-GPU `torchrun`
   launch against `configs/config_hpc.yml`; override `GPUS_PER_NODE` or
   `CONFIG_PATH` at submit time when needed.
   Only rank 0 prints logs and runs inference; validation metrics are computed on rank 0 and broadcast to the others. If no argument is provided the script checks the first CLI argument, then `$DINOV3SEG_CONFIG`, and finally searches upward for `configs/config_hpc.yml`.

3. **Observe logs**: The logger honors three verbosity levels (`error`, `info`, `debug`), can print timestamps, and optionally mirrors output to a log file. Configure it via the `logging` block.

## Configuration Reference

The YAML file drives everything. Each section mirrors a phase and shares defaults if a specific value is missing.

The shipped profiles under `configs/` are now the primary reference: every
user-facing key is commented inline, including blank values and override
behavior.

Important cache semantics:
- `dataset.cache_features` and `prepare.cache_features` control whether cached
  `.pt` tiles include precomputed DINO features. They do not disable tile
  caching itself.
- `paths.label_path` is read during `prepare` when tiles are built. Training
  usually reads cached `.pt` tiles from `processed_dir`, not the label TIFF
  directly.
- If you change `label_path`, `dataset.tile_filter`, or tile size settings, use
  a fresh `processed_dir` or delete the old cached tiles before rerunning
  `prepare`.
- `paths.processed_dir` is the shared default cache root; `train.processed_dir`
  and other phase-local `processed_dir` keys override it for that phase only.

Repository layout:
- `configs/` contains the shipped example/local/HPC YAML profiles.
- `docs/` contains architecture notes, changelog, model notes, and style guidance.
- `README.md` stays at the repository root as the entry point.

```yaml
resources:
  seed: 1337
  omp_threads: 4
  matmul_precision: high
  distributed: false        # set true when launching with torchrun
  dist_backend: nccl        # backend for DDP

logging:
  level: info
  timestamps: true
  file: logs/run.log

paths:
  raw_images_dir: /path/to/imagery
  label_path: /path/to/labels.tif
  processed_dir: /path/to/cache

dataset:
  cache_features: false
  tile_filter:
    enabled: true
    mode: foreground_any
    foreground_labels: [1]
  augmentations:
    enable: true
    hflip: true
    vflip: true
    rotate90: true
    allow_feature_mismatch: false
    cutout:
      enable: true
      prob: 0.25
      min_frac: 0.08
      max_frac: 0.22
      num_holes: 1
      fill: 0.0
    gridmask:
      enable: true
      prob: 0.20
      d_min: 48
      d_max: 128
      ratio: 0.5
      rotate: true
      fill: 0.0
    color_jitter:
      enable: true
      prob: 0.35
      brightness: 0.25
      contrast: 0.25
      saturation: 0.12
      hue: 0.04
  splits:
    train_list: splits/train.txt
    val_list: splits/val.txt
  validation:
    enabled: true
    allowed_labels: [0, 1]
    ignore_index: 255
    out_of_range_policy: map_to_ignore
    require_finite_features: true
    require_finite_images: true

# `dataset.tile_filter` is applied while preparing new cached tiles.
# If a cache already exists, point to a fresh processed_dir (or clean old tiles) to re-filter it.

model:
  backbone: facebook/dinov3-vitl16-pretrain-sat493m
  layers: [5, 11, 17, 23]
  head: unet_v2          # dino_dense_probe | dino_segdino_light | unet | unet_v2 | unet_lite | unet_lite_plus | unet_nano | unet_nano_fapm | unet_topo_fusion | maskformer
  num_classes: 2
  dino_channels: 1024
  dense_probe:
    norm_type: batchnorm # batchnorm | syncbn | groupnorm | none
    groupnorm_groups: 32
  segdino_light:
    proj_dim: 128
    activation: gelu      # gelu | relu
    dropout: 0.0
    strict_layers: true
  fusion:
    enable: true
    hidden: 64
    layer_hidden: 128
    max_layers: 6
    save_maps: false
  lora:
    enable: true
    rank: 8
    alpha: 16.0
    dropout: 0.0
    freeze_base: true
  boundary_gate:
    enable: true
    scale: 0.1
    clamp: true

prepare:
  enable: true
  tile_size: 512
  device: cuda

verify:
  enable: true
  workers: 8

train:
  enable: true
  processed_dir: /path/to/cache
  weights_dir: weights
  batch_size: 4
  epochs: 30
  muon_lr: 0.02
  muon_wd:        # optional; defaults to adamw_wd when unset
  muon_update_scale: 0.2
  muon_adjust_lr_for_shape: true
  adamw_lr: 0.001
  adamw_wd: 0.01
  momentum: 0.95
  patience: 10
  val_fraction: 0.2
  num_workers: 4
  grad_accum_steps: 1
  compile: false
  ema_decay: 0.0
  plots:
    epoch:
      enable: true
      dir: output/plot  # fallback only when MLflow logging is disabled
      cmap: tab20
      pairs: 4
      seed_offset: 1000
      metric_class_index: 1
    xai:
      enable: true
      class_index: 1
      topk_channels: 5
      cam_layer_mode: last_requested_layer
      render_attn_rollout: true
      pca:
        enable: true
        layer_mode: same_as_cam
      branch_importance:
        enable: true
        class_index: 1
        max_samples: 4
      channel_tracking:
        enable: true
        max_samples: 64
        top_k_per_sample: 5
        top_n_stable: 10
        min_presence: 0.05
        save_json: true
      module:
        enable: true
        every_n_epochs: 5
        max_samples: 8
        save_maps: true
        boundary_band_px: 3
        gate_threshold: 0.5
        entropy_eps: 1.0e-8
        strict: false
        enable_lora_ratio: true
        enable_topology_panels: true
  loss:
    main:
      ce_weight: 1.0
      dice_weight: 1.0
      aux_weight: 0.4
      label_smoothing: 0.1
    # Focal is weight-driven: 0.0 disables it while CE stays active.
    focal:
      weight: 0.0
      gamma: 2.0
      alpha:
    boundary:
      weight: 0.1
      kernel_size: 3
  topology:
    skeleton_weight: 0.0
    weight: 0.0
    class_index: 1
    iters: 10
    on_aux: true
    downsample: 1
  stability:
    amp:
      enabled: auto      # auto | on | off
      dtype: bf16        # bf16 | fp16
    loss_fp32: true
    grad_clip_norm: 1.0
    max_abs_logit_warn: 80.0
    nonfinite:
      action: stop_run   # stop_run | stop_epoch | skip_batch
      max_consecutive_batches: 2
      max_total_batches_per_epoch: 5
      save_bad_batch_sample: true
    check_params_every_steps: 50

inference:
  enable: false
  input_tif: /path/to/scene.tif
  output_tif: test/output_prediction.tif
  checkpoint: weights/unet_v2_best.pth
  tile_size: 512
  overlap: 0.25
  merge:
    mode: center_weighted
  tta:
    horizontal_flip: true
    vertical_flip: false
  explain:
    enable: true
    output_dir: plots  # fallback only when MLflow logging is disabled
    class_index: 1
    dashboard_layout: "2x2"
    pred_overlay_color: [120, 190, 255]
    pred_overlay_alpha: 0.28
    tile_debug_enable: false
  vector:
    enable: false
    target_epsg: 4326
    append: true
    foreground_class: 1
```

Set `enable: true` for any section you want to run. The `paths` block provides base directories shared across phases, while individual sections can override them (e.g., use a different `processed_dir` for training vs. verification).
Model options for topology-fusion heads are grouped under `model.fusion`, `model.lora`, and `model.boundary_gate`; legacy flat keys are still supported for backward compatibility.

Inference input selection:
- Set exactly one source: `inference.input_tif` or `inference.input_dir`.
- `input_dir` now runs each file through the same sliding-window tiled inference + merge path used by `input_tif`, then writes outputs to `output_dir`.
- Inference explainability now writes one scene-level figure per input image by default: RGB, light-blue prediction overlay, Grad-CAM, and class probability.
- `inference.vector` appends all scene-level foreground predictions into one cumulative shapefile in `EPSG:4326` under the run artifact tree (`artifacts/vectors/inference/predictions_4326.shp`).

## Logging & Timing

- The custom `VerbosityLogger` prints `[LEVEL] message` lines with optional timestamps and can also tee logs to disk (`logging.file`).
- Every phase (`prepare`, `verify`, `train`, `inference`) runs inside a `TimedBlock`, so you see start/finish messages and durations.
- Inner loops log progress periodically (e.g., every 10 training batches or every 50 inference tiles) when verbosity permits.

## Distributed Training

- Set `resources.distributed: true` in the config and launch via `torchrun --standalone --nproc_per_node=<gpus> main.py config.yml`.
- `segmentation.sh` requests 2 GPUs by default and launches `torchrun` with the
  visible GPU count unless `GPUS_PER_NODE` is overridden.
- Training uses `DistributedDataParallel` with `DistributedSampler`; rank 0 handles logging, validation loops, checkpointing, and inference while other ranks stay silent and focus on SGD.
- Validation metrics (loss, mIoU) and early-stopping signals are broadcast to every rank so they can exit cleanly at the same epoch. Inference automatically runs only on rank 0 to avoid duplicate outputs.

## Decoder Registry

The `model.head` key selects one of the decoders registered under `models/`:

| Head        | File             | Highlights                                                        |
|-------------|------------------|-------------------------------------------------------------------|
| `dino_dense_probe` | `models/dino_dense_probe.py` | Dense linear-probe baseline on last-layer DINO tokens (`norm -> 1x1 conv -> upsample`). |
| `dino_segdino_light` | `models/dino_segdino_light.py` | SegDINO-style lightweight multi-layer fusion head (`per-layer 1x1 -> align -> concat -> fuse`). |
| `unet`      | `models/unet.py` | Baseline DinoUNet with stacked UpBlocks and raw-image skip.       |
| `unet_v2`   | `models/unet_v2.py` | Adds Spatial Prior Module + Fidelity-Aware projections + deep supervision. |
| `unet_lite` | `models/UnetLite.py` | Lightweight DinoUNet variant with reduced channels for faster training/inference. |
| `unet_lite_plus` | `models/unet_lite_plus.py` | Opt-in Lite+ variant using interpolate+conv upsampling, GN+GELU residual blocks, and lightweight gated H/4 fusion. |
| `unet_nano` | `models/unet_nano.py` | Aggressively compact decoder with GroupNorm, GELU, Dropout2d, and late RGB fusion at H/4 and H/2. |
| `unet_nano_fapm` | `models/unet_nano_fapm.py` | Nano variant with low-rank split-and-modulate projections (NanoFAPM) plus a lightweight boundary branch fused into final logits. |
| `unet_topo_fusion` | `models/unet_topo_fusion.py` | Topology-aware variant with learned DINO layer fusion, LoRA-style projection adapters, boundary-feature gating, and a skeleton branch for soft-clDice supervision. |
| `maskformer`| `models/maskformer.py` | Pixel decoder fused with transformer mask head (MaskFormer style).       |

Adding a new decoder only requires implementing `SegmentationHead`, registering it in `models/__init__.py`, and referencing it via `model.head`.

## Utilities

- `utils/data.py` handles tiling, label alignment, feature extraction, cache verification, and the `PrecomputedDataset`. It supports an optional foreground-label tile filter during preparation (`dataset.tile_filter`), applies optional train-time augmentations (flips/rotations + color jitter/cutout/gridmask), validates finiteness/label ranges, and supports region-based splits. Image-only augmentations are skipped by default when cached features are enabled unless `dataset.augmentations.allow_feature_mismatch` is set to `true`.
- `utils/losses.py` implements the combined segmentation losses (CE or focal + Dice + optional boundary BCE + optional skeleton BCE + optional soft-clDice topology term) for main/aux outputs.
- `utils/metrics.py` accumulates per-class IoU/Dice; we early-stop on validation mIoU instead of loss.
- `utils/optim.py` contains the Muon optimizer (matrix-aware momentum with orthogonalization, decoupled Muon weight decay, and paper-style shape-aware update scaling), AdamW handling, and a configurable EarlyStopping helper that works for min/max metrics.
- `utils/logging.py` exposes the verbosity logger (`stdout` + optional file) and `TimedBlock` context manager.
- `config.py` reads the YAML file, honors the `$DINOV3SEG_CONFIG` override, and searches upward from the working directory if no path is provided.
- `scripts/export_metrics_csv.py` converts `artifacts/metrics.jsonl` into a flat CSV for thesis tables/plots.

- **Training extras:** gradient accumulation, optional `torch.compile`, Muon+AdamW with OneCycleLR, model EMA, configurable CE/focal + Dice (+ optional boundary BCE/skeleton BCE/soft-clDice topology) losses, fp32-loss mixed precision, gradient clipping, parameter finite checks, Muon routing that keeps embeddings/bias-like params on AdamW, per-epoch validation grids (4 tile pairs by default) with per-tile IoU/F1, and optional epoch-level XAI panels (`epoch_XXXX_xai.png`) with DINO CLS/rollout focus, Grad-CAM overlays, per-sample DINO PCA (PC1-3), top-k influential DINO channel maps, gradient-based branch importance (`image` vs `dino`), per-layer DINO connection importance trends, Lite+ gate importance, per-epoch branch-importance trendlines (`branch_importance_trends.png`), per-epoch DINO-layer trendlines (`dino_layer_importance_trends.png`), per-epoch channel-importance artifacts (bar chart + trends + heatmap + JSON summaries), and module-specific diagnostics under `plots/xai/module/` (layer-fusion argmax/entropy + region bars, gate boundary ROC panels, boundary error-reduction maps, LoRA ratio maps/histograms, and topology skeleton overlays with trend plots).
- **Inference extras:** sliding-window streaming directly from disk, configurable overlap with probability blending, AMP, and optional flip-based test-time augmentation.
- **MLflow traces:** epoch metrics include explicit validation aliases (`train.val_miou`, `train.val_iou`, `train.val_f1`, `train.val_mdice`), full loss decomposition (`train.loss_*` + `train.val_loss_*`), split learning-rate traces (`train.lr_muon`, `train.lr_adamw`), branch + DINO-layer importance means, and model size settings (`model_total_params`, `model_trainable_params`, `model_non_trainable_params`) as params/tags. Artifacts are grouped under `artifacts/plots/{metrics,xai,inference}` per run.
- With MLflow enabled, training and inference plots are written directly under the active run artifact tree (`artifacts/plots/...`) to avoid mixed local output folders.

Branch-importance interpretation:
- Higher `image` importance means predictions are more sensitive to RGB content.
- Higher `dino` importance means predictions rely more on DINO feature tensors.
- Higher per-layer DINO importance means that specific configured DINO skip
  connection contributes more strongly to the final segmentation output.
- For `unet_lite_plus`, higher gate importance means stronger pass-through of the H/4 RGB prior skip.
- Channel-importance plots show which DINO channels dominate on average each epoch; rising concentration in a few channels can indicate specialization, while diffuse usage suggests broader feature reliance.

## Testing

Every function ships with doctests to keep behavior well documented. Run them with:

```bash
python -m doctest main.py
PYTHONPATH=. python - <<'PY'
import doctest, importlib
for mod in [
    "models.base",
    "models.dino_dense_probe",
    "models.dino_segdino_light",
    "models.unet",
    "models.unet_v2",
    "models.unet_topo_fusion",
    "models.maskformer",
    "models.__init__",
    "utils.data",
    "utils.optim",
    "utils.logging",
    "utils.losses",
    "utils.metrics",
    "config",
]:
    doctest.testmod(importlib.import_module(mod))
PY
```

## Dependencies

- PyTorch (CUDA build recommended)
- `transformers`
- Geospatial stack: `rasterio`, `tifffile`, `shapely`, `fiona`
- Misc: `numpy`, `tqdm`, `PyYAML`

Install via:

```bash
pip install torch torchvision transformers rasterio tifffile shapely fiona tqdm pyyaml
```

## Notes

- Large imagery and label rasters should share a CRS; the tiling pipeline reprojects labels when needed.
- Cache verification deletes unreadable or semantically invalid `.pt` files, so rerun `prepare` if the dataset was partially generated.
- Inference now streams tiles from disk, supports overlapping windows with probability blending, runs under AMP, and can average flip-based TTA predictions.
- Cached tiles are used for training while inference recomputes DINO features on the fly.

With the YAML-driven approach, you can version-control experiment configs, schedule recurring training jobs, and keep logs consistent across runs.
