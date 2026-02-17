# DINOv3Seg Experiments

This repository provides a research-grade segmentation pipeline that keeps a frozen **DINOv3** backbone and swaps different decoder heads (classic U-Net, SPM/FAPM-enhanced U-Net, and a MaskFormer-style transformer). The pipeline now runs entirely from a YAML configuration file, supports structured verbosity with timestamps, and records durations for every major phase.

## Quick Start

1. **Copy the example config** and tailor it to your environment:
   ```bash
   cp config.example.yml config.yml
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
   Only rank 0 prints logs and runs inference; validation metrics are computed on rank 0 and broadcast to the others. If no argument is provided the script checks the first CLI argument, then `$DINOV3SEG_CONFIG`, and finally searches upward for `config.yml`.

3. **Observe logs**: The logger honors three verbosity levels (`error`, `info`, `debug`), can print timestamps, and optionally mirrors output to a log file. Configure it via the `logging` block.

## Configuration Reference

The YAML file drives everything. Each section mirrors a phase and shares defaults if a specific value is missing.

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
  head: unet_v2          # unet | unet_v2 | unet_lite | unet_lite_plus | maskformer
  num_classes: 2
  dino_channels: 1024

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
  adamw_lr: 0.001
  adamw_wd: 0.01
  momentum: 0.95
  patience: 10
  val_fraction: 0.2
  num_workers: 4
  grad_accum_steps: 1
  compile: false
  ema_decay: 0.0
  epoch_plot: true
  epoch_plot_dir: output/plot  # fallback only when MLflow logging is disabled
  epoch_plot_cmap: tab20
  epoch_plot_pairs: 4
  epoch_plot_seed_offset: 1000
  epoch_plot_metric_class_index: 1
  epoch_plot_xai_enable: true
  epoch_plot_xai_class_index: 1
  epoch_plot_xai_topk_channels: 5
  epoch_plot_xai_cam_layer_mode: last_requested_layer
  epoch_plot_xai_render_attn_rollout: true
  epoch_plot_xai_pca_enable: true
  epoch_plot_xai_pca_layer_mode: same_as_cam
  epoch_plot_xai_branch_importance_enable: true
  epoch_plot_xai_branch_importance_class_index: 1
  epoch_plot_xai_branch_importance_max_samples: 4
  epoch_plot_xai_channel_tracking_enable: true
  epoch_plot_xai_channel_tracking_max_samples: 64
  epoch_plot_xai_channel_top_k_per_sample: 5
  epoch_plot_xai_channel_top_n_stable: 10
  epoch_plot_xai_channel_min_presence: 0.05
  epoch_plot_xai_channel_save_json: true
  loss:
    ce_weight: 1.0
    dice_weight: 1.0
    aux_weight: 0.4
    label_smoothing: 0.1
    use_focal: false
    focal_gamma: 2.0
    focal_alpha:
    boundary_weight: 0.1
    boundary_kernel_size: 3
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
  tta:
    horizontal_flip: true
    vertical_flip: false
  explain:
    enable: true
    output_dir: plots  # fallback only when MLflow logging is disabled
    class_index: 1
    dashboard_layout: "4x3"
    pca_enable: true
    pca_layer_mode: last_requested_layer
```

Set `enable: true` for any section you want to run. The `paths` block provides base directories shared across phases, while individual sections can override them (e.g., use a different `processed_dir` for training vs. verification).

Inference input selection:
- Set exactly one source: `inference.input_tif` or `inference.input_dir`.
- `input_dir` now runs each file through the same sliding-window tiled inference + merge path used by `input_tif`, then writes outputs to `output_dir`.

## Logging & Timing

- The custom `VerbosityLogger` prints `[LEVEL] message` lines with optional timestamps and can also tee logs to disk (`logging.file`).
- Every phase (`prepare`, `verify`, `train`, `inference`) runs inside a `TimedBlock`, so you see start/finish messages and durations.
- Inner loops log progress periodically (e.g., every 10 training batches or every 50 inference tiles) when verbosity permits.

## Distributed Training

- Set `resources.distributed: true` in the config and launch via `torchrun --standalone --nproc_per_node=<gpus> main.py config.yml`.
- Training uses `DistributedDataParallel` with `DistributedSampler`; rank 0 handles logging, validation loops, checkpointing, and inference while other ranks stay silent and focus on SGD.
- Validation metrics (loss, mIoU) and early-stopping signals are broadcast to every rank so they can exit cleanly at the same epoch. Inference automatically runs only on rank 0 to avoid duplicate outputs.

## Decoder Registry

The `model.head` key selects one of the decoders registered under `models/`:

| Head        | File             | Highlights                                                        |
|-------------|------------------|-------------------------------------------------------------------|
| `unet`      | `models/unet.py` | Baseline DinoUNet with stacked UpBlocks and raw-image skip.       |
| `unet_v2`   | `models/unet_v2.py` | Adds Spatial Prior Module + Fidelity-Aware projections + deep supervision. |
| `unet_lite` | `models/UnetLite.py` | Lightweight DinoUNet variant with reduced channels for faster training/inference. |
| `unet_lite_plus` | `models/unet_lite_plus.py` | Opt-in Lite+ variant using interpolate+conv upsampling, GN+GELU residual blocks, and lightweight gated H/4 fusion. |
| `unet_nano` | `models/unet_nano.py` | Aggressively compact decoder with GroupNorm, GELU, Dropout2d, and late RGB fusion at H/4 and H/2. |
| `unet_nano_fapm` | `models/unet_nano_fapm.py` | Nano variant with low-rank split-and-modulate projections (NanoFAPM) plus a lightweight boundary branch fused into final logits. |
| `maskformer`| `models/maskformer.py` | Pixel decoder fused with transformer mask head (MaskFormer style).       |

Adding a new decoder only requires implementing `SegmentationHead`, registering it in `models/__init__.py`, and referencing it via `model.head`.

## Utilities

- `utils/data.py` handles tiling, label alignment, feature extraction, cache verification, and the `PrecomputedDataset`. It supports an optional foreground-label tile filter during preparation (`dataset.tile_filter`), applies optional train-time augmentations (flips/rotations + color jitter/cutout/gridmask), validates finiteness/label ranges, and supports region-based splits. Image-only augmentations are skipped by default when cached features are enabled unless `dataset.augmentations.allow_feature_mismatch` is set to `true`.
- `utils/losses.py` implements the combined segmentation losses (CE or focal + Dice + optional boundary BCE) for main/aux outputs.
- `utils/metrics.py` accumulates per-class IoU/Dice; we early-stop on validation mIoU instead of loss.
- `utils/optim.py` contains the Muon optimizer (matrix-aware momentum with orthogonalization), AdamW handling, and a configurable EarlyStopping helper that works for min/max metrics.
- `utils/logging.py` exposes the verbosity logger (`stdout` + optional file) and `TimedBlock` context manager.
- `config.py` reads the YAML file, honors the `$DINOV3SEG_CONFIG` override, and searches upward from the working directory if no path is provided.

- **Training extras:** gradient accumulation, optional `torch.compile`, Muon+AdamW with OneCycleLR, model EMA, configurable CE/focal + Dice (+ optional boundary BCE) loss, fp32-loss mixed precision, gradient clipping, parameter finite checks, per-epoch validation grids (4 tile pairs by default) with per-tile IoU/F1, and optional epoch-level XAI panels (`epoch_XXXX_xai.png`) with DINO CLS/rollout focus, Grad-CAM overlays, per-sample DINO PCA (PC1-3), top-k influential DINO channel maps, gradient-based branch importance (`image` vs `dino`), per-layer DINO connection importance trends, Lite+ gate importance, per-epoch branch-importance trendlines (`branch_importance_trends.png`), per-epoch DINO-layer trendlines (`dino_layer_importance_trends.png`), and per-epoch channel-importance artifacts (bar chart + trends + heatmap + JSON summaries).
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
    "models.unet",
    "models.unet_v2",
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
- Geospatial stack: `rasterio`, `tifffile`, `shapely`
- Misc: `numpy`, `tqdm`, `PyYAML`

Install via:

```bash
pip install torch torchvision transformers rasterio tifffile shapely tqdm pyyaml
```

## Notes

- Large imagery and label rasters should share a CRS; the tiling pipeline reprojects labels when needed.
- Cache verification deletes unreadable or semantically invalid `.pt` files, so rerun `prepare` if the dataset was partially generated.
- Inference now streams tiles from disk, supports overlapping windows with probability blending, runs under AMP, and can average flip-based TTA predictions.
- Cached tiles are used for training while inference recomputes DINO features on the fly.

With the YAML-driven approach, you can version-control experiment configs, schedule recurring training jobs, and keep logs consistent across runs.
