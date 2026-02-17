# ARCHITECTURE

## Goal
Provide a config-driven segmentation pipeline with pluggable heads, reproducible runs, and
MLflow-compatible artifacts for research workflows.

## Folder Structure
- `main.py`: Thin CLI entry point for running the pipeline.
- `pipeline/`: Phase runner, hooks, processors, and tracking utilities.
- `models/`: Segmentation heads (U-Net variants, MaskFormer-style head).
- `utils/`: Data preparation, losses, metrics, optimization helpers, logging.
- `config.py`: YAML configuration loader.

## Phase Orchestration
- **Phase base class:** Standardizes enable checks, timing, and error handling.
- **PhaseRunner:** Executes phases in order and coordinates hooks/processors.
- **Hooks:** Lifecycle callbacks (run/phase/epoch/batch/tile) for extensibility.
- **Processors:** Pre/post phase modules for snapshotting and summaries.
- **Stability policy:** `train.stability` controls AMP mode/dtype, fp32 loss, gradient clipping,
  non-finite handling, and checkpoint safety gates.
- **Dataset validation policy:** `dataset.validation` defines finite checks and allowed label
  values for both dataloading and cache verification.
- **Tile intake policy:** `dataset.tile_filter` can keep only tiles containing foreground labels
  during prepare, reducing background-only training samples.

## Tracking & Artifacts
- MLflow-compatible file layout under `mlruns/<experiment_id>/<run_id>/`.
- `artifacts/metrics.jsonl` for lightweight visualization.
- `artifacts/run_summary.json` for run metadata and phase outputs.
- Epoch logging emits explicit validation aliases (mIoU/IoU/F1) plus decomposed
  train/validation loss components for richer MLflow dashboards.
- Epoch XAI logging can emit branch-importance metrics (image-vs-DINO gradient
  sensitivity), per-layer DINO connection importance trends, Lite+ gate
  importance summaries, and epoch-wise DINO channel importance evolution
  artifacts (bar/trend/heatmap + JSON summaries).
- Plot artifacts are grouped per run under `artifacts/plots/metrics`,
  `artifacts/plots/xai`, and `artifacts/plots/inference` to reduce clutter.
- Decoder family includes opt-in lightweight variants (`unet_lite`,
  `unet_lite_plus`, `unet_nano`) so users can trade compute for quality
  without changing pipeline wiring. `unet_nano` keeps the deep path tiny and
  adds RGB priors only in late decoder stages (H/4, H/2).

## Design Principles
- **Modularity:** Small, focused modules with explicit contracts.
- **Documentation:** Docstrings + doctests for public symbols.
- **Minimal diffs:** Avoid structural churn unless necessary.

## Workflow
1. Prepare tiles and features (optional)
   Foreground-aware filtering can be applied here (`dataset.tile_filter`) so only tiles with
   configured target labels are cached.
2. Verify cached tiles (readability + semantic checks) (optional)
3. Train segmentation head with per-epoch validation visualization panels (optional),
   including optional XAI dashboards (DINO attention, Grad-CAM, PCA feature maps,
   top-k feature channels, and channel-importance evolution plots). Training
   augmentation includes geometric transforms and
   optional image-only regularizers (color jitter/cutout/gridmask), which are
   cache-safe by default when precomputed features are used.
4. Run inference (optional)
   Both `input_tif` and `input_dir` modes use the same sliding-window tiled inference
   and merge path to keep behavior consistent and memory-bounded on large inputs.
