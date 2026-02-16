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

## Tracking & Artifacts
- MLflow-compatible file layout under `mlruns/<experiment_id>/<run_id>/`.
- `artifacts/metrics.jsonl` for lightweight visualization.
- `artifacts/run_summary.json` for run metadata and phase outputs.

## Design Principles
- **Modularity:** Small, focused modules with explicit contracts.
- **Documentation:** Docstrings + doctests for public symbols.
- **Minimal diffs:** Avoid structural churn unless necessary.

## Workflow
1. Prepare tiles and features (optional)
2. Verify cached tiles (readability + semantic checks) (optional)
3. Train segmentation head with per-epoch validation visualization panels (optional),
   including optional XAI dashboards (DINO attention, Grad-CAM, top-k feature channels)
4. Run inference (optional)
