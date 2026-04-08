"""Inference phase implementation."""

from __future__ import annotations

import glob
import os
import tempfile
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.windows import Window
from transformers import AutoImageProcessor, AutoModel

from models import build_head
from utils import TimedBlock, extract_multiscale_features
from utils.data.core import (
    build_tile_grid_layout,
    coverage_window_positions,
    read_label_window_for_image_bounds,
)

from ..constants import DEFAULT_DEVICE
from ..context import InferenceError, PhaseOutcome, RunContext
from ..data_splits import _read_name_list
from ..inference_checkpoint import (
    extract_checkpoint_state_dict,
    resolve_inference_checkpoint,
    validate_checkpoint_compatibility,
)
from ..inference_utils import (
    append_prediction_shapefile,
    backup_prediction_raster,
    build_blend_weight_mask,
    build_cumulative_raster_backup_path,
    build_dashboard,
    build_tta_transforms,
    compute_gradcam_map,
    compute_xai_maps,
    ensure_cumulative_prediction_raster,
    extract_prediction_features,
    hold_prediction_raster_lock,
    normalize_map,
    overlay_binary_mask,
    overlay_heatmap,
    upsample_map,
    write_prediction_to_cumulative_raster,
)
from ..phase_runner import Phase
from ..plotting import resolve_cam_layer
from ..train_utils import (
    head_uses_backbone_features,
    normalize_forward_output,
    resolve_model_patch_size,
)
from ..utils import get_model_config, resolve_path


def _read_input_manifest_entries(path: Path) -> list[str]:
    """Read one directory-inference manifest into ordered non-comment entries.

    Text manifests keep their historical behavior of ignoring blank/comment
    lines, while YAML/JSON manifests reuse the nested list flattening from the
    split-loader helper.

    Args:
        path (Path): Manifest path.

    Returns:
        list[str]: Ordered manifest entries.
    """

    if path.suffix.lower() in {".yml", ".yaml", ".json"}:
        return [entry for entry in _read_name_list(str(path)) if str(entry).strip()]
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _resolve_directory_input_paths(
    *,
    manifest_path: Path,
    input_dir: str | None,
    glob_pattern: str,
) -> list[str]:
    """Resolve one directory-mode manifest into concrete GeoTIFF paths.

    Entries may be literal absolute/relative file paths or source-scene stems
    such as ``dop20_592000_5982000_1km_20cm`` that should be matched against
    files inside ``input_dir``.

    Args:
        manifest_path (Path): Manifest path on disk.
        input_dir (str | None): Optional source-image directory.
        glob_pattern (str): File-discovery glob used when matching scene stems.

    Returns:
        list[str]: Ordered, deduplicated input raster paths.

    Raises:
        InferenceError: If one or more manifest entries cannot be resolved.
    """

    entries = _read_input_manifest_entries(manifest_path)
    if not entries:
        return []

    input_root = Path(input_dir).expanduser() if input_dir else None
    available_by_stem: dict[str, list[str]] = {}
    available_by_name: dict[str, list[str]] = {}
    if input_root is not None:
        for candidate in sorted(glob.glob(str(input_root / glob_pattern))):
            resolved = str(Path(candidate).resolve())
            available_by_stem.setdefault(Path(candidate).stem, []).append(resolved)
            available_by_name.setdefault(Path(candidate).name, []).append(resolved)

    resolved_paths: list[str] = []
    seen: set[str] = set()
    unresolved: list[str] = []
    ambiguous: list[str] = []
    for entry in entries:
        raw_entry = str(entry).strip()
        if not raw_entry:
            continue
        direct_candidate = Path(raw_entry).expanduser()
        manifest_relative = (manifest_path.parent / raw_entry).resolve()
        input_relative = (
            (input_root / raw_entry).resolve() if input_root is not None else None
        )
        matches: list[str] = []
        if direct_candidate.is_absolute():
            if direct_candidate.exists():
                matches = [str(direct_candidate.resolve())]
        elif manifest_relative.exists():
            matches = [str(manifest_relative)]
        elif input_relative is not None and input_relative.exists():
            matches = [str(input_relative)]
        else:
            entry_name = Path(raw_entry).name
            entry_stem = Path(raw_entry).stem if Path(raw_entry).suffix else entry_name
            matches = list(available_by_name.get(entry_name, []))
            if not matches:
                matches = list(available_by_stem.get(entry_stem, []))
        if len(matches) > 1:
            ambiguous.append(raw_entry)
            continue
        if not matches:
            unresolved.append(raw_entry)
            continue
        resolved = matches[0]
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved_paths.append(resolved)
    if ambiguous or unresolved:
        problems: list[str] = []
        if unresolved:
            sample = ", ".join(unresolved[:3])
            problems.append(f"unresolved entries: {sample}")
        if ambiguous:
            sample = ", ".join(ambiguous[:3])
            problems.append(f"ambiguous entries: {sample}")
        raise InferenceError(
            "Failed to resolve inference.input_paths_file %s (%s)."
            % (manifest_path, "; ".join(problems))
        )
    return resolved_paths


class InferencePhase(Phase):
    """Phase for sliding-window inference.

    Examples:
        >>> InferencePhase.name
        'inference'
    """

    name = "inference"
    config_key = "inference"

    def is_enabled(self, context: RunContext) -> bool:
        """Return True when inference should execute on this rank.

        Args:
            context (RunContext): Active run context.

        Returns:
            bool: True when inference should run.
        """

        if not context.dist_ctx.is_main:
            return False
        infer_cfg = context.config.get("inference", context.config.get("infer", {}))
        return bool(infer_cfg and infer_cfg.get("enable", False))

    def execute(self, context: RunContext) -> PhaseOutcome:
        """Run sliding-window inference over a large raster.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts from the phase.

        Raises:
            InferenceError: If inference fails unexpectedly.
        """

        try:
            return self._infer(context)
        except Exception as exc:
            raise InferenceError(str(exc)) from exc

    def _infer(self, context: RunContext) -> PhaseOutcome:
        """Internal inference implementation.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts from the phase.
        """

        infer_cfg = context.config.get("inference", context.config.get("infer", {}))
        paths_cfg = context.config.get("paths", {})
        model_cfg = get_model_config(context.config)
        device = torch.device(infer_cfg.get("device", DEFAULT_DEVICE))
        uses_backbone_features = head_uses_backbone_features(model_cfg["head"])
        processor = None
        backbone = None
        if uses_backbone_features:
            processor = AutoImageProcessor.from_pretrained(model_cfg["backbone"])
            backbone = (
                AutoModel.from_pretrained(model_cfg["backbone"]).eval().to(device)
            )
        head = build_head(
            model_cfg["head"],
            num_classes=model_cfg["num_classes"],
            dino_channels=model_cfg["dino_channels"],
            model_cfg=context.config.get("model", {}),
        ).to(device)
        checkpoint, checkpoint_source = resolve_inference_checkpoint(context, infer_cfg)
        context.logger.info(
            "Loading inference checkpoint %s (source=%s, head=%s, num_classes=%s, strict=true)"
            % (
                checkpoint,
                checkpoint_source,
                model_cfg["head"],
                model_cfg["num_classes"],
            )
        )
        loaded_checkpoint = torch.load(checkpoint, map_location=device)
        state_dict = extract_checkpoint_state_dict(loaded_checkpoint)
        validate_checkpoint_compatibility(head, state_dict)
        head.load_state_dict(state_dict, strict=True)
        head.eval()
        input_dir = infer_cfg.get("input_dir")
        input_tif = infer_cfg.get("input_tif")
        output_dir = infer_cfg.get("output_dir")
        output_tif = infer_cfg.get("output_tif")
        tile_size = infer_cfg.get("tile_size", 512)
        ps = resolve_model_patch_size(model_cfg["backbone"], model_cfg["head"])
        overlap_cfg = infer_cfg.get("overlap", 0.0)
        overlap_px = (
            int(tile_size * overlap_cfg) if overlap_cfg < 1 else int(overlap_cfg)
        )
        stride = max(1, tile_size - overlap_px)
        tta_transforms = build_tta_transforms(infer_cfg.get("tta", {}))
        autocast = torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()
        explain_cfg = infer_cfg.get("explain", {})
        explain_enabled = bool(explain_cfg.get("enable", False))
        plots_dir = explain_cfg.get("output_dir")
        if explain_enabled and context.mlflow_logger is not None:
            plots_dir = str(context.mlflow_logger.artifacts_dir / "plots" / "inference")
        class_index = int(explain_cfg.get("class_index", 1))
        explain_cam_layer = resolve_cam_layer(
            [int(layer_id) for layer_id in model_cfg["layers"]],
            "last_requested_layer",
        )
        layout = explain_cfg.get("dashboard_layout", "2x2")
        plot_every_n = explain_cfg.get("plot_every_n")
        tile_debug_enable = bool(explain_cfg.get("tile_debug_enable", False))
        overlay_color = tuple(
            int(value)
            for value in explain_cfg.get("pred_overlay_color", [120, 190, 255])
        )
        overlay_alpha = float(explain_cfg.get("pred_overlay_alpha", 0.28))
        merge_cfg = infer_cfg.get("merge", {})
        merge_mode = str(merge_cfg.get("mode", "center_weighted"))
        vector_cfg = infer_cfg.get("vector", {})
        vector_enabled = bool(vector_cfg.get("enable", False))
        vector_target_epsg = int(vector_cfg.get("target_epsg", 4326))
        vector_append = bool(vector_cfg.get("append", True))
        foreground_class = int(vector_cfg.get("foreground_class", class_index))
        glob_pattern = infer_cfg.get("glob", "*.tif")
        input_paths_file = str(infer_cfg.get("input_paths_file", "") or "")
        label_path = resolve_path(context.config, infer_cfg, "label_path", "")
        if not label_path and isinstance(paths_cfg, dict):
            label_path = str(paths_cfg.get("label_path", "") or "")

        def _infer_one_tif(
            input_tif_path: str,
            output_tif_path: str | None,
            plot_prefix: str,
            vector_output_path: str | None = None,
            cumulative_output_path: str | None = None,
        ) -> dict[str, float]:
            """Run tiled inference for a single input raster.

            Args:
                input_tif_path (str): Source raster path.
                output_tif_path (str | None): Optional destination prediction path.
                plot_prefix (str): Prefix for explainability dashboard filenames.
                vector_output_path (str | None): Shared shapefile output path.
                cumulative_output_path (str | None): Shared cumulative raster path.

            Returns:
                dict[str, float]: Per-file metrics with `tiles_total`.
            """

            label_grid_enabled = bool(label_path and os.path.exists(label_path))
            strict_label_grid = cumulative_output_path is not None
            with rasterio.open(input_tif_path) as src:
                profile = src.profile.copy()
                height, width = src.height, src.width
                channels = src.count
                source_transform = src.transform
                source_crs = src.crs
                scene_rgb = None
                output_transform = source_transform
                output_crs = source_crs
                output_height = height
                output_width = width
                tile_size_eff = tile_size
                stride_eff = stride
                if label_grid_enabled:
                    try:
                        if strict_label_grid:
                            with rasterio.open(label_path) as template_src:
                                output_crs = template_src.crs
                                template_bounds = src.bounds
                                if source_crs != output_crs:
                                    template_bounds = rasterio.warp.transform_bounds(
                                        source_crs,
                                        output_crs,
                                        *src.bounds,
                                        densify_pts=21,
                                    )
                                output_window = (
                                    rasterio.windows.from_bounds(
                                        *template_bounds,
                                        transform=template_src.transform,
                                    )
                                    .round_offsets()
                                    .round_lengths()
                                )
                                output_window = Window(
                                    col_off=int(output_window.col_off),
                                    row_off=int(output_window.row_off),
                                    width=max(1, int(output_window.width)),
                                    height=max(1, int(output_window.height)),
                                )
                                output_transform = rasterio.windows.transform(
                                    output_window,
                                    template_src.transform,
                                )
                                output_height = int(output_window.height)
                                output_width = int(output_window.width)
                        else:
                            _, label_meta = read_label_window_for_image_bounds(
                                input_tif_path,
                                label_path,
                            )
                            output_transform = label_meta["transform"]
                            output_crs = label_meta["crs"]
                            output_height = int(label_meta["height"])
                            output_width = int(label_meta["width"])
                        label_layout = build_tile_grid_layout(
                            input_tif_path,
                            label_path,
                            requested_tile_size=tile_size,
                            patch_size=ps,
                        )
                        tile_size_eff = label_layout.image_tile_size
                        overlap_px_eff = (
                            int(tile_size_eff * overlap_cfg)
                            if overlap_cfg < 1
                            else int(overlap_cfg)
                        )
                        stride_eff = max(1, tile_size_eff - overlap_px_eff)
                        if explain_enabled:
                            scene_rgb = np.transpose(
                                src.read(
                                    [1, 2, 3],
                                    out_shape=(3, output_height, output_width),
                                    resampling=Resampling.bilinear,
                                ),
                                (1, 2, 0),
                            )
                    except Exception as exc:
                        if strict_label_grid:
                            raise InferenceError(
                                "Directory inference requires successful label-grid "
                                "alignment for every scene; failed for %s: %s"
                                % (os.path.basename(input_tif_path), exc)
                            ) from exc
                        label_grid_enabled = False
                        context.logger.warning(
                            "Falling back to image-grid inference output for %s: %s"
                            % (os.path.basename(input_tif_path), exc)
                        )
                if explain_enabled and scene_rgb is None:
                    scene_rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
            if channels != 3:
                raise InferenceError(
                    f"Expected 3-band imagery: {os.path.basename(input_tif_path)}"
                )
            gradcam_tiles_attempted = 0
            gradcam_tiles_succeeded = 0
            gradcam_tiles_failed = 0
            gradcam_first_failure = None
            weight_cache: dict[tuple[int, int], np.ndarray] = {}
            y_positions = coverage_window_positions(height, tile_size_eff, stride_eff)
            x_positions = coverage_window_positions(width, tile_size_eff, stride_eff)
            total_tiles = len(y_positions) * len(x_positions)
            scene_window_size = 1024
            context.logger.info(
                "Running inference on %s tiles with stride %s and effective tile "
                "size %s." % (total_tiles, stride_eff, tile_size_eff)
            )
            local_plot_every_n = plot_every_n
            if explain_enabled:
                assert plots_dir is not None
                os.makedirs(plots_dir, exist_ok=True)
                if local_plot_every_n is None:
                    local_plot_every_n = 10 if total_tiles > 50 else 1
            tile_counter = 0
            phase_label = f"Inference phase ({os.path.basename(input_tif_path)})"
            with tempfile.TemporaryDirectory(prefix="scene_infer_") as temp_dir:
                with (
                    rasterio.open(input_tif_path) as src,
                    TimedBlock(context.logger, phase_label),
                ):
                    prob_accum = np.memmap(
                        os.path.join(temp_dir, "prob_accum.dat"),
                        dtype=np.float32,
                        mode="w+",
                        shape=(model_cfg["num_classes"], output_height, output_width),
                    )
                    prob_accum[:] = 0.0
                    weight_accum = np.memmap(
                        os.path.join(temp_dir, "weight_accum.dat"),
                        dtype=np.float32,
                        mode="w+",
                        shape=(output_height, output_width),
                    )
                    weight_accum[:] = 0.0
                    gradcam_accum = None
                    if explain_enabled:
                        gradcam_accum = np.memmap(
                            os.path.join(temp_dir, "gradcam_accum.dat"),
                            dtype=np.float32,
                            mode="w+",
                            shape=(output_height, output_width),
                        )
                        gradcam_accum[:] = 0.0

                    for y in y_positions:
                        for x in x_positions:
                            tile_counter += 1
                            y_max = min(y + tile_size_eff, height)
                            x_max = min(x + tile_size_eff, width)
                            window = Window.from_slices((y, y_max), (x, x_max))
                            mask_tile = src.read_masks(window=window)
                            if not np.any(mask_tile):
                                continue
                            img_tile = src.read(window=window, boundless=False)
                            img_tile = np.transpose(img_tile, (1, 2, 0))
                            img_tile_raw = img_tile
                            orig_h, orig_w = img_tile.shape[:2]
                            pad_h = max(0, tile_size_eff - orig_h)
                            pad_w = max(0, tile_size_eff - orig_w)
                            if pad_h or pad_w:
                                context.logger.info(
                                    "Scene %s is smaller than one full inference tile; "
                                    "reflect-padding %sx%s up to %sx%s for tile %s."
                                    % (
                                        os.path.basename(input_tif_path),
                                        orig_h,
                                        orig_w,
                                        tile_size_eff,
                                        tile_size_eff,
                                        tile_counter,
                                    )
                                )
                                img_tile = np.pad(
                                    img_tile,
                                    ((0, pad_h), (0, pad_w), (0, 0)),
                                    mode="reflect",
                                )
                            tile_probs = np.zeros(
                                (model_cfg["num_classes"], orig_h, orig_w),
                                dtype=np.float32,
                            )
                            for transform in tta_transforms:
                                aug_img = transform.apply(img_tile)
                                img_tile_norm = (
                                    aug_img.astype(np.float32) / 255.0
                                ).astype(np.float32)
                                img_t = (
                                    torch.from_numpy(img_tile_norm)
                                    .permute(2, 0, 1)
                                    .unsqueeze(0)
                                    .to(device)
                                )
                                feats_batched = []
                                if uses_backbone_features:
                                    assert (
                                        backbone is not None and processor is not None
                                    )
                                    feats = extract_multiscale_features(
                                        aug_img.astype(np.float32),
                                        backbone,
                                        processor,
                                        device,
                                        model_cfg["layers"],
                                        ps=ps,
                                    )
                                    feats_batched = [
                                        f.to(device).unsqueeze(0) for f in feats
                                    ]
                                with torch.no_grad(), autocast:
                                    payload = normalize_forward_output(
                                        head(img_t, feats_batched)
                                    )
                                    logits = payload["logits"]
                                    logits = transform.invert_logits(logits)
                                    if logits.shape[-2:] != img_t.shape[-2:]:
                                        logits = F.interpolate(
                                            logits,
                                            size=img_t.shape[-2:],
                                            mode="bilinear",
                                            align_corners=False,
                                        )
                                    probs = (
                                        torch.softmax(logits, dim=1)
                                        .squeeze(0)
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )
                                probs = probs[:, :orig_h, :orig_w]
                                tile_probs += probs
                            tile_probs /= len(tta_transforms)
                            gradcam_tile = None
                            if explain_enabled:
                                if uses_backbone_features:
                                    assert (
                                        backbone is not None and processor is not None
                                    )
                                    gradcam_tiles_attempted += 1
                                    gradcam_result = compute_gradcam_map(
                                        img_tile_raw.astype(np.float32),
                                        backbone,
                                        head,
                                        processor,
                                        device,
                                        model_cfg["layers"],
                                        ps,
                                        class_index,
                                        cam_layer=explain_cam_layer,
                                        logger=None,
                                    )
                                    gradcam_tile = upsample_map(
                                        np.asarray(
                                            gradcam_result["cam_map"], dtype=np.float32
                                        ),
                                        orig_h,
                                        orig_w,
                                    )
                                    if bool(gradcam_result["success"]):
                                        gradcam_tiles_succeeded += 1
                                    else:
                                        gradcam_tiles_failed += 1
                                        if gradcam_first_failure is None:
                                            gradcam_first_failure = (
                                                "scene=%s tile=%s layer=%s stage=%s reason=%s"
                                                % (
                                                    os.path.basename(input_tif_path),
                                                    tile_counter,
                                                    gradcam_result.get(
                                                        "selected_layer"
                                                    ),
                                                    gradcam_result.get("failure_stage"),
                                                    gradcam_result.get(
                                                        "failure_reason"
                                                    ),
                                                )
                                            )
                                            context.logger.warning(
                                                "Grad-CAM fallback engaged; using zero maps. %s"
                                                % gradcam_first_failure
                                            )
                                else:
                                    gradcam_tile = np.zeros(
                                        (orig_h, orig_w), dtype=np.float32
                                    )
                            blend_key = (orig_h, orig_w)
                            if label_grid_enabled:
                                tile_bounds = src.window_bounds(window)
                                label_bounds = tile_bounds
                                if source_crs != output_crs:
                                    label_bounds = rasterio.warp.transform_bounds(
                                        source_crs,
                                        output_crs,
                                        *tile_bounds,
                                        densify_pts=21,
                                    )
                                label_window = (
                                    rasterio.windows.from_bounds(
                                        *label_bounds,
                                        transform=output_transform,
                                    )
                                    .round_offsets()
                                    .round_lengths()
                                )
                                label_window = Window(
                                    col_off=max(0, int(label_window.col_off)),
                                    row_off=max(0, int(label_window.row_off)),
                                    width=max(1, int(label_window.width)),
                                    height=max(1, int(label_window.height)),
                                )
                                row_end = min(
                                    int(label_window.row_off + label_window.height),
                                    output_height,
                                )
                                col_end = min(
                                    int(label_window.col_off + label_window.width),
                                    output_width,
                                )
                                target_h = max(
                                    1,
                                    row_end - int(label_window.row_off),
                                )
                                target_w = max(
                                    1,
                                    col_end - int(label_window.col_off),
                                )
                                tile_probs = (
                                    F.interpolate(
                                        torch.from_numpy(tile_probs).unsqueeze(0),
                                        size=(target_h, target_w),
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                                    .squeeze(0)
                                    .numpy()
                                )
                                if explain_enabled and gradcam_tile is not None:
                                    gradcam_tile = upsample_map(
                                        np.asarray(gradcam_tile, dtype=np.float32),
                                        target_h,
                                        target_w,
                                    )
                                blend_key = (target_h, target_w)
                                y_slice = slice(int(label_window.row_off), row_end)
                                x_slice = slice(int(label_window.col_off), col_end)
                            else:
                                y_slice = slice(y, y_max)
                                x_slice = slice(x, x_max)
                            blend_mask = weight_cache.get(blend_key)
                            if blend_mask is None:
                                blend_mask = build_blend_weight_mask(
                                    blend_key[0],
                                    blend_key[1],
                                    mode=merge_mode,
                                )
                                weight_cache[blend_key] = blend_mask
                            if (
                                explain_enabled
                                and tile_debug_enable
                                and local_plot_every_n
                                and tile_counter % local_plot_every_n == 0
                            ):
                                try:
                                    rgb = np.clip(img_tile_raw, 0, 255).astype(np.uint8)
                                    _, _, class_prob = compute_xai_maps(
                                        tile_probs, class_index
                                    )
                                    pred_tile = tile_probs.argmax(axis=0).astype(
                                        np.uint8
                                    )
                                    if rgb.shape[:2] != pred_tile.shape:
                                        rgb = np.transpose(
                                            src.read(
                                                [1, 2, 3],
                                                window=window,
                                                out_shape=(
                                                    3,
                                                    pred_tile.shape[0],
                                                    pred_tile.shape[1],
                                                ),
                                                resampling=Resampling.bilinear,
                                            ),
                                            (1, 2, 0),
                                        ).astype(np.uint8)
                                    overlay_pred = overlay_binary_mask(
                                        rgb,
                                        pred_tile == foreground_class,
                                        color=overlay_color,
                                        alpha=overlay_alpha,
                                    )
                                    gradcam_overlay = overlay_heatmap(
                                        rgb,
                                        np.asarray(gradcam_tile, dtype=np.float32),
                                        cmap="magma",
                                        alpha=0.45,
                                    )
                                    plot_path = os.path.join(
                                        plots_dir,
                                        f"{plot_prefix}_tile_y{y}_x{x}_compact.png",
                                    )
                                    build_dashboard(
                                        plot_path,
                                        rgb,
                                        overlay_pred,
                                        gradcam_overlay,
                                        class_prob,
                                        layout=layout,
                                    )
                                except Exception:
                                    context.logger.error(
                                        "XAI plotting failed for tile y=%s x=%s\n%s"
                                        % (y, x, traceback.format_exc())
                                    )
                            prob_accum[:, y_slice, x_slice] += (
                                tile_probs * blend_mask[None, ...]
                            )
                            weight_accum[y_slice, x_slice] += blend_mask
                            if (
                                explain_enabled
                                and gradcam_tile is not None
                                and gradcam_accum is not None
                            ):
                                gradcam_accum[y_slice, x_slice] += (
                                    gradcam_tile * blend_mask
                                )
                            if tile_counter % 50 == 0 or tile_counter == total_tiles:
                                context.logger.info(
                                    f"Inference progress: {tile_counter}/{total_tiles} tiles."
                                )
                                context.hook_manager.on_inference_tile(
                                    context,
                                    self.name,
                                    tile_counter,
                                    total_tiles,
                                )
                    pred_full = np.memmap(
                        os.path.join(temp_dir, "pred_full.dat"),
                        dtype=np.uint8,
                        mode="w+",
                        shape=(output_height, output_width),
                    )
                    for row_off in range(0, output_height, scene_window_size):
                        row_end = min(output_height, row_off + scene_window_size)
                        for col_off in range(0, output_width, scene_window_size):
                            col_end = min(output_width, col_off + scene_window_size)
                            weights = np.asarray(
                                weight_accum[row_off:row_end, col_off:col_end],
                                dtype=np.float32,
                            )
                            weights[weights == 0] = 1.0
                            probs = np.asarray(
                                prob_accum[:, row_off:row_end, col_off:col_end],
                                dtype=np.float32,
                            )
                            probs /= weights[None, ...]
                            pred_full[row_off:row_end, col_off:col_end] = probs.argmax(
                                axis=0
                            ).astype(np.uint8)
                            prob_accum[:, row_off:row_end, col_off:col_end] = probs
                    profile.update(
                        dtype=rasterio.uint8,
                        count=1,
                        nodata=0,
                        transform=output_transform,
                        crs=output_crs,
                        height=output_height,
                        width=output_width,
                    )
                    if output_tif_path is not None:
                        os.makedirs(
                            os.path.dirname(output_tif_path) or ".", exist_ok=True
                        )
                        with rasterio.open(output_tif_path, "w", **profile) as dst:
                            for row_off in range(0, output_height, scene_window_size):
                                row_end = min(
                                    output_height, row_off + scene_window_size
                                )
                                for col_off in range(
                                    0, output_width, scene_window_size
                                ):
                                    col_end = min(
                                        output_width, col_off + scene_window_size
                                    )
                                    dst.write(
                                        np.asarray(
                                            pred_full[row_off:row_end, col_off:col_end],
                                            dtype=np.uint8,
                                        ),
                                        1,
                                        window=Window(
                                            col_off=col_off,
                                            row_off=row_off,
                                            width=col_end - col_off,
                                            height=row_end - row_off,
                                        ),
                                    )
                        context.logger.info(f"Saved prediction to {output_tif_path}")
                    scene_metrics = {"tiles_total": float(total_tiles)}
                    if cumulative_output_path is not None:
                        write_prediction_to_cumulative_raster(
                            cumulative_output_path,
                            pred_full,
                            output_transform,
                            output_crs,
                        )
                        context.logger.info(
                            "Updated cumulative prediction raster %s for %s"
                            % (cumulative_output_path, os.path.basename(input_tif_path))
                        )
                        scene_metrics["cumulative_updates"] = 1.0
                    if gradcam_tiles_attempted > 0:
                        scene_metrics["gradcam_tiles_attempted"] = float(
                            gradcam_tiles_attempted
                        )
                        scene_metrics["gradcam_tiles_succeeded"] = float(
                            gradcam_tiles_succeeded
                        )
                        scene_metrics["gradcam_tiles_failed"] = float(
                            gradcam_tiles_failed
                        )
                        summary_message = (
                            "Grad-CAM summary for %s :: attempted=%s succeeded=%s failed=%s layer=%s"
                            % (
                                os.path.basename(input_tif_path),
                                gradcam_tiles_attempted,
                                gradcam_tiles_succeeded,
                                gradcam_tiles_failed,
                                explain_cam_layer,
                            )
                        )
                        if gradcam_first_failure is not None:
                            summary_message += " first_failure=" + gradcam_first_failure
                        context.logger.info(summary_message)
                    if explain_enabled and scene_rgb is not None:
                        try:
                            assert gradcam_accum is not None
                            gradcam_scene = np.zeros(
                                (output_height, output_width),
                                dtype=np.float32,
                            )
                            class_prob_scene = np.zeros(
                                (output_height, output_width),
                                dtype=np.float32,
                            )
                            for row_off in range(0, output_height, scene_window_size):
                                row_end = min(
                                    output_height, row_off + scene_window_size
                                )
                                for col_off in range(
                                    0, output_width, scene_window_size
                                ):
                                    col_end = min(
                                        output_width, col_off + scene_window_size
                                    )
                                    weights = np.asarray(
                                        weight_accum[row_off:row_end, col_off:col_end],
                                        dtype=np.float32,
                                    )
                                    weights[weights == 0] = 1.0
                                    gradcam_scene[row_off:row_end, col_off:col_end] = (
                                        np.asarray(
                                            gradcam_accum[
                                                row_off:row_end,
                                                col_off:col_end,
                                            ],
                                            dtype=np.float32,
                                        )
                                        / weights
                                    )
                                    class_prob_scene[
                                        row_off:row_end, col_off:col_end
                                    ] = np.asarray(
                                        prob_accum[
                                            class_index,
                                            row_off:row_end,
                                            col_off:col_end,
                                        ],
                                        dtype=np.float32,
                                    )
                            gradcam_scene = normalize_map(gradcam_scene)
                            scene_rgb_uint8 = np.clip(scene_rgb, 0, 255).astype(
                                np.uint8
                            )
                            overlay_pred = overlay_binary_mask(
                                scene_rgb_uint8,
                                pred_full == foreground_class,
                                color=overlay_color,
                                alpha=overlay_alpha,
                            )
                            gradcam_overlay = overlay_heatmap(
                                scene_rgb_uint8,
                                gradcam_scene,
                                cmap="magma",
                                alpha=0.45,
                            )
                            plot_path = os.path.join(
                                plots_dir,
                                f"{plot_prefix}_scene_dashboard.png",
                            )
                            build_dashboard(
                                plot_path,
                                scene_rgb_uint8,
                                overlay_pred,
                                gradcam_overlay,
                                class_prob_scene,
                                layout=layout,
                            )
                            scene_metrics["scene_plot"] = 1.0
                        except Exception:
                            context.logger.error(
                                "Scene XAI plotting failed for %s\n%s"
                                % (plot_prefix, traceback.format_exc())
                            )
                    if vector_enabled and vector_output_path is not None:
                        features = extract_prediction_features(
                            pred_full,
                            output_transform,
                            output_crs,
                            source_id=plot_prefix,
                            run_id=context.run_id,
                            foreground_class=foreground_class,
                            target_epsg=vector_target_epsg,
                        )
                        append_prediction_shapefile(
                            vector_output_path,
                            features,
                            target_epsg=vector_target_epsg,
                            append=vector_append,
                        )
                        scene_metrics["vector_features"] = float(len(features))
                    return scene_metrics

        directory_mode = bool(input_dir or input_paths_file)
        if directory_mode and input_tif:
            raise InferenceError(
                "Set only one input source: either inference.input_tif or "
                "inference.input_dir/inference.input_paths_file."
            )
        if directory_mode:
            if not label_path or not os.path.exists(label_path):
                raise InferenceError(
                    "Directory inference requires a valid label_path so the shared "
                    "prediction GeoTIFF can inherit CRS, resolution, and grid "
                    "alignment."
                )
            if input_paths_file:
                manifest_path = Path(input_paths_file).expanduser()
                if not manifest_path.is_absolute():
                    config_root = (
                        Path(context.config_path).resolve().parent
                        if context.config_path
                        else Path.cwd()
                    )
                    manifest_path = config_root / manifest_path
                if not manifest_path.exists():
                    raise InferenceError(
                        f"inference.input_paths_file not found: {manifest_path}"
                    )
                tile_files = _resolve_directory_input_paths(
                    manifest_path=manifest_path,
                    input_dir=input_dir,
                    glob_pattern=glob_pattern,
                )
            else:
                tile_files = sorted(glob.glob(os.path.join(input_dir, glob_pattern)))
            if not tile_files:
                input_desc = str(manifest_path) if input_paths_file else str(input_dir)
                raise InferenceError(f"No input files found in {input_desc}")
            cumulative_output_path = str(
                output_tif
                or context.run_dir
                / "artifacts"
                / "rasters"
                / "inference"
                / "predictions.tif"
            )
            with rasterio.open(label_path) as template_src:
                template_crs = template_src.crs
                template_transform = template_src.transform
            cumulative_window = None
            for tile_path in tile_files:
                try:
                    build_tile_grid_layout(
                        tile_path,
                        label_path,
                        requested_tile_size=tile_size,
                        patch_size=ps,
                    )
                except Exception:
                    continue
                with rasterio.open(tile_path) as src:
                    tile_bounds = src.bounds
                    if src.crs != template_crs:
                        tile_bounds = rasterio.warp.transform_bounds(
                            src.crs,
                            template_crs,
                            *src.bounds,
                            densify_pts=21,
                        )
                tile_window = (
                    rasterio.windows.from_bounds(
                        *tile_bounds,
                        transform=template_transform,
                    )
                    .round_offsets()
                    .round_lengths()
                )
                tile_window = Window(
                    col_off=int(tile_window.col_off),
                    row_off=int(tile_window.row_off),
                    width=max(1, int(tile_window.width)),
                    height=max(1, int(tile_window.height)),
                )
                if cumulative_window is None:
                    cumulative_window = tile_window
                    continue
                row_off = min(
                    int(cumulative_window.row_off),
                    int(tile_window.row_off),
                )
                col_off = min(
                    int(cumulative_window.col_off),
                    int(tile_window.col_off),
                )
                row_end = max(
                    int(cumulative_window.row_off + cumulative_window.height),
                    int(tile_window.row_off + tile_window.height),
                )
                col_end = max(
                    int(cumulative_window.col_off + cumulative_window.width),
                    int(tile_window.col_off + tile_window.width),
                )
                cumulative_window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=col_end - col_off,
                    height=row_end - row_off,
                )
            if cumulative_window is None:
                raise InferenceError(
                    "No input files in %s could align to the configured label grid."
                    % (input_dir or input_paths_file)
                )
            cumulative_backup_path = None
            with hold_prediction_raster_lock(cumulative_output_path):
                existed_before = os.path.exists(cumulative_output_path)
                ensure_cumulative_prediction_raster(
                    cumulative_output_path,
                    label_path,
                    template_window=cumulative_window,
                )
                if existed_before:
                    cumulative_backup_path = build_cumulative_raster_backup_path(
                        cumulative_output_path,
                        context.run_id,
                    )
                    if not os.path.exists(cumulative_backup_path):
                        backup_prediction_raster(
                            cumulative_output_path,
                            cumulative_backup_path,
                        )
                        context.logger.info(
                            "Backed up cumulative prediction raster to %s"
                            % cumulative_backup_path
                        )
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                if explain_enabled:
                    default_plot_root = (
                        output_dir
                        or os.path.dirname(cumulative_output_path or "")
                        or str(context.run_dir / "artifacts" / "plots" / "inference")
                    )
                    plots_dir = plots_dir or os.path.join(default_plot_root, "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                vector_output_path = None
                if vector_enabled:
                    context.logger.warning(
                        "Skipping inference.vector export in directory mode because "
                        "predictions are now accumulated into one GeoTIFF."
                    )
                total_tiles = 0.0
                total_gradcam_tiles_attempted = 0.0
                total_gradcam_tiles_succeeded = 0.0
                total_gradcam_tiles_failed = 0.0
                total_vector_features = 0.0
                total_cumulative_updates = 0.0
                total_skipped_alignment = 0.0
                for idx, tile_path in enumerate(tile_files, start=1):
                    base = os.path.splitext(os.path.basename(tile_path))[0]
                    context.logger.info(
                        "Running file inference %s/%s: %s"
                        % (idx, len(tile_files), base)
                    )
                    try:
                        file_metrics = _infer_one_tif(
                            tile_path,
                            None,
                            plot_prefix=base,
                            vector_output_path=vector_output_path,
                            cumulative_output_path=cumulative_output_path,
                        )
                    except InferenceError as exc:
                        message = str(exc)
                        if "label-grid alignment" in message:
                            total_skipped_alignment += 1.0
                            context.logger.warning(
                                "Skipping %s due to label-grid alignment failure: %s"
                                % (base, message)
                            )
                            continue
                        raise
                    total_tiles += float(file_metrics.get("tiles_total", 0.0))
                    total_gradcam_tiles_attempted += float(
                        file_metrics.get("gradcam_tiles_attempted", 0.0)
                    )
                    total_gradcam_tiles_succeeded += float(
                        file_metrics.get("gradcam_tiles_succeeded", 0.0)
                    )
                    total_gradcam_tiles_failed += float(
                        file_metrics.get("gradcam_tiles_failed", 0.0)
                    )
                    total_vector_features += float(
                        file_metrics.get("vector_features", 0.0)
                    )
                    total_cumulative_updates += float(
                        file_metrics.get("cumulative_updates", 0.0)
                    )
            metrics = {
                "files_total": float(len(tile_files)),
                "tiles_total": total_tiles,
                "gradcam_tiles_attempted": total_gradcam_tiles_attempted,
                "gradcam_tiles_succeeded": total_gradcam_tiles_succeeded,
                "gradcam_tiles_failed": total_gradcam_tiles_failed,
                "vector_features": total_vector_features,
                "cumulative_updates": total_cumulative_updates,
                "files_skipped_alignment": total_skipped_alignment,
                "files_skipped_overlap": 0.0,
            }
            artifacts = {
                "checkpoint": checkpoint,
                "output_tif": cumulative_output_path,
            }
            if explain_enabled and plots_dir is not None:
                artifacts["plots_dir"] = plots_dir
            if vector_enabled and vector_output_path is not None:
                artifacts["vector_output"] = vector_output_path
            if cumulative_backup_path is not None:
                artifacts["output_tif_backup"] = cumulative_backup_path
            return PhaseOutcome(metrics=metrics, artifacts=artifacts)
        if not input_tif:
            raise InferenceError(
                "Either inference.input_tif or inference.input_dir must be set."
            )
        if not output_tif:
            raise InferenceError("output_tif is required when input_tif is set")
        if explain_enabled:
            plots_dir = plots_dir or os.path.join(os.path.dirname(output_tif), "plots")
            os.makedirs(plots_dir, exist_ok=True)
        vector_output_path = None
        if vector_enabled:
            vector_output_path = str(
                context.run_dir
                / "artifacts"
                / "vectors"
                / "inference"
                / "predictions_4326.shp"
            )
        single_prefix = os.path.splitext(os.path.basename(input_tif))[0]
        metrics = _infer_one_tif(
            input_tif,
            output_tif,
            plot_prefix=single_prefix,
            vector_output_path=vector_output_path,
        )
        artifacts = {"output_tif": output_tif, "checkpoint": checkpoint}
        if explain_enabled and plots_dir is not None:
            artifacts["plots_dir"] = plots_dir
        if vector_enabled and vector_output_path is not None:
            artifacts["vector_output"] = vector_output_path
        return PhaseOutcome(metrics=metrics, artifacts=artifacts)
