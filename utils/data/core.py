"""
Data-handling utilities: tiling GeoTIFFs, caching features, validation, and
dataset loader.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds

if TYPE_CHECKING:
    from utils.logging import VerbosityLogger


CACHE_META_FILENAME = "cache_meta.json"
SUPERVISION_GRID_MODE = "native_label_grid"
DEFAULT_DATASET_VALIDATION_CONFIG = {
    "enabled": True,
    "allowed_labels": (0, 1),
    "ignore_index": 255,
    "out_of_range_policy": "map_to_ignore",
    "require_finite_features": True,
    "require_finite_images": True,
}
DEFAULT_TILE_FILTER_CONFIG = {
    "enabled": False,
    "mode": "foreground_any",
    "foreground_labels": (1,),
}


@dataclass(frozen=True)
class TileGridLayout:
    """Describe the paired image-grid and label-grid tile dimensions.

    Args:
        image_tile_size (int): Output image tile size in pixels.
        label_tile_size (int): Output label tile size in pixels.
        image_resolution (float): Image raster resolution in map units per pixel.
        label_resolution (float): Label raster resolution in map units per pixel.
        scale_factor (int): Integer ratio ``label_resolution / image_resolution``.

    Examples:
        >>> TileGridLayout(480, 96, 0.2, 1.0, 5).label_tile_size
        96
    """

    image_tile_size: int
    label_tile_size: int
    image_resolution: float
    label_resolution: float
    scale_factor: int


EDGE_POLICY_DROP_PARTIAL = "drop_partial"


def _pixel_size(transform: Affine) -> tuple[float, float]:
    """Return absolute pixel size from an affine transform.

    Args:
        transform (Affine): Raster affine transform.

    Returns:
        tuple[float, float]: Pixel width and height.

    Examples:
        >>> _pixel_size(Affine(2.0, 0.0, 0.0, 0.0, -3.0, 0.0))
        (2.0, 3.0)
    """

    return abs(float(transform.a)), abs(float(transform.e))


def build_tile_grid_layout(
    image_path: str,
    label_path: str,
    requested_tile_size: int,
    patch_size: int,
) -> TileGridLayout:
    """Build compatible image/label tile sizes for label-grid supervision.

    The image tile size is reduced, if needed, so it is compatible with both
    the backbone patch size and the native label-grid scale factor.

    Args:
        image_path (str): Input image path.
        label_path (str): Label raster path.
        requested_tile_size (int): Requested image tile size in pixels.
        patch_size (int): Backbone patch size in image pixels.

    Returns:
        TileGridLayout: Compatible image and label tile dimensions.

    Raises:
        ValueError: If the imagery/label grids do not support an integer
            scale-factor relationship in one CRS.

    Examples:
        >>> build_tile_grid_layout(  # doctest: +SKIP
        ...     "image.tif",
        ...     "labels.tif",
        ...     requested_tile_size=512,
        ...     patch_size=16,
        ... )
        TileGridLayout(...)
    """

    with rasterio.open(image_path) as src_img, rasterio.open(label_path) as src_lab:
        if src_img.crs != src_lab.crs:
            raise ValueError(
                "Native label-grid supervision currently requires imagery and "
                "labels to share one CRS."
            )
        img_res_x, img_res_y = _pixel_size(src_img.transform)
        lab_res_x, lab_res_y = _pixel_size(src_lab.transform)
    if not math.isclose(img_res_x, img_res_y, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("Image raster must have square pixels for label-grid tiling.")
    if not math.isclose(lab_res_x, lab_res_y, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("Label raster must have square pixels for label-grid tiling.")
    scale = lab_res_x / img_res_x
    rounded_scale = int(round(scale))
    if rounded_scale < 1 or not math.isclose(
        scale, rounded_scale, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError(
            "Label-grid supervision requires label resolution to be an integer "
            "multiple of image resolution."
        )
    compatible_multiple = math.lcm(int(patch_size), int(rounded_scale))
    image_tile_size = (
        int(requested_tile_size) // compatible_multiple
    ) * compatible_multiple
    if image_tile_size < compatible_multiple:
        raise ValueError(
            "Requested tile_size is too small for the image/label resolution "
            "ratio and patch-size constraints."
        )
    label_tile_size = image_tile_size // rounded_scale
    return TileGridLayout(
        image_tile_size=image_tile_size,
        label_tile_size=label_tile_size,
        image_resolution=img_res_x,
        label_resolution=lab_res_x,
        scale_factor=rounded_scale,
    )


def full_fit_window_positions(total_size: int, tile_size: int) -> list[int]:
    """Return fixed-size starts whose full tiles fit inside one dimension.

    Args:
        total_size (int): Raster width or height in pixels.
        tile_size (int): Tile size in pixels.

    Returns:
        list[int]: Start offsets for fully fitting windows.

    Examples:
        >>> full_fit_window_positions(8, 4)
        [0, 4]
        >>> full_fit_window_positions(10, 4)
        [0, 4]
        >>> full_fit_window_positions(3, 4)
        []
    """

    if total_size < tile_size:
        return []
    return list(range(0, (total_size - tile_size) + 1, tile_size))


def coverage_window_positions(
    total_size: int, tile_size: int, stride: int
) -> list[int]:
    """Return sliding-window starts that cover one full dimension.

    Args:
        total_size (int): Raster width or height in pixels.
        tile_size (int): Tile size in pixels.
        stride (int): Sliding-window stride in pixels.

    Returns:
        list[int]: Start offsets whose last window aligns to the border.

    Examples:
        >>> coverage_window_positions(8, 4, 4)
        [0, 4]
        >>> coverage_window_positions(10, 4, 4)
        [0, 4, 6]
        >>> coverage_window_positions(11, 4, 4)
        [0, 4, 7]
        >>> coverage_window_positions(3, 4, 4)
        [0]
    """

    if stride <= 0:
        raise ValueError("stride must be positive")
    if total_size <= tile_size:
        return [0]
    positions = list(range(0, total_size, stride))
    last_start = total_size - tile_size
    if positions[-1] != last_start:
        positions[-1] = last_start
    deduped: list[int] = []
    for pos in positions:
        if not deduped or deduped[-1] != pos:
            deduped.append(pos)
    return deduped


def _tile_window_positions(total_size: int, tile_size: int) -> list[int]:
    """Backward-compatible alias for coverage-style fixed-size tiling.

    Args:
        total_size (int): Raster width or height in pixels.
        tile_size (int): Tile size in pixels.

    Returns:
        list[int]: Coverage-aligned start offsets.
    """

    return coverage_window_positions(total_size, tile_size, tile_size)


def read_label_window_for_image_bounds(
    image_path: str,
    label_path: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the native label-grid subset covering one image footprint.

    Args:
        image_path (str): Input image path.
        label_path (str): Label raster path.

    Returns:
        tuple[np.ndarray, dict[str, Any]]: Label array and metadata including
        CRS, transform, bounds, width, and height.

    Raises:
        ValueError: If no overlap exists between the image and label rasters.

    Examples:
        >>> read_label_window_for_image_bounds(  # doctest: +SKIP
        ...     "image.tif",
        ...     "labels.tif",
        ... )
        (array(...), {'crs': ...})
        >>> isinstance(  # doctest: +SKIP
        ...     read_label_window_for_image_bounds("image.tif", "labels.tif")[1],
        ...     dict,
        ... )
        True
    """

    with rasterio.open(image_path) as src_img, rasterio.open(label_path) as src_lab:
        if src_img.crs == src_lab.crs:
            image_bounds_in_label = src_img.bounds
        else:
            image_bounds_in_label = transform_bounds(
                src_img.crs,
                src_lab.crs,
                *src_img.bounds,
                densify_pts=21,
            )
        label_window = (
            from_bounds(
                *image_bounds_in_label,
                transform=src_lab.transform,
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
        label_array = src_lab.read(1, window=label_window, boundless=True, fill_value=0)
        transform = rasterio.windows.transform(label_window, src_lab.transform)
        bounds = window_bounds(label_window, src_lab.transform)
        meta = {
            "crs": src_lab.crs,
            "transform": transform,
            "bounds": bounds,
            "width": int(label_array.shape[1]),
            "height": int(label_array.shape[0]),
        }
        return label_array, meta


def read_image_tile_for_label_bounds(
    image_path: str,
    bounds: tuple[float, float, float, float],
    image_tile_size: int,
) -> np.ndarray:
    """Read one image tile resampled onto bounds defined in image CRS.

    Args:
        image_path (str): Input image path.
        bounds (tuple[float, float, float, float]): Tile bounds in image CRS.
        image_tile_size (int): Output tile size in pixels.

    Returns:
        np.ndarray: RGB image tile shaped ``(H, W, C)``.

    Examples:
        >>> read_image_tile_for_label_bounds(  # doctest: +SKIP
        ...     "image.tif",
        ...     (0.0, 0.0, 10.0, 10.0),
        ...     image_tile_size=64,
        ... ).shape
        (64, 64, 3)
        >>> read_image_tile_for_label_bounds(  # doctest: +SKIP
        ...     "image.tif",
        ...     (0.0, 0.0, 5.0, 5.0),
        ...     image_tile_size=32,
        ... ).dtype
        dtype('float32')
    """

    with rasterio.open(image_path) as src_img:
        destination = np.zeros(
            (src_img.count, int(image_tile_size), int(image_tile_size)),
            dtype=np.float32,
        )
        left, bottom, right, top = bounds
        dst_transform = from_origin(
            left,
            top,
            (right - left) / float(image_tile_size),
            (top - bottom) / float(image_tile_size),
        )
        for band_index in range(1, src_img.count + 1):
            reproject(
                source=rasterio.band(src_img, band_index),
                destination=destination[band_index - 1],
                src_transform=src_img.transform,
                src_crs=src_img.crs,
                dst_transform=dst_transform,
                dst_crs=src_img.crs,
                resampling=Resampling.bilinear,
            )
    return np.transpose(destination, (1, 2, 0))


def _normalize_dataset_validation_cfg(
    validation_cfg: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Return normalized dataset validation settings.

    Args:
        validation_cfg (Optional[dict[str, Any]]): Raw validation config.

    Returns:
        dict[str, Any]: Normalized validation config.
    """

    cfg = dict(DEFAULT_DATASET_VALIDATION_CONFIG)
    if validation_cfg:
        cfg.update(validation_cfg)
    allowed_raw = cfg.get("allowed_labels", (0, 1))
    if isinstance(allowed_raw, (list, tuple, set)):
        allowed_labels = tuple(sorted({int(v) for v in allowed_raw}))
    else:
        allowed_labels = (0, 1)
    if not allowed_labels:
        allowed_labels = (0, 1)
    policy = str(cfg.get("out_of_range_policy", "map_to_ignore")).lower()
    if policy not in {"map_to_ignore", "error"}:
        policy = "map_to_ignore"
    ignore_index = cfg.get("ignore_index", 255)
    cfg["allowed_labels"] = allowed_labels
    cfg["ignore_index"] = None if ignore_index is None else int(ignore_index)
    cfg["out_of_range_policy"] = policy
    cfg["enabled"] = bool(cfg.get("enabled", True))
    cfg["require_finite_features"] = bool(cfg.get("require_finite_features", True))
    cfg["require_finite_images"] = bool(cfg.get("require_finite_images", True))
    return cfg


def _label_validity_mask(
    label: torch.Tensor,
    allowed_labels: Sequence[int],
) -> torch.Tensor:
    """Build a mask of valid class labels.

    Args:
        label (torch.Tensor): Label tensor.
        allowed_labels (Sequence[int]): Allowed class indices.

    Returns:
        torch.Tensor: Boolean mask where labels are valid.
    """

    mask = torch.zeros_like(label, dtype=torch.bool)
    for cls in allowed_labels:
        mask |= label == int(cls)
    return mask


def _normalize_tile_filter_cfg(
    tile_filter_cfg: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Return normalized tile-filter settings.

    Args:
        tile_filter_cfg (Optional[dict[str, Any]]): Raw tile-filter config.

    Returns:
        dict[str, Any]: Normalized tile-filter config.
    """

    cfg = dict(DEFAULT_TILE_FILTER_CONFIG)
    if tile_filter_cfg:
        cfg.update(tile_filter_cfg)
    mode = str(cfg.get("mode", "foreground_any")).strip().lower()
    if mode not in {"foreground_any"}:
        mode = "foreground_any"
    labels_raw = cfg.get("foreground_labels", (1,))
    if isinstance(labels_raw, (list, tuple, set)):
        labels = tuple(sorted({int(v) for v in labels_raw}))
    else:
        labels = (1,)
    if not labels:
        labels = (1,)
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["mode"] = mode
    cfg["foreground_labels"] = labels
    return cfg


def _tile_passes_label_filter(
    label: np.ndarray | torch.Tensor,
    tile_filter_cfg: dict[str, Any],
) -> bool:
    """Return whether a tile satisfies the label-content filter.

    Args:
        label (np.ndarray | torch.Tensor): Tile label array.
        tile_filter_cfg (dict[str, Any]): Normalized tile-filter config.

    Returns:
        bool: ``True`` when tile should be kept.
    """

    if not tile_filter_cfg.get("enabled", False):
        return True
    if tile_filter_cfg.get("mode") != "foreground_any":
        return True
    label_np = (
        label.detach().cpu().numpy()
        if isinstance(label, torch.Tensor)
        else np.asarray(label)
    )
    if label_np.size == 0:
        return False
    if not np.isfinite(label_np).all():
        label_np = np.nan_to_num(label_np, nan=-1)
    label_int = label_np.astype(np.int64, copy=False)
    for cls in tile_filter_cfg.get("foreground_labels", (1,)):
        if np.any(label_int == int(cls)):
            return True
    return False


def _write_tile_payload_atomic(payload: dict[str, Any], save_path: str) -> bool:
    """Write one cached tile atomically without clobbering an existing tile.

    Args:
        payload (dict[str, Any]): Tile payload to persist.
        save_path (str): Final `.pt` path.

    Returns:
        bool: ``True`` when the tile was written, ``False`` when another writer
        already created the destination.

    Examples:
        >>> import tempfile
        >>> tmpdir = tempfile.mkdtemp()
        >>> path = os.path.join(tmpdir, "tile.pt")
        >>> _write_tile_payload_atomic({"x": torch.tensor([1])}, path)
        True
        >>> _write_tile_payload_atomic({"x": torch.tensor([2])}, path)
        False
    """

    temp_path = f"{save_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(payload, temp_path)
        try:
            os.link(temp_path, save_path)
        except FileExistsError:
            return False
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _sanitize_label_tensor(
    label: torch.Tensor,
    validation_cfg: dict[str, Any],
    source: str,
) -> torch.Tensor:
    """Validate and optionally sanitize invalid label values.

    Args:
        label (torch.Tensor): Label tensor.
        validation_cfg (dict[str, Any]): Normalized validation config.
        source (str): Source identifier for error messages.

    Returns:
        torch.Tensor: Sanitized label tensor.

    Raises:
        ValueError: If invalid labels are found and policy is ``error``.
    """

    if not validation_cfg.get("enabled", True):
        return label
    label = label.long()
    valid_mask = _label_validity_mask(label, validation_cfg["allowed_labels"])
    if valid_mask.all():
        return label
    invalid_values = torch.unique(label[~valid_mask]).cpu().tolist()
    policy = validation_cfg["out_of_range_policy"]
    if policy == "error":
        raise ValueError(
            f"Invalid labels {invalid_values[:10]} in {source}; "
            f"allowed={validation_cfg['allowed_labels']}"
        )
    ignore_index = validation_cfg.get("ignore_index")
    if ignore_index is None:
        raise ValueError(
            f"Invalid labels {invalid_values[:10]} in {source} and no ignore_index set"
        )
    sanitized = label.clone()
    sanitized[~valid_mask] = int(ignore_index)
    return sanitized


def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    """Convert an RGB tensor in [0, 1] to HSV.

    Args:
        image (torch.Tensor): RGB image tensor shaped (3, H, W).

    Returns:
        torch.Tensor: HSV image tensor shaped (3, H, W).
    """

    eps = 1e-8
    r, g, b = image[0], image[1], image[2]
    maxc, argmax = torch.max(image, dim=0)
    minc = torch.min(image, dim=0).values
    delta = maxc - minc
    saturation = torch.where(maxc > eps, delta / (maxc + eps), torch.zeros_like(maxc))
    hue = torch.zeros_like(maxc)
    valid = delta > eps
    delta_safe = delta + eps
    r_max = (argmax == 0) & valid
    g_max = (argmax == 1) & valid
    b_max = (argmax == 2) & valid
    hue = torch.where(r_max, ((g - b) / delta_safe) % 6.0, hue)
    hue = torch.where(g_max, ((b - r) / delta_safe) + 2.0, hue)
    hue = torch.where(b_max, ((r - g) / delta_safe) + 4.0, hue)
    hue = (hue / 6.0) % 1.0
    value = maxc
    return torch.stack((hue, saturation, value), dim=0)


def _hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an HSV tensor in [0, 1] to RGB.

    Args:
        image (torch.Tensor): HSV image tensor shaped (3, H, W).

    Returns:
        torch.Tensor: RGB image tensor shaped (3, H, W).
    """

    h, s, v = image[0], image[1], image[2]
    h6 = (h % 1.0) * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - torch.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = torch.zeros_like(v)
    g = torch.zeros_like(v)
    b = torch.zeros_like(v)
    m0 = i == 0
    m1 = i == 1
    m2 = i == 2
    m3 = i == 3
    m4 = i == 4
    m5 = i == 5
    r = torch.where(m0, v, r)
    g = torch.where(m0, t, g)
    b = torch.where(m0, p, b)
    r = torch.where(m1, q, r)
    g = torch.where(m1, v, g)
    b = torch.where(m1, p, b)
    r = torch.where(m2, p, r)
    g = torch.where(m2, v, g)
    b = torch.where(m2, t, b)
    r = torch.where(m3, p, r)
    g = torch.where(m3, q, g)
    b = torch.where(m3, v, b)
    r = torch.where(m4, t, r)
    g = torch.where(m4, p, g)
    b = torch.where(m4, v, b)
    r = torch.where(m5, v, r)
    g = torch.where(m5, p, g)
    b = torch.where(m5, q, b)
    return torch.stack((r, g, b), dim=0)


def _apply_color_jitter(img: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    """Apply color jitter augmentation to an image tensor.

    Args:
        img (torch.Tensor): Image tensor shaped (C, H, W) in [0, 1].
        cfg (dict[str, Any]): Color jitter config.

    Returns:
        torch.Tensor: Augmented image tensor.
    """

    if not cfg.get("enable", False):
        return img
    prob = float(cfg.get("prob", 0.0))
    if prob <= 0 or random.random() >= prob:
        return img
    out = img.clone()
    brightness = max(0.0, float(cfg.get("brightness", 0.0)))
    contrast = max(0.0, float(cfg.get("contrast", 0.0)))
    saturation = max(0.0, float(cfg.get("saturation", 0.0)))
    hue = max(0.0, min(0.5, float(cfg.get("hue", 0.0))))
    if brightness > 0:
        b_factor = 1.0 + random.uniform(-brightness, brightness)
        out = out * b_factor
    if contrast > 0:
        c_factor = 1.0 + random.uniform(-contrast, contrast)
        mean = out.mean(dim=(1, 2), keepdim=True)
        out = (out - mean) * c_factor + mean
    if saturation > 0 and out.shape[0] >= 3:
        s_factor = 1.0 + random.uniform(-saturation, saturation)
        gray = out[:3].mean(dim=0, keepdim=True)
        out[:3] = (out[:3] - gray) * s_factor + gray
    if hue > 0 and out.shape[0] >= 3:
        h_shift = random.uniform(-hue, hue)
        hsv = _rgb_to_hsv(out[:3])
        hsv[0] = (hsv[0] + h_shift) % 1.0
        out[:3] = _hsv_to_rgb(hsv)
    return torch.clamp(out, 0.0, 1.0)


def _apply_cutout(img: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    """Apply CutOut augmentation to an image tensor.

    Args:
        img (torch.Tensor): Image tensor shaped (C, H, W) in [0, 1].
        cfg (dict[str, Any]): CutOut config.

    Returns:
        torch.Tensor: Augmented image tensor.
    """

    if not cfg.get("enable", False):
        return img
    prob = float(cfg.get("prob", 0.0))
    if prob <= 0 or random.random() >= prob:
        return img
    out = img.clone()
    _, height, width = out.shape
    min_frac = max(0.0, float(cfg.get("min_frac", 0.08)))
    max_frac = max(min_frac, float(cfg.get("max_frac", 0.22)))
    num_holes = max(1, int(cfg.get("num_holes", 1)))
    fill = float(cfg.get("fill", 0.0))
    base = max(1, min(height, width))
    for _ in range(num_holes):
        frac = random.uniform(min_frac, max_frac)
        side = max(1, int(round(frac * base)))
        half = side // 2
        center_y = random.randint(0, max(0, height - 1))
        center_x = random.randint(0, max(0, width - 1))
        y0 = max(0, center_y - half)
        y1 = min(height, y0 + side)
        x0 = max(0, center_x - half)
        x1 = min(width, x0 + side)
        out[:, y0:y1, x0:x1] = fill
    return torch.clamp(out, 0.0, 1.0)


def _apply_gridmask(img: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    """Apply GridMask augmentation to an image tensor.

    Args:
        img (torch.Tensor): Image tensor shaped (C, H, W) in [0, 1].
        cfg (dict[str, Any]): GridMask config.

    Returns:
        torch.Tensor: Augmented image tensor.
    """

    if not cfg.get("enable", False):
        return img
    prob = float(cfg.get("prob", 0.0))
    if prob <= 0 or random.random() >= prob:
        return img
    out = img.clone()
    _, height, width = out.shape
    d_min = max(2, int(cfg.get("d_min", 48)))
    d_max = max(d_min, int(cfg.get("d_max", 128)))
    ratio = max(0.0, min(1.0, float(cfg.get("ratio", 0.5))))
    fill = float(cfg.get("fill", 0.0))
    period = random.randint(d_min, d_max)
    hole = max(1, int(round(period * ratio)))
    mask = torch.ones((height, width), dtype=out.dtype, device=out.device)
    offset_y = random.randint(0, max(0, period - 1))
    offset_x = random.randint(0, max(0, period - 1))
    for y0 in range(-period + offset_y, height, period):
        y1 = min(height, y0 + hole)
        ys = max(0, y0)
        if ys >= y1:
            continue
        for x0 in range(-period + offset_x, width, period):
            x1 = min(width, x0 + hole)
            xs = max(0, x0)
            if xs >= x1:
                continue
            mask[ys:y1, xs:x1] = 0.0
    if cfg.get("rotate", False):
        k = random.randint(0, 3)
        if k:
            mask = torch.rot90(mask, k, dims=(0, 1))
    mask = mask.unsqueeze(0)
    out = out * mask + fill * (1.0 - mask)
    return torch.clamp(out, 0.0, 1.0)


def _cache_subdir_name(
    tile_size: int,
    cache_features: bool,
    patch_size: int | None = None,
    edge_policy: str | None = None,
) -> str:
    """Build a cache subdirectory name for tile size and feature mode.

    Args:
        tile_size (int): Tile size in pixels.
        cache_features (bool): Whether features are cached.
        patch_size (int | None): Effective patch-size compatibility requirement
            for no-feature caches.
        edge_policy (str | None): Optional edge-window policy identifier for
            no-feature caches.

    Returns:
        str: Subdirectory name.

    Examples:
        >>> _cache_subdir_name(512, True)
        'tiles_512_feat_labelgrid'
        >>> _cache_subdir_name(1024, False, patch_size=16, edge_policy="drop_partial")
        'tiles_1024_nofeat_ps16_drop_partial_labelgrid'
    """

    suffix = "feat" if cache_features else "nofeat"
    if not cache_features and patch_size is not None:
        edge_suffix = (
            f"_{str(edge_policy).strip().lower()}" if edge_policy is not None else ""
        )
        return (
            f"tiles_{tile_size}_{suffix}_ps{int(patch_size)}" f"{edge_suffix}_labelgrid"
        )
    return f"tiles_{tile_size}_{suffix}_labelgrid"


def _cache_meta_path(cache_dir: str) -> str:
    """Return the metadata path for a cache directory.

    Args:
        cache_dir (str): Cache directory path.

    Returns:
        str: Metadata file path.

    Examples:
        >>> _cache_meta_path("/tmp/cache")
        '/tmp/cache/cache_meta.json'
    """

    return os.path.join(cache_dir, CACHE_META_FILENAME)


def _load_cache_metadata(cache_dir: str) -> dict[str, Any] | None:
    """Load cache metadata if present.

    Args:
        cache_dir (str): Cache directory path.

    Returns:
        dict[str, Any] | None: Metadata if available.

    Examples:
        >>> _load_cache_metadata("/tmp/does_not_exist") is None
        True
    """

    meta_path = _cache_meta_path(cache_dir)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_cache_metadata(
    cache_dir: str,
    tile_size: int,
    cache_features: bool,
    model_name: str | None,
    layers: Sequence[int] | None,
    patch_size: int | None,
    edge_policy: str | None = None,
) -> None:
    """Write cache metadata for a tile directory.

    Args:
        cache_dir (str): Cache directory path.
        tile_size (int): Tile size in pixels.
        cache_features (bool): Whether features are cached.
        model_name (str | None): Backbone model name.
        layers (Sequence[int] | None): Backbone layer indices.
        patch_size (int | None): Effective patch-size compatibility requirement.
        edge_policy (str | None): Optional edge-window policy identifier.

    Returns:
        None: Metadata is written to disk.
    """

    meta = {
        "tile_size": tile_size,
        "cache_features": cache_features,
        "model_name": model_name,
        "layers": list(layers) if layers is not None else None,
        "patch_size": None if patch_size is None else int(patch_size),
        "edge_policy": None if edge_policy is None else str(edge_policy),
        "supervision_grid_mode": SUPERVISION_GRID_MODE,
    }
    meta_path = _cache_meta_path(cache_dir)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_cache_metadata(
    meta: dict[str, Any],
    tile_size: int | None,
    cache_features: bool | None,
    model_name: str | None,
    layers: Sequence[int] | None,
    patch_size: int | None = None,
    edge_policy: str | None = None,
    allow_layer_subset: bool = False,
) -> None:
    """Validate cache metadata against expected settings.

    Args:
        meta (dict[str, Any]): Metadata loaded from cache.
        tile_size (int | None): Expected tile size.
        cache_features (bool | None): Expected cache_features setting.
        model_name (str | None): Expected model name.
        layers (Sequence[int] | None): Expected backbone layers.
        patch_size (int | None): Expected patch-size compatibility requirement
            for no-feature caches.
        edge_policy (str | None): Expected prepare-time edge policy.
        allow_layer_subset (bool): Allow the requested layer list to be a subset
            of the cached layer list.

    Raises:
        ValueError: If a metadata value does not match expectations.
    """

    mismatches = []
    if tile_size is not None and meta.get("tile_size") != tile_size:
        mismatches.append(f"tile_size={meta.get('tile_size')} expected {tile_size}")
    if cache_features is not None and meta.get("cache_features") != cache_features:
        mismatches.append(
            f"cache_features={meta.get('cache_features')} expected {cache_features}"
        )
    if model_name is not None and meta.get("model_name") != model_name:
        mismatches.append(f"model_name={meta.get('model_name')} expected {model_name}")
    if patch_size is not None:
        cached_patch_size = meta.get("patch_size")
        if cached_patch_size is None:
            if int(patch_size) != 1:
                mismatches.append(
                    f"patch_size={cached_patch_size} expected {int(patch_size)}"
                )
        elif int(cached_patch_size) != int(patch_size):
            mismatches.append(
                f"patch_size={cached_patch_size} expected {int(patch_size)}"
            )
    cached_edge_policy = meta.get("edge_policy")
    if (
        edge_policy is not None
        and cached_edge_policy != edge_policy
        and not (
            cached_edge_policy is None
            and patch_size is not None
            and int(patch_size) == 1
            and str(edge_policy) == EDGE_POLICY_DROP_PARTIAL
        )
    ):
        mismatches.append(f"edge_policy={cached_edge_policy} expected {edge_policy}")
    if layers is not None:
        cached_layers = meta.get("layers")
        expected_layers = [int(layer_id) for layer_id in layers]
        if allow_layer_subset:
            cached_layer_ids = (
                [int(layer_id) for layer_id in cached_layers]
                if isinstance(cached_layers, list)
                else None
            )
            if cached_layer_ids is None or any(
                layer_id not in cached_layer_ids for layer_id in expected_layers
            ):
                mismatches.append(
                    f"layers={meta.get('layers')} expected superset of {expected_layers}"
                )
        elif cached_layers != expected_layers:
            mismatches.append(f"layers={meta.get('layers')} expected {expected_layers}")
    if meta.get("supervision_grid_mode") != SUPERVISION_GRID_MODE:
        mismatches.append(
            "supervision_grid_mode="
            f"{meta.get('supervision_grid_mode')} expected {SUPERVISION_GRID_MODE}"
        )
    if mismatches:
        raise ValueError("Cache metadata mismatch: " + "; ".join(mismatches))


def resolve_cache_dir_for_prepare(
    base_dir: str,
    tile_size: int,
    cache_features: bool,
    model_name: str,
    layers: Sequence[int],
    patch_size: int | None = None,
    edge_policy: str | None = None,
    logger: Optional["VerbosityLogger"] = None,
) -> str:
    """Return the cache directory for prepare, creating it if needed.

    Args:
        base_dir (str): Base cache directory.
        tile_size (int): Tile size in pixels.
        cache_features (bool): Whether features are cached.
        model_name (str): Backbone model name.
        layers (Sequence[int]): Backbone layer indices.
        patch_size (int | None): Effective patch-size compatibility requirement
            for no-feature caches.
        edge_policy (str | None): Expected prepare-time edge policy for
            no-feature caches.
        logger (VerbosityLogger | None): Optional logger.

    Returns:
        str: Resolved cache directory.
    """

    expected_model_name = model_name if cache_features else None
    expected_layers = layers if cache_features else None
    expected_patch_size = None if cache_features else patch_size
    expected_edge_policy = None if cache_features else edge_policy
    meta = _load_cache_metadata(base_dir)
    if meta is not None:
        _validate_cache_metadata(
            meta,
            tile_size,
            cache_features,
            expected_model_name,
            expected_layers,
            patch_size=expected_patch_size,
            edge_policy=expected_edge_policy,
            allow_layer_subset=bool(cache_features),
        )
        return base_dir

    cache_dir = os.path.join(
        base_dir,
        _cache_subdir_name(
            tile_size,
            cache_features,
            patch_size=expected_patch_size,
            edge_policy=expected_edge_policy,
        ),
    )
    os.makedirs(cache_dir, exist_ok=True)
    meta = _load_cache_metadata(cache_dir)
    if meta is not None:
        _validate_cache_metadata(
            meta,
            tile_size,
            cache_features,
            expected_model_name,
            expected_layers,
            patch_size=expected_patch_size,
            edge_policy=expected_edge_policy,
            allow_layer_subset=bool(cache_features),
        )
    else:
        _write_cache_metadata(
            cache_dir,
            tile_size,
            cache_features,
            expected_model_name,
            expected_layers,
            expected_patch_size,
            expected_edge_policy,
        )
    if logger and glob.glob(os.path.join(base_dir, "*.pt")):
        logger.info(
            "Legacy cached tiles detected in %s; writing new tiles to %s."
            % (base_dir, cache_dir)
        )
    return cache_dir


def resolve_cache_dir_for_train(
    base_dir: str,
    tile_size: int | None,
    cache_features: bool | None,
    patch_size: int | None = None,
    edge_policy: str | None = None,
    logger: Optional["VerbosityLogger"] = None,
) -> str:
    """Return the cache directory for training/verification.

    Args:
        base_dir (str): Base cache directory.
        tile_size (int | None): Expected tile size.
        cache_features (bool | None): Expected cache_features setting.
        patch_size (int | None): Effective patch-size compatibility requirement
            for no-feature caches.
        edge_policy (str | None): Expected prepare-time edge policy for
            no-feature caches.
        logger (VerbosityLogger | None): Optional logger.

    Returns:
        str: Resolved cache directory.

    Raises:
        ValueError: If multiple matching cache directories are found.
    """

    expected_patch_size = None if cache_features else patch_size
    expected_edge_policy = None if cache_features else edge_policy
    meta = _load_cache_metadata(base_dir)
    if meta is not None:
        _validate_cache_metadata(
            meta,
            tile_size,
            cache_features,
            None,
            None,
            patch_size=expected_patch_size,
            edge_policy=expected_edge_policy,
        )
        return base_dir

    derived = None
    if tile_size is not None and cache_features is not None:
        derived = os.path.join(
            base_dir,
            _cache_subdir_name(
                tile_size,
                cache_features,
                patch_size=expected_patch_size,
                edge_policy=expected_edge_policy,
            ),
        )
        if os.path.exists(derived):
            meta = _load_cache_metadata(derived)
            if meta is not None:
                _validate_cache_metadata(
                    meta,
                    tile_size,
                    cache_features,
                    None,
                    None,
                    patch_size=expected_patch_size,
                    edge_policy=expected_edge_policy,
                )
            return derived

    cache_dirs = []
    if os.path.isdir(base_dir):
        for entry in os.scandir(base_dir):
            if not entry.is_dir():
                continue
            meta = _load_cache_metadata(entry.path)
            if meta is None:
                if (
                    not cache_features
                    and tile_size is not None
                    and expected_patch_size is not None
                    and int(expected_patch_size) == 1
                    and expected_edge_policy == EDGE_POLICY_DROP_PARTIAL
                    and entry.name == _cache_subdir_name(int(tile_size), False)
                    and glob.glob(os.path.join(entry.path, "*.pt"))
                ):
                    cache_dirs.append(entry.path)
                continue
            try:
                _validate_cache_metadata(
                    meta,
                    tile_size,
                    cache_features,
                    None,
                    None,
                    patch_size=expected_patch_size,
                    edge_policy=expected_edge_policy,
                )
            except ValueError:
                continue
            cache_dirs.append(entry.path)
    if len(cache_dirs) == 1:
        if logger:
            logger.info("Using cached tiles from %s." % cache_dirs[0])
        return cache_dirs[0]
    if len(cache_dirs) > 1:
        raise ValueError(
            "Multiple cached tile directories found; set prepare.tile_size "
            "or point processed_dir to a specific cache directory."
        )
    if glob.glob(os.path.join(base_dir, "*.pt")):
        if expected_patch_size is not None and int(expected_patch_size) > 1:
            raise ValueError(
                "Legacy no-feature cached tiles in %s are missing patch-size "
                "or edge-policy metadata required for patch_size=%s. Re-run "
                "prepare or point processed_dir to a compatible cache directory."
                % (base_dir, int(expected_patch_size))
            )
        return base_dir
    return derived or base_dir


def build_tile_payload_metadata(
    *,
    requested_tile_size: int,
    layout: TileGridLayout,
    patch_size: int,
    edge_policy: str = EDGE_POLICY_DROP_PARTIAL,
) -> dict[str, int | str]:
    """Build per-tile geometry metadata stored inside cached payloads.

    Args:
        requested_tile_size (int): Requested image tile size from config.
        layout (TileGridLayout): Effective image/label tile layout.
        patch_size (int): Backbone patch size.
        edge_policy (str): Prepare-time edge policy identifier.

    Returns:
        dict[str, int | str]: Geometry metadata for one cached tile.

    Examples:
        >>> meta = build_tile_payload_metadata(
        ...     requested_tile_size=512,
        ...     layout=TileGridLayout(480, 96, 0.2, 1.0, 5),
        ...     patch_size=16,
        ... )
        >>> (meta["image_tile_size"], meta["label_tile_size"], meta["patch_size"])
        (480, 96, 16)
    """

    return {
        "requested_tile_size": int(requested_tile_size),
        "image_tile_size": int(layout.image_tile_size),
        "label_tile_size": int(layout.label_tile_size),
        "scale_factor": int(layout.scale_factor),
        "patch_size": int(patch_size),
        "edge_policy": str(edge_policy),
        "supervision_grid_mode": SUPERVISION_GRID_MODE,
    }


def extract_multiscale_features(
    image_hw3: np.ndarray,
    model,
    processor,
    device: torch.device,
    layers: Sequence[int],
    ps: int = 14,
) -> List[torch.Tensor]:
    """
    Run DINO backbone once on a tile and slice hidden states into feature maps.

    Args:
        image_hw3 (np.ndarray): Image array in HWC format.
        model: DINO backbone model.
        processor: Image processor for the backbone.
        device (torch.device): Device for inference.
        layers (Sequence[int]): Backbone layer indices to extract.
        ps (int): Patch size for the backbone.

    Returns:
        List[torch.Tensor]: Feature maps per requested layer.

    >>> import types
    >>> class DummyBatch(dict):
    ...     def to(self, device):
    ...         return self
    >>> class DummyProcessor:
    ...     def __call__(self, images, return_tensors=None, do_resize=None, do_center_crop=None):
    ...         batch = DummyBatch()
    ...         batch["pixel_values"] = torch.randn(1, 3, 14, 14)
    ...         return batch
    >>> class DummyModel(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.config = types.SimpleNamespace(num_register_tokens=0)
    ...     def forward(self, **kwargs):
    ...         hidden = tuple(torch.randn(1, 197, 4) for _ in range(24))
    ...         return types.SimpleNamespace(hidden_states=hidden)
    >>> dummy_processor = DummyProcessor()
    >>> dummy_model = DummyModel()
    >>> feats = extract_multiscale_features(
    ...     np.random.rand(14, 14, 3).astype(np.float32),
    ...     dummy_model,
    ...     dummy_processor,
    ...     torch.device("cpu"),
    ...     layers=[0],
    ...     ps=14,
    ... )
    >>> len(feats)
    1
    """

    inputs = processor(
        images=image_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)
    R = getattr(model.config, "num_register_tokens", 0)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states
    _, _, Hproc, Wproc = inputs["pixel_values"].shape
    feature_maps = []
    for layer_idx in layers:
        layer_output = hidden_states[layer_idx]
        patch_tokens = layer_output[:, 1 + R :, :]
        Hp, Wp = Hproc // ps, Wproc // ps
        feats = patch_tokens.reshape(1, Hp, Wp, -1).permute(0, 3, 1, 2)
        feature_maps.append(feats.squeeze(0).cpu())
    return feature_maps


def process_image_tiles_no_features(
    img_path: str,
    label_path: str,
    output_dir: str,
    tile_size: int,
    patch_size: int = 16,
    tile_filter_cfg: dict[str, Any] | None = None,
    max_tiles: int | None = None,
    counter: Any | None = None,
    lock: Any | None = None,
    stop_event: Any | None = None,
) -> dict:
    """Process one image into label-grid-supervised tiles without DINO features.

    Args:
        img_path (str): Path to the input image.
        label_path (str): Path to the label raster.
        output_dir (str): Output directory for tiles.
        tile_size (int): Tile size in pixels.
        tile_filter_cfg (dict[str, Any] | None): Optional tile-label filter config.
        patch_size (int): Backbone patch size used to derive compatible tiles.
        max_tiles (int | None): Optional tile limit.
        counter (multiprocessing.Value | None): Shared tile counter.
        lock (multiprocessing.Lock | None): Shared lock for counter.
        stop_event (multiprocessing.Event | None): Shared stop flag.

    Returns:
        dict: Status and tile counts for the processed image.

    Examples:
        >>> process_image_tiles_no_features(  # doctest: +SKIP
        ...     "image.tif",
        ...     "labels.tif",
        ...     "cache",
        ...     tile_size=512,
        ... )
        {'status': 'ok', 'tiles_written': ...}
    """

    tile_filter = _normalize_tile_filter_cfg(tile_filter_cfg)
    os.makedirs(output_dir, exist_ok=True)
    if stop_event is not None and stop_event.is_set():
        return {"status": "stopped", "tiles_written": 0, "skipped_no_foreground": 0}
    try:
        layout = build_tile_grid_layout(
            img_path,
            label_path,
            requested_tile_size=tile_size,
            patch_size=patch_size,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error_type": "label_alignment_error",
            "error": str(exc),
        }
    tiles_written = 0
    skipped_no_foreground = 0
    tile_meta = build_tile_payload_metadata(
        requested_tile_size=tile_size,
        layout=layout,
        patch_size=patch_size,
        edge_policy=EDGE_POLICY_DROP_PARTIAL,
    )
    with rasterio.open(img_path) as src_img, rasterio.open(label_path) as src_lab:
        if src_img.crs != src_lab.crs:
            return {
                "status": "error",
                "error_type": "label_alignment_error",
                "error": "native label-grid supervision requires one shared CRS",
            }
        label_window = (
            from_bounds(
                *src_img.bounds,
                transform=src_lab.transform,
            )
            .round_offsets()
            .round_lengths()
        )
        row_positions = full_fit_window_positions(
            int(label_window.height),
            layout.label_tile_size,
        )
        col_positions = full_fit_window_positions(
            int(label_window.width),
            layout.label_tile_size,
        )
        if not row_positions or not col_positions:
            return {
                "status": "ok",
                "tiles_written": 0,
                "skipped_no_foreground": 0,
            }
        for row_off in row_positions:
            for col_off in col_positions:
                if stop_event is not None and stop_event.is_set():
                    return {
                        "status": "stopped",
                        "tiles_written": tiles_written,
                        "skipped_no_foreground": skipped_no_foreground,
                    }
                if max_tiles is not None and counter is not None:
                    if counter.value >= max_tiles:
                        if stop_event is not None:
                            stop_event.set()
                        return {
                            "status": "limit",
                            "tiles_written": tiles_written,
                            "skipped_no_foreground": skipped_no_foreground,
                        }
                tile_window = Window(
                    col_off=int(label_window.col_off) + int(col_off),
                    row_off=int(label_window.row_off) + int(row_off),
                    width=layout.label_tile_size,
                    height=layout.label_tile_size,
                )
                tile_bounds = window_bounds(tile_window, src_lab.transform)
                tile_name = (
                    f"{Path(img_path).stem}_y{int(tile_window.row_off)}_x"
                    f"{int(tile_window.col_off)}.pt"
                )
                save_path = os.path.join(output_dir, tile_name)
                if os.path.exists(save_path):
                    continue
                lbl_crop = src_lab.read(
                    1,
                    window=tile_window,
                    boundless=True,
                    fill_value=0,
                )
                try:
                    img_crop = read_image_tile_for_label_bounds(
                        img_path,
                        tile_bounds,
                        layout.image_tile_size,
                    )
                except Exception as exc:
                    return {
                        "status": "error",
                        "error_type": "image_read_error",
                        "error": str(exc),
                    }
                if img_crop.max() == 0:
                    continue
                if not _tile_passes_label_filter(lbl_crop, tile_filter):
                    skipped_no_foreground += 1
                    continue
                if np.isnan(img_crop).any():
                    img_crop = np.nan_to_num(img_crop)
                if max_tiles is not None and counter is not None and lock is not None:
                    with lock:
                        if counter.value >= max_tiles:
                            if stop_event is not None:
                                stop_event.set()
                            return {
                                "status": "limit",
                                "tiles_written": tiles_written,
                                "skipped_no_foreground": skipped_no_foreground,
                            }
                        counter.value += 1
                payload = {
                    "image": torch.from_numpy(img_crop),
                    "features": [],
                    "label": lbl_crop,
                    "tile_meta": tile_meta,
                }
                try:
                    wrote_tile = _write_tile_payload_atomic(payload, save_path)
                except Exception as exc:
                    return {
                        "status": "error",
                        "error_type": "tile_write_error",
                        "error": str(exc),
                    }
                if not wrote_tile:
                    continue
                tiles_written += 1
    return {
        "status": "ok",
        "tiles_written": tiles_written,
        "skipped_no_foreground": skipped_no_foreground,
    }


def subset_label_to_image_bounds(img_path: str, lab_path: str) -> np.ndarray:
    """Read the native label-grid subset covering one image footprint.

    Args:
        img_path (str): Path to the input image.
        lab_path (str): Path to the label raster.

    Returns:
        np.ndarray: Aligned label array.

    >>> subset_label_to_image_bounds("image.tif", "labels.tif")  # doctest: +SKIP
    array(...)
    """

    labels_aligned, _ = read_label_window_for_image_bounds(img_path, lab_path)
    return labels_aligned
