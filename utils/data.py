"""
Data-handling utilities: tiling GeoTIFFs, caching features, validation, and
dataset loader.
"""

from __future__ import annotations

import concurrent.futures
import gc
import glob
import json
import multiprocessing
import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import reproject
from shapely.geometry import box
from tifffile import imread
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

if TYPE_CHECKING:
    from utils.logging import VerbosityLogger


CACHE_META_FILENAME = "cache_meta.json"
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


def _cache_subdir_name(tile_size: int, cache_features: bool) -> str:
    """Build a cache subdirectory name for tile size and feature mode.

    Args:
        tile_size (int): Tile size in pixels.
        cache_features (bool): Whether features are cached.

    Returns:
        str: Subdirectory name.

    Examples:
        >>> _cache_subdir_name(512, True)
        'tiles_512_feat'
        >>> _cache_subdir_name(1024, False)
        'tiles_1024_nofeat'
    """

    suffix = "feat" if cache_features else "nofeat"
    return f"tiles_{tile_size}_{suffix}"


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
) -> None:
    """Write cache metadata for a tile directory.

    Args:
        cache_dir (str): Cache directory path.
        tile_size (int): Tile size in pixels.
        cache_features (bool): Whether features are cached.
        model_name (str | None): Backbone model name.
        layers (Sequence[int] | None): Backbone layer indices.

    Returns:
        None: Metadata is written to disk.
    """

    meta = {
        "tile_size": tile_size,
        "cache_features": cache_features,
        "model_name": model_name,
        "layers": list(layers) if layers is not None else None,
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
) -> None:
    """Validate cache metadata against expected settings.

    Args:
        meta (dict[str, Any]): Metadata loaded from cache.
        tile_size (int | None): Expected tile size.
        cache_features (bool | None): Expected cache_features setting.
        model_name (str | None): Expected model name.
        layers (Sequence[int] | None): Expected backbone layers.

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
    if layers is not None and meta.get("layers") != list(layers):
        mismatches.append(f"layers={meta.get('layers')} expected {list(layers)}")
    if mismatches:
        raise ValueError("Cache metadata mismatch: " + "; ".join(mismatches))


def resolve_cache_dir_for_prepare(
    base_dir: str,
    tile_size: int,
    cache_features: bool,
    model_name: str,
    layers: Sequence[int],
    logger: Optional["VerbosityLogger"] = None,
) -> str:
    """Return the cache directory for prepare, creating it if needed.

    Args:
        base_dir (str): Base cache directory.
        tile_size (int): Tile size in pixels.
        cache_features (bool): Whether features are cached.
        model_name (str): Backbone model name.
        layers (Sequence[int]): Backbone layer indices.
        logger (VerbosityLogger | None): Optional logger.

    Returns:
        str: Resolved cache directory.
    """

    meta = _load_cache_metadata(base_dir)
    if meta is not None:
        _validate_cache_metadata(meta, tile_size, cache_features, model_name, layers)
        return base_dir

    cache_dir = os.path.join(base_dir, _cache_subdir_name(tile_size, cache_features))
    os.makedirs(cache_dir, exist_ok=True)
    meta = _load_cache_metadata(cache_dir)
    if meta is not None:
        _validate_cache_metadata(meta, tile_size, cache_features, model_name, layers)
    else:
        _write_cache_metadata(cache_dir, tile_size, cache_features, model_name, layers)
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
    logger: Optional["VerbosityLogger"] = None,
) -> str:
    """Return the cache directory for training/verification.

    Args:
        base_dir (str): Base cache directory.
        tile_size (int | None): Expected tile size.
        cache_features (bool | None): Expected cache_features setting.
        logger (VerbosityLogger | None): Optional logger.

    Returns:
        str: Resolved cache directory.

    Raises:
        ValueError: If multiple matching cache directories are found.
    """

    meta = _load_cache_metadata(base_dir)
    if meta is not None:
        _validate_cache_metadata(meta, tile_size, cache_features, None, None)
        return base_dir

    derived = None
    if tile_size is not None and cache_features is not None:
        derived = os.path.join(base_dir, _cache_subdir_name(tile_size, cache_features))
        if os.path.exists(derived):
            meta = _load_cache_metadata(derived)
            if meta is not None:
                _validate_cache_metadata(meta, tile_size, cache_features, None, None)
            return derived

    cache_dirs = []
    if os.path.isdir(base_dir):
        for entry in os.scandir(base_dir):
            if not entry.is_dir():
                continue
            meta = _load_cache_metadata(entry.path)
            if meta is None:
                continue
            if tile_size is not None and meta.get("tile_size") != tile_size:
                continue
            if (
                cache_features is not None
                and meta.get("cache_features") != cache_features
            ):
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
        return base_dir
    return derived or base_dir


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
    tile_filter_cfg: dict[str, Any] | None = None,
    max_tiles: int | None = None,
    counter: Any | None = None,
    lock: Any | None = None,
    stop_event: Any | None = None,
) -> dict:
    """Process one image into tiles without DINO features.

    Args:
        img_path (str): Path to the input image.
        label_path (str): Path to the label raster.
        output_dir (str): Output directory for tiles.
        tile_size (int): Tile size in pixels.
        tile_filter_cfg (dict[str, Any] | None): Optional tile-label filter config.
        max_tiles (int | None): Optional tile limit.
        counter (multiprocessing.Value | None): Shared tile counter.
        lock (multiprocessing.Lock | None): Shared lock for counter.
        stop_event (multiprocessing.Event | None): Shared stop flag.

    Returns:
        dict: Status and tile counts for the processed image.
    """

    tile_filter = _normalize_tile_filter_cfg(tile_filter_cfg)
    if stop_event is not None and stop_event.is_set():
        return {"status": "stopped", "tiles_written": 0, "skipped_no_foreground": 0}
    try:
        full_img = imread(img_path)
        full_label = subset_label_to_image_bounds(img_path, label_path)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    h, w, _ = full_img.shape
    tiles_written = 0
    skipped_no_foreground = 0
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
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
            y_min, x_min = y, x
            y_max, x_max = y + tile_size, x + tile_size
            if y_max > h:
                y_min, y_max = h - tile_size, h
            if x_max > w:
                x_min, x_max = w - tile_size, w
            tile_name = f"{Path(img_path).stem}_y{y_min}_x{x_min}.pt"
            save_path = os.path.join(output_dir, tile_name)
            if os.path.exists(save_path):
                continue
            img_crop = full_img[y_min:y_max, x_min:x_max, :]
            lbl_crop = full_label[y_min:y_max, x_min:x_max]
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
            }
            temp_path = save_path + ".tmp"
            torch.save(payload, temp_path)
            os.rename(temp_path, save_path)
            tiles_written += 1
    return {
        "status": "ok",
        "tiles_written": tiles_written,
        "skipped_no_foreground": skipped_no_foreground,
    }


def subset_label_to_image_bounds(img_path: str, lab_path: str) -> np.ndarray:
    """
    Crop or reproject the label raster so it aligns with the image tile.

    Args:
        img_path (str): Path to the input image.
        lab_path (str): Path to the label raster.

    Returns:
        np.ndarray: Aligned label array.

    >>> subset_label_to_image_bounds("image.tif", "labels.tif")  # doctest: +SKIP
    array(...)
    """

    with rasterio.open(img_path) as src_img:
        img_bounds = src_img.bounds
        img_meta = src_img.meta.copy()
        img_crs = src_img.crs
        H, W = src_img.shape
    with rasterio.open(lab_path) as src_lab:
        if src_lab.crs == img_crs:
            geom = [box(*img_bounds).__geo_interface__]
            out_image, _ = mask(src_lab, geom, crop=True)
            if out_image.shape[1] != H or out_image.shape[2] != W:
                t_lbl = torch.from_numpy(out_image).float().unsqueeze(0)
                t_lbl = F.interpolate(t_lbl, size=(H, W), mode="nearest")
                labels_aligned = t_lbl.squeeze(0).squeeze(0).numpy()
            else:
                labels_aligned = out_image[0]
        else:
            new_meta = img_meta.copy()
            new_meta.update(dtype=src_lab.dtypes[0], count=1)
            with MemoryFile() as mem:
                with mem.open(**new_meta) as dst:
                    reproject(
                        source=rasterio.band(src_lab, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src_lab.transform,
                        src_crs=src_lab.crs,
                        dst_transform=img_meta["transform"],
                        dst_crs=img_crs,
                        dst_width=img_meta["width"],
                        dst_height=img_meta["height"],
                        resampling=Resampling.nearest,
                    )
                    labels_aligned = dst.read(1)
    return labels_aligned


def _check_single_file(file_path: str) -> str | None:
    """
    Validate that a cached tile can be read.

    Args:
        file_path (str): Path to the cached tile.

    Returns:
        str | None: Path of the corrupt file, if any.

    >>> import tempfile
    >>> tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    >>> torch.save({"x": torch.tensor([1])}, tmp.name)
    >>> _check_single_file(tmp.name)
    """

    try:
        torch.load(file_path, weights_only=False, map_location="cpu")
        return None
    except Exception:
        return file_path


def verify_tile_semantics(
    file_path: str,
    validation_cfg: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Validate readability + semantic integrity for one cached tile file.

    Args:
        file_path (str): Path to tile file.
        validation_cfg (Optional[dict[str, Any]]): Dataset validation config.

    Returns:
        dict[str, Any]: Semantic validation result summary.
    """

    cfg = _normalize_dataset_validation_cfg(validation_cfg)
    result = {
        "path": file_path,
        "valid": True,
        "read_error": False,
        "nonfinite_image": False,
        "nonfinite_features": False,
        "bad_labels": False,
        "invalid_label_values": [],
    }
    try:
        try:
            data = torch.load(file_path, weights_only=False, map_location="cpu")
        except TypeError:
            data = torch.load(file_path, map_location="cpu")
    except Exception:
        result["valid"] = False
        result["read_error"] = True
        return result
    if not cfg["enabled"]:
        return result
    image = data.get("image")
    if image is None:
        result["valid"] = False
        result["nonfinite_image"] = True
        return result
    image_t = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    image_t = image_t.float()
    if cfg["require_finite_images"] and not torch.isfinite(image_t).all():
        result["valid"] = False
        result["nonfinite_image"] = True
    features = data.get("features", [])
    if cfg["require_finite_features"]:
        for feat in features:
            feat_t = feat if isinstance(feat, torch.Tensor) else torch.as_tensor(feat)
            if not torch.isfinite(feat_t.float()).all():
                result["valid"] = False
                result["nonfinite_features"] = True
                break
    label = data.get("label")
    if label is None:
        result["valid"] = False
        result["bad_labels"] = True
        return result
    label_t = label if isinstance(label, torch.Tensor) else torch.as_tensor(label)
    if not torch.isfinite(label_t.float()).all():
        result["valid"] = False
        result["bad_labels"] = True
        return result
    label_t = label_t.long()
    valid_mask = _label_validity_mask(label_t, cfg["allowed_labels"])
    if not valid_mask.all():
        result["valid"] = False
        result["bad_labels"] = True
        result["invalid_label_values"] = torch.unique(label_t[~valid_mask]).tolist()
    return result


def verify_and_clean_dataset_fast(
    output_dir: str,
    num_workers: int | None = None,
    logger: Optional["VerbosityLogger"] = None,
    validation_cfg: Optional[dict[str, Any]] = None,
) -> dict[str, int]:
    """
    Spawn workers to make sure each cached tile is readable; delete corrupt ones.

    Args:
        output_dir (str): Directory containing cached tiles.
        num_workers (int | None): Worker count for verification.
        logger (Optional["VerbosityLogger"]): Logger instance.
        validation_cfg (Optional[dict[str, Any]]): Dataset validation options.

    >>> summary = verify_and_clean_dataset_fast("/tmp", num_workers=1)  # doctest: +SKIP
    >>> isinstance(summary, dict)  # doctest: +SKIP
    True
    """

    cfg = _normalize_dataset_validation_cfg(validation_cfg)
    summary = {
        "tiles_total": 0,
        "tiles_removed": 0,
        "tiles_corrupt": 0,
        "tiles_nonfinite": 0,
        "tiles_bad_labels": 0,
    }
    files = glob.glob(os.path.join(output_dir, "*.pt"))
    if not files:
        if logger:
            logger.info("No cached tiles found for verification.")
        return summary
    summary["tiles_total"] = len(files)
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    invalid_results: list[dict[str, Any]] = []
    if logger:
        logger.info(
            f"Verifying {len(files)} cached tiles"
            f"{' with semantic checks' if cfg['enabled'] else ''}."
        )
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(verify_tile_semantics, f, cfg) for f in files]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(files), desc="Verifying"
        ):
            result = future.result()
            if not result["valid"]:
                invalid_results.append(result)
    for result in invalid_results:
        f = result["path"]
        reason = "corrupt"
        if result.get("nonfinite_image") or result.get("nonfinite_features"):
            reason = "non-finite values"
        elif result.get("bad_labels"):
            reason = "invalid labels"
        if result.get("read_error"):
            summary["tiles_corrupt"] += 1
        if result.get("nonfinite_image") or result.get("nonfinite_features"):
            summary["tiles_nonfinite"] += 1
        if result.get("bad_labels"):
            summary["tiles_bad_labels"] += 1
        try:
            os.remove(f)
            summary["tiles_removed"] += 1
            if logger:
                logger.error(f"Removed tile ({reason}) {f}")
        except OSError:
            if logger:
                logger.error(f"Failed to remove tile {f}")
    return summary


def prepare_data_tiles(
    img_dir: str,
    label_path: str,
    output_dir: str,
    model_name: str,
    layers: Sequence[int],
    device: torch.device,
    tile_size: int = 512,
    cache_features: bool = True,
    tile_filter_cfg: Optional[dict[str, Any]] = None,
    workers: int | None = None,
    max_tiles: int | None = None,
    logger: Optional["VerbosityLogger"] = None,
) -> None:
    """
    Tile raw GeoTIFFs, align labels, and pre-compute DINO feature tensors.

    Args:
        img_dir (str): Directory of input imagery.
        label_path (str): Label raster path.
        output_dir (str): Output directory for cached tiles.
        model_name (str): Backbone model name.
        layers (Sequence[int]): Backbone layer indices to extract.
        device (torch.device): Device for inference.
        tile_size (int): Tile size in pixels.
        cache_features (bool): Whether to cache DINO features on disk.
        tile_filter_cfg (Optional[dict[str, Any]]): Optional tile-label filter config.
        workers (int | None): Number of worker processes for tiling.
        max_tiles (int | None): Optional tile limit for preparation.
        logger (Optional["VerbosityLogger"]): Logger instance.

    >>> # Light-touch doctest ensures function signature works by calling with
    >>> # a fake directory (no images). Should exit early with no errors.
    >>> import tempfile
    >>> tmp_imgs = tempfile.mkdtemp()
    >>> tmp_out = tempfile.mkdtemp()
    >>> prepare_data_tiles(
    ...     img_dir=tmp_imgs,
    ...     label_path="/tmp/nonexistent.tif",
    ...     output_dir=tmp_out,
    ...     model_name="facebook/dinov3-vitl16-pretrain-sat493m",
    ...     layers=[5],
    ...     device=torch.device("cpu"),
    ... )  # doctest: +SKIP
    """

    def _log_info(message: str) -> None:
        """Emit an info message to the logger or stdout.

        Args:
            message (str): Message text to emit.
        """

        if logger:
            logger.info(message)
        else:
            print(message)

    def _log_debug(message: str) -> None:
        """Emit a debug message to the logger when enabled.

        Args:
            message (str): Message text to emit.
        """

        if logger:
            logger.debug(message)

    def _format_eta(seconds: float) -> str:
        """Format seconds as HH:MM:SS.

        Args:
            seconds (float): Remaining seconds estimate.

        Returns:
            str: Formatted ETA string.

        Examples:
            >>> _format_eta(65.2)
            '00:01:05'
        """

        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    _log_info("--- PHASE 1: TILING & PRE-COMPUTING ---")
    os.makedirs(output_dir, exist_ok=True)
    existing = glob.glob(os.path.join(output_dir, "*.pt"))
    if existing:
        _log_info(f"[INFO] Found {len(existing)} existing tiles.")
    if max_tiles is not None and max_tiles <= 0:
        max_tiles = None
    tile_filter = _normalize_tile_filter_cfg(tile_filter_cfg)
    if tile_filter["enabled"]:
        _log_info(
            "Tile label filter enabled: mode=%s foreground_labels=%s"
            % (tile_filter["mode"], list(tile_filter["foreground_labels"]))
        )
    count_existing = cache_features
    if max_tiles is not None and count_existing and existing:
        if len(existing) >= max_tiles:
            _log_info("Max tiles already satisfied by existing cache. Skipping tiling.")
            return
    if workers is None:
        workers = 1
    if cache_features and workers > 1:
        _log_info("cache_features enabled; using a single worker for tiling.")
        workers = 1
    processor = None
    model = None
    if cache_features:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).eval().to(device)
    image_paths = glob.glob(os.path.join(img_dir, "*.tif"))
    if max_tiles is not None:
        random.shuffle(image_paths)
    ps = 14 if "vitl14" in model_name else 16
    total_images = len(image_paths)
    start_time = time.time()
    tiles_written = len(existing) if count_existing else 0
    if workers > 1 and not cache_features:
        counter = None
        lock = None
        stop_event = None
        manager = None
        if max_tiles is not None:
            manager = multiprocessing.Manager()
            counter = manager.Value("i", tiles_written)
            lock = manager.Lock()
            stop_event = manager.Event()

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        submit_idx = 0
        completed = 0
        cancelled = 0
        errors = 0
        limit_hits = 0
        skipped_no_foreground = 0
        image_durations: list[float] = []
        shutdown_wait = 0.0
        limit_detected_at: float | None = None
        max_pending = max(workers * 2, workers)
        iterator = iter(image_paths)
        pending: dict[concurrent.futures.Future, tuple[str, float]] = {}

        def _submit_next() -> bool:
            """Submit one image tiling job.

            Returns:
                bool: ``True`` when a task was submitted, ``False`` if exhausted.
            """
            nonlocal submit_idx
            try:
                next_img = next(iterator)
            except StopIteration:
                return False
            submit_idx += 1
            future = executor.submit(
                process_image_tiles_no_features,
                next_img,
                label_path,
                output_dir,
                tile_size,
                tile_filter,
                max_tiles,
                counter,
                lock,
                stop_event,
            )
            pending[future] = (next_img, time.time())
            return True

        try:
            while len(pending) < max_pending and _submit_next():
                pass

            while pending:
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    img_path, submitted_at = pending.pop(future)
                    completed += 1
                    image_durations.append(time.time() - submitted_at)
                    basename = os.path.splitext(os.path.basename(img_path))[0]
                    try:
                        result = future.result()
                    except Exception as exc:
                        errors += 1
                        _log_debug(f"Skipping corrupted image {basename}: {exc}")
                        continue
                    skipped_no_foreground += int(result.get("skipped_no_foreground", 0))
                    status = result.get("status")
                    if status == "limit":
                        limit_hits += 1
                        if limit_detected_at is None:
                            limit_detected_at = time.time()
                            _log_info(
                                "Reached max tiles during multiprocessing. "
                                "Stopping new submissions."
                            )
                        if stop_event is not None:
                            stop_event.set()
                        continue
                    if status == "stopped":
                        continue
                    if status in {"error"}:
                        errors += 1
                        _log_debug(
                            f"Skipping corrupted image {basename}: {result.get('error')}."
                        )
                        continue
                    elapsed = time.time() - start_time
                    eta = _format_eta(
                        (elapsed / max(1, completed)) * (total_images - completed)
                    )
                    _log_info(
                        f"Processing image {completed}/{total_images} (ETA {eta}): {basename}"
                    )

                if limit_detected_at is not None:
                    for queued_future in list(pending):
                        if queued_future.cancel():
                            cancelled += 1
                            pending.pop(queued_future, None)
                    continue

                while len(pending) < max_pending and _submit_next():
                    pass
        finally:
            shutdown_start = time.time()
            executor.shutdown(wait=True, cancel_futures=True)
            shutdown_wait = time.time() - shutdown_start
            if manager is not None:
                manager.shutdown()

        avg_image_time = (
            (sum(image_durations) / len(image_durations)) if image_durations else 0.0
        )
        drain_wait = (
            (time.time() - limit_detected_at) if limit_detected_at is not None else 0.0
        )
        _log_info(
            "Multiprocessing tiling summary :: submitted=%d completed=%d "
            "cancelled=%d errors=%d limit_hits=%d avg_image_time=%.2fs "
            "drain_wait=%.2fs executor_shutdown_wait=%.2fs "
            "skipped_no_foreground=%d"
            % (
                submit_idx,
                completed,
                cancelled,
                errors,
                limit_hits,
                avg_image_time,
                drain_wait,
                shutdown_wait,
                skipped_no_foreground,
            )
        )
        return

    for idx, img_path in enumerate(
        tqdm(image_paths, desc="Processing Large Images"), start=1
    ):
        if max_tiles is not None and tiles_written >= max_tiles:
            _log_info("Reached max tiles. Stopping tiling.")
            return
        basename = os.path.splitext(os.path.basename(img_path))[0]
        elapsed = time.time() - start_time
        eta = _format_eta((elapsed / max(1, idx)) * (total_images - idx))
        _log_info(f"Processing image {idx}/{total_images} (ETA {eta}): {basename}")
        torch.cuda.empty_cache()
        gc.collect()
        try:
            full_img = imread(img_path)
            full_label = subset_label_to_image_bounds(img_path, label_path)
        except Exception:
            _log_debug(f"Skipping corrupted image {basename}.")
            continue
        H, W, _ = full_img.shape
        skipped_no_foreground = 0
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                if max_tiles is not None and tiles_written >= max_tiles:
                    _log_info("Reached max tiles. Stopping tiling.")
                    return
                y_min, x_min = y, x
                y_max, x_max = y + tile_size, x + tile_size
                if y_max > H:
                    y_min, y_max = H - tile_size, H
                if x_max > W:
                    x_min, x_max = W - tile_size, W
                tile_name = f"{basename}_y{y_min}_x{x_min}.pt"
                save_path = os.path.join(output_dir, tile_name)
                if os.path.exists(save_path):
                    _log_debug(f"Tile already exists: {tile_name}")
                    continue
                img_crop = full_img[y_min:y_max, x_min:x_max, :]
                lbl_crop = full_label[y_min:y_max, x_min:x_max]
                if img_crop.max() == 0:
                    _log_debug(f"Skipping zero tile {tile_name}")
                    continue
                if not _tile_passes_label_filter(lbl_crop, tile_filter):
                    skipped_no_foreground += 1
                    _log_debug(f"Skipping no-foreground tile {tile_name}")
                    continue
                if np.isnan(img_crop).any():
                    img_crop = np.nan_to_num(img_crop)
                    _log_debug(f"NaNs detected and replaced for tile {tile_name}")
                temp_path: str | None = None
                try:
                    feats = []
                    if cache_features:
                        feats = extract_multiscale_features(
                            img_crop,
                            model,
                            processor,
                            device,
                            layers,
                            ps=ps,
                        )
                    payload = {
                        "image": torch.from_numpy(img_crop),
                        "features": [f.cpu() for f in feats] if feats else [],
                        "label": lbl_crop,
                    }
                    temp_path = save_path + ".tmp"
                    torch.save(payload, temp_path)
                    os.rename(temp_path, save_path)
                    del feats, payload, img_crop, lbl_crop
                    tiles_written += 1
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        del img_crop
                        torch.cuda.empty_cache()
                        gc.collect()
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)
                        continue
                    raise e
        if tile_filter["enabled"] and skipped_no_foreground > 0:
            _log_info(
                "Image %s: skipped_no_foreground=%d" % (basename, skipped_no_foreground)
            )
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    _log_info("Phase 1 Complete.")


class PrecomputedDataset(Dataset):
    """
    Lazy dataset that loads cached tiles on demand.
    """

    def __init__(
        self,
        processed_dir: str,
        augmentation_cfg: Optional[dict] = None,
        file_subset: Optional[List[str]] = None,
        validation_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Index every cached tile path.

        Args:
            processed_dir (str): Directory containing cached tiles.
            augmentation_cfg (Optional[dict]): Augmentation configuration.
            file_subset (Optional[List[str]]): Optional subset of files.
            validation_cfg (Optional[dict[str, Any]]): Validation policy settings.

        >>> import tempfile
        >>> tmpdir = tempfile.mkdtemp()
        >>> sample = os.path.join(tmpdir, "sample.pt")
        >>> torch.save(
        ...     {
        ...         "image": torch.zeros(4, 4, 3),
        ...         "features": [torch.zeros(1, 1, 1)],
        ...         "label": np.zeros((4, 4)),
        ...     },
        ...     sample,
        ... )
        >>> ds = PrecomputedDataset(tmpdir)
        >>> len(ds)
        1
        """

        if file_subset is not None:
            self.processed_files = file_subset
        else:
            self.processed_files = sorted(
                glob.glob(os.path.join(processed_dir, "*.pt"))
            )
        if not self.processed_files:
            raise ValueError(f"No .pt files found in {processed_dir}.")
        self.augmentation_cfg = augmentation_cfg or {}
        self.validation_cfg = _normalize_dataset_validation_cfg(validation_cfg)

    def __len__(self) -> int:
        """
        Number of cached tiles.

        Returns:
            int: Number of cached tiles.

        >>> ds = PrecomputedDataset.__new__(PrecomputedDataset)
        >>> ds.processed_files = [1, 2, 3]
        >>> len(ds)
        3
        """

        return len(self.processed_files)

    def __getitem__(self, idx: int):
        """
        Load the tile, normalize RGB image, and return label tensor.

        Args:
            idx (int): Index of the tile to load.

        Returns:
            tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]: Image, features, and label.

        >>> import tempfile
        >>> tmpdir = tempfile.mkdtemp()
        >>> sample = os.path.join(tmpdir, "sample.pt")
        >>> torch.save({
        ...     "image": torch.zeros(4, 4, 3),
        ...     "features": [torch.zeros(3, 3, 3) for _ in range(4)],
        ...     "label": np.zeros((4, 4)),
        ... }, sample)
        >>> ds = PrecomputedDataset(tmpdir)
        >>> img, feats, label = ds[0]
        >>> img.shape[0]
        3
        """

        try:
            data = torch.load(self.processed_files[idx], weights_only=False)
        except TypeError:
            data = torch.load(self.processed_files[idx])
        image_raw = data["image"]
        image_tensor = (
            image_raw
            if isinstance(image_raw, torch.Tensor)
            else torch.as_tensor(image_raw)
        )
        img = image_tensor.permute(2, 0, 1).float() / 255.0
        features = [
            feat if isinstance(feat, torch.Tensor) else torch.as_tensor(feat)
            for feat in data.get("features", [])
        ]
        label_raw = data["label"]
        if isinstance(label_raw, torch.Tensor):
            label_seg = label_raw.long()
        else:
            label_seg = torch.from_numpy(np.asarray(label_raw).astype(np.int64)).long()
        img, features, label_seg = self._validate_sample(
            img, features, label_seg, self.processed_files[idx]
        )
        img, features, label_seg = self._apply_augmentations(img, features, label_seg)
        return img, features, label_seg

    def _validate_sample(
        self,
        img: torch.Tensor,
        features: List[torch.Tensor],
        label: torch.Tensor,
        source: str,
    ) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Validate finite values and label ranges for one sample.

        Args:
            img (torch.Tensor): Image tensor.
            features (List[torch.Tensor]): Feature tensors.
            label (torch.Tensor): Label tensor.
            source (str): Source tile path.

        Returns:
            tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]: Validated tensors.
        """

        cfg = self.validation_cfg
        if not cfg.get("enabled", True):
            return img, features, label
        if cfg["require_finite_images"] and not torch.isfinite(img).all():
            raise ValueError(f"Non-finite image values in {source}")
        if cfg["require_finite_features"]:
            for idx, feat in enumerate(features):
                if not torch.isfinite(feat.float()).all():
                    raise ValueError(
                        f"Non-finite feature values in {source} (feature {idx})"
                    )
        if not torch.isfinite(label.float()).all():
            raise ValueError(f"Non-finite label values in {source}")
        label = _sanitize_label_tensor(label, cfg, source)
        return img, features, label

    def _apply_augmentations(
        self,
        img: torch.Tensor,
        features: List[torch.Tensor],
        label: torch.Tensor,
    ) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Apply optional augmentations to image, features, and label.

        Args:
            img (torch.Tensor): Image tensor.
            features (List[torch.Tensor]): Feature tensors.
            label (torch.Tensor): Label tensor.

        Returns:
            tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]: Augmented outputs.
        """

        cfg = self.augmentation_cfg
        if not cfg or not cfg.get("enable", False):
            return img, features, label
        feats = [f.clone() for f in features]
        # Random rotation (multiples of 90 degrees)
        if cfg.get("rotate90", False):
            k = random.randint(0, 3)
            if k:
                img = torch.rot90(img, k, dims=(1, 2))
                label = torch.rot90(label, k, dims=(0, 1))
                feats = [torch.rot90(f, k, dims=(1, 2)) for f in feats]
        # Horizontal flip
        if cfg.get("hflip", False) and random.random() < 0.5:
            img = torch.flip(img, dims=(2,))
            label = torch.flip(label, dims=(1,))
            feats = [torch.flip(f, dims=(2,)) for f in feats]
        # Vertical flip
        if cfg.get("vflip", False) and random.random() < 0.5:
            img = torch.flip(img, dims=(1,))
            label = torch.flip(label, dims=(0,))
            feats = [torch.flip(f, dims=(1,)) for f in feats]
        allow_feature_mismatch = bool(cfg.get("allow_feature_mismatch", False))
        has_cached_features = len(feats) > 0
        if allow_feature_mismatch or not has_cached_features:
            img = _apply_color_jitter(img, cfg.get("color_jitter", {}))
            img = _apply_cutout(img, cfg.get("cutout", {}))
            img = _apply_gridmask(img, cfg.get("gridmask", {}))
        return img, feats, label
