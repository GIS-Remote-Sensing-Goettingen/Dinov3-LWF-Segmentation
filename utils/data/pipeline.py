"""Tile verification, preparation pipeline, and dataset class."""

from __future__ import annotations

import concurrent.futures
import gc
import glob
import multiprocessing
import os
import random
import time
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from .core import (
    _apply_color_jitter,
    _apply_cutout,
    _apply_gridmask,
    _label_validity_mask,
    _normalize_dataset_validation_cfg,
    _normalize_tile_filter_cfg,
    _sanitize_label_tensor,
    _tile_passes_label_filter,
    _write_tile_payload_atomic,
    build_tile_grid_layout,
    extract_multiscale_features,
    process_image_tiles_no_features,
    read_image_tile_for_label_bounds,
)

if TYPE_CHECKING:
    from utils.logging import VerbosityLogger


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
                ps,
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
                        _log_debug(f"Prepare worker failed for {basename}: {exc}")
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
                        error_type = str(result.get("error_type", "prepare_error"))
                        error_text = str(result.get("error", "unknown prepare error"))
                        if error_type == "image_read_error":
                            _log_debug(
                                f"Skipping unreadable image {basename}: {error_text}"
                            )
                        elif error_type == "label_alignment_error":
                            _log_debug(
                                "Skipping label alignment failure for %s: %s"
                                % (basename, error_text)
                            )
                        elif error_type == "tile_write_error":
                            _log_debug(
                                "Stopping prepare for %s due to tile write error: %s"
                                % (basename, error_text)
                            )
                        else:
                            _log_debug(f"Prepare failed for {basename}: {error_text}")
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
            layout = build_tile_grid_layout(
                img_path,
                label_path,
                requested_tile_size=tile_size,
                patch_size=ps,
            )
        except Exception as exc:
            _log_debug(f"Skipping label alignment failure for {basename}: {exc}")
            continue
        skipped_no_foreground = 0
        with rasterio.open(img_path) as src_img, rasterio.open(label_path) as src_lab:
            if src_img.crs != src_lab.crs:
                _log_debug(
                    f"Skipping label alignment failure for {basename}: "
                    "native label-grid supervision requires one shared CRS"
                )
                continue
            label_window = (
                rasterio.windows.from_bounds(
                    *src_img.bounds,
                    transform=src_lab.transform,
                )
                .round_offsets()
                .round_lengths()
            )
            row_positions = range(
                0,
                max(int(label_window.height), 1),
                layout.label_tile_size,
            )
            col_positions = range(
                0,
                max(int(label_window.width), 1),
                layout.label_tile_size,
            )
            row_positions = list(row_positions) or [0]
            col_positions = list(col_positions) or [0]
            if row_positions[-1] != max(
                int(label_window.height) - layout.label_tile_size, 0
            ):
                row_positions[-1] = max(
                    int(label_window.height) - layout.label_tile_size, 0
                )
            if col_positions[-1] != max(
                int(label_window.width) - layout.label_tile_size, 0
            ):
                col_positions[-1] = max(
                    int(label_window.width) - layout.label_tile_size, 0
                )
            row_positions = list(dict.fromkeys(row_positions))
            col_positions = list(dict.fromkeys(col_positions))
            for row_off in row_positions:
                for col_off in col_positions:
                    if max_tiles is not None and tiles_written >= max_tiles:
                        _log_info("Reached max tiles. Stopping tiling.")
                        return
                    tile_window = rasterio.windows.Window(
                        col_off=int(label_window.col_off) + int(col_off),
                        row_off=int(label_window.row_off) + int(row_off),
                        width=layout.label_tile_size,
                        height=layout.label_tile_size,
                    )
                    tile_bounds = rasterio.windows.bounds(
                        tile_window, src_lab.transform
                    )
                    tile_name = (
                        f"{basename}_y{int(tile_window.row_off)}_x"
                        f"{int(tile_window.col_off)}.pt"
                    )
                    save_path = os.path.join(output_dir, tile_name)
                    if os.path.exists(save_path):
                        _log_debug(f"Tile already exists: {tile_name}")
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
                        _log_debug(f"Skipping unreadable image tile {tile_name}: {exc}")
                        continue
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
                        wrote_tile = _write_tile_payload_atomic(payload, save_path)
                        if not wrote_tile:
                            del feats, payload, img_crop, lbl_crop
                            _log_debug(f"Tile already exists after race: {tile_name}")
                            continue
                        del feats, payload, img_crop, lbl_crop
                        tiles_written += 1
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            del img_crop
                            torch.cuda.empty_cache()
                            gc.collect()
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
        requested_layers: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Index every cached tile path.

        Args:
            processed_dir (str): Directory containing cached tiles.
            augmentation_cfg (Optional[dict]): Augmentation configuration.
            file_subset (Optional[List[str]]): Optional subset of files.
            validation_cfg (Optional[dict[str, Any]]): Validation policy settings.
            requested_layers (Optional[Sequence[int]]): Optional layer ids to
                select from cached feature tensors.

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
        self.requested_layers = (
            [int(layer_id) for layer_id in requested_layers]
            if requested_layers is not None
            else None
        )
        self.cached_layers = self._load_cached_layers(processed_dir)

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
        features = self._select_requested_features(features, self.processed_files[idx])
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

    def _load_cached_layers(self, processed_dir: str) -> list[int] | None:
        """Load cached layer ids from cache metadata when available.

        Args:
            processed_dir (str): Directory containing cached tiles.

        Returns:
            list[int] | None: Cached layer ids, or `None` if metadata is absent.
        """

        meta_path = os.path.join(processed_dir, "cache_meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            import json

            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError, TypeError):
            return None
        layers = meta.get("layers")
        if not isinstance(layers, list):
            return None
        try:
            return [int(layer_id) for layer_id in layers]
        except (TypeError, ValueError):
            return None

    def _select_requested_features(
        self,
        features: List[torch.Tensor],
        source: str,
    ) -> List[torch.Tensor]:
        """Return cached features filtered to the requested layer ids.

        Args:
            features (List[torch.Tensor]): Cached feature tensors.
            source (str): Source tile path for clearer error messages.

        Returns:
            List[torch.Tensor]: Requested feature tensors in requested-layer order.
        """

        if not features or self.requested_layers is None:
            return features
        if self.cached_layers is None:
            raise ValueError(
                "Requested cached-layer selection but cache metadata does not "
                f"declare layer ids for {source}."
            )
        index_by_layer = {
            int(layer_id): idx for idx, layer_id in enumerate(self.cached_layers)
        }
        selected: List[torch.Tensor] = []
        for layer_id in self.requested_layers:
            if layer_id not in index_by_layer:
                raise ValueError(
                    f"Cached features in {source} do not include requested layer "
                    f"{layer_id}; available layers: {self.cached_layers}"
                )
            selected.append(features[index_by_layer[layer_id]])
        return selected

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
