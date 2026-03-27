"""Helpers for inference outputs, explainability, and test-time augmentation."""

from __future__ import annotations

import fcntl
import os
import shutil
from contextlib import contextmanager
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from .train_utils import normalize_forward_output


def _gradcam_result(
    zero_map: np.ndarray,
    *,
    selected_layer: int | None,
    success: bool,
    failure_stage: str | None = None,
    failure_reason: str | None = None,
    cam_map: np.ndarray | None = None,
    top_indices: list[int] | None = None,
    top_scores: list[float] | None = None,
    top_maps: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Build one standardized Grad-CAM result payload.

    Args:
        zero_map (np.ndarray): Fallback CAM map shape template.
        selected_layer (int | None): Requested CAM layer id.
        success (bool): Whether extraction succeeded.
        failure_stage (str | None): Failure stage identifier.
        failure_reason (str | None): Human-readable failure reason.
        cam_map (np.ndarray | None): Computed Grad-CAM map.
        top_indices (list[int] | None): Top channel indices.
        top_scores (list[float] | None): Top channel scores.
        top_maps (list[np.ndarray] | None): Top channel activation maps.

    Returns:
        dict[str, Any]: Standardized Grad-CAM result payload.
    """

    return {
        "success": bool(success),
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "selected_layer": selected_layer,
        "cam_map": np.asarray(
            cam_map if cam_map is not None else zero_map, dtype=np.float32
        ),
        "top_indices": list(top_indices or []),
        "top_scores": [float(score) for score in (top_scores or [])],
        "top_maps": [
            np.asarray(top_map, dtype=np.float32) for top_map in (top_maps or [])
        ],
    }


def _log_gradcam_failure(
    logger: Any | None,
    *,
    failure_stage: str,
    failure_reason: str,
) -> None:
    """Emit one Grad-CAM failure message when a logger is available.

    Args:
        logger (Any | None): Optional logger.
        failure_stage (str): Failure stage identifier.
        failure_reason (str): Human-readable failure reason.
    """

    if logger:
        logger.info(
            "Grad-CAM extraction failed at %s: %s" % (failure_stage, failure_reason)
        )


class TTATransform:
    """Test-time augmentation transform wrapper.

    Args:
        name (str): Transform name (none, hflip, vflip).

    Examples:
        >>> TTATransform("hflip").name
        'hflip'
    """

    def __init__(self, name: str) -> None:
        """Initialize the transform.

        Args:
            name (str): Transform name.
        """

        self.name = name

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the augmentation to a numpy image.

        Args:
            image (np.ndarray): Input image (H, W, C).

        Returns:
            np.ndarray: Augmented image.
        """

        if self.name == "hflip":
            return np.flip(image, axis=1).copy()
        if self.name == "vflip":
            return np.flip(image, axis=0).copy()
        return image

    def invert_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Invert the augmentation on logits.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            torch.Tensor: Inverted logits.
        """

        if self.name == "hflip":
            return torch.flip(logits, dims=(3,))
        if self.name == "vflip":
            return torch.flip(logits, dims=(2,))
        return logits


def build_tta_transforms(cfg: dict) -> list[TTATransform]:
    """Build TTA transform list from configuration.

    Args:
        cfg (dict): TTA configuration block.

    Returns:
        list[TTATransform]: Transform instances.

    Examples:
        >>> transforms = build_tta_transforms({"horizontal_flip": True})
        >>> [t.name for t in transforms]
        ['none', 'hflip']
    """

    transforms = [TTATransform("none")]
    if cfg.get("horizontal_flip"):
        transforms.append(TTATransform("hflip"))
    if cfg.get("vertical_flip"):
        transforms.append(TTATransform("vflip"))
    return transforms


def normalize_map(values: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1].

    Args:
        values (np.ndarray): Input array.

    Returns:
        np.ndarray: Normalized array.

    Examples:
        >>> normalize_map(np.array([0.0, 1.0])).tolist()
        [0.0, 1.0]
    """

    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def build_blend_weight_mask(
    height: int,
    width: int,
    mode: str = "center_weighted",
    min_weight: float = 0.05,
) -> np.ndarray:
    """Build a spatial blend mask for tiled inference merging.

    Args:
        height (int): Tile height.
        width (int): Tile width.
        mode (str): Blend mode (`center_weighted` or `uniform`).
        min_weight (float): Lower bound used to avoid zero-weight borders.

    Returns:
        np.ndarray: Blend mask with shape `(height, width)`.

    Examples:
        >>> mask = build_blend_weight_mask(3, 3)
        >>> float(mask[1, 1]) > float(mask[0, 0])
        True
        >>> build_blend_weight_mask(2, 2, mode="uniform").tolist()
        [[1.0, 1.0], [1.0, 1.0]]
    """

    height = max(1, int(height))
    width = max(1, int(width))
    if mode == "uniform":
        return np.ones((height, width), dtype=np.float32)
    if mode != "center_weighted":
        raise ValueError(f"Unsupported blend mode: {mode}")

    min_weight = float(np.clip(min_weight, 0.0, 1.0))

    def _axis_weights(length: int) -> np.ndarray:
        """Build one-dimensional center-emphasized weights.

        Args:
            length (int): Number of pixels along one spatial axis.

        Returns:
            np.ndarray: One-dimensional blend weights.
        """

        if length <= 1:
            return np.ones((length,), dtype=np.float32)
        coords = np.linspace(-1.0, 1.0, num=length, dtype=np.float32)
        weights = 0.5 * (1.0 + np.cos(np.pi * coords))
        weights = min_weight + (1.0 - min_weight) * weights
        return weights.astype(np.float32)

    mask = np.outer(_axis_weights(height), _axis_weights(width)).astype(np.float32)
    max_value = float(mask.max())
    if max_value <= 0.0:
        return np.ones((height, width), dtype=np.float32)
    return (mask / max_value).astype(np.float32)


def upsample_map(values: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Upsample a 2D map to the target spatial size.

    Args:
        values (np.ndarray): 2D array.
        target_h (int): Target height.
        target_w (int): Target width.

    Returns:
        np.ndarray: Upsampled array.

    Examples:
        >>> up = upsample_map(np.array([[1.0]], dtype=np.float32), 2, 3)
        >>> up.shape
        (2, 3)
        >>> float(up[0, 0])
        1.0
    """

    tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0).float()
    up = F.interpolate(
        tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return up.squeeze(0).squeeze(0).cpu().numpy()


def upsample_rgb_map(values: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Upsample an RGB map to the target spatial size.

    Args:
        values (np.ndarray): RGB map as (H, W, 3).
        target_h (int): Target height.
        target_w (int): Target width.

    Returns:
        np.ndarray: Upsampled RGB map in [0, 1].

    Examples:
        >>> rgb = np.array([[[1.0, 0.0, 0.5]]], dtype=np.float32)
        >>> up = upsample_rgb_map(rgb, 2, 2)
        >>> up.shape
        (2, 2, 3)
        >>> up[0, 0].tolist()
        [1.0, 0.0, 0.5]
    """

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        return np.zeros((target_h, target_w, 3), dtype=np.float32)
    channels = [upsample_map(values[..., idx], target_h, target_w) for idx in range(3)]
    upsampled = np.stack(channels, axis=-1).astype(np.float32)
    return np.clip(upsampled, 0.0, 1.0)


def compute_feature_pca_rgb(
    feature_map: torch.Tensor | np.ndarray,
    n_components: int = 3,
) -> np.ndarray:
    """Project a feature map to PCA RGB channels.

    Args:
        feature_map (torch.Tensor | np.ndarray): Feature map as (B, C, H, W) or
            (C, H, W).
        n_components (int): Number of PCA components to compute, up to 3.

    Returns:
        np.ndarray: PCA RGB map as (H, W, 3) in [0, 1].
    """

    if isinstance(feature_map, torch.Tensor):
        fmap_t = feature_map.detach().float()
    else:
        fmap_t = torch.as_tensor(feature_map).detach().float()
    if fmap_t.dim() == 4:
        fmap_t = fmap_t[0]
    if fmap_t.dim() != 3:
        return np.zeros((1, 1, 3), dtype=np.float32)
    channels, height, width = fmap_t.shape
    if channels <= 0 or height <= 0 or width <= 0:
        return np.zeros((max(1, height), max(1, width), 3), dtype=np.float32)
    patches = fmap_t.permute(1, 2, 0).reshape(-1, channels)
    patches = patches - patches.mean(dim=0, keepdim=True)
    if int(patches.shape[0]) <= 1:
        return np.zeros((height, width, 3), dtype=np.float32)
    q = min(max(1, int(n_components)), 3, int(channels), int(patches.shape[0]))
    try:
        _, _, v = torch.pca_lowrank(patches, q=q, center=False)
        projected = patches @ v[:, :q]
    except Exception:
        return np.zeros((height, width, 3), dtype=np.float32)
    projected_np = projected.detach().cpu().numpy().reshape(height, width, q)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for idx in range(q):
        component = projected_np[..., idx]
        if abs(float(component.min())) > abs(float(component.max())):
            component = -component
        rgb[..., idx] = normalize_map(component)
    return rgb


def _get_attention_backend(backbone: torch.nn.Module) -> str | None:
    """Return the configured attention backend when available.

    Args:
        backbone (torch.nn.Module): Backbone model instance.

    Returns:
        str | None: Attention backend (for example ``"sdpa"``/``"eager"``), or
        ``None`` when unavailable.
    """

    cfg = getattr(backbone, "config", None)
    if cfg is None:
        return None
    for attr in ("_attn_implementation", "attn_implementation"):
        value = getattr(cfg, attr, None)
        if value is not None:
            return str(value)
    return None


def _set_attention_backend(backbone: torch.nn.Module, backend: str) -> bool:
    """Set the backbone attention backend if supported.

    Args:
        backbone (torch.nn.Module): Backbone model instance.
        backend (str): Target backend name.

    Returns:
        bool: ``True`` when the requested backend was applied.
    """

    setter = getattr(backbone, "set_attn_implementation", None)
    if callable(setter):
        try:
            setter(backend)
            return True
        except Exception:
            pass
    cfg = getattr(backbone, "config", None)
    if cfg is None:
        return False
    changed = False
    for attr in ("_attn_implementation", "attn_implementation"):
        if hasattr(cfg, attr):
            try:
                setattr(cfg, attr, backend)
                changed = True
            except Exception:
                continue
    return changed


def _compute_hidden_state_proxy_maps(
    hidden_states: Any,
    hp: int,
    wp: int,
    register_tokens: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute proxy focus maps from hidden states.

    This fallback is used when attention tensors are unavailable.

    Args:
        hidden_states (Any): Hidden state tuple from the backbone forward pass.
        hp (int): Patch-grid height.
        wp (int): Patch-grid width.
        register_tokens (int): Number of register tokens before patch tokens.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: Proxy CLS and rollout-style maps,
        normalized to ``[0, 1]``. Returns ``None`` when unavailable.
    """

    if hidden_states is None:
        return None
    patch_tokens_expected = int(hp * wp)
    layer_maps: list[np.ndarray] = []
    for layer_output in hidden_states:
        if layer_output is None or layer_output.dim() != 3:
            continue
        if layer_output.shape[0] < 1 or layer_output.shape[1] <= 1 + register_tokens:
            continue
        cls_token = layer_output[:, 0:1, :]
        patch_tokens = layer_output[:, 1 + register_tokens :, :]
        if int(patch_tokens.shape[1]) < patch_tokens_expected:
            continue
        patch_tokens = patch_tokens[:, :patch_tokens_expected, :]
        cls_norm = F.normalize(cls_token, dim=-1)
        patch_norm = F.normalize(patch_tokens, dim=-1)
        sim = torch.matmul(cls_norm, patch_norm.transpose(1, 2)).squeeze(1)
        sim_map = sim.reshape(hp, wp).detach().cpu().numpy()
        layer_maps.append(normalize_map(sim_map))
    if not layer_maps:
        return None
    cls_proxy = layer_maps[-1]
    rollout_proxy = normalize_map(np.mean(np.stack(layer_maps, axis=0), axis=0))
    return cls_proxy, rollout_proxy


def compute_attention_maps(
    image_hw3: np.ndarray,
    backbone: torch.nn.Module,
    processor: Any,
    device: torch.device,
    ps: int,
    logger: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Compute CLS and rollout attention maps for a single image.

    Args:
        image_hw3 (np.ndarray): Input image in HWC format.
        backbone (torch.nn.Module): DINO backbone.
        processor (object): Image processor.
        device (torch.device): Device for inference.
        ps (int): Patch size for the backbone.
        logger (Any | None): Optional logger for fallback events.

    Returns:
        tuple[np.ndarray, np.ndarray, bool]: CLS map, rollout map, and
        a flag indicating whether attentions were available.
    """

    proc = processor
    inputs = proc(
        images=image_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)
    r_tokens = getattr(backbone.config, "num_register_tokens", 0)
    _, _, h_proc, w_proc = inputs["pixel_values"].shape
    hp, wp = h_proc // ps, w_proc // ps
    zero_map = np.zeros((hp, wp), dtype=np.float32)
    supported_attention_backends = {"eager", "eager_paged", "flex_attention"}

    def _maps_from_attentions(attentions: Any) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute CLS and rollout maps from model attentions.

        Args:
            attentions (Any): Backbone attention tensors.

        Returns:
            tuple[np.ndarray, np.ndarray] | None: Normalized maps, or ``None`` if
            attentions are missing/invalid.
        """

        if attentions is None:
            return None
        if isinstance(attentions, torch.Tensor):
            raw_layers = [attentions]
        elif isinstance(attentions, (list, tuple)):
            raw_layers = list(attentions)
        else:
            return None

        try:
            reduced_attentions: list[torch.Tensor] = []
            for layer in raw_layers:
                if layer is None or not isinstance(layer, torch.Tensor):
                    continue
                if layer.dim() == 4:
                    reduced = layer.mean(dim=1)
                elif layer.dim() == 3:
                    reduced = layer
                else:
                    continue
                if reduced.shape[0] < 1:
                    continue
                reduced_attentions.append(reduced)
            if not reduced_attentions:
                return None

            last = reduced_attentions[-1]
            tokens = int(last.shape[-1])
            if int(last.shape[-2]) != tokens:
                return None

            valid_attentions = [
                attn
                for attn in reduced_attentions
                if int(attn.shape[-2]) == tokens and int(attn.shape[-1]) == tokens
            ]
            if not valid_attentions:
                return None

            last = valid_attentions[-1]
            cls_attn = last[:, 0, 1 + r_tokens :]
            if int(cls_attn.shape[-1]) < int(hp * wp):
                return None
            cls_map = cls_attn[:, : hp * wp].reshape(hp, wp).detach().cpu().numpy()

            identity = torch.eye(tokens, device=last.device, dtype=last.dtype)
            rollout = identity.unsqueeze(0).repeat(last.shape[0], 1, 1)
            for attn in valid_attentions:
                if attn.device != rollout.device or attn.dtype != rollout.dtype:
                    attn = attn.to(device=rollout.device, dtype=rollout.dtype)
                attn = attn + identity
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                rollout = attn @ rollout

            rollout_cls = rollout[:, 0, 1 + r_tokens :]
            if int(rollout_cls.shape[-1]) < int(hp * wp):
                return None
            rollout_map = (
                rollout_cls[:, : hp * wp].reshape(hp, wp).detach().cpu().numpy()
            )
            return normalize_map(cls_map), normalize_map(rollout_map)
        except Exception:
            return None

    try:
        out = None
        original_backend = _get_attention_backend(backbone)
        eager_switched = False
        if (
            original_backend
            and original_backend not in supported_attention_backends
            and original_backend != "eager"
        ):
            eager_switched = _set_attention_backend(backbone, "eager")
        try:
            with torch.no_grad():
                out = backbone(
                    **inputs,
                    output_attentions=True,
                )
            attn_maps = _maps_from_attentions(getattr(out, "attentions", None))
            if attn_maps is not None:
                cls_map, rollout_map = attn_maps
                return cls_map, rollout_map, True

            if not eager_switched and original_backend and original_backend != "eager":
                eager_switched = _set_attention_backend(backbone, "eager")
            if eager_switched:
                with torch.no_grad():
                    out = backbone(
                        **inputs,
                        output_attentions=True,
                    )
                attn_maps = _maps_from_attentions(getattr(out, "attentions", None))
                if attn_maps is not None:
                    cls_map, rollout_map = attn_maps
                    if logger:
                        logger.info(
                            "Attention maps extracted after switching backend to eager."
                        )
                    return cls_map, rollout_map, True
        finally:
            if eager_switched and original_backend:
                _set_attention_backend(backbone, original_backend)

        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            with torch.no_grad():
                out_hidden = backbone(**inputs, output_hidden_states=True)
            hidden_states = getattr(out_hidden, "hidden_states", None)
        proxy_maps = _compute_hidden_state_proxy_maps(
            hidden_states,
            hp=hp,
            wp=wp,
            register_tokens=r_tokens,
        )
        if proxy_maps is not None:
            if logger:
                logger.info(
                    "Backbone attentions unavailable; using hidden-state proxy maps."
                )
            cls_map, rollout_map = proxy_maps
            return cls_map, rollout_map, True
        if logger:
            logger.info("Backbone returned no attentions.")
        return zero_map, zero_map, False
    except Exception:
        if logger:
            logger.info("Attention extraction failed.")
        return zero_map, zero_map, False


def compute_gradcam_map(
    image_hw3: np.ndarray,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    processor: Any,
    device: torch.device,
    layers: list[int],
    ps: int,
    class_index: int,
    cam_layer: int | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Compute a Grad-CAM map using the DINO backbone and head.

    Args:
        image_hw3 (np.ndarray): Input image in HWC format.
        backbone (torch.nn.Module): DINO backbone.
        head (torch.nn.Module): Segmentation head.
        processor (Any): Image processor.
        device (torch.device): Device for inference.
        layers (list[int]): Backbone layers used by the head.
        ps (int): Patch size for the backbone.
        class_index (int): Target class index for Grad-CAM.
        cam_layer (int | None): Explicit layer index for CAM extraction.
        logger (Any | None): Optional logger for errors.

    Returns:
        dict[str, Any]: Structured Grad-CAM result payload.
    """

    return compute_gradcam_with_topk_channels(
        image_hw3=image_hw3,
        backbone=backbone,
        head=head,
        processor=processor,
        device=device,
        layers=layers,
        ps=ps,
        class_index=class_index,
        topk_channels=1,
        cam_layer=cam_layer,
        logger=logger,
    )


def compute_branch_importance(
    head: torch.nn.Module,
    image: torch.Tensor,
    features: list[torch.Tensor],
    class_index: int,
    logger: Any | None = None,
) -> dict[str, float]:
    """Estimate relative importance of RGB image input vs. DINO feature inputs.

    The metric is gradient-based: we backpropagate a scalar target from model
    logits and compare average absolute gradient magnitudes on image and
    projected DINO feature tensors.

    Args:
        head (torch.nn.Module): Segmentation head.
        image (torch.Tensor): Input image tensor with shape (1, C, H, W).
        features (list[torch.Tensor]): Feature tensors passed to the head.
        class_index (int): Target class index for attribution.
        logger (Any | None): Optional logger for failures.

    Returns:
        dict[str, float]: Branch-importance summary with keys
        `img_importance`, `dino_importance`, and `img_to_dino_ratio`.

    Examples:
        >>> result = compute_branch_importance(  # doctest: +SKIP
        ...     head=head,
        ...     image=torch.randn(1, 3, 32, 32),
        ...     features=[torch.randn(1, 8, 4, 4)],
        ...     class_index=1,
        ... )
        >>> "img_importance" in result  # doctest: +SKIP
        True
    """

    if image.dim() != 4 or image.shape[0] != 1:
        return {
            "img_importance": 0.5,
            "dino_importance": 0.5,
            "img_to_dino_ratio": 1.0,
        }
    if not features:
        return {
            "img_importance": 1.0,
            "dino_importance": 0.0,
            "img_to_dino_ratio": float("inf"),
        }
    try:
        image_var = image.detach().clone().requires_grad_(True)
        feature_vars = [feat.detach().clone().requires_grad_(True) for feat in features]
        with torch.enable_grad():
            if hasattr(head, "forward_with_aux"):
                logits, _ = head.forward_with_aux(image_var, feature_vars)
            else:
                logits = head(image_var, feature_vars)
            if logits.dim() == 4 and 0 <= class_index < int(logits.shape[1]):
                target = logits[:, class_index].mean()
            else:
                target = logits.mean()
            grads = torch.autograd.grad(
                target,
                [image_var, *feature_vars],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        img_grad = grads[0] if grads else None
        img_grad_score = (
            float(img_grad.detach().abs().mean().item())
            if img_grad is not None
            else 0.0
        )
        dino_grad_scores = [
            float(grad.detach().abs().mean().item())
            for grad in grads[1:]
            if grad is not None
        ]
        dino_grad_score = float(np.mean(dino_grad_scores)) if dino_grad_scores else 0.0
        denom = max(img_grad_score + dino_grad_score, 1e-12)
        img_importance = img_grad_score / denom
        dino_importance = dino_grad_score / denom
        ratio = img_grad_score / max(dino_grad_score, 1e-12)
        return {
            "img_importance": float(img_importance),
            "dino_importance": float(dino_importance),
            "img_to_dino_ratio": float(ratio),
        }
    except Exception as exc:
        if logger:
            logger.info(f"Branch importance failed; using neutral split. ({exc})")
        return {
            "img_importance": 0.5,
            "dino_importance": 0.5,
            "img_to_dino_ratio": 1.0,
        }
    finally:
        head.zero_grad(set_to_none=True)


def compute_dino_layer_importance(
    head: torch.nn.Module,
    image: torch.Tensor,
    features: list[torch.Tensor],
    layer_ids: list[int],
    class_index: int,
    logger: Any | None = None,
) -> dict[int, float]:
    """Estimate relative importance of each DINO feature connection.

    Importance is computed from mean absolute gradients on each feature tensor
    and normalized to sum to 1.0 across the provided feature list.

    Args:
        head (torch.nn.Module): Segmentation head.
        image (torch.Tensor): Input image tensor with shape (1, C, H, W).
        features (list[torch.Tensor]): Feature tensors passed to the head.
        layer_ids (list[int]): DINO layer identifiers for each feature tensor.
        class_index (int): Target class index for attribution.
        logger (Any | None): Optional logger for failures.

    Returns:
        dict[int, float]: Mapping from DINO layer id to normalized importance.

    Examples:
        >>> layer_imp = compute_dino_layer_importance(  # doctest: +SKIP
        ...     head=head,
        ...     image=torch.randn(1, 3, 32, 32),
        ...     features=[torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)],
        ...     layer_ids=[7, 14],
        ...     class_index=1,
        ... )
        >>> 7 in layer_imp  # doctest: +SKIP
        True
    """

    if not features:
        return {}
    if len(layer_ids) != len(features):
        layer_ids = list(range(len(features)))
    try:
        image_var = image.detach().clone().requires_grad_(True)
        feature_vars = [feat.detach().clone().requires_grad_(True) for feat in features]
        with torch.enable_grad():
            if hasattr(head, "forward_with_aux"):
                logits, _ = head.forward_with_aux(image_var, feature_vars)
            else:
                logits = head(image_var, feature_vars)
            if logits.dim() == 4 and 0 <= class_index < int(logits.shape[1]):
                target = logits[:, class_index].mean()
            else:
                target = logits.mean()
            grads = torch.autograd.grad(
                target,
                feature_vars,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        raw_scores = [
            float(grad.detach().abs().mean().item()) if grad is not None else 0.0
            for grad in grads
        ]
        denom = max(float(sum(raw_scores)), 1e-12)
        return {
            int(layer_id): float(score / denom)
            for layer_id, score in zip(layer_ids, raw_scores)
        }
    except Exception as exc:
        if logger:
            logger.info(f"DINO layer importance failed; using uniform split. ({exc})")
        uniform = 1.0 / max(1, len(layer_ids))
        return {int(layer_id): float(uniform) for layer_id in layer_ids}
    finally:
        head.zero_grad(set_to_none=True)


def compute_gradcam_with_topk_channels(
    image_hw3: np.ndarray,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    processor: Any,
    device: torch.device,
    layers: list[int],
    ps: int,
    class_index: int,
    topk_channels: int = 5,
    cam_layer: int | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Compute Grad-CAM and top-k influential channel maps.

    Args:
        image_hw3 (np.ndarray): Input image in HWC format.
        backbone (torch.nn.Module): DINO backbone.
        head (torch.nn.Module): Segmentation head.
        processor (Any): Image processor.
        device (torch.device): Device for inference.
        layers (list[int]): Backbone layers used by the head.
        ps (int): Patch size for the backbone.
        class_index (int): Target class index for Grad-CAM.
        topk_channels (int): Number of channel maps to return.
        cam_layer (int | None): Explicit layer index for CAM, defaults to last layer.
        logger (Any | None): Optional logger for errors.

    Returns:
        dict[str, Any]: Dict with keys `success`, `failure_stage`,
        `failure_reason`, `selected_layer`, `cam_map`, `top_indices`,
        `top_scores`, and `top_maps`.

    Examples:
        >>> result = compute_gradcam_with_topk_channels(  # doctest: +SKIP
        ...     image_hw3=np.zeros((32, 32, 3), dtype=np.float32),
        ...     backbone=backbone,
        ...     head=head,
        ...     processor=processor,
        ...     device=torch.device("cpu"),
        ...     layers=[1],
        ...     ps=16,
        ...     class_index=0,
        ... )
        >>> "cam_map" in result  # doctest: +SKIP
        True
    """

    inputs = processor(
        images=image_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)
    r_tokens = getattr(backbone.config, "num_register_tokens", 0)
    img_norm = (image_hw3.astype(np.float32) / 255.0).astype(np.float32)
    img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h_proc, w_proc = inputs["pixel_values"].shape
    hp, wp = h_proc // ps, w_proc // ps
    zero_map = np.zeros((hp, wp), dtype=np.float32)
    topk = max(1, int(topk_channels))
    if not layers:
        reason = "No backbone layers configured for Grad-CAM."
        _log_gradcam_failure(logger, failure_stage="no_layers", failure_reason=reason)
        return _gradcam_result(
            zero_map,
            selected_layer=None,
            success=False,
            failure_stage="no_layers",
            failure_reason=reason,
        )
    selected_layer = cam_layer if cam_layer is not None else layers[-1]
    if selected_layer not in {int(layer_id) for layer_id in layers}:
        reason = (
            f"Requested CAM layer {selected_layer} is not present in configured "
            f"layers {list(layers)}."
        )
        _log_gradcam_failure(
            logger,
            failure_stage="selected_layer_missing",
            failure_reason=reason,
        )
        return _gradcam_result(
            zero_map,
            selected_layer=selected_layer,
            success=False,
            failure_stage="selected_layer_missing",
            failure_reason=reason,
        )
    try:
        with torch.enable_grad():
            backbone.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            out = backbone(**inputs, output_hidden_states=True)
            hidden_states = out.hidden_states
            feat_maps = []
            cam_feature = None
            for layer_idx in layers:
                layer_output = hidden_states[layer_idx]
                patch_tokens = layer_output[:, 1 + r_tokens :, :]
                feats = patch_tokens.reshape(1, hp, wp, -1).permute(0, 3, 1, 2)
                if not feats.requires_grad:
                    feats = feats.requires_grad_()
                if layer_idx == selected_layer:
                    cam_feature = feats
                    cam_feature.retain_grad()
                feat_maps.append(feats)
            if cam_feature is None:
                reason = (
                    f"Selected CAM layer {selected_layer} was not found in the "
                    "backbone hidden states."
                )
                _log_gradcam_failure(
                    logger,
                    failure_stage="selected_layer_missing",
                    failure_reason=reason,
                )
                return _gradcam_result(
                    zero_map,
                    selected_layer=selected_layer,
                    success=False,
                    failure_stage="selected_layer_missing",
                    failure_reason=reason,
                )
            payload = normalize_forward_output(head(img_t, feat_maps))
            logits = payload["logits"]
            if logits.dim() == 4 and 0 <= class_index < int(logits.shape[1]):
                target = logits[:, class_index].mean()
            else:
                target = logits.mean()
            target.backward()
            grads = cam_feature.grad
            if grads is None:
                reason = "Gradients were not retained for the selected CAM feature map."
                _log_gradcam_failure(
                    logger,
                    failure_stage="missing_gradients",
                    failure_reason=reason,
                )
                return _gradcam_result(
                    zero_map,
                    selected_layer=selected_layer,
                    success=False,
                    failure_stage="missing_gradients",
                    failure_reason=reason,
                )
            weights = grads.mean(dim=(2, 3), keepdim=True)
            weighted_feature = weights * cam_feature
            cam = torch.relu(weighted_feature.sum(dim=1))
            cam_map = normalize_map(cam.squeeze(0).detach().cpu().numpy())
            channel_scores = weighted_feature.detach().abs().mean(dim=(0, 2, 3)).cpu()
            channel_count = int(channel_scores.shape[0])
            keep = min(topk, channel_count)
            if keep <= 0:
                return _gradcam_result(
                    zero_map,
                    selected_layer=selected_layer,
                    success=True,
                    cam_map=cam_map,
                )
            top_scores_t, top_indices_t = torch.topk(channel_scores, k=keep)
            top_indices = [int(idx) for idx in top_indices_t.tolist()]
            top_scores = [float(score) for score in top_scores_t.tolist()]
            top_maps: list[np.ndarray] = []
            for idx in top_indices:
                fmap = cam_feature[0, idx].detach().cpu().numpy()
                top_maps.append(normalize_map(fmap))
            return _gradcam_result(
                zero_map,
                selected_layer=selected_layer,
                success=True,
                cam_map=cam_map,
                top_indices=top_indices,
                top_scores=top_scores,
                top_maps=top_maps,
            )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        _log_gradcam_failure(
            logger,
            failure_stage="exception",
            failure_reason=reason,
        )
        return _gradcam_result(
            zero_map,
            selected_layer=selected_layer,
            success=False,
            failure_stage="exception",
            failure_reason=reason,
        )


def build_dashboard(
    output_path: str,
    rgb: np.ndarray,
    overlay_pred: np.ndarray,
    gradcam_overlay: np.ndarray,
    class_prob: np.ndarray,
    layout: str = "2x2",
) -> None:
    """Create a compact scene-level inference dashboard.

    Args:
        output_path (str): PNG output path.
        rgb (np.ndarray): RGB image.
        overlay_pred (np.ndarray): RGB image with prediction overlay.
        gradcam_overlay (np.ndarray): RGB image with Grad-CAM overlay.
        class_prob (np.ndarray): Class probability map.
        layout (str): Layout string, defaults to `2x2`.

    Examples:
        >>> callable(build_dashboard)
        True
    """

    import matplotlib.pyplot as plt

    rows, cols = 2, 2
    if layout == "1x4":
        rows, cols = 1, 4
    panels = [
        ("RGB", rgb, None),
        ("Prediction Overlay", overlay_pred, None),
        ("Grad-CAM", gradcam_overlay, None),
        ("Class Probability", class_prob, "magma"),
    ]
    required_slots = len(panels)
    if required_slots > rows * cols:
        cols = int(np.ceil(required_slots / max(1, rows)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx >= len(panels):
            ax.axis("off")
            continue
        title, data, cmap = panels[idx]
        if data.ndim == 2:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_xai_maps(
    probs: np.ndarray,
    class_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute confidence, entropy, and class probability maps.

    Args:
        probs (np.ndarray): Class probabilities (C, H, W).
        class_index (int): Class index for class probability.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Confidence, entropy, class prob.

    Examples:
        >>> conf, ent, cls = compute_xai_maps(
        ...     np.array([[[0.2]], [[0.8]]], dtype=np.float32),
        ...     class_index=1,
        ... )
        >>> conf.shape, ent.shape, cls.shape
        ((1, 1), (1, 1), (1, 1))
    """

    confidence = probs.max(axis=0)
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=0)
    if 0 <= class_index < probs.shape[0]:
        class_prob = probs[class_index]
    else:
        class_prob = np.zeros_like(confidence)
    return normalize_map(confidence), normalize_map(entropy), normalize_map(class_prob)


def overlay_binary_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (120, 190, 255),
    alpha: float = 0.28,
) -> np.ndarray:
    """Overlay a binary mask onto an RGB image.

    Args:
        rgb (np.ndarray): RGB image (H, W, 3).
        mask (np.ndarray): Binary or label mask.
        color (tuple[int, int, int]): Overlay RGB color.
        alpha (float): Overlay alpha.

    Returns:
        np.ndarray: Overlay image.

    Examples:
        >>> rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        >>> mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        >>> out = overlay_binary_mask(rgb, mask, color=(100, 150, 200), alpha=0.5)
        >>> out[0, 1].tolist()
        [50, 75, 100]
    """

    rgb_uint8 = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
    mask_bool = np.asarray(mask) > 0
    if not mask_bool.any():
        return rgb_uint8.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    overlay = rgb_uint8.astype(np.float32).copy()
    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color_arr.reshape(
        3
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def overlay_heatmap(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    cmap: str = "magma",
    alpha: float = 0.4,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Overlay a heatmap onto an RGB image.

    Args:
        rgb (np.ndarray): RGB image (H, W, 3).
        heatmap (np.ndarray): Heatmap in [0, 1].
        cmap (str): Matplotlib colormap name.
        alpha (float): Overlay alpha.
        vmin (float | None): Optional lower bound used to normalize the heatmap
            before color mapping.
        vmax (float | None): Optional upper bound used to normalize the heatmap
            before color mapping.

    Returns:
        np.ndarray: Overlay image.

    Examples:
        >>> rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        >>> heatmap = np.ones((2, 2), dtype=np.float32)
        >>> overlay_heatmap(rgb, heatmap).shape
        (2, 2, 3)
    """

    import matplotlib.cm as cm

    rgb_float = rgb.astype(np.float32) / 255.0
    heatmap_float = np.asarray(heatmap, dtype=np.float32)
    if vmin is not None or vmax is not None:
        lo = (
            float(vmin)
            if vmin is not None and np.isfinite(vmin)
            else float(np.nanmin(heatmap_float))
        )
        hi = (
            float(vmax)
            if vmax is not None and np.isfinite(vmax)
            else float(np.nanmax(heatmap_float))
        )
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            heatmap_float = np.clip((heatmap_float - lo) / (hi - lo), 0.0, 1.0)
        else:
            heatmap_float = np.clip(heatmap_float, 0.0, 1.0)
    colored = cm.get_cmap(cmap)(heatmap_float)[..., :3]
    overlay = (1 - alpha) * rgb_float + alpha * colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return overlay


def _iter_raster_windows(
    height: int,
    width: int,
    window_size: int = 4096,
) -> list[Window]:
    """Return fixed-size windows covering one raster.

    Args:
        height (int): Raster height in pixels.
        width (int): Raster width in pixels.
        window_size (int): Maximum window width/height in pixels.

    Returns:
        list[Window]: Coverage windows spanning the raster extent.
    """

    windows: list[Window] = []
    for row_off in range(0, int(height), int(window_size)):
        row_end = min(int(height), row_off + int(window_size))
        for col_off in range(0, int(width), int(window_size)):
            col_end = min(int(width), col_off + int(window_size))
            windows.append(
                Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=col_end - col_off,
                    height=row_end - row_off,
                )
            )
    return windows


def _normalize_template_window(window: Window) -> Window:
    """Return one integer-valued raster window.

    Args:
        window (Window): Candidate raster window.

    Returns:
        Window: Integer-valued window with positive width and height.
    """

    return Window(
        col_off=int(window.col_off),
        row_off=int(window.row_off),
        width=max(1, int(window.width)),
        height=max(1, int(window.height)),
    )


def _gtiff_block_size(length: int, preferred: int = 1024) -> int | None:
    """Return one GeoTIFF-compatible tiled block size when feasible.

    Args:
        length (int): Raster width or height.
        preferred (int): Preferred block size.

    Returns:
        int | None: Block size rounded down to a multiple of 16, or ``None``
        when the raster edge is too small for tiled blocks.

    Examples:
        >>> _gtiff_block_size(1024)
        1024
        >>> _gtiff_block_size(30)
        16
        >>> _gtiff_block_size(8) is None
        True
    """

    length = int(length)
    if length < 16:
        return None
    capped = min(int(preferred), length)
    return max(16, (capped // 16) * 16)


def ensure_cumulative_prediction_raster(
    output_path: str,
    template_path: str,
    *,
    template_window: Window | None = None,
    dtype: str = "uint8",
    fill_value: int = 0,
    compress: str = "deflate",
    num_threads: str | int | None = "ALL_CPUS",
) -> bool:
    """Create or validate one cumulative prediction GeoTIFF.

    Args:
        output_path (str): Destination GeoTIFF path.
        template_path (str): Template GeoTIFF providing the target grid.
        template_window (Window | None): Optional window on the template grid
            defining the output raster extent.
        dtype (str): Output raster dtype.
        fill_value (int): Background value for empty pixels.
        compress (str): GeoTIFF compression codec.
        num_threads (str | int | None): Optional GeoTIFF compression thread
            hint used for large tiled outputs.

    Returns:
        bool: ``True`` when a new blank raster was created.

    Examples:
        >>> callable(ensure_cumulative_prediction_raster)
        True
    """

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(template_path) as template:
        template_profile = template.profile.copy()
        template_crs = template.crs
        template_transform = template.transform
        target_window = (
            _normalize_template_window(template_window)
            if template_window is not None
            else Window(0, 0, int(template.width), int(template.height))
        )
        template_transform = rasterio.windows.transform(
            target_window,
            template_transform,
        )
        template_width = int(target_window.width)
        template_height = int(target_window.height)
        output_profile = template_profile
        output_profile.update(
            driver="GTiff",
            count=1,
            dtype=dtype,
            nodata=fill_value,
            compress=compress or template_profile.get("compress", "deflate"),
            transform=template_transform,
            width=template_width,
            height=template_height,
            BIGTIFF="IF_SAFER",
        )
        block_x = _gtiff_block_size(template_width)
        block_y = _gtiff_block_size(template_height)
        if block_x is not None and block_y is not None:
            output_profile.update(
                tiled=True,
                blockxsize=block_x,
                blockysize=block_y,
            )
        if num_threads is not None:
            output_profile["num_threads"] = str(num_threads)
        if os.path.exists(output_path):
            with rasterio.open(output_path) as existing:
                if (
                    existing.count != 1
                    or existing.crs != template_crs
                    or existing.transform != template_transform
                    or int(existing.width) != template_width
                    or int(existing.height) != template_height
                ):
                    raise ValueError(
                        "Existing cumulative raster does not match the expected "
                        "cumulative output grid."
                    )
            return False

        with rasterio.open(output_path, "w", **output_profile) as dst:
            for window in _iter_raster_windows(template_height, template_width):
                dst.write(
                    np.full(
                        (int(window.height), int(window.width)),
                        fill_value,
                        dtype=np.dtype(dtype),
                    ),
                    1,
                    window=window,
                )
    return True


def build_cumulative_raster_backup_path(output_path: str, run_id: str) -> str:
    """Return the one-time backup path for a cumulative GeoTIFF.

    Args:
        output_path (str): Destination GeoTIFF path.
        run_id (str): Active run identifier.

    Returns:
        str: Backup path in the same directory as ``output_path``.

    Examples:
        >>> build_cumulative_raster_backup_path("predictions.tif", "run1")
        'predictions_preupdate_run1.tif'
    """

    root, ext = os.path.splitext(output_path)
    suffix = ext or ".tif"
    return f"{root}_preupdate_{run_id}{suffix}"


@contextmanager
def hold_prediction_raster_lock(output_path: str) -> Any:
    """Hold one exclusive lock for a shared prediction raster.

    Args:
        output_path (str): Destination cumulative GeoTIFF path.

    Yields:
        Any: Open lock-file handle held for the duration of the context.

    Examples:
        >>> callable(hold_prediction_raster_lock)
        True
    """

    lock_path = f"{output_path}.lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def backup_prediction_raster(output_path: str, backup_path: str) -> None:
    """Copy one prediction GeoTIFF to a backup path.

    Args:
        output_path (str): Source GeoTIFF path.
        backup_path (str): Destination backup path.

    Examples:
        >>> callable(backup_prediction_raster)
        True
    """

    os.makedirs(os.path.dirname(backup_path) or ".", exist_ok=True)
    shutil.copy2(output_path, backup_path)


def write_prediction_to_cumulative_raster(
    output_path: str,
    prediction: np.ndarray,
    prediction_transform: Any,
    prediction_crs: Any,
) -> Window:
    """Write one scene prediction into the matching cumulative-raster window.

    Args:
        output_path (str): Destination cumulative GeoTIFF path.
        prediction (np.ndarray): Scene prediction array on one output grid.
        prediction_transform (Any): Affine transform for ``prediction``.
        prediction_crs (Any): CRS for ``prediction``.

    Returns:
        Window: Destination write window inside ``output_path``.

    Examples:
        >>> callable(write_prediction_to_cumulative_raster)
        True
    """

    pred_array = np.array(prediction, dtype=np.uint8, copy=False)
    if pred_array.ndim != 2:
        raise ValueError("prediction must be a 2D array")
    with rasterio.open(output_path, "r+") as dst:
        pred_bounds = array_bounds(
            pred_array.shape[0],
            pred_array.shape[1],
            prediction_transform,
        )
        pred_crs_obj = rasterio.crs.CRS.from_user_input(prediction_crs)
        if dst.crs != pred_crs_obj:
            pred_bounds = transform_bounds(
                pred_crs_obj,
                dst.crs,
                *pred_bounds,
                densify_pts=21,
            )
        target_window = (
            from_bounds(*pred_bounds, transform=dst.transform)
            .round_offsets()
            .round_lengths()
        )
        target_window = _normalize_template_window(target_window)
        src_row_off = max(0, -int(target_window.row_off))
        src_col_off = max(0, -int(target_window.col_off))
        dst_row_off = max(0, int(target_window.row_off))
        dst_col_off = max(0, int(target_window.col_off))
        row_end = min(int(target_window.row_off + target_window.height), dst.height)
        col_end = min(int(target_window.col_off + target_window.width), dst.width)
        write_height = row_end - dst_row_off
        write_width = col_end - dst_col_off
        if write_height <= 0 or write_width <= 0:
            raise ValueError(
                "Prediction window falls outside the cumulative raster extent."
            )
        write_window = Window(
            col_off=dst_col_off,
            row_off=dst_row_off,
            width=write_width,
            height=write_height,
        )
        dst.write(
            pred_array[
                src_row_off : src_row_off + write_height,
                src_col_off : src_col_off + write_width,
            ],
            1,
            window=write_window,
        )
        return write_window


def extract_prediction_features(
    pred_mask: np.ndarray,
    transform: Any,
    source_crs: Any,
    source_id: str,
    run_id: str,
    foreground_class: int = 1,
    target_epsg: int = 4326,
) -> list[dict[str, Any]]:
    """Polygonize a foreground prediction mask and reproject it to EPSG:4326.

    Args:
        pred_mask (np.ndarray): Predicted class raster.
        transform (Any): Raster affine transform.
        source_crs (Any): Source raster CRS.
        source_id (str): Source scene identifier.
        run_id (str): Run identifier stored on each feature.
        foreground_class (int): Foreground class value to polygonize.
        target_epsg (int): Target EPSG code.

    Returns:
        list[dict[str, Any]]: Fiona-style features ready for writing.

    Examples:
        >>> callable(extract_prediction_features)
        True
    """

    import rasterio
    from rasterio.features import shapes
    from rasterio.warp import transform_geom

    source_crs_obj = None
    if source_crs is not None:
        source_crs_obj = rasterio.crs.CRS.from_user_input(source_crs)
    if source_crs_obj is None:
        raise ValueError("Source raster CRS is required for vector export.")
    target_crs = rasterio.crs.CRS.from_epsg(int(target_epsg))

    pred_arr = np.asarray(pred_mask, dtype=np.uint8)
    foreground_value = int(foreground_class)
    foreground_mask = pred_arr == foreground_value
    if not foreground_mask.any():
        return []

    features: list[dict[str, Any]] = []
    for geometry, value in shapes(pred_arr, mask=foreground_mask, transform=transform):
        if int(value) != foreground_value:
            continue
        reproj_geometry = transform_geom(
            source_crs_obj,
            target_crs,
            geometry,
            antimeridian_cutting=True,
            precision=6,
        )
        features.append(
            {
                "geometry": reproj_geometry,
                "properties": {
                    "source_id": str(source_id)[:80],
                    "class_id": foreground_value,
                    "run_id": str(run_id)[:80],
                },
            }
        )
    return features


def append_prediction_shapefile(
    output_path: str,
    features: list[dict[str, Any]],
    target_epsg: int = 4326,
    append: bool = True,
) -> None:
    """Append prediction features to a shapefile dataset.

    Args:
        output_path (str): Destination `.shp` path.
        features (list[dict[str, Any]]): Fiona-style features.
        target_epsg (int): Target CRS EPSG code.
        append (bool): Whether to append to an existing shapefile.

    Examples:
        >>> callable(append_prediction_shapefile)
        True
    """

    if not features:
        return

    import fiona
    import rasterio

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    schema = {
        "geometry": "Polygon",
        "properties": {
            "source_id": "str:80",
            "class_id": "int",
            "run_id": "str:80",
        },
    }
    mode = "a" if append and os.path.exists(output_path) else "w"
    target_crs = rasterio.crs.CRS.from_epsg(int(target_epsg))
    with fiona.open(
        output_path,
        mode=mode,
        driver="ESRI Shapefile",
        schema=schema,
        crs_wkt=target_crs.to_wkt(),
    ) as sink:
        for feature in features:
            sink.write(feature)
