"""Helpers for inference and test-time augmentation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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


def upsample_map(values: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Upsample a 2D map to the target spatial size.

    Args:
        values (np.ndarray): 2D array.
        target_h (int): Target height.
        target_w (int): Target width.

    Returns:
        np.ndarray: Upsampled array.
    """

    tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0).float()
    up = F.interpolate(
        tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return up.squeeze(0).squeeze(0).cpu().numpy()


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
    try:
        with torch.no_grad():
            out = backbone(**inputs, output_attentions=True)
        attentions = out.attentions
        if attentions is None:
            if logger:
                logger.info("Backbone returned no attentions; using zeros.")
            zeros = np.zeros((hp, wp), dtype=np.float32)
            return zeros, zeros, False
        last = attentions[-1].mean(dim=1)
        cls_attn = last[:, 0, 1 + r_tokens :]
        cls_map = cls_attn.reshape(hp, wp).detach().cpu().numpy()
        tokens = last.shape[-1]
        rollout = torch.eye(tokens, device=last.device).unsqueeze(0)
        rollout = rollout.repeat(last.shape[0], 1, 1)
        for layer in attentions:
            attn = layer.mean(dim=1)
            attn = attn + torch.eye(tokens, device=attn.device)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            rollout = attn @ rollout
        rollout_cls = rollout[:, 0, 1 + r_tokens :].reshape(hp, wp)
        rollout_map = rollout_cls.detach().cpu().numpy()
        return normalize_map(cls_map), normalize_map(rollout_map), True
    except Exception:
        if logger:
            logger.info("Attention extraction failed; using zeros.")
        zeros = np.zeros((hp, wp), dtype=np.float32)
        return zeros, zeros, False


def compute_gradcam_map(
    image_hw3: np.ndarray,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    processor: Any,
    device: torch.device,
    layers: list[int],
    ps: int,
    class_index: int,
    logger: Any | None = None,
) -> np.ndarray:
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
        logger (Any | None): Optional logger for errors.

    Returns:
        np.ndarray: Grad-CAM map in [0, 1].
    """

    inputs = processor(
        images=image_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)
    R = getattr(backbone.config, "num_register_tokens", 0)
    img_norm = (image_hw3.astype(np.float32) / 255.0).astype(np.float32)
    img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.enable_grad():
        backbone.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        out = backbone(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states
        _, _, h_proc, w_proc = inputs["pixel_values"].shape
        hp, wp = h_proc // ps, w_proc // ps
        feat_maps = []
        cam_layer = layers[-1]
        cam_feature = None
        for layer_idx in layers:
            layer_output = hidden_states[layer_idx]
            patch_tokens = layer_output[:, 1 + R :, :]
            feats = patch_tokens.reshape(1, hp, wp, -1).permute(0, 3, 1, 2)
            if layer_idx == cam_layer:
                cam_feature = feats
                cam_feature.retain_grad()
            feat_maps.append(feats)
        if cam_feature is None:
            if logger:
                logger.info("Grad-CAM layer not found; using zeros.")
            return np.zeros((hp, wp), dtype=np.float32)
        logits = head(img_t, feat_maps)
        if logits.dim() == 4:
            target = logits[:, class_index].mean()
        else:
            target = logits.mean()
        target.backward()
        grads = cam_feature.grad
        if grads is None:
            if logger:
                logger.info("Grad-CAM gradients missing; using zeros.")
            return np.zeros((hp, wp), dtype=np.float32)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * cam_feature).sum(dim=1)
        cam = torch.relu(cam)
        cam_map = cam.squeeze(0).detach().cpu().numpy()
        return normalize_map(cam_map)


def build_dashboard(
    output_path: str,
    rgb: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    entropy: np.ndarray,
    class_prob: np.ndarray,
    attn_cls: np.ndarray,
    attn_rollout: np.ndarray,
    overlay_pred: np.ndarray,
    overlay_attn: np.ndarray,
    layout: str = "4x3",
) -> None:
    """Create a dashboard plot with multiple subplots.

    Args:
        output_path (str): PNG output path.
        rgb (np.ndarray): RGB image.
        pred (np.ndarray): Prediction mask.
        confidence (np.ndarray): Confidence map.
        entropy (np.ndarray): Entropy map.
        class_prob (np.ndarray): Class probability map.
        attn_cls (np.ndarray): CLS attention map.
        attn_rollout (np.ndarray): Rollout attention map.
        overlay_pred (np.ndarray): RGB + prediction overlay.
        overlay_attn (np.ndarray): RGB + attention overlay.
        layout (str): Layout string, defaults to 4x3.
    """

    import matplotlib.pyplot as plt

    rows, cols = 4, 3
    if layout == "3x3":
        rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)
    panels = [
        ("RGB", rgb, None),
        ("Prediction", pred, "tab20"),
        ("Confidence", confidence, "magma"),
        ("Entropy", entropy, "magma"),
        ("Class Prob", class_prob, "magma"),
        ("Attn CLS", attn_cls, "viridis"),
        ("Attn Rollout", attn_rollout, "viridis"),
        ("Overlay Pred", overlay_pred, None),
        ("Overlay Attn", overlay_attn, None),
    ]
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
    """

    confidence = probs.max(axis=0)
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=0)
    if 0 <= class_index < probs.shape[0]:
        class_prob = probs[class_index]
    else:
        class_prob = np.zeros_like(confidence)
    return normalize_map(confidence), normalize_map(entropy), normalize_map(class_prob)


def overlay_heatmap(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    cmap: str = "magma",
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a heatmap onto an RGB image.

    Args:
        rgb (np.ndarray): RGB image (H, W, 3).
        heatmap (np.ndarray): Heatmap in [0, 1].
        cmap (str): Matplotlib colormap name.
        alpha (float): Overlay alpha.

    Returns:
        np.ndarray: Overlay image.
    """

    import matplotlib.cm as cm

    rgb_float = rgb.astype(np.float32) / 255.0
    colored = cm.get_cmap(cmap)(heatmap)[..., :3]
    overlay = (1 - alpha) * rgb_float + alpha * colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return overlay
