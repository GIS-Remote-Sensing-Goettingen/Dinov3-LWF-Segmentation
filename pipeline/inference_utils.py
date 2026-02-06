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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CLS and rollout attention maps for a single image.

    Args:
        image_hw3 (np.ndarray): Input image in HWC format.
        backbone (torch.nn.Module): DINO backbone.
        processor (object): Image processor.
        device (torch.device): Device for inference.
        ps (int): Patch size for the backbone.
        logger (Any | None): Optional logger for fallback events.

    Returns:
        tuple[np.ndarray, np.ndarray]: CLS and rollout attention maps.
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
            return zeros, zeros
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
        return normalize_map(cls_map), normalize_map(rollout_map)
    except Exception:
        if logger:
            logger.info("Attention extraction failed; using zeros.")
        zeros = np.zeros((hp, wp), dtype=np.float32)
        return zeros, zeros


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
