"""Helpers for inference and test-time augmentation."""

from __future__ import annotations

import numpy as np
import torch


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
