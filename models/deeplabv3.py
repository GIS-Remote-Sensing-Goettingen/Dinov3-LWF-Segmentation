"""Torchvision DeepLabV3 image-only baseline head.

This wraps the official torchvision ``deeplabv3_resnet50`` implementation so
the repo can compare a standard RGB-only baseline against the DINO-aware heads
without introducing a new dependency surface.
"""

from __future__ import annotations

from typing import Any

import torch
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50

from .base import SegmentationHead


class DeepLabV3Head(SegmentationHead):
    """Official torchvision DeepLabV3-ResNet50 adapter.

    The model consumes only RGB tensors and ignores the legacy DINO feature
    list expected by the shared head interface.
    """

    def __init__(self, num_classes: int, dino_channels: int) -> None:
        """Build the DeepLabV3 baseline.

        Args:
            num_classes (int): Number of semantic classes.
            dino_channels (int): Unused legacy registry argument.
        """

        super().__init__()
        _ = dino_channels
        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
            num_classes=num_classes,
            aux_loss=True,
        )

    def forward(
        self,
        image: torch.Tensor,
        features: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Return main and auxiliary logits on the input image grid.

        Args:
            image (torch.Tensor): Input RGB batch.
            features (list[torch.Tensor]): Ignored legacy DINO feature list.

        Returns:
            dict[str, Any]: Payload compatible with ``normalize_forward_output``.
        """

        _ = features
        outputs = self.model(image)
        return {
            "logits": outputs["out"],
            "aux_logits": outputs.get("aux"),
        }
