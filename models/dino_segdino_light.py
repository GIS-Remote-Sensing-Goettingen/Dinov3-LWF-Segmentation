"""
Paper-like lightweight SegDINO decoder on frozen DINO features.

This head reformulates each selected DINO hidden-state grid to a shared channel
width, aligns them to a common token grid, concatenates them, and applies a
minimal per-pixel MLP decoder to predict segmentation logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import SegmentationHead

_REFORM_CHANNELS = 128


class DinoSegDinoLightHead(SegmentationHead):
    """Minimal paper-like SegDINO decoder for frozen multi-layer DINO features.

    Examples:
        >>> head = DinoSegDinoLightHead(
        ...     num_classes=2,
        ...     dino_channels=32,
        ...     num_layers=4,
        ... )
        >>> img = torch.randn(1, 3, 128, 128)
        >>> feats = [torch.randn(1, 32, 8, 8) for _ in range(4)]
        >>> out = head(img, feats)
        >>> tuple(out.shape)
        (1, 2, 128, 128)
    """

    def __init__(
        self,
        num_classes: int,
        dino_channels: int,
        num_layers: int,
    ) -> None:
        """Initialize the fixed paper-like SegDINO decoder.

        Args:
            num_classes (int): Number of output classes.
            dino_channels (int): Channel count of each DINO feature map.
            num_layers (int): Expected number of selected DINO layers.
        """

        super().__init__()
        self.expected_layers = max(1, int(num_layers))
        reform_channels = int(_REFORM_CHANNELS)
        self.reform = nn.ModuleList(
            [
                nn.Conv2d(
                    int(dino_channels),
                    reform_channels,
                    kernel_size=1,
                    bias=True,
                )
                for _ in range(self.expected_layers)
            ]
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(
                self.expected_layers * reform_channels,
                reform_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(reform_channels, int(num_classes), kernel_size=1, bias=True),
        )

    def _select_features(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Validate the selected DINO feature maps.

        Args:
            features (list[torch.Tensor]): Input feature maps.

        Returns:
            list[torch.Tensor]: Feature maps to decode.

        Raises:
            ValueError: If the feature count does not match ``model.layers``.
        """

        if len(features) != self.expected_layers:
            raise ValueError(
                "DinoSegDinoLightHead expected "
                f"{self.expected_layers} features (matching model.layers), "
                f"got {len(features)}."
            )
        return features

    def _forward_impl(
        self, image: torch.Tensor, features: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute segmentation logits.

        Args:
            image (torch.Tensor): Input image tensor.
            features (list[torch.Tensor]): DINO feature tensors.

        Returns:
            torch.Tensor: Upsampled logits tensor.
        """

        selected = self._select_features(features)
        target_h, target_w = selected[0].shape[-2:]
        reformed: list[torch.Tensor] = []
        for idx, feat in enumerate(selected):
            reform = self.reform[idx](feat)
            if reform.shape[-2:] != (target_h, target_w):
                reform = F.interpolate(
                    reform,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            reformed.append(reform)
        logits = self.mlp(torch.cat(reformed, dim=1))
        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(
                logits, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits

    def forward_with_aux(
        self, image: torch.Tensor, features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, None]:
        """Forward pass returning logits and aux placeholder.

        Args:
            image (torch.Tensor): Input image tensor.
            features (list[torch.Tensor]): DINO feature tensors.

        Returns:
            tuple[torch.Tensor, None]: Logits and no auxiliary logits.
        """

        return self._forward_impl(image, features), None

    def forward(
        self, image: torch.Tensor, features: list[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass returning logits only.

        Args:
            image (torch.Tensor): Input image tensor.
            features (list[torch.Tensor]): DINO feature tensors.

        Returns:
            torch.Tensor: Logits tensor.
        """

        return self._forward_impl(image, features)
