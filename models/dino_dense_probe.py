"""
Dense linear probe head on top of frozen DINO feature maps.

This baseline follows the dense-probing spirit: apply a lightweight token-wise
classifier on a normalized last-layer feature grid, then upsample to image
resolution.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .base import SegmentationHead

NormType = Literal["batchnorm", "syncbn", "groupnorm", "none"]


def _group_count(channels: int, max_groups: int = 32) -> int:
    """Return a valid GroupNorm group count.

    Args:
        channels (int): Channel count.
        max_groups (int): Upper bound for number of groups.

    Returns:
        int: Valid group count dividing `channels`.

    Examples:
        >>> _group_count(32)
        32
        >>> _group_count(10)
        10
    """

    upper = min(max_groups, channels)
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _make_norm(
    channels: int,
    norm_type: NormType = "batchnorm",
    groupnorm_groups: int = 32,
) -> nn.Module:
    """Build normalization layer for dense probing.

    Args:
        channels (int): Input channel count.
        norm_type (NormType): Normalization type.
        groupnorm_groups (int): Group count hint for GroupNorm.

    Returns:
        nn.Module: Normalization layer.
    """

    key = str(norm_type).strip().lower()
    if key == "batchnorm":
        return nn.BatchNorm2d(channels)
    if key == "syncbn":
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return nn.SyncBatchNorm(channels)
        return nn.BatchNorm2d(channels)
    if key == "groupnorm":
        groups = min(max(1, int(groupnorm_groups)), int(channels))
        groups = _group_count(int(channels), max_groups=groups)
        return nn.GroupNorm(groups, channels)
    return nn.Identity()


class DinoDenseProbeHead(SegmentationHead):
    """Dense linear probe on the last DINO feature map.

    Examples:
        >>> head = DinoDenseProbeHead(num_classes=2, dino_channels=64)
        >>> img = torch.randn(1, 3, 128, 128)
        >>> feats = [torch.randn(1, 64, 8, 8), torch.randn(1, 64, 8, 8)]
        >>> out = head(img, feats)
        >>> tuple(out.shape)
        (1, 2, 128, 128)
    """

    def __init__(
        self,
        num_classes: int,
        dino_channels: int,
        norm_type: NormType = "batchnorm",
        groupnorm_groups: int = 32,
    ) -> None:
        """Initialize the dense probe head.

        Args:
            num_classes (int): Number of output classes.
            dino_channels (int): Channel count of DINO features.
            norm_type (NormType): Normalization layer type.
            groupnorm_groups (int): Group count hint when using GroupNorm.
        """

        super().__init__()
        self.norm = _make_norm(
            channels=int(dino_channels),
            norm_type=norm_type,
            groupnorm_groups=groupnorm_groups,
        )
        self.classifier = nn.Conv2d(
            int(dino_channels), int(num_classes), kernel_size=1, bias=True
        )

    def _forward_impl(
        self,
        image: torch.Tensor,
        features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute logits.

        Args:
            image (torch.Tensor): Input image tensor.
            features (list[torch.Tensor]): DINO feature tensors.

        Returns:
            torch.Tensor: Upsampled logits tensor.
        """

        if not features:
            raise ValueError("DinoDenseProbeHead requires at least one feature map.")
        tokens = features[-1]
        logits = self.classifier(self.norm(tokens))
        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(
                logits, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits

    def forward_with_aux(
        self,
        image: torch.Tensor,
        features: list[torch.Tensor],
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
