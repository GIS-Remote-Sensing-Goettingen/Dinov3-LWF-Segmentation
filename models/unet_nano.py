"""
DINO-only Nano U-Net head for aggressive capacity reduction.

Architecture overview:
- No RGB spatial-prior branch (SPM) to keep the model minimal and force
  reliance on DINO semantic features.
- Fidelity-aware projections compress 4 DINO scales into compact widths
  [64, 64, 32, 32] from deep to shallow.
- Decoder uses tiny GroupNorm+GELU blocks with Dropout2d regularization.
- Deep supervision logits are emitted at H/8.
- Final prediction upsamples from H/8 to full resolution.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .base import SegmentationHead
from .unet_v2 import FidelityAwareProjection


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Select a valid GroupNorm divisor for a channel count.

    Args:
        channels (int): Channel count to normalize.
        max_groups (int): Upper bound for the number of groups.

    Returns:
        int: Largest divisor of `channels` not exceeding `max_groups`.
    """

    upper = min(max_groups, channels)
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class NanoDoubleConv(nn.Module):
    """Ultra-light convolution block with spatial dropout regularization.

    Examples:
        >>> block = NanoDoubleConv(8, 16, dropout_rate=0.1)
        >>> x = torch.randn(1, 8, 16, 16)
        >>> tuple(block(x).shape)
        (1, 16, 16, 16)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize the Nano block.

        Args:
            in_channels (int): Input channel count.
            out_channels (int): Output channel count.
            dropout_rate (float): Drop probability for Dropout2d.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout2d(p=float(dropout_rate)),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block on an input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        return self.conv(x)


class DinoUNetNanoHead(SegmentationHead):
    """Aggressively compact decoder head targeting shape-biased segmentation.

    Examples:
        >>> head = DinoUNetNanoHead(num_classes=2, dino_channels=64)
        >>> img = torch.randn(1, 3, 256, 256)
        >>> feats = [
        ...     torch.randn(1, 64, 32, 32),
        ...     torch.randn(1, 64, 16, 16),
        ...     torch.randn(1, 64, 8, 8),
        ...     torch.randn(1, 64, 4, 4),
        ... ]
        >>> logits, aux = head.forward_with_aux(img, feats)
        >>> tuple(logits.shape), tuple(aux.shape)
        ((1, 2, 256, 256), (1, 2, 32, 32))
    """

    def __init__(self, num_classes: int, dino_channels: int) -> None:
        """Build the Nano segmentation head.

        Args:
            num_classes (int): Number of output classes.
            dino_channels (int): DINO feature channel count per scale.
        """

        super().__init__()
        self.fapm1 = FidelityAwareProjection(dino_channels, 64)
        self.fapm2 = FidelityAwareProjection(dino_channels, 64)
        self.fapm3 = FidelityAwareProjection(dino_channels, 32)
        self.fapm4 = FidelityAwareProjection(dino_channels, 32)

        self.bottleneck = NanoDoubleConv(64, 64, dropout_rate=0.1)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = NanoDoubleConv(64 + 64, 64, dropout_rate=0.1)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = NanoDoubleConv(32 + 32, 32, dropout_rate=0.1)

        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv3 = NanoDoubleConv(32 + 32, 32, dropout_rate=0.1)

        self.ds_head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.final_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def _concat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Align skip features to decoder resolution before concatenation.

        Args:
            x (torch.Tensor): Decoder tensor.
            skip (torch.Tensor): Skip tensor.

        Returns:
            torch.Tensor: Concatenated tensor.
        """

        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(
                skip, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return torch.cat([x, skip], dim=1)

    def _forward_impl(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Shared forward implementation.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale DINO features from shallow to deep.

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: Main logits,
            aux logits, and intermediate tensors.
        """

        d_shallow = self.fapm4(features[0])  # H/8
        d_mid1 = self.fapm3(features[1])  # H/16
        d_mid2 = self.fapm2(features[2])  # H/32
        d_deep = self.fapm1(features[3])  # H/64

        x = self.bottleneck(d_deep)
        bottleneck_feat = x

        x = self.up1(x)
        x = self.conv1(self._concat(x, d_mid2))

        x = self.up2(x)
        x = self.conv2(self._concat(x, d_mid1))

        x = self.up3(x)
        x = self.conv3(self._concat(x, d_shallow))
        decoder_h8 = x

        aux_logits = self.ds_head(x)

        x = self.final_up(x)
        if x.shape[-2:] != image.shape[-2:]:
            x = F.interpolate(
                x, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        logits = self.final_conv(x)

        extras = {
            "bottleneck_features": bottleneck_feat,
            "decoder_h8": decoder_h8,
        }
        return logits, aux_logits, extras

    def forward_with_aux(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning main and auxiliary logits.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale backbone feature tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Main logits and aux logits.
        """

        logits, aux_logits, _ = self._forward_impl(image, features)
        return logits, aux_logits

    def forward_with_extras(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning logits and explainability intermediates.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale backbone feature tensors.

        Returns:
            dict[str, torch.Tensor]: Payload with logits, aux logits, and extras.
        """

        logits, aux_logits, extras = self._forward_impl(image, features)
        payload = {"logits": logits, "aux_logits": aux_logits}
        payload.update(extras)
        return payload

    def forward(self, image: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass returning only main logits.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale backbone feature tensors.

        Returns:
            torch.Tensor: Main segmentation logits.
        """

        logits, _ = self.forward_with_aux(image, features)
        return logits
