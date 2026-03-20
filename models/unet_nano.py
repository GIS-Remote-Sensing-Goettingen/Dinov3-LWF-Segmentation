"""
Nano U-Net head with DINO backbone features and late RGB prior fusion.

Architecture overview:
- DINO path: fidelity-aware projections compress up to 4 backbone scales into
  compact widths [64, 64, 32, 32] from deep to shallow.
- Decoder path: tiny GroupNorm+GELU blocks with Dropout2d regularization.
- Deep supervision logits are emitted at H/8.
- Late RGB fusion: Spatial Prior Module (SPM) features are fused at H/4 and H/2
  to recover boundary detail without widening the deep decoder. When fewer than
  4 DINO layers are configured, missing shallow skip sources are dropped.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .base import SegmentationHead
from .unet_v2 import FidelityAwareProjection, SpatialPriorModule


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
    """Aggressively compact decoder head with late RGB boundary fusion.

    The head accepts 1 to 4 DINO feature maps ordered from shallowest to
    deepest. With fewer than 4 maps, the deepest available feature still drives
    the bottleneck path while missing shallow skip concatenations are omitted.

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
        self.spm = SpatialPriorModule(in_channels=3, base_channels=16)

        self.fapm1 = FidelityAwareProjection(dino_channels, 64)
        self.fapm2 = FidelityAwareProjection(dino_channels, 64)
        self.fapm3 = FidelityAwareProjection(dino_channels, 32)
        self.fapm4 = FidelityAwareProjection(dino_channels, 32)

        self.bottleneck = NanoDoubleConv(64, 64, dropout_rate=0.1)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = NanoDoubleConv(64 + 64, 64, dropout_rate=0.1)
        self.conv1_no_skip = NanoDoubleConv(64, 64, dropout_rate=0.1)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = NanoDoubleConv(32 + 32, 32, dropout_rate=0.1)
        self.conv2_no_skip = NanoDoubleConv(32, 32, dropout_rate=0.1)

        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv3 = NanoDoubleConv(32 + 32, 32, dropout_rate=0.1)
        self.conv3_no_skip = NanoDoubleConv(32, 32, dropout_rate=0.1)

        self.ds_head = nn.Conv2d(32, num_classes, kernel_size=1)

        # Late fusion with RGB priors (mirrors lite/lite+ strategy, kept minimal).
        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv4 = NanoDoubleConv(32 + 32, 32, dropout_rate=0.1)

        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv5 = NanoDoubleConv(16 + 16, 16, dropout_rate=0.1)

        self.final_up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

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

    def _resolve_dino_connections(
        self,
        features: List[torch.Tensor],
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        """Map 1-4 input features onto the fixed Nano decoder roles.

        Args:
            features (List[torch.Tensor]): DINO features ordered shallowest to deepest.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            Deep bottleneck feature plus optional mid2, mid1, and shallow skips.
        """

        feature_count = len(features)
        if feature_count < 1 or feature_count > 4:
            raise ValueError(
                "DinoUNetNanoHead requires 1 to 4 DINO feature maps, "
                f"got {feature_count}."
            )
        d_deep = self.fapm1(features[-1])  # deepest selected layer
        d_mid2 = self.fapm2(features[-2]) if feature_count >= 2 else None
        d_mid1 = self.fapm3(features[-3]) if feature_count >= 3 else None
        d_shallow = self.fapm4(features[-4]) if feature_count >= 4 else None
        return d_deep, d_mid2, d_mid1, d_shallow

    def _decode_stage(
        self,
        x: torch.Tensor,
        upsampler: nn.Module,
        with_skip: nn.Module,
        no_skip: nn.Module,
        skip: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one decoder stage with an optional DINO skip connection.

        Args:
            x (torch.Tensor): Decoder activation before upsampling.
            upsampler (nn.Module): Stage upsampling module.
            with_skip (nn.Module): Convolution block used when a DINO skip exists.
            no_skip (nn.Module): Convolution block used when the DINO skip is absent.
            skip (torch.Tensor | None): Optional DINO skip tensor for this stage.
        """

        x = upsampler(x)
        if skip is None:
            return no_skip(x)
        return with_skip(self._concat(x, skip))

    def _forward_impl(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Shared forward implementation.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): One to four DINO features from shallow
                to deep.

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: Main logits,
            aux logits, and intermediate tensors.
        """

        spm_h2, spm_h4 = self.spm(image)  # H/2 and H/4

        d_deep, d_mid2, d_mid1, d_shallow = self._resolve_dino_connections(features)

        x = self.bottleneck(d_deep)
        bottleneck_feat = x

        x = self._decode_stage(x, self.up1, self.conv1, self.conv1_no_skip, d_mid2)
        x = self._decode_stage(x, self.up2, self.conv2, self.conv2_no_skip, d_mid1)
        x = self._decode_stage(x, self.up3, self.conv3, self.conv3_no_skip, d_shallow)
        decoder_h8 = x

        aux_logits = self.ds_head(x)

        # Late RGB fusion at H/4.
        x = self.up4(x)
        x = self.conv4(self._concat(x, spm_h4))
        decoder_h4 = x

        # Late RGB fusion at H/2.
        x = self.up5(x)
        x = self.conv5(self._concat(x, spm_h2))
        decoder_h2 = x

        x = self.final_up(x)
        if x.shape[-2:] != image.shape[-2:]:
            x = F.interpolate(
                x, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        logits = self.final_conv(x)

        extras = {
            "bottleneck_features": bottleneck_feat,
            "decoder_h8": decoder_h8,
            "decoder_h4": decoder_h4,
            "decoder_h2": decoder_h2,
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

    def forward(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass returning only main logits.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale backbone feature tensors.

        Returns:
            torch.Tensor: Main segmentation logits.
        """

        logits, _ = self.forward_with_aux(image, features)
        return logits
