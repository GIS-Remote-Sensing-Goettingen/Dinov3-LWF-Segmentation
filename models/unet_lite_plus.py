"""
Enhanced lightweight DinoUNet decoder with safer resizing and stronger fusion.

Architecture overview:
- RGB prior path: lightweight Spatial Prior Module (SPM) at H/2 and H/4.
- DINO path: fidelity-aware projections from 4 backbone scales.
- Decoder: interpolation + projection upsampling (no transposed-conv resizing
  hacks), GroupNorm+GELU residual blocks, and deep supervision at H/8.
- Fusion: lightweight attention gate on H/4 SPM skip, then late H/2 fusion.
- Explainability hooks: optional intermediate tensors via `forward_with_extras`.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .base import SegmentationHead
from .unet_v2 import FidelityAwareProjection, SpatialPriorModule


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Pick a GroupNorm group count that divides channel size.

    Args:
        channels (int): Channel count requiring normalization groups.
        max_groups (int): Upper bound for group count.

    Returns:
        int: Valid group count dividing `channels`.
    """

    upper = min(max_groups, channels)
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    """Small residual block with GroupNorm and GELU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the residual convolution block.

        Args:
            in_channels (int): Input channel count.
            out_channels (int): Output channel count.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.GELU()
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        residual = self.proj(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act(out + residual)
        return out


class UpsampleProject(nn.Module):
    """Interpolation-based upsampling followed by projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the upsample projection module.

        Args:
            in_channels (int): Input channel count.
            out_channels (int): Output channel count.
        """

        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """Upsample to target size and project channels.

        Args:
            x (torch.Tensor): Input tensor.
            target_size (tuple[int, int]): Target spatial size (H, W).

        Returns:
            torch.Tensor: Upsampled and projected tensor.
        """

        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class AttentionGateLite(nn.Module):
    """Lightweight attention gate for filtering skip features."""

    def __init__(
        self, gate_channels: int, skip_channels: int, inter_channels: int
    ) -> None:
        """Initialize the attention gate.

        Args:
            gate_channels (int): Decoder gating tensor channels.
            skip_channels (int): Skip tensor channels.
            inter_channels (int): Intermediate channel width.
        """

        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(inter_channels), inter_channels),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(inter_channels), inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.GELU()

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Filter skip features with decoder gating signal.

        Args:
            gate (torch.Tensor): Decoder feature tensor.
            skip (torch.Tensor): Skip feature tensor.

        Returns:
            torch.Tensor: Gated skip tensor.
        """

        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(
                gate, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        psi = self.act(self.w_g(gate) + self.w_x(skip))
        psi = self.psi(psi)
        return skip * psi


class DinoUNetLitePlusHead(SegmentationHead):
    """Lite+ DinoUNet head with safer resizing and stronger lightweight fusion.

    Examples:
        >>> head = DinoUNetLitePlusHead(num_classes=2, dino_channels=64)
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
        """Initialize the Lite+ segmentation head.

        Args:
            num_classes (int): Number of segmentation classes.
            dino_channels (int): DINO feature channel count.
        """

        super().__init__()
        self.spm = SpatialPriorModule(in_channels=3, base_channels=16)

        # Slightly wider shallow projection to preserve texture cues.
        self.fapm1 = FidelityAwareProjection(dino_channels, 128)
        self.fapm2 = FidelityAwareProjection(dino_channels, 64)
        self.fapm3 = FidelityAwareProjection(dino_channels, 32)
        self.fapm4 = FidelityAwareProjection(dino_channels, 32)

        self.bottleneck = ResidualConvBlock(128, 128)
        self.up1 = UpsampleProject(128, 64)
        self.conv1 = ResidualConvBlock(64 + 64, 64)
        self.up2 = UpsampleProject(64, 32)
        self.conv2 = ResidualConvBlock(32 + 32, 32)
        self.up3 = UpsampleProject(32, 32)
        self.conv3 = ResidualConvBlock(32 + 32, 32)

        self.ds_head1 = nn.Conv2d(32, num_classes, kernel_size=1)

        self.up4 = UpsampleProject(32, 32)
        self.gate_h4 = AttentionGateLite(32, 32, 16)
        self.conv4 = ResidualConvBlock(32 + 32, 32)
        self.up5 = UpsampleProject(32, 16)
        self.conv5 = ResidualConvBlock(16 + 16, 16)
        self.final_up = UpsampleProject(16, 16)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def _concat(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Align tensors spatially and concatenate.

        Args:
            decoder (torch.Tensor): Decoder tensor.
            skip (torch.Tensor): Skip tensor.

        Returns:
            torch.Tensor: Concatenated tensor.
        """

        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(
                decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return torch.cat([decoder, skip], dim=1)

    def _forward_impl(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Shared forward implementation.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale backbone features.

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: Main
            logits, aux logits, and intermediate tensors.
        """

        spm_h2, spm_h4 = self.spm(image)
        d_shallow = self.fapm4(features[0])
        d_mid1 = self.fapm3(features[1])
        d_mid2 = self.fapm2(features[2])
        d_deep = self.fapm1(features[3])

        x = self.bottleneck(d_deep)
        bottleneck_feat = x

        x = self.up1(x, target_size=d_mid2.shape[-2:])
        x = self.conv1(self._concat(x, d_mid2))

        x = self.up2(x, target_size=d_mid1.shape[-2:])
        x = self.conv2(self._concat(x, d_mid1))

        x = self.up3(x, target_size=d_shallow.shape[-2:])
        x = self.conv3(self._concat(x, d_shallow))
        decoder_h8 = x
        ds_out = self.ds_head1(x)

        x = self.up4(x, target_size=spm_h4.shape[-2:])
        spm_h4_gated = self.gate_h4(x, spm_h4)
        x = self.conv4(self._concat(x, spm_h4_gated))
        decoder_h4 = x

        x = self.up5(x, target_size=spm_h2.shape[-2:])
        x = self.conv5(self._concat(x, spm_h2))
        decoder_h2 = x

        x = self.final_up(x, target_size=image.shape[-2:])
        logits = self.final_conv(x)

        extras = {
            "bottleneck_features": bottleneck_feat,
            "decoder_h8": decoder_h8,
            "decoder_h4": decoder_h4,
            "decoder_h2": decoder_h2,
        }
        return logits, ds_out, extras

    def forward_with_aux(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning main and deep-supervision logits.

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
        """Forward pass returning logits plus intermediate tensors.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale backbone feature tensors.

        Returns:
            dict[str, torch.Tensor]: Mapping with logits, aux logits, and extras.
        """

        logits, aux_logits, extras = self._forward_impl(image, features)
        payload = {"logits": logits, "aux_logits": aux_logits}
        payload.update(extras)
        return payload

    def forward(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward returning only main logits.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale backbone feature tensors.

        Returns:
            torch.Tensor: Main segmentation logits.
        """

        logits, _ = self.forward_with_aux(image, features)
        return logits
