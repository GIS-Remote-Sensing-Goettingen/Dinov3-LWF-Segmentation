"""
Nano U-Net head with low-rank fidelity-aware projections and boundary branch.

Architecture overview:
- DINO path: each selected transformer feature passes through NanoFAPM
  (split-specific vs context branch, FiLM modulation, depthwise refinement).
- Decoder path: compact GroupNorm+GELU blocks with Dropout2d regularization.
- Deep supervision logits are emitted at H/8.
- Late RGB fusion: Spatial Prior Module (SPM) features are fused at H/4 and H/2.
- Boundary branch: a tiny 1x1 head predicts edge logits and is fused into
  segmentation logits as a lightweight boundary correction.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .base import SegmentationHead
from .unet_v2 import SpatialPriorModule


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
    """Ultra-light convolution block with spatial dropout regularization."""

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


class NanoFAPM(nn.Module):
    """Low-rank context modulation block for DINO feature projection.

    Examples:
        >>> block = NanoFAPM(in_channels=64, out_channels=32, context_rank=4)
        >>> x = torch.randn(1, 64, 8, 8)
        >>> tuple(block(x).shape)
        (1, 32, 8, 8)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_rank: int = 8,
    ) -> None:
        """Initialize the NanoFAPM block.

        Args:
            in_channels (int): Input channels from DINO features.
            out_channels (int): Output channels for decoder fusion.
            context_rank (int): Low-rank context width.
        """

        super().__init__()
        rank = max(1, int(context_rank))
        self.specific_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.specific_norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.specific_act = nn.GELU()

        self.context_proj = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rank, out_channels * 2, 1, bias=True),
            nn.Sigmoid(),
        )

        self.refine_depthwise = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )
        self.refine_pointwise = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.refine_gate = nn.Sigmoid()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and modulate DINO features.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Modulated output tensor.
        """

        specific = self.specific_act(self.specific_norm(self.specific_proj(x)))
        style = self.modulator(self.context_proj(x))
        gamma, beta = torch.chunk(style, 2, dim=1)
        modulated = specific * (1.0 + gamma) + beta
        refined = self.refine_depthwise(modulated)
        refined = self.refine_pointwise(refined)
        gate = self.refine_gate(refined)
        return gate * modulated + self.shortcut(x)


class DinoUNetNanoFAPMHead(SegmentationHead):
    """Compact decoder with NanoFAPM projections and boundary branch.

    Examples:
        >>> head = DinoUNetNanoFAPMHead(num_classes=2, dino_channels=64)
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
        >>> payload = head.forward_with_extras(img, feats)
        >>> tuple(payload["edge_logits"].shape)
        (1, 1, 256, 256)
    """

    def __init__(
        self,
        num_classes: int,
        dino_channels: int,
        boundary_fusion_weight: float = 0.1,
    ) -> None:
        """Build the Nano FAPM segmentation head.

        Args:
            num_classes (int): Number of output classes.
            dino_channels (int): DINO feature channel count per scale.
            boundary_fusion_weight (float): Edge-logit fusion scale.
        """

        super().__init__()
        self.boundary_fusion_weight = float(boundary_fusion_weight)
        self.spm = SpatialPriorModule(in_channels=3, base_channels=16)

        self.fapm1 = NanoFAPM(dino_channels, 64, context_rank=8)
        self.fapm2 = NanoFAPM(dino_channels, 64, context_rank=8)
        self.fapm3 = NanoFAPM(dino_channels, 32, context_rank=4)
        self.fapm4 = NanoFAPM(dino_channels, 32, context_rank=4)

        self.bottleneck = NanoDoubleConv(64, 64, dropout_rate=0.1)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = NanoDoubleConv(64 + 64, 64, dropout_rate=0.1)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = NanoDoubleConv(32 + 32, 32, dropout_rate=0.1)

        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv3 = NanoDoubleConv(32 + 32, 32, dropout_rate=0.1)

        self.ds_head = nn.Conv2d(32, num_classes, kernel_size=1)

        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv4 = NanoDoubleConv(32 + 32, 32, dropout_rate=0.1)

        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv5 = NanoDoubleConv(16 + 16, 16, dropout_rate=0.1)

        self.final_up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(16, 1, kernel_size=1)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Shared forward implementation.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale DINO features.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
            Fused logits, aux logits, edge logits, and intermediate tensors.
        """

        spm_h2, spm_h4 = self.spm(image)

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

        x = self.up4(x)
        x = self.conv4(self._concat(x, spm_h4))
        decoder_h4 = x

        x = self.up5(x)
        x = self.conv5(self._concat(x, spm_h2))
        decoder_h2 = x

        x = self.final_up(x)
        if x.shape[-2:] != image.shape[-2:]:
            x = F.interpolate(
                x, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        decoder_h1 = x
        mask_logits = self.final_conv(x)
        edge_logits = self.boundary_head(x)
        fused_logits = mask_logits + self.boundary_fusion_weight * edge_logits

        extras = {
            "mask_logits": mask_logits,
            "edge_logits": edge_logits,
            "bottleneck_features": bottleneck_feat,
            "decoder_h8": decoder_h8,
            "decoder_h4": decoder_h4,
            "decoder_h2": decoder_h2,
            "decoder_h1": decoder_h1,
        }
        return fused_logits, aux_logits, edge_logits, extras

    def forward_with_aux(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning main and auxiliary logits.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale backbone feature tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Main fused logits and aux logits.
        """

        logits, aux_logits, _, _ = self._forward_impl(image, features)
        return logits, aux_logits

    def forward_with_extras(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning logits and explainability intermediates.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale backbone feature tensors.

        Returns:
            dict[str, torch.Tensor]: Payload with logits, aux logits and extras.
        """

        logits, aux_logits, edge_logits, extras = self._forward_impl(image, features)
        payload = {
            "logits": logits,
            "aux_logits": aux_logits,
            "edge_logits": edge_logits,
        }
        payload.update(extras)
        return payload

    def forward(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Run the segmentation head and return fused logits.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): Multiscale backbone feature tensors.

        Returns:
            torch.Tensor: Fused segmentation logits.
        """

        logits, _, _, _ = self._forward_impl(image, features)
        return logits

