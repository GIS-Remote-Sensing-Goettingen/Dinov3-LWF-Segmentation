"""
Topology-aware DINO U-Net head with learned layer fusion and boundary gating.

Architecture overview:
- DINO layer fusion: spatially varying softmax over selected hidden-state layers.
- Low-rank projection adapters: LoRA-style 1x1 residual updates on frozen base
  projections for parameter-efficient DINO-to-decoder transfer.
- Pyramid synthesis: ViT layer outputs share one patch grid, so multi-resolution
  taps are created with learned strided downsampling.
- Decoder: compact GN+GELU blocks with late RGB prior fusion at H/4 and H/2.
- Shape stream: boundary logits and a gated refinement map modulate final
  decoder features before segmentation logits.
- Topology stream: skeleton logits at H/8 for topology-aware supervision.
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn.functional as F
from torch import nn

from .base import SegmentationHead
from .unet_v2 import SpatialPriorModule


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Select a valid GroupNorm divisor for a channel count.

    Args:
        channels (int): Channel count to normalize.
        max_groups (int): Upper bound for group count.

    Returns:
        int: Largest divisor of `channels` not exceeding `max_groups`.

    Examples:
        >>> _group_count(32)
        8
        >>> _group_count(10)
        5
    """

    upper = min(max_groups, channels)
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class NanoDoubleConv(nn.Module):
    """Compact convolution block with GroupNorm + GELU + optional Dropout2d.

    Examples:
        >>> block = NanoDoubleConv(8, 16, dropout_rate=0.1)
        >>> tuple(block(torch.randn(1, 8, 4, 4)).shape)
        (1, 16, 4, 4)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize the block.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            dropout_rate (float): Dropout2d probability.
        """

        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=float(dropout_rate)))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.GELU(),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        return self.block(x)


class LowRankAdapterProjection(nn.Module):
    """LoRA-style projection: frozen base 1x1 + low-rank residual update.

    Examples:
        >>> proj = LowRankAdapterProjection(64, 32, rank=4, alpha=8.0)
        >>> x = torch.randn(1, 64, 8, 8)
        >>> tuple(proj(x).shape)
        (1, 32, 8, 8)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 8,
        alpha: float = 16.0,
        freeze_base: bool = True,
        lora_dropout: float = 0.0,
    ) -> None:
        """Initialize the low-rank projection.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            rank (int): Low-rank width.
            alpha (float): LoRA scale numerator.
            freeze_base (bool): Freeze base projection weights.
            lora_dropout (float): Dropout applied to LoRA path input.
        """

        super().__init__()
        self.rank = max(1, int(rank))
        self.scale = float(alpha) / float(self.rank)
        self.base = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.lora_down = nn.Conv2d(in_channels, self.rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(self.rank, out_channels, kernel_size=1, bias=False)
        self.dropout = (
            nn.Dropout2d(p=float(lora_dropout)) if lora_dropout > 0 else nn.Identity()
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.GELU()

        nn.init.zeros_(self.lora_up.weight)
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input with a frozen-base + low-rank residual update.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Projected output.
        """

        base = self.base(x)
        update = self.lora_up(self.lora_down(self.dropout(x))) * self.scale
        return self.act(self.norm(base + update))


class LayerFusionMixer(nn.Module):
    """Learn spatially varying layer weights from joint multi-layer context.

    Examples:
        >>> mixer = LayerFusionMixer(channels=16, max_layers=4, hidden=8)
        >>> feats = [torch.randn(2, 16, 4, 4) for _ in range(3)]
        >>> fused, alpha, means = mixer(feats)
        >>> fused.shape, alpha.shape, means.shape
        (torch.Size([2, 16, 4, 4]), torch.Size([2, 3, 4, 4]), torch.Size([3]))
    """

    def __init__(self, channels: int, max_layers: int, hidden: int = 64) -> None:
        """Initialize the fusion module.

        Args:
            channels (int): Feature channel width.
            max_layers (int): Maximum number of layers to fuse.
            hidden (int): Hidden width for score prediction.
        """

        super().__init__()
        self.channels = int(channels)
        self.max_layers = max(1, int(max_layers))
        self.hidden = max(1, int(hidden))
        self.net = nn.Sequential(
            nn.Conv2d(
                self.max_layers * self.channels,
                self.hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(self.hidden, self.max_layers, kernel_size=1, bias=True),
        )

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse layer features.

        Args:
            features (list[torch.Tensor]): Layer feature tensors with identical shape.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Fused feature map, layer weight maps, and layer mean weights.
        """

        layer_count = len(features)
        if layer_count == 0:
            raise ValueError("LayerFusionMixer requires at least one feature map.")
        if layer_count > self.max_layers:
            raise ValueError(
                f"Received {layer_count} layers, but max_layers={self.max_layers}."
            )
        c0 = int(features[0].shape[1])
        h0, w0 = int(features[0].shape[2]), int(features[0].shape[3])
        if c0 != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels per layer, got {c0}."
            )
        for idx, feat in enumerate(features[1:], start=1):
            if int(feat.shape[1]) != self.channels:
                raise ValueError(
                    f"Layer {idx} has {int(feat.shape[1])} channels; "
                    f"expected {self.channels}."
                )
            if tuple(feat.shape[-2:]) != (h0, w0):
                raise ValueError(
                    f"Layer {idx} has shape {tuple(feat.shape[-2:])}; "
                    f"expected {(h0, w0)}."
                )
        if layer_count == 1:
            alpha = torch.ones(
                features[0].shape[0],
                1,
                features[0].shape[2],
                features[0].shape[3],
                device=features[0].device,
                dtype=features[0].dtype,
            )
            mean = torch.ones(1, device=features[0].device, dtype=features[0].dtype)
            return features[0], alpha, mean
        mixer_in = torch.cat(features, dim=1)
        if layer_count < self.max_layers:
            pad = features[0].new_zeros(
                features[0].shape[0],
                (self.max_layers - layer_count) * self.channels,
                h0,
                w0,
            )
            mixer_in = torch.cat([mixer_in, pad], dim=1)
        scores = self.net(mixer_in)
        if layer_count < self.max_layers:
            scores[:, layer_count:, :, :] = -1e9
        scores = scores[:, :layer_count, :, :]
        alpha = torch.softmax(scores, dim=1)
        stacked = torch.stack(features, dim=1)
        fused = (stacked * alpha.unsqueeze(2)).sum(dim=1)
        layer_means = alpha.mean(dim=(0, 2, 3))
        return fused, alpha, layer_means


class DinoUNetTopoFusionHead(SegmentationHead):
    """Thin-structure head with learned DINO fusion and topology stream.

    Examples:
        >>> head = DinoUNetTopoFusionHead(num_classes=2, dino_channels=64)
        >>> img = torch.randn(1, 3, 256, 256)
        >>> feats = [torch.randn(1, 64, 16, 16) for _ in range(4)]
        >>> payload = head.forward_with_extras(img, feats)
        >>> tuple(payload["logits"].shape), tuple(payload["aux_logits"].shape)
        ((1, 2, 256, 256), (1, 2, 32, 32))
        >>> tuple(payload["skeleton_logits"].shape)
        (1, 1, 32, 32)
    """

    def __init__(
        self,
        num_classes: int,
        dino_channels: int,
        fusion_hidden: int = 64,
        layer_fusion_hidden: int | None = None,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_freeze_base: bool = True,
        boundary_gate_scale: float = 0.1,
        boundary_gate_clamp: bool = True,
        max_layers_for_fusion: int = 6,
        layer_mix_maps_enable: bool = False,
    ) -> None:
        """Initialize the head.

        Args:
            num_classes (int): Number of segmentation classes.
            dino_channels (int): Input DINO feature channels.
            fusion_hidden (int): Hidden channels for fused DINO map.
            layer_fusion_hidden (int | None): Layer-mixer hidden width.
                Defaults to min(128, 4 * fusion_hidden).
            lora_rank (int): Low-rank adapter width.
            lora_alpha (float): LoRA scaling numerator.
            lora_dropout (float): LoRA path dropout.
            lora_freeze_base (bool): Freeze base 1x1 projection.
            boundary_gate_scale (float): Boundary gate strength.
            boundary_gate_clamp (bool): Clamp gate multiplier to [1, 1+s].
            max_layers_for_fusion (int): Maximum layers consumed from input list.
            layer_mix_maps_enable (bool): Include full alpha maps in payload.
        """

        super().__init__()
        fusion_hidden = int(fusion_hidden)
        mixer_hidden = (
            min(128, max(8, 4 * fusion_hidden))
            if layer_fusion_hidden is None
            else max(1, int(layer_fusion_hidden))
        )
        self.boundary_gate_scale = max(0.0, float(boundary_gate_scale))
        self.boundary_gate_clamp = bool(boundary_gate_clamp)
        self.max_layers_for_fusion = max(1, int(max_layers_for_fusion))
        self.layer_mix_maps_enable = bool(layer_mix_maps_enable)

        self.spm = SpatialPriorModule(in_channels=3, base_channels=16)
        self.layer_proj = LowRankAdapterProjection(
            in_channels=dino_channels,
            out_channels=fusion_hidden,
            rank=lora_rank,
            alpha=lora_alpha,
            freeze_base=lora_freeze_base,
            lora_dropout=lora_dropout,
        )
        self.layer_mixer = LayerFusionMixer(
            channels=fusion_hidden,
            max_layers=self.max_layers_for_fusion,
            hidden=mixer_hidden,
        )
        self.mix_refine = NanoDoubleConv(fusion_hidden, fusion_hidden, dropout_rate=0.05)

        self.tap_shallow = nn.Conv2d(fusion_hidden, 32, kernel_size=1, bias=False)
        self.tap_mid = nn.Sequential(
            nn.Conv2d(
                fusion_hidden,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(64), 64),
            nn.GELU(),
        )
        self.tap_deep = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.GELU(),
        )

        self.bottleneck = NanoDoubleConv(64, 64, dropout_rate=0.1)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = NanoDoubleConv(64 + 64, 64, dropout_rate=0.1)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = NanoDoubleConv(32 + 32, 32, dropout_rate=0.1)

        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv3 = NanoDoubleConv(32, 32, dropout_rate=0.1)
        self.ds_head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.skeleton_head = nn.Conv2d(32, 1, kernel_size=1)

        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv4 = NanoDoubleConv(32 + 32, 32, dropout_rate=0.1)

        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv5 = NanoDoubleConv(16 + 16, 16, dropout_rate=0.1)

        self.final_up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.final_refine = NanoDoubleConv(16, 16, dropout_rate=0.0)
        self.edge_feat = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.edge_feat_norm = nn.GroupNorm(_group_count(16), 16)
        self.edge_feat_act = nn.GELU()
        self.edge_logits = nn.Conv2d(16, 1, kernel_size=1)
        self.boundary_gate = nn.Conv2d(16, 1, kernel_size=1)
        self.mask_logits = nn.Conv2d(16, num_classes, kernel_size=1)

    def _concat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Align and concatenate skip features.

        Args:
            x (torch.Tensor): Decoder tensor.
            skip (torch.Tensor): Skip tensor.

        Returns:
            torch.Tensor: Concatenated tensor.
        """

        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(
                skip,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat([x, skip], dim=1)

    def _build_dino_pyramid(
        self,
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, fuse, and downsample DINO features into decoder taps.

        Args:
            features (list[torch.Tensor]): Input DINO feature tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Deep, mid, shallow taps; alpha map; and per-layer mean alpha.
        """

        selected = list(features[: self.max_layers_for_fusion])
        if len(selected) == 0:
            raise ValueError(
                "DinoUNetTopoFusionHead requires at least one DINO feature map."
            )
        projected = [self.layer_proj(feat) for feat in selected]
        fused, alpha, alpha_mean = self.layer_mixer(projected)
        fused = self.mix_refine(fused)
        shallow = self.tap_shallow(fused)
        mid = self.tap_mid(fused)
        deep = self.tap_deep(mid)
        return deep, mid, shallow, alpha, alpha_mean

    def _forward_impl(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> dict[str, Any]:
        """Run shared forward implementation.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): DINO feature tensors.

        Returns:
            dict[str, Any]: Output payload with logits and intermediates.
        """

        spm_h2, spm_h4 = self.spm(image)
        deep, mid, shallow, alpha, alpha_mean = self._build_dino_pyramid(features)

        x = self.bottleneck(deep)
        x = self.conv1(self._concat(self.up1(x), mid))
        x = self.conv2(self._concat(self.up2(x), shallow))
        x = self.conv3(self.up3(x))

        aux_logits = self.ds_head(x)
        skeleton_logits = self.skeleton_head(x)
        expected_aux = (int(shallow.shape[-2]) * 2, int(shallow.shape[-1]) * 2)
        if tuple(aux_logits.shape[-2:]) != expected_aux:
            raise RuntimeError(
                "Aux resolution mismatch: expected "
                f"{expected_aux}, got {tuple(aux_logits.shape[-2:])}. "
                "Ensure DINO features and decoder taps share the same effective patch grid."
            )

        x = self.conv4(self._concat(self.up4(x), spm_h4))
        x = self.conv5(self._concat(self.up5(x), spm_h2))
        x = self.final_up(x)
        if x.shape[-2:] != image.shape[-2:]:
            x = F.interpolate(
                x, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        x = self.final_refine(x)

        edge_feat = self.edge_feat_act(self.edge_feat_norm(self.edge_feat(x)))
        edge_logits = self.edge_logits(edge_feat)
        gate = torch.sigmoid(self.boundary_gate(edge_feat))
        multiplier = 1.0 + self.boundary_gate_scale * gate
        if self.boundary_gate_clamp:
            multiplier = torch.clamp(multiplier, min=1.0, max=1.0 + self.boundary_gate_scale)
        x_refined = x * multiplier
        logits = self.mask_logits(x_refined)

        payload: dict[str, Any] = {
            "logits": logits,
            "aux_logits": aux_logits,
            "edge_logits": edge_logits,
            "skeleton_logits": skeleton_logits,
            "layer_mix_weights_mean": alpha_mean,
            "gate_mean": float(gate.mean().item()),
            "gate_std": float(gate.std(unbiased=False).item()),
        }
        if self.layer_mix_maps_enable:
            payload["layer_mix_maps"] = alpha
        return payload

    def forward_with_aux(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass and return main + auxiliary logits.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): DINO feature tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Main and auxiliary logits.
        """

        payload = self._forward_impl(image, features)
        return payload["logits"], payload["aux_logits"]

    def forward_with_extras(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> dict[str, Any]:
        """Run forward pass and return full payload.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): DINO feature tensors.

        Returns:
            dict[str, Any]: Output payload with optional explainability tensors.
        """

        return self._forward_impl(image, features)

    def forward(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Run the segmentation head.

        Args:
            image (torch.Tensor): Input image tensor.
            features (List[torch.Tensor]): DINO feature tensors.

        Returns:
            torch.Tensor: Segmentation logits.
        """

        return self._forward_impl(image, features)["logits"]
