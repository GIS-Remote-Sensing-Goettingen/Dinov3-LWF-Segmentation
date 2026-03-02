"""
SegDINO-style lightweight multi-layer fusion head.

This baseline projects multiple DINO hidden-state grids with per-layer 1x1
convolutions, aligns them to a common token grid, concatenates, fuses, and
predicts segmentation logits with a shallow head.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .base import SegmentationHead

ActivationName = Literal["gelu", "relu"]


def _group_count(channels: int, max_groups: int = 16) -> int:
    """Return a valid GroupNorm group count dividing the channel size.

    Args:
        channels (int): Channel count.
        max_groups (int): Upper bound for number of groups.

    Returns:
        int: Largest valid group count not exceeding ``max_groups``.

    Examples:
        >>> _group_count(32, max_groups=16)
        16
    """

    upper = min(max(1, int(max_groups)), int(channels))
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _build_activation(name: ActivationName) -> nn.Module:
    """Create activation module.

    Args:
        name (ActivationName): Activation name.

    Returns:
        nn.Module: Activation layer.
    """

    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    return nn.GELU()


class DinoSegDinoLightHead(SegmentationHead):
    """Multi-layer lightweight segmentation head for frozen DINO features.

    Examples:
        >>> head = DinoSegDinoLightHead(
        ...     num_classes=2,
        ...     dino_channels=32,
        ...     num_layers=4,
        ...     proj_dim=16,
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
        proj_dim: int = 128,
        activation: ActivationName = "gelu",
        dropout: float = 0.0,
        strict_layers: bool = True,
    ) -> None:
        """Initialize the SegDINO-light head.

        Args:
            num_classes (int): Number of output classes.
            dino_channels (int): Channel count of each DINO feature map.
            num_layers (int): Expected number of feature layers.
            proj_dim (int): Per-layer projection width.
            activation (ActivationName): Activation type in fusion block.
            dropout (float): Dropout probability in fusion block.
            strict_layers (bool): Enforce exact feature-count matching.
        """

        super().__init__()
        self.expected_layers = max(1, int(num_layers))
        self.strict_layers = bool(strict_layers)
        proj_dim_i = int(proj_dim)
        norm_groups = _group_count(proj_dim_i, max_groups=16)
        self.proj = nn.ModuleList(
            [
                nn.Conv2d(int(dino_channels), proj_dim_i, kernel_size=1, bias=False)
                for _ in range(self.expected_layers)
            ]
        )
        self.proj_norm = nn.ModuleList(
            [nn.GroupNorm(norm_groups, proj_dim_i) for _ in range(self.expected_layers)]
        )
        fuse_layers: list[nn.Module] = [
            nn.Conv2d(
                self.expected_layers * proj_dim_i,
                proj_dim_i,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(norm_groups, proj_dim_i),
            _build_activation(activation),
        ]
        if float(dropout) > 0:
            fuse_layers.append(nn.Dropout2d(p=float(dropout)))
        self.fuse = nn.Sequential(*fuse_layers)
        self.out = nn.Conv2d(proj_dim_i, int(num_classes), kernel_size=1, bias=True)
        self._init_output_layer()

    def _init_output_layer(self) -> None:
        """Initialize output logits conservatively to avoid early saturation.

        Returns:
            None: Updates output-layer parameters in-place.
        """

        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def _select_features(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Select and validate feature maps.

        Args:
            features (list[torch.Tensor]): Input feature maps.

        Returns:
            list[torch.Tensor]: Feature maps to fuse.
        """

        if len(features) == 0:
            raise ValueError("DinoSegDinoLightHead requires at least one feature map.")
        if self.strict_layers and len(features) != self.expected_layers:
            raise ValueError(
                "DinoSegDinoLightHead expected "
                f"{self.expected_layers} features (matching model.layers), "
                f"got {len(features)}."
            )
        if len(features) < self.expected_layers:
            raise ValueError(
                "DinoSegDinoLightHead received fewer features than expected: "
                f"{len(features)} < {self.expected_layers}."
            )
        return features[: self.expected_layers]

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
        projected: list[torch.Tensor] = []
        for idx, feat in enumerate(selected):
            proj = self.proj_norm[idx](self.proj[idx](feat))
            if proj.shape[-2:] != (target_h, target_w):
                proj = F.interpolate(
                    proj,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(proj)
        fused = self.fuse(torch.cat(projected, dim=1))
        logits = self.out(fused)
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
