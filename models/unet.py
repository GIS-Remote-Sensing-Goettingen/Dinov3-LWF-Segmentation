"""Standard image-only U-Net segmentation head.

The topology follows the original U-Net design from Ronneberger et al.:
a symmetric contracting/expanding path with double 3x3 convolutions,
2x2 max-pooling, transpose-convolution upsampling, and skip concatenations.

For pipeline compatibility this implementation keeps padded 3x3 convolutions,
so logits stay on the same spatial grid as the input image instead of using
the paper's valid-convolution cropping scheme.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .base import SegmentationHead


class DoubleConv(nn.Module):
    """Apply the paper-style pair of 3x3 convolutions with ReLU activations.

    >>> block = DoubleConv(3, 8)
    >>> tuple(block(torch.randn(1, 3, 16, 16)).shape)
    (1, 8, 16, 16)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build one U-Net convolution block.

        Args:
            in_channels (int): Input channel count.
            out_channels (int): Output channel count.
        """

        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the refined feature map.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Refined tensor after two convolutions.
        """

        return self.layers(x)


class UpBlock(nn.Module):
    """Upsample, concatenate the encoder skip, and refine the feature map.

    >>> up = UpBlock(32, 16)
    >>> x = torch.randn(1, 32, 8, 8)
    >>> skip = torch.randn(1, 16, 16, 16)
    >>> tuple(up(x, skip).shape)
    (1, 16, 16, 16)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build one expanding-path block.

        Args:
            in_channels (int): Decoder input channel count.
            out_channels (int): Output channel count after fusion.
        """

        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Fuse decoder and encoder features on one spatial scale.

        Args:
            x (torch.Tensor): Decoder feature map to upsample.
            skip (torch.Tensor): Encoder skip tensor on the target scale.

        Returns:
            torch.Tensor: Refined decoder tensor after skip fusion.
        """

        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(
                skip,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class DinoUNetHead(SegmentationHead):
    """Standard image-only U-Net with the registry-compatible head name.

    The head keeps the historical ``DinoUNetHead`` class name so existing
    registry wiring does not change, but it ignores the DINO feature list and
    operates only on the RGB image tensor.

    >>> head = DinoUNetHead(num_classes=2, dino_channels=1024)
    >>> image = torch.randn(1, 3, 128, 128)
    >>> logits = head(image, [])
    >>> tuple(logits.shape)
    (1, 2, 128, 128)
    """

    def __init__(self, num_classes: int, dino_channels: int) -> None:
        """Build the standard U-Net head.

        Args:
            num_classes (int): Number of output classes.
            dino_channels (int): Unused legacy registry argument.
        """

        super().__init__()
        _ = dino_channels
        encoder_channels = [64, 128, 256, 512]
        bottleneck_channels = 1024
        self.encoder_blocks = nn.ModuleList(
            [
                DoubleConv(3, encoder_channels[0]),
                DoubleConv(encoder_channels[0], encoder_channels[1]),
                DoubleConv(encoder_channels[1], encoder_channels[2]),
                DoubleConv(encoder_channels[2], encoder_channels[3]),
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(encoder_channels[-1], bottleneck_channels)
        self.decoder_blocks = nn.ModuleList(
            [
                UpBlock(bottleneck_channels, encoder_channels[3]),
                UpBlock(encoder_channels[3], encoder_channels[2]),
                UpBlock(encoder_channels[2], encoder_channels[1]),
                UpBlock(encoder_channels[1], encoder_channels[0]),
            ]
        )
        self.final_conv = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(
        self,
        image: torch.Tensor,
        features: List[torch.Tensor],
    ) -> torch.Tensor:
        """Segment the image with a standard U-Net forward pass.

        Args:
            image (torch.Tensor): Input image tensor shaped ``(B, 3, H, W)``.
            features (List[torch.Tensor]): Ignored legacy DINO feature list.

        Returns:
            torch.Tensor: Full-resolution class logits.
        """

        _ = features
        skips: list[torch.Tensor] = []
        x = image
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder(x, skip)
        return self.final_conv(x)
