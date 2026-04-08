"""Hugging Face Mask2Former semantic RGB-only baseline head.

This adapter keeps the repo's shared segmentation-head contract while training
with Mask2Former's native mask-classification loss. The wrapped model consumes
only RGB tensors and ignores the legacy DINO feature list.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)

from .base import SegmentationHead

_DEFAULT_MODEL_PATH = (
    "/user/davide.mattioli/u20330/Dinov3-LWF-Segmentation/weights/hf/facebook/"
    "mask2former-swin-base-ade-semantic"
)
_DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)


def build_mask2former_targets(
    labels: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert semantic labels into one mask/class set per sample.

    Args:
        labels (torch.Tensor): Semantic labels shaped ``(B, H, W)`` or ``(H, W)``.
        num_classes (int): Number of semantic classes.
        ignore_index (int | None): Optional ignore label.
        image_size (tuple[int, int] | None): Optional output size on the image grid.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: ``mask_labels`` and
        ``class_labels`` lists compatible with HF Mask2Former.

    Examples:
        >>> labels = torch.tensor([[[0, 1], [1, 255]]])
        >>> masks, classes = build_mask2former_targets(
        ...     labels,
        ...     num_classes=2,
        ...     ignore_index=255,
        ...     image_size=(4, 4),
        ... )
        >>> tuple(masks[0].shape), classes[0].tolist()
        ((2, 4, 4), [0, 1])
    """

    if labels.ndim == 2:
        labels = labels.unsqueeze(0)
    label_tensor = labels.long()
    if image_size is not None and label_tensor.shape[-2:] != tuple(image_size):
        label_tensor = F.interpolate(
            label_tensor.unsqueeze(1).float(),
            size=image_size,
            mode="nearest",
        ).squeeze(1).long()

    mask_labels: list[torch.Tensor] = []
    class_labels: list[torch.Tensor] = []
    for sample in label_tensor:
        valid = torch.ones_like(sample, dtype=torch.bool)
        if ignore_index is not None:
            valid = sample != int(ignore_index)
        valid_sample = sample[valid]
        present_classes = [
            int(class_id)
            for class_id in torch.unique(valid_sample).tolist()
            if 0 <= int(class_id) < int(num_classes)
        ]
        if not present_classes:
            empty_masks = sample.new_zeros((0, *sample.shape), dtype=torch.float32)
            empty_classes = sample.new_zeros((0,), dtype=torch.long)
            mask_labels.append(empty_masks)
            class_labels.append(empty_classes)
            continue
        sample_masks = [
            ((sample == class_id) & valid).to(dtype=torch.float32)
            for class_id in present_classes
        ]
        mask_labels.append(torch.stack(sample_masks, dim=0))
        class_labels.append(
            torch.tensor(present_classes, device=sample.device, dtype=torch.long)
        )
    return mask_labels, class_labels


def semantic_logits_from_mask2former_outputs(
    class_queries_logits: torch.Tensor,
    masks_queries_logits: torch.Tensor,
    *,
    target_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Convert Mask2Former query outputs into semantic logits.

    Args:
        class_queries_logits (torch.Tensor): Query-class logits with shape
            ``(B, Q, C + 1)``.
        masks_queries_logits (torch.Tensor): Query masks with shape
            ``(B, Q, H, W)``.
        target_size (tuple[int, int] | None): Optional output ``(H, W)``.

    Returns:
        torch.Tensor: Semantic logits shaped ``(B, C, H, W)``.

    Examples:
        >>> class_logits = torch.randn(2, 4, 3)
        >>> mask_logits = torch.randn(2, 4, 8, 8)
        >>> semantic_logits_from_mask2former_outputs(
        ...     class_logits,
        ...     mask_logits,
        ...     target_size=(16, 16),
        ... ).shape
        torch.Size([2, 2, 16, 16])
    """

    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()
    semantic_logits = torch.einsum("bqc,bqhw->bchw", masks_classes, masks_probs)
    if target_size is not None and semantic_logits.shape[-2:] != tuple(target_size):
        semantic_logits = F.interpolate(
            semantic_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    return semantic_logits


class Mask2FormerSemanticHead(SegmentationHead):
    """Real HF Mask2Former semantic baseline for RGB-only experiments."""

    def __init__(
        self,
        num_classes: int,
        dino_channels: int,
        *,
        model_name_or_path: str = _DEFAULT_MODEL_PATH,
        preprocessor_name_or_path: str | None = None,
        use_pretrained: bool = True,
    ) -> None:
        """Build the HF Mask2Former semantic baseline.

        Args:
            num_classes (int): Number of semantic output classes.
            dino_channels (int): Unused legacy registry argument.
            model_name_or_path (str): Local HF checkpoint/config directory.
            preprocessor_name_or_path (str | None): Local image-processor path.
            use_pretrained (bool): Whether to load pretrained weights from the
                local checkpoint directory.
        """

        super().__init__()
        _ = dino_channels
        self.num_classes = int(num_classes)
        self.model_name_or_path = str(model_name_or_path)
        self.preprocessor_name_or_path = str(
            preprocessor_name_or_path or model_name_or_path
        )
        self.use_pretrained = bool(use_pretrained)

        processor = AutoImageProcessor.from_pretrained(
            self.preprocessor_name_or_path,
            local_files_only=True,
        )
        image_mean = getattr(processor, "image_mean", None) or _DEFAULT_IMAGE_MEAN
        image_std = getattr(processor, "image_std", None) or _DEFAULT_IMAGE_STD
        mean_tensor = torch.tensor(image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        std_tensor = torch.tensor(image_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("image_mean", mean_tensor, persistent=False)
        self.register_buffer("image_std", std_tensor, persistent=False)

        if self.use_pretrained:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.model_name_or_path,
                local_files_only=True,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = Mask2FormerConfig.from_pretrained(
                self.model_name_or_path,
                local_files_only=True,
            )
            config.num_labels = self.num_classes
            self.model = Mask2FormerForUniversalSegmentation(config)

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the staged processor's channel normalization.

        Args:
            image (torch.Tensor): Input RGB tensor on the repo image grid.

        Returns:
            torch.Tensor: Channel-normalized image tensor.
        """

        if image.shape[1] != int(self.image_mean.shape[1]):
            raise ValueError(
                "Mask2Former expects %s input channels but received %s."
                % (int(self.image_mean.shape[1]), int(image.shape[1]))
            )
        return (image - self.image_mean) / self.image_std.clamp_min(1e-6)

    def _pixel_mask_for(self, image: torch.Tensor) -> torch.Tensor:
        """Return an all-valid pixel mask for unpadded tiles.

        Args:
            image (torch.Tensor): Input RGB tensor.

        Returns:
            torch.Tensor: Boolean all-ones mask with image spatial size.
        """

        batch, _, height, width = image.shape
        return torch.ones(
            batch,
            height,
            width,
            device=image.device,
            dtype=torch.bool,
        )

    def _forward_outputs(self, image: torch.Tensor) -> Any:
        """Run one forward pass without semantic targets.

        Args:
            image (torch.Tensor): Input RGB batch.

        Returns:
            Any: Raw HF Mask2Former output object.
        """

        normalized = self._normalize_image(image)
        return self.model(
            pixel_values=normalized,
            pixel_mask=self._pixel_mask_for(image),
            return_dict=True,
        )

    def _logits_from_outputs(self, outputs: Any, image: torch.Tensor) -> torch.Tensor:
        """Project query outputs back into semantic logits.

        Args:
            outputs (Any): Raw HF output object with query logits.
            image (torch.Tensor): Input RGB batch whose size defines the target
                semantic-logit grid.

        Returns:
            torch.Tensor: Semantic logits on the image grid.
        """

        return semantic_logits_from_mask2former_outputs(
            outputs.class_queries_logits,
            outputs.masks_queries_logits,
            target_size=(int(image.shape[-2]), int(image.shape[-1])),
        )

    def forward(
        self,
        image: torch.Tensor,
        features: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Return semantic logits on the input image grid.

        Args:
            image (torch.Tensor): Input RGB batch.
            features (list[torch.Tensor]): Ignored legacy DINO feature list.

        Returns:
            dict[str, Any]: Payload with semantic logits.
        """

        _ = features
        outputs = self._forward_outputs(image)
        return {"logits": self._logits_from_outputs(outputs, image)}

    def forward_with_native_loss(
        self,
        image: torch.Tensor,
        features: list[torch.Tensor],
        labels: torch.Tensor,
        *,
        ignore_index: int | None = None,
    ) -> dict[str, Any]:
        """Run Mask2Former with native mask-classification supervision.

        Args:
            image (torch.Tensor): Input RGB batch.
            features (list[torch.Tensor]): Ignored legacy DINO feature list.
            labels (torch.Tensor): Semantic labels on the repo label grid.
            ignore_index (int | None): Optional ignored label id.

        Returns:
            dict[str, Any]: Payload with semantic logits and native loss.
        """

        _ = features
        normalized = self._normalize_image(image)
        image_size = (int(image.shape[-2]), int(image.shape[-1]))
        mask_labels, class_labels = build_mask2former_targets(
            labels,
            num_classes=self.num_classes,
            ignore_index=ignore_index,
            image_size=image_size,
        )
        outputs = self.model(
            pixel_values=normalized,
            pixel_mask=self._pixel_mask_for(image),
            mask_labels=mask_labels,
            class_labels=class_labels,
            return_dict=True,
        )
        return {
            "logits": self._logits_from_outputs(outputs, image),
            "native_loss": outputs.loss,
        }
