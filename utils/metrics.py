"""
Utilities for computing segmentation metrics such as IoU and Dice.
"""

from __future__ import annotations

from typing import Dict

import torch


class SegmentationMetrics:
    """
    Maintains a confusion matrix to compute IoU/Dice scores.

    >>> metrics = SegmentationMetrics(num_classes=2)
    >>> preds = torch.tensor([[0, 1], [1, 1]])
    >>> targets = torch.tensor([[0, 1], [0, 1]])
    >>> metrics.update(preds, targets)
    >>> summary = metrics.compute()
    >>> round(float(summary["per_class_iou"][0]), 3)
    0.5
    """

    def __init__(self, num_classes: int) -> None:
        """Initialize the metric accumulator.

        Args:
            num_classes (int): Number of segmentation classes.
        """

        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update the confusion matrix with a batch.

        Args:
            preds (torch.Tensor): Predicted class indices.
            targets (torch.Tensor): Ground-truth class indices.
        """

        preds = preds.view(-1).long()
        targets = targets.view(-1).long()
        mask = (targets >= 0) & (targets < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]
        if preds.numel() == 0:
            return
        indices = self.num_classes * targets + preds
        confusion = torch.bincount(indices, minlength=self.num_classes**2).double()
        self.confusion += confusion.view(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, torch.Tensor | float]:
        """Compute IoU and Dice metrics from the confusion matrix.

        Returns:
            Dict[str, torch.Tensor | float]: Metrics including per-class values and means.
        """

        tp = torch.diag(self.confusion)
        fp = self.confusion.sum(dim=0) - tp
        fn = self.confusion.sum(dim=1) - tp
        denom = tp + fp + fn
        iou = torch.where(denom > 0, tp / denom.clamp(min=1e-8), torch.zeros_like(tp))
        dice = torch.where(
            (2 * tp + fp + fn) > 0,
            2 * tp / (2 * tp + fp + fn).clamp(min=1e-8),
            torch.zeros_like(tp),
        )
        return {
            "per_class_iou": iou.cpu(),
            "per_class_dice": dice.cpu(),
            "miou": iou.mean().item(),
            "mdice": dice.mean().item(),
        }
