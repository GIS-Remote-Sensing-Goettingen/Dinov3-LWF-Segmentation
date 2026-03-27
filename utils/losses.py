"""
Segmentation losses combining cross-entropy and Dice components.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

LOSS_COMPONENT_KEYS: tuple[str, ...] = (
    "loss_total",
    "loss_main_ce",
    "loss_main_focal",
    "loss_main_dice",
    "loss_aux_ce",
    "loss_aux_focal",
    "loss_aux_dice",
    "loss_edge_bce",
    "loss_skeleton_bce",
    "loss_topology_cldice",
    "loss_weighted_main",
    "loss_weighted_aux",
    "loss_weighted_edge",
    "loss_weighted_skeleton",
    "loss_weighted_topology",
    "skeleton_prob_mean",
    "skeleton_prob_p95",
    "skeleton_pred_pos_rate",
)


class DiceLoss(nn.Module):
    """
    Multiclass Dice loss operating on logits and integer targets.

    >>> _ = torch.manual_seed(0)
    >>> loss = DiceLoss(num_classes=2)
    >>> logits = torch.randn(1, 2, 4, 4)
    >>> targets = torch.zeros(1, 4, 4, dtype=torch.long)
    >>> round(loss(logits, targets).item(), 4)
    0.683
    """

    def __init__(
        self, num_classes: int, eps: float = 1e-6, ignore_index: Optional[int] = None
    ) -> None:
        """Initialize the Dice loss.

        Args:
            num_classes (int): Number of segmentation classes.
            eps (float): Numerical stability constant.
            ignore_index (Optional[int]): Optional ignore index.
        """

        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the Dice loss.

        Args:
            logits (torch.Tensor): Logits tensor.
            targets (torch.Tensor): Integer target labels.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """

        probs = torch.softmax(logits, dim=1)
        targets = targets.long()
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            if not mask.any():
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            probs = probs * mask.unsqueeze(1)
            targets = torch.where(mask, targets, torch.zeros_like(targets))
        one_hot = F.one_hot(
            targets.clamp(min=0, max=self.num_classes - 1), self.num_classes
        )
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        cardinality = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


def compute_boundary_targets(
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    kernel_size: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute binary edge targets and valid masks from segmentation labels.

    Args:
        targets (torch.Tensor): Integer labels with shape (B, H, W) or (H, W).
        num_classes (int): Number of semantic classes.
        ignore_index (Optional[int]): Optional ignore label.
        kernel_size (int): Laplacian kernel size. Only 3 is currently supported.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Edge targets and valid mask, both with
        shape (B, 1, H, W).

    Examples:
        >>> labels = torch.tensor([[[0, 0, 1], [0, 1, 1], [0, 0, 1]]])
        >>> edge, mask = compute_boundary_targets(labels, num_classes=2)
        >>> edge.shape, mask.shape
        (torch.Size([1, 1, 3, 3]), torch.Size([1, 1, 3, 3]))
    """

    if kernel_size != 3:
        raise ValueError("compute_boundary_targets currently supports kernel_size=3")
    if targets.ndim == 2:
        targets = targets.unsqueeze(0)
    targets = targets.long()
    valid_mask = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid_mask = targets != int(ignore_index)
    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
    safe_targets = safe_targets.clamp(min=0, max=num_classes - 1)
    one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.float() * valid_mask.unsqueeze(1).float()

    laplacian = torch.tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
        device=targets.device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    kernel = laplacian.repeat(num_classes, 1, 1, 1)
    response = F.conv2d(one_hot, kernel, padding=1, groups=num_classes).abs()
    edge_targets = (response > 0).any(dim=1, keepdim=True).float()
    edge_mask = valid_mask.unsqueeze(1).float()
    edge_targets = edge_targets * edge_mask
    return edge_targets, edge_mask


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Apply differentiable soft erosion.

    Args:
        x (torch.Tensor): Input map with shape (B, 1, H, W).

    Returns:
        torch.Tensor: Soft-eroded map.

    Examples:
        >>> x = torch.zeros(1, 1, 5, 5)
        >>> x[:, :, 2, 2] = 1.0
        >>> tuple(soft_erode(x).shape)
        (1, 1, 5, 5)
    """

    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """Apply differentiable soft dilation.

    Args:
        x (torch.Tensor): Input map with shape (B, 1, H, W).

    Returns:
        torch.Tensor: Soft-dilated map.

    Examples:
        >>> x = torch.zeros(1, 1, 5, 5)
        >>> x[:, :, 2, 2] = 1.0
        >>> float(soft_dilate(x).max()) >= 1.0
        True
    """

    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    """Apply differentiable opening operation.

    Args:
        x (torch.Tensor): Input map with shape (B, 1, H, W).

    Returns:
        torch.Tensor: Opened map.

    Examples:
        >>> x = torch.zeros(1, 1, 5, 5)
        >>> x[:, :, 2, 2] = 1.0
        >>> tuple(soft_open(x).shape)
        (1, 1, 5, 5)
    """

    return soft_dilate(soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Approximate a soft skeleton via iterative morphology.

    Args:
        x (torch.Tensor): Input map with shape (B, 1, H, W).
        iters (int): Number of refinement iterations.

    Returns:
        torch.Tensor: Soft skeleton map.

    Examples:
        >>> m = torch.zeros(1, 1, 5, 5)
        >>> m[:, :, 2, :] = 1.0
        >>> s = soft_skeletonize(m, iters=5)
        >>> tuple(s.shape)
        (1, 1, 5, 5)
    """

    x = x.clamp(min=0.0, max=1.0)
    skel = F.relu(x - soft_open(x))
    iterations = max(1, int(iters))
    for _ in range(iterations - 1):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(min=0.0, max=1.0)


def soft_cldice_loss(
    pred_fg: torch.Tensor,
    target_fg: torch.Tensor,
    iters: int = 10,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute soft-clDice loss for connectivity preservation.

    Args:
        pred_fg (torch.Tensor): Foreground probability map (B, 1, H, W).
        target_fg (torch.Tensor): Binary target foreground map (B, 1, H, W).
        iters (int): Skeletonization iterations.
        smooth (float): Numerical stability term.

    Returns:
        torch.Tensor: Scalar soft-clDice loss.

    Examples:
        >>> p = torch.zeros(1, 1, 8, 8)
        >>> t = torch.zeros(1, 1, 8, 8)
        >>> p[:, :, 3, 1:7] = 0.8
        >>> t[:, :, 3, 1:7] = 1.0
        >>> float(soft_cldice_loss(p, t, iters=5)) < 0.5
        True
    """

    pred_fg = pred_fg.clamp(min=0.0, max=1.0)
    target_fg = target_fg.clamp(min=0.0, max=1.0)
    skel_pred = soft_skeletonize(pred_fg, iters=iters)
    skel_true = soft_skeletonize(target_fg, iters=iters)
    tprec = (skel_pred * target_fg).sum(dim=(1, 2, 3))
    tprec = (tprec + smooth) / (skel_pred.sum(dim=(1, 2, 3)) + smooth)
    tsens = (skel_true * pred_fg).sum(dim=(1, 2, 3))
    tsens = (tsens + smooth) / (skel_true.sum(dim=(1, 2, 3)) + smooth)
    cldice = (2.0 * tprec * tsens + smooth) / (tprec + tsens + smooth)
    return 1.0 - cldice.mean()


class SegmentationLoss(nn.Module):
    """
    Combined cross-entropy and Dice loss with optional auxiliary output.

    >>> _ = torch.manual_seed(0)
    >>> loss_fn = SegmentationLoss(num_classes=2, ce_weight=1.0, dice_weight=1.0)
    >>> logits = torch.randn(1, 2, 4, 4)
    >>> targets = torch.zeros(1, 4, 4, dtype=torch.long)
    >>> round(loss_fn(logits, targets).item(), 4)
    1.6594
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        focal_weight: float = 0.0,
        dice_weight: float = 1.0,
        aux_weight: float = 0.4,
        class_weights: Optional[List[float]] = None,
        ignore_index: Optional[int] = None,
        label_smoothing: float = 0.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        boundary_weight: float = 0.0,
        skeleton_weight: float = 0.0,
        skeleton_pos_weight: float = 1.0,
        topology_weight: float = 0.0,
        topology_class_index: int = 1,
        topology_iters: int = 10,
        topology_on_aux: bool = True,
        topology_downsample: int = 1,
    ) -> None:
        """Initialize the combined segmentation loss.

        Args:
            num_classes (int): Number of segmentation classes.
            ce_weight (float): Cross-entropy weight.
            focal_weight (float): Focal-loss weight.
            dice_weight (float): Dice loss weight.
            aux_weight (float): Auxiliary loss weight.
            class_weights (Optional[List[float]]): Optional class weights.
            ignore_index (Optional[int]): Optional ignore index.
            label_smoothing (float): Cross-entropy label smoothing value.
            use_focal (bool): Legacy toggle to replace CE with focal loss.
            focal_gamma (float): Focal focusing parameter.
            focal_alpha (Optional[float]): Optional focal alpha weight in [0, 1].
            boundary_weight (float): Weight for boundary BCE supervision.
            skeleton_weight (float): Weight for skeleton BCE supervision.
            skeleton_pos_weight (float): Positive-class weight for the skeleton
                BCE term to counter the extreme sparsity of 1-pixel skeletons.
            topology_weight (float): Weight for soft-clDice topology supervision.
            topology_class_index (int): Foreground class index for topology loss.
            topology_iters (int): Soft skeletonization iteration count.
            topology_on_aux (bool): Use aux logits/targets for topology by default.
            topology_downsample (int): Downsample factor when topology_on_aux is false.
        """

        super().__init__()
        self.ce_weight = float(ce_weight)
        self.focal_weight = max(float(focal_weight), 0.0)
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.label_smoothing = min(max(float(label_smoothing), 0.0), 0.999)
        self.use_focal = bool(use_focal)
        self.focal_gamma = max(float(focal_gamma), 0.0)
        self.focal_alpha = (
            None if focal_alpha is None else min(max(float(focal_alpha), 0.0), 1.0)
        )
        if self.use_focal and self.focal_weight <= 0.0:
            self.focal_weight = self.ce_weight
            self.ce_weight = 0.0
        self.boundary_weight = max(float(boundary_weight), 0.0)
        self.skeleton_weight = max(float(skeleton_weight), 0.0)
        self.skeleton_pos_weight = max(float(skeleton_pos_weight), 1.0)
        self.topology_weight = max(float(topology_weight), 0.0)
        self.topology_class_index = max(0, int(topology_class_index))
        self.topology_iters = max(1, int(topology_iters))
        self.topology_on_aux = bool(topology_on_aux)
        self.topology_downsample = max(1, int(topology_downsample))
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weight_tensor, persistent=False)
        else:
            self.class_weights = None
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.num_classes = num_classes

    def _ce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the cross-entropy loss.

        Args:
            logits (torch.Tensor): Logits tensor.
            targets (torch.Tensor): Integer target labels.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """

        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
            label_smoothing=self.label_smoothing,
        )

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multiclass focal loss from logits and integer targets.

        Args:
            logits (torch.Tensor): Logits tensor.
            targets (torch.Tensor): Integer target labels.

        Returns:
            torch.Tensor: Scalar focal loss tensor.
        """

        ignore = self.ignore_index if self.ignore_index is not None else -100
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=ignore,
            reduction="none",
        )
        valid_mask = targets != ignore
        pt = torch.exp(-ce)
        focal = ((1.0 - pt).clamp(min=0.0) ** self.focal_gamma) * ce
        if self.focal_alpha is not None:
            focal = focal * self.focal_alpha
        focal = focal * valid_mask.float()
        denom = valid_mask.float().sum().clamp_min(1.0)
        return focal.sum() / denom

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: Optional[torch.Tensor] = None,
        aux_targets: Optional[torch.Tensor] = None,
        edge_logits: Optional[torch.Tensor] = None,
        edge_targets: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        skeleton_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the combined loss with optional auxiliary output.

        Args:
            logits (torch.Tensor): Main logits tensor.
            targets (torch.Tensor): Target labels.
            aux_logits (Optional[torch.Tensor]): Auxiliary logits tensor.
            aux_targets (Optional[torch.Tensor]): Auxiliary targets tensor.
            edge_logits (Optional[torch.Tensor]): Boundary logits tensor.
            edge_targets (Optional[torch.Tensor]): Binary boundary targets.
            edge_mask (Optional[torch.Tensor]): Optional boundary valid mask.
            skeleton_logits (Optional[torch.Tensor]): Skeleton logits tensor.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """

        return self.compute_components(
            logits=logits,
            targets=targets,
            aux_logits=aux_logits,
            aux_targets=aux_targets,
            edge_logits=edge_logits,
            edge_targets=edge_targets,
            edge_mask=edge_mask,
            skeleton_logits=skeleton_logits,
        )["loss_total"]

    def compute_components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: Optional[torch.Tensor] = None,
        aux_targets: Optional[torch.Tensor] = None,
        edge_logits: Optional[torch.Tensor] = None,
        edge_targets: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        skeleton_logits: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute weighted and unweighted loss components.

        Args:
            logits (torch.Tensor): Main logits tensor.
            targets (torch.Tensor): Target labels.
            aux_logits (Optional[torch.Tensor]): Auxiliary logits tensor.
            aux_targets (Optional[torch.Tensor]): Auxiliary targets tensor.
            edge_logits (Optional[torch.Tensor]): Boundary logits tensor.
            edge_targets (Optional[torch.Tensor]): Binary boundary targets.
            edge_mask (Optional[torch.Tensor]): Optional boundary valid mask.
            skeleton_logits (Optional[torch.Tensor]): Skeleton logits tensor.

        Returns:
            dict[str, torch.Tensor]: Loss components keyed by
            `LOSS_COMPONENT_KEYS`.
        """

        zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
        main_ce = zero
        main_focal = zero
        main_dice = zero
        aux_ce = zero
        aux_focal = zero
        aux_dice = zero
        edge_bce = zero
        skeleton_bce = zero
        topology_cldice = zero
        skeleton_prob_mean = zero
        skeleton_prob_p95 = zero
        skeleton_pred_pos_rate = zero

        if self.ce_weight > 0:
            main_ce = self._ce_loss(logits, targets)
        if self.focal_weight > 0:
            main_focal = self._focal_loss(logits, targets)
        if self.dice_weight:
            main_dice = self.dice(logits, targets)
        if aux_logits is not None and aux_targets is not None and self.aux_weight > 0:
            if self.ce_weight > 0:
                aux_ce = self._ce_loss(aux_logits, aux_targets)
            if self.focal_weight > 0:
                aux_focal = self._focal_loss(aux_logits, aux_targets)
            if self.dice_weight:
                aux_dice = self.dice(aux_logits, aux_targets)
        if (
            edge_logits is not None
            and edge_targets is not None
            and self.boundary_weight > 0
        ):
            edge_map = F.binary_cross_entropy_with_logits(
                edge_logits,
                edge_targets.float(),
                reduction="none",
            )
            if edge_mask is not None:
                mask = edge_mask.float()
                edge_bce = (edge_map * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                edge_bce = edge_map.mean()

        class_idx = min(self.topology_class_index, self.num_classes - 1)
        if self.topology_weight > 0:
            topology_logits = logits
            topology_targets = targets
            if (
                self.topology_on_aux
                and aux_logits is not None
                and aux_targets is not None
            ):
                topology_logits = aux_logits
                topology_targets = aux_targets

            valid_mask = torch.ones_like(topology_targets, dtype=torch.bool)
            if self.ignore_index is not None:
                valid_mask = topology_targets != int(self.ignore_index)
            safe_targets = torch.where(
                valid_mask,
                topology_targets.clamp(min=0, max=self.num_classes - 1),
                torch.zeros_like(topology_targets),
            )
            target_fg = (safe_targets == class_idx).float().unsqueeze(1)
            target_fg = target_fg * valid_mask.unsqueeze(1).float()

            if topology_logits.shape[1] == 1:
                pred_fg = torch.sigmoid(topology_logits)
            else:
                pred_fg = torch.softmax(topology_logits, dim=1)[
                    :, class_idx : class_idx + 1
                ]
            pred_fg = pred_fg * valid_mask.unsqueeze(1).float()

            if not self.topology_on_aux and self.topology_downsample > 1:
                scale = 1.0 / float(self.topology_downsample)
                pred_fg = F.interpolate(
                    pred_fg,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                )
                target_fg = F.interpolate(
                    target_fg,
                    size=pred_fg.shape[-2:],
                    mode="nearest",
                )

            topology_cldice = soft_cldice_loss(
                pred_fg,
                target_fg,
                iters=self.topology_iters,
            )

        if skeleton_logits is not None:
            with torch.no_grad():
                skel_prob_metrics = torch.sigmoid(skeleton_logits.detach().float())
                if skel_prob_metrics.numel() > 0:
                    skeleton_prob_mean = skel_prob_metrics.mean()
                    skeleton_prob_p95 = torch.quantile(
                        skel_prob_metrics.reshape(-1), 0.95
                    )
                    skeleton_pred_pos_rate = (skel_prob_metrics >= 0.5).float().mean()

        if skeleton_logits is not None and self.skeleton_weight > 0:
            target_for_skeleton = targets
            if target_for_skeleton.ndim == 2:
                target_for_skeleton = target_for_skeleton.unsqueeze(0)
            target_for_skeleton = (
                F.interpolate(
                    target_for_skeleton.unsqueeze(1).float(),
                    size=skeleton_logits.shape[-2:],
                    mode="nearest",
                )
                .squeeze(1)
                .long()
            )
            valid_skel = torch.ones_like(target_for_skeleton, dtype=torch.bool)
            if self.ignore_index is not None:
                valid_skel = target_for_skeleton != int(self.ignore_index)
            safe_skel = torch.where(
                valid_skel,
                target_for_skeleton.clamp(min=0, max=self.num_classes - 1),
                torch.zeros_like(target_for_skeleton),
            )
            skel_fg = (safe_skel == class_idx).float().unsqueeze(1)
            skel_fg = skel_fg * valid_skel.unsqueeze(1).float()
            with torch.no_grad():
                skeleton_target = soft_skeletonize(skel_fg, iters=self.topology_iters)
            skel_bce_map = F.binary_cross_entropy_with_logits(
                skeleton_logits,
                skeleton_target,
                pos_weight=torch.full(
                    (1,),
                    self.skeleton_pos_weight,
                    dtype=skeleton_logits.dtype,
                    device=skeleton_logits.device,
                ),
                reduction="none",
            )
            skel_mask = valid_skel.unsqueeze(1).float()
            skeleton_bce = (skel_bce_map * skel_mask).sum() / skel_mask.sum().clamp_min(
                1.0
            )

        weighted_main = (
            self.ce_weight * main_ce
            + self.focal_weight * main_focal
            + self.dice_weight * main_dice
        )
        weighted_aux = self.aux_weight * (
            self.ce_weight * aux_ce
            + self.focal_weight * aux_focal
            + self.dice_weight * aux_dice
        )
        weighted_edge = self.boundary_weight * edge_bce
        weighted_skeleton = self.skeleton_weight * skeleton_bce
        weighted_topology = self.topology_weight * topology_cldice
        total = (
            weighted_main
            + weighted_aux
            + weighted_edge
            + weighted_skeleton
            + weighted_topology
        )
        return {
            "loss_total": total,
            "loss_main_ce": main_ce,
            "loss_main_focal": main_focal,
            "loss_main_dice": main_dice,
            "loss_aux_ce": aux_ce,
            "loss_aux_focal": aux_focal,
            "loss_aux_dice": aux_dice,
            "loss_edge_bce": edge_bce,
            "loss_skeleton_bce": skeleton_bce,
            "loss_topology_cldice": topology_cldice,
            "loss_weighted_main": weighted_main,
            "loss_weighted_aux": weighted_aux,
            "loss_weighted_edge": weighted_edge,
            "loss_weighted_skeleton": weighted_skeleton,
            "loss_weighted_topology": weighted_topology,
            "skeleton_prob_mean": skeleton_prob_mean,
            "skeleton_prob_p95": skeleton_prob_p95,
            "skeleton_pred_pos_rate": skeleton_pred_pos_rate,
        }
