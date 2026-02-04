"""Training-related helpers and utilities."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import SegmentationLoss, SegmentationMetrics, VerbosityLogger


class ModelEMA:
    """Maintain an exponential moving average of model parameters.

    Args:
        model (torch.nn.Module): Model to track.
        decay (float): EMA decay factor.

    Examples:
        >>> model = torch.nn.Linear(2, 2)
        >>> ema = ModelEMA(model, decay=0.9)
        >>> isinstance(ema.ema_model, torch.nn.Module)
        True
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        """Initialize the EMA tracker.

        Args:
            model (torch.nn.Module): Model to track.
            decay (float): EMA decay factor.
        """

        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self.decay = decay

    def update(self, model: torch.nn.Module) -> None:
        """Update EMA weights from the current model.

        Args:
            model (torch.nn.Module): Model with current weights.
        """

        with torch.no_grad():
            ema_params = dict(self.ema_model.named_parameters())
            model_params = dict(model.named_parameters())
            for name, param in model_params.items():
                if name in ema_params:
                    ema_params[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )
            ema_buffers = dict(self.ema_model.named_buffers())
            for name, buf in model.named_buffers():
                if name in ema_buffers:
                    ema_buffers[name].copy_(buf)


def extract_multiscale_features_batch(
    images: torch.Tensor,
    model: Any,
    processor: Any,
    device: torch.device,
    layers: list[int],
    ps: int,
) -> list[torch.Tensor]:
    """Extract multiscale features for a batch of images.

    Args:
        images (torch.Tensor): Image batch in CHW format, normalized to [0, 1].
        model (Any): Backbone model instance.
        processor (Any): Image processor instance.
        device (torch.device): Device for inference.
        layers (list[int]): Backbone layer indices to extract.
        ps (int): Patch size for the backbone.

    Returns:
        list[torch.Tensor]: Feature maps per requested layer (B, C, H/ps, W/ps).

    Examples:
        >>> callable(extract_multiscale_features_batch)
        True
    """

    images_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    if images_np.max() <= 1.5:
        images_np = (images_np * 255.0).astype("uint8")
    inputs = processor(
        images=list(images_np),
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)
    R = getattr(model.config, "num_register_tokens", 0)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states
    _, _, h_proc, w_proc = inputs["pixel_values"].shape
    hp, wp = h_proc // ps, w_proc // ps
    feature_maps: list[torch.Tensor] = []
    batch_size = images.shape[0]
    for layer_idx in layers:
        layer_output = hidden_states[layer_idx]
        patch_tokens = layer_output[:, 1 + R :, :]
        feats = patch_tokens.reshape(batch_size, hp, wp, -1).permute(0, 3, 1, 2)
        feature_maps.append(feats)
    return feature_maps


def move_features_to_device(
    features: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    """Clone and push cached feature tensors to the target device.

    Args:
        features (list[torch.Tensor]): Feature tensors.
        device (torch.device): Target device.

    Returns:
        list[torch.Tensor]: Feature tensors on the target device.

    Examples:
        >>> feats = [torch.ones(1, 2, 2, 2)]
        >>> move_features_to_device(feats, torch.device("cpu"))[0].device.type
        'cpu'
    """

    return [f.to(device) for f in features]


def align_labels_to_logits(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Align label tensor spatial dimensions with logits.

    Args:
        y (torch.Tensor): Label tensor.
        logits (torch.Tensor): Logits tensor.

    Returns:
        torch.Tensor: Aligned label tensor.

    Examples:
        >>> y = torch.zeros(1, 2, 2).long()
        >>> logits = torch.zeros(1, 2, 4, 4)
        >>> align_labels_to_logits(y, logits).shape
        torch.Size([1, 4, 4])
    """

    if y.ndim == 2:
        y = y.unsqueeze(0)
    if logits.shape[-2:] == y.shape[-2:]:
        return y
    y_expanded = y.unsqueeze(1).float()
    aligned = F.interpolate(y_expanded, size=logits.shape[-2:], mode="nearest")
    return aligned.squeeze(1).long()


def split_params_for_muon(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into Muon-compatible and AdamW groups.

    Args:
        model (torch.nn.Module): Model to split parameters for.

    Returns:
        tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]: Muon params and AdamW params.

    Examples:
        >>> module = torch.nn.Linear(4, 4)
        >>> muon_params, adamw_params = split_params_for_muon(module)
        >>> all(p.ndim >= 2 for p in muon_params)
        True
    """

    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for _, p in model.named_parameters():
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader | None,
    loss_fn: SegmentationLoss,
    device: torch.device,
    use_amp: bool,
    logger: VerbosityLogger | None = None,
    num_classes: int = 2,
    cache_features: bool = True,
    backbone: Any | None = None,
    processor: Any | None = None,
    layers: list[int] | None = None,
    ps: int = 16,
) -> tuple[float, dict[str, Any]]:
    """Evaluate the model on the validation set.

    Args:
        model (torch.nn.Module): Model to evaluate.
        loader (DataLoader | None): Validation loader.
        loss_fn (SegmentationLoss): Loss function.
        device (torch.device): Device to run on.
        use_amp (bool): Whether to use AMP.
        logger (VerbosityLogger | None): Logger for debug messages.
        num_classes (int): Number of classes.
        cache_features (bool): Whether cached features are available.
        backbone (Any | None): DINO backbone for on-the-fly extraction.
        processor (Any | None): Image processor for on-the-fly extraction.
        layers (list[int] | None): Backbone layers to extract.
        ps (int): Patch size for the backbone.

    Returns:
        tuple[float, dict[str, Any]]: Average loss and metrics summary.

    Examples:
        >>> callable(evaluate)
        True
    """

    if loader is None:
        zeros = torch.zeros(num_classes)
        return 0.0, {
            "per_class_iou": zeros,
            "per_class_dice": zeros,
            "miou": 0.0,
            "mdice": 0.0,
        }
    model.eval()
    total = 0.0
    metrics = SegmentationMetrics(num_classes)
    autocast = torch.cuda.amp.autocast() if use_amp else nullcontext()
    with torch.no_grad():
        for batch_idx, (img, features, y) in enumerate(loader, 1):
            img = img.to(device)
            y = y.to(device)
            if cache_features and features:
                feats = move_features_to_device(features, device)
            else:
                if backbone is None or processor is None or layers is None:
                    raise ValueError(
                        "Backbone/processor/layers required for on-the-fly eval"
                    )
                feats = extract_multiscale_features_batch(
                    img,
                    backbone,
                    processor,
                    device,
                    layers,
                    ps,
                )
            model_call = cast(Any, model)
            with autocast:
                if hasattr(model_call, "forward_with_aux"):
                    logits, aux_logits = model_call.forward_with_aux(img, feats)
                else:
                    logits = model_call(img, feats)
                    aux_logits = None
                target_main = align_labels_to_logits(y, logits)
                target_aux = (
                    align_labels_to_logits(y, aux_logits)
                    if aux_logits is not None
                    else None
                )
                loss = loss_fn(
                    logits,
                    target_main,
                    aux_logits=aux_logits,
                    aux_targets=target_aux,
                )
            total += loss.item()
            preds = logits.argmax(dim=1)
            metrics.update(preds.cpu(), target_main.cpu())
            if logger and batch_idx % 10 == 0:
                logger.debug(
                    f"[Val] batch {batch_idx}/{len(loader)} "
                    f"loss={loss.item():.4f} "
                    f"running mIoU={metrics.compute()['miou']:.4f}"
                )
    avg_loss = total / len(loader)
    metric_summary = metrics.compute()
    if logger:
        logger.debug(
            f"Validation summary :: loss={avg_loss:.4f}, "
            f"mIoU={metric_summary['miou']:.4f}, mDice={metric_summary['mdice']:.4f}"
        )
    return avg_loss, metric_summary
