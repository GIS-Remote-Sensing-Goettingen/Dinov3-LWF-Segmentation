"""Training-related helpers and utilities."""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import SegmentationLoss, SegmentationMetrics, VerbosityLogger
from utils.losses import LOSS_COMPONENT_KEYS, compute_boundary_targets

from .context import StabilityConfig

_PATCH_CROP_WARNED: set[tuple[int, int, int]] = set()
_ADAMW_ONLY_HEADS: frozenset[str] = frozenset(
    {"dino_dense_probe", "dino_segdino_light"}
)


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


def align_to_patch_grid(
    image: torch.Tensor,
    labels: torch.Tensor | None,
    patch_size: int,
    logger: VerbosityLogger | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Crop tensors to the nearest lower patch-size multiple.

    Args:
        image (torch.Tensor): Image tensor shaped (B, C, H, W).
        labels (torch.Tensor | None): Optional labels shaped (B, H, W) or (H, W).
        patch_size (int): Backbone patch size.
        logger (VerbosityLogger | None): Optional logger for one-time crop warnings.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: Cropped image and labels.

    Examples:
        >>> img = torch.randn(1, 3, 33, 35)
        >>> y = torch.zeros(1, 33, 35).long()
        >>> i2, y2 = align_to_patch_grid(img, y, patch_size=16)
        >>> tuple(i2.shape), tuple(y2.shape)
        ((1, 3, 32, 32), (1, 32, 32))
    """

    if patch_size <= 1:
        return image, labels
    h, w = int(image.shape[-2]), int(image.shape[-1])
    h_eff = max((h // patch_size) * patch_size, patch_size)
    w_eff = max((w // patch_size) * patch_size, patch_size)
    if h_eff == h and w_eff == w:
        return image, labels
    if logger is not None:
        key = (h, w, int(patch_size))
        if key not in _PATCH_CROP_WARNED:
            _PATCH_CROP_WARNED.add(key)
            logger.warning(
                "Cropping inputs from "
                f"{h}x{w} to {h_eff}x{w_eff} to match DINO patch size {patch_size}."
            )
    image = image[..., :h_eff, :w_eff]
    if labels is None:
        return image, labels
    if labels.ndim == 2:
        labels = labels[:h_eff, :w_eff]
    elif labels.ndim >= 3:
        labels = labels[..., :h_eff, :w_eff]
    return image, labels


def forward_with_optional_extras(
    model_call: Any,
    image: torch.Tensor,
    features: list[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    dict[str, Any],
]:
    """Forward a model while collecting optional aux and boundary logits.

    Args:
        model_call (Any): Model or wrapper with forward methods.
        image (torch.Tensor): Input image tensor.
        features (list[torch.Tensor]): Multiscale feature tensors.

    Returns:
        tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            dict[str, Any],
        ]:
        Main logits, aux logits, optional edge logits, optional skeleton logits,
        and raw payload extras.

    Examples:
        >>> class Dummy:
        ...     def forward_with_aux(self, image, features):
        ...         return features[0], None
        >>> logits, aux, edge, skel, payload = forward_with_optional_extras(
        ...     Dummy(),
        ...     torch.randn(1, 3, 2, 2),
        ...     [torch.randn(1, 2, 2, 2)],
        ... )
        >>> aux is None and edge is None and skel is None and payload == {}
        True
    """

    if hasattr(model_call, "forward_with_extras"):
        payload = cast(dict[str, Any], model_call.forward_with_extras(image, features))
        logits = cast(torch.Tensor, payload["logits"])
        aux_logits = cast(torch.Tensor | None, payload.get("aux_logits"))
        edge_logits = cast(torch.Tensor | None, payload.get("edge_logits"))
        skeleton_logits = cast(torch.Tensor | None, payload.get("skeleton_logits"))
        return logits, aux_logits, edge_logits, skeleton_logits, payload
    if hasattr(model_call, "forward_with_aux"):
        logits, aux_logits = model_call.forward_with_aux(image, features)
        return logits, aux_logits, None, None, {}
    logits = model_call(image, features)
    return logits, None, None, None, {}


def build_boundary_targets(
    labels: torch.Tensor,
    edge_logits: torch.Tensor | None,
    num_classes: int,
    ignore_index: int | None,
    kernel_size: int = 3,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build boundary targets aligned to optional edge logits.

    Args:
        labels (torch.Tensor): Integer label tensor.
        edge_logits (torch.Tensor | None): Optional boundary logits.
        num_classes (int): Number of classes in labels.
        ignore_index (int | None): Optional ignore index.
        kernel_size (int): Laplacian kernel size.

    Returns:
        tuple[torch.Tensor | None, torch.Tensor | None]: Edge targets and valid mask.

    Examples:
        >>> labels = torch.tensor([[[0, 1], [1, 1]]])
        >>> edge_t, edge_m = build_boundary_targets(
        ...     labels=labels,
        ...     edge_logits=torch.randn(1, 1, 2, 2),
        ...     num_classes=2,
        ...     ignore_index=None,
        ... )
        >>> edge_t is not None and edge_m is not None
        True
    """

    if edge_logits is None:
        return None, None
    edge_labels = align_labels_to_logits(labels, edge_logits)
    return compute_boundary_targets(
        edge_labels,
        num_classes=num_classes,
        ignore_index=ignore_index,
        kernel_size=kernel_size,
    )


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


def use_adamw_only_for_head(head_name: str) -> bool:
    """Return whether a head should use the AdamW-only optimizer path.

    Args:
        head_name (str): Model head registry key.

    Returns:
        bool: ``True`` when the head should avoid Muon updates.

    Examples:
        >>> use_adamw_only_for_head("dino_segdino_light")
        True
        >>> use_adamw_only_for_head("unet_topo_fusion")
        False
    """

    return str(head_name).strip().lower() in _ADAMW_ONLY_HEADS


def should_warn_high_logit(batch_max_abs_logit: float, threshold: float) -> bool:
    """Return whether batch logits exceed the configured warning threshold.

    Args:
        batch_max_abs_logit (float): Maximum absolute logit in the batch.
        threshold (float): Warning threshold.

    Returns:
        bool: ``True`` if a warning should be emitted.

    Examples:
        >>> should_warn_high_logit(120.0, 80.0)
        True
        >>> should_warn_high_logit(40.0, 80.0)
        False
        >>> should_warn_high_logit(float("nan"), 80.0)
        False
    """

    return math.isfinite(batch_max_abs_logit) and (
        batch_max_abs_logit > float(threshold)
    )


def build_autocast(
    use_amp: bool,
    amp_dtype: str = "bf16",
):
    """Build an autocast context manager with dtype selection.

    Args:
        use_amp (bool): Whether autocast is enabled.
        amp_dtype (str): Preferred dtype ("bf16" or "fp16").

    Returns:
        context manager: Autocast context or nullcontext.

    Examples:
        >>> ctx = build_autocast(use_amp=False)
        >>> type(ctx).__name__
        'nullcontext'
    """

    if not use_amp:
        return nullcontext()
    if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def count_nonfinite_parameters(model: torch.nn.Module) -> int:
    """Count non-finite parameter values for a model.

    Args:
        model (torch.nn.Module): Model to inspect.

    Returns:
        int: Number of NaN/Inf parameter entries.

    Examples:
        >>> layer = torch.nn.Linear(2, 2)
        >>> count_nonfinite_parameters(layer)
        0
    """

    count = 0
    with torch.no_grad():
        for param in model.parameters():
            count += int((~torch.isfinite(param)).sum().item())
    return count


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
    stability: StabilityConfig | None = None,
    boundary_kernel_size: int = 3,
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
        stability (StabilityConfig | None): Stability policy controls.
        boundary_kernel_size (int): Kernel size for boundary target extraction.

    Returns:
        tuple[float, dict[str, Any]]: Average loss and metrics summary.

    Examples:
        >>> callable(evaluate)
        True
    """

    stability_cfg = stability or StabilityConfig()
    loss_component_sums = {key: 0.0 for key in LOSS_COMPONENT_KEYS}
    if loader is None:
        zeros = torch.zeros(num_classes)
        return 0.0, {
            "per_class_iou": zeros,
            "per_class_dice": zeros,
            "miou": 0.0,
            "mdice": 0.0,
            "nonfinite_val_batches": 0.0,
            "nonfinite_val_loss_batches": 0.0,
            "max_abs_logit": 0.0,
            **loss_component_sums,
        }
    model.eval()
    total = 0.0
    counted_loss_batches = 0
    metrics = SegmentationMetrics(num_classes)
    nonfinite_val_batches = 0
    nonfinite_val_loss_batches = 0
    max_abs_logit = 0.0
    gate_mean_sum = 0.0
    gate_std_sum = 0.0
    gate_stat_batches = 0
    layer_mix_sum: torch.Tensor | None = None
    layer_mix_count = 0
    autocast = build_autocast(use_amp=use_amp, amp_dtype=stability_cfg.amp_dtype)
    with torch.no_grad():
        for batch_idx, (img, features, y) in enumerate(loader, 1):
            img = img.to(device)
            y = y.to(device)
            img, y = align_to_patch_grid(img, y, patch_size=ps, logger=logger)
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
                logits, aux_logits, edge_logits, skeleton_logits, payload = (
                    forward_with_optional_extras(
                        model_call,
                        img,
                        feats,
                    )
                )
                target_main = align_labels_to_logits(y, logits)
                target_aux = (
                    align_labels_to_logits(y, aux_logits)
                    if aux_logits is not None
                    else None
                )
                edge_targets, edge_mask = build_boundary_targets(
                    labels=y,
                    edge_logits=edge_logits,
                    num_classes=loss_fn.num_classes,
                    ignore_index=loss_fn.ignore_index,
                    kernel_size=boundary_kernel_size,
                )
                logits_for_loss = logits.float() if stability_cfg.loss_fp32 else logits
                aux_for_loss = (
                    aux_logits.float()
                    if aux_logits is not None and stability_cfg.loss_fp32
                    else aux_logits
                )
                edge_for_loss = (
                    edge_logits.float()
                    if edge_logits is not None and stability_cfg.loss_fp32
                    else edge_logits
                )
                skeleton_for_loss = (
                    skeleton_logits.float()
                    if skeleton_logits is not None and stability_cfg.loss_fp32
                    else skeleton_logits
                )
                components = loss_fn.compute_components(
                    logits_for_loss,
                    target_main,
                    aux_logits=aux_for_loss,
                    aux_targets=target_aux,
                    edge_logits=edge_for_loss,
                    edge_targets=edge_targets,
                    edge_mask=edge_mask,
                    skeleton_logits=skeleton_for_loss,
                )
                loss = components["loss_total"]
            gate_mean = payload.get("gate_mean")
            gate_std = payload.get("gate_std")
            if gate_mean is not None and gate_std is not None:
                gate_mean_sum += float(torch.as_tensor(gate_mean).detach().item())
                gate_std_sum += float(torch.as_tensor(gate_std).detach().item())
                gate_stat_batches += 1
            layer_mix = payload.get("layer_mix_weights_mean")
            if layer_mix is not None:
                layer_vec = torch.as_tensor(layer_mix).detach().float().cpu()
                if layer_mix_sum is None:
                    layer_mix_sum = layer_vec.clone()
                else:
                    dim = min(int(layer_mix_sum.numel()), int(layer_vec.numel()))
                    layer_mix_sum[:dim] += layer_vec[:dim]
                layer_mix_count += 1
            batch_max_abs_logit = float(
                torch.nan_to_num(
                    logits.detach().float().abs(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                .max()
                .item()
            )
            max_abs_logit = max(max_abs_logit, batch_max_abs_logit)
            if not torch.isfinite(logits.detach()).all():
                nonfinite_val_batches += 1
                if stability_cfg.nonfinite_action == "stop_run":
                    raise RuntimeError(
                        f"Non-finite validation logits at batch {batch_idx}/{len(loader)}"
                    )
                if stability_cfg.nonfinite_action == "stop_epoch":
                    break
                continue
            if not torch.isfinite(loss):
                nonfinite_val_loss_batches += 1
                if stability_cfg.nonfinite_action == "stop_run":
                    raise RuntimeError(
                        f"Non-finite validation loss at batch {batch_idx}/{len(loader)}"
                    )
                if stability_cfg.nonfinite_action == "stop_epoch":
                    break
                continue
            total += loss.item()
            counted_loss_batches += 1
            for key in LOSS_COMPONENT_KEYS:
                loss_component_sums[key] += float(components[key].detach().item())
            preds = logits.argmax(dim=1)
            metrics.update(preds.cpu(), target_main.cpu())
            if logger and batch_idx % 10 == 0:
                logger.debug(
                    f"[Val] batch {batch_idx}/{len(loader)} "
                    f"loss={loss.item():.4f} "
                    f"running mIoU={metrics.compute()['miou']:.4f}"
                )
    avg_loss = float("nan")
    if counted_loss_batches:
        avg_loss = total / counted_loss_batches
    metric_summary = metrics.compute()
    if counted_loss_batches:
        for key in LOSS_COMPONENT_KEYS:
            metric_summary[key] = loss_component_sums[key] / counted_loss_batches
    else:
        for key in LOSS_COMPONENT_KEYS:
            metric_summary[key] = float("nan")
    metric_summary["nonfinite_val_batches"] = float(nonfinite_val_batches)
    metric_summary["nonfinite_val_loss_batches"] = float(nonfinite_val_loss_batches)
    metric_summary["max_abs_logit"] = float(max_abs_logit)
    if gate_stat_batches > 0:
        metric_summary["gate_mean"] = float(gate_mean_sum / gate_stat_batches)
        metric_summary["gate_std"] = float(gate_std_sum / gate_stat_batches)
    if layer_mix_sum is not None and layer_mix_count > 0:
        layer_mean = layer_mix_sum / float(layer_mix_count)
        layer_ids = layers or list(range(int(layer_mean.numel())))
        for idx in range(int(layer_mean.numel())):
            layer_id = layer_ids[idx] if idx < len(layer_ids) else idx
            metric_summary[f"layer_mix_{int(layer_id)}_mean"] = float(
                layer_mean[idx].item()
            )
    if logger:
        logger.debug(
            f"Validation summary :: loss={avg_loss:.4f}, "
            f"mIoU={metric_summary['miou']:.4f}, mDice={metric_summary['mdice']:.4f}"
        )
        if nonfinite_val_loss_batches > 0 and torch.isfinite(
            torch.tensor(metric_summary["miou"])
        ):
            logger.error(
                "Validation has non-finite loss batches while mIoU remains finite."
            )
    return avg_loss, metric_summary
