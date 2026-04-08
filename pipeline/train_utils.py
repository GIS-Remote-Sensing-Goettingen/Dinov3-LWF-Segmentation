"""Training-related helpers and utilities."""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from utils import SegmentationLoss, SegmentationMetrics, VerbosityLogger
from utils.losses import LOSS_COMPONENT_KEYS, compute_boundary_targets
from utils.optim import Muon

from .context import StabilityConfig

_PATCH_CROP_WARNED: set[tuple[int, int, int]] = set()
_ADAMW_ONLY_HEADS: frozenset[str] = frozenset(
    {
        "deeplabv3",
        "dino_dense_probe",
        "dino_segdino_light",
        "mask2former_semantic",
        "unet",
    }
)
_IMAGE_ONLY_HEADS: frozenset[str] = frozenset(
    {"deeplabv3", "mask2former_semantic", "unet"}
)
_AUX_LOGIT_HEADS: frozenset[str] = frozenset(
    {
        "deeplabv3",
        "unet_v2",
        "unet_lite",
        "unet_lite_plus",
        "unet_nano",
        "unet_nano_fapm",
        "unet_topo_fusion",
    }
)
_NATIVE_LOSS_HEADS: frozenset[str] = frozenset({"mask2former_semantic"})


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


class NormalizedForwardAdapter(nn.Module):
    """Adapt segmentation heads so ``forward`` always returns a payload dict.

    This keeps custom aux/edge outputs reachable through wrappers such as DDP,
    which only invoke the module's ``forward`` method.

    Args:
        head (nn.Module): Segmentation head to normalize.

    Examples:
        >>> class Dummy(nn.Module):
        ...     def forward_with_aux(self, image, features):
        ...         return features[0], features[0][:, :1]
        >>> adapter = NormalizedForwardAdapter(Dummy())
        >>> payload = adapter(torch.randn(1, 3, 2, 2), [torch.randn(1, 2, 2, 2)])
        >>> sorted(payload.keys())
        ['aux_logits', 'edge_logits', 'logits', 'skeleton_logits']
    """

    def __init__(self, head: nn.Module) -> None:
        """Store the wrapped segmentation head.

        Args:
            head (nn.Module): Segmentation head whose outputs need normalization.

        Examples:
            >>> adapter = NormalizedForwardAdapter(torch.nn.Identity())
            >>> isinstance(adapter.head, torch.nn.Module)
            True
        """

        super().__init__()
        self.head = head

    def forward(
        self,
        image: torch.Tensor,
        features: list[torch.Tensor],
        *,
        labels: torch.Tensor | None = None,
        ignore_index: int | None = None,
    ) -> dict[str, Any]:
        """Run the wrapped head and normalize outputs into one payload.

        Args:
            image (torch.Tensor): Input image tensor.
            features (list[torch.Tensor]): Multiscale feature tensors.
            labels (torch.Tensor | None): Optional semantic labels used only by
                native-loss heads.
            ignore_index (int | None): Optional ignore label passed through to
                native-loss heads.

        Returns:
            dict[str, Any]: Normalized payload containing logits and optional
            aux/boundary/topology entries.

        Examples:
            >>> class Dummy(nn.Module):
            ...     def forward(self, image, features):
            ...         _ = image
            ...         return features[0]
            >>> adapter = NormalizedForwardAdapter(Dummy())
            >>> payload = adapter(torch.randn(1, 3, 2, 2), [torch.randn(1, 2, 2, 2)])
            >>> payload["aux_logits"] is None
            True
        """

        if labels is not None and hasattr(self.head, "forward_with_native_loss"):
            raw_output = cast(Any, self.head).forward_with_native_loss(
                image,
                features,
                labels,
                ignore_index=ignore_index,
            )
        elif hasattr(self.head, "forward_with_extras"):
            raw_output = cast(Any, self.head).forward_with_extras(image, features)
        elif hasattr(self.head, "forward_with_aux"):
            logits, aux_logits = cast(Any, self.head).forward_with_aux(image, features)
            raw_output = {"logits": logits, "aux_logits": aux_logits}
        else:
            raw_output = cast(Any, self.head)(image, features)
        return normalize_forward_output(raw_output)


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


def normalize_forward_output(raw_output: Any) -> dict[str, Any]:
    """Normalize model outputs into a payload with optional extras.

    Args:
        raw_output (Any): Tensor logits, legacy tuple outputs, or a payload dict.

    Returns:
        dict[str, Any]: Payload with at least ``logits``, ``aux_logits``,
        ``edge_logits``, and ``skeleton_logits`` keys.

    Examples:
        >>> payload = normalize_forward_output(torch.randn(1, 2, 4, 4))
        >>> payload["aux_logits"] is None and payload["edge_logits"] is None
        True
        >>> payload = normalize_forward_output({
        ...     "logits": torch.randn(1, 2, 4, 4),
        ...     "extra": 1,
        ... })
        >>> payload["extra"]
        1
    """

    if isinstance(raw_output, dict):
        payload = dict(raw_output)
        if "logits" not in payload:
            raise ValueError("Model payload dict must include a 'logits' entry.")
    elif isinstance(raw_output, tuple):
        if len(raw_output) != 2:
            raise TypeError(
                "Tuple model outputs must be (logits, aux_logits) for normalization."
            )
        logits, aux_logits = raw_output
        payload = {"logits": logits, "aux_logits": aux_logits}
    elif isinstance(raw_output, torch.Tensor):
        payload = {"logits": raw_output}
    else:
        raise TypeError(
            "Model output must be a tensor, (logits, aux_logits) tuple, "
            f"or payload dict, got {type(raw_output).__name__}."
        )
    payload.setdefault("aux_logits", None)
    payload.setdefault("edge_logits", None)
    payload.setdefault("skeleton_logits", None)
    return payload


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


def align_logits_to_labels(
    logits: torch.Tensor | None,
    labels: torch.Tensor,
) -> torch.Tensor | None:
    """Align logits to the native label grid.

    Args:
        logits (torch.Tensor | None): Logits tensor or ``None``.
        labels (torch.Tensor): Label tensor defining the target grid.

    Returns:
        torch.Tensor | None: Logits resized to label spatial dimensions.

    Examples:
        >>> y = torch.zeros(1, 2, 2).long()
        >>> logits = torch.zeros(1, 2, 4, 4)
        >>> align_logits_to_labels(logits, y).shape
        torch.Size([1, 2, 2, 2])
        >>> align_logits_to_labels(None, y) is None
        True
    """

    if logits is None:
        return None
    if labels.ndim == 2:
        labels = labels.unsqueeze(0)
    if logits.shape[-2:] == labels.shape[-2:]:
        return logits
    return F.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


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
    if labels.shape[-2:] != (h, w):
        return image, labels
    if labels.ndim == 2:
        labels = labels[:h_eff, :w_eff]
    elif labels.ndim >= 3:
        labels = labels[..., :h_eff, :w_eff]
    return image, labels


def require_patch_grid_compatible(
    image: torch.Tensor,
    patch_size: int,
    *,
    source: str,
) -> None:
    """Fail fast when cached/training tensors are incompatible with DINO patches.

    Args:
        image (torch.Tensor): Image tensor shaped ``(B, C, H, W)``.
        patch_size (int): Backbone patch size.
        source (str): Error-message context.

    Raises:
        ValueError: If the spatial size is not patch-grid compatible.

    Examples:
        >>> require_patch_grid_compatible(torch.zeros(1, 3, 32, 32), 16, source="ok")
        >>> require_patch_grid_compatible(torch.zeros(1, 3, 33, 32), 16, source="bad")
        Traceback (most recent call last):
        ...
        ValueError: bad has image size 33x32, which is incompatible with DINO patch size 16. ...
    """

    if patch_size <= 1:
        return
    height = int(image.shape[-2])
    width = int(image.shape[-1])
    if (height % int(patch_size)) != 0 or (width % int(patch_size)) != 0:
        raise ValueError(
            f"{source} has image size {height}x{width}, which is incompatible with "
            f"DINO patch size {int(patch_size)}. Rebuild the cache with "
            "DINO-compatible tiling instead of relying on runtime cropping."
        )


def forward_with_optional_extras(
    model_call: Any,
    image: torch.Tensor,
    features: list[torch.Tensor],
    require_aux_logits: bool = False,
    labels: torch.Tensor | None = None,
    ignore_index: int | None = None,
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
        require_aux_logits (bool): Whether aux logits must be present.
        labels (torch.Tensor | None): Optional semantic labels for native-loss
            heads.
        ignore_index (int | None): Optional ignore label for native-loss heads.

    Returns:
        tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            dict[str, Any],
        ]:
        Main logits, aux logits, optional edge logits, optional skeleton logits,
        and normalized payload extras.

    Examples:
        >>> class Dummy:
        ...     def forward_with_aux(self, image, features):
        ...         return features[0], None
        >>> logits, aux, edge, skel, payload = forward_with_optional_extras(
        ...     Dummy(),
        ...     torch.randn(1, 3, 2, 2),
        ...     [torch.randn(1, 2, 2, 2)],
        ... )
        >>> aux is None and edge is None and skel is None
        True
        >>> payload["edge_logits"] is None and payload["skeleton_logits"] is None
        True
    """

    native_loss_fn = getattr(model_call, "forward_with_native_loss", None)
    wrapped_head = getattr(getattr(model_call, "module", model_call), "head", None)
    wrapped_native_loss_fn = (
        getattr(wrapped_head, "forward_with_native_loss", None)
        if wrapped_head is not None
        else None
    )
    if labels is not None and callable(native_loss_fn):
        raw_output = cast(Any, native_loss_fn)(
            image,
            features,
            labels,
            ignore_index=ignore_index,
        )
    elif labels is not None and callable(wrapped_native_loss_fn):
        raw_output = cast(Any, model_call)(
            image,
            features,
            labels=labels,
            ignore_index=ignore_index,
        )
    elif hasattr(model_call, "forward_with_extras"):
        raw_output = cast(Any, model_call).forward_with_extras(image, features)
    elif hasattr(model_call, "forward_with_aux"):
        raw_output = cast(Any, model_call).forward_with_aux(image, features)
    else:
        raw_output = cast(Any, model_call)(image, features)
    payload = normalize_forward_output(raw_output)
    logits = cast(torch.Tensor, payload["logits"])
    aux_logits = cast(torch.Tensor | None, payload.get("aux_logits"))
    edge_logits = cast(torch.Tensor | None, payload.get("edge_logits"))
    skeleton_logits = cast(torch.Tensor | None, payload.get("skeleton_logits"))
    if require_aux_logits and aux_logits is None:
        model_name = type(model_call).__name__
        raise RuntimeError(
            "Auxiliary supervision is enabled, but the model forward path did not "
            f"return aux logits. Received wrapper type '{model_name}'."
        )
    return logits, aux_logits, edge_logits, skeleton_logits, payload


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

    embedding_param_ids = {
        id(param)
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding)
        for param in module.parameters(recurse=False)
    }
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for _, p in model.named_parameters():
        if id(p) in embedding_param_ids or p.ndim < 2:
            adamw_params.append(p)
        elif p.ndim >= 2:
            muon_params.append(p)
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


def head_uses_backbone_features(head_name: str) -> bool:
    """Return whether a head consumes DINO backbone features.

    Args:
        head_name (str): Model head registry key.

    Returns:
        bool: ``True`` when the head expects DINO feature tensors.

    Examples:
        >>> head_uses_backbone_features("unet")
        False
        >>> head_uses_backbone_features("unet_nano")
        True
    """

    return str(head_name).strip().lower() not in _IMAGE_ONLY_HEADS


def head_supports_aux_logits(head_name: str) -> bool:
    """Return whether a head exposes auxiliary supervision logits.

    Args:
        head_name (str): Model head registry key.

    Returns:
        bool: ``True`` when the head returns auxiliary logits.

    Examples:
        >>> head_supports_aux_logits("unet")
        False
        >>> head_supports_aux_logits("unet_nano")
        True
    """

    return str(head_name).strip().lower() in _AUX_LOGIT_HEADS


def head_uses_native_loss(head_name: str) -> bool:
    """Return whether a head optimizes through its own native objective.

    Args:
        head_name (str): Model head registry key.

    Returns:
        bool: ``True`` when the shared CE/Dice loss should be bypassed.

    Examples:
        >>> head_uses_native_loss("mask2former_semantic")
        True
        >>> head_uses_native_loss("deeplabv3")
        False
    """

    return str(head_name).strip().lower() in _NATIVE_LOSS_HEADS


def resolve_model_patch_size(backbone_name: str, head_name: str) -> int:
    """Return the spatial compatibility multiple required by the active head.

    Args:
        backbone_name (str): Backbone model identifier.
        head_name (str): Segmentation head registry key.

    Returns:
        int: Patch-size multiple. Image-only heads return ``1``.

    Examples:
        >>> resolve_model_patch_size("facebook/dinov3-vitl16-pretrain-sat493m", "unet")
        1
        >>> resolve_model_patch_size("facebook/dinov3-vitl16-pretrain-sat493m", "unet_nano")
        16
    """

    if not head_uses_backbone_features(head_name):
        return 1
    return 14 if "vitl14" in str(backbone_name) else 16


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


def build_native_loss_components(loss: torch.Tensor) -> dict[str, torch.Tensor]:
    """Wrap one native head loss into the shared component dictionary.

    Args:
        loss (torch.Tensor): Scalar native-loss tensor.

    Returns:
        dict[str, torch.Tensor]: Component mapping compatible with the rest of
        the training/validation logging code.

    Examples:
        >>> comps = build_native_loss_components(torch.tensor(2.5))
        >>> float(comps["loss_total"]), float(comps["loss_weighted_main"])
        (2.5, 2.5)
    """

    zero = torch.zeros_like(loss)
    components = {key: zero.clone() for key in LOSS_COMPONENT_KEYS}
    components["loss_total"] = loss
    components["loss_weighted_main"] = loss
    return components


def resolve_lr_metrics(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> tuple[float, float, float]:
    """Resolve generic and component learning-rate metrics safely.

    Args:
        optimizer (torch.optim.Optimizer): Active optimizer instance.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Active scheduler.

    Returns:
        tuple[float, float, float]:
        ``(lr, lr_muon, lr_adamw)`` metrics for logging.

    Examples:
        >>> p = torch.nn.Parameter(torch.ones(1))
        >>> opt = torch.optim.AdamW([p], lr=1e-3)
        >>> sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=1, steps_per_epoch=1)
        >>> lr, lr_muon, lr_adamw = resolve_lr_metrics(opt, sch)
        >>> lr_muon == 0.0 and lr >= 0.0 and lr_adamw >= 0.0
        True
    """

    lr = float(scheduler.get_last_lr()[0])
    if isinstance(optimizer, Muon):
        group0 = optimizer.param_groups[0]
        return lr, lr, float(group0.get("adamw_lr", group0.get("lr", lr)))
    group0 = optimizer.param_groups[0]
    return lr, 0.0, float(group0.get("lr", lr))


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
    requires_backbone_features: bool = True,
    require_aux_logits: bool = False,
    uses_native_loss: bool = False,
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
        requires_backbone_features (bool): Whether the head needs DINO features.
        require_aux_logits (bool): Whether aux logits must be present.
        uses_native_loss (bool): Whether to optimize through a head-provided
            native objective instead of the shared CE/Dice loss.

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
            require_patch_grid_compatible(
                img,
                ps,
                source=f"Validation batch {batch_idx}",
            )
            if not requires_backbone_features:
                feats = []
            elif cache_features and features:
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
                        require_aux_logits=require_aux_logits,
                        labels=y if uses_native_loss else None,
                        ignore_index=loss_fn.ignore_index,
                    )
                )
                logits = cast(torch.Tensor, align_logits_to_labels(logits, y))
                aux_logits = align_logits_to_labels(aux_logits, y)
                edge_logits = align_logits_to_labels(edge_logits, y)
                skeleton_logits = align_logits_to_labels(skeleton_logits, y)
                target_main = y if y.ndim == 3 else y.unsqueeze(0)
                native_loss = payload.get("native_loss")
                if uses_native_loss:
                    if native_loss is None:
                        raise RuntimeError(
                            "Native-loss head did not return payload['native_loss']."
                        )
                    loss = torch.as_tensor(native_loss)
                    components = build_native_loss_components(loss)
                else:
                    target_aux = target_main if aux_logits is not None else None
                    edge_targets, edge_mask = build_boundary_targets(
                        labels=y,
                        edge_logits=edge_logits,
                        num_classes=loss_fn.num_classes,
                        ignore_index=loss_fn.ignore_index,
                        kernel_size=boundary_kernel_size,
                    )
                    logits_for_loss = (
                        logits.float() if stability_cfg.loss_fp32 else logits
                    )
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
