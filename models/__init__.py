"""
Factories for segmentation heads.

This keeps the main script simple: pick a head by string name and instantiate
it with consistent defaults.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, cast

from .base import SegmentationHead
from .maskformer import DinoMaskFormerHead
from .unet import DinoUNetHead
from .unet_lite_plus import DinoUNetLitePlusHead
from .unet_nano import DinoUNetNanoHead
from .unet_nano_fapm import DinoUNetNanoFAPMHead
from .unet_topo_fusion import DinoUNetTopoFusionHead
from .unet_v2 import DinoUNetV2Head
from .UnetLite import DinoUNetLiteHead

HeadBuilder = Callable[[int, int], SegmentationHead]


def available_heads() -> Dict[str, HeadBuilder]:
    """
    Return the set of supported segmentation head builders.

    Returns:
        Dict[str, HeadBuilder]: Mapping of head names to builders.

    >>> sorted(available_heads().keys()) == [
    ...     "maskformer",
    ...     "unet",
    ...     "unet_lite",
    ...     "unet_lite_plus",
    ...     "unet_nano",
    ...     "unet_nano_fapm",
    ...     "unet_topo_fusion",
    ...     "unet_v2",
    ... ]
    True
    """

    return {
        "unet": DinoUNetHead,
        "unet_v2": DinoUNetV2Head,
        "maskformer": DinoMaskFormerHead,
        "unet_lite": DinoUNetLiteHead,
        "unet_lite_plus": DinoUNetLitePlusHead,
        "unet_nano": DinoUNetNanoHead,
        "unet_nano_fapm": DinoUNetNanoFAPMHead,
        "unet_topo_fusion": DinoUNetTopoFusionHead,
    }


def build_head(
    name: str,
    num_classes: int,
    dino_channels: int,
    model_cfg: dict[str, Any] | None = None,
) -> SegmentationHead:
    """
    Build a segmentation head by name.

    Args:
        name (str): Head name.
        num_classes (int): Number of classes.
        dino_channels (int): DINO feature channel count.
        model_cfg (dict[str, Any] | None): Optional model config overrides.

    Returns:
        SegmentationHead: Instantiated head module.

    >>> head = build_head("unet", num_classes=2, dino_channels=1024)
    >>> hasattr(head, "forward")
    True
    """

    registry = available_heads()
    if name not in registry:
        raise ValueError(f"Unknown head '{name}'. Choose from: {sorted(registry)}")
    if name == "unet_topo_fusion":
        cfg = model_cfg or {}
        enable_layer_fusion = bool(cfg.get("enable_layer_fusion", True))
        enable_lora = bool(cfg.get("enable_lora", True))
        enable_boundary_gate = bool(cfg.get("enable_boundary_gate", True))
        max_layers_for_fusion = int(cfg.get("max_layers_for_fusion", 6))
        if not enable_layer_fusion:
            max_layers_for_fusion = 1
        lora_alpha = float(cfg.get("lora_alpha", 16.0))
        if not enable_lora:
            lora_alpha = 0.0
        boundary_gate_scale = float(cfg.get("boundary_gate_scale", 0.1))
        if not enable_boundary_gate:
            boundary_gate_scale = 0.0
        return DinoUNetTopoFusionHead(
            num_classes=num_classes,
            dino_channels=dino_channels,
            fusion_hidden=int(cfg.get("fusion_hidden", 64)),
            layer_fusion_hidden=(
                None
                if cfg.get("layer_fusion_hidden") is None
                else int(cfg.get("layer_fusion_hidden"))
            ),
            lora_rank=int(cfg.get("lora_rank", 8)),
            lora_alpha=lora_alpha,
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            lora_freeze_base=bool(cfg.get("lora_freeze_base", True)),
            boundary_gate_scale=boundary_gate_scale,
            boundary_gate_clamp=bool(cfg.get("boundary_gate_clamp", True)),
            max_layers_for_fusion=max_layers_for_fusion,
            layer_mix_maps_enable=bool(cfg.get("layer_mix_maps_enable", False)),
        )
    builder = cast(Any, registry[name])
    return builder(num_classes=num_classes, dino_channels=dino_channels)
