"""
Factories for segmentation heads.

This keeps the main script simple: pick a head by string name and instantiate
it with consistent defaults.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, cast

from .base import SegmentationHead
from .deeplabv3 import DeepLabV3Head
from .dino_dense_probe import DinoDenseProbeHead
from .dino_segdino_light import DinoSegDinoLightHead
from .maskformer import DinoMaskFormerHead
from .unet import DinoUNetHead
from .unet_lite_plus import DinoUNetLitePlusHead
from .unet_nano import DinoUNetNanoHead
from .unet_nano_fapm import DinoUNetNanoFAPMHead
from .unet_topo_fusion import DinoUNetTopoFusionHead
from .unet_v2 import DinoUNetV2Head
from .UnetLite import DinoUNetLiteHead

HeadBuilder = Callable[[int, int], SegmentationHead]


def _subcfg(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """
    Return a nested model-config block or an empty mapping.

    Args:
        cfg (dict[str, Any]): Parent model config mapping.
        key (str): Nested key to read.

    Returns:
        dict[str, Any]: Nested mapping when available, else ``{}``.
    """
    value = cfg.get(key, {})
    return value if isinstance(value, dict) else {}


def available_heads() -> Dict[str, HeadBuilder]:
    """
    Return the set of supported segmentation head builders.

    Returns:
        Dict[str, HeadBuilder]: Mapping of head names to builders.

    >>> sorted(available_heads().keys()) == [
    ...     "deeplabv3",
    ...     "dino_dense_probe",
    ...     "dino_segdino_light",
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
        "deeplabv3": DeepLabV3Head,
        "dino_dense_probe": DinoDenseProbeHead,
        "dino_segdino_light": DinoSegDinoLightHead,
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
    cfg = model_cfg or {}
    if name == "dino_dense_probe":
        dense_cfg = _subcfg(cfg, "dense_probe")
        return DinoDenseProbeHead(
            num_classes=num_classes,
            dino_channels=dino_channels,
            norm_type=str(dense_cfg.get("norm_type", "batchnorm")).strip().lower(),
            groupnorm_groups=int(dense_cfg.get("groupnorm_groups", 32)),
        )
    if name == "dino_segdino_light":
        if "segdino_light" in cfg:
            raise ValueError(
                "model.segdino_light is no longer supported. "
                "The dino_segdino_light head now uses fixed paper-like defaults "
                "and is configured only by model.layers."
            )
        layers = cfg.get("layers", [])
        if not isinstance(layers, list) or not layers:
            raise ValueError(
                "dino_segdino_light requires a non-empty model.layers list."
            )
        return DinoSegDinoLightHead(
            num_classes=num_classes,
            dino_channels=dino_channels,
            num_layers=int(len(layers)),
        )
    if name == "unet_topo_fusion":
        fusion_cfg = _subcfg(cfg, "fusion")
        lora_cfg = _subcfg(cfg, "lora")
        boundary_cfg = _subcfg(cfg, "boundary_gate")

        # Prefer nested keys; keep legacy flat keys for backward compatibility.
        enable_layer_fusion = bool(
            fusion_cfg.get("enable", cfg.get("enable_layer_fusion", True))
        )
        max_layers_for_fusion = int(
            fusion_cfg.get("max_layers", cfg.get("max_layers_for_fusion", 6))
        )
        if not enable_layer_fusion:
            max_layers_for_fusion = 1

        enable_lora = bool(lora_cfg.get("enable", cfg.get("enable_lora", True)))
        lora_alpha = float(lora_cfg.get("alpha", cfg.get("lora_alpha", 16.0)))
        if not enable_lora:
            lora_alpha = 0.0

        enable_boundary_gate = bool(
            boundary_cfg.get("enable", cfg.get("enable_boundary_gate", True))
        )
        boundary_gate_scale = float(
            boundary_cfg.get("scale", cfg.get("boundary_gate_scale", 0.1))
        )
        if not enable_boundary_gate:
            boundary_gate_scale = 0.0
        layer_fusion_hidden_raw = fusion_cfg.get(
            "layer_hidden", cfg.get("layer_fusion_hidden")
        )

        return DinoUNetTopoFusionHead(
            num_classes=num_classes,
            dino_channels=dino_channels,
            fusion_hidden=int(fusion_cfg.get("hidden", cfg.get("fusion_hidden", 64))),
            layer_fusion_hidden=(
                None
                if layer_fusion_hidden_raw is None
                else int(layer_fusion_hidden_raw)
            ),
            lora_rank=int(lora_cfg.get("rank", cfg.get("lora_rank", 8))),
            lora_alpha=lora_alpha,
            lora_dropout=float(lora_cfg.get("dropout", cfg.get("lora_dropout", 0.0))),
            lora_freeze_base=bool(
                lora_cfg.get("freeze_base", cfg.get("lora_freeze_base", True))
            ),
            boundary_gate_scale=boundary_gate_scale,
            boundary_gate_clamp=bool(
                boundary_cfg.get("clamp", cfg.get("boundary_gate_clamp", True))
            ),
            max_layers_for_fusion=max_layers_for_fusion,
            layer_mix_maps_enable=bool(
                fusion_cfg.get("save_maps", cfg.get("layer_mix_maps_enable", False))
            ),
        )
    builder = cast(Any, registry[name])
    return builder(num_classes=num_classes, dino_channels=dino_channels)
