"""Training configuration parsing helpers for nested and legacy keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResolvedLossConfig:
    """Resolved loss settings used to build `SegmentationLoss`."""

    ce_weight: float
    focal_weight: float
    dice_weight: float
    aux_weight: float
    class_weights: list[float] | None
    ignore_index: int | None
    label_smoothing: float
    focal_gamma: float
    focal_alpha: float | None
    boundary_weight: float
    boundary_kernel_size: int
    skeleton_weight: float
    topology_weight: float
    topology_class_index: int
    topology_iters: int
    topology_on_aux: bool
    topology_downsample: int


@dataclass(frozen=True)
class ResolvedPlotConfig:
    """Resolved plotting and XAI settings used by training."""

    enabled: bool
    root_dir: str
    cmap: str
    pairs: int
    seed_offset: int
    metric_class_index: int
    xai_enable: bool
    xai_class_index: int
    xai_topk_channels: int
    xai_channel_top_k_per_sample: int
    xai_cam_layer_mode: str
    xai_render_rollout: bool
    xai_pca_enable: bool
    xai_pca_layer_mode: str
    xai_branch_importance_enable: bool
    xai_branch_importance_class_index: int
    xai_branch_importance_max_samples: int
    xai_channel_tracking_enable: bool
    xai_channel_tracking_max_samples: int
    xai_channel_top_n_stable: int
    xai_channel_min_presence: float
    xai_channel_save_json: bool
    xai_module_cfg: dict[str, Any]


def parse_train_loss_config(
    train_cfg: dict[str, Any],
    dataset_ignore_index: int | None = None,
) -> ResolvedLossConfig:
    """Resolve train-loss settings from nested and legacy keys.

    Args:
        train_cfg (dict[str, Any]): Train config section.
        dataset_ignore_index (int | None): Validation ignore index fallback.

    Returns:
        ResolvedLossConfig: Parsed loss settings.

    Examples:
        >>> cfg = {
        ...     "loss": {
        ...         "main": {"ce_weight": 1.0, "dice_weight": 1.0},
        ...         "focal": {"weight": 0.0, "gamma": 2.0},
        ...     },
        ...     "topology": {"weight": 0.2, "skeleton_weight": 0.1},
        ... }
        >>> out = parse_train_loss_config(cfg, dataset_ignore_index=255)
        >>> (out.ce_weight, out.focal_weight, out.topology_weight, out.ignore_index)
        (1.0, 0.0, 0.2, 255)
    """

    loss_cfg = train_cfg.get("loss", {})
    loss_main_cfg = loss_cfg.get("main", {}) if isinstance(loss_cfg, dict) else {}
    loss_focal_cfg = loss_cfg.get("focal", {}) if isinstance(loss_cfg, dict) else {}
    loss_boundary_cfg = (
        loss_cfg.get("boundary", {}) if isinstance(loss_cfg, dict) else {}
    )
    loss_topology_cfg = (
        loss_cfg.get("topology", {}) if isinstance(loss_cfg, dict) else {}
    )
    train_topology_cfg = train_cfg.get("topology", {})
    topology_cfg: dict[str, Any] = {}
    if isinstance(loss_topology_cfg, dict):
        topology_cfg.update(loss_topology_cfg)
    if isinstance(train_topology_cfg, dict):
        topology_cfg.update(train_topology_cfg)

    ce_weight = float(loss_main_cfg.get("ce_weight", loss_cfg.get("ce_weight", 1.0)))
    focal_weight = loss_focal_cfg.get("weight")
    if focal_weight is None:
        use_focal_legacy = bool(loss_cfg.get("use_focal", False))
        focal_weight = ce_weight if use_focal_legacy else 0.0
        if use_focal_legacy and ce_weight > 0:
            ce_weight = 0.0

    ignore_index = loss_cfg.get("ignore_index")
    if ignore_index is None:
        ignore_index = dataset_ignore_index

    return ResolvedLossConfig(
        ce_weight=ce_weight,
        focal_weight=float(focal_weight),
        dice_weight=float(
            loss_main_cfg.get("dice_weight", loss_cfg.get("dice_weight", 1.0))
        ),
        aux_weight=float(
            loss_main_cfg.get("aux_weight", loss_cfg.get("aux_weight", 0.4))
        ),
        class_weights=loss_cfg.get("class_weights"),
        ignore_index=ignore_index,
        label_smoothing=float(
            loss_main_cfg.get("label_smoothing", loss_cfg.get("label_smoothing", 0.0))
        ),
        focal_gamma=float(
            loss_focal_cfg.get("gamma", loss_cfg.get("focal_gamma", 2.0))
        ),
        focal_alpha=loss_focal_cfg.get("alpha", loss_cfg.get("focal_alpha")),
        boundary_weight=float(
            loss_boundary_cfg.get("weight", loss_cfg.get("boundary_weight", 0.1))
        ),
        boundary_kernel_size=max(
            3,
            int(
                loss_boundary_cfg.get(
                    "kernel_size", loss_cfg.get("boundary_kernel_size", 3)
                )
            ),
        ),
        skeleton_weight=float(
            topology_cfg.get("skeleton_weight", loss_cfg.get("skeleton_weight", 0.0))
        ),
        topology_weight=float(
            topology_cfg.get("weight", loss_cfg.get("topology_weight", 0.0))
        ),
        topology_class_index=int(
            topology_cfg.get("class_index", loss_cfg.get("topology_class_index", 1))
        ),
        topology_iters=int(
            topology_cfg.get("iters", loss_cfg.get("topology_iters", 10))
        ),
        topology_on_aux=bool(
            topology_cfg.get("on_aux", loss_cfg.get("topology_on_aux", True))
        ),
        topology_downsample=int(
            topology_cfg.get("downsample", loss_cfg.get("topology_downsample", 1))
        ),
    )


def parse_train_plot_config(train_cfg: dict[str, Any]) -> ResolvedPlotConfig:
    """Resolve train-plot settings from nested and legacy keys.

    Args:
        train_cfg (dict[str, Any]): Train config section.

    Returns:
        ResolvedPlotConfig: Parsed plotting settings.

    Examples:
        >>> cfg = {
        ...     "plots": {
        ...         "epoch": {"enable": True, "pairs": 2},
        ...         "xai": {"enable": True, "module": {"enable": False}},
        ...     }
        ... }
        >>> out = parse_train_plot_config(cfg)
        >>> (out.enabled, out.pairs, out.xai_enable, out.xai_module_cfg["enable"])
        (True, 2, True, False)
    """

    plots_cfg = (
        train_cfg.get("plots", {})
        if isinstance(train_cfg.get("plots", {}), dict)
        else {}
    )
    epoch_plot_cfg = (
        plots_cfg.get("epoch", {})
        if isinstance(plots_cfg.get("epoch", {}), dict)
        else {}
    )
    xai_plot_cfg = (
        plots_cfg.get("xai", {}) if isinstance(plots_cfg.get("xai", {}), dict) else {}
    )
    xai_pca_cfg = (
        xai_plot_cfg.get("pca", {})
        if isinstance(xai_plot_cfg.get("pca", {}), dict)
        else {}
    )
    xai_branch_cfg = (
        xai_plot_cfg.get("branch_importance", {})
        if isinstance(xai_plot_cfg.get("branch_importance", {}), dict)
        else {}
    )
    xai_channel_cfg = (
        xai_plot_cfg.get("channel_tracking", {})
        if isinstance(xai_plot_cfg.get("channel_tracking", {}), dict)
        else {}
    )
    xai_module_nested_cfg = (
        xai_plot_cfg.get("module", {})
        if isinstance(xai_plot_cfg.get("module", {}), dict)
        else {}
    )
    xai_module_cfg: dict[str, Any] = {
        "enable": bool(train_cfg.get("epoch_plot_xai_module_enable", True)),
        "every_n_epochs": int(train_cfg.get("epoch_plot_xai_module_every_n_epochs", 5)),
        "max_samples": int(train_cfg.get("epoch_plot_xai_module_max_samples", 8)),
        "save_maps": bool(train_cfg.get("epoch_plot_xai_module_save_maps", True)),
        "boundary_band_px": int(
            train_cfg.get("epoch_plot_xai_module_boundary_band_px", 3)
        ),
        "gate_threshold": float(
            train_cfg.get("epoch_plot_xai_module_gate_threshold", 0.5)
        ),
        "entropy_eps": float(
            train_cfg.get("epoch_plot_xai_module_entropy_eps", 1.0e-8)
        ),
        "strict": bool(train_cfg.get("epoch_plot_xai_module_strict", False)),
        "enable_lora_ratio": bool(
            train_cfg.get("epoch_plot_xai_module_enable_lora_ratio", True)
        ),
        "enable_topology_panels": bool(
            train_cfg.get("epoch_plot_xai_module_enable_topology_panels", True)
        ),
    }
    if xai_module_nested_cfg:
        xai_module_cfg.update(xai_module_nested_cfg)

    metric_class_index = int(
        epoch_plot_cfg.get(
            "metric_class_index",
            train_cfg.get("epoch_plot_metric_class_index", 1),
        )
    )
    xai_class_index = int(
        xai_plot_cfg.get(
            "class_index",
            train_cfg.get("epoch_plot_xai_class_index", metric_class_index),
        )
    )
    xai_topk_channels = max(
        1,
        int(
            xai_plot_cfg.get(
                "topk_channels",
                train_cfg.get("epoch_plot_xai_topk_channels", 5),
            )
        ),
    )

    return ResolvedPlotConfig(
        enabled=bool(epoch_plot_cfg.get("enable", train_cfg.get("epoch_plot", False))),
        root_dir=str(
            epoch_plot_cfg.get(
                "dir",
                train_cfg.get("epoch_plot_dir", "output/plot"),
            )
        ),
        cmap=str(epoch_plot_cfg.get("cmap", train_cfg.get("epoch_plot_cmap", "tab20"))),
        pairs=max(
            1, int(epoch_plot_cfg.get("pairs", train_cfg.get("epoch_plot_pairs", 4)))
        ),
        seed_offset=int(
            epoch_plot_cfg.get(
                "seed_offset",
                train_cfg.get("epoch_plot_seed_offset", 1000),
            )
        ),
        metric_class_index=metric_class_index,
        xai_enable=bool(
            xai_plot_cfg.get("enable", train_cfg.get("epoch_plot_xai_enable", False))
        ),
        xai_class_index=xai_class_index,
        xai_topk_channels=xai_topk_channels,
        xai_channel_top_k_per_sample=max(
            1,
            int(
                xai_channel_cfg.get(
                    "top_k_per_sample",
                    train_cfg.get(
                        "epoch_plot_xai_channel_top_k_per_sample", xai_topk_channels
                    ),
                )
            ),
        ),
        xai_cam_layer_mode=str(
            xai_plot_cfg.get(
                "cam_layer_mode",
                train_cfg.get("epoch_plot_xai_cam_layer_mode", "last_requested_layer"),
            )
        ),
        xai_render_rollout=bool(
            xai_plot_cfg.get(
                "render_attn_rollout",
                train_cfg.get("epoch_plot_xai_render_attn_rollout", True),
            )
        ),
        xai_pca_enable=bool(
            xai_pca_cfg.get("enable", train_cfg.get("epoch_plot_xai_pca_enable", True))
        ),
        xai_pca_layer_mode=str(
            xai_pca_cfg.get(
                "layer_mode",
                train_cfg.get("epoch_plot_xai_pca_layer_mode", "same_as_cam"),
            )
        ),
        xai_branch_importance_enable=bool(
            xai_branch_cfg.get(
                "enable",
                train_cfg.get("epoch_plot_xai_branch_importance_enable", True),
            )
        ),
        xai_branch_importance_class_index=int(
            xai_branch_cfg.get(
                "class_index",
                train_cfg.get(
                    "epoch_plot_xai_branch_importance_class_index",
                    xai_class_index,
                ),
            )
        ),
        xai_branch_importance_max_samples=max(
            1,
            int(
                xai_branch_cfg.get(
                    "max_samples",
                    train_cfg.get("epoch_plot_xai_branch_importance_max_samples", 4),
                )
            ),
        ),
        xai_channel_tracking_enable=bool(
            xai_channel_cfg.get(
                "enable",
                train_cfg.get("epoch_plot_xai_channel_tracking_enable", True),
            )
        ),
        xai_channel_tracking_max_samples=max(
            1,
            int(
                xai_channel_cfg.get(
                    "max_samples",
                    train_cfg.get("epoch_plot_xai_channel_tracking_max_samples", 64),
                )
            ),
        ),
        xai_channel_top_n_stable=max(
            1,
            int(
                xai_channel_cfg.get(
                    "top_n_stable",
                    train_cfg.get("epoch_plot_xai_channel_top_n_stable", 10),
                )
            ),
        ),
        xai_channel_min_presence=min(
            1.0,
            max(
                0.0,
                float(
                    xai_channel_cfg.get(
                        "min_presence",
                        train_cfg.get("epoch_plot_xai_channel_min_presence", 0.05),
                    )
                ),
            ),
        ),
        xai_channel_save_json=bool(
            xai_channel_cfg.get(
                "save_json",
                train_cfg.get("epoch_plot_xai_channel_save_json", True),
            )
        ),
        xai_module_cfg=xai_module_cfg,
    )
