"""Pipeline configuration helpers and orchestration utilities."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from utils import VerbosityLogger

from .constants import (
    DEFAULT_DINO_CHANNELS,
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_HEAD,
    DEFAULT_LAYERS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TRACKING_DIR,
)
from .context import DatasetValidationConfig, DistContext, RunContext, StabilityConfig
from .tracking import (
    ConfigSnapshotProcessor,
    Hook,
    HookManager,
    MetricsWriter,
    MlflowFileLogger,
    MlflowHook,
    Processor,
    RunSummaryProcessor,
)


def setup_distributed(resources_cfg: dict) -> DistContext:
    """Initialize distributed context from the resources config.

    Args:
        resources_cfg (dict): Resource configuration block.

    Returns:
        DistContext: Initialized distributed context.

    Raises:
        RuntimeError: If distributed training is requested but unavailable.

    Examples:
        >>> setup_distributed({"distributed": False}).enabled
        False
    """

    dist_flag = resources_cfg.get("distributed", False)
    ctx = DistContext()
    if not dist_flag:
        return ctx
    if not dist.is_available():
        raise RuntimeError(
            "distributed training requested but torch.distributed unavailable"
        )
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        raise RuntimeError(
            "Distributed mode requires torchrun/launch to set RANK and WORLD_SIZE"
        )
    backend = resources_cfg.get("dist_backend", "nccl")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    ctx.enabled = True
    ctx.rank = rank
    ctx.world_size = world_size
    ctx.local_rank = local_rank
    return ctx


def cleanup_distributed(ctx: DistContext) -> None:
    """Shutdown distributed process group if active.

    Args:
        ctx (DistContext): Distributed context.

    Examples:
        >>> cleanup_distributed(DistContext())
    """

    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def apply_resource_config(config: dict) -> None:
    """Apply thread, seed, and precision settings from the config.

    Args:
        config (dict): Configuration dictionary.

    Examples:
        >>> apply_resource_config({"resources": {"seed": 1}})
    """

    res_cfg = config.get("resources", {})
    threads = res_cfg.get("omp_threads")
    if threads:
        for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[env_var] = str(threads)
    precision = res_cfg.get("matmul_precision", "high")
    torch.set_float32_matmul_precision(precision)
    seed = res_cfg.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = res_cfg.get("cudnn_benchmark", True)


def build_logger(config: dict, run_id: str, enabled: bool = True) -> VerbosityLogger:
    """Create a VerbosityLogger using the logging configuration.

    Args:
        config (dict): Configuration dictionary.
        run_id (str): Run identifier used for per-run log files.
        enabled (bool): Whether logging is enabled.

    Returns:
        VerbosityLogger: Configured logger instance.

    Examples:
        >>> logger = build_logger(
        ...     {"logging": {"level": "debug", "timestamps": False, "per_run": False}},
        ...     run_id="demo",
        ... )
        >>> logger.debug("configured")
        [DEBUG] configured
    """

    logging_cfg = config.get("logging", {})
    level = logging_cfg.get("level", "info")
    timestamps = logging_cfg.get("timestamps", True)
    log_file = logging_cfg.get("file")
    per_run = bool(logging_cfg.get("per_run", True))
    if per_run:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = logging_cfg.get("dir")
        if log_dir:
            base = logging_cfg.get("base_name", "run")
            log_file = str(Path(log_dir) / f"{base}_{timestamp}_{run_id}.log")
        elif log_file:
            path = Path(log_file)
            base = path.stem or "run"
            log_file = str(path.with_name(f"{base}_{timestamp}_{run_id}.log"))
        else:
            log_file = str(Path("logs") / f"run_{timestamp}_{run_id}.log")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    return VerbosityLogger(
        level=level, timestamps=timestamps, log_file=log_file, enabled=enabled
    )


def section_enabled(config: dict, name: str) -> bool:
    """Return True if the named section has enable=true.

    Args:
        config (dict): Configuration dictionary.
        name (str): Section name.

    Returns:
        bool: True if the section is enabled.

    Examples:
        >>> section_enabled({"prepare": {"enable": True}}, "prepare")
        True
    """

    section = config.get(name, {})
    return bool(section.get("enable", False))


def resolve_path(
    config: dict,
    section: dict,
    key: str,
    fallback: str,
    legacy_keys: tuple[str, ...] = (),
) -> str:
    """Resolve a path from a section, falling back to global paths or defaults.

    Args:
        config (dict): Configuration dictionary.
        section (dict): Phase-specific section.
        key (str): Configuration key to resolve.
        fallback (str): Fallback path.
        legacy_keys (tuple[str, ...]): Optional backward-compatible aliases
            checked after `key`.

    Returns:
        str: Resolved path string.

    Examples:
        >>> cfg = {"paths": {"processed_dir": "/tmp/proc"}}
        >>> resolve_path(cfg, {"processed_dir": "/custom"}, "processed_dir", "/default")
        '/custom'
        >>> cfg = {"paths": {"raw_images_dir": "/tmp/imgs"}}
        >>> resolve_path(cfg, {}, "raw_images_dir", "/default", ("img_dir",))
        '/tmp/imgs'
        >>> cfg = {"paths": {"img_dir": "/tmp/legacy"}}
        >>> resolve_path(cfg, {}, "raw_images_dir", "/default", ("img_dir",))
        '/tmp/legacy'
    """

    paths_cfg = config.get("paths", {})
    direct = section.get(key) or paths_cfg.get(key)
    if direct:
        return direct
    for legacy_key in legacy_keys:
        legacy_value = section.get(legacy_key) or paths_cfg.get(legacy_key)
        if legacy_value:
            return legacy_value
    return fallback


def get_model_config(config: dict) -> dict[str, Any]:
    """Return model configuration with defaults applied.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict[str, Any]: Model configuration.

    Examples:
        >>> get_model_config({"model": {"head": "maskformer"}})["head"]
        'maskformer'
    """

    model_cfg = config.get("model", {})
    return {
        "backbone": model_cfg.get("backbone", DEFAULT_MODEL_NAME),
        "layers": model_cfg.get("layers", DEFAULT_LAYERS),
        "head": model_cfg.get("head", DEFAULT_HEAD),
        "num_classes": model_cfg.get("num_classes", DEFAULT_NUM_CLASSES),
        "dino_channels": model_cfg.get("dino_channels", DEFAULT_DINO_CHANNELS),
    }


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model, unwrapping DDP if needed.

    Args:
        model (torch.nn.Module): Model possibly wrapped in DDP.

    Returns:
        torch.nn.Module: Unwrapped model instance.

    Examples:
        >>> unwrap_model(torch.nn.Linear(2, 2)).__class__.__name__
        'Linear'
    """

    from torch.nn.parallel import DistributedDataParallel as DDP

    if isinstance(model, DDP):
        return model.module
    return model


def get_hook_option(config: dict, key: str, default: Any) -> Any:
    """Return a hook option value from the pipeline config.

    Args:
        config (dict): Configuration dictionary.
        key (str): Hook option key.
        default (Any): Default value if the key is missing.

    Returns:
        Any: Resolved option value.

    Examples:
        >>> get_hook_option({"pipeline": {"hook_options": {"x": 1}}}, "x", 0)
        1
    """

    pipeline_cfg = config.get("pipeline", {})
    options = pipeline_cfg.get("hook_options", {})
    return options.get(key, default)


def parse_stability_config(config: dict) -> StabilityConfig:
    """Parse numeric stability options from the training config.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        StabilityConfig: Parsed stability settings with safe defaults.

    Examples:
        >>> cfg = {"train": {"stability": {"nonfinite": {"action": "skip_batch"}}}}
        >>> parse_stability_config(cfg).nonfinite_action
        'skip_batch'
    """

    train_cfg = config.get("train", {})
    stable_cfg = train_cfg.get("stability", {})
    amp_cfg = stable_cfg.get("amp", {})
    nonfinite_cfg = stable_cfg.get("nonfinite", {})
    amp_enabled = str(
        amp_cfg.get("enabled", stable_cfg.get("amp_enabled", "auto"))
    ).lower()
    if amp_enabled in {"true", "1", "yes"}:
        amp_enabled = "on"
    if amp_enabled in {"false", "0", "no"}:
        amp_enabled = "off"
    if amp_enabled not in {"auto", "on", "off"}:
        amp_enabled = "auto"
    amp_dtype = str(amp_cfg.get("dtype", stable_cfg.get("amp_dtype", "bf16"))).lower()
    if amp_dtype not in {"bf16", "fp16"}:
        amp_dtype = "bf16"
    action = str(
        nonfinite_cfg.get("action", stable_cfg.get("nonfinite_action", "stop_run"))
    ).lower()
    if action not in {"stop_run", "stop_epoch", "skip_batch"}:
        action = "stop_run"
    check_params_every_steps = int(stable_cfg.get("check_params_every_steps", 50) or 50)
    return StabilityConfig(
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        loss_fp32=bool(stable_cfg.get("loss_fp32", True)),
        grad_clip_norm=max(0.0, float(stable_cfg.get("grad_clip_norm", 1.0))),
        max_abs_logit_warn=max(1.0, float(stable_cfg.get("max_abs_logit_warn", 80.0))),
        nonfinite_action=action,
        nonfinite_max_consecutive_batches=max(
            1,
            int(
                nonfinite_cfg.get(
                    "max_consecutive_batches",
                    stable_cfg.get("nonfinite_max_consecutive_batches", 2),
                )
            ),
        ),
        nonfinite_max_total_batches_per_epoch=max(
            1,
            int(
                nonfinite_cfg.get(
                    "max_total_batches_per_epoch",
                    stable_cfg.get("nonfinite_max_total_batches_per_epoch", 5),
                )
            ),
        ),
        save_bad_batch_sample=bool(
            nonfinite_cfg.get(
                "save_bad_batch_sample",
                stable_cfg.get("save_bad_batch_sample", True),
            )
        ),
        check_params_every_steps=max(1, check_params_every_steps),
    )


def parse_dataset_validation_config(config: dict) -> DatasetValidationConfig:
    """Parse dataset semantic validation settings from the config.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        DatasetValidationConfig: Parsed dataset validation settings.

    Examples:
        >>> cfg = {"dataset": {"validation": {"allowed_labels": [0, 1, 2]}}}
        >>> parse_dataset_validation_config(cfg).allowed_labels
        (0, 1, 2)
    """

    dataset_cfg = config.get("dataset", {})
    validation_cfg = dataset_cfg.get("validation", {})
    allowed_raw = validation_cfg.get("allowed_labels", [0, 1])
    allowed_labels: tuple[int, ...]
    if isinstance(allowed_raw, (list, tuple)):
        allowed_labels = tuple(sorted({int(v) for v in allowed_raw}))
    else:
        allowed_labels = (0, 1)
    if not allowed_labels:
        allowed_labels = (0, 1)
    ignore_index_raw = validation_cfg.get("ignore_index", 255)
    ignore_index = None if ignore_index_raw is None else int(ignore_index_raw)
    policy = str(validation_cfg.get("out_of_range_policy", "map_to_ignore")).lower()
    if policy not in {"map_to_ignore", "error"}:
        policy = "map_to_ignore"
    return DatasetValidationConfig(
        enabled=bool(validation_cfg.get("enabled", True)),
        allowed_labels=allowed_labels,
        ignore_index=ignore_index,
        out_of_range_policy=policy,
        require_finite_features=bool(
            validation_cfg.get("require_finite_features", True)
        ),
        require_finite_images=bool(validation_cfg.get("require_finite_images", True)),
    )


def collect_run_params(config: dict) -> dict[str, str]:
    """Collect a curated set of run parameters for tracking.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict[str, str]: Parameter key/value pairs.

    Examples:
        >>> params = collect_run_params({"model": {"head": "unet"}})
        >>> params["model.head"]
        'unet'
    """

    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    resources_cfg = config.get("resources", {})
    dataset_cfg = config.get("dataset", {})
    stability_cfg = parse_stability_config(config)
    validation_cfg = parse_dataset_validation_config(config)
    muon_wd = train_cfg.get("muon_wd")
    if muon_wd is None:
        muon_wd = train_cfg.get("adamw_wd", 0.01)
    params: dict[str, str] = {
        "model.head": str(model_cfg.get("head", DEFAULT_HEAD)),
        "model.backbone": str(model_cfg.get("backbone", DEFAULT_MODEL_NAME)),
        "model.num_classes": str(model_cfg.get("num_classes", DEFAULT_NUM_CLASSES)),
        "train.batch_size": str(train_cfg.get("batch_size", 4)),
        "train.epochs": str(train_cfg.get("epochs", 30)),
        "train.muon_lr": str(train_cfg.get("muon_lr", 0.02)),
        "train.muon_wd": str(muon_wd),
        "train.muon_update_scale": str(train_cfg.get("muon_update_scale", 0.2)),
        "train.muon_adjust_lr_for_shape": str(
            train_cfg.get("muon_adjust_lr_for_shape", True)
        ),
        "train.adamw_lr": str(train_cfg.get("adamw_lr", 0.001)),
        "train.adamw_wd": str(train_cfg.get("adamw_wd", 0.01)),
        "resources.seed": str(resources_cfg.get("seed", "")),
        "resources.distributed": str(resources_cfg.get("distributed", False)),
        "dataset.augmentations": str(
            dataset_cfg.get("augmentations", {}).get("enable", False)
        ),
        "train.stability.amp_enabled": stability_cfg.amp_enabled,
        "train.stability.amp_dtype": stability_cfg.amp_dtype,
        "train.stability.nonfinite_action": stability_cfg.nonfinite_action,
        "dataset.validation.enabled": str(validation_cfg.enabled),
    }
    return params


def build_hooks(config: dict, mlflow_logger: MlflowFileLogger | None) -> list[Hook]:
    """Instantiate hooks from configuration.

    Args:
        config (dict): Configuration dictionary.
        mlflow_logger (MlflowFileLogger | None): MLflow logger if enabled.

    Returns:
        list[Hook]: Hook instances.

    Examples:
        >>> build_hooks({}, None)
        []
    """

    pipeline_cfg = config.get("pipeline", {})
    hook_names = pipeline_cfg.get("hooks")
    if hook_names is None:
        hook_names = ["mlflow"] if mlflow_logger else []
    hooks: list[Hook] = []
    log_batch_metrics = bool(get_hook_option(config, "log_batch_metrics", False))
    log_batch_interval = int(get_hook_option(config, "log_batch_interval", 10))
    for name in hook_names:
        if name == "mlflow" and mlflow_logger:
            hooks.append(
                MlflowHook(
                    mlflow_logger,
                    log_batch_metrics=log_batch_metrics,
                    log_batch_interval=log_batch_interval,
                )
            )
    return hooks


def build_processors(config: dict) -> list[Processor]:
    """Instantiate pre/post processors from configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        list[Processor]: Processor instances.

    Examples:
        >>> [p.__class__.__name__ for p in build_processors({})]
        ['ConfigSnapshotProcessor', 'RunSummaryProcessor']
    """

    pipeline_cfg = config.get("pipeline", {})
    before = pipeline_cfg.get("processors", {}).get("before", ["config_snapshot"])
    after = pipeline_cfg.get("processors", {}).get("after", ["run_summary"])
    processors: list[Processor] = []
    registry = {
        "config_snapshot": ConfigSnapshotProcessor,
        "run_summary": RunSummaryProcessor,
    }
    for name in before + after:
        builder = registry.get(name)
        if builder:
            processors.append(builder())
    return processors


def build_run_context(
    config: dict, logger: VerbosityLogger, dist_ctx: DistContext, run_id: str
) -> RunContext:
    """Create the RunContext with MLflow logging and hooks.

    Args:
        config (dict): Configuration dictionary.
        logger (VerbosityLogger): Logger instance.
        dist_ctx (DistContext): Distributed context.
        run_id (str): Run identifier for tracking and logging.

    Returns:
        RunContext: Initialized run context.

    Examples:
        >>> cfg = {"tracking": {"mlflow": {"enable": False}}}
        >>> ctx = build_run_context(
        ...     cfg,
        ...     VerbosityLogger(level="info", timestamps=False),
        ...     DistContext(),
        ...     run_id="demo",
        ... )
        >>> isinstance(ctx, RunContext)
        True
    """

    tracking_cfg = config.get("tracking", {}).get("mlflow", {})
    enable_tracking = tracking_cfg.get("enable", True)
    tracking_dir = tracking_cfg.get("tracking_dir", DEFAULT_TRACKING_DIR)
    experiment_id = str(tracking_cfg.get("experiment_id", DEFAULT_EXPERIMENT_ID))
    run_name = tracking_cfg.get("run_name")
    mlflow_logger = None
    run_dir = Path(".")
    if enable_tracking and dist_ctx.is_main:
        mlflow_logger = MlflowFileLogger(
            tracking_dir=tracking_dir,
            experiment_id=experiment_id,
            run_id=run_id,
            run_name=run_name,
            tags=tracking_cfg.get("tags", {}),
        )
        run_dir = mlflow_logger.run_dir
    metrics_writer = (
        MetricsWriter(run_dir / "artifacts" / "metrics.jsonl")
        if mlflow_logger
        else None
    )
    config["_run_params"] = collect_run_params(config)
    hooks = build_hooks(config, mlflow_logger)
    hook_manager = HookManager(hooks)
    continue_on_error = bool(
        config.get("resources", {}).get("continue_on_error", False)
    )
    stability = parse_stability_config(config)
    dataset_validation = parse_dataset_validation_config(config)
    return RunContext(
        config=config,
        logger=logger,
        dist_ctx=dist_ctx,
        mlflow_logger=mlflow_logger,
        hook_manager=hook_manager,
        metrics_writer=metrics_writer,
        run_dir=run_dir,
        experiment_id=experiment_id,
        run_id=run_id,
        start_time=time.time(),
        config_path=config.get("_config_path"),
        continue_on_error=continue_on_error,
        stability=stability,
        dataset_validation=dataset_validation,
        run_results=[],
    )
