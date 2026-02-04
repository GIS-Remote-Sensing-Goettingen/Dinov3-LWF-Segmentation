"""Entry point for the config-driven segmentation pipeline."""

from __future__ import annotations

import sys
import uuid

from pipeline.constants import ensure_env_defaults
from pipeline.context import ConfigError
from pipeline.phase_runner import PhaseRunner
from pipeline.phases import InferencePhase, PreparePhase, TrainPhase, VerifyPhase
from pipeline.utils import (
    apply_resource_config,
    build_logger,
    build_processors,
    build_run_context,
    cleanup_distributed,
    setup_distributed,
)
from utils import load_config


def main(config_path: str | None = None) -> None:
    """Load a YAML configuration file and execute the enabled phases.

    Args:
        config_path (str | None): Optional configuration path.

    Raises:
        ConfigError: If the configuration cannot be loaded.

    Examples:
        >>> callable(main)
        True
    """

    ensure_env_defaults()
    candidate = config_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    try:
        config = load_config(candidate)
    except Exception as exc:
        raise ConfigError(str(exc)) from exc
    apply_resource_config(config)
    dist_ctx = setup_distributed(config.get("resources", {}))
    run_id = uuid.uuid4().hex
    logger = build_logger(config, run_id=run_id, enabled=dist_ctx.is_main)
    logger.info(
        f"Loaded configuration from {config.get('_config_path', 'embedded dict')}"
    )
    context = build_run_context(config, logger, dist_ctx, run_id=run_id)
    phases = [PreparePhase(), VerifyPhase(), TrainPhase(), InferencePhase()]
    processors = build_processors(config)
    runner = PhaseRunner(phases=phases, processors=processors)
    results = runner.run(context)
    status = "FINISHED"
    if any(result.status == "failed" for result in results):
        status = "FAILED"
    if context.mlflow_logger:
        context.mlflow_logger.close(status)
    logger.info("All enabled phases completed.")
    cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
