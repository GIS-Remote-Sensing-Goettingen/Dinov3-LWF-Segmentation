"""Phase base class and runner for the pipeline."""

from __future__ import annotations

import time

from .context import PhaseError, PhaseOutcome, PhaseResult, RunContext
from .utils import section_enabled


class Phase:
    """Base class for pipeline phases.

    Subclasses should implement execute() and set name/config_key attributes.

    Examples:
        >>> isinstance(Phase, type)
        True
    """

    name: str = "phase"
    config_key: str = "phase"

    def is_enabled(self, context: RunContext) -> bool:
        """Return True when the phase should execute.

        Args:
            context (RunContext): Active run context.

        Returns:
            bool: True when the phase is enabled.
        """

        return section_enabled(context.config, self.config_key)

    def execute(self, context: RunContext) -> PhaseOutcome:
        """Execute the phase work and return metrics/artifacts.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseOutcome: Metrics and artifacts produced by the phase.
        """

        raise NotImplementedError("Phase subclasses must implement execute().")

    def run(self, context: RunContext) -> PhaseResult:
        """Execute the phase with timing, logging, and error handling.

        Args:
            context (RunContext): Active run context.

        Returns:
            PhaseResult: Completed phase result.
        """

        if not self.is_enabled(context):
            return PhaseResult(
                name=self.name,
                status="skipped",
                start_time=time.time(),
                end_time=time.time(),
                duration_s=0.0,
                metrics={},
                artifacts={},
                error=None,
            )

        start_time = time.time()
        try:
            outcome = self.execute(context)
            end_time = time.time()
            result = PhaseResult(
                name=self.name,
                status="success",
                start_time=start_time,
                end_time=end_time,
                duration_s=end_time - start_time,
                metrics=outcome.metrics,
                artifacts=outcome.artifacts,
                error=None,
            )
            if outcome.metrics:
                context.hook_manager.on_metrics(context, self.name, 0, outcome.metrics)
            return result
        except Exception as exc:
            end_time = time.time()
            error = PhaseError(type=type(exc).__name__, message=str(exc), details=None)
            context.logger.error(f"Phase '{self.name}' failed: {exc}")
            context.hook_manager.on_error(context, self.name, error)
            return PhaseResult(
                name=self.name,
                status="failed",
                start_time=start_time,
                end_time=end_time,
                duration_s=end_time - start_time,
                metrics={},
                artifacts={},
                error=error,
            )


class PhaseRunner:
    """Run phases in sequence with hooks and processors.

    Args:
        phases (list[Phase]): Pipeline phases to run.
        processors (list[Processor]): Pre/post processors.

    Examples:
        >>> PhaseRunner([], []).phases
        []
    """

    def __init__(self, phases: list[Phase], processors: list) -> None:
        """Initialize the phase runner.

        Args:
            phases (list[Phase]): Phase instances.
            processors (list): Processor instances.
        """

        self.phases = phases
        self.processors = processors

    def run(self, context: RunContext) -> list[PhaseResult]:
        """Execute all phases in order.

        Args:
            context (RunContext): Active run context.

        Returns:
            list[PhaseResult]: Phase results.
        """

        results: list[PhaseResult] = []
        context.run_results = results
        context.hook_manager.on_run_start(context)
        for phase in self.phases:
            context.hook_manager.on_phase_start(context, phase.name)
            placeholder = PhaseResult(
                name=phase.name,
                status="pending",
                start_time=0.0,
                end_time=0.0,
                duration_s=0.0,
                metrics={},
                artifacts={},
                error=None,
            )
            for processor in self.processors:
                if processor.should_run(context, phase.name, "before"):
                    processor.run(context, placeholder, "before")
            result = phase.run(context)
            results.append(result)
            for processor in self.processors:
                if processor.should_run(context, phase.name, "after"):
                    processor.run(context, result, "after")
            context.hook_manager.on_phase_end(context, result)
            if result.status == "failed" and not context.continue_on_error:
                break
        context.hook_manager.on_run_end(context, results)
        return results
