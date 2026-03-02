"""Safety helper tests for training utilities."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.train_utils import (  # noqa: E402
    should_warn_high_logit,
    use_adamw_only_for_head,
)


def test_use_adamw_only_for_baseline_heads() -> None:
    """Ensure baseline lightweight heads route to AdamW-only optimization.

    This verifies the stability-oriented optimizer routing helper.

    Examples:
        >>> True
        True
    """

    assert use_adamw_only_for_head("dino_dense_probe")
    assert use_adamw_only_for_head("dino_segdino_light")
    assert not use_adamw_only_for_head("unet_lite")


def test_should_warn_high_logit_uses_batch_value() -> None:
    """Ensure high-logit warning helper is batch-local and finite-safe.

    The helper must ignore non-finite values and warn only on threshold breach.

    Examples:
        >>> True
        True
    """

    assert should_warn_high_logit(120.0, 80.0)
    assert not should_warn_high_logit(40.0, 80.0)
    assert not should_warn_high_logit(float("nan"), 80.0)
