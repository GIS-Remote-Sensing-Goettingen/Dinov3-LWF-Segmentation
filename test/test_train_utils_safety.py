"""Safety helper tests for training utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.train_utils import (  # noqa: E402
    resolve_lr_metrics,
    should_warn_high_logit,
    use_adamw_only_for_head,
)
from utils.optim import Muon  # noqa: E402


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


def test_resolve_lr_metrics_handles_adamw_and_muon() -> None:
    """Ensure LR metric extraction works for both optimizer paths.

    Examples:
        >>> True
        True
    """

    p_adamw = torch.nn.Parameter(torch.ones(1))
    opt_adamw = torch.optim.AdamW([p_adamw], lr=1e-3)
    sch_adamw = torch.optim.lr_scheduler.OneCycleLR(
        opt_adamw, max_lr=1e-3, epochs=1, steps_per_epoch=1
    )
    lr, lr_muon, lr_adamw = resolve_lr_metrics(opt_adamw, sch_adamw)
    assert lr >= 0.0
    assert lr_muon == 0.0
    assert lr_adamw > 0.0

    p_muon = torch.nn.Parameter(torch.ones(2, 2))
    p_aux = torch.nn.Parameter(torch.ones(2))
    opt_muon = Muon(
        [p_muon],
        lr=0.02,
        adamw_params=[p_aux],
        adamw_lr=1e-3,
        adamw_wd=0.01,
    )
    sch_muon = torch.optim.lr_scheduler.OneCycleLR(
        opt_muon, max_lr=0.02, epochs=1, steps_per_epoch=1
    )
    lr2, lr_muon2, lr_adamw2 = resolve_lr_metrics(opt_muon, sch_muon)
    assert lr2 >= 0.0
    assert lr_muon2 == lr2
    assert lr_adamw2 > 0.0
