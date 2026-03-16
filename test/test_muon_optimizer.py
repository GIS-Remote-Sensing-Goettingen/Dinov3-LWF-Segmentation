"""Regression tests for the Muon optimizer."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.train_utils import split_params_for_muon  # noqa: E402
from pipeline.utils import collect_run_params  # noqa: E402
from utils import optim as optim_module  # noqa: E402
from utils.optim import Muon  # noqa: E402


def test_muon_reuses_adamw_weight_decay_by_default() -> None:
    """Muon should default to the AdamW decay value when unset.

    This pins the fallback behavior for configs that only define `adamw_wd`.

    Examples:
        >>> True
        True
    """

    param = torch.nn.Parameter(torch.eye(2))
    param.grad = torch.zeros_like(param)
    opt = Muon([param], lr=0.1, momentum=0.0, nesterov=False, adamw_wd=0.25)

    opt.step()

    expected = torch.eye(2) * (1 - 0.1 * 0.25)
    assert torch.allclose(param.data, expected)


def test_muon_shape_adjustment_scales_update_magnitude() -> None:
    """Shape-aware scaling should multiply updates by sqrt(max(rows, cols)).

    The paper-inspired scale factor should change magnitude without changing direction.

    Examples:
        >>> True
        True
    """

    grad = torch.tensor([[1.0, 2.0, -1.0, 0.5], [0.25, -0.5, 3.0, 1.5]])
    base = torch.nn.Parameter(torch.zeros_like(grad))
    shaped = torch.nn.Parameter(torch.zeros_like(grad))
    base.grad = grad.clone()
    shaped.grad = grad.clone()

    opt_base = Muon(
        [base],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        muon_wd=0.0,
        muon_update_scale=1.0,
        muon_adjust_lr_for_shape=False,
        ns_steps=5,
    )
    opt_shaped = Muon(
        [shaped],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        muon_wd=0.0,
        muon_update_scale=1.0,
        muon_adjust_lr_for_shape=True,
        ns_steps=5,
    )

    opt_base.step()
    opt_shaped.step()

    base_delta = (-base.data).norm().item()
    shaped_delta = (-shaped.data).norm().item()

    assert base_delta > 0.0
    assert math.isclose(
        shaped_delta / base_delta,
        math.sqrt(max(grad.shape)),
        rel_tol=1e-4,
        abs_tol=1e-4,
    )


def test_muon_skips_nonfinite_candidate_without_mutating_parameter() -> None:
    """A non-finite candidate parameter should not be committed.

    This ensures failed Muon steps do not silently corrupt model weights.

    Examples:
        >>> True
        True
    """

    param = torch.nn.Parameter(
        torch.full((2, 2), torch.finfo(torch.float32).max / 4, dtype=torch.float32)
    )
    before = param.detach().clone()
    param.grad = torch.zeros_like(param)
    opt = Muon(
        [param],
        lr=2.0,
        momentum=0.0,
        nesterov=False,
        muon_wd=1e31,
        muon_update_scale=0.0,
    )

    opt.step()

    assert torch.equal(param.data, before)
    assert opt.last_step_stats["muon_params_skipped"] == 1
    assert opt.last_step_stats["muon_params_updated"] == 0


def test_muon_skips_nonfinite_orthogonalized_update() -> None:
    """A non-finite orthogonalized update should leave the parameter unchanged.

    Muon should count the step as skipped and keep the parameter intact.

    Examples:
        >>> True
        True
    """

    param = torch.nn.Parameter(torch.ones(2, 2))
    before = param.detach().clone()
    param.grad = torch.ones_like(param)
    opt = Muon([param], lr=0.1, muon_wd=0.0)
    original = optim_module.zeropower_via_newtonschulz5

    def _broken(*args, **kwargs):
        """Return a deliberately invalid Muon update for safety testing.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.
        """

        return torch.full((2, 2), float("inf"))

    optim_module.zeropower_via_newtonschulz5 = _broken
    try:
        opt.step()
    finally:
        optim_module.zeropower_via_newtonschulz5 = original

    assert torch.equal(param.data, before)
    assert opt.last_step_stats["muon_params_skipped"] == 1
    assert opt.last_step_stats["muon_params_updated"] == 0


def test_split_params_for_muon_keeps_embeddings_on_adamw() -> None:
    """Embeddings should stay on the AdamW side even though they are 2D.

    This guards the module-aware parameter split used by the Muon path.

    Examples:
        >>> True
        True
    """

    class TinyModule(torch.nn.Module):
        """Small module with both embedding and linear weights."""

        def __init__(self) -> None:
            """Build the test module layers.

            The split helper should route the two parameter types differently.
            """

            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.linear = torch.nn.Linear(4, 4)

    module = TinyModule()
    muon_params, adamw_params = split_params_for_muon(module)

    muon_ids = {id(param) for param in muon_params}
    adamw_ids = {id(param) for param in adamw_params}

    assert id(module.linear.weight) in muon_ids
    assert id(module.linear.bias) in adamw_ids
    assert id(module.embedding.weight) in adamw_ids


def test_collect_run_params_includes_muon_scaling_settings() -> None:
    """Run-parameter snapshots should expose Muon scaling configuration.

    Tracking should record the effective Muon defaults used by a run.

    Examples:
        >>> True
        True
    """

    params = collect_run_params(
        {
            "train": {
                "muon_lr": 0.02,
                "adamw_wd": 0.05,
                "muon_update_scale": 0.2,
                "muon_adjust_lr_for_shape": True,
            }
        }
    )

    assert params["train.muon_lr"] == "0.02"
    assert params["train.muon_wd"] == "0.05"
    assert params["train.muon_update_scale"] == "0.2"
    assert params["train.muon_adjust_lr_for_shape"] == "True"
