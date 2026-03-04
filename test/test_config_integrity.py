"""Config integrity tests for shipped YAML profiles.

These tests enforce that the maintained config files stay synchronized in key
surface and remain loadable by the current training/model wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import available_heads, build_head  # noqa: E402
from pipeline.train_config import (  # noqa: E402
    parse_train_loss_config,
    parse_train_plot_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = (
    REPO_ROOT / "config.example.yml",
    REPO_ROOT / "config_hpc.yml",
    REPO_ROOT / "config_local.yml",
)


def _load_config(path: Path) -> dict[str, Any]:
    """Load one YAML config as a mapping.

    Args:
        path (Path): YAML file path.

    Returns:
        dict[str, Any]: Parsed YAML mapping.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "cfg.yml"
        ...     _ = p.write_text("a: 1\\n", encoding="utf-8")
        ...     _load_config(p)["a"]
        1
    """

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"{path} must contain a top-level mapping.")
    return data


def _flatten_keys(mapping: dict[str, Any], prefix: str = "") -> set[str]:
    """Flatten nested mapping keys into dotted paths.

    Args:
        mapping (dict[str, Any]): Source mapping.
        prefix (str): Prefix for recursive calls.

    Returns:
        set[str]: Flattened dotted key set.

    Examples:
        >>> sorted(_flatten_keys({"a": {"b": 1}, "c": 2}))
        ['a', 'a.b', 'c']
    """

    keys: set[str] = set()
    for key, value in mapping.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        keys.add(dotted)
        if isinstance(value, dict):
            keys |= _flatten_keys(value, dotted)
    return keys


def test_configs_parse_as_mappings() -> None:
    """Validate that shipped configs parse as dictionaries.

    Examples:
        >>> True
        True
    """

    for path in CONFIG_PATHS:
        cfg = _load_config(path)
        assert isinstance(cfg, dict), f"{path.name} did not parse as a dict."


def test_configs_key_surface_is_synchronized() -> None:
    """Ensure all shipped configs expose the same key surface.

    Examples:
        >>> True
        True
    """

    flattened = {path.name: _flatten_keys(_load_config(path)) for path in CONFIG_PATHS}
    reference_name = "config.example.yml"
    reference = flattened[reference_name]
    for name, keys in flattened.items():
        missing = sorted(reference - keys)
        extra = sorted(keys - reference)
        assert keys == reference, (
            f"{name} key mismatch vs {reference_name}; "
            f"missing={missing}, extra={extra}"
        )


def test_model_heads_are_registered_and_buildable() -> None:
    """Validate that each config selects a registered, buildable head.

    Examples:
        >>> True
        True
    """

    registry = available_heads()
    for path in CONFIG_PATHS:
        cfg = _load_config(path)
        model_cfg = cfg.get("model", {})
        assert isinstance(model_cfg, dict), f"{path.name} model section must be a dict."
        head_name = str(model_cfg.get("head", "")).strip()
        assert head_name in registry, (
            f"{path.name} uses unknown model.head={head_name!r}; "
            f"available={sorted(registry)}"
        )
        built = build_head(
            name=head_name,
            num_classes=int(model_cfg.get("num_classes", 2)),
            dino_channels=int(model_cfg.get("dino_channels", 1024)),
            model_cfg=model_cfg,
        )
        assert hasattr(built, "forward"), f"{path.name} head did not build correctly."


def test_train_parsers_accept_all_configs() -> None:
    """Validate train parser viability across shipped configs.

    Examples:
        >>> True
        True
    """

    for path in CONFIG_PATHS:
        cfg = _load_config(path)
        train_cfg = cfg.get("train", {})
        dataset_cfg = cfg.get("dataset", {})
        assert isinstance(train_cfg, dict), f"{path.name} train section must be a dict."
        assert isinstance(
            dataset_cfg, dict
        ), f"{path.name} dataset section must be a dict."
        validation_cfg = dataset_cfg.get("validation", {})
        ignore_index = None
        if isinstance(validation_cfg, dict):
            ignore_index = validation_cfg.get("ignore_index")
        resolved_loss = parse_train_loss_config(
            train_cfg, dataset_ignore_index=ignore_index
        )
        resolved_plot = parse_train_plot_config(train_cfg)
        assert resolved_loss.dice_weight >= 0.0
        assert resolved_loss.ce_weight >= 0.0
        assert resolved_plot.pairs >= 1


def test_core_model_values_are_viable() -> None:
    """Validate core model value sanity for shipped configs.

    Examples:
        >>> True
        True
    """

    for path in CONFIG_PATHS:
        cfg = _load_config(path)
        model_cfg = cfg.get("model", {})
        assert isinstance(model_cfg, dict), f"{path.name} model section must be a dict."
        layers = model_cfg.get("layers", [])
        assert isinstance(layers, list), f"{path.name} model.layers must be a list."
        assert layers, f"{path.name} model.layers cannot be empty."
        assert all(
            isinstance(x, int) for x in layers
        ), f"{path.name} model.layers must contain only integers."
        num_classes = int(model_cfg.get("num_classes", 0))
        dino_channels = int(model_cfg.get("dino_channels", 0))
        assert num_classes >= 2, f"{path.name} model.num_classes must be >= 2."
        assert dino_channels > 0, f"{path.name} model.dino_channels must be > 0."
