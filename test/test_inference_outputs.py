"""Inference output helper tests."""

from __future__ import annotations

import sys
from pathlib import Path

import fiona
import numpy as np
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.inference_utils import (  # noqa: E402
    append_prediction_shapefile,
    build_blend_weight_mask,
    extract_prediction_features,
    overlay_binary_mask,
)


def test_center_weight_mask_emphasizes_tile_center() -> None:
    """Center-weighted merge masks should favor central pixels.

    This guards the seam-reduction weighting used during scene assembly.
    """

    mask = build_blend_weight_mask(5, 5, mode="center_weighted")
    assert mask.shape == (5, 5)
    assert float(mask[2, 2]) > float(mask[0, 0])
    assert float(mask.min()) > 0.0


def test_overlay_binary_mask_tints_only_foreground() -> None:
    """Prediction overlay should tint foreground pixels and keep background.

    This keeps the unified inference figure readable while preserving context.
    """

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    overlay = overlay_binary_mask(rgb, mask, color=(120, 190, 255), alpha=0.5)
    assert overlay[0, 0].tolist() == [0, 0, 0]
    assert overlay[0, 1].tolist() == [60, 95, 127]


def test_prediction_shapefile_append_uses_epsg4326(tmp_path: Path) -> None:
    """Vector export should append features into one EPSG:4326 shapefile.

    The append path should preserve the cumulative dataset and CRS metadata.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """

    output_path = tmp_path / "predictions_4326.shp"
    transform = from_origin(12.0, 46.0, 0.01, 0.01)
    mask_a = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    mask_b = np.array([[0, 0], [0, 1]], dtype=np.uint8)

    features_a = extract_prediction_features(
        mask_a,
        transform,
        "EPSG:4326",
        source_id="scene_a",
        run_id="run_1",
    )
    features_b = extract_prediction_features(
        mask_b,
        transform,
        "EPSG:4326",
        source_id="scene_b",
        run_id="run_1",
    )

    append_prediction_shapefile(str(output_path), features_a, target_epsg=4326)
    append_prediction_shapefile(str(output_path), features_b, target_epsg=4326)

    with fiona.open(output_path) as src:
        records = list(src)
        assert len(records) == 2
        assert str(src.crs).lower().find("4326") != -1
        assert {record["properties"]["source_id"] for record in records} == {
            "scene_a",
            "scene_b",
        }
