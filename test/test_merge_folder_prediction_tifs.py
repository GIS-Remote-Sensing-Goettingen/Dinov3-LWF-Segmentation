"""Folder-level prediction merge script tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_merge_module():
    """Load the folder-merge script module from disk.

    Returns:
        object: Imported module object.

    Examples:
        >>> callable(_load_merge_module)
        True
    """

    module_path = REPO_ROOT / "scripts" / "merge_folder_prediction_tifs.py"
    spec = importlib.util.spec_from_file_location(
        "merge_folder_prediction_tifs", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_test_geotiff(path: Path, data: np.ndarray, *, transform) -> None:
    """Write one single-band GeoTIFF fixture.

    Args:
        path (Path): Output TIFF path.
        data (np.ndarray): Raster data as `(H, W)`.
        transform: Raster transform.

    Examples:
        >>> callable(_write_test_geotiff)
        True
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=int(data.shape[0]),
        width=int(data.shape[1]),
        count=1,
        dtype=str(data.dtype),
        crs="EPSG:25832",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data[np.newaxis, ...])


def test_merge_folder_prediction_tifs_merges_default_folder_outputs(
    tmp_path: Path,
) -> None:
    """Folder merge script should resolve and merge folder-level TIFFs.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    module = _load_merge_module()
    batches_root = tmp_path / "batches"
    for idx, folder_name in enumerate(["folder1_infer", "folder2_infer"]):
        _write_test_geotiff(
            batches_root / folder_name / "merged" / "predictions.tif",
            np.full((1, 2), idx + 1, dtype=np.uint8),
            transform=from_origin(float(idx * 2), 1.0, 1.0, 1.0),
        )

    resolved = module.resolve_folder_prediction_tifs(
        batches_root=batches_root,
        folder_names=["folder1_infer", "folder2_infer"],
    )
    assert len(resolved) == 2

    output_tif = tmp_path / "all_merged" / "predictions.tif"
    merged_path = module.merge_folder_prediction_tifs(
        batches_root=batches_root,
        folder_names=["folder1_infer", "folder2_infer"],
        output_tif=output_tif,
    )

    with rasterio.open(merged_path) as src:
        data = src.read(1)
        assert src.width == 4
        assert src.height == 1
    assert data.tolist() == [[1, 1, 2, 2]]
