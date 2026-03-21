"""Batch inference orchestrator tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_launch_module():
    """Load the batch launcher module from disk.

    Returns:
        object: Imported module object.

    Examples:
        >>> callable(_load_launch_module)
        True
    """

    module_path = REPO_ROOT / "scripts" / "launch_batched_inference.py"
    spec = importlib.util.spec_from_file_location(
        "launch_batched_inference", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_test_geotiff(
    path: Path,
    data: np.ndarray,
    *,
    transform,
    crs: str = "EPSG:25832",
) -> None:
    """Write one small GeoTIFF fixture.

    Args:
        path (Path): Output TIFF path.
        data (np.ndarray): Raster data, either `(H, W)` or `(H, W, C)`.
        transform: Raster transform.
        crs (str): CRS string.

    Examples:
        >>> callable(_write_test_geotiff)
        True
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        write_data = data[np.newaxis, ...]
        count = 1
    else:
        write_data = np.transpose(data, (2, 0, 1))
        count = int(data.shape[2])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=int(data.shape[0]),
        width=int(data.shape[1]),
        count=count,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(write_data)


def _write_template_config(path: Path, input_dir: Path) -> dict:
    """Write one minimal HPC-style config used by launcher tests.

    Args:
        path (Path): Output config path.
        input_dir (Path): Directory scanned for inference images.

    Returns:
        dict: Written config payload.

    Examples:
        >>> callable(_write_template_config)
        True
    """

    payload = {
        "resources": {"distributed": True},
        "logging": {
            "level": "info",
            "file": str(path.parent / "run.log"),
            "per_run": True,
        },
        "paths": {"label_path": str(path.parent / "label_template.tif")},
        "prepare": {"enable": False},
        "verify": {"enable": False},
        "train": {"enable": False},
        "inference": {
            "enable": True,
            "device": "cpu",
            "input_tif": "",
            "input_dir": str(input_dir),
            "input_paths_file": "",
            "output_tif": "",
            "output_dir": "",
            "glob": "*.tif",
            "checkpoint": str(path.parent / "checkpoint.pth"),
            "tile_size": 960,
            "overlap": 0.25,
            "merge": {"mode": "center_weighted"},
            "tta": {"horizontal_flip": False, "vertical_flip": False},
            "explain": {"enable": False, "output_dir": "plots"},
            "vector": {"enable": False},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def test_launch_batched_inference_dry_run_writes_configs_and_scripts(
    tmp_path: Path,
) -> None:
    """Dry-run launch should write batch-local configs and Slurm scripts.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    module = _load_launch_module()
    image_dir = tmp_path / "images"
    for idx in range(3):
        (image_dir / f"scene_{idx:03d}.tif").parent.mkdir(parents=True, exist_ok=True)
        (image_dir / f"scene_{idx:03d}.tif").touch()
    template_config = tmp_path / "configs" / "config_hpc.yml"
    original_config = _write_template_config(template_config, image_dir)

    orchestration_root = module.launch_batched_inference(
        job_name="demo",
        batch_size=2,
        template_config_path=template_config,
        template_slurm_path=REPO_ROOT / "segmentation.sh",
        output_root=tmp_path / "output",
        max_retries=3,
        dry_run=True,
    )

    generated_cfg = yaml.safe_load(
        (orchestration_root / "configs" / "batch_000.yml").read_text(encoding="utf-8")
    )
    assert (
        yaml.safe_load(template_config.read_text(encoding="utf-8")) == original_config
    )
    assert generated_cfg["tracking"]["mlflow"]["enable"] is False
    assert generated_cfg["resources"]["distributed"] is False
    assert generated_cfg["logging"]["per_run"] is False
    assert generated_cfg["inference"]["input_dir"] == ""
    assert generated_cfg["inference"]["input_tif"] == ""
    assert generated_cfg["inference"]["input_paths_file"].endswith(
        "images_batch_000.txt"
    )
    assert generated_cfg["inference"]["output_tif"].endswith(
        "runs/batch_000/predictions.tif"
    )
    assert (orchestration_root / "slurm" / "worker_batch_000.sh").exists()
    assert (orchestration_root / "slurm" / "controller.sh").exists()
    worker_script = (orchestration_root / "slurm" / "worker_batch_000.sh").read_text(
        encoding="utf-8"
    )
    assert 'cd "${WORK_DIR}"' in worker_script
    assert 'python -u "${REPO_ROOT}/main.py" "${CONFIG_PATH}"' in worker_script
    status = module._load_json(orchestration_root / "status.json")
    assert len(status["batches"]) == 2
    assert status["worker_job_ids"] == [None, None]
    assert status["controller_job_ids"] == [None]


def test_merge_batch_prediction_tifs_uses_batch_order_overwrite(
    tmp_path: Path,
) -> None:
    """Merge should overwrite earlier batch pixels with later batch order.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    module = _load_launch_module()
    batch_a = tmp_path / "batch_a.tif"
    batch_b = tmp_path / "batch_b.tif"
    output_tif = tmp_path / "merged" / "predictions.tif"

    _write_test_geotiff(
        batch_a,
        np.array([[1, 1], [1, 1]], dtype=np.uint8),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
    )
    _write_test_geotiff(
        batch_b,
        np.array([[2, 2], [2, 2]], dtype=np.uint8),
        transform=from_origin(1.0, 2.0, 1.0, 1.0),
    )

    merged_path = module.merge_batch_prediction_tifs(
        batch_tifs=[str(batch_a), str(batch_b)],
        output_tif=str(output_tif),
    )

    with rasterio.open(merged_path) as src:
        data = src.read(1)
        assert src.width == 3
        assert src.height == 2
    assert data.tolist() == [[1, 2, 2], [1, 2, 2]]


def test_run_controller_merges_completed_batch_outputs(
    tmp_path: Path,
) -> None:
    """Controller should merge completed batch TIFFs into one final output.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    module = _load_launch_module()
    image_dir = tmp_path / "images"
    for idx in range(2):
        (image_dir / f"scene_{idx:03d}.tif").parent.mkdir(parents=True, exist_ok=True)
        (image_dir / f"scene_{idx:03d}.tif").touch()
    template_config = tmp_path / "configs" / "config_hpc.yml"
    _write_template_config(template_config, image_dir)

    orchestration_root = module.launch_batched_inference(
        job_name="controller_demo",
        batch_size=1,
        template_config_path=template_config,
        template_slurm_path=REPO_ROOT / "segmentation.sh",
        output_root=tmp_path / "output",
        max_retries=1,
        dry_run=True,
    )

    batch_0_output = orchestration_root / "runs" / "batch_000" / "predictions.tif"
    batch_1_output = orchestration_root / "runs" / "batch_001" / "predictions.tif"
    _write_test_geotiff(
        batch_0_output,
        np.array([[1, 1]], dtype=np.uint8),
        transform=from_origin(0.0, 1.0, 1.0, 1.0),
    )
    _write_test_geotiff(
        batch_1_output,
        np.array([[2, 2]], dtype=np.uint8),
        transform=from_origin(2.0, 1.0, 1.0, 1.0),
    )
    (orchestration_root / "runs" / "batch_000" / "batch_complete.json").write_text(
        '{"status":"complete"}\n',
        encoding="utf-8",
    )
    (orchestration_root / "runs" / "batch_001" / "batch_complete.json").write_text(
        '{"status":"complete"}\n',
        encoding="utf-8",
    )

    module.run_controller(orchestration_root=orchestration_root, dry_run=True)

    merged_output = orchestration_root / "merged" / "predictions.tif"
    with rasterio.open(merged_output) as src:
        data = src.read(1)
        assert src.width == 4
        assert src.height == 1
    assert data.tolist() == [[1, 1, 2, 2]]
    final_status = module._load_json(orchestration_root / "final_status.json")
    assert final_status["status"] == "success"
