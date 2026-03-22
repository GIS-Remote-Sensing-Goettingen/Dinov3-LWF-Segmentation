"""Launch and manage one Slurm worker per fixed-size inference batch.

Examples:
    >>> isinstance(REPO_ROOT.name, str)
    True
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window, from_bounds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.inference_utils import (  # noqa: E402
    ensure_cumulative_prediction_raster,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "batches"
DEFAULT_TEMPLATE_CONFIG = REPO_ROOT / "configs" / "config_hpc.yml"
DEFAULT_TEMPLATE_SLURM = REPO_ROOT / "segmentation.sh"
SBATCH_ID_RE = re.compile(r"Submitted batch job (\d+)")
logger = logging.getLogger(__name__)


def _resolve_relative_path(path: str | Path, *, base_dir: Path) -> Path:
    """Resolve one path relative to a chosen base directory.

    Args:
        path (str | Path): Input path string or object.
        base_dir (Path): Directory used for relative resolution.

    Returns:
        Path: Absolute resolved path.

    Examples:
        >>> resolved = _resolve_relative_path("cfg.yml", base_dir=Path("/tmp"))
        >>> str(resolved)
        '/tmp/cfg.yml'
    """

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML file as a mapping.

    Args:
        path (Path): YAML file path.

    Returns:
        dict[str, Any]: Parsed mapping.

    Examples:
        >>> callable(_load_yaml)
        True
    """

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must define a mapping: {path}")
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write one YAML mapping to disk.

    Args:
        path (Path): Output YAML path.
        payload (dict[str, Any]): Mapping payload.

    Examples:
        >>> callable(_write_yaml)
        True
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load one JSON file or return a default mapping.

    Args:
        path (Path): JSON file path.
        default (dict[str, Any] | None): Optional fallback mapping.

    Returns:
        dict[str, Any]: Parsed or fallback mapping.

    Examples:
        >>> _load_json(Path('/tmp/missing-batch.json'), {'ok': True})['ok']
        True
    """

    if not path.exists():
        return {} if default is None else dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file with stable formatting.

    Args:
        path (Path): Output JSON path.
        payload (dict[str, Any]): Mapping payload.

    Examples:
        >>> callable(_write_json)
        True
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _chunk_paths(paths: list[str], batch_size: int) -> list[list[str]]:
    """Split paths into contiguous fixed-size batches.

    Args:
        paths (list[str]): Ordered path list.
        batch_size (int): Maximum paths per batch.

    Returns:
        list[list[str]]: Contiguous path chunks.

    Examples:
        >>> _chunk_paths(['a', 'b', 'c'], 2)
        [['a', 'b'], ['c']]
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [paths[idx : idx + batch_size] for idx in range(0, len(paths), batch_size)]


def resolve_inference_paths_from_config(template_config_path: Path) -> list[str]:
    """Resolve directory-inference image paths from one template config.

    Args:
        template_config_path (Path): Base YAML config path.

    Returns:
        list[str]: Sorted absolute input raster paths.

    Examples:
        >>> callable(resolve_inference_paths_from_config)
        True
    """

    cfg = _load_yaml(template_config_path)
    infer_cfg = cfg.get("inference", {})
    input_dir_raw = str(infer_cfg.get("input_dir", "") or "").strip()
    if not input_dir_raw:
        raise ValueError(
            f"{template_config_path} must define inference.input_dir for batch launch."
        )
    input_dir = _resolve_relative_path(
        input_dir_raw,
        base_dir=template_config_path.parent,
    )
    glob_pattern = str(infer_cfg.get("glob", "*.tif") or "*.tif")
    matches = sorted(str(path.resolve()) for path in input_dir.glob(glob_pattern))
    if not matches:
        raise ValueError(
            f"No inference inputs found under {input_dir} with glob {glob_pattern!r}."
        )
    return matches


def build_inference_batches(
    *,
    template_config_path: Path,
    batch_size: int,
    output_root: Path,
    job_name: str,
) -> tuple[Path, list[Path]]:
    """Resolve input images once and write deterministic batch manifest files.

    Args:
        template_config_path (Path): Base YAML config path.
        batch_size (int): Maximum images per batch.
        output_root (Path): Parent directory for orchestration outputs.
        job_name (str): Job/orchestration name.

    Returns:
        tuple[Path, list[Path]]: Orchestration root and manifest paths.

    Examples:
        >>> callable(build_inference_batches)
        True
    """

    orchestration_root = output_root / job_name
    if (orchestration_root / "manifest.json").exists():
        raise ValueError(
            f"orchestration root already exists and looks active: {orchestration_root}"
        )
    paths = resolve_inference_paths_from_config(template_config_path)
    batches = _chunk_paths(paths, batch_size)
    manifest_paths: list[Path] = []
    batches_dir = orchestration_root / "batches"
    for idx, batch_paths in enumerate(batches):
        manifest_path = batches_dir / f"images_batch_{idx:03d}.txt"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            "\n".join(batch_paths) + ("\n" if batch_paths else ""),
            encoding="utf-8",
        )
        manifest_paths.append(manifest_path)
    _write_json(
        orchestration_root / "manifest.json",
        {
            "job_name": job_name,
            "batch_size": batch_size,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "template_config": str(template_config_path),
            "template_slurm": str(DEFAULT_TEMPLATE_SLURM),
            "total_images": len(paths),
            "num_batches": len(manifest_paths),
            "batch_manifests": [str(path) for path in manifest_paths],
            "input_paths": paths,
        },
    )
    return orchestration_root, manifest_paths


def _batch_run_dir(orchestration_root: Path, batch_idx: int) -> Path:
    """Return the batch-local run directory.

    Args:
        orchestration_root (Path): Root orchestration directory.
        batch_idx (int): Zero-based batch index.

    Returns:
        Path: Batch run directory.

    Examples:
        >>> _batch_run_dir(Path('/tmp/root'), 2).name
        'batch_002'
    """

    return orchestration_root / "runs" / f"batch_{batch_idx:03d}"


def _batch_output_tif(orchestration_root: Path, batch_idx: int) -> Path:
    """Return the batch-local cumulative prediction TIFF path.

    Args:
        orchestration_root (Path): Root orchestration directory.
        batch_idx (int): Zero-based batch index.

    Returns:
        Path: Batch output TIFF path.

    Examples:
        >>> _batch_output_tif(Path('/tmp/root'), 1).name
        'predictions.tif'
    """

    return _batch_run_dir(orchestration_root, batch_idx) / "predictions.tif"


def _batch_complete_marker(orchestration_root: Path, batch_idx: int) -> Path:
    """Return the completion marker path for one batch.

    Args:
        orchestration_root (Path): Root orchestration directory.
        batch_idx (int): Zero-based batch index.

    Returns:
        Path: Batch completion marker path.

    Examples:
        >>> _batch_complete_marker(Path('/tmp/root'), 1).name
        'batch_complete.json'
    """

    return _batch_run_dir(orchestration_root, batch_idx) / "batch_complete.json"


def _write_batch_configs(
    *,
    orchestration_root: Path,
    template_config_path: Path,
    batch_manifests: list[Path],
) -> list[Path]:
    """Write one derived YAML config per batch without touching the template.

    Args:
        orchestration_root (Path): Root orchestration directory.
        template_config_path (Path): Base YAML config path.
        batch_manifests (list[Path]): Per-batch image manifest files.

    Returns:
        list[Path]: Generated config paths.

    Examples:
        >>> callable(_write_batch_configs)
        True
    """

    base_cfg = _load_yaml(template_config_path)
    config_paths: list[Path] = []
    for idx, manifest_path in enumerate(batch_manifests):
        batch_cfg = copy.deepcopy(base_cfg)
        batch_run_dir = _batch_run_dir(orchestration_root, idx)
        inference_cfg = batch_cfg.setdefault("inference", {})
        explain_cfg = inference_cfg.setdefault("explain", {})
        batch_cfg.setdefault("prepare", {})["enable"] = False
        batch_cfg.setdefault("verify", {})["enable"] = False
        batch_cfg.setdefault("train", {})["enable"] = False
        inference_cfg["enable"] = True
        inference_cfg["input_tif"] = ""
        inference_cfg["input_dir"] = ""
        inference_cfg["input_paths_file"] = str(manifest_path)
        inference_cfg["output_tif"] = str(_batch_output_tif(orchestration_root, idx))
        inference_cfg["output_dir"] = str(batch_run_dir)
        if bool(explain_cfg.get("enable", False)):
            explain_cfg["output_dir"] = str(batch_run_dir / "plots")
        logging_cfg = batch_cfg.setdefault("logging", {})
        logging_cfg["file"] = str(batch_run_dir / "run.log")
        logging_cfg["per_run"] = False
        batch_cfg.setdefault("resources", {})["distributed"] = False
        batch_cfg.setdefault("tracking", {}).setdefault("mlflow", {})["enable"] = False
        config_path = orchestration_root / "configs" / f"batch_{idx:03d}.yml"
        _write_yaml(config_path, batch_cfg)
        config_paths.append(config_path)
    return config_paths


def _strip_sbatch_overrides(sbatch_lines: list[str]) -> list[str]:
    """Drop template scheduler directives that the renderer owns explicitly.

    Args:
        sbatch_lines (list[str]): Template `#SBATCH` lines.

    Returns:
        list[str]: Preserved scheduler lines.

    Examples:
        >>> _strip_sbatch_overrides(['#SBATCH --job-name=x', '#SBATCH --mem=1G'])
        ['#SBATCH --mem=1G']
    """

    override_prefixes = (
        "#SBATCH --job-name=",
        "#SBATCH --output=",
        "#SBATCH --error=",
        "#SBATCH --array=",
    )
    return [
        line
        for line in sbatch_lines
        if not any(line.startswith(prefix) for prefix in override_prefixes)
    ]


def _repo_root_setup_lines() -> list[str]:
    """Return the shell stanza that validates and enters the repo root.

    Returns:
        list[str]: Shell lines.

    Examples:
        >>> any('REPO_ROOT=' in line for line in _repo_root_setup_lines())
        True
    """

    return [
        f"REPO_ROOT={shlex.quote(str(REPO_ROOT))}",
        'if [ ! -f "${REPO_ROOT}/main.py" ]; then',
        '  echo "missing repo root main.py: ${REPO_ROOT}" >&2',
        "  exit 1",
        "fi",
        'cd "${REPO_ROOT}"',
    ]


def _render_slurm_script(
    *,
    template_path: Path,
    job_name: str,
    stdout_path: Path,
    stderr_path: Path,
    command: str,
) -> str:
    """Render one derived Slurm script from a read-only template.

    Args:
        template_path (Path): Source Slurm template.
        job_name (str): Rendered job name.
        stdout_path (Path): Rendered stdout path.
        stderr_path (Path): Rendered stderr path.
        command (str): Custom command block appended after template setup.

    Returns:
        str: Executable Slurm script content.

    Examples:
        >>> callable(_render_slurm_script)
        True
    """

    lines = template_path.read_text(encoding="utf-8").splitlines()
    shebang = lines[0] if lines and lines[0].startswith("#!") else "#!/usr/bin/env bash"
    sbatch_lines: list[str] = []
    body_start = 1
    for idx, line in enumerate(lines[1:], start=1):
        if line.startswith("#SBATCH"):
            sbatch_lines.append(line)
            body_start = idx + 1
            continue
        if not line.strip():
            body_start = idx + 1
            continue
        body_start = idx
        break
    body_lines = lines[body_start:]
    cut_idx = len(body_lines)
    for idx, line in enumerate(body_lines):
        stripped = line.strip()
        if stripped.startswith("detect_allocated_gpus()") or stripped.startswith(
            "CONFIG_PATH="
        ):
            cut_idx = idx
            break
    preserved_body = "\n".join(body_lines[:cut_idx]).rstrip()
    body_sections = [
        preserved_body,
        "\n".join(_repo_root_setup_lines()),
        command.strip(),
    ]
    rendered_body = "\n\n".join(
        section for section in body_sections if section
    ).rstrip()
    script_lines = [
        shebang,
        *_strip_sbatch_overrides(sbatch_lines),
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={stdout_path}",
        f"#SBATCH --error={stderr_path}",
        "",
        rendered_body,
        "",
    ]
    return "\n".join(script_lines)


def _write_executable(path: Path, content: str) -> Path:
    """Write one executable helper script.

    Args:
        path (Path): Output script path.
        content (str): Script contents.

    Returns:
        Path: Written script path.

    Examples:
        >>> callable(_write_executable)
        True
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _worker_script_command(
    *,
    config_path: Path,
    run_dir: Path,
    output_tif: Path,
    complete_marker: Path,
) -> str:
    """Build one worker command block for a batch.

    Args:
        config_path (Path): Batch config path.
        run_dir (Path): Batch run directory.
        output_tif (Path): Expected batch prediction TIFF.
        complete_marker (Path): Batch completion marker path.

    Returns:
        str: Shell command block.

    Examples:
        >>> '--config' not in _worker_script_command(
        ...     config_path=Path('/tmp/cfg.yml'),
        ...     run_dir=Path('/tmp/run'),
        ...     output_tif=Path('/tmp/run/predictions.tif'),
        ...     complete_marker=Path('/tmp/run/batch_complete.json'),
        ... )
        True
    """

    return "\n".join(
        [
            f"WORK_DIR={shlex.quote(str(run_dir))}",
            f"CONFIG_PATH={shlex.quote(str(config_path))}",
            f"OUTPUT_TIF={shlex.quote(str(output_tif))}",
            f"COMPLETE_MARKER={shlex.quote(str(complete_marker))}",
            'mkdir -p "${WORK_DIR}"',
            'rm -f "${COMPLETE_MARKER}"',
            'echo "repo_root=${REPO_ROOT}"',
            'echo "work_dir=${WORK_DIR}"',
            'echo "config_path=${CONFIG_PATH}"',
            'echo "output_tif=${OUTPUT_TIF}"',
            "python --version",
            "nvidia-smi || true",
            'cd "${WORK_DIR}"',
            'python -u "${REPO_ROOT}/main.py" "${CONFIG_PATH}"',
            'if [ ! -f "${OUTPUT_TIF}" ]; then',
            '  echo "missing batch prediction output: ${OUTPUT_TIF}" >&2',
            "  exit 1",
            "fi",
            'printf \'{"status":"complete","output_tif":"%s"}\\n\' "${OUTPUT_TIF}" > "${COMPLETE_MARKER}"',
        ]
    )


def _controller_command(orchestration_root: Path) -> str:
    """Build one self-referential controller command block.

    Args:
        orchestration_root (Path): Root orchestration directory.

    Returns:
        str: Shell command block.

    Examples:
        >>> '--controller' in _controller_command(Path('/tmp/root'))
        True
    """

    return (
        f'python -u "{REPO_ROOT / "scripts" / "launch_batched_inference.py"}" '
        f'--controller --orchestration-root "{orchestration_root}"'
    )


def _write_slurm_scripts(
    *,
    orchestration_root: Path,
    template_slurm_path: Path,
    job_name: str,
    config_paths: list[Path],
) -> tuple[list[Path], Path]:
    """Render worker and controller Slurm scripts.

    Args:
        orchestration_root (Path): Root orchestration directory.
        template_slurm_path (Path): Slurm template path.
        job_name (str): Orchestration job name.
        config_paths (list[Path]): Batch config paths.

    Returns:
        tuple[list[Path], Path]: Worker scripts and controller script.

    Examples:
        >>> callable(_write_slurm_scripts)
        True
    """

    slurm_dir = orchestration_root / "slurm"
    worker_scripts: list[Path] = []
    for idx, config_path in enumerate(config_paths):
        worker_scripts.append(
            _write_executable(
                slurm_dir / f"worker_batch_{idx:03d}.sh",
                _render_slurm_script(
                    template_path=template_slurm_path,
                    job_name=f"{job_name}_batch_{idx:03d}",
                    stdout_path=slurm_dir / f"worker_batch_{idx:03d}_%j.out",
                    stderr_path=slurm_dir / f"worker_batch_{idx:03d}_%j.err",
                    command=_worker_script_command(
                        config_path=config_path,
                        run_dir=_batch_run_dir(orchestration_root, idx),
                        output_tif=_batch_output_tif(orchestration_root, idx),
                        complete_marker=_batch_complete_marker(orchestration_root, idx),
                    ),
                ),
            )
        )
    controller_script = _write_executable(
        slurm_dir / "controller.sh",
        _render_slurm_script(
            template_path=template_slurm_path,
            job_name=f"{job_name}_controller",
            stdout_path=slurm_dir / "controller_%j.out",
            stderr_path=slurm_dir / "controller_%j.err",
            command=_controller_command(orchestration_root),
        ),
    )
    return worker_scripts, controller_script


def _submit_sbatch(
    *,
    script_path: Path,
    extra_args: list[str] | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Submit one `sbatch` command or return a dry-run record.

    Args:
        script_path (Path): Slurm script path.
        extra_args (list[str] | None): Optional additional sbatch args.
        dry_run (bool): When true, skip the real submission.

    Returns:
        dict[str, Any]: Submission metadata.

    Examples:
        >>> _submit_sbatch(
        ...     script_path=Path('/tmp/demo.sh'),
        ...     extra_args=['--dependency', 'afterany:1'],
        ...     dry_run=True,
        ... )['command'][-1]
        '/tmp/demo.sh'
    """

    command = ["sbatch", *(extra_args or []), str(script_path)]
    if dry_run:
        return {"job_id": None, "command": command}
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    match = SBATCH_ID_RE.search(proc.stdout.strip())
    if not match:
        raise RuntimeError(f"unable to parse sbatch output: {proc.stdout!r}")
    return {"job_id": match.group(1), "command": command, "stdout": proc.stdout.strip()}


def _dependency_args(job_ids: list[str | None]) -> list[str]:
    """Build one `afterany` dependency argument for controller submissions.

    Args:
        job_ids (list[str | None]): Worker job ids.

    Returns:
        list[str]: Additional `sbatch` arguments.

    Examples:
        >>> _dependency_args(['1', '2'])
        ['--dependency', 'afterany:1:2']
    """

    valid_ids = [str(job_id) for job_id in job_ids if job_id not in (None, "")]
    if not valid_ids:
        return []
    return ["--dependency", f"afterany:{':'.join(valid_ids)}"]


def _is_batch_complete(batch: dict[str, Any]) -> bool:
    """Return whether one batch finished successfully.

    Args:
        batch (dict[str, Any]): Batch status entry.

    Returns:
        bool: True when the completion marker and TIFF are both valid.

    Examples:
        >>> _is_batch_complete({'complete_marker': '/tmp/missing', 'output_tif': '/tmp/x.tif'})
        False
    """

    marker_path = Path(batch["complete_marker"])
    output_tif = Path(batch["output_tif"])
    if not marker_path.exists() or not output_tif.exists():
        return False
    try:
        with rasterio.open(output_tif) as src:
            return int(src.width) > 0 and int(src.height) > 0
    except Exception:
        return False


def _initial_status(
    *,
    orchestration_root: Path,
    batch_manifests: list[Path],
    config_paths: list[Path],
    worker_scripts: list[Path],
    max_retries: int,
    batch_size: int,
) -> dict[str, Any]:
    """Build the initial orchestration status payload.

    Args:
        orchestration_root (Path): Root orchestration directory.
        batch_manifests (list[Path]): Batch manifest paths.
        config_paths (list[Path]): Batch config paths.
        worker_scripts (list[Path]): Batch worker scripts.
        max_retries (int): Maximum worker retries.
        batch_size (int): Images per batch.

    Returns:
        dict[str, Any]: Status payload.

    Examples:
        >>> callable(_initial_status)
        True
    """

    batches = []
    for idx, (manifest_path, config_path, worker_script) in enumerate(
        zip(batch_manifests, config_paths, worker_scripts)
    ):
        image_count = len(
            [
                line
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        batches.append(
            {
                "batch_id": idx,
                "images_file": str(manifest_path),
                "expected_images": image_count,
                "config_path": str(config_path),
                "worker_script": str(worker_script),
                "run_dir": str(_batch_run_dir(orchestration_root, idx)),
                "output_tif": str(_batch_output_tif(orchestration_root, idx)),
                "complete_marker": str(_batch_complete_marker(orchestration_root, idx)),
                "retry_count": 0,
                "status": "pending",
                "last_worker_job_id": None,
            }
        )
    return {
        "state": "submitted",
        "batch_size": batch_size,
        "max_retries": max_retries,
        "worker_job_ids": [],
        "controller_job_ids": [],
        "batches": batches,
    }


def _refresh_status(orchestration_root: Path) -> dict[str, Any]:
    """Refresh batch completion state from batch-local outputs.

    Args:
        orchestration_root (Path): Root orchestration directory.

    Returns:
        dict[str, Any]: Refreshed status payload.

    Examples:
        >>> callable(_refresh_status)
        True
    """

    status_path = orchestration_root / "status.json"
    status = _load_json(status_path)
    if not status:
        raise ValueError(f"missing status file: {status_path}")
    for batch in status.get("batches", []):
        batch["status"] = "complete" if _is_batch_complete(batch) else "incomplete"
    _write_json(status_path, status)
    return status


def _reset_raster_to_zero(path: Path) -> None:
    """Fill one raster with zeros in place.

    Args:
        path (Path): Raster path.

    Examples:
        >>> callable(_reset_raster_to_zero)
        True
    """

    with rasterio.open(path, "r+") as dst:
        window_size = 1024
        for row_off in range(0, int(dst.height), window_size):
            row_end = min(int(dst.height), row_off + window_size)
            for col_off in range(0, int(dst.width), window_size):
                col_end = min(int(dst.width), col_off + window_size)
                dst.write(
                    np.zeros((row_end - row_off, col_end - col_off), dtype=np.uint8),
                    1,
                    window=Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=col_end - col_off,
                        height=row_end - row_off,
                    ),
                )


def _read_raster_window(
    source_path: str,
    window: Window,
) -> tuple[Window, np.ndarray]:
    """Read one raster window from disk.

    Args:
        source_path (str): Source raster path.
        window (Window): Source window.

    Returns:
        tuple[Window, np.ndarray]: Window plus the corresponding pixel data.

    Examples:
        >>> callable(_read_raster_window)
        True
    """

    with rasterio.open(source_path) as src:
        data = src.read(1, window=window)
    return window, data


def _copy_prediction_raster_into_merge_output(
    *,
    output_tif: str,
    source_tif: str,
    read_workers: int,
) -> None:
    """Copy one aligned prediction TIFF into the shared merge output.

    Args:
        output_tif (str): Destination cumulative TIFF path.
        source_tif (str): Source TIFF path.
        read_workers (int): Number of parallel read workers.

    Examples:
        >>> callable(_copy_prediction_raster_into_merge_output)
        True
    """

    with rasterio.open(source_tif) as src, rasterio.open(output_tif, "r+") as dst:
        source_window = from_bounds(*src.bounds, transform=dst.transform)
        source_window = Window(
            col_off=int(round(source_window.col_off)),
            row_off=int(round(source_window.row_off)),
            width=max(1, int(round(source_window.width))),
            height=max(1, int(round(source_window.height))),
        )
        source_windows = [window for _, window in src.block_windows(1)]
        if not source_windows:
            source_windows = [
                Window(col_off=0, row_off=0, width=int(src.width), height=int(src.height))
            ]
        worker_count = max(1, int(read_workers))
        if worker_count == 1:
            for window in source_windows:
                data = src.read(1, window=window)
                dst.write(
                    data,
                    1,
                    window=Window(
                        col_off=int(source_window.col_off + window.col_off),
                        row_off=int(source_window.row_off + window.row_off),
                        width=int(window.width),
                        height=int(window.height),
                    ),
                )
            return

        pending: dict[Future, Window] = {}
        pending_limit = max(worker_count * 2, 2)
        window_iter = iter(source_windows)
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            while True:
                while len(pending) < pending_limit:
                    try:
                        window = next(window_iter)
                    except StopIteration:
                        break
                    future = pool.submit(_read_raster_window, source_tif, window)
                    pending[future] = window
                if not pending:
                    break
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    window = pending.pop(future)
                    _, data = future.result()
                    dst.write(
                        data,
                        1,
                        window=Window(
                            col_off=int(source_window.col_off + window.col_off),
                            row_off=int(source_window.row_off + window.row_off),
                            width=int(window.width),
                            height=int(window.height),
                        ),
                    )


def merge_batch_prediction_tifs(
    *,
    batch_tifs: list[str],
    output_tif: str,
    read_workers: int | None = None,
) -> str:
    """Merge batch-level prediction TIFFs into one cumulative GeoTIFF.

    Args:
        batch_tifs (list[str]): Batch prediction TIFF paths in merge order.
        output_tif (str): Final merged GeoTIFF path.
        read_workers (int | None): Optional number of parallel source-window
            readers used during the merge copy phase.

    Returns:
        str: Final merged GeoTIFF path.

    Examples:
        >>> callable(merge_batch_prediction_tifs)
        True
    """

    if not batch_tifs:
        raise ValueError("batch_tifs cannot be empty")
    output_path = Path(output_tif)
    first_path = Path(batch_tifs[0])
    merge_read_workers = (
        max(1, min(8, os.cpu_count() or 1))
        if read_workers is None
        else max(1, int(read_workers))
    )
    union_bounds: tuple[float, float, float, float] | None = None
    ref_transform = None
    ref_crs = None
    ref_res_x = None
    ref_res_y = None
    for tif_path_str in batch_tifs:
        tif_path = Path(tif_path_str)
        with rasterio.open(tif_path) as src:
            if ref_crs is None:
                ref_crs = src.crs
                ref_transform = src.transform
                ref_res_x = float(src.transform.a)
                ref_res_y = float(src.transform.e)
            else:
                if src.crs != ref_crs:
                    raise ValueError("batch TIFF CRS mismatch during merge")
                if (
                    not np.isclose(float(src.transform.a), float(ref_res_x))
                    or not np.isclose(float(src.transform.e), float(ref_res_y))
                    or not np.isclose(float(src.transform.b), float(ref_transform.b))
                    or not np.isclose(float(src.transform.d), float(ref_transform.d))
                ):
                    raise ValueError(
                        "batch TIFF resolution/rotation mismatch during merge"
                    )
                col_shift = (float(src.transform.c) - float(ref_transform.c)) / float(
                    ref_res_x
                )
                row_shift = (float(src.transform.f) - float(ref_transform.f)) / float(
                    ref_res_y
                )
                if not np.isclose(col_shift, round(col_shift), atol=1e-6):
                    raise ValueError("batch TIFF x-origin is not aligned to merge grid")
                if not np.isclose(row_shift, round(row_shift), atol=1e-6):
                    raise ValueError("batch TIFF y-origin is not aligned to merge grid")
            bounds = src.bounds
            if union_bounds is None:
                union_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            else:
                union_bounds = (
                    min(union_bounds[0], bounds.left),
                    min(union_bounds[1], bounds.bottom),
                    max(union_bounds[2], bounds.right),
                    max(union_bounds[3], bounds.top),
                )
    assert ref_transform is not None
    assert union_bounds is not None
    template_window = from_bounds(*union_bounds, transform=ref_transform)
    template_window = Window(
        col_off=int(round(template_window.col_off)),
        row_off=int(round(template_window.row_off)),
        width=max(1, int(round(template_window.width))),
        height=max(1, int(round(template_window.height))),
    )
    created = ensure_cumulative_prediction_raster(
        str(output_path),
        str(first_path),
        template_window=template_window,
        num_threads="ALL_CPUS",
    )
    if not created:
        _reset_raster_to_zero(output_path)
    for tif_path_str in batch_tifs:
        _copy_prediction_raster_into_merge_output(
            output_tif=str(output_path),
            source_tif=tif_path_str,
            read_workers=merge_read_workers,
        )
    return str(output_path)


def launch_batched_inference(
    *,
    job_name: str,
    batch_size: int,
    template_config_path: Path,
    template_slurm_path: Path,
    output_root: Path,
    max_retries: int,
    dry_run: bool,
) -> Path:
    """Build batch artifacts and submit worker/controller Slurm jobs.

    Args:
        job_name (str): Orchestration job name.
        batch_size (int): Maximum images per batch.
        template_config_path (Path): Base YAML config path.
        template_slurm_path (Path): Slurm template path.
        output_root (Path): Root directory for orchestration outputs.
        max_retries (int): Maximum controller retries for incomplete batches.
        dry_run (bool): When true, skip real `sbatch` submissions.

    Returns:
        Path: Orchestration root.

    Examples:
        >>> callable(launch_batched_inference)
        True
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    orchestration_root, batch_manifests = build_inference_batches(
        template_config_path=template_config_path,
        batch_size=batch_size,
        output_root=output_root,
        job_name=job_name,
    )
    config_paths = _write_batch_configs(
        orchestration_root=orchestration_root,
        template_config_path=template_config_path,
        batch_manifests=batch_manifests,
    )
    worker_scripts, controller_script = _write_slurm_scripts(
        orchestration_root=orchestration_root,
        template_slurm_path=template_slurm_path,
        job_name=job_name,
        config_paths=config_paths,
    )
    manifest = _load_json(orchestration_root / "manifest.json")
    manifest.update(
        {
            "template_slurm": str(template_slurm_path),
            "worker_scripts": [str(path) for path in worker_scripts],
            "controller_script": str(controller_script),
            "config_paths": [str(path) for path in config_paths],
            "repo_root": str(REPO_ROOT),
        }
    )
    _write_json(orchestration_root / "manifest.json", manifest)
    status = _initial_status(
        orchestration_root=orchestration_root,
        batch_manifests=batch_manifests,
        config_paths=config_paths,
        worker_scripts=worker_scripts,
        max_retries=max_retries,
        batch_size=batch_size,
    )
    worker_submissions: list[dict[str, Any]] = []
    worker_job_ids: list[str | None] = []
    for batch in status["batches"]:
        submission = _submit_sbatch(
            script_path=Path(batch["worker_script"]),
            extra_args=None,
            dry_run=dry_run,
        )
        batch["last_worker_job_id"] = submission["job_id"]
        worker_submissions.append(
            {"batch_id": batch["batch_id"], "submission": submission}
        )
        status["worker_job_ids"].append(submission["job_id"])
        worker_job_ids.append(submission["job_id"])
    controller_submission = _submit_sbatch(
        script_path=controller_script,
        extra_args=_dependency_args(worker_job_ids),
        dry_run=dry_run,
    )
    status["controller_job_ids"].append(controller_submission["job_id"])
    _write_json(orchestration_root / "status.json", status)
    _write_json(
        orchestration_root / "submission.json",
        {
            "dry_run": dry_run,
            "worker_submissions": worker_submissions,
            "controller_submission": controller_submission,
        },
    )
    return orchestration_root


def submit_controller_only(
    *,
    orchestration_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Submit only the existing controller script for one orchestration root.

    Args:
        orchestration_root (Path): Root orchestration directory.
        dry_run (bool): When true, skip the real `sbatch` submission.

    Returns:
        dict[str, Any]: Submission metadata.

    Examples:
        >>> callable(submit_controller_only)
        True
    """

    manifest = _load_json(orchestration_root / "manifest.json")
    status = _load_json(orchestration_root / "status.json")
    controller_script = manifest.get("controller_script")
    if not controller_script:
        raise ValueError(
            f"missing controller_script in {orchestration_root / 'manifest.json'}"
        )
    submission = _submit_sbatch(
        script_path=Path(controller_script),
        extra_args=None,
        dry_run=dry_run,
    )
    status.setdefault("controller_job_ids", []).append(submission["job_id"])
    status["state"] = "controller_submitted"
    _write_json(orchestration_root / "status.json", status)
    _write_json(
        orchestration_root / "submission.json",
        {
            **_load_json(orchestration_root / "submission.json"),
            "manual_controller_submission": submission,
        },
    )
    return submission


def run_controller(*, orchestration_root: Path, dry_run: bool) -> None:
    """Retry incomplete batches or merge finished batch outputs.

    Args:
        orchestration_root (Path): Root orchestration directory.
        dry_run (bool): When true, skip real `sbatch` submissions.

    Examples:
        >>> callable(run_controller)
        True
    """

    manifest = _load_json(orchestration_root / "manifest.json")
    status = _refresh_status(orchestration_root)
    incomplete = [
        batch
        for batch in status.get("batches", [])
        if batch.get("status") != "complete"
    ]
    if not incomplete:
        merged_dir = orchestration_root / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_output = merge_batch_prediction_tifs(
            batch_tifs=[batch["output_tif"] for batch in status["batches"]],
            output_tif=str(merged_dir / "predictions.tif"),
        )
        status["state"] = "complete"
        _write_json(orchestration_root / "status.json", status)
        _write_json(
            orchestration_root / "final_status.json",
            {"status": "success", "merged_output_tif": merged_output},
        )
        return

    retryable = [
        batch
        for batch in incomplete
        if int(batch.get("retry_count", 0)) < int(status.get("max_retries", 0))
    ]
    if not retryable:
        status["state"] = "failed_incomplete"
        _write_json(orchestration_root / "status.json", status)
        _write_json(
            orchestration_root / "final_status.json",
            {
                "status": "failed_incomplete",
                "incomplete_batches": [
                    {
                        "batch_id": batch["batch_id"],
                        "output_tif": batch["output_tif"],
                        "complete_marker": batch["complete_marker"],
                        "retry_count": batch["retry_count"],
                    }
                    for batch in incomplete
                ],
            },
        )
        raise SystemExit("incomplete batches remain and retry limit is exhausted")

    retry_submissions: list[dict[str, Any]] = []
    retry_job_ids: list[str | None] = []
    for batch in retryable:
        batch["retry_count"] = int(batch["retry_count"]) + 1
        batch["status"] = "retrying"
        submission = _submit_sbatch(
            script_path=Path(batch["worker_script"]),
            extra_args=None,
            dry_run=dry_run,
        )
        batch["last_worker_job_id"] = submission["job_id"]
        status["worker_job_ids"].append(submission["job_id"])
        retry_job_ids.append(submission["job_id"])
        retry_submissions.append(
            {"batch_id": batch["batch_id"], "submission": submission}
        )
    controller_submission = _submit_sbatch(
        script_path=Path(manifest["controller_script"]),
        extra_args=_dependency_args(retry_job_ids),
        dry_run=dry_run,
    )
    status["controller_job_ids"].append(controller_submission["job_id"])
    status["state"] = "retry_submitted"
    _write_json(orchestration_root / "status.json", status)
    _write_json(
        orchestration_root / "submission.json",
        {
            **_load_json(orchestration_root / "submission.json"),
            "last_retry_worker_submissions": retry_submissions,
            "last_retry_controller_submission": controller_submission,
        },
    )


def main() -> None:
    """CLI entrypoint for batch-based Slurm inference launches.

    Examples:
        >>> callable(main)
        True
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", help="Name for the orchestration root.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Maximum number of images per worker batch.",
    )
    parser.add_argument(
        "--template-config",
        default=str(DEFAULT_TEMPLATE_CONFIG),
        help="Read-only base YAML config used to derive batch configs.",
    )
    parser.add_argument(
        "--template-slurm",
        default=str(DEFAULT_TEMPLATE_SLURM),
        help="Read-only Slurm template used to render worker/controller scripts.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where orchestration manifests, scripts, runs, and merges live.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum controller retries for incomplete batches.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render artifacts without calling sbatch.",
    )
    parser.add_argument(
        "--controller",
        action="store_true",
        help="Internal mode: inspect progress, retry incomplete batches, and merge outputs.",
    )
    parser.add_argument(
        "--submit-controller",
        action="store_true",
        help="Submit only the existing controller script for an orchestration root.",
    )
    parser.add_argument(
        "--orchestration-root",
        help="Existing orchestration root for controller-oriented modes.",
    )
    args = parser.parse_args()

    if args.controller:
        if not args.orchestration_root:
            raise ValueError("--orchestration-root is required with --controller")
        run_controller(
            orchestration_root=_resolve_relative_path(
                args.orchestration_root,
                base_dir=Path.cwd(),
            ),
            dry_run=bool(args.dry_run),
        )
        return

    if args.submit_controller:
        if not args.orchestration_root:
            raise ValueError(
                "--orchestration-root is required with --submit-controller"
            )
        submit_controller_only(
            orchestration_root=_resolve_relative_path(
                args.orchestration_root,
                base_dir=Path.cwd(),
            ),
            dry_run=bool(args.dry_run),
        )
        return

    if not args.job_name:
        raise ValueError("--job-name is required for launch mode")
    launch_batched_inference(
        job_name=args.job_name,
        batch_size=int(args.batch_size),
        template_config_path=_resolve_relative_path(
            args.template_config,
            base_dir=Path.cwd(),
        ),
        template_slurm_path=_resolve_relative_path(
            args.template_slurm,
            base_dir=Path.cwd(),
        ),
        output_root=_resolve_relative_path(args.output_root, base_dir=Path.cwd()),
        max_retries=int(args.max_retries),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
