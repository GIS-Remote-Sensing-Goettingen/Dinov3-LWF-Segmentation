"""Merge multiple folder-level prediction TIFFs into one final GeoTIFF.

Examples:
    >>> isinstance(REPO_ROOT.name, str)
    True
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from launch_batched_inference import merge_batch_prediction_tifs  # noqa: E402

DEFAULT_BATCHES_ROOT = REPO_ROOT / "output" / "batches"
DEFAULT_FOLDERS = (
    "folder1_infer",
    "folder2_infer",
    "folder3_infer",
    "folder4_infer",
)
logger = logging.getLogger(__name__)


def resolve_folder_prediction_tifs(
    *,
    batches_root: Path,
    folder_names: list[str],
) -> list[str]:
    """Resolve the merged prediction TIFF for each folder orchestration root.

    Args:
        batches_root (Path): Root directory containing folder orchestration runs.
        folder_names (list[str]): Folder orchestration names to merge.

    Returns:
        list[str]: Resolved merged TIFF paths in the requested order.

    Examples:
        >>> callable(resolve_folder_prediction_tifs)
        True
    """

    tif_paths: list[str] = []
    for folder_name in folder_names:
        tif_path = batches_root / folder_name / "merged" / "predictions.tif"
        if not tif_path.exists():
            raise FileNotFoundError(f"missing merged prediction TIFF: {tif_path}")
        tif_paths.append(str(tif_path.resolve()))
    return tif_paths


def merge_folder_prediction_tifs(
    *,
    batches_root: Path,
    folder_names: list[str],
    output_tif: Path,
    read_workers: int | None = None,
) -> str:
    """Merge the selected folder-level merged TIFFs into one final raster.

    Args:
        batches_root (Path): Root directory containing folder orchestration runs.
        folder_names (list[str]): Folder orchestration names to merge.
        output_tif (Path): Destination final merged TIFF path.
        read_workers (int | None): Optional number of parallel source-window
            readers used during the merge copy phase.

    Returns:
        str: Final merged TIFF path.

    Examples:
        >>> callable(merge_folder_prediction_tifs)
        True
    """

    tif_paths = resolve_folder_prediction_tifs(
        batches_root=batches_root,
        folder_names=folder_names,
    )
    logger.info("Merging folder predictions: %s", ", ".join(folder_names))
    return merge_batch_prediction_tifs(
        batch_tifs=tif_paths,
        output_tif=str(output_tif),
        read_workers=read_workers,
    )


def main() -> None:
    """CLI entrypoint for folder-level prediction TIFF merging.

    Examples:
        >>> callable(main)
        True
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches-root",
        default=str(DEFAULT_BATCHES_ROOT),
        help="Root directory containing folder orchestration outputs.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=list(DEFAULT_FOLDERS),
        help="Folder orchestration names to merge, in overwrite order.",
    )
    parser.add_argument(
        "--output-tif",
        required=True,
        help="Destination final merged TIFF path.",
    )
    parser.add_argument(
        "--read-workers",
        type=int,
        default=None,
        help="Optional number of parallel source-window readers.",
    )
    args = parser.parse_args()
    output_path = merge_folder_prediction_tifs(
        batches_root=Path(args.batches_root).expanduser().resolve(),
        folder_names=list(args.folders),
        output_tif=Path(args.output_tif).expanduser().resolve(),
        read_workers=args.read_workers,
    )
    print(output_path)


if __name__ == "__main__":
    main()
