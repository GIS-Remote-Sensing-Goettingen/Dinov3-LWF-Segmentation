#!/usr/bin/env bash
#SBATCH --job-name=rasterize_labels
#SBATCH --output=rasterize_labels_%j.out
#SBATCH --error=rasterize_labels_%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=scc-gpu
#SBATCH -G A100:1

set -euo pipefail

module load miniforge3 gcc cuda
source activate "${SEGEDGE_CONDA_ENV:-/mnt/vast-standard/home/davide.mattioli/u20330/all}"

: "${REPO_ROOT:=/user/davide.mattioli/u20330/Dinov3-LWF-Segmentation}"

cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

: "${VECTOR_PATH:=${REPO_ROOT}/crf/union.shp}"
: "${REFERENCE_PATH:=/user/davide.mattioli/u20330/planet_labels_2022.tif}"
: "${OUTPUT_PATH:=${REPO_ROOT}/crf/new_labels.tif}"
: "${REFERENCE_GLOB:=*.tif}"
: "${VECTOR_CRS:=EPSG:25832}"
: "${WINDOW_SIZE:=0}"
: "${STREAM_THRESHOLD_PIXELS:=50000000}"
: "${WORKERS:=${SLURM_CPUS_PER_TASK:-1}}"
: "${LOG_LEVEL:=INFO}"
: "${OVERWRITE:=0}"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "repo_root=${REPO_ROOT}"
echo "vector_path=${VECTOR_PATH}"
echo "reference_path=${REFERENCE_PATH}"
echo "output_path=${OUTPUT_PATH}"
echo "reference_glob=${REFERENCE_GLOB}"
echo "vector_crs=${VECTOR_CRS}"
echo "window_size=${WINDOW_SIZE}"
echo "stream_threshold_pixels=${STREAM_THRESHOLD_PIXELS}"
echo "workers=${WORKERS}"
echo "log_level=${LOG_LEVEL}"
echo "overwrite=${OVERWRITE}"

nvidia-smi || true
python --version
python - <<'PY'
import fiona
import rasterio

print(f"fiona={fiona.__version__}")
print(f"rasterio={rasterio.__version__}")
PY

cmd=(
  python -u ./scripts/rasterize_vector_labels.py
  "${VECTOR_PATH}"
  "${REFERENCE_PATH}"
  "${OUTPUT_PATH}"
  --glob "${REFERENCE_GLOB}"
  --vector-crs "${VECTOR_CRS}"
  --stream-threshold-pixels "${STREAM_THRESHOLD_PIXELS}"
  --workers "${WORKERS}"
  --log-level "${LOG_LEVEL}"
)

if [[ "${WINDOW_SIZE}" -gt 0 ]]; then
  cmd+=(--window-size "${WINDOW_SIZE}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
