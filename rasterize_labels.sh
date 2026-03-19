#!/usr/bin/env bash
#SBATCH --job-name=rasterize_labels
#SBATCH --output=rasterize_labels_%j.out
#SBATCH --error=rasterize_labels_%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=scc-gpu
#SBATCH -G A100:1

set -euo pipefail

module load miniforge3 gcc cuda
source activate "${SEGEDGE_CONDA_ENV:-/mnt/vast-standard/home/davide.mattioli/u20330/all}"

: "${REPO_ROOT:=/user/davide.mattioli/u20330/Dinov3-LWF-Segmentation}"
: "${RASTERIZE_CONFIG_PATH:=${REPO_ROOT}/configs/rasterize_labels.yml}"

cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "repo_root=${REPO_ROOT}"
echo "rasterize_config_path=${RASTERIZE_CONFIG_PATH}"

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
  "${RASTERIZE_CONFIG_PATH}"
)

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
