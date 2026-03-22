#!/usr/bin/env bash
#SBATCH --job-name=merge_all_folders
#SBATCH --partition=scc-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=merge_all_folders_%j.out
#SBATCH --error=merge_all_folders_%j.err

set -euo pipefail

module load miniforge3 gcc cuda
source activate "${SEGEDGE_CONDA_ENV:-/mnt/vast-standard/home/davide.mattioli/u20330/all}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1

BATCHES_ROOT="${BATCHES_ROOT:-${REPO_ROOT}/output/batches}"
OUTPUT_TIF="${OUTPUT_TIF:-${BATCHES_ROOT}/all_folders_merged/predictions.tif}"
READ_WORKERS="${READ_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
FOLDERS="${FOLDERS:-folder1_infer folder2_infer folder3_infer folder4_infer}"
MERGE_SCRIPT="${REPO_ROOT}/scripts/merge_folder_prediction_tifs.py"

if [[ ! -f "${MERGE_SCRIPT}" ]]; then
  echo "Could not find merge script at ${MERGE_SCRIPT}." >&2
  exit 1
fi

echo "repo_root=${REPO_ROOT}"
echo "batches_root=${BATCHES_ROOT}"
echo "output_tif=${OUTPUT_TIF}"
echo "folders=${FOLDERS}"
echo "read_workers=${READ_WORKERS}"
python --version

python "${MERGE_SCRIPT}" \
  --batches-root "${BATCHES_ROOT}" \
  --folders ${FOLDERS} \
  --output-tif "${OUTPUT_TIF}" \
  --read-workers "${READ_WORKERS}"
