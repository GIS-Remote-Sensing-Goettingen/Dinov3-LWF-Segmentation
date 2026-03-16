#!/usr/bin/env bash
#SBATCH --job-name=segmentation
#SBATCH --output=segmentation_%j.out
#SBATCH --error=segmentation_%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=scc-gpu
#SBATCH -G H100:2

set -euo pipefail

module load miniforge3 gcc cuda
# Activate env (allow override)
source activate "${SEGEDGE_CONDA_ENV:-/mnt/vast-standard/home/davide.mattioli/u20330/all}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

detect_allocated_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local devices=()
    IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES}"
    echo "${#devices[@]}"
    return
  fi
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    echo "${SLURM_GPUS_ON_NODE}"
    return
  fi
  echo 2
}

ALLOCATED_GPUS="$(detect_allocated_gpus)"
GPUS_PER_NODE="${GPUS_PER_NODE:-${ALLOCATED_GPUS}}"
CONFIG_PATH="${CONFIG_PATH:-configs/config_hpc.yml}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "${GPUS_PER_NODE}" -gt "${ALLOCATED_GPUS}" ]]; then
  echo "Requested GPUS_PER_NODE=${GPUS_PER_NODE} but only ${ALLOCATED_GPUS} GPUs are visible." >&2
  exit 1
fi

# Show GPU and env info (useful for debugging)
nvidia-smi || true
echo "repo_root=${REPO_ROOT}"
echo "config_path=${CONFIG_PATH}"
echo "allocated_gpus=${ALLOCATED_GPUS}"
echo "gpus_per_node=${GPUS_PER_NODE}"
echo "master_port=${MASTER_PORT}"
python --version
python -m torch.utils.collect_env

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  ./main.py "${CONFIG_PATH}"
