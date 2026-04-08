#!/usr/bin/env bash
set -euo pipefail

# Download the locally staged Hugging Face Mask2Former semantic checkpoint used
# by configs/thesis_runs/R13_mask2former_semantic_split_s1337.yml.
#
# Intended usage on the cluster:
#   srun --pty -p standard96s:test -N 1 -c 1 -C inet /bin/bash
#   cd /path/to/Dinov3-LWF-Segmentation
#   bash scripts/download_mask2former_semantic_weights.sh
#
# Optional overrides:
#   MASK2FORMER_REPO_ID
#   MASK2FORMER_TARGET_DIR
#   http_proxy / https_proxy / ftp_proxy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MASK2FORMER_REPO_ID="${MASK2FORMER_REPO_ID:-facebook/mask2former-swin-base-ade-semantic}"
MASK2FORMER_TARGET_DIR="${MASK2FORMER_TARGET_DIR:-/user/davide.mattioli/u20330/Dinov3-LWF-Segmentation/weights/hf/facebook/mask2former-swin-base-ade-semantic}"
export MASK2FORMER_REPO_ID
export MASK2FORMER_TARGET_DIR

export http_proxy="${http_proxy:-http://www-cache.gwdg.de:3128}"
export https_proxy="${https_proxy:-http://www-cache.gwdg.de:3128}"
export ftp_proxy="${ftp_proxy:-http://www-cache.gwdg.de:3128}"

mkdir -p "${MASK2FORMER_TARGET_DIR}"

echo "repo_root=${REPO_ROOT}"
echo "repo_id=${MASK2FORMER_REPO_ID}"
echo "target_dir=${MASK2FORMER_TARGET_DIR}"
echo "http_proxy=${http_proxy}"

python3 - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MASK2FORMER_REPO_ID"],
    local_dir=os.environ["MASK2FORMER_TARGET_DIR"],
    local_dir_use_symlinks=False,
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.bin",
        "*.model",
        "*.txt",
    ],
)
print("done")
PY

echo
echo "Downloaded files:"
ls -lh "${MASK2FORMER_TARGET_DIR}"
