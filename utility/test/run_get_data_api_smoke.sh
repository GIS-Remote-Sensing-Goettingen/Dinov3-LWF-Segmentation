#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/utility/test/download_smoke}"
MAX_TILES="${MAX_TILES:-2}"
MAX_WORKERS="${MAX_WORKERS:-2}"
TILE_ORIGIN="${TILE_ORIGIN:-}"
START_X="${START_X:-}"
START_Y="${START_Y:-}"
BLOCK_SIZE_KM="${BLOCK_SIZE_KM:-}"

mkdir -p "${OUT_DIR}"

cmd=(
  python
  "${REPO_ROOT}/utility/get_data_api.py"
  --out-dir "${OUT_DIR}"
  --max-workers "${MAX_WORKERS}"
  --max-tiles "${MAX_TILES}"
)

if [[ -n "${TILE_ORIGIN}" ]]; then
  cmd+=(--tile-origin "${TILE_ORIGIN}")
fi

if [[ -n "${START_X}" || -n "${START_Y}" || -n "${BLOCK_SIZE_KM}" ]]; then
  if [[ -z "${START_X}" || -z "${START_Y}" || -z "${BLOCK_SIZE_KM}" ]]; then
    echo "START_X, START_Y, and BLOCK_SIZE_KM must all be set together" >&2
    exit 1
  fi
  if ! [[ "${START_X}" =~ ^-?[0-9]+$ && "${START_Y}" =~ ^-?[0-9]+$ && "${BLOCK_SIZE_KM}" =~ ^[0-9]+$ ]]; then
    echo "START_X, START_Y, and BLOCK_SIZE_KM must be integer values" >&2
    exit 1
  fi
  if (( BLOCK_SIZE_KM <= 0 )); then
    echo "BLOCK_SIZE_KM must be > 0" >&2
    exit 1
  fi
  for ((x = START_X; x < START_X + BLOCK_SIZE_KM * 1000; x += 1000)); do
    for ((y = START_Y; y < START_Y + BLOCK_SIZE_KM * 1000; y += 1000)); do
      cmd+=(--tile-origin "${x},${y}")
    done
  done
fi

cmd+=("$@")
"${cmd[@]}"
