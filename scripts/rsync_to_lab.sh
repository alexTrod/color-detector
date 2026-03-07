#!/usr/bin/env bash
set -euo pipefail

# Sync identify repo to alex@lab:/home/alex/identify
# Usage: REMOTE_HOST=alex@lab [SYNC_DATA=1] ./scripts/rsync_to_lab.sh

REMOTE_HOST="${REMOTE_HOST:-alex@lab}"
REMOTE_DIR="${REMOTE_DIR:-/home/alex/identify}"
SYNC_DATA="${SYNC_DATA:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDENTIFY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${REMOTE_HOST}" ]]; then
  echo "Set REMOTE_HOST (e.g. REMOTE_HOST=alex@lab)"
  exit 1
fi

echo "[1/2] Ensure remote directory exists"
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

echo "[2/2] Rsync identify -> ${REMOTE_HOST}:${REMOTE_DIR}"
RSYNC_ARGS=(
  -azP
  --exclude ".git"
  --exclude "__pycache__"
  --exclude ".ipynb_checkpoints"
  --exclude "*.pyc"
  --exclude ".conda"
  --exclude "runs/zuna_*"
)
if [[ "${SYNC_DATA}" != "1" ]]; then
  RSYNC_ARGS+=(--exclude "data/raw_samples")
  RSYNC_ARGS+=(--exclude "data/processed/epochs")
  RSYNC_ARGS+=(--exclude "data/processed/epochs_aug")
  RSYNC_ARGS+=(--exclude "data/processed/epochs_yoto_tones")
  RSYNC_ARGS+=(--exclude "data/processed/fif_for_zuna")
fi
rsync "${RSYNC_ARGS[@]}" "${IDENTIFY_ROOT}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Done. Run notebook on server: ssh ${REMOTE_HOST} 'cd ${REMOTE_DIR} && python -m jupyter nbconvert --to notebook --execute notebooks/yoto_labram_pipeline.ipynb --output yoto_executed.ipynb'"
