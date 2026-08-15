#!/usr/bin/env bash
set -euo pipefail
: "${VAST_HOST:?set VAST_HOST=user@host}"
: "${VAST_ROOT:?set VAST_ROOT=/workspace/g05-run}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${G05_PULL_DEST:-${ROOT}/artifacts/g05_vastai}"
mkdir -p "${DEST}"
rsync -a --partial --info=progress2 "${VAST_HOST}:${VAST_ROOT}/RoboDojo/artifacts/g05_robodojo/" "${DEST}/artifacts/g05_robodojo/"
rsync -a --partial --info=progress2 "${VAST_HOST}:${VAST_ROOT}/G05/outputs/" "${DEST}/outputs/" || true
rsync -a --partial --info=progress2 "${VAST_HOST}:${VAST_ROOT}/logs/" "${DEST}/logs/" || true
