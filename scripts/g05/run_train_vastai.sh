#!/usr/bin/env bash
set -euo pipefail
: "${G05_ROOT:?set G05_ROOT on the training instance}"
: "${G05_BASE_ASSETS:?set G05_BASE_ASSETS on the training instance}"
: "${ROBODOJO_LEROBOT_V30_ROOT:?set dataset path}"
: "${ROBODOJO_SIDECAR:?set sidecar path}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export G05_SIDECAR_JSONL="${G05_SIDECAR_JSONL:-${ROOT}/artifacts/g05_robodojo/subgoal_samples.jsonl}"
export G05_TRAIN_MODE="${G05_TRAIN_MODE:-ar_fm}"
export G05_ACTION_SOURCE="${G05_ACTION_SOURCE:-ar_fm}"
export G05_RESUME="${G05_RESUME:-}"
python3 "${ROOT}/scripts/g05/preflight.py" --manifest "${G05_SIDECAR_JSONL}"
cd "${ROOT}/XPolicyLab/policy/G05"
exec bash train.sh RoboDojo g05_12task_ar_fm arx_x5 joint "${G05_SEED:-42}" "${G05_GPUS:-0,1,2,3}" "$@"
