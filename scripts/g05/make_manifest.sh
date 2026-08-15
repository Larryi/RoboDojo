#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${ROBODOJO_LEROBOT_V30_ROOT:-${ROOT}/data/lerobot_v30_joint}"
SIDECAR="${ROBODOJO_SIDECAR:-/mnt/pqssd/RoboInter/RoboInterTools/annotations/lerobot_v30_joint.sqlite3}"
OUT="${1:-${ROOT}/artifacts/g05_robodojo}"
mkdir -p "${OUT}"
python3 "${ROOT}/scripts/g05/sidecar_manifest.py" \
  --sidecar "${SIDECAR}" --expected-tasks "${G05_EXPECTED_TASKS:-12}" \
  --chunk "${G05_ACTION_CHUNK:-16}" --stride "${G05_SAMPLE_STRIDE:-16}" \
  --manifest "${OUT}/subgoal_samples.jsonl" --summary "${OUT}/sidecar_summary.json"
python3 "${ROOT}/scripts/g05/balanced_sampler.py" \
  --input "${OUT}/subgoal_samples.jsonl" --output "${OUT}/balanced_samples.jsonl" \
  --samples "${G05_SAMPLES_PER_EPOCH:-34744}" --seed "${G05_SAMPLER_SEED:-42}"
printf '%s\n' "dataset=${DATASET}" "sidecar=${SIDECAR}" "output=${OUT}"
