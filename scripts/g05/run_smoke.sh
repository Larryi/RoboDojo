#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 -m py_compile "${ROOT}/scripts/g05/sidecar_manifest.py" "${ROOT}/scripts/g05/balanced_sampler.py" "${ROOT}/scripts/g05/preflight.py"
ROBODOJO_SIDECAR="${ROBODOJO_SIDECAR:-/mnt/pqssd/RoboInter/RoboInterTools/annotations/lerobot_v30_joint.sqlite3}" \
  bash "${ROOT}/scripts/g05/make_manifest.sh" "${G05_ARTIFACT_DIR:-${ROOT}/artifacts/g05_smoke}"
bash -n "${ROOT}/scripts/g05/"*.sh "${ROOT}/XPolicyLab/policy/G05/"*.sh
echo "G05 CPU/config smoke passed; no model/data download and no training launched."
