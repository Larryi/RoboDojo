#!/usr/bin/env bash
set -euo pipefail
# Run inside the VastAI instance. No credentials or model/data downloads are embedded.
ROOT="${ROBODOJO_ROOT:-/workspace/RoboDojo}"
python3 -m venv "${ROOT}/.venv-g05" 2>/dev/null || true
"${ROOT}/.venv-g05/bin/python" -m pip install --upgrade pip
if [[ -f "${G05_ROOT:-}/GalaxeaVLA/pyproject.toml" ]]; then
  "${ROOT}/.venv-g05/bin/pip" install -e "${G05_ROOT}/GalaxeaVLA"
fi
echo "Bootstrap complete. Install Isaac/G05 dependencies using the official G0.5 instructions and then run preflight.py."
