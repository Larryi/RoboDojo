#!/usr/bin/env bash
set -Eeuo pipefail

SSH_TARGET="${1:?Usage: $0 user@host [ssh_port]}"
SSH_PORT="${2:-${VAST_SSH_PORT:-22}}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_ROOT="${G05_VAST_ROOT:-/workspace/g05-robodojo}"
SSH_OPTS=(-p "${SSH_PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

if [[ -n "${G05_VAST_SECRETS:-}" ]]; then
  SECRETS_FILE="${G05_VAST_SECRETS}"
elif [[ -f "${HOME}/.config/robodojo/g05_vast.env" ]]; then
  SECRETS_FILE="${HOME}/.config/robodojo/g05_vast.env"
else
  # One-command path: reuse the local HF CLI token and exported optional
  # credentials without putting secrets into SSH command-line arguments.
  command -v hf >/dev/null 2>&1 || { echo "Install/authenticate hf or create g05_vast.env" >&2; exit 2; }
  HF_TOKEN="${HF_TOKEN:-$(hf auth token 2>/dev/null || true)}"
  [[ -n "${HF_TOKEN}" ]] || { echo "No HF token. Run hf auth login or create g05_vast.env" >&2; exit 2; }
  SECRETS_FILE="$(mktemp "${TMPDIR:-/tmp}/g05-vast-env.XXXXXX")"
  chmod 600 "${SECRETS_FILE}"
  trap 'rm -f "${SECRETS_FILE}"' EXIT
  emit_env() { local key="$1" value="${!1-}"; printf '%s=%q\n' "${key}" "${value}" >> "${SECRETS_FILE}"; }
  emit_env HF_TOKEN
  for key in MODELSCOPE_API_TOKEN WANDB_API_KEY SERVERCHAN_SENDKEY HF_DATASET_REPO HF_OUTPUT_REPO \
    G05_GPUS G05_SAVE_INTERVAL_STEPS G05_KEEP_CHECKPOINTS G05_DATASET_SOURCE G05_TRAIN_TASK G05_TRAIN_ARGS; do
    emit_env "${key}"
  done
fi
[[ -f "${SECRETS_FILE}" ]] || { echo "Missing ${SECRETS_FILE}" >&2; exit 2; }

ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "mkdir -p '${REMOTE_ROOT}/.secrets' '${REMOTE_ROOT}/logs'"
scp "${SSH_OPTS[@]}" "${SECRETS_FILE}" "${SSH_TARGET}:${REMOTE_ROOT}/.secrets/g05.env"
scp "${SSH_OPTS[@]}" "${ROOT}/scripts/g05/vast_remote_train.sh" "${SSH_TARGET}:${REMOTE_ROOT}/vast_remote_train.sh"
ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "chmod 600 '${REMOTE_ROOT}/.secrets/g05.env' && chmod 700 '${REMOTE_ROOT}/vast_remote_train.sh' && nohup '${REMOTE_ROOT}/vast_remote_train.sh' >> '${REMOTE_ROOT}/logs/launcher.log' 2>&1 </dev/null & echo TRAIN_PID=\$!"
echo "Remote G05 training launched."
echo "Logs: ssh -p ${SSH_PORT} ${SSH_TARGET} tail -f ${REMOTE_ROOT}/logs/launcher.log"
echo "Status: ssh -p ${SSH_PORT} ${SSH_TARGET} cat ${REMOTE_ROOT}/runs/*/status.txt"
