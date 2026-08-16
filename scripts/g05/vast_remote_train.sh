#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

WORK_ROOT="${G05_VAST_ROOT:-/workspace/g05-robodojo}"
SECRETS_FILE="${G05_SECRETS_FILE:-${WORK_ROOT}/.secrets/g05.env}"
mkdir -p "${WORK_ROOT}/.secrets" "${WORK_ROOT}/logs"
[[ -f "${SECRETS_FILE}" ]] || { echo "missing ${SECRETS_FILE}" >&2; exit 2; }
set -a
source "${SECRETS_FILE}"
set +a

: "${HF_TOKEN:?HF_TOKEN is required for dataset/model downloads and final upload}"
: "${HF_DATASET_REPO:=larryi/RoboDojo-G05-12task}"
: "${HF_DATASET_SOURCE_REPO:=RoboDojo-Benchmark/GOAI-2026}"
: "${HF_DATASET_SOURCE_PATH:=data/lerobot_v30_joint}"
: "${HF_OUTPUT_REPO:=larryi/G05-RoboDojo-12task}"
: "${ROBO_DOJO_REPO_URL:=https://github.com/Larryi/RoboDojo.git}"
: "${ROBO_DOJO_BRANCH:=codex/g05-vastai-training}"
: "${XPL_REPO_URL:=https://github.com/Larryi/XPolicyLab.git}"
: "${XPL_BRANCH:=codex/g05-subgoal-sidecar}"
: "${G05_REPO_URL:=https://github.com/OpenGalaxea/GalaxeaVLA.git}"
: "${G05_REF:=main}"
: "${MODELSCOPE_DATASET_REPO:=RoboDojo-Benchmark/RoboDojo}"
: "${MODELSCOPE_CKPT_PREFIX:=ckpt/RoboDojo/G05/RoboDojo-sim-arx_x5-joint-0}"
: "${G05_CHECKPOINT_SOURCE:=modelscope}"
: "${HF_CHECKPOINT_REPO:=}"
: "${HF_G05_BASE_REPO:=OpenGalaxea/G05}"
: "${HF_G05_PROCESSOR_PATH:=qwen3_5_2b_base_processor}"
: "${HF_G05_ACTION_TOKENIZER_PATH:=action_tokenizer.pt}"
: "${G05_GPUS:=0}"
: "${G05_SAVE_INTERVAL_STEPS:=5000}"
: "${G05_KEEP_CHECKPOINTS:=1}"
: "${G05_AUTO_RESUME:=1}"
: "${G05_RUN_ID:=g05_robodojo_$(date +%Y%m%d_%H%M%S)}"
: "${G05_TRAIN_TASK:=real/g0plus_xpolicylab_finetune}"
: "${G05_DATASET_SOURCE:=huggingface}"

RUN_ROOT="${WORK_ROOT}/runs/${G05_RUN_ID}"
REPO_ROOT="${WORK_ROOT}/RoboDojo"
G05_ROOT="${WORK_ROOT}/G05"
DATA_ROOT="${G05_DATA_ROOT:-${WORK_ROOT}/data/lerobot_v30_joint}"
MODEL_ROOT="${WORK_ROOT}/models/robodojo"
ASSET_ROOT="${WORK_ROOT}/models/g05-assets"
VENV="${WORK_ROOT}/venv-g05"
PYTHON_310="${G05_PYTHON:-$(command -v python3.10 || true)}"
LOG="${RUN_ROOT}/logs/train.log"
mkdir -p "${RUN_ROOT}/logs" "${WORK_ROOT}/cache" "${MODEL_ROOT}" "${ASSET_ROOT}"
exec > >(tee -a "${LOG}") 2>&1

notify() {
  local title="$1" body="${2:-}" url=""
  if [[ "${SERVERCHAN_SENDKEY:-}" =~ ^sctp([0-9]+)t ]]; then
    url="https://${BASH_REMATCH[1]}.push.ft07.com/send/${SERVERCHAN_SENDKEY}.send"
  elif [[ -n "${SERVERCHAN_SENDKEY:-}" ]]; then
    url="https://sctapi.ftqq.com/${SERVERCHAN_SENDKEY}.send"
  fi
  if [[ -n "${url}" ]] && command -v curl >/dev/null 2>&1; then
    curl --fail --silent --show-error --max-time 20 --retry 3 -X POST "${url}" \
      -H 'Content-Type: application/x-www-form-urlencoded' \
      --data-urlencode "text=${title}" --data-urlencode "desp=${body}" >/dev/null || true
  else
    echo "[notify disabled] ${title}"
  fi
}

write_status() {
  local state="$1"
  printf '%s\n' "state=${state}" "run_id=${G05_RUN_ID}" "updated=$(date -Is)" \
    "log=${LOG}" > "${RUN_ROOT}/status.txt"
}

cleanup() {
  local rc=$?
  set +e
  [[ -z "${PRUNE_PID:-}" ]] || kill "${PRUNE_PID}" 2>/dev/null || true
  if (( rc == 0 )); then
    write_status success
    notify "G05训练完成" "Run: ${G05_RUN_ID}\nHF: https://huggingface.co/${HF_OUTPUT_REPO}"
  else
    write_status failed
    notify "G05训练失败" "Run: ${G05_RUN_ID}\nExit: ${rc}\nLog: ${LOG}"
  fi
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_status preparing
notify "G05 Vast训练启动" "Run: ${G05_RUN_ID}\nHost: $(hostname)\nGPUs: ${G05_GPUS}"

git_clone_or_update() {
  local url="$1" branch="$2" dest="$3"
  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" fetch --depth=1 origin "${branch}"
    git -C "${dest}" checkout -B "${branch}" "FETCH_HEAD"
  else
    rm -rf "${dest}"
    git clone --depth=1 --branch "${branch}" "${url}" "${dest}"
  fi
}

git_clone_or_update "${ROBO_DOJO_REPO_URL}" "${ROBO_DOJO_BRANCH}" "${REPO_ROOT}"
# The parent gitlink points at the fork commit, so clone the fork explicitly.
rm -rf "${REPO_ROOT}/XPolicyLab"
git clone --depth=1 --branch "${XPL_BRANCH}" "${XPL_REPO_URL}" "${REPO_ROOT}/XPolicyLab"
git_clone_or_update "${G05_REPO_URL}" "${G05_REF}" "${G05_ROOT}"

export HF_HOME="${WORK_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${WORK_ROOT}/cache/datasets"
export TRANSFORMERS_CACHE="${WORK_ROOT}/cache/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ -z "${PYTHON_310}" && -x "$(command -v conda 2>/dev/null || true)" ]]; then
  CONDA_PREFIX_G05="${WORK_ROOT}/conda-g05"
  if [[ ! -x "${CONDA_PREFIX_G05}/bin/python" ]]; then
    conda create -y -p "${CONDA_PREFIX_G05}" python=3.10.16
  fi
  PYTHON_310="${CONDA_PREFIX_G05}/bin/python"
fi
if [[ -z "${PYTHON_310}" ]]; then
  UV_BOOTSTRAP="${WORK_ROOT}/uv-bootstrap"
  if [[ ! -x "${UV_BOOTSTRAP}/bin/uv" ]]; then
    python3 -m venv "${UV_BOOTSTRAP}"
    "${UV_BOOTSTRAP}/bin/python" -m pip install --upgrade pip uv
  fi
  "${UV_BOOTSTRAP}/bin/uv" python install 3.10.16
  PYTHON_310="$("${UV_BOOTSTRAP}/bin/uv" python find 3.10.16)"
fi
[[ -x "${PYTHON_310}" ]] || {
  echo "Unable to provision Python 3.10.16 for G05. Set G05_PYTHON to a Python 3.10 executable." >&2
  exit 3
}

if [[ -x "${VENV}/bin/python" ]] && ! "${VENV}/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)'; then
  rm -rf "${VENV}"
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  "${PYTHON_310}" -m venv "${VENV}"
fi
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/pip" install --upgrade "huggingface_hub[cli]" modelscope wandb
if [[ -f "${G05_ROOT}/pyproject.toml" ]]; then
  "${VENV}/bin/pip" install -e "${G05_ROOT}"
elif [[ -f "${G05_ROOT}/GalaxeaVLA/pyproject.toml" ]]; then
  G05_ROOT="${G05_ROOT}/GalaxeaVLA"
  "${VENV}/bin/pip" install -e "${G05_ROOT}"
fi
export PATH="${VENV}/bin:${PATH}"
hf auth whoami >/dev/null

write_status downloading
hf download "${HF_DATASET_REPO}" --repo-type dataset --local-dir "${WORK_ROOT}/annotations"
export ROBODOJO_SIDECAR="${WORK_ROOT}/annotations/annotations/lerobot_v30_joint.sqlite3"
[[ -f "${ROBODOJO_SIDECAR}" ]] || ROBODOJO_SIDECAR="${WORK_ROOT}/annotations/lerobot_v30_joint.sqlite3"
[[ -f "${ROBODOJO_SIDECAR}" ]] || { echo "Sidecar not found" >&2; exit 4; }

if [[ "${G05_DATASET_SOURCE}" == "huggingface" && ! -f "${DATA_ROOT}/meta/info.json" ]]; then
  HF_DATA_ROOT="${WORK_ROOT}/robodojo-hf-data"
  hf download "${HF_DATASET_SOURCE_REPO}" --repo-type dataset \
    --include "${HF_DATASET_SOURCE_PATH}/**" --local-dir "${HF_DATA_ROOT}"
  DATA_ROOT="${HF_DATA_ROOT}/${HF_DATASET_SOURCE_PATH}"
elif [[ "${G05_DATASET_SOURCE}" == "modelscope" && ! -f "${DATA_ROOT}/meta/info.json" ]]; then
  if [[ -n "${MODELSCOPE_API_TOKEN:-}" ]]; then
    modelscope login --token "${MODELSCOPE_API_TOKEN}" || true
  fi
  modelscope download --dataset "${MODELSCOPE_DATASET_REPO}" \
    --include "data/lerobot_v30_joint/**" --local_dir "${WORK_ROOT}/robodojo-modelscope"
  rm -rf "${DATA_ROOT}"
  mkdir -p "${WORK_ROOT}/data"
  if [[ -d "${WORK_ROOT}/robodojo-modelscope/data/lerobot_v30_joint" ]]; then
    mv "${WORK_ROOT}/robodojo-modelscope/data/lerobot_v30_joint" "${DATA_ROOT}"
  fi
fi
[[ -f "${DATA_ROOT}/meta/info.json" ]] || {
  echo "LeRobot dataset missing at ${DATA_ROOT}. Check HF_DATASET_SOURCE_REPO and HF_DATASET_SOURCE_PATH." >&2
  exit 5
}

if [[ -n "${G05_INIT_CKPT:-}" && -d "${G05_INIT_CKPT}" ]]; then
  echo "[checkpoint] using user-provided G05_INIT_CKPT=${G05_INIT_CKPT}"
elif [[ ! -f "${MODEL_ROOT}/.downloaded" ]]; then
  if [[ "${G05_CHECKPOINT_SOURCE}" == "huggingface" ]]; then
    : "${HF_CHECKPOINT_REPO:?Set HF_CHECKPOINT_REPO when G05_CHECKPOINT_SOURCE=huggingface}"
    hf download "${HF_CHECKPOINT_REPO}" --repo-type model --local-dir "${MODEL_ROOT}"
  else
    if [[ -n "${MODELSCOPE_API_TOKEN:-}" ]]; then
      modelscope login --token "${MODELSCOPE_API_TOKEN}" || true
    fi
    modelscope download --dataset "${MODELSCOPE_DATASET_REPO}" \
      --include "${MODELSCOPE_CKPT_PREFIX}/**" --local_dir "${MODEL_ROOT}"
  fi
  touch "${MODEL_ROOT}/.downloaded"
fi
export G05_BASE_ASSETS="${ASSET_ROOT}"
if [[ ! -f "${ASSET_ROOT}/${HF_G05_PROCESSOR_PATH}/config.json" || ! -f "${ASSET_ROOT}/${HF_G05_ACTION_TOKENIZER_PATH}" ]]; then
  echo "[assets] downloading G05 processor and action tokenizer from ${HF_G05_BASE_REPO}"
  hf download "${HF_G05_BASE_REPO}" --repo-type model \
    --include "${HF_G05_PROCESSOR_PATH}/**" "${HF_G05_ACTION_TOKENIZER_PATH}" \
    --local-dir "${ASSET_ROOT}"
fi
export G05_ACTION_TOKENIZER_PATH="${ASSET_ROOT}/${HF_G05_ACTION_TOKENIZER_PATH}"
G05_PROCESSOR_DIR="${ASSET_ROOT}/${HF_G05_PROCESSOR_PATH}"
export G05_PROCESSOR_DIR
if [[ -z "${G05_INIT_CKPT:-}" ]]; then
  G05_INIT_CKPT="$(find "${MODEL_ROOT}" -type f \( -name model_state_dict.pt -o -name checkpoint.pt -o -name checkpoint \) | sort | head -1 || true)"
  [[ -n "${G05_INIT_CKPT}" ]] && G05_INIT_CKPT="$(dirname "${G05_INIT_CKPT}")"
fi
export G05_INIT_CKPT

if [[ "${G05_AUTO_RESUME}" == "1" && -z "${G05_RESUME:-}" ]]; then
  G05_RESUME="$(find "${WORK_ROOT}/runs" -type d \( -name 'step_*' -o -name 'global_step_*' -o -name 'checkpoint-*' \) ! -path "${RUN_ROOT}/*" | sort -V | tail -1 || true)"
  export G05_RESUME
  [[ -z "${G05_RESUME}" ]] || echo "[resume] automatically using ${G05_RESUME}"
fi

write_status preparing_data
export ROBODOJO_LEROBOT_V30_ROOT="${DATA_ROOT}"
export G05_ACTION_CHUNK="${G05_ACTION_CHUNK:-32}"
export G05_ARTIFACT_DIR="${RUN_ROOT}/artifacts"
export G05_SAMPLES_PER_EPOCH="${G05_SAMPLES_PER_EPOCH:-34744}"
bash "${REPO_ROOT}/scripts/g05/make_manifest.sh" "${G05_ARTIFACT_DIR}"
export G05_SIDECAR_JSONL="${G05_ARTIFACT_DIR}/subgoal_samples.jsonl"
export G05_SUBGOAL_MANIFEST="${G05_SIDECAR_JSONL}"
export G05_BALANCED_MANIFEST="${G05_ARTIFACT_DIR}/balanced_samples.jsonl"

export G05_ROOT
export G05_OUTPUT_ROOT="${RUN_ROOT}/outputs"
export G05_SAVE_INTERVAL_STEPS
export G05_TASK_CONFIG="${G05_TRAIN_TASK}"
export G05_GPUS
export GALAXEA_FM_OUTPUT_DIR="${G05_OUTPUT_ROOT}"
export GALAXEA_CKPT_RUN_ID="${G05_RUN_ID}"
export GALAXEA_FM_DATASET_STATS_CACHE_DIR="${WORK_ROOT}/cache/dataset_stats"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-g05-robodojo}"
mkdir -p "${G05_OUTPUT_ROOT}" "${GALAXEA_FM_DATASET_STATS_CACHE_DIR}"

# The RoboDojo checkpoint's Hydra config contains the publisher's absolute
# processor path. Override it with the processor downloaded above.
G05_TRAIN_ARGS="${G05_TRAIN_ARGS:-} model.model_arch.hf_processor_path=${G05_PROCESSOR_DIR}"
export G05_TRAIN_ARGS

prune_loop() {
  while true; do
    mapfile -t checkpoints < <(find "${G05_OUTPUT_ROOT}" -type d \( -name 'step_*' -o -name 'global_step_*' -o -name 'checkpoint-*' \) -mmin +2 | sort -V)
    if (( ${#checkpoints[@]} > G05_KEEP_CHECKPOINTS )); then
      local remove_count=$((${#checkpoints[@]} - G05_KEEP_CHECKPOINTS))
      for ((i=0; i<remove_count; i++)); do
        rm -rf "${checkpoints[i]}"
        echo "[prune] removed ${checkpoints[i]}"
      done
    fi
    sleep "${G05_PRUNE_INTERVAL:-60}"
  done
}

write_status training
notify "G05训练开始" "Run: ${G05_RUN_ID}\nInit: ${G05_INIT_CKPT}\nDataset: ${DATA_ROOT}"
prune_loop &
PRUNE_PID=$!
set +e
bash "${REPO_ROOT}/scripts/g05/run_train_vastai.sh" ${G05_TRAIN_ARGS:-}
train_rc=$?
set -e
kill "${PRUNE_PID}" 2>/dev/null || true
wait "${PRUNE_PID}" 2>/dev/null || true
(( train_rc == 0 )) || exit "${train_rc}"

write_status uploading
latest="$(find "${G05_OUTPUT_ROOT}" -type d \( -name 'step_*' -o -name 'global_step_*' -o -name 'checkpoint-*' \) | sort -V | tail -1)"
[[ -n "${latest}" && -d "${latest}" ]] || { echo "No final checkpoint found" >&2; exit 6; }
hf repos create "${HF_OUTPUT_REPO}" --type model --public --exist-ok
hf upload "${HF_OUTPUT_REPO}" "${latest}" "${G05_RUN_ID}/$(basename "${latest}")" \
  --repo-type model --commit-message "G05 training final checkpoint ${G05_RUN_ID}"
hf upload "${HF_OUTPUT_REPO}" "${RUN_ROOT}/artifacts" "${G05_RUN_ID}/artifacts" \
  --repo-type model --commit-message "G05 training manifest ${G05_RUN_ID}"
write_status complete
