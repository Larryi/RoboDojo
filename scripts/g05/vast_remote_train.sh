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
: "${G05_CHECKPOINT_SOURCE:=huggingface}"
: "${HF_CHECKPOINT_REPO:=RoboDojo-Benchmark/RoboDojo}"
: "${HF_CHECKPOINT_PATH:=${MODELSCOPE_CKPT_PREFIX}}"
: "${HF_G05_BASE_REPO:=OpenGalaxea/G05}"
: "${HF_G05_PROCESSOR_PATH:=qwen3_5_2b_base_processor}"
: "${HF_G05_ACTION_TOKENIZER_PATH:=action_tokenizer.pt}"
: "${G05_GPUS:=0}"
: "${G05_TORCH_VERSION:=2.7.1}"
: "${G05_TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu128}"
: "${G05_SAVE_INTERVAL_STEPS:=5000}"
: "${G05_KEEP_CHECKPOINTS:=1}"
: "${G05_AUTO_RESUME:=1}"
: "${G05_RUN_ID:=g05_robodojo_$(date +%Y%m%d_%H%M%S)}"
: "${G05_TRAIN_TASK:=robodojo_g05}"
: "${G05_USE_SIDECAR:=0}"
: "${G05_TORCH_COMPILE:=0}"
: "${G05_BATCH_SIZE:=10}"
# Older secrets files may still carry the legacy PaliGemma task. It is not
# compatible with the Qwen3.5 RoboDojo checkpoint; transparently migrate it.
if [[ "${G05_TRAIN_TASK}" == "real/g0plus_xpolicylab_finetune" ]]; then
  echo "[config] replacing legacy G0Plus task with native RoboDojo G05 task"
  G05_TRAIN_TASK="robodojo_g05"
fi
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
: "${HF_HUB_ENABLE_HF_TRANSFER:=1}"
export HF_HUB_ENABLE_HF_TRANSFER
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
DEPS_MARKER="${VENV}/.g05_deps_ready"
if [[ -f "${G05_ROOT}/GalaxeaVLA/pyproject.toml" ]]; then
  G05_ROOT="${G05_ROOT}/GalaxeaVLA"
fi
if [[ -f "${DEPS_MARKER}" ]] && ! "${VENV}/bin/python" - "${G05_TORCH_VERSION}" <<'PY'
import sys
import torch
raise SystemExit(0 if torch.__version__.split("+")[0] == sys.argv[1] else 1)
PY
then
  echo "[deps] Torch version mismatch; reinstalling the G05 dependency set"
  rm -f "${DEPS_MARKER}"
fi
if [[ ! -f "${DEPS_MARKER}" ]]; then
  "${VENV}/bin/python" -m pip install --upgrade pip
  "${VENV}/bin/pip" install --upgrade "huggingface_hub[cli]" modelscope wandb
  if [[ "${HF_HUB_ENABLE_HF_TRANSFER}" == "1" ]]; then
    if ! "${VENV}/bin/pip" install --upgrade hf_transfer; then
      echo "[hf] hf_transfer unavailable; falling back to standard Hugging Face downloads" >&2
      export HF_HUB_ENABLE_HF_TRANSFER=0
    fi
  fi
if [[ -f "${G05_ROOT}/pyproject.toml" ]]; then
  "${VENV}/bin/pip" install -e "${G05_ROOT}"
fi
  # RTX PRO 6000 Blackwell is sm_120. Install the CUDA 12.8 wheel family.
  "${VENV}/bin/pip" install --upgrade \
    "torch==${G05_TORCH_VERSION}" \
    --index-url "${G05_TORCH_INDEX_URL}"
  touch "${DEPS_MARKER}"
else
  echo "[deps] reusing ${VENV}; dependency installation already completed"
fi
# The released G05 V3 loader currently has two compatibility issues with this
# dataset: its in-memory path leaves hf_dataset unset while video timestamp
# queries still dereference it, and video_backend is accepted by the config
# but not forwarded from BaseLerobotDataset to MultiLeRobotDataset. The
# manifest-aware V3 loader also exposes a logical subset length while its
# parent loader still needs to read the original global frame index. Patch the
# cloned source idempotently at runtime so the launcher remains self-contained.
"${VENV}/bin/python" - "${G05_ROOT}/src/g05/data/base_lerobot_dataset.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old_param = "        load_images: Optional[bool] = None,\n        in_memory: bool = False,\n"
new_param = "        load_images: Optional[bool] = None,\n        video_backend: Optional[str] = None,\n        in_memory: bool = False,\n"
if "video_backend: Optional[str] = None" not in text:
    if old_param not in text:
        raise SystemExit(f"Cannot patch video_backend parameter in {path}")
    text = text.replace(old_param, new_param, 1)
old_call = "                load_images=self.load_images,\n                in_memory=self.in_memory,\n"
new_call = "                load_images=self.load_images,\n                video_backend=video_backend,\n                in_memory=self.in_memory,\n"
if "                video_backend=video_backend,\n" not in text:
    if old_call not in text:
        raise SystemExit(f"Cannot patch video_backend forwarding in {path}")
    text = text.replace(old_call, new_call, 1)

# A manifest-aware subclass can expose len(self) == len(manifest), while its
# __getitem__ translates a manifest position to an original global frame
# index. The base loader must therefore validate/retry against the physical
# dataset length, not the subclass's logical length.
if "physical_len = self.multi_dataset.num_frames" not in text:
    old_check = "        if idx >= len(self):\n"
    old_error = "            raise IndexError(f\"Index {idx} out of bounds {len(self)}.\")\n"
    if old_check not in text or old_error not in text:
        raise SystemExit(f"Cannot patch manifest index bounds in {path}")
    text = text.replace(
        old_check,
        "        physical_len = self.multi_dataset.num_frames\n"
        "        if idx < 0 or idx >= physical_len:\n",
        1,
    )
    text = text.replace(
        old_error,
        "            raise IndexError(f\"Index {idx} out of bounds {physical_len}.\")\n",
        1,
    )

old_retry = "sample_idx = np.random.randint(len(self))"
new_retry = "sample_idx = np.random.randint(physical_len)"
if old_retry in text:
    text = text.replace(old_retry, new_retry, 1)
old_retry_base = "sample_idx = np.random.randint(BaseLerobotDataset.__len__(self))"
if old_retry_base in text:
    text = text.replace(old_retry_base, new_retry, 1)

old_index_translation = """        else:
            sample_idx = idx + self._start_idx
"""
new_index_translation = """        else:
            if getattr(self, "_manifest_global_index_mode", False):
                sample_idx = idx
            else:
                sample_idx = idx + self._start_idx
"""
if new_index_translation not in text:
    if old_index_translation not in text:
        raise SystemExit(f"Cannot patch manifest global index translation in {path}")
    text = text.replace(old_index_translation, new_index_translation, 1)

path.write_text(text)
print(f"[loader] patched {path}")
PY
"${VENV}/bin/python" - "${G05_ROOT}/src/g05/data/galaxea_lerobot_dataset.py" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
if "self._sidecar_rows = {}" not in text:
    text = text.replace("import torch\n", "import json\nfrom pathlib import Path\n\nimport torch\n", 1)
    old_params = "        tolerance_s: Optional[float] = None,\n        **kwargs,\n"
    new_params = "        tolerance_s: Optional[float] = None,\n        subgoal_manifest: Optional[str] = None,\n        balanced_manifest: Optional[str] = None,\n        preserve_global_task: bool = True,\n        action_chunk_boundary: Optional[str] = None,\n        **kwargs,\n"
    if old_params not in text:
        raise SystemExit(f"Cannot add subgoal parameters in {path}")
    text = text.replace(old_params, new_params, 1)
    marker = "        self._future_task_offset = future_task_offset\n"
    init = ("        self.preserve_global_task = preserve_global_task\n"
            "        self.action_chunk_boundary = action_chunk_boundary\n"
            "        self._sidecar_rows = {}\n"
            "        self._manifest_global_indices = None\n"
            "        manifest_path = balanced_manifest if is_training_set and balanced_manifest else subgoal_manifest\n"
            "        if manifest_path:\n"
            "            self._load_subgoal_manifest(Path(manifest_path))\n")
    if marker not in text:
        raise SystemExit(f"Cannot initialize subgoal manifest in {path}")
    text = text.replace(marker, init + marker, 1)
    old_len = ("    def __len__(self):\n"
               "        if hasattr(self, \"_overfit_len\"):\n"
               "            return self._overfit_len\n")
    new_len = ("    def __len__(self):\n"
               "        if self._manifest_global_indices is not None:\n"
               "            return len(self._manifest_global_indices)\n"
               "        if hasattr(self, \"_overfit_len\"):\n"
               "            return self._overfit_len\n")
    if old_len not in text:
        raise SystemExit(f"Cannot patch subgoal dataset length in {path}")
    text = text.replace(old_len, new_len, 1)
    old_additional = ("    def _get_additional_data(self, sample, lerobot_sample):\n"
                      "        sample[\"coarse_task\"] = lerobot_sample[\"coarse_task\"]\n"
                      "        return sample\n")
    new_additional = ("    def _get_additional_data(self, sample, lerobot_sample):\n"
                      "        global_task = lerobot_sample.get(\"coarse_task\", lerobot_sample.get(\"task\", \"\"))\n"
                      "        sample[\"coarse_task\"] = global_task\n"
                      "        sample[\"_is_subgoal\"] = False\n"
                      "        if self._sidecar_rows:\n"
                      "            episode_index = lerobot_sample.get(\"episode_index\")\n"
                      "            frame_index = lerobot_sample.get(\"frame_index\")\n"
                      "            if hasattr(episode_index, \"item\"):\n"
                      "                episode_index = episode_index.item()\n"
                      "            if hasattr(frame_index, \"item\"):\n"
                      "                frame_index = frame_index.item()\n"
                      "            row = self._sidecar_rows.get((int(episode_index), int(frame_index)))\n"
                      "            if row:\n"
                      "                if self.preserve_global_task:\n"
                      "                    sample[\"coarse_task\"] = row[\"task\"]\n"
                      "                sample[\"task\"] = row[\"task\"]\n"
                      "                sample[\"atomic_task\"] = row[\"subtask\"].removeprefix(\"Subtask: \").strip()\n"
                      "                sample[\"_is_subgoal\"] = bool(row.get(\"is_subgoal\", row.get(\"subtask\", \"\") != row.get(\"task\", \"\")))\n"
                      "                sample[\"_subgoal_action_horizon\"] = int(row.get(\"action_horizon\", 0))\n"
                      "                if self.action_chunk_boundary == \"segment\" and \"action_is_pad\" in sample:\n"
                      "                    horizon = max(0, min(int(row[\"action_horizon\"]), sample[\"action_is_pad\"].shape[0]))\n"
                      "                    sample[\"action_is_pad\"] = sample[\"action_is_pad\"].clone()\n"
                      "                    sample[\"action_is_pad\"][horizon:] = True\n"
                      "        return sample\n")
    if old_additional not in text:
        raise SystemExit(f"Cannot patch subgoal task data in {path}")
    text = text.replace(old_additional, new_additional, 1)
    marker = "    def _get_ee_start_moving_step_of_episode(self, episode_idx: int) -> int:\n"
    methods = ("    def _load_subgoal_manifest(self, path: Path):\n"
               "        if not path.is_file():\n"
               "            raise FileNotFoundError(f\"G0.5 subgoal manifest not found: {path}\")\n"
               "        rows = []\n"
               "        with path.open(encoding=\"utf-8\") as handle:\n"
               "            for line in handle:\n"
               "                row = json.loads(line)\n"
               "                if \"episode_index\" in row and \"frame_index\" in row:\n"
               "                    rows.append(row)\n"
               "        episode_count = len(self.episode_data_index[\"from\"])\n"
               "        train_cutoff = int(episode_count * (1.0 - self.val_set_proportion))\n"
               "        selected = []\n"
               "        for source_row in rows:\n"
               "            row = dict(source_row)\n"
               "            episode = int(row[\"episode_index\"])\n"
               "            frame = int(row[\"frame_index\"])\n"
               "            if episode < 0 or episode >= episode_count:\n"
               "                continue\n"
               "            if (episode < train_cutoff) != self.is_training_set:\n"
               "                continue\n"
               "            self._sidecar_rows[(episode, frame)] = row\n"
               "            selected.append(int(self.episode_data_index[\"from\"][episode]) + frame)\n"
               "        if not selected:\n"
               "            split = \"training\" if self.is_training_set else \"validation\"\n"
               "            raise RuntimeError(f\"No {split} samples in {path}\")\n"
               "        self._manifest_global_indices = selected\n"
               "        print(f\"[subgoal] loaded {path}: {len(selected)} samples\")\n\n")
    if marker not in text:
        raise SystemExit(f"Cannot insert subgoal manifest loader in {path}")
    text = text.replace(marker, methods + marker, 1)
    old_getitem = ("    def __getitem__(self, idx):\n"
                   "        if idx >= len(self):\n"
                   "            raise IndexError(f\"Index {idx} out of bounds.\")\n\n"
                   "        if hasattr(self, \"_overfit_indices\"):\n"
                   "            return super().__getitem__(idx)\n")
    new_getitem = ("    def __getitem__(self, idx):\n"
                   "        if idx < 0 or idx >= len(self):\n"
                   "            raise IndexError(f\"Index {idx} out of bounds {len(self)}.\")\n\n"
                   "        if self._manifest_global_indices is not None:\n"
                   "            original_idx = self._manifest_global_indices[idx]\n"
                   "            self._manifest_global_index_mode = True\n"
                   "            try:\n"
                   "                return super().__getitem__(original_idx)\n"
                   "            finally:\n"
                   "                self._manifest_global_index_mode = False\n\n"
                   "        if hasattr(self, \"_overfit_indices\"):\n"
                   "            return super().__getitem__(idx)\n")
    if old_getitem not in text:
        raise SystemExit(f"Cannot patch subgoal item mapping in {path}")
    text = text.replace(old_getitem, new_getitem, 1)
old_oracle_conditioning = "                sample[\"task\"] = row[\"subtask\"]\n"
new_hierarchical_conditioning = ("                sample[\"task\"] = row[\"task\"]\n"
                                 "                sample[\"atomic_task\"] = row[\"subtask\"].removeprefix(\"Subtask: \").strip()\n")
if old_oracle_conditioning in text:
    text = text.replace(old_oracle_conditioning, new_hierarchical_conditioning, 1)
if "sample[\"_is_subgoal\"]" not in text:
    marker = "        sample[\"coarse_task\"] = global_task\n"
    replacement = marker + "        sample[\"_is_subgoal\"] = False\n"
    if marker not in text:
        raise SystemExit(f"Cannot add subgoal sample marker in {path}")
    text = text.replace(marker, replacement, 1)
    marker = "                sample[\"atomic_task\"] = row[\"subtask\"].removeprefix(\"Subtask: \").strip()\n"
    replacement = (marker
                   + "                sample[\"_is_subgoal\"] = bool(row.get(\"is_subgoal\", row.get(\"subtask\", \"\") != row.get(\"task\", \"\")))\n"
                   + "                sample[\"_subgoal_action_horizon\"] = int(row.get(\"action_horizon\", 0))\n")
    if marker not in text:
        raise SystemExit(f"Cannot add subgoal sample metadata in {path}")
    text = text.replace(marker, replacement, 1)
path.write_text(text)
print(f"[subgoal] patched {path}")
PY
"${VENV}/bin/python" - "${G05_ROOT}/src/g05/models/g05/g05_policy.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "        loss_value_dict = {k: v.detach() for k, v in loss_dict.items()}\n"
new = ("        loss_value_dict = {k: v.detach() for k, v in loss_dict.items()}\n"
       "        # Explicit W&B accounting for samples selected by the subgoal manifest.\n"
       "        # The aggregate loss above already includes every sample; these fields\n"
       "        # expose the denominator and subgoal coverage without affecting gradients.\n"
       "        _subgoal_flags = [bool(s.get(\"_is_subgoal\", False)) for s in samples]\n"
       "        _subgoal_count = sum(_subgoal_flags)\n"
       "        _subgoal_fraction = _subgoal_count / max(1, len(_subgoal_flags))\n"
       "        _subgoal_horizons = [int(s.get(\"_subgoal_action_horizon\", 0)) for s, flag in zip(samples, _subgoal_flags) if flag]\n"
       "        loss_value_dict[\"train/subgoal_samples\"] = torch.tensor(float(_subgoal_count), device=device)\n"
       "        loss_value_dict[\"train/total_samples\"] = torch.tensor(float(len(_subgoal_flags)), device=device)\n"
       "        loss_value_dict[\"train/subgoal_fraction\"] = torch.tensor(float(_subgoal_fraction), device=device)\n"
       "        loss_value_dict[\"train/subgoal_action_horizon_mean\"] = torch.tensor(float(sum(_subgoal_horizons) / len(_subgoal_horizons)) if _subgoal_horizons else 0.0, device=device)\n")
if "train/subgoal_fraction" not in text:
    if old not in text:
        raise SystemExit(f"Cannot add subgoal W&B metrics in {path}")
    text = text.replace(old, new, 1)
    path.write_text(text)
print(f"[metrics] subgoal W&B metrics patched {path}")
PY
"${VENV}/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"[torch] {torch.__version__} CUDA={torch.version.cuda} GPU={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")'
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
    hf download "${HF_CHECKPOINT_REPO}" --repo-type dataset \
      --include "${HF_CHECKPOINT_PATH}/**" --local-dir "${MODEL_ROOT}"
  else
    if [[ -n "${MODELSCOPE_API_TOKEN:-}" ]]; then
      modelscope login --token "${MODELSCOPE_API_TOKEN}" || true
    fi
    modelscope download --dataset "${MODELSCOPE_DATASET_REPO}" \
      --include "${MODELSCOPE_CKPT_PREFIX}/**" --local_dir "${MODEL_ROOT}"
  fi
  touch "${MODEL_ROOT}/.downloaded"
fi
CHECKPOINT_CONFIG="${MODEL_ROOT}/${HF_CHECKPOINT_PATH}/.hydra/config.yaml"
[[ -f "${CHECKPOINT_CONFIG}" ]] || { echo "G05 checkpoint Hydra config missing: ${CHECKPOINT_CONFIG}" >&2; exit 8; }
mkdir -p "${G05_ROOT}/configs/task"
{
  echo "# @package _global_"
  cat "${CHECKPOINT_CONFIG}"
} > "${G05_ROOT}/configs/task/robodojo_g05.yaml"
# The published config contains the original machine-local dataset path.
sed -i \
  "s#/personal/tianxing/RoboDojo/data/RoboDojo_lerobot_v30_video#${DATA_ROOT}#g" \
  "${G05_ROOT}/configs/task/robodojo_g05.yaml"
G05_VIDEO_BACKEND="${G05_VIDEO_BACKEND:-pyav}"
if grep -Eq "^[[:space:]]+video_backend:" "${G05_ROOT}/configs/task/robodojo_g05.yaml"; then
  sed -i -E "s/^([[:space:]]+video_backend:)[[:space:]]*.*/\\1 ${G05_VIDEO_BACKEND}/" \
    "${G05_ROOT}/configs/task/robodojo_g05.yaml"
else
  sed -i "/^[[:space:]]*lerobot_ds_version:[[:space:]]*['\"]\?3\.0['\"]\?$/a\\      video_backend: ${G05_VIDEO_BACKEND}" \
    "${G05_ROOT}/configs/task/robodojo_g05.yaml"
fi
echo "[dataset] video_backend=$(awk '/^[[:space:]]+video_backend:/{print $2; exit}' "${G05_ROOT}/configs/task/robodojo_g05.yaml")"
# The published config enables in_memory, but G05 V3 video timestamp queries
# are not compatible with that mode. Keep the parquet mmap path instead.
sed -i -E 's/^  in_memory:[[:space:]]*true$/  in_memory: false/' \
  "${G05_ROOT}/configs/task/robodojo_g05.yaml"
echo "[dataset] in_memory=$(awk '/^  in_memory:/{print $2; exit}' "${G05_ROOT}/configs/task/robodojo_g05.yaml")"
# FLA's Triton gated-delta kernel currently fails to lower for Blackwell
# (sm_120) with the installed Torch/Triton pair. G05 ships a numerically
# compatible pure-PyTorch implementation; use it unless explicitly changed.
G05_LINEAR_ATTN_BACKEND="${G05_LINEAR_ATTN_BACKEND:-torch}"
sed -i -E "s/(^[[:space:]]*linear_attn_backend:)[[:space:]]*.*/\\1 ${G05_LINEAR_ATTN_BACKEND}/" \
  "${G05_ROOT}/configs/task/robodojo_g05.yaml"
echo "[model] linear_attn_backend=$(awk '/^[[:space:]]+linear_attn_backend:/{print $2; exit}' "${G05_ROOT}/configs/task/robodojo_g05.yaml")"
# The published task config leaves eval_steps null, but finetune.py performs
# modulo arithmetic on it after the first optimizer steps.
if grep -Eq '^eval_steps:[[:space:]]*null[[:space:]]*$' "${G05_ROOT}/configs/task/robodojo_g05.yaml"; then
  sed -i -E 's/^eval_steps:[[:space:]]*null[[:space:]]*$/eval_steps: 1000/' \
    "${G05_ROOT}/configs/task/robodojo_g05.yaml"
fi
echo "[train] eval_steps=$(awk '/^eval_steps:/{print $2; exit}' "${G05_ROOT}/configs/task/robodojo_g05.yaml")"
"${VENV}/bin/python" - "${G05_ROOT}/configs/task/robodojo_g05.yaml" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

path = Path(sys.argv[1])
cfg = OmegaConf.load(path)
OmegaConf.set_struct(cfg, False)
cfg.model.model_arch.predict_cot = True
cfg.model.model_arch.action_attend_cot = True
cfg.model.model_arch.discrete_action = False
cfg.model.model_arch.continuous_action = True
cfg.model.model_arch.return_continuous_action = True
cfg.model.model_arch.ar.ce_weight = 1.0
cfg.model.processor.drop_high_level_prob = 0.0
cfg.model.processor.samples_builder = {
    "_target_": "g05.data_processor.processor.samples_builder.SubtaskCoTBuilderFMOnly",
    "num_input_images": "${model.model_arch.num_input_images}",
    "image_sizes": "${model.processor.camera_size_config}",
}
path.write_text("# @package _global_\n\n" + OmegaConf.to_yaml(cfg), encoding="utf-8")
print("[subgoal] enabled main-task -> predicted-subgoal -> FM-action training")
PY
"${VENV}/bin/python" - "${G05_ROOT}/configs/task/robodojo_g05.yaml" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(Path(sys.argv[1]))
required = (
    "resume_ckpt",
    "checkpointing_steps",
    "model.batch_size",
    "model.grad_accumulation_steps",
    "model.pretrained_ckpt",
    "model.model_arch.hf_processor_path",
    "model.model_arch.predict_cot",
    "model.model_arch.action_attend_cot",
    "model.model_arch.ar.ce_weight",
    "model.processor.drop_high_level_prob",
    "model.processor.samples_builder._target_",
)
missing = []
for path in required:
    node = cfg
    for part in path.split("."):
        if part not in node:
            missing.append(path)
            break
        node = node[part]
if missing:
    raise SystemExit("G05 checkpoint config missing expected keys: " + ", ".join(missing))
print("[config] RoboDojo G05 checkpoint schema validated")
PY
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
if [[ -n "${G05_INIT_CKPT:-}" && -d "${G05_INIT_CKPT}" ]]; then
  G05_INIT_CKPT="$(find "${G05_INIT_CKPT}" -type f \( -name model_state_dict.pt -o -name checkpoint.pt -o -name checkpoint -o -name model.pt \) | sort | head -1 || true)"
fi
if [[ -z "${G05_INIT_CKPT:-}" ]]; then
  G05_INIT_CKPT="$(find "${MODEL_ROOT}" -type f \( -name model_state_dict.pt -o -name checkpoint.pt -o -name checkpoint \) | sort | head -1 || true)"
fi
[[ -f "${G05_INIT_CKPT:-}" ]] || { echo "G05 checkpoint file not found: ${G05_INIT_CKPT:-<empty>}" >&2; exit 9; }
echo "[checkpoint] using file ${G05_INIT_CKPT}"
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
G05_TRAIN_ARGS="${G05_TRAIN_ARGS:-}"
# Migrate only the resume key from older launchers; batch size and gradient
# accumulation belong under model in the published RoboDojo checkpoint config.
G05_TRAIN_ARGS="${G05_TRAIN_ARGS//checkpoint.resume/resume_ckpt}"
# The published checkpoint leaves both max_epochs and max_steps null, but the
# released finetune.py requires one of them to build the LR scheduler.
if [[ " ${G05_TRAIN_ARGS} " != *" model.max_steps="* && " ${G05_TRAIN_ARGS} " != *" model.max_epochs="* ]]; then
  G05_MAX_STEPS="${G05_MAX_STEPS:-100000}"
  G05_TRAIN_ARGS+=" model.max_steps=${G05_MAX_STEPS}"
fi
G05_TRAIN_ARGS+=" model.model_arch.hf_processor_path=${G05_PROCESSOR_DIR}"
# Native RoboDojo G05 uses the embodiment_datasets namespace. The leading
# '+' is required because these fields are absent from the published config's
# structured schema. Keep these overrides here so subgoal is enabled even
# when the legacy XPolicyLab sidecar switch is disabled.
G05_TRAIN_ARGS+=" +data.embodiment_datasets.robodojo.subgoal_manifest=${G05_SUBGOAL_MANIFEST}"
G05_TRAIN_ARGS+=" +data.embodiment_datasets.robodojo.balanced_manifest=${G05_BALANCED_MANIFEST}"
G05_TRAIN_ARGS+=" +data.embodiment_datasets.robodojo.preserve_global_task=true"
G05_TRAIN_ARGS+=" +data.embodiment_datasets.robodojo.action_chunk_boundary=segment"
# Throughput/LR settings validated for the Blackwell 96 GiB instance.
G05_TRAIN_ARGS+=" model.batch_size=${G05_BATCH_SIZE}"
G05_TRAIN_ARGS+=" model.learning_rate=0.0001"
G05_TRAIN_ARGS+=" model.lr_min_ratio=0.1"
G05_TRAIN_ARGS+=" model.constant_end_ratio=0.9"
G05_TRAIN_ARGS+=" model.num_workers=16"
G05_TRAIN_ARGS+=" model.prefetch_factor=4"
if [[ "${G05_TORCH_COMPILE}" == "1" ]]; then
  G05_TRAIN_ARGS+=" model.use_torch_compile=true"
fi
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
