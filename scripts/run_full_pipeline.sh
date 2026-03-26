#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSDF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="tsdf"
CONDA_EXE="${CONDA_EXE:-conda}"
PYTHON_BIN=""
SETUP_ENV=1
INSTALL_TOOLS=1
DOWNLOAD_TUM=1
DOWNLOAD_SCANOBJECTNN=1
DOWNLOAD_MODELNET40=0
RUN_RECON=1
RUN_MASKS=1
RUN_FUSE=1
RUN_TRAIN=1
CONFIG_PATH="${TSDF_ROOT}/configs/rgbd/tum/fr3_office.yaml"
TARGET_CLASS="keyboard"
MASK_MAX_FRAMES=100
MASK_FRAME_STRIDE=1
MASK_DEVICE="cuda"
TRAIN_MODELS="curvenet,pointnet++,pct,gdanet,dgcnn,pointnet"
TRAIN_DATASET_TYPE="scanobjectnn"
TRAIN_EPOCHS=150
TRAIN_BATCH_SIZE=32
TRAIN_WORKERS=4
TRAIN_DEVICE="cuda"
TRAIN_EXTRA_ARGS=()

log() {
  echo
  echo "[run_full_pipeline] $*"
}

print_help() {
  cat <<'EOF'
Usage:
  bash scripts/run_full_pipeline.sh [options] [-- extra training args]

Description:
  One-click pipeline for:
  1. creating/updating the tsdf environment
  2. installing helper packages and official model sources
  3. downloading datasets
  4. reconstructing a TUM RGB-D scene
  5. extracting keyboard masks and fusing a keyboard point cloud
  6. training multiple classifiers with skip-on-failure behavior

Examples:
  bash scripts/run_full_pipeline.sh
  bash scripts/run_full_pipeline.sh --epochs 1 --batch-size 8 --workers 0
  bash scripts/run_full_pipeline.sh --skip-env --skip-modelnet40 --models dgcnn,pointnet
  bash scripts/run_full_pipeline.sh --config configs/rgbd/tum/fr3_office.yaml --target-class keyboard

Options:
  --env-name NAME
  --python BIN
  --conda-exe BIN
  --config PATH
  --target-class NAME
  --mask-max-frames N
  --mask-frame-stride N
  --mask-device NAME
  --models LIST
  --dataset-type TYPE
  --epochs N
  --batch-size N
  --workers N
  --device NAME
  --skip-env
  --skip-install
  --skip-tum
  --skip-scanobjectnn
  --skip-modelnet40
  --download-modelnet40
  --skip-recon
  --skip-masks
  --skip-fuse
  --skip-train
  --help, -h

Notes:
  1. ModelNet40 download requires Kaggle CLI credentials.
  2. PointNet++ / PCT may still fail if CUDA toolkit and pointnet2_ops are unavailable.
     The training wrapper will skip failed models and continue.
  3. Extra args after `--` are forwarded to scripts/train_models.sh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --conda-exe)
      CONDA_EXE="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --target-class)
      TARGET_CLASS="$2"
      shift 2
      ;;
    --mask-max-frames)
      MASK_MAX_FRAMES="$2"
      shift 2
      ;;
    --mask-frame-stride)
      MASK_FRAME_STRIDE="$2"
      shift 2
      ;;
    --mask-device)
      MASK_DEVICE="$2"
      shift 2
      ;;
    --models)
      TRAIN_MODELS="$2"
      shift 2
      ;;
    --dataset-type)
      TRAIN_DATASET_TYPE="$2"
      shift 2
      ;;
    --epochs)
      TRAIN_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --workers)
      TRAIN_WORKERS="$2"
      shift 2
      ;;
    --device)
      TRAIN_DEVICE="$2"
      shift 2
      ;;
    --skip-env)
      SETUP_ENV=0
      shift
      ;;
    --skip-install)
      INSTALL_TOOLS=0
      shift
      ;;
    --skip-tum)
      DOWNLOAD_TUM=0
      shift
      ;;
    --skip-scanobjectnn)
      DOWNLOAD_SCANOBJECTNN=0
      shift
      ;;
    --skip-modelnet40)
      DOWNLOAD_MODELNET40=0
      shift
      ;;
    --download-modelnet40)
      DOWNLOAD_MODELNET40=1
      shift
      ;;
    --skip-recon)
      RUN_RECON=0
      shift
      ;;
    --skip-masks)
      RUN_MASKS=0
      shift
      ;;
    --skip-fuse)
      RUN_FUSE=0
      shift
      ;;
    --skip-train)
      RUN_TRAIN=0
      shift
      ;;
    --help|-h)
      print_help
      exit 0
      ;;
    --)
      shift
      TRAIN_EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help to see supported options."
      exit 1
      ;;
  esac
done

if [[ "${PYTHON_BIN}" == "" ]]; then
  if command -v "${CONDA_EXE}" >/dev/null 2>&1; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
    PYTHON_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

run_python() {
  "${PYTHON_BIN}" "$@"
}

maybe_setup_env() {
  if [[ "${SETUP_ENV}" -ne 1 ]]; then
    log "Skipping conda environment setup."
    return
  fi
  if ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
    log "Conda not found. Skipping environment creation."
    return
  fi
  if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    log "Creating conda environment ${ENV_NAME} from environment_py310.yml"
    "${CONDA_EXE}" env create -n "${ENV_NAME}" -f "${TSDF_ROOT}/environment_py310.yml"
  else
    log "Updating conda environment ${ENV_NAME} from environment_py310.yml"
    "${CONDA_EXE}" env update -n "${ENV_NAME}" -f "${TSDF_ROOT}/environment_py310.yml" --prune
  fi
}

maybe_install_tools() {
  if [[ "${INSTALL_TOOLS}" -ne 1 ]]; then
    log "Skipping tool installation."
    return
  fi

  log "Installing common Python helpers"
  run_python -m pip install --upgrade pip
  run_python -m pip install tqdm kaggle

  if [[ ! -d "${TSDF_ROOT}/third_party/ModelNet40-C/.git" ]]; then
    log "Cloning ModelNet40-C"
    git clone https://github.com/jiachens/ModelNet40-C.git "${TSDF_ROOT}/third_party/ModelNet40-C"
  else
    log "ModelNet40-C already exists. Skipping clone."
  fi

  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    log "CUDA toolkit detected at ${CUDA_HOME}; attempting pointnet2_ops install"
    (
      cd "${TSDF_ROOT}/third_party/ModelNet40-C/PCT_Pytorch/pointnet2_ops_lib"
      run_python setup.py install || true
    )
  else
    log "CUDA toolkit not detected. Skipping pointnet2_ops install."
  fi
}

maybe_download_tum() {
  if [[ "${DOWNLOAD_TUM}" -ne 1 ]]; then
    log "Skipping TUM sample download."
    return
  fi
  log "Downloading TUM RGB-D samples"
  run_python "${TSDF_ROOT}/dataset/download_tum_rgbd_samples.py"
}

maybe_download_scanobjectnn() {
  if [[ "${DOWNLOAD_SCANOBJECTNN}" -ne 1 ]]; then
    log "Skipping ScanObjectNN download."
    return
  fi
  log "Downloading ScanObjectNN"
  run_python "${TSDF_ROOT}/dataset/download_scanobjectnn.py"
}

maybe_download_modelnet40() {
  if [[ "${DOWNLOAD_MODELNET40}" -ne 1 ]]; then
    log "Skipping ModelNet40 download."
    return
  fi
  log "Downloading ModelNet40 from Kaggle"
  run_python "${TSDF_ROOT}/dataset/download_modelnet40_kaggle.py"
}

maybe_reconstruct_scene() {
  if [[ "${RUN_RECON}" -ne 1 ]]; then
    log "Skipping TUM TSDF reconstruction."
    return
  fi
  log "Running TUM TSDF reconstruction"
  run_python "${TSDF_ROOT}/3d_construction/run_tum_rgbd_tsdf.py" \
    --config "${CONFIG_PATH}"
}

maybe_generate_masks() {
  if [[ "${RUN_MASKS}" -ne 1 ]]; then
    log "Skipping keyboard mask generation."
    return
  fi
  log "Generating ${TARGET_CLASS} masks from TUM RGB-D"
  run_python "${TSDF_ROOT}/mask_generation/generate_tum_masks_maskrcnn.py" \
    --config "${CONFIG_PATH}" \
    --target-class "${TARGET_CLASS}" \
    --separate-instances \
    --save-preview \
    --max-frames "${MASK_MAX_FRAMES}" \
    --frame-stride "${MASK_FRAME_STRIDE}" \
    --device "${MASK_DEVICE}"
}

maybe_fuse_object() {
  if [[ "${RUN_FUSE}" -ne 1 ]]; then
    log "Skipping masked object fusion."
    return
  fi
  config_stem="$(basename "${CONFIG_PATH}")"
  config_stem="${config_stem%.*}"
  mask_output="${TSDF_ROOT}/mask_generation/outputs/${config_stem}_${TARGET_CLASS}"
  log "Fusing ${TARGET_CLASS} point cloud from masks in ${mask_output}"
  run_python "${TSDF_ROOT}/3d_construction/fuse_tum_mask_object_pcd.py" \
    --config "${CONFIG_PATH}" \
    --mask-dir "${mask_output}/masks" \
    --mask-list "${mask_output}/mask.txt" \
    --largest-component \
    --voxel-downsample 0.005 \
    --remove-statistical-outlier
}

maybe_train_models() {
  if [[ "${RUN_TRAIN}" -ne 1 ]]; then
    log "Skipping multi-model training."
    return
  fi
  log "Training models: ${TRAIN_MODELS}"
  bash "${TSDF_ROOT}/scripts/train_models.sh" \
    --python "${PYTHON_BIN}" \
    --models "${TRAIN_MODELS}" \
    --dataset-type "${TRAIN_DATASET_TYPE}" \
    --scanobjectnn-root "${TSDF_ROOT}/data/ScanObjectNN" \
    --epochs "${TRAIN_EPOCHS}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --workers "${TRAIN_WORKERS}" \
    --device "${TRAIN_DEVICE}" \
    "${TRAIN_EXTRA_ARGS[@]}"
}

log "Repo root: ${TSDF_ROOT}"
log "Python: ${PYTHON_BIN}"

maybe_setup_env
maybe_install_tools
maybe_download_tum
maybe_download_scanobjectnn
maybe_download_modelnet40
maybe_reconstruct_scene
maybe_generate_masks
maybe_fuse_object
maybe_train_models

log "Full pipeline finished."
