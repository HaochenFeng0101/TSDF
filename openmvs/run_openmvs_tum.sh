#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSDF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${TSDF_ROOT}/configs/rgbd/tum/fr3_office.yaml"
WORKSPACE_NAME=""
OPENMVS_BIN_DIR="${OPENMVS_BIN:-}"
REEXPORT=0
FRAME_STRIDE=5
MAX_FRAMES=500
MAX_DT=0.08
INPUT_WIDTH=""
INPUT_HEIGHT=""
RESOLUTION_LEVEL=1
MAX_RESOLUTION=2560
MIN_RESOLUTION=640
NUMBER_VIEWS=5
NUMBER_VIEWS_FUSE=2
ITERS=3
GEOMETRIC_ITERS=2
MAX_THREADS=8
REMOVE_DMAPS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --workspace-name)
      WORKSPACE_NAME="$2"
      shift 2
      ;;
    --reexport)
      REEXPORT=1
      shift
      ;;
    --frame-stride)
      FRAME_STRIDE="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --max-dt)
      MAX_DT="$2"
      shift 2
      ;;
    --input-width)
      INPUT_WIDTH="$2"
      shift 2
      ;;
    --input-height)
      INPUT_HEIGHT="$2"
      shift 2
      ;;
    --resolution-level)
      RESOLUTION_LEVEL="$2"
      shift 2
      ;;
    --max-resolution)
      MAX_RESOLUTION="$2"
      shift 2
      ;;
    --min-resolution)
      MIN_RESOLUTION="$2"
      shift 2
      ;;
    --number-views)
      NUMBER_VIEWS="$2"
      shift 2
      ;;
    --number-views-fuse)
      NUMBER_VIEWS_FUSE="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --geometric-iters)
      GEOMETRIC_ITERS="$2"
      shift 2
      ;;
    --max-threads)
      MAX_THREADS="$2"
      shift 2
      ;;
    --keep-dmaps)
      REMOVE_DMAPS=0
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage:
  bash openmvs/run_openmvs_tum.sh [options]

Options:
  --config PATH
  --workspace-name NAME
  --reexport
  --frame-stride N
  --max-frames N
  --max-dt FLOAT
  --input-width N
  --input-height N
  --resolution-level N
  --max-resolution N
  --min-resolution N
  --number-views N
  --number-views-fuse N
  --iters N
  --geometric-iters N
  --max-threads N
  --keep-dmaps

Examples:
  bash openmvs/run_openmvs_tum.sh --workspace-name fr3_office_openmvs
  bash openmvs/run_openmvs_tum.sh --workspace-name fr3_office_openmvs --reexport --frame-stride 5 --max-frames 80
  bash openmvs/run_openmvs_tum.sh --workspace-name fr3_office_openmvs --resolution-level 0 --number-views 7
EOF
      exit 0
      ;;
    *)
      if [[ "$1" != --* && "${CONFIG_PATH}" == "${TSDF_ROOT}/configs/rgbd/tum/fr3_office.yaml" ]]; then
        CONFIG_PATH="$1"
      elif [[ -z "${WORKSPACE_NAME}" && "$1" != --* ]]; then
        WORKSPACE_NAME="$1"
      else
        echo "Unknown argument: $1"
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "${WORKSPACE_NAME}" ]]; then
  WORKSPACE_NAME="$(basename "${CONFIG_PATH%.*}")_openmvs"
fi

LOG_ROOT="${TSDF_ROOT}/log/oppenmvs/${WORKSPACE_NAME}"
mkdir -p "${LOG_ROOT}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
INTERFACE_STDOUT_LOG="${LOG_ROOT}/${RUN_STAMP}_interfacecolmap.stdout.log"
DENSIFY_STDOUT_LOG="${LOG_ROOT}/${RUN_STAMP}_densifypointcloud.stdout.log"

if [[ -z "${OPENMVS_BIN_DIR}" ]]; then
  if [[ -d "${TSDF_ROOT}/openmvs/OpenMVS/build/bin" ]]; then
    OPENMVS_BIN_DIR="${TSDF_ROOT}/openmvs/OpenMVS/build/bin"
  elif [[ -d "${TSDF_ROOT}/openmvs/_build/OpenMVS/bin" ]]; then
    OPENMVS_BIN_DIR="${TSDF_ROOT}/openmvs/_build/OpenMVS/bin"
  elif [[ -d "${TSDF_ROOT}/openmvs/OpenMVS/build/bin/x64/Release" ]]; then
    OPENMVS_BIN_DIR="${TSDF_ROOT}/openmvs/OpenMVS/build/bin/x64/Release"
  fi
fi

if [[ -z "${OPENMVS_BIN_DIR}" ]]; then
  echo "Could not find the OpenMVS binary directory."
  echo "Please run: bash openmvs/setup_openmvs.sh"
  echo "Or set OPENMVS_BIN=/path/to/OpenMVS/build/bin"
  exit 1
fi

INTERFACE_COLMAP="${OPENMVS_BIN_DIR}/InterfaceCOLMAP"
DENSIFY_POINTCLOUD="${OPENMVS_BIN_DIR}/DensifyPointCloud"

if [[ ! -x "${INTERFACE_COLMAP}" || ! -x "${DENSIFY_POINTCLOUD}" ]]; then
  echo "OpenMVS executables are incomplete: ${OPENMVS_BIN_DIR}"
  exit 1
fi

WORKSPACE_DIR="${TSDF_ROOT}/openmvs/workspaces/${WORKSPACE_NAME}"
COLMAP_DIR="${WORKSPACE_DIR}/colmap"
SPARSE_DIR="${COLMAP_DIR}/sparse"
IMAGES_DIR="${WORKSPACE_DIR}/images"
SCENE_MVS="${WORKSPACE_DIR}/scene.mvs"
DENSE_MVS="${WORKSPACE_DIR}/scene_dense.mvs"

if [[ "${REEXPORT}" -eq 1 || ! -d "${SPARSE_DIR}" || ! -d "${IMAGES_DIR}" ]]; then
  if [[ "${REEXPORT}" -eq 1 ]]; then
    echo "Re-exporting the TUM sequence into the workspace."
  else
    echo "Workspace not found. Exporting the TUM sequence first."
  fi
  EXPORT_CMD=(
    python "${SCRIPT_DIR}/export_tum_to_openmvs.py"
    --config "${CONFIG_PATH}"
    --workspace-name "${WORKSPACE_NAME}"
    --frame-stride "${FRAME_STRIDE}"
    --max-frames "${MAX_FRAMES}"
    --max-dt "${MAX_DT}"
  )
  if [[ -n "${INPUT_WIDTH}" ]]; then
    EXPORT_CMD+=(--input-width "${INPUT_WIDTH}")
  fi
  if [[ -n "${INPUT_HEIGHT}" ]]; then
    EXPORT_CMD+=(--input-height "${INPUT_HEIGHT}")
  fi
  "${EXPORT_CMD[@]}"
else
  echo "Using existing workspace: ${WORKSPACE_DIR}"
fi

echo "Logs will be stored in: ${LOG_ROOT}"
pushd "${LOG_ROOT}" >/dev/null

echo "1/2 Running InterfaceCOLMAP"
"${INTERFACE_COLMAP}" \
  -i "${COLMAP_DIR}" \
  -o "${SCENE_MVS}" \
  --image-folder "${IMAGES_DIR}" \
  2>&1 | tee "${INTERFACE_STDOUT_LOG}"

echo "2/2 Running DensifyPointCloud"
"${DENSIFY_POINTCLOUD}" \
  -i "${SCENE_MVS}" \
  -o "${DENSE_MVS}" \
  --working-folder "${WORKSPACE_DIR}" \
  --resolution-level "${RESOLUTION_LEVEL}" \
  --max-resolution "${MAX_RESOLUTION}" \
  --min-resolution "${MIN_RESOLUTION}" \
  --number-views "${NUMBER_VIEWS}" \
  --number-views-fuse "${NUMBER_VIEWS_FUSE}" \
  --iters "${ITERS}" \
  --geometric-iters "${GEOMETRIC_ITERS}" \
  --remove-dmaps "${REMOVE_DMAPS}" \
  --max-threads "${MAX_THREADS}" \
  2>&1 | tee "${DENSIFY_STDOUT_LOG}"

popd >/dev/null

echo "OpenMVS finished. Generated files in workspace:"
find "${WORKSPACE_DIR}" -maxdepth 1 \( -name '*.mvs' -o -name '*.ply' -o -name '*.dmap' \) -print
echo "Workspace directory: ${WORKSPACE_DIR}"
echo "Run logs:"
echo "  ${INTERFACE_STDOUT_LOG}"
echo "  ${DENSIFY_STDOUT_LOG}"
