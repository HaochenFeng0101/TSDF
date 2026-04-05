#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSDF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${TSDF_ROOT}/configs/rgbd/tum/fr3_office.yaml"
WORKSPACE_NAME=""
OPENMVS_BIN_DIR="${OPENMVS_BIN:-}"
REEXPORT=0
FRAME_STRIDE=10
MAX_FRAMES=120
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
MAX_THREADS=0
REMOVE_DMAPS=1
RECONSTRUCT=1
REFINE=1
TEXTURE=1
MESH_DECIMATE=1
MESH_REMOVE_SPURIOUS=20
MESH_CLOSE_HOLES=30
MESH_SMOOTH=2
FREE_SPACE_SUPPORT=1
REFINE_SCALES=3
REFINE_MAX_VIEWS=8
TEXTURE_MAX_SIZE=1024
TEXTURE_VIRTUAL_FACE_IMAGES=0
EXPORT_STAGE_PCD=1
STAGE_PCD_POINTS=500000

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
    --skip-reconstruct)
      RECONSTRUCT=0
      shift
      ;;
    --skip-refine)
      REFINE=0
      shift
      ;;
    --skip-texture)
      TEXTURE=0
      shift
      ;;
    --mesh-decimate)
      MESH_DECIMATE="$2"
      shift 2
      ;;
    --mesh-remove-spurious)
      MESH_REMOVE_SPURIOUS="$2"
      shift 2
      ;;
    --mesh-close-holes)
      MESH_CLOSE_HOLES="$2"
      shift 2
      ;;
    --mesh-smooth)
      MESH_SMOOTH="$2"
      shift 2
      ;;
    --free-space-support)
      FREE_SPACE_SUPPORT="$2"
      shift 2
      ;;
    --refine-scales)
      REFINE_SCALES="$2"
      shift 2
      ;;
    --refine-max-views)
      REFINE_MAX_VIEWS="$2"
      shift 2
      ;;
    --texture-max-size)
      TEXTURE_MAX_SIZE="$2"
      shift 2
      ;;
    --texture-virtual-face-images)
      TEXTURE_VIRTUAL_FACE_IMAGES="$2"
      shift 2
      ;;
    --no-stage-pcd)
      EXPORT_STAGE_PCD=0
      shift
      ;;
    --stage-pcd-points)
      STAGE_PCD_POINTS="$2"
      shift 2
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
  --skip-reconstruct
  --skip-refine
  --skip-texture
  --mesh-decimate FLOAT
  --mesh-remove-spurious N
  --mesh-close-holes N
  --mesh-smooth N
  --free-space-support 0|1
  --refine-scales N
  --refine-max-views N
  --texture-max-size N
  --texture-virtual-face-images N
  --no-stage-pcd
  --stage-pcd-points N

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
RECONSTRUCT_STDOUT_LOG="${LOG_ROOT}/${RUN_STAMP}_reconstructmesh.stdout.log"
REFINE_STDOUT_LOG="${LOG_ROOT}/${RUN_STAMP}_refinemesh.stdout.log"
TEXTURE_STDOUT_LOG="${LOG_ROOT}/${RUN_STAMP}_texturemesh.stdout.log"

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
RECONSTRUCT_MESH="${OPENMVS_BIN_DIR}/ReconstructMesh"
REFINE_MESH="${OPENMVS_BIN_DIR}/RefineMesh"
TEXTURE_MESH="${OPENMVS_BIN_DIR}/TextureMesh"

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
DENSE_PLY="${WORKSPACE_DIR}/scene_dense.ply"
MESH_PLY="${WORKSPACE_DIR}/scene_mesh.ply"
REFINE_PLY="${WORKSPACE_DIR}/scene_mesh_refine.ply"
TEXTURE_PLY="${WORKSPACE_DIR}/scene_mesh_refine_texture.ply"
SEED_PLY="${WORKSPACE_DIR}/seed_from_depth.ply"

export_stage_pcd() {
  local input_ply="$1"
  local output_pcd="$2"
  if [[ "${EXPORT_STAGE_PCD}" -ne 1 ]]; then
    return 0
  fi
  if [[ ! -f "${input_ply}" ]]; then
    return 0
  fi
  python3 "${SCRIPT_DIR}/export_stage_pcd.py" \
    "${input_ply}" \
    "${output_pcd}" \
    --sample-points "${STAGE_PCD_POINTS}"
}

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

export_stage_pcd "${SEED_PLY}" "${WORKSPACE_DIR}/seed_from_depth.pcd"
export_stage_pcd "${DENSE_PLY}" "${WORKSPACE_DIR}/scene_dense.pcd"

if [[ "${RECONSTRUCT}" -eq 1 ]]; then
  echo "3/5 Running ReconstructMesh"
  "${RECONSTRUCT_MESH}" \
    -i "${DENSE_MVS}" \
    -p "${DENSE_PLY}" \
    -o "${MESH_PLY}" \
    --working-folder "${WORKSPACE_DIR}" \
    --free-space-support "${FREE_SPACE_SUPPORT}" \
    --decimate "${MESH_DECIMATE}" \
    --remove-spurious "${MESH_REMOVE_SPURIOUS}" \
    --close-holes "${MESH_CLOSE_HOLES}" \
    --smooth "${MESH_SMOOTH}" \
    --max-threads "${MAX_THREADS}" \
    2>&1 | tee "${RECONSTRUCT_STDOUT_LOG}"

  export_stage_pcd "${MESH_PLY}" "${WORKSPACE_DIR}/scene_mesh.pcd"
fi

if [[ "${REFINE}" -eq 1 && -f "${MESH_PLY}" ]]; then
  echo "4/5 Running RefineMesh"
  "${REFINE_MESH}" \
    -i "${DENSE_MVS}" \
    -m "${MESH_PLY}" \
    -o "${REFINE_PLY}" \
    --working-folder "${WORKSPACE_DIR}" \
    --resolution-level "${RESOLUTION_LEVEL}" \
    --min-resolution "${MIN_RESOLUTION}" \
    --max-views "${REFINE_MAX_VIEWS}" \
    --scales "${REFINE_SCALES}" \
    --close-holes "${MESH_CLOSE_HOLES}" \
    --max-threads "${MAX_THREADS}" \
    2>&1 | tee "${REFINE_STDOUT_LOG}"

  export_stage_pcd "${REFINE_PLY}" "${WORKSPACE_DIR}/scene_mesh_refine.pcd"
fi

if [[ "${TEXTURE}" -eq 1 ]]; then
  TEXTURE_INPUT_MESH="${REFINE_PLY}"
  if [[ ! -f "${TEXTURE_INPUT_MESH}" ]]; then
    TEXTURE_INPUT_MESH="${MESH_PLY}"
  fi
  if [[ -f "${TEXTURE_INPUT_MESH}" ]]; then
    echo "5/5 Running TextureMesh"
    "${TEXTURE_MESH}" \
      -i "${DENSE_MVS}" \
      -m "${TEXTURE_INPUT_MESH}" \
      -o "${TEXTURE_PLY}" \
      --working-folder "${WORKSPACE_DIR}" \
      --resolution-level "${RESOLUTION_LEVEL}" \
      --min-resolution "${MIN_RESOLUTION}" \
      --virtual-face-images "${TEXTURE_VIRTUAL_FACE_IMAGES}" \
      --max-texture-size "${TEXTURE_MAX_SIZE}" \
      --max-threads "${MAX_THREADS}" \
      2>&1 | tee "${TEXTURE_STDOUT_LOG}"

    export_stage_pcd "${TEXTURE_PLY}" "${WORKSPACE_DIR}/scene_mesh_refine_texture.pcd"
  fi
fi

popd >/dev/null

echo "OpenMVS finished. Generated files in workspace:"
find "${WORKSPACE_DIR}" -maxdepth 1 \( -name '*.mvs' -o -name '*.ply' -o -name '*.dmap' \) -print
echo "Workspace directory: ${WORKSPACE_DIR}"
echo "Stage summary:"
if [[ -f "${WORKSPACE_DIR}/seed_from_depth.pcd" ]]; then
  echo "  seed_from_depth.pcd"
  echo "    Direct RGB-D back-projection from the input sequence."
  echo "    Fastest baseline; keeps raw depth noise and view inconsistencies."
fi
if [[ -f "${WORKSPACE_DIR}/scene_dense.pcd" ]]; then
  echo "  scene_dense.pcd"
  echo "    Output of DensifyPointCloud."
  echo "    Improves density and multi-view consistency over seed_from_depth.pcd."
fi
if [[ -f "${WORKSPACE_DIR}/scene_mesh.pcd" ]]; then
  echo "  scene_mesh.pcd"
  echo "    Sampled from the ReconstructMesh surface."
  echo "    Improves structural completeness by turning the dense cloud into a connected surface."
fi
if [[ -f "${WORKSPACE_DIR}/scene_mesh_refine.pcd" ]]; then
  echo "  scene_mesh_refine.pcd"
  echo "    Sampled from the RefineMesh surface."
  echo "    Improves geometry quality, boundaries, and local detail over scene_mesh.pcd."
fi
if [[ -f "${WORKSPACE_DIR}/scene_mesh_refine_texture.pcd" ]]; then
  echo "  scene_mesh_refine_texture.pcd"
  echo "    Sampled from the TextureMesh result."
  echo "    Same refined geometry as the final mesh, but with image-based color information."
fi
echo "Run logs:"
echo "  ${INTERFACE_STDOUT_LOG}"
echo "  ${DENSIFY_STDOUT_LOG}"
if [[ -f "${RECONSTRUCT_STDOUT_LOG}" ]]; then
  echo "  ${RECONSTRUCT_STDOUT_LOG}"
fi
if [[ -f "${REFINE_STDOUT_LOG}" ]]; then
  echo "  ${REFINE_STDOUT_LOG}"
fi
if [[ -f "${TEXTURE_STDOUT_LOG}" ]]; then
  echo "  ${TEXTURE_STDOUT_LOG}"
fi
