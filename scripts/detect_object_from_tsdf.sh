#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSDF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="configs/rgbd/tum/fr3_office.yaml"
TARGET_CLASS="chair"
YOLO_MODEL="yolov8x-seg.pt"
CLASSIFIER="pointnet"
TRACK_ID=""
NO_VISUALIZE=0
CHECKPOINT=""
LABELS=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/detect_object_from_tsdf.sh [options]

Options:
  --config PATH
  --target-class NAME
  --yolo-model PATH
  --classifier pointmlp|pointnet2|pointnet
  --track-id N
  --checkpoint PATH
  --labels PATH
  --no-visualize
  --help, -h
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --target-class)
      TARGET_CLASS="$2"
      shift 2
      ;;
    --yolo-model)
      YOLO_MODEL="$2"
      shift 2
      ;;
    --classifier)
      CLASSIFIER="$2"
      shift 2
      ;;
    --track-id)
      TRACK_ID="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --labels)
      LABELS="$2"
      shift 2
      ;;
    --no-visualize)
      NO_VISUALIZE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

cd "${TSDF_ROOT}"

CONFIG_STEM="$(basename "${CONFIG_PATH%.*}")"
MODEL_STEM="$(basename "${YOLO_MODEL%.*}")"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASK_OUTPUT_DIR="mask_generation/outputs/${CONFIG_STEM}_${TARGET_CLASS}_${MODEL_STEM}_${STAMP}"
FUSE_OUTPUT_DIR="3d_construction/outputs/fuse_obj_${CONFIG_STEM}_${TARGET_CLASS}_${STAMP}"

echo "[1/3] Generating 2D masks"
python3 mask_generation/generate_tum_masks_yolo.py \
  --config "${CONFIG_PATH}" \
  --model "${YOLO_MODEL}" \
  --target-class "${TARGET_CLASS}" \
  --output-dir "${MASK_OUTPUT_DIR}"

echo "[2/3] Fusing 3D object point cloud"
FUSE_CMD=(
  python3 3d_construction/fuse_tum_mask_object_pcd.py
  "${MASK_OUTPUT_DIR}"
  --output-dir "${FUSE_OUTPUT_DIR}"
)
if [[ -n "${TRACK_ID}" ]]; then
  FUSE_CMD+=(--track-id "${TRACK_ID}")
fi
"${FUSE_CMD[@]}"

FUSED_PCD="${FUSE_OUTPUT_DIR}/fused_object.pcd"
if [[ ! -f "${FUSED_PCD}" ]]; then
  echo "Expected fused object point cloud was not generated: ${FUSED_PCD}"
  exit 1
fi

echo "[3/3] Classifying fused object"
VALIDATE_CMD=()
case "${CLASSIFIER}" in
  pointmlp)
    VALIDATE_CMD=(python3 detection/validate/validate_pointmlp_own_object.py "${FUSED_PCD}" --use-all-points)
    ;;
  pointnet2)
    VALIDATE_CMD=(python3 detection/validate/validate_pointnet2_own_object.py "${FUSED_PCD}" --use-all-points)
    ;;
  pointnet)
    VALIDATE_CMD=(python3 detection/pointnet/validate_own_object.py "${FUSED_PCD}" --use-all-points)
    ;;
  *)
    echo "Unsupported classifier: ${CLASSIFIER}"
    exit 1
    ;;
esac

if [[ -n "${CHECKPOINT}" ]]; then
  VALIDATE_CMD+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${LABELS}" ]]; then
  VALIDATE_CMD+=(--labels "${LABELS}")
fi
if [[ "${NO_VISUALIZE}" -eq 1 ]]; then
  VALIDATE_CMD+=(--no-visualize)
fi
"${VALIDATE_CMD[@]}"

echo ""
echo "Mask output: ${MASK_OUTPUT_DIR}"
echo "Fused object: ${FUSED_PCD}"
