#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSDF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TSDF_ROOT}/detection/modelnet40c/train.py"

MODELS="curvenet,gdanet,dgcnn,pointnet"
DATASET_TYPE="modelnet40"
DATA_ROOT=""
TRAIN_H5=""
TEST_H5=""
LABELS=""
SCANOBJECTNN_ROOT="${TSDF_ROOT}/data/ScanObjectNN"
SCANOBJECTNN_VARIANT="pb_t50_rs"
SCANOBJECTNN_NO_BG=0
MODELNET40_ROOT="${TSDF_ROOT}/data/ModelNet40"
MODELNET40_SAMPLE_METHOD="surface"
EPOCHS=150
BATCH_SIZE=32
NUM_POINTS=1024
LR="1e-3"
WEIGHT_DECAY="1e-4"
OPTIMIZER="adamw"
MOMENTUM="0.9"
LABEL_SMOOTHING="0.0"
LOSS_NAME=""
FEATURE_TRANSFORM_WEIGHT="1e-3"
SIMPLEVIEW_FEAT_SIZE=16
USE_CLASS_WEIGHTS=0
GRAD_CLIP="1.0"
WORKERS=4
SEED=0
AMP=0
DEVICE=""
USE_WANDB=0
WANDB_PROJECT="TSDF-OfficialCls"
OUTPUT_ROOT="${TSDF_ROOT}/model"
EXTRA_ARGS=()
SUCCESS_MODELS=()
FAILED_MODELS=()

normalize_model_name() {
  local raw="$1"
  local normalized
  normalized="$(echo "${raw}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  case "${normalized}" in
    pointnet++)
      echo "pointnet2"
      ;;
    pointnet2)
      echo "pointnet2"
      ;;
    *)
      echo "${normalized}"
      ;;
  esac
}

print_help() {
  cat <<'EOF'
Usage:
  bash scripts/train_models.sh [options]

Description:
  Train one or more point cloud classifiers, including:
  CurveNet, PointNet++, PCT, GDANet, DGCNN, and PointNet.

Examples:
  bash scripts/train_models.sh
  bash scripts/train_models.sh --models curvenet,pointnet++,pct --dataset-type modelnet40
  bash scripts/train_models.sh --models dgcnn --dataset-type scanobjectnn --scanobjectnn-root data/ScanObjectNN
  bash scripts/train_models.sh --models pointnet --dataset-type dir --data-root data/my_cls_dataset

Options:
  --models LIST                  Comma-separated model list.
                                 Supported names: curvenet, pointnet++, pointnet2, pct, gdanet, dgcnn, pointnet
  --python BIN                   Python executable to use.
  --dataset-type TYPE            One of: modelnet40, scanobjectnn, dir, h5
  --data-root PATH               Required for --dataset-type dir
  --train-h5 PATH                Required for --dataset-type h5
  --test-h5 PATH                 Required for --dataset-type h5
  --labels PATH                  Required for --dataset-type h5
  --scanobjectnn-root PATH
  --scanobjectnn-variant NAME
  --scanobjectnn-no-bg
  --modelnet40-root PATH
  --modelnet40-sample-method M   One of: surface, vertices
  --epochs N
  --batch-size N
  --num-points N
  --lr FLOAT
  --weight-decay FLOAT
  --optimizer NAME               One of: adamw, sgd
  --momentum FLOAT
  --label-smoothing FLOAT
  --loss-name NAME               One of: cross_entropy, smooth
  --feature-transform-weight FLOAT
  --simpleview-feat-size N
  --use-class-weights
  --grad-clip FLOAT
  --workers N
  --seed N
  --amp
  --device NAME
  --use-wandb
  --wandb-project NAME
  --output-root PATH             Root folder for model outputs.
  --help, -h                     Show this help message.

Notes:
  1. Each model is trained sequentially.
  2. If one model fails, the script skips it and continues with the next model.
  3. Checkpoints are saved to:
     OUTPUT_ROOT/<model_name>/
  4. Extra unknown args after a standalone -- are forwarded to the Python trainer.

Forward extra args example:
  bash scripts/train_models.sh --models pct -- --wandb-run-name pct_exp1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODELS="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dataset-type)
      DATASET_TYPE="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --train-h5)
      TRAIN_H5="$2"
      shift 2
      ;;
    --test-h5)
      TEST_H5="$2"
      shift 2
      ;;
    --labels)
      LABELS="$2"
      shift 2
      ;;
    --scanobjectnn-root)
      SCANOBJECTNN_ROOT="$2"
      shift 2
      ;;
    --scanobjectnn-variant)
      SCANOBJECTNN_VARIANT="$2"
      shift 2
      ;;
    --scanobjectnn-no-bg)
      SCANOBJECTNN_NO_BG=1
      shift
      ;;
    --modelnet40-root)
      MODELNET40_ROOT="$2"
      shift 2
      ;;
    --modelnet40-sample-method)
      MODELNET40_SAMPLE_METHOD="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-points)
      NUM_POINTS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --weight-decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --optimizer)
      OPTIMIZER="$2"
      shift 2
      ;;
    --momentum)
      MOMENTUM="$2"
      shift 2
      ;;
    --label-smoothing)
      LABEL_SMOOTHING="$2"
      shift 2
      ;;
    --loss-name)
      LOSS_NAME="$2"
      shift 2
      ;;
    --feature-transform-weight)
      FEATURE_TRANSFORM_WEIGHT="$2"
      shift 2
      ;;
    --simpleview-feat-size)
      SIMPLEVIEW_FEAT_SIZE="$2"
      shift 2
      ;;
    --use-class-weights)
      USE_CLASS_WEIGHTS=1
      shift
      ;;
    --grad-clip)
      GRAD_CLIP="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --amp)
      AMP=1
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --use-wandb)
      USE_WANDB=1
      shift
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --help|-h)
      print_help
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help to see supported options."
      exit 1
      ;;
  esac
done

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Training entrypoint not found: ${TRAIN_SCRIPT}"
  exit 1
fi

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"

if [[ ${#MODEL_LIST[@]} -eq 0 ]]; then
  echo "No models were provided."
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "Training script: ${TRAIN_SCRIPT}"
echo "Dataset type: ${DATASET_TYPE}"
echo "Models: ${MODELS}"
echo "Output root: ${OUTPUT_ROOT}"

for raw_model in "${MODEL_LIST[@]}"; do
  input_model="$(echo "${raw_model}" | xargs)"
  if [[ -z "${input_model}" ]]; then
    continue
  fi
  model="$(normalize_model_name "${input_model}")"

  output_dir="${OUTPUT_ROOT}/${model}"
  cmd=(
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
    --model-name "${input_model}"
    --dataset-type "${DATASET_TYPE}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-points "${NUM_POINTS}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --optimizer "${OPTIMIZER}"
    --momentum "${MOMENTUM}"
    --label-smoothing "${LABEL_SMOOTHING}"
    --feature-transform-weight "${FEATURE_TRANSFORM_WEIGHT}"
    --simpleview-feat-size "${SIMPLEVIEW_FEAT_SIZE}"
    --grad-clip "${GRAD_CLIP}"
    --workers "${WORKERS}"
    --seed "${SEED}"
    --scanobjectnn-root "${SCANOBJECTNN_ROOT}"
    --scanobjectnn-variant "${SCANOBJECTNN_VARIANT}"
    --modelnet40-root "${MODELNET40_ROOT}"
    --modelnet40-sample-method "${MODELNET40_SAMPLE_METHOD}"
    --wandb-project "${WANDB_PROJECT}"
    --output-dir "${output_dir}"
  )

  case "${DATASET_TYPE}" in
    dir)
      if [[ -z "${DATA_ROOT}" ]]; then
        echo "--data-root is required when --dataset-type dir"
        exit 1
      fi
      cmd+=(--data-root "${DATA_ROOT}")
      ;;
    h5)
      if [[ -z "${TRAIN_H5}" || -z "${TEST_H5}" || -z "${LABELS}" ]]; then
        echo "--train-h5, --test-h5, and --labels are required when --dataset-type h5"
        exit 1
      fi
      cmd+=(--train-h5 "${TRAIN_H5}" --test-h5 "${TEST_H5}" --labels "${LABELS}")
      ;;
    scanobjectnn)
      if [[ "${SCANOBJECTNN_NO_BG}" -eq 1 ]]; then
        cmd+=(--scanobjectnn-no-bg)
      fi
      ;;
    modelnet40)
      :
      ;;
    *)
      echo "Unsupported dataset type: ${DATASET_TYPE}"
      exit 1
      ;;
  esac

  if [[ -n "${LOSS_NAME}" ]]; then
    cmd+=(--loss-name "${LOSS_NAME}")
  fi
  if [[ "${USE_CLASS_WEIGHTS}" -eq 1 ]]; then
    cmd+=(--use-class-weights)
  fi
  if [[ "${AMP}" -eq 1 ]]; then
    cmd+=(--amp)
  fi
  if [[ -n "${DEVICE}" ]]; then
    cmd+=(--device "${DEVICE}")
  fi
  if [[ "${USE_WANDB}" -eq 1 ]]; then
    cmd+=(--use-wandb)
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo
  echo "============================================================"
  echo "Starting training for model: ${input_model} -> ${model}"
  echo "Checkpoint directory: ${output_dir}"
  echo "Command: ${cmd[*]}"
  echo "============================================================"

  if "${cmd[@]}"; then
    echo "Model ${input_model} finished successfully."
    SUCCESS_MODELS+=("${input_model}")
  else
    status=$?
    echo "Model ${input_model} failed with exit code ${status}. Skipping to the next model."
    FAILED_MODELS+=("${input_model}")
  fi
done

echo
echo "Training summary:"
echo "  successful: ${#SUCCESS_MODELS[@]}"
for model in "${SUCCESS_MODELS[@]}"; do
  echo "    - ${model}"
done
echo "  failed: ${#FAILED_MODELS[@]}"
for model in "${FAILED_MODELS[@]}"; do
  echo "    - ${model}"
done

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  exit 1
fi

echo "All requested trainings finished successfully."
