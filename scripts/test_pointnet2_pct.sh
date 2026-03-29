#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSDF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_TYPE="scanobjectnn"
SCANOBJECTNN_ROOT="${TSDF_ROOT}/data/ScanObjectNN"
EPOCHS=1
BATCH_SIZE=8
WORKERS=0
DEVICE="cuda"
LOG_DIR="${TSDF_ROOT}/log/model_smoke_tests"
EXTRA_ARGS=()

SUCCESS_MODELS=()
FAILED_MODELS=()

print_help() {
  cat <<'EOF'
Usage:
  bash scripts/test_pointnet2_pct.sh [options] [-- extra trainer args]

Description:
  Smoke-test whether PointNet++ and PCT can start training on this server.
  Each model is run separately with a lightweight configuration.

Examples:
  bash scripts/test_pointnet2_pct.sh --python /root/miniconda3/envs/tsdf/bin/python
  bash scripts/test_pointnet2_pct.sh --epochs 1 --batch-size 4 --workers 0

Options:
  --python BIN
  --dataset-type TYPE
  --scanobjectnn-root PATH
  --epochs N
  --batch-size N
  --workers N
  --device NAME
  --log-dir PATH
  --help, -h
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dataset-type)
      DATASET_TYPE="$2"
      shift 2
      ;;
    --scanobjectnn-root)
      SCANOBJECTNN_ROOT="$2"
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
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
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

mkdir -p "${LOG_DIR}"

run_one_model() {
  local model="$1"
  local stdout_log="${LOG_DIR}/${model}.out.log"
  local stderr_log="${LOG_DIR}/${model}.err.log"

  local cmd=(
    "${PYTHON_BIN}" "${TSDF_ROOT}/detection/modelnet40c/train.py"
    --model-name "${model}"
    --dataset-type "${DATASET_TYPE}"
    --scanobjectnn-root "${SCANOBJECTNN_ROOT}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --device "${DEVICE}"
  )

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo
  echo "============================================================"
  echo "Testing model: ${model}"
  echo "stdout log: ${stdout_log}"
  echo "stderr log: ${stderr_log}"
  echo "Command: ${cmd[*]}"
  echo "============================================================"

  if "${cmd[@]}" >"${stdout_log}" 2>"${stderr_log}"; then
    echo "Smoke test passed: ${model}"
    SUCCESS_MODELS+=("${model}")
  else
    status=$?
    echo "Smoke test failed: ${model} (exit code ${status})"
    echo "Last stderr lines:"
    tail -n 20 "${stderr_log}" || true
    FAILED_MODELS+=("${model}")
  fi
}

run_one_model "pointnet++"
run_one_model "pct"

echo
echo "Smoke test summary:"
echo "  successful: ${#SUCCESS_MODELS[@]}"
for model in "${SUCCESS_MODELS[@]}"; do
  echo "    - ${model}"
done
echo "  failed: ${#FAILED_MODELS[@]}"
for model in "${FAILED_MODELS[@]}"; do
  echo "    - ${model}"
done

echo
echo "Logs saved under: ${LOG_DIR}"

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  exit 1
fi
