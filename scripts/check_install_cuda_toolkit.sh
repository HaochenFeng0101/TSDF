#!/usr/bin/env bash
set -euo pipefail

CUDA_VERSION_MAJOR_MINOR="12-4"
CUDA_VERSION_DOTTED="12.4"
CUDA_REPO_DISTRO="ubuntu2204"
CUDA_REPO_ARCH="x86_64"
CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_DISTRO}/${CUDA_REPO_ARCH}/${CUDA_KEYRING_DEB}"
CUDA_DEFAULT_HOME="/usr/local/cuda-${CUDA_VERSION_DOTTED}"

log() {
  echo
  echo "[check_install_cuda_toolkit] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

detect_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "${CUDA_HOME}"
    return
  fi
  if [[ -x "${CUDA_DEFAULT_HOME}/bin/nvcc" ]]; then
    echo "${CUDA_DEFAULT_HOME}"
    return
  fi
  if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    echo "/usr/local/cuda"
    return
  fi
  echo ""
}

print_status() {
  local detected_home="$1"
  log "System info"
  uname -a || true
  cat /etc/os-release || true

  log "Driver info"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found"
  fi

  log "CUDA toolkit status"
  echo "CUDA_HOME=${CUDA_HOME:-}"
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    nvcc --version || true
  else
    echo "nvcc not found in PATH"
  fi

  if [[ -n "${detected_home}" ]]; then
    echo "Detected CUDA toolkit home: ${detected_home}"
  else
    echo "No CUDA toolkit detected"
  fi
}

install_cuda_toolkit() {
  require_cmd wget
  require_cmd sudo
  require_cmd dpkg
  require_cmd apt-get

  log "Installing CUDA toolkit ${CUDA_VERSION_DOTTED} from NVIDIA's Ubuntu 22.04 repository"
  log "Reference: NVIDIA CUDA Installation Guide for Linux (CUDA 12.4.1), Ubuntu network repo"

  sudo apt-get update
  sudo apt-get install -y build-essential linux-headers-"$(uname -r)"

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' EXIT
  cd "${tmpdir}"

  wget "${CUDA_KEYRING_URL}"
  sudo dpkg -i "${CUDA_KEYRING_DEB}"
  sudo apt-get update

  # Lock to the 12.4 toolkit package to avoid pulling a newer CUDA major/minor.
  sudo apt-get install -y "cuda-toolkit-${CUDA_VERSION_MAJOR_MINOR}"

  cd - >/dev/null
}

write_shell_exports() {
  local cuda_home="$1"
  if [[ -z "${cuda_home}" ]]; then
    return
  fi

  cat <<EOF
export CUDA_HOME=${cuda_home}
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
EOF
}

main() {
  local detected_home
  detected_home="$(detect_cuda_home)"
  print_status "${detected_home}"

  if [[ -n "${detected_home}" ]]; then
    log "CUDA toolkit is already available. No installation needed."
    log "Recommended shell exports"
    write_shell_exports "${detected_home}"
    exit 0
  fi

  install_cuda_toolkit

  detected_home="$(detect_cuda_home)"
  print_status "${detected_home}"

  if [[ -z "${detected_home}" ]]; then
    echo "CUDA toolkit installation finished, but nvcc is still not detected."
    echo "You may need to open a new shell or manually export CUDA_HOME."
    exit 1
  fi

  log "CUDA toolkit installation succeeded."
  log "Recommended shell exports"
  write_shell_exports "${detected_home}"
}

main "$@"
