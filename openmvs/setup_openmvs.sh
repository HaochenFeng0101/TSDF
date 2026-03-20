#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSDF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPENMVS_SRC_DIR="${TSDF_ROOT}/openmvs/OpenMVS"
THIRD_PARTY_DIR="${TSDF_ROOT}/openmvs/third_party"
BUILD_ROOT_DIR="${TSDF_ROOT}/openmvs/_build"
INSTALL_ROOT_DIR="${TSDF_ROOT}/openmvs/_install"
OPENMVS_BUILD_DIR="${OPENMVS_SRC_DIR}/build"
NANOFLANN_SRC_DIR="${THIRD_PARTY_DIR}/nanoflann"
NANOFLANN_BUILD_DIR="${BUILD_ROOT_DIR}/nanoflann"
NANOFLANN_INSTALL_DIR="${INSTALL_ROOT_DIR}/nanoflann"
NANOFLANN_CONFIG="${NANOFLANN_INSTALL_DIR}/lib/cmake/nanoflann/nanoflannConfig.cmake"
VCG_SRC_DIR="${THIRD_PARTY_DIR}/vcglib"
OPENMVS_IO_CMAKE="${OPENMVS_SRC_DIR}/libs/IO/CMakeLists.txt"

if [[ ! -d "${OPENMVS_SRC_DIR}" ]]; then
  git clone --recurse-submodules https://github.com/cdcseacave/openMVS.git "${OPENMVS_SRC_DIR}"
else
  echo "Source directory already exists: ${OPENMVS_SRC_DIR}"
fi

python3 -c '
from pathlib import Path
path = Path("'"${OPENMVS_IO_CMAKE}"'")
text = path.read_text(encoding="utf-8")
old = "pkg_check_modules(${PREFIX} REQUIRED IMPORTED_TARGET ${MODULE_NAME})"
new = "pkg_check_modules(${PREFIX} QUIET IMPORTED_TARGET ${MODULE_NAME})"
if old in text:
    path.write_text(text.replace(old, new), encoding="utf-8")
'

mkdir -p "${THIRD_PARTY_DIR}"
mkdir -p "${BUILD_ROOT_DIR}" "${INSTALL_ROOT_DIR}"

if [[ ! -f "${NANOFLANN_CONFIG}" ]]; then
  if [[ ! -d "${NANOFLANN_SRC_DIR}" ]]; then
    git clone --recurse-submodules https://github.com/jlblancoc/nanoflann.git "${NANOFLANN_SRC_DIR}"
  else
    echo "nanoflann source directory already exists: ${NANOFLANN_SRC_DIR}"
  fi

  cmake -S "${NANOFLANN_SRC_DIR}" -B "${NANOFLANN_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${NANOFLANN_INSTALL_DIR}"
  cmake --build "${NANOFLANN_BUILD_DIR}" -j"$(nproc)"
  cmake --install "${NANOFLANN_BUILD_DIR}"
fi

if [[ ! -d "${VCG_SRC_DIR}" ]]; then
  git clone --recurse-submodules https://github.com/cnr-isti-vclab/vcglib.git "${VCG_SRC_DIR}"
else
  echo "vcglib source directory already exists: ${VCG_SRC_DIR}"
fi

VCG_ROOT="${VCG_SRC_DIR}" cmake -S "${OPENMVS_SRC_DIR}" -B "${OPENMVS_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${NANOFLANN_INSTALL_DIR}" \
  -DOpenMVS_BUILD_VIEWER=OFF \
  -DOpenMVS_ENABLE_TESTS=OFF \
  -DOpenMVS_USE_PYTHON=OFF
cmake --build "${OPENMVS_BUILD_DIR}" -j"$(nproc)"

echo "OpenMVS build finished."
echo "Binaries are usually available in: ${OPENMVS_BUILD_DIR}/bin"
