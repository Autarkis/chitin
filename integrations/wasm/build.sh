#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COACD_SRC="${COACD_SRC:-/tmp/coacd-src}"
BUILD_DIR="${SCRIPT_DIR}/build"
OUT_DIR="${SCRIPT_DIR}/dist"

if [ ! -f "${COACD_SRC}/public/coacd.h" ]; then
    echo "CoACD source not found at ${COACD_SRC}"
    echo "Clone it: git clone --depth 1 --recurse-submodules https://github.com/SarahWeiii/CoACD.git ${COACD_SRC}"
    exit 1
fi

if ! command -v emcc &>/dev/null; then
    echo "emcc not found. Source emsdk_env.sh first."
    exit 1
fi

echo "Building CoACD static library for WASM..."
mkdir -p "${BUILD_DIR}/coacd"
cd "${BUILD_DIR}/coacd"
emcmake cmake "${COACD_SRC}" \
    -DWITH_3RD_PARTY_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -Wno-dev 2>&1 | grep -v "^--" || true
emmake make coacd -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" 2>&1

echo ""
echo "Compiling WASM module with Embind..."
mkdir -p "${OUT_DIR}"
em++ \
    -O3 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME=createCoACD \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=67108864 \
    -s ENVIRONMENT=web,worker \
    -s SINGLE_FILE=0 \
    --bind \
    -std=c++20 \
    -I"${COACD_SRC}/public" \
    -I"${COACD_SRC}/3rd/cdt/CDT/include" \
    -DWITH_3RD_PARTY_LIBS=0 \
    -DDISABLE_SPDLOG \
    "${SCRIPT_DIR}/src/coacd_bind.cpp" \
    "${BUILD_DIR}/coacd/libcoacd.a" \
    -o "${OUT_DIR}/coacd.js"

WASM_SIZE=$(wc -c < "${OUT_DIR}/coacd.wasm" | tr -d ' ')
JS_SIZE=$(wc -c < "${OUT_DIR}/coacd.js" | tr -d ' ')
echo ""
echo "Build complete:"
echo "  ${OUT_DIR}/coacd.js    (${JS_SIZE} bytes)"
echo "  ${OUT_DIR}/coacd.wasm  (${WASM_SIZE} bytes)"
