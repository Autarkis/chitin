#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# CoACD's C++ CoACD() entry point is version-locked to the 18-argument signature
# that src/coacd_bind.cpp calls positionally. Pin the tag so an upstream API
# change can't silently break the build; bump it deliberately, in lockstep with
# the binding. Override with COACD_REF=<tag> to test another release.
COACD_REF="${COACD_REF:-1.0.11}"
COACD_SRC="${COACD_SRC:-/tmp/coacd-src}"
BUILD_DIR="${SCRIPT_DIR}/build"
OUT_DIR="${SCRIPT_DIR}/dist"

if ! command -v emcc &>/dev/null; then
    echo "emcc not found. Source emsdk_env.sh first."
    exit 1
fi

if [ ! -f "${COACD_SRC}/public/coacd.h" ]; then
    echo "CoACD source not at ${COACD_SRC}; cloning ${COACD_REF}..."
    git clone --depth 1 --branch "${COACD_REF}" --recurse-submodules \
        https://github.com/SarahWeiii/CoACD.git "${COACD_SRC}"
fi

NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "Building CoACD static library for WASM (CoACD ${COACD_REF})..."
mkdir -p "${BUILD_DIR}/coacd"
cd "${BUILD_DIR}/coacd"
# We link libcoacd.a directly and never `make install`, so skip install-rule
# generation. CoACD 1.0.11's install(EXPORT "CoACDTargets") fails its dependency
# check under this configure ("_coacd" requires "coacd" not in any export set);
# skipping install rules removes that error at the source rather than masking a
# failed configure with `|| true`.
emcmake cmake "${COACD_SRC}" \
    -DWITH_3RD_PARTY_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SKIP_INSTALL_RULES=ON \
    -Wno-dev
emmake make coacd -j"${NPROC}"

echo ""
echo "Compiling WASM module with Embind..."
mkdir -p "${OUT_DIR}"
# ENVIRONMENT includes node so the module can be loaded and functionally tested
# in CI (test/decompose.test.cjs); web and worker are the shipping targets.
em++ \
    -O3 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME=createCoACD \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=67108864 \
    -s ENVIRONMENT=web,worker,node \
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
