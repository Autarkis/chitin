#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# CoACD's C++ CoACD() entry point is version-locked to the 18-argument signature
# that src/coacd_bind.cpp calls positionally. Pin the tag so an upstream API
# change can't silently break the build; bump it deliberately, in lockstep with
# the binding. Override with COACD_REF=<tag> to test another release.
COACD_REF="${COACD_REF:-1.0.11}"
COACD_SRC="${COACD_SRC:-/tmp/coacd-src}"
# PoissonRecon has no release tags. Pin to a commit hash; bump deliberately.
POISSON_REF="${POISSON_REF:-262b0f5}"
POISSON_SRC="${POISSON_SRC:-/tmp/poisson-src}"
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

if [ ! -f "${POISSON_SRC}/Src/Reconstructors.h" ]; then
    echo "PoissonRecon source not at ${POISSON_SRC}; cloning..."
    git clone https://github.com/mkazhdan/PoissonRecon.git "${POISSON_SRC}"
    git -C "${POISSON_SRC}" checkout "${POISSON_REF}"
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
# EXPORT_ES6 emits a real ES module (coacd.mjs) so `import(url)` loads it cleanly
# from a CDN in the browser and from Node -- a MODULARIZE UMD would throw on its
# `module`/`exports` references when imported as ESM. ENVIRONMENT includes node
# so the module can be functionally tested in CI (test/decompose.test.mjs); web
# and worker are the shipping targets.
em++ \
    -O3 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
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
    -o "${OUT_DIR}/coacd.mjs"

echo ""
echo "Compiling PoissonRecon WASM module with Embind..."
# PoissonRecon is header-only (template-heavy). No static library build needed —
# poisson_bind.cpp includes the headers and instantiates the templates directly.
# The library's ThreadPool falls back to serial execution in single-threaded WASM.
#
# 32-bit fix: NestedVector uses `((size_t)1) << (LogSize * (Depth+1))` with
# LogSize=20, Depth=1 → shift count 40, which overflows wasm32's 32-bit size_t.
# Reducing NESTED_VECTOR_LEVELS to 0 keeps only the base case (1<<20 = 1M entries,
# sufficient for Poisson depths ≤ 10). The define is unguarded in PreProcessor.h,
# so we patch it in-place before compiling.
sed -i 's/^#define NESTED_VECTOR_LEVELS 1/#define NESTED_VECTOR_LEVELS 0/' \
    "${POISSON_SRC}/Src/PreProcessor.h"
em++ \
    -O3 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s EXPORT_NAME=createPoisson \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=67108864 \
    -s ENVIRONMENT=web,worker,node \
    -s SINGLE_FILE=0 \
    --bind \
    -std=c++20 \
    -I"${POISSON_SRC}/Src" \
    "${SCRIPT_DIR}/src/poisson_bind.cpp" \
    -o "${OUT_DIR}/poisson.mjs"

COACD_WASM_SIZE=$(wc -c < "${OUT_DIR}/coacd.wasm" | tr -d ' ')
COACD_JS_SIZE=$(wc -c < "${OUT_DIR}/coacd.mjs" | tr -d ' ')
POISSON_WASM_SIZE=$(wc -c < "${OUT_DIR}/poisson.wasm" | tr -d ' ')
POISSON_JS_SIZE=$(wc -c < "${OUT_DIR}/poisson.mjs" | tr -d ' ')
echo ""
echo "Build complete:"
echo "  ${OUT_DIR}/coacd.mjs     (${COACD_JS_SIZE} bytes)"
echo "  ${OUT_DIR}/coacd.wasm    (${COACD_WASM_SIZE} bytes)"
echo "  ${OUT_DIR}/poisson.mjs   (${POISSON_JS_SIZE} bytes)"
echo "  ${OUT_DIR}/poisson.wasm  (${POISSON_WASM_SIZE} bytes)"
