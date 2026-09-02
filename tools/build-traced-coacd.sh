#!/usr/bin/env bash
# Build traced CoACD DLL from pinned upstream + instrumentation patches.
# Requires: git, cmake, MSVC (via VS Developer prompt or vcvarsall).
set -euo pipefail

UPSTREAM_REPO="https://github.com/mhamber/CoACD.git"
UPSTREAM_TAG="v1.0.14"
UPSTREAM_SHA="1401ce2"
EXPECTED_DLL_SHA="dd295d37ad6579545f1017c7125bfe8daab65b52a9ff1853a104b8a2851853d3"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/patches"
BUILD_DIR="$SCRIPT_DIR/coacd-build"

echo "=== Traced CoACD build ==="
echo "Upstream: $UPSTREAM_TAG ($UPSTREAM_SHA)"

# Clean checkout
rm -rf "$BUILD_DIR"
git clone --depth 1 --branch "$UPSTREAM_TAG" "$UPSTREAM_REPO" "$BUILD_DIR"
cd "$BUILD_DIR"

actual_sha=$(git rev-parse --short HEAD)
if [ "$actual_sha" != "$UPSTREAM_SHA" ]; then
    echo "ERROR: upstream SHA mismatch: expected $UPSTREAM_SHA, got $actual_sha"
    exit 1
fi

# Apply patches
echo "Applying instrumentation patches..."
git apply "$PATCH_DIR/0001-trace-concatenated-stream-trace-hook-with-ZIP64-flus.patch"
git apply "$PATCH_DIR/0002-trace-v2-instrumentation-hooks-in-clip-cost-mcts-pro.patch"
echo "Patches applied."

# Build
mkdir -p build && cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release

# Locate DLL
DLL=$(find . -name "lib_coacd.dll" -path "*/Release/*" | head -1)
if [ -z "$DLL" ]; then
    echo "ERROR: lib_coacd.dll not found in build output"
    exit 1
fi

# Verify hash
actual_dll_sha=$(sha256sum "$DLL" | cut -d' ' -f1)
echo "DLL SHA-256: $actual_dll_sha"
if [ "$actual_dll_sha" != "$EXPECTED_DLL_SHA" ]; then
    echo "WARNING: DLL hash differs from contract."
    echo "  Expected: $EXPECTED_DLL_SHA"
    echo "  Got:      $actual_dll_sha"
    echo "This is expected if toolchain versions differ. Update BUILD_CONTRACT.md."
else
    echo "DLL hash matches contract."
fi

echo ""
echo "DLL at: $(realpath "$DLL")"
echo "To deploy: cp \"$DLL\" \"\$(python -c 'import coacd; print(coacd.__path__[0])')/lib_coacd.dll\""
