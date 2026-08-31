#!/usr/bin/env bash
set -euo pipefail

# Publish npm packages whose local version is ahead of the registry.
# Called by changesets/action after a "Version Packages" PR merges.
#
# chitin-wasm is excluded — its WASM artifacts require an Emscripten build
# that only runs in the dedicated release-wasm.yml workflow (triggered by
# a wasm-v* tag). Changesets still manages its version and CHANGELOG.

publish_if_newer() {
  local dir="$1"
  local name
  name=$(node -p "require('./${dir}/package.json').name")
  local local_ver
  local_ver=$(node -p "require('./${dir}/package.json').version")
  local registry_ver
  registry_ver=$(npm view "${name}" version 2>/dev/null || echo "0.0.0")

  if [ "$local_ver" = "$registry_ver" ]; then
    echo "✓ ${name}@${local_ver} already published — skipping"
    return 0
  fi

  echo "→ Publishing ${name}@${local_ver} (registry has ${registry_ver})"
  (cd "$dir" && npm ci && npm run build && npm publish --access public)
  echo "✓ ${name}@${local_ver} published"
}

echo "=== chitin-wasm ==="
echo "⊘ @autarkis/chitin-wasm is excluded (requires Emscripten build; use release-wasm.yml)"

echo ""
echo "=== chitin-lite ==="
publish_if_newer "integrations/wasm-lite"

echo ""
echo "=== chitin-web ==="
publish_if_newer "integrations/web"
