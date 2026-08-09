#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$1" = "--help" ]; then
  echo "start_demo.sh"
  echo "Builds the local Chitin browser packages, starts Collider Lab on port 4179,"
  echo "and opens it in the default browser. Keep the terminal open while using it."
  exit 0
fi

open_url() {
  local url="$1"
  if command -v open >/dev/null 2>&1; then
    open "$url"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url"
  fi
}

if page="$(curl -s --max-time 2 'http://127.0.0.1:4179/' 2>/dev/null)" && \
   printf '%s' "$page" | grep -q '<title>Chitin Collider Lab</title>'; then
  echo "Chitin Collider Lab is already running. Opening it now..."
  open_url "http://127.0.0.1:4179/"
  exit 0
fi

if ! command -v npm >/dev/null 2>&1; then
  echo
  echo "Chitin Collider Lab needs Node.js and npm."
  echo "Install the current Node.js LTS release, then run start_demo.sh again."
  echo
  exit 1
fi

build_package() {
  local dir="$1"
  (
    cd "$dir"
    if [ ! -f "node_modules/.bin/tsc" ]; then
      echo "Installing dependencies for $dir..."
      npm ci
    fi
    echo "Building $dir..."
    npm run build
  )
}

echo "Preparing Chitin Collider Lab..."

if ! build_package "../../integrations/wasm-lite"; then
  echo
  echo "The demo could not be prepared. Review the error above, then try again."
  echo
  exit 1
fi

if ! build_package "../../integrations/web"; then
  echo
  echo "The demo could not be prepared. Review the error above, then try again."
  echo
  exit 1
fi

if [ ! -f "node_modules/vite/bin/vite.js" ]; then
  echo "Installing demo dependencies..."
  if ! npm ci; then
    echo
    echo "The demo could not be prepared. Review the error above, then try again."
    echo
    exit 1
  fi
fi

echo
echo "Opening http://127.0.0.1:4179/"
echo "Keep this window open. Press Ctrl+C here to stop the demo."
echo
npm start
