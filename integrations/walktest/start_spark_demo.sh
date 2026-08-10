#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$1" = "--help" ]; then
  echo "start_spark_demo.sh"
  echo "Builds the local Chitin browser packages, starts the Spark demo on port 3219,"
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

if page="$(curl -s --max-time 2 'http://127.0.0.1:3219/spark-demo' 2>/dev/null)" && \
   printf '%s' "$page" | grep -q '<title>Chitin .* Spark.*Hull Shadows on Splats</title>'; then
  echo "Chitin Spark demo is already running. Opening it now..."
  open_url "http://127.0.0.1:3219/spark-demo"
  exit 0
fi

if ! command -v npm >/dev/null 2>&1; then
  echo
  echo "The Chitin Spark demo needs Node.js and npm."
  echo "Install the current Node.js LTS release, then run start_spark_demo.sh again."
  echo
  exit 1
fi

if [ ! -f "node_modules/serve/build/main.js" ]; then
  echo "Installing demo dependencies..."
  if ! npm ci; then
    echo
    echo "The demo could not be prepared. Review the error above, then try again."
    echo
    exit 1
  fi
fi

echo "Preparing the Chitin Spark demo..."
if ! npm run prepare:release-candidate; then
  echo
  echo "The demo could not be prepared. Review the error above, then try again."
  echo
  exit 1
fi
if ! npm run build:spark-demo; then
  echo
  echo "The demo could not be prepared. Review the error above, then try again."
  echo
  exit 1
fi

echo
echo "Keep this window open. Press Ctrl+C here to stop the demo."
echo
npm run serve:spark
