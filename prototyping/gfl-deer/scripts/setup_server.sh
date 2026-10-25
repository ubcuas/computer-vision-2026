#!/usr/bin/env bash
# Same pinned stack as the verified T4 run, with server-local paths.
set -euo pipefail
: "${DEER_WORKSPACE:?Set DEER_WORKSPACE to an absolute external data directory}"
export GFL_ENV="${GFL_ENV:-$DEER_WORKSPACE/.venv-gfl}"
mkdir -p "$DEER_WORKSPACE"
bash "$(dirname "$0")/setup_colab.sh"
