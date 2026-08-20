#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEPS_DIR="${DEPS_DIR:-$PROJECT_DIR/.deps}"
export LIBERO_IMPL_DIR="${LIBERO_PRO_DIR:-$DEPS_DIR/LIBERO-PRO}"
export LIBERO_OVERLAY="${LIBERO_PRO_OVERLAY:-$DEPS_DIR/libero-pro-python-overlay}"
export LIBERO_CONFIG_PATH="${LIBERO_PRO_CONFIG:-$DEPS_DIR/libero-pro-config}"
export PYTHONPATH="$DEPS_DIR/robosuite-1.4.0:$LIBERO_OVERLAY:$DEPS_DIR/cosmos-policy/.venv/lib/python3.10/site-packages:$DEPS_DIR/.venv-libero/lib/python3.10/site-packages:$LIBERO_IMPL_DIR:$DEPS_DIR/cosmos-policy:$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PROJECT_DIR/scripts/run_openpi_persistent_benchmark.sh" "$@"
