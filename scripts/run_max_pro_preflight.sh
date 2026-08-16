#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
DEPS_DIR="${DEPS_DIR:-/vepfs/zijian/alter-wam-deps}"
export LIBERO_IMPL_DIR="${LIBERO_PRO_DIR:-$DEPS_DIR/LIBERO-PRO}"
export LIBERO_OVERLAY="${LIBERO_PRO_OVERLAY:-$DEPS_DIR/libero-pro-python-overlay}"
export LIBERO_CONFIG_PATH="${LIBERO_PRO_CONFIG:-$DEPS_DIR/libero-pro-config}"
exec "$PROJECT_DIR/scripts/run_max_hard_preflight.sh" "$@"
