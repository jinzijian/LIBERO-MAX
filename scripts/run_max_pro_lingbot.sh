#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEPS_DIR="${DEPS_DIR:-$PROJECT_DIR/.deps}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$DEPS_DIR/lingbot-va}"
PYTHON="${PYTHON:-$LINGBOT_ROOT/.venv/bin/python}"
ACTIVE_SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
LIBERO_IMPL_DIR="${LIBERO_IMPL_DIR:-${LIBERO_PRO_DIR:-$DEPS_DIR/LIBERO-PRO}}"
LIBERO_OVERLAY="${LIBERO_OVERLAY:-${LIBERO_PRO_OVERLAY:-$DEPS_DIR/libero-pro-python-overlay}}"
CHECKPOINT="${LINGBOT_CHECKPOINT:-$DEPS_DIR/model-assets/lingbot-va-libero-long}"
GPUS="${GPUS:-0,1,2,3}"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_PRO_CONFIG:-$DEPS_DIR/libero-pro-config}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$ACTIVE_SITE_PACKAGES:$DEPS_DIR/robosuite-1.4.0:$LIBERO_OVERLAY:$DEPS_DIR/.venv-libero/lib/python3.10/site-packages:$LIBERO_IMPL_DIR:$LINGBOT_ROOT:$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - <<'PY'
import mujoco

if mujoco.__version__ != "3.2.6" or not hasattr(mujoco.MjModel, "mesh_scale"):
    raise SystemExit(
        "LingBot LIBERO runtime requires mujoco==3.2.6 with MjModel.mesh_scale; "
        "install it in the active model venv"
    )
PY

command=(
  "$PYTHON" "$PROJECT_DIR/scripts/run_lingbot_persistent_benchmark.py"
  "$(realpath "$1")" --output-root "$(realpath -m "$2")"
  --gpus "$GPUS" --lingbot-root "$LINGBOT_ROOT" --checkpoint "$CHECKPOINT"
)
if [[ -n "${MAX_CASES_PER_SHARD:-}" ]]; then
  command+=(--max-cases-per-shard "$MAX_CASES_PER_SHARD")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  command+=(--resume)
fi
exec "${command[@]}"
