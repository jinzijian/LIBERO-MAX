#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
DEPS_DIR="${DEPS_DIR:-/vepfs/zijian/alter-wam-deps}"
FASTWAM_ROOT="${FASTWAM_ROOT:-$DEPS_DIR/FastWAM}"
PYTHON="${PYTHON:-$FASTWAM_ROOT/.venv/bin/python}"
LIBERO_IMPL_DIR="${LIBERO_PRO_DIR:-$DEPS_DIR/LIBERO-PRO}"
LIBERO_OVERLAY="${LIBERO_PRO_OVERLAY:-$DEPS_DIR/libero-pro-python-overlay}"
CHECKPOINT="${FASTWAM_CHECKPOINT:-$DEPS_DIR/model-assets/fastwam-libero/libero_uncond_2cam224.pt}"
DATASET_STATS="${FASTWAM_DATASET_STATS:-$DEPS_DIR/model-assets/fastwam-libero/libero_uncond_2cam224_dataset_stats.json}"
GPUS="${GPUS:-0,1,2,3}"

export LIBERO_CONFIG_PATH="${LIBERO_PRO_CONFIG:-$DEPS_DIR/libero-pro-config}"
export HF_HOME="${HF_HOME:-$DEPS_DIR/hf-cache-fastwam}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$DEPS_DIR/robosuite-1.4.0:$LIBERO_OVERLAY:$DEPS_DIR/.venv-libero/lib/python3.10/site-packages:$LIBERO_IMPL_DIR:$FASTWAM_ROOT:$FASTWAM_ROOT/experiments/libero:$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

command=(
  "$PYTHON" "$PROJECT_DIR/scripts/run_fastwam_persistent_benchmark.py"
  "$(realpath "$1")" --output-root "$(realpath -m "$2")"
  --gpus "$GPUS" --fastwam-root "$FASTWAM_ROOT"
  --checkpoint "$CHECKPOINT" --dataset-stats "$DATASET_STATS"
)
if [[ -n "${MAX_CASES_PER_SHARD:-}" ]]; then
  command+=(--max-cases-per-shard "$MAX_CASES_PER_SHARD")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  command+=(--resume)
fi
exec "${command[@]}"
