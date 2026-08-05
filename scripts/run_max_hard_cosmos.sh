#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
DEPS_DIR="${DEPS_DIR:-/vepfs/zijian/alter-wam-deps}"
PYTHON="${PYTHON:-$DEPS_DIR/cosmos-policy/.venv/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
T5_EMBEDDINGS="${T5_EMBEDDINGS:-$DEPS_DIR/cosmos-assets/Cosmos-Policy-LIBERO-Predict2-2B/libero_plus_t5_embeddings.pkl}"
MANIFEST="$(realpath "$1")"
mkdir -p "$2"
OUTPUT_ROOT="$(realpath "$2")"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$DEPS_DIR/libero-plus-config}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$DEPS_DIR/robosuite-1.4.0:$DEPS_DIR/libero-plus-python-overlay:$DEPS_DIR/cosmos-policy/.venv/lib/python3.10/site-packages:$DEPS_DIR/.venv-libero/lib/python3.10/site-packages:$DEPS_DIR/LIBERO-plus:$DEPS_DIR/cosmos-policy:$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

command=(
  "$PYTHON"
  "$PROJECT_DIR/scripts/run_cosmos_persistent_benchmark.py"
  "$MANIFEST"
  --output-root "$OUTPUT_ROOT"
  --gpus "$GPUS"
  --t5-embeddings "$T5_EMBEDDINGS"
)
if [[ -n "${MAX_CASES_PER_SHARD:-}" ]]; then
  command+=(--max-cases-per-shard "$MAX_CASES_PER_SHARD")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  command+=(--resume)
fi
exec "${command[@]}"
