#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
DEPS_DIR="${DEPS_DIR:-/vepfs/zijian/alter-wam-deps}"
PYTHON="${PYTHON:-$DEPS_DIR/cosmos-policy/.venv/bin/python}"
BUILD_ROOT="${BUILD_ROOT:-$PROJECT_DIR/artifacts/libero_max_v1_release_build}"
RELEASE_DIR="${RELEASE_DIR:-$PROJECT_DIR/benchmark/v1}"
NUM_SHARDS="${NUM_SHARDS:-8}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_CONFIG_PATH="$DEPS_DIR/libero-config"
export PYTHONPATH="$PROJECT_DIR/src:$DEPS_DIR/robosuite-1.4.0:$DEPS_DIR/cosmos-policy/.venv/lib/python3.10/site-packages:$DEPS_DIR/.venv-libero/lib/python3.10/site-packages:$DEPS_DIR/LIBERO:$DEPS_DIR/cosmos-policy"

core_manifest="$BUILD_ROOT/core_filtered.json"
full_manifest="$BUILD_ROOT/full_filtered.json"
preflight_dir="$BUILD_ROOT/preflight-final"
merged_preflight="$BUILD_ROOT/physical_preflight.json"
feasibility_filter="$BUILD_ROOT/feasibility_filter.json"
calibrated_catalog="$BUILD_ROOT/task_catalog_calibrated.json"

for required in \
  "$core_manifest" \
  "$full_manifest" \
  "$feasibility_filter" \
  "$calibrated_catalog"; do
  if [[ ! -s "$required" ]]; then
    echo "missing final-preflight input: $required" >&2
    exit 1
  fi
done

mkdir -p "$preflight_dir"
cd "$PROJECT_DIR"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" \
    scripts/preflight_manifest_interventions.py "$core_manifest" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$shard" \
    --output "$preflight_dir/shard-$shard.json" \
    > "$preflight_dir/shard-$shard.log" 2>&1 &
  pids+=("$!")
done

shard_status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    shard_status=1
  fi
done

set +e
"$PYTHON" scripts/merge_preflight_reports.py \
  "$preflight_dir"/shard-*.json \
  --output "$merged_preflight"
merge_status=$?
set -e
if (( shard_status != 0 || merge_status != 0 )); then
  echo "filtered v1 physical preflight incomplete; inspect $BUILD_ROOT" >&2
  exit 1
fi

"$PYTHON" scripts/freeze_v1_release.py \
  --catalog "$calibrated_catalog" \
  --core "$core_manifest" \
  --full "$full_manifest" \
  --preflight "$merged_preflight" \
  --feasibility-filter "$feasibility_filter" \
  --output-dir "$RELEASE_DIR"

echo "LIBERO-MAX v1 test sets frozen at $RELEASE_DIR"
