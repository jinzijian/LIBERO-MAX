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

mkdir -p "$BUILD_ROOT/preflight"
cd "$PROJECT_DIR"

raw_catalog="$BUILD_ROOT/task_catalog_raw.json"
calibrated_catalog="$BUILD_ROOT/task_catalog_calibrated.json"
calibration_report="$BUILD_ROOT/relocation_calibration.json"
core_manifest="$BUILD_ROOT/core_candidate.json"
full_manifest="$BUILD_ROOT/full_candidate.json"
merged_preflight="$BUILD_ROOT/physical_preflight.json"

"$PYTHON" scripts/build_libero_task_catalog.py \
  --bddl-root "$DEPS_DIR/LIBERO/libero/libero/bddl_files" \
  --output "$raw_catalog"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/calibrate_relocation_directions.py \
  "$raw_catalog" \
  --output-catalog "$calibrated_catalog" \
  --report "$calibration_report"

"$PYTHON" scripts/build_randomized_v1_manifest.py \
  "$calibrated_catalog" --profile core --output "$core_manifest"
"$PYTHON" scripts/build_randomized_v1_manifest.py \
  "$calibrated_catalog" --profile full --output "$full_manifest"

pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" \
    scripts/preflight_manifest_interventions.py "$core_manifest" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$shard" \
    --output "$BUILD_ROOT/preflight/shard-$shard.json" \
    > "$BUILD_ROOT/preflight/shard-$shard.log" 2>&1 &
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
  "$BUILD_ROOT"/preflight/shard-*.json \
  --output "$merged_preflight"
merge_status=$?
set -e
if (( shard_status != 0 || merge_status != 0 )); then
  echo "v1 physical preflight incomplete; inspect $BUILD_ROOT" >&2
  exit 1
fi

"$PYTHON" scripts/freeze_v1_release.py \
  --catalog "$calibrated_catalog" \
  --core "$core_manifest" \
  --full "$full_manifest" \
  --preflight "$merged_preflight" \
  --output-dir "$RELEASE_DIR"

echo "LIBERO-MAX v1 test sets frozen at $RELEASE_DIR"
