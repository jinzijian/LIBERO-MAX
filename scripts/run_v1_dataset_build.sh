#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEPS_DIR="${DEPS_DIR:-$PROJECT_DIR/.deps}"
PYTHON="${PYTHON:-$DEPS_DIR/cosmos-policy/.venv/bin/python}"
BUILD_ROOT="${BUILD_ROOT:-$PROJECT_DIR/artifacts/libero_max_v1_release_build}"
RELEASE_DIR="${RELEASE_DIR:-$PROJECT_DIR/benchmark/v1}"
NUM_SHARDS="${NUM_SHARDS:-8}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_CONFIG_PATH="$DEPS_DIR/libero-config"
export PYTHONPATH="$PROJECT_DIR/src:$DEPS_DIR/robosuite-1.4.0:$DEPS_DIR/cosmos-policy/.venv/lib/python3.10/site-packages:$DEPS_DIR/.venv-libero/lib/python3.10/site-packages:$DEPS_DIR/LIBERO:$DEPS_DIR/cosmos-policy"

mkdir -p "$BUILD_ROOT"
cd "$PROJECT_DIR"

raw_catalog="$BUILD_ROOT/task_catalog_raw.json"
calibrated_catalog="$BUILD_ROOT/task_catalog_calibrated.json"
calibration_report="$BUILD_ROOT/relocation_calibration.json"
core_candidate="$BUILD_ROOT/core_candidate.json"
full_candidate="$BUILD_ROOT/full_candidate.json"
candidate_preflight="$BUILD_ROOT/candidate_physical_preflight.json"
feasibility_filter="$BUILD_ROOT/feasibility_filter.json"
core_manifest="$BUILD_ROOT/core_filtered.json"
full_manifest="$BUILD_ROOT/full_filtered.json"
merged_preflight="$BUILD_ROOT/physical_preflight.json"

"$PYTHON" scripts/build_libero_task_catalog.py \
  --bddl-root "$DEPS_DIR/LIBERO/libero/libero/bddl_files" \
  --output "$raw_catalog"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/calibrate_relocation_directions.py \
  "$raw_catalog" \
  --output-catalog "$calibrated_catalog" \
  --report "$calibration_report"

"$PYTHON" scripts/build_randomized_v1_manifest.py \
  "$calibrated_catalog" --profile core --output "$core_candidate"
"$PYTHON" scripts/build_randomized_v1_manifest.py \
  "$calibrated_catalog" --profile full --output "$full_candidate"

run_preflight_round() {
  local manifest="$1"
  local round_dir="$2"
  local output="$3"
  mkdir -p "$round_dir"
  local pids=()
  local shard
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" \
      scripts/preflight_manifest_interventions.py "$manifest" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      --output "$round_dir/shard-$shard.json" \
      > "$round_dir/shard-$shard.log" 2>&1 &
    pids+=("$!")
  done
  local shard_status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      shard_status=1
    fi
  done
  local merge_status=0
  "$PYTHON" scripts/merge_preflight_reports.py \
    "$round_dir"/shard-*.json --output "$output" || merge_status=$?
  if (( shard_status != 0 || merge_status != 0 )); then
    return 1
  fi
}

candidate_status=0
run_preflight_round \
  "$core_candidate" "$BUILD_ROOT/preflight-candidate" "$candidate_preflight" \
  || candidate_status=$?
if [[ ! -s "$candidate_preflight" ]]; then
  echo "candidate physical preflight did not produce a complete report" >&2
  exit 1
fi
if (( candidate_status != 0 )); then
  echo "candidate preflight found infeasible configurations; applying filter"
fi

"$PYTHON" scripts/prune_infeasible_configurations.py \
  --core "$core_candidate" \
  --full "$full_candidate" \
  --preflight "$candidate_preflight" \
  --output-core "$core_manifest" \
  --output-full "$full_manifest" \
  --report "$feasibility_filter"

if ! run_preflight_round \
  "$core_manifest" "$BUILD_ROOT/preflight-final" "$merged_preflight"; then
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
