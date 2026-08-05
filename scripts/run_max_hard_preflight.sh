#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
DEPS_DIR="${DEPS_DIR:-/vepfs/zijian/alter-wam-deps}"
PYTHON="${PYTHON:-$DEPS_DIR/cosmos-policy/.venv/bin/python}"
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
MANIFEST="$(realpath "$1")"
mkdir -p "$2"
OUTPUT_ROOT="$(realpath "$2")"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$DEPS_DIR/libero-plus-config}"
export PYTHONPATH="$DEPS_DIR/robosuite-1.4.0:$DEPS_DIR/libero-plus-python-overlay:$DEPS_DIR/cosmos-policy/.venv/lib/python3.10/site-packages:$DEPS_DIR/.venv-libero/lib/python3.10/site-packages:$DEPS_DIR/LIBERO-plus:$DEPS_DIR/cosmos-policy:$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

IFS=',' read -r -a gpu_ids <<< "$GPUS_CSV"
if (( ${#gpu_ids[@]} == 0 )); then
  echo "GPUS must contain at least one physical GPU id" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/shards" "$OUTPUT_ROOT/logs"
for (( shard=0; shard<${#gpu_ids[@]}; shard++ )); do
  if [[ -e "$OUTPUT_ROOT/shards/shard-$shard.json" ]]; then
    echo "refusing to overwrite existing shard: $OUTPUT_ROOT/shards/shard-$shard.json" >&2
    exit 1
  fi
done

cd "$PROJECT_DIR"
pids=()
for (( shard=0; shard<${#gpu_ids[@]}; shard++ )); do
  gpu="${gpu_ids[$shard]}"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" "$PYTHON" \
    scripts/preflight_manifest_interventions.py "$MANIFEST" \
    --num-shards "${#gpu_ids[@]}" \
    --shard-index "$shard" \
    --output "$OUTPUT_ROOT/shards/shard-$shard.json" \
    > "$OUTPUT_ROOT/logs/shard-$shard.log" 2>&1 &
  pids+=("$!")
done

worker_status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    worker_status=1
  fi
done

set +e
"$PYTHON" scripts/merge_preflight_reports.py \
  "$OUTPUT_ROOT"/shards/shard-*.json \
  --output "$OUTPUT_ROOT/physical_preflight.json"
merge_status=$?
set -e
if (( worker_status != 0 || merge_status != 0 )); then
  echo "preflight found invalid configurations; inspect $OUTPUT_ROOT" >&2
  exit 1
fi

echo "preflight complete: $OUTPUT_ROOT/physical_preflight.json"
