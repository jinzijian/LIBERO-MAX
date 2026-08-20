#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEPS_DIR="${DEPS_DIR:-$PROJECT_DIR/.deps}"
COSMOS_PYTHON="${COSMOS_PYTHON:-$DEPS_DIR/cosmos-policy/.venv/bin/python}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/examples/manifests/cosmos_physical_pilot_v0.1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/artifacts/cosmos_physical_pilot_v0.1_seed195_20260803}"
GPUS="${GPUS:-0,1,2,3,4}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"
IDLE_POLLS="${IDLE_POLLS:-10}"
MAX_GPU_MEMORY_MIB="${MAX_GPU_MEMORY_MIB:-1000}"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%S%z
}

if [[ -n "$WAIT_FOR_PID" ]]; then
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    printf '%s waiting_for_pid=%s\n' "$(timestamp)" "$WAIT_FOR_PID"
    sleep "$POLL_SECONDS"
  done
fi

stable=0
while (( stable < IDLE_POLLS )); do
  competing="$(pgrep -fc 'run_cosmos_policy_libero_subset.py|run_cosmos_policy_libero_plus_tta_manifest.sh' || true)"
  busy="$({
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
  } | awk -F',' -v selected="$GPUS" -v limit="$MAX_GPU_MEMORY_MIB" '
    BEGIN {
      n = split(selected, ids, ",")
      for (i = 1; i <= n; i++) wanted[ids[i] + 0] = 1
    }
    {
      gpu = $1 + 0
      memory = $2 + 0
      if (wanted[gpu] && memory >= limit) count++
    }
    END { print count + 0 }
  ')"
  if (( competing == 0 && busy == 0 )); then
    stable=$((stable + 1))
  else
    stable=0
  fi
  printf '%s competing=%s busy_selected_gpus=%s stable=%s/%s\n' \
    "$(timestamp)" "$competing" "$busy" "$stable" "$IDLE_POLLS"
  if (( stable < IDLE_POLLS )); then
    sleep "$POLL_SECONDS"
  fi
done

cd "$PROJECT_DIR"
printf '%s pilot_start output=%s gpus=%s\n' "$(timestamp)" "$OUTPUT_ROOT" "$GPUS"
PYTHONPATH=src "$COSMOS_PYTHON" scripts/run_cosmos_benchmark.py \
  "$MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --gpus "$GPUS" \
  --resume
PYTHONPATH=src "$COSMOS_PYTHON" scripts/aggregate_cosmos_benchmark.py "$OUTPUT_ROOT"
printf '%s pilot_complete output=%s\n' "$(timestamp)" "$OUTPUT_ROOT"
