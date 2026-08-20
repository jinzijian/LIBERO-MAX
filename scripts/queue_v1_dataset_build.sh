#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
POLL_SECONDS="${POLL_SECONDS:-30}"
IDLE_POLLS="${IDLE_POLLS:-10}"
MAX_GPU_MEMORY_MIB="${MAX_GPU_MEMORY_MIB:-1000}"

stable=0
while (( stable < IDLE_POLLS )); do
  competing="$(pgrep -fc 'run_cosmos_policy_libero_subset.py|run_cosmos_policy_libero_plus_tta_manifest.sh|run_cosmos_benchmark.py' || true)"
  busy="$({
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  } | awk -v limit="$MAX_GPU_MEMORY_MIB" '$1 >= limit {count++} END {print count + 0}')"
  if (( competing == 0 && busy == 0 )); then
    stable=$((stable + 1))
  else
    stable=0
  fi
  printf '%s competing=%s busy_gpus=%s stable=%s/%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S%z)" \
    "$competing" "$busy" "$stable" "$IDLE_POLLS"
  if (( stable < IDLE_POLLS )); then
    sleep "$POLL_SECONDS"
  fi
done

cd "$PROJECT_DIR"
bash scripts/run_v1_dataset_build.sh
