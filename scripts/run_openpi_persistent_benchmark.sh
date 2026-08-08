#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
DEPS_DIR="${DEPS_DIR:-/vepfs/zijian/alter-wam-deps}"
OPENPI_DIR="${OPENPI_DIR:-$DEPS_DIR/openpi}"
OPENPI_PYTHON="${OPENPI_PYTHON:-$OPENPI_DIR/.venv/bin/python}"
OPENPI_SITE_PACKAGES="$("$OPENPI_PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
CLIENT_PYTHON="${CLIENT_PYTHON:-$DEPS_DIR/cosmos-policy/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$DEPS_DIR/openpi-assets/pi05_libero}"
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
PORT_BASE="${PORT_BASE:-8100}"
QUERY_INTERVAL="${QUERY_INTERVAL:-5}"
MANIFEST="$(realpath "$1")"
mkdir -p "$2"
OUTPUT_ROOT="$(realpath "$2")"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

IFS=',' read -r -a gpu_ids <<< "$GPUS_CSV"
if (( ${#gpu_ids[@]} == 0 )); then
  echo "GPUS must contain at least one physical GPU id" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
RUN_MANIFEST="$OUTPUT_ROOT/manifest.json"
"$CLIENT_PYTHON" - "$MANIFEST" "$RUN_MANIFEST" "$QUERY_INTERVAL" <<'PY'
import json, pathlib, sys
source, output, query_interval = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
manifest = json.loads(source.read_text(encoding="utf-8"))
manifest["protocol"]["query_interval"] = query_interval
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
PYTHONPATH="$OPENPI_SITE_PACKAGES:$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$OPENPI_PYTHON" - "$OPENPI_DIR" "$CHECKPOINT" "$OUTPUT_ROOT/run_config.json" \
  "$GPUS_CSV" "$QUERY_INTERVAL" "$PORT_BASE" <<'PY'
import importlib.metadata
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1]).resolve()
checkpoint = pathlib.Path(sys.argv[2]).resolve()
output = pathlib.Path(sys.argv[3])
gpus = [item for item in sys.argv[4].split(",") if item]

def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None

checkpoint_bytes = (
    sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    if checkpoint.is_dir()
    else checkpoint.stat().st_size
)
payload = {
    "model": "pi0.5-LIBERO",
    "source_revision": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "checkpoint": str(checkpoint),
    "checkpoint_bytes": checkpoint_bytes,
    "physical_gpus": gpus,
    "workers": len(gpus),
    "query_interval": int(sys.argv[5]),
    "port_base": int(sys.argv[6]),
    "deterministic": True,
    "deterministic_flow_noise": True,
    "control_replay_before_event": True,
    "rollout_videos_disabled": True,
    "runtime_versions": {
        package: version(package)
        for package in ("openpi-client", "jax", "numpy", "torch")
    },
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
server_pids=()
worker_pids=()
cleanup() {
  for pid in "${worker_pids[@]:-}" "${server_pids[@]:-}"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT INT TERM

cd "$PROJECT_DIR"
for gpu in "${gpu_ids[@]}"; do
  port="$((PORT_BASE + gpu))"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  PYTHONPATH="$OPENPI_SITE_PACKAGES:$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$OPENPI_PYTHON" scripts/serve_openpi_deterministic.py \
      --port "$port" \
      --checkpoint "$CHECKPOINT" \
      > "$OUTPUT_ROOT/server-gpu$gpu.log" 2>&1 &
  server_pids+=("$!")
done

for gpu in "${gpu_ids[@]}"; do
  port="$((PORT_BASE + gpu))"
  "$CLIENT_PYTHON" - "$port" <<'PY'
import socket, sys, time
port = int(sys.argv[1])
for _ in range(600):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            raise SystemExit(0)
    except OSError:
        time.sleep(1)
raise SystemExit("policy server did not become ready on port %d" % port)
PY
done

for (( shard=0; shard<${#gpu_ids[@]}; shard++ )); do
  gpu="${gpu_ids[$shard]}"
  command=(
    "$CLIENT_PYTHON" scripts/run_openpi_persistent_shard.py "$RUN_MANIFEST"
    --output-root "$OUTPUT_ROOT"
    --shard-index "$shard"
    --num-shards "${#gpu_ids[@]}"
    --gpu-id "$gpu"
    --port-base "$PORT_BASE"
    --query-interval "$QUERY_INTERVAL"
  )
  if [[ "${RESUME:-0}" == "1" ]]; then command+=(--resume); fi
  if [[ -n "${MAX_CASES_PER_SHARD:-}" ]]; then
    command+=(--max-cases "$MAX_CASES_PER_SHARD")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
    "${command[@]}" > "$OUTPUT_ROOT/worker-gpu$gpu.log" 2>&1 &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
