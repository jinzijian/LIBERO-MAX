#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
PYTHON="${PYTHON:-/vepfs/zijian/alter-wam-deps/cosmos-policy/.venv/bin/python}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/benchmark/max5600/libero_max_5600.json}"
COSMOS_DELTA="${COSMOS_DELTA:-$PROJECT_DIR/artifacts/max5600/cosmos_policy_delta}"
COSMOS_CORE="${COSMOS_CORE:-$PROJECT_DIR/artifacts/max_hard/cosmos_core_final}"
COSMOS_OUTPUT="${COSMOS_OUTPUT:-$PROJECT_DIR/artifacts/max5600/cosmos_policy}"
PI05_OUTPUT="${PI05_OUTPUT:-$PROJECT_DIR/artifacts/max5600/pi05_libero}"
FINAL_ROOT="${FINAL_ROOT:-$PROJECT_DIR/artifacts/max5600/finalization}"

mkdir -p "$FINAL_ROOT"
cd "$PROJECT_DIR"

while pgrep -f "run_cosmos_persistent_shard.py .*cosmos_policy_delta" >/dev/null \
  || pgrep -f "run_openpi_persistent_shard.py .*pi05_libero" >/dev/null; do
  sleep 60
done

set +e
PYTHONPATH=src "$PYTHON" scripts/compose_paired_run.py \
  "$MANIFEST" "$COSMOS_DELTA" "$COSMOS_CORE" \
  --output-root "$COSMOS_OUTPUT" \
  >"$FINAL_ROOT/compose_cosmos.log" 2>&1
compose_status=$?
PYTHONPATH=src "$PYTHON" scripts/aggregate_cosmos_benchmark.py \
  "$COSMOS_OUTPUT" >"$FINAL_ROOT/aggregate_cosmos.log" 2>&1
cosmos_status=$?
PYTHONPATH=src "$PYTHON" scripts/aggregate_cosmos_benchmark.py \
  "$PI05_OUTPUT" >"$FINAL_ROOT/aggregate_pi05.log" 2>&1
pi05_status=$?
set -e

PYTHONPATH=src "$PYTHON" - "$COSMOS_OUTPUT" "$PI05_OUTPUT" <<'PY'
import json, pathlib, sys
for root_value in sys.argv[1:]:
    root = pathlib.Path(root_value)
    summary = json.loads((root / "benchmark_summary.json").read_text(encoding="utf-8"))
    if not summary["coverage"].get("execution_complete"):
        raise SystemExit("run is not execution-complete: %s" % root)
PY

for spec in \
  "cosmos2_intent:results/v1/runs/cosmos2_intent" \
  "pi05_intent:results/v1/runs/pi05_intent"; do
  name="${spec%%:*}"
  source="${spec#*:}"
  PYTHONPATH=src "$PYTHON" scripts/materialize_result_subset.py \
    "$source" benchmark/max5600/intent_96.json \
    "$FINAL_ROOT/$name" >"$FINAL_ROOT/materialize_$name.log"
done

PYTHONPATH=src "$PYTHON" scripts/build_paper_tables.py \
  --run "Cosmos-Policy=$COSMOS_OUTPUT" \
  --run "pi0.5-LIBERO=$PI05_OUTPUT" \
  --run "Cosmos-Policy-Intent=$FINAL_ROOT/cosmos2_intent" \
  --run "pi0.5-LIBERO-Intent=$FINAL_ROOT/pi05_intent" \
  --output-dir "$FINAL_ROOT/tables" \
  >"$FINAL_ROOT/build_tables.log"

"$PYTHON" - "$FINAL_ROOT" "$compose_status" "$cosmos_status" "$pi05_status" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
payload = {
    "execution_complete": True,
    "compose_exit_status": int(sys.argv[2]),
    "cosmos_aggregate_exit_status": int(sys.argv[3]),
    "pi05_aggregate_exit_status": int(sys.argv[4]),
}
(root / "DONE.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
