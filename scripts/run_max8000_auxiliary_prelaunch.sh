#!/usr/bin/env bash
set -euo pipefail

# Prelaunch the two q16 comparison models on otherwise idle GPUs while the
# main paper queue owns GPUs 0-3. Every model receives a real simulator smoke
# before the frozen 800-case run. The main queue consumes the completion
# markers and resumes partial roots if this process is interrupted.

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
DEPS_DIR="${DEPS_DIR:-/vepfs/zijian/alter-wam-deps}"
PYTHON="${PYTHON:-$DEPS_DIR/cosmos-policy/.venv/bin/python}"
GPUS="${GPUS:-4,5,6,7}"
SMOKE_GPU="${SMOKE_GPU:-${GPUS%%,*}}"
FINAL_ROOT="${FINAL_ROOT:-$PROJECT_DIR/artifacts/max8000/paper_final}"
MANIFEST="$PROJECT_DIR/benchmark/max8000/libero_max_pro_model_comparison_800.json"
SMOKE_MANIFEST="$FINAL_ROOT/work/pro-runtime-compatibility-smoke.json"

mkdir -p "$FINAL_ROOT/logs" "$FINAL_ROOT/work"
cd "$PROJECT_DIR"
export PROJECT_DIR DEPS_DIR
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$SMOKE_MANIFEST" ]]; then
  "$PYTHON" - "$MANIFEST" "$SMOKE_MANIFEST" <<'PY'
import json, pathlib, sys
source, output = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(source.read_text(encoding="utf-8"))
wanted_categories = (
    "initial_pose_position_angle",
    "object_shape",
    "view_occlusion",
)
selected = []
for category in wanted_categories:
    candidates = [
        case
        for case in manifest["cases"]
        if case["substrate_category"] == "LIBERO-PRO/" + category
    ]
    if not candidates:
        raise SystemExit("frozen compatibility-smoke category is missing: " + category)
    selected.append(min(candidates, key=lambda case: case["case_id"]))
manifest["benchmark_id"] += "-pro-runtime-compatibility-smoke"
manifest["cases"] = selected
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

run_model() {
  local label="$1"
  local launcher="$2"
  local smoke_root="$FINAL_ROOT/work/${label}-compatibility-smoke"
  local raw_root="$FINAL_ROOT/work/${label}-comparison-800-raw"

  if [[ ! -f "$smoke_root/PAPER_RUN_COMPLETE" ]]; then
    set +e
    GPUS="$SMOKE_GPU" RESUME=1 bash "$launcher" "$SMOKE_MANIFEST" "$smoke_root" \
      >"$FINAL_ROOT/logs/${label}-prelaunch-smoke.log" 2>&1
    local smoke_status=$?
    set -e
    printf '%s\n' "$smoke_status" \
      >"$FINAL_ROOT/logs/${label}-prelaunch-smoke.exit_status"
    set +e
    "$PYTHON" scripts/aggregate_cosmos_benchmark.py "$smoke_root" \
      --require-render-qa \
      >"$FINAL_ROOT/logs/${label}-prelaunch-smoke-aggregate.log" 2>&1
    set -e
    "$PYTHON" - "$smoke_root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "benchmark_summary.json").read_text(encoding="utf-8"))
if not summary["coverage"].get("execution_complete"):
    raise SystemExit("prelaunch smoke is not execution-complete: %s" % root)
PY
    touch "$smoke_root/PAPER_RUN_COMPLETE"
  fi

  mkdir -p "$raw_root"
  echo "$$" >"$raw_root/PRELAUNCH_PID"
  if [[ ! -f "$raw_root/RAW_ROLLOUT_FINISHED" ]]; then
    set +e
    GPUS="$GPUS" RESUME=1 bash "$launcher" "$MANIFEST" "$raw_root" \
      >"$FINAL_ROOT/logs/${label}-prelaunch-800.log" 2>&1
    local status=$?
    set -e
    printf '%s\n' "$status" \
      >"$FINAL_ROOT/logs/${label}-prelaunch-800.exit_status"
    touch "$raw_root/RAW_ROLLOUT_FINISHED"
  fi
}

run_model fastwam "$PROJECT_DIR/scripts/run_max_pro_fastwam.sh"
run_model lingbot "$PROJECT_DIR/scripts/run_max_pro_lingbot.sh"
touch "$FINAL_ROOT/AUXILIARY_PRELAUNCH_DONE"
