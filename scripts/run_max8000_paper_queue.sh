#!/usr/bin/env bash
set -euo pipefail

# Durable, evidence-gated queue for the complete LIBERO-MAX paper evaluation.
# Expected model failures (including trigger-unreached episodes) may make a
# rollout launcher exit nonzero. Only the full-denominator aggregator decides
# whether a run is complete; infrastructure gaps are repaired separately.

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
DEPS_DIR="${DEPS_DIR:-/vepfs/zijian/alter-wam-deps}"
PYTHON="${PYTHON:-$DEPS_DIR/cosmos-policy/.venv/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
FINAL_ROOT="${FINAL_ROOT:-$PROJECT_DIR/artifacts/max8000/paper_final}"
COSMOS_PID_FILE="${COSMOS_PID_FILE:-$PROJECT_DIR/artifacts/max8000/cosmos_pro_hard_2400.launcher.pid}"

BASE_MANIFEST="$PROJECT_DIR/benchmark/max5600/libero_max_5600.json"
PRO_MANIFEST="$PROJECT_DIR/benchmark/max8000_candidate/libero_max_pro_hard_2400.json"
COMBINED_MANIFEST="$PROJECT_DIR/benchmark/max8000_candidate/libero_max_8000.json"
COMPARISON_MANIFEST="$PROJECT_DIR/benchmark/max8000_candidate/libero_max_pro_model_comparison_800.json"
BASE_PREFLIGHT="$PROJECT_DIR/benchmark/max5600/physical_preflight.json"
PRO_PREFLIGHT="$PROJECT_DIR/benchmark/max8000_candidate/pro_physical_preflight.json"
BASE_T5="$DEPS_DIR/cosmos-assets/Cosmos-Policy-LIBERO-Predict2-2B/libero_plus_t5_embeddings.pkl"
PRO_T5="$DEPS_DIR/cosmos-assets/Cosmos-Policy-LIBERO-Predict2-2B/libero_max_pro_t5_embeddings.pkl"

mkdir -p "$FINAL_ROOT/logs" "$FINAL_ROOT/work"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$FINAL_ROOT/queue.log"
}

run_rollouts() {
  local label="$1"
  shift
  log "START rollout $label"
  set +e
  "$@" >"$FINAL_ROOT/logs/$label.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" >"$FINAL_ROOT/logs/$label.exit_status"
  log "END rollout $label launcher_status=$status"
}

assert_complete() {
  "$PYTHON" - "$1" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "benchmark_summary.json").read_text(encoding="utf-8"))
coverage = summary["coverage"]
if not coverage.get("execution_complete"):
    raise SystemExit(
        "run is not execution-complete: %s missing=%d invalid=%d blocking=%d"
        % (
            root,
            len(coverage.get("missing", [])),
            len(coverage.get("invalid", {})),
            len(coverage.get("blocking_terminal_invalid", {})),
        )
    )
PY
}

repair_count() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["cases"]))
PY
}

launch_cosmos_base() {
  GPUS="$GPUS" T5_EMBEDDINGS="$BASE_T5" \
    bash scripts/run_max_hard_cosmos.sh "$1" "$2"
}

launch_cosmos_pro() {
  GPUS="$GPUS" T5_EMBEDDINGS="$PRO_T5" \
    bash scripts/run_max_pro_cosmos.sh "$1" "$2"
}

launch_pi_base() {
  GPUS="$GPUS" QUERY_INTERVAL=5 \
    bash scripts/run_openpi_persistent_benchmark.sh "$1" "$2"
}

launch_pi_pro_q5() {
  GPUS="$GPUS" QUERY_INTERVAL=5 bash scripts/run_max_pro_openpi.sh "$1" "$2"
}

launch_pi_pro_q16() {
  GPUS="$GPUS" QUERY_INTERVAL=16 bash scripts/run_max_pro_openpi.sh "$1" "$2"
}

launch_fastwam() {
  GPUS="$GPUS" bash scripts/run_max_pro_fastwam.sh "$1" "$2"
}

launch_lingbot() {
  GPUS="$GPUS" bash scripts/run_max_pro_lingbot.sh "$1" "$2"
}

finalize_with_repairs() {
  local label="$1"
  local manifest="$2"
  local raw_root="$3"
  local final_root="$4"
  local launcher="$5"
  local current_root="$raw_root"

  if [[ -f "$final_root/PAPER_RUN_COMPLETE" ]]; then
    log "SKIP finalized $label"
    return
  fi

  for attempt in 1 2 3; do
    local audit_root="$FINAL_ROOT/work/${label}-audit-$attempt"
    mkdir -p "$audit_root"
    set +e
    "$PYTHON" scripts/aggregate_cosmos_benchmark.py "$current_root" \
      --manifest "$manifest" --output-dir "$audit_root" \
      >"$FINAL_ROOT/logs/${label}-aggregate-$attempt.log" 2>&1
    set -e
    local repair_manifest="$FINAL_ROOT/work/${label}-repair-$attempt.json"
    "$PYTHON" scripts/build_infrastructure_repair_manifest.py \
      "$manifest" "$audit_root/benchmark_summary.json" "$repair_manifest" \
      >"$FINAL_ROOT/logs/${label}-repair-select-$attempt.log"
    local count
    count="$(repair_count "$repair_manifest")"
    if [[ "$count" == "0" ]]; then
      break
    fi
    log "$label infrastructure repair round=$attempt cases=$count"
    local repair_root="$FINAL_ROOT/work/${label}-repair-run-$attempt"
    run_rollouts "${label}-repair-$attempt" "$launcher" "$repair_manifest" "$repair_root"
    local composed="$FINAL_ROOT/work/${label}-composed-$attempt"
    "$PYTHON" scripts/compose_paired_run.py "$manifest" \
      "$repair_root" "$current_root" --output-root "$composed" \
      >"$FINAL_ROOT/logs/${label}-compose-$attempt.log"
    current_root="$composed"
  done

  "$PYTHON" scripts/compose_paired_run.py "$manifest" "$current_root" \
    --output-root "$final_root" >"$FINAL_ROOT/logs/${label}-compose-final.log"
  "$PYTHON" scripts/aggregate_cosmos_benchmark.py "$final_root" \
    >"$FINAL_ROOT/logs/${label}-aggregate-final.log"
  assert_complete "$final_root"
  touch "$final_root/PAPER_RUN_COMPLETE"
  log "FINALIZED $label"
}

make_smoke_manifest() {
  "$PYTHON" - "$COMPARISON_MANIFEST" "$1" <<'PY'
import json, pathlib, sys
source, output = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(source.read_text(encoding="utf-8"))
wanted = "pro-task-10-t00-i04-target_relocation-d0-p195"
manifest["benchmark_id"] += "-runtime-smoke"
manifest["cases"] = [case for case in manifest["cases"] if case["case_id"] == wanted]
if len(manifest["cases"]) != 1:
    raise SystemExit("frozen runtime-smoke case is missing")
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_smoke() {
  local label="$1"
  local launcher="$2"
  local root="$FINAL_ROOT/work/${label}-smoke"
  if [[ -f "$root/PAPER_RUN_COMPLETE" ]]; then
    return
  fi
  run_rollouts "${label}-smoke" "$launcher" "$FINAL_ROOT/work/runtime-smoke.json" "$root"
  set +e
  "$PYTHON" scripts/aggregate_cosmos_benchmark.py "$root" \
    >"$FINAL_ROOT/logs/${label}-smoke-aggregate.log" 2>&1
  set -e
  assert_complete "$root"
  touch "$root/PAPER_RUN_COMPLETE"
  log "SMOKE passed $label"
}

wait_for_file() {
  local path="$1"
  local waited=0
  while [[ ! -x "$path" ]]; do
    if (( waited >= 21600 )); then
      log "ERROR timed out waiting for $path"
      return 1
    fi
    sleep 60
    waited=$((waited + 60))
  done
}

if [[ -f "$FINAL_ROOT/PAPER_QUEUE_DONE" ]]; then
  log "paper queue already complete"
  exit 0
fi

if [[ -f "$COSMOS_PID_FILE" ]]; then
  cosmos_pid="$(cat "$COSMOS_PID_FILE")"
  while kill -0 "$cosmos_pid" 2>/dev/null; do
    completed="$(find artifacts/max8000/cosmos_pro_hard_2400/cases -name DONE 2>/dev/null | wc -l)"
    failed="$(find artifacts/max8000/cosmos_pro_hard_2400/cases -name FAILED 2>/dev/null | wc -l)"
    log "WAIT Cosmos PRO-Hard done=$completed failed=$failed planned=2400"
    sleep 60
  done
fi

wait_for_file "$DEPS_DIR/openpi/.venv/bin/python"
wait_for_file "$DEPS_DIR/FastWAM/.venv/bin/python"
wait_for_file "$DEPS_DIR/lingbot-va/.venv/bin/python"
make_smoke_manifest "$FINAL_ROOT/work/runtime-smoke.json"

# Finish both Cosmos tracks first. Trigger-unreached terminal traces remain in
# the denominator and are not selected for repair.
finalize_with_repairs \
  cosmos-base "$BASE_MANIFEST" artifacts/max5600/cosmos_policy \
  "$FINAL_ROOT/runs/cosmos_base_5600" launch_cosmos_base
finalize_with_repairs \
  cosmos-pro "$PRO_MANIFEST" artifacts/max8000/cosmos_pro_hard_2400 \
  "$FINAL_ROOT/runs/cosmos_pro_2400" launch_cosmos_pro

mkdir -p "$FINAL_ROOT/runs/cosmos_max8000"
set +e
"$PYTHON" scripts/aggregate_cosmos_benchmark.py \
  "$FINAL_ROOT/runs/cosmos_base_5600" \
  --case-root "$FINAL_ROOT/runs/cosmos_pro_2400" \
  --manifest "$COMBINED_MANIFEST" \
  --output-dir "$FINAL_ROOT/runs/cosmos_max8000" \
  >"$FINAL_ROOT/logs/cosmos-max8000-aggregate.log" 2>&1
set -e
assert_complete "$FINAL_ROOT/runs/cosmos_max8000"

mkdir -p "$FINAL_ROOT/runs/model_comparison/cosmos"
"$PYTHON" scripts/aggregate_cosmos_benchmark.py \
  "$FINAL_ROOT/runs/cosmos_pro_2400" \
  --manifest "$COMPARISON_MANIFEST" \
  --output-dir "$FINAL_ROOT/runs/model_comparison/cosmos" \
  >"$FINAL_ROOT/logs/cosmos-comparison-aggregate.log"
assert_complete "$FINAL_ROOT/runs/model_comparison/cosmos"

# Render all eight interventions and one audited, real Cosmos action replay
# while the queue exclusively owns GPUs 0-3.
export LIBERO_IMPL_DIR="$DEPS_DIR/LIBERO-PRO"
export LIBERO_OVERLAY="$DEPS_DIR/libero-pro-python-overlay"
export LIBERO_CONFIG_PATH="$DEPS_DIR/libero-pro-config"
export PYTHONPATH="$DEPS_DIR/robosuite-1.4.0:$LIBERO_OVERLAY:$DEPS_DIR/cosmos-policy/.venv/lib/python3.10/site-packages:$DEPS_DIR/.venv-libero/lib/python3.10/site-packages:$LIBERO_IMPL_DIR:$DEPS_DIR/cosmos-policy:$PROJECT_DIR/src:$PROJECT_DIR"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  "$PYTHON" scripts/render_benchmark_media.py "$PRO_MANIFEST" \
    --preflight "$PRO_PREFLIGHT" --output-dir "$FINAL_ROOT/media" \
    >"$FINAL_ROOT/logs/render-media.log"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  "$PYTHON" scripts/render_rollout_replay.py "$COMPARISON_MANIFEST" \
    "$FINAL_ROOT/runs/cosmos_pro_2400" \
    pro-task-10-t00-i04-target_relocation-d0-p195 \
    "$FINAL_ROOT/media/cosmos-target-relocation-rollout.gif" \
    >"$FINAL_ROOT/logs/render-rollout-replay.log"

# Main pi0.5 runs: native q5 on all 8,000 cases, plus a separate q16 rerun on
# the frozen 800-case model-comparison subset.
run_smoke pi05-q5 launch_pi_pro_q5
finalize_with_repairs \
  pi05-base-q5 artifacts/max5600/pi05_libero/manifest.json \
  artifacts/max5600/pi05_libero "$FINAL_ROOT/runs/pi05_base_5600_q5" \
  launch_pi_base
run_rollouts pi05-pro-2400-q5 launch_pi_pro_q5 "$PRO_MANIFEST" \
  "$FINAL_ROOT/work/pi05-pro-2400-q5-raw"
finalize_with_repairs \
  pi05-pro-q5 "$FINAL_ROOT/work/pi05-pro-2400-q5-raw/manifest.json" \
  "$FINAL_ROOT/work/pi05-pro-2400-q5-raw" \
  "$FINAL_ROOT/runs/pi05_pro_2400_q5" launch_pi_pro_q5

"$PYTHON" - "$COMBINED_MANIFEST" "$FINAL_ROOT/work/libero_max_8000_q5.json" <<'PY'
import json, pathlib, sys
source, output = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(source.read_text(encoding="utf-8"))
manifest["protocol"]["query_interval"] = 5
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
mkdir -p "$FINAL_ROOT/runs/pi05_max8000_q5"
set +e
"$PYTHON" scripts/aggregate_cosmos_benchmark.py \
  "$FINAL_ROOT/runs/pi05_base_5600_q5" \
  --case-root "$FINAL_ROOT/runs/pi05_pro_2400_q5" \
  --manifest "$FINAL_ROOT/work/libero_max_8000_q5.json" \
  --output-dir "$FINAL_ROOT/runs/pi05_max8000_q5" \
  >"$FINAL_ROOT/logs/pi05-max8000-aggregate.log" 2>&1
set -e
assert_complete "$FINAL_ROOT/runs/pi05_max8000_q5"

run_rollouts pi05-comparison-800-q16 launch_pi_pro_q16 "$COMPARISON_MANIFEST" \
  "$FINAL_ROOT/work/pi05-comparison-800-q16-raw"
finalize_with_repairs \
  pi05-comparison-q16 "$FINAL_ROOT/work/pi05-comparison-800-q16-raw/manifest.json" \
  "$FINAL_ROOT/work/pi05-comparison-800-q16-raw" \
  "$FINAL_ROOT/runs/model_comparison/pi05" launch_pi_pro_q16

# q16 cross-model comparison on exactly the same 800 frozen cases.
run_smoke fastwam launch_fastwam
run_rollouts fastwam-comparison-800 launch_fastwam "$COMPARISON_MANIFEST" \
  "$FINAL_ROOT/work/fastwam-comparison-800-raw"
finalize_with_repairs \
  fastwam-comparison "$COMPARISON_MANIFEST" \
  "$FINAL_ROOT/work/fastwam-comparison-800-raw" \
  "$FINAL_ROOT/runs/model_comparison/fastwam" launch_fastwam

run_smoke lingbot launch_lingbot
run_rollouts lingbot-comparison-800 launch_lingbot "$COMPARISON_MANIFEST" \
  "$FINAL_ROOT/work/lingbot-comparison-800-raw"
finalize_with_repairs \
  lingbot-comparison "$COMPARISON_MANIFEST" \
  "$FINAL_ROOT/work/lingbot-comparison-800-raw" \
  "$FINAL_ROOT/runs/model_comparison/lingbot" launch_lingbot

mkdir -p "$FINAL_ROOT/tables/main" "$FINAL_ROOT/tables/model_comparison" \
  "$FINAL_ROOT/tables/intent" "$FINAL_ROOT/human_review"
"$PYTHON" scripts/build_paper_tables.py \
  --run "Cosmos-Policy-q16=$FINAL_ROOT/runs/cosmos_max8000" \
  --run "pi0.5-LIBERO-q5=$FINAL_ROOT/runs/pi05_max8000_q5" \
  --output-dir "$FINAL_ROOT/tables/main" \
  >"$FINAL_ROOT/logs/build-main-tables.log"
"$PYTHON" scripts/build_paper_tables.py \
  --run "Cosmos-Policy=$FINAL_ROOT/runs/model_comparison/cosmos" \
  --run "pi0.5-LIBERO=$FINAL_ROOT/runs/model_comparison/pi05" \
  --run "FastWAM-LIBERO=$FINAL_ROOT/runs/model_comparison/fastwam" \
  --run "LingBot-VA=$FINAL_ROOT/runs/model_comparison/lingbot" \
  --output-dir "$FINAL_ROOT/tables/model_comparison" \
  >"$FINAL_ROOT/logs/build-model-comparison-tables.log"
"$PYTHON" scripts/build_paper_tables.py \
  --run "Cosmos-Policy-Intent=results/v1/runs/cosmos2_intent" \
  --run "pi0.5-LIBERO-Intent=results/v1/runs/pi05_intent" \
  --output-dir "$FINAL_ROOT/tables/intent" \
  >"$FINAL_ROOT/logs/build-intent-tables.log"

"$PYTHON" scripts/build_human_review_queue.py "$COMBINED_MANIFEST" \
  --preflight "$BASE_PREFLIGHT" --preflight "$PRO_PREFLIGHT" \
  --run "Cosmos-Policy=$FINAL_ROOT/runs/cosmos_max8000" \
  --run "pi0.5-LIBERO=$FINAL_ROOT/runs/pi05_max8000_q5" \
  --run "FastWAM-LIBERO=$FINAL_ROOT/runs/model_comparison/fastwam" \
  --run "LingBot-VA=$FINAL_ROOT/runs/model_comparison/lingbot" \
  --limit 300 --minimum-score 5 \
  --output-dir "$FINAL_ROOT/human_review" \
  >"$FINAL_ROOT/logs/build-human-review.log"

"$PYTHON" - "$FINAL_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
runs = {}
for summary_path in sorted((root / "runs").glob("**/benchmark_summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    coverage = summary["coverage"]
    runs[str(summary_path.parent.relative_to(root))] = {
        "planned": coverage["planned"],
        "triggered": coverage["completed"],
        "trigger_unreached": coverage.get("trigger_unreached", 0),
        "execution_complete": coverage.get("execution_complete", False),
        "control_accuracy": summary["end_to_end_metrics"]["control"]["accuracy_on_planned"],
        "intervention_accuracy": summary["end_to_end_metrics"]["intervention"]["accuracy_on_planned"],
        "paired_delta": summary["end_to_end_metrics"]["paired_robustness_delta_on_planned"],
    }
payload = {"paper_experiments_complete": all(row["execution_complete"] for row in runs.values()), "runs": runs}
(root / "experiment_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not payload["paper_experiments_complete"]:
    raise SystemExit("at least one paper run is incomplete")
PY

touch "$FINAL_ROOT/PAPER_QUEUE_DONE"
log "PAPER QUEUE COMPLETE"
