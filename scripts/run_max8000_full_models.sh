#!/usr/bin/env bash
set -euo pipefail

# Complete the paper matrix on the full MAX-8000 denominator. The historical
# 800-case comparison remains a reusable frozen subset; it is composed with
# an outcome-independent 1,600-case PRO complement rather than rerun.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEPS_DIR="${DEPS_DIR:-$PROJECT_DIR/.deps}"
PYTHON="${PYTHON:-$DEPS_DIR/cosmos-policy/.venv/bin/python}"
FINAL_ROOT="${FINAL_ROOT:-$PROJECT_DIR/artifacts/max8000/paper_final}"
GPUS="${GPUS:-0,1,2,3}"
MODELS="${MODELS:-fastwam}"
FINALIZE_PUBLICATION="${FINALIZE_PUBLICATION:-1}"
WAIT_FOR_PID_FILE="${WAIT_FOR_PID_FILE:-}"

BASE_MANIFEST="$PROJECT_DIR/benchmark/max5600/libero_max_5600.json"
PRO_MANIFEST="$PROJECT_DIR/benchmark/max8000/libero_max_pro_hard_2400.json"
COMBINED_MANIFEST="$PROJECT_DIR/benchmark/max8000/libero_max_8000.json"
COMPARISON_MANIFEST="$PROJECT_DIR/benchmark/max8000/libero_max_pro_model_comparison_800.json"
PRO_REMAINDER="$FINAL_ROOT/work/libero_max_pro_remaining_1600.json"
BASE_PREFLIGHT="$PROJECT_DIR/benchmark/max5600/physical_preflight.json"
PRO_PREFLIGHT="$PROJECT_DIR/benchmark/max8000_candidate/pro_physical_preflight.json"

mkdir -p "$FINAL_ROOT/logs" "$FINAL_ROOT/work" "$FINAL_ROOT/full_models"
touch "$FINAL_ROOT/FULL_MODELS_QUEUE_PENDING"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "$FINAL_ROOT/full-models.log"
}

wait_for_pid_file() {
  local path="$1"
  [[ -n "$path" ]] || return 0
  while [[ ! -f "$path" ]]; do
    log "WAIT pid file $path"
    sleep 60
  done
  local pid
  pid="$(cat "$path")"
  while kill -0 "$pid" 2>/dev/null; do
    log "WAIT predecessor pid=$pid"
    sleep 60
  done
  log "PREDECESSOR finished pid=$pid"
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

ensure_rollouts() {
  local label="$1"
  local launcher="$2"
  local manifest="$3"
  local root="$4"
  if [[ -f "$root/RAW_ROLLOUT_FINISHED" ]]; then
    log "SKIP completed raw rollout $label"
    return
  fi
  RESUME=1 run_rollouts "$label" "$launcher" "$manifest" "$root"
  touch "$root/RAW_ROLLOUT_FINISHED"
}

wait_for_raw_rollout() {
  local label="$1"
  local root="$2"
  while [[ ! -f "$root/RAW_ROLLOUT_FINISHED" ]]; do
    local completed failed
    completed="$(find "$root/cases" -mindepth 2 -maxdepth 2 -name DONE 2>/dev/null | wc -l)"
    failed="$(find "$root/cases" -mindepth 2 -maxdepth 2 -name FAILED 2>/dev/null | wc -l)"
    log "WAIT $label completed=$completed terminal_failed=$failed planned=800"
    sleep 60
  done
}

make_base_smoke_manifest() {
  local output="$FINAL_ROOT/work/base-runtime-smoke.json"
  if [[ -f "$output" ]]; then
    return
  fi
  "$PYTHON" - "$BASE_MANIFEST" "$output" <<'PY'
import json, pathlib, sys
source, output = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(source.read_text(encoding="utf-8"))
manifest["benchmark_id"] += "-runtime-smoke"
manifest["cases"] = manifest["cases"][:1]
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
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

launch_fastwam_base() {
  LIBERO_IMPL_DIR="$DEPS_DIR/LIBERO-plus" \
    LIBERO_OVERLAY="$DEPS_DIR/libero-plus-python-overlay" \
    LIBERO_CONFIG_PATH="$DEPS_DIR/libero-plus-config" \
    GPUS="$GPUS" bash scripts/run_max_pro_fastwam.sh "$1" "$2"
}

launch_fastwam_pro() {
  GPUS="$GPUS" bash scripts/run_max_pro_fastwam.sh "$1" "$2"
}

launch_lingbot_base() {
  LIBERO_IMPL_DIR="$DEPS_DIR/LIBERO-plus" \
    LIBERO_OVERLAY="$DEPS_DIR/libero-plus-python-overlay" \
    LIBERO_CONFIG_PATH="$DEPS_DIR/libero-plus-config" \
    GPUS="$GPUS" bash scripts/run_max_pro_lingbot.sh "$1" "$2"
}

launch_lingbot_pro() {
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
    "$PYTHON" scripts/aggregate_cosmos_benchmark.py "$final_root" \
      --require-render-qa >"$FINAL_ROOT/logs/${label}-revalidate.log" 2>&1
    assert_complete "$final_root"
    log "SKIP finalized $label"
    return
  fi
  for attempt in 1 2 3; do
    local audit_root="$FINAL_ROOT/work/${label}-audit-$attempt"
    mkdir -p "$audit_root"
    set +e
    "$PYTHON" scripts/aggregate_cosmos_benchmark.py "$current_root" \
      --require-render-qa --manifest "$manifest" --output-dir "$audit_root" \
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
    RESUME=1 run_rollouts \
      "${label}-repair-$attempt" "$launcher" "$repair_manifest" "$repair_root"
    local composed="$FINAL_ROOT/work/${label}-composed-$attempt"
    "$PYTHON" scripts/compose_paired_run.py "$manifest" \
      "$repair_root" "$current_root" --output-root "$composed" \
      >"$FINAL_ROOT/logs/${label}-compose-$attempt.log"
    current_root="$composed"
  done
  "$PYTHON" scripts/compose_paired_run.py "$manifest" "$current_root" \
    --output-root "$final_root" >"$FINAL_ROOT/logs/${label}-compose-final.log"
  "$PYTHON" scripts/aggregate_cosmos_benchmark.py "$final_root" \
    --require-render-qa >"$FINAL_ROOT/logs/${label}-aggregate-final.log"
  assert_complete "$final_root"
  touch "$final_root/PAPER_RUN_COMPLETE"
  log "FINALIZED $label"
}

aggregate_max8000() {
  local label="$1"
  local base_root="$2"
  local pro_root="$3"
  local output_root="$4"
  mkdir -p "$output_root"
  "$PYTHON" scripts/aggregate_cosmos_benchmark.py "$base_root" \
    --case-root "$pro_root" --manifest "$COMBINED_MANIFEST" \
    --output-dir "$output_root" --require-render-qa \
    >"$FINAL_ROOT/logs/${label}-max8000-aggregate.log" 2>&1
  assert_complete "$output_root"
}

run_base_smoke() {
  local model="$1"
  local launcher="$2"
  local root="$FINAL_ROOT/work/${model}-base-runtime-smoke"
  if [[ -f "$root/PAPER_RUN_COMPLETE" ]]; then
    return
  fi
  make_base_smoke_manifest
  local GPUS="${GPUS%%,*}"
  run_rollouts "${model}-base-runtime-smoke" "$launcher" \
    "$FINAL_ROOT/work/base-runtime-smoke.json" "$root"
  "$PYTHON" scripts/aggregate_cosmos_benchmark.py "$root" \
    --manifest "$FINAL_ROOT/work/base-runtime-smoke.json" \
    --require-render-qa >"$FINAL_ROOT/logs/${model}-base-smoke-aggregate.log"
  assert_complete "$root"
  touch "$root/PAPER_RUN_COMPLETE"
  log "BASE runtime smoke passed model=$model"
}

run_full_model() {
  local model="$1"
  local base_launcher pro_launcher comparison_root
  case "$model" in
    fastwam)
      base_launcher=launch_fastwam_base
      pro_launcher=launch_fastwam_pro
      comparison_root="$FINAL_ROOT/work/fastwam-comparison-800-raw"
      ;;
    lingbot)
      base_launcher=launch_lingbot_base
      pro_launcher=launch_lingbot_pro
      comparison_root="$FINAL_ROOT/work/lingbot-comparison-800-raw"
      ;;
    *)
      log "ERROR unsupported full model: $model"
      return 2
      ;;
  esac
  if [[ -f "$FINAL_ROOT/full_models/$model/DONE" ]]; then
    log "SKIP full MAX-8000 model=$model"
    return
  fi
  wait_for_raw_rollout "$model comparison subset" "$comparison_root"
  run_base_smoke "$model" "$base_launcher"
  local base_raw="$FINAL_ROOT/work/${model}-base-5600-q16-raw"
  local base_final="$FINAL_ROOT/runs/${model}_base_5600_q16"
  ensure_rollouts "${model}-base-5600-q16" "$base_launcher" \
    "$BASE_MANIFEST" "$base_raw"
  finalize_with_repairs "${model}-base-q16" "$BASE_MANIFEST" \
    "$base_raw" "$base_final" "$base_launcher"

  local pro_remaining_raw="$FINAL_ROOT/work/${model}-pro-remaining-1600-q16-raw"
  ensure_rollouts "${model}-pro-remaining-1600-q16" "$pro_launcher" \
    "$PRO_REMAINDER" "$pro_remaining_raw"
  local pro_raw="$FINAL_ROOT/work/${model}-pro-2400-q16-composed"
  "$PYTHON" scripts/compose_paired_run.py "$PRO_MANIFEST" \
    "$comparison_root" "$pro_remaining_raw" --output-root "$pro_raw" \
    >"$FINAL_ROOT/logs/${model}-pro-compose-800-plus-1600.log"
  local pro_final="$FINAL_ROOT/runs/${model}_pro_2400_q16"
  finalize_with_repairs "${model}-pro-q16" "$PRO_MANIFEST" \
    "$pro_raw" "$pro_final" "$pro_launcher"
  aggregate_max8000 "$model" "$base_final" "$pro_final" \
    "$FINAL_ROOT/runs/${model}_max8000_q16"
  mkdir -p "$FINAL_ROOT/full_models/$model"
  touch "$FINAL_ROOT/full_models/$model/DONE"
  log "FULL MAX-8000 COMPLETE model=$model"
}

wait_for_full_model() {
  local model="$1"
  while [[ ! -f "$FINAL_ROOT/full_models/$model/DONE" ]]; do
    log "WAIT full MAX-8000 model=$model"
    sleep 60
  done
}

rebuild_publication() {
  assert_complete "$FINAL_ROOT/runs/pi05_max8000_q5"
  mkdir -p "$FINAL_ROOT/full_models/pi05"
  touch "$FINAL_ROOT/full_models/pi05/DONE"
  wait_for_full_model fastwam
  if [[ -f "$FINAL_ROOT/PAPER_QUEUE_DONE" ]]; then
    mv "$FINAL_ROOT/PAPER_QUEUE_DONE" "$FINAL_ROOT/PAPER_QUEUE_SUBSET_DONE"
  fi
  mkdir -p "$FINAL_ROOT/tables/main" "$FINAL_ROOT/tables/tracks" \
    "$FINAL_ROOT/tables/model_comparison" "$FINAL_ROOT/tables/intent" \
    "$FINAL_ROOT/tables/ablation" "$FINAL_ROOT/figures/main" \
    "$FINAL_ROOT/figures/model_comparison" "$FINAL_ROOT/human_review"
  "$PYTHON" scripts/build_paper_tables.py \
    --run "Cosmos-Policy-q16=$FINAL_ROOT/runs/cosmos_max8000" \
    --run "pi0.5-LIBERO-q5=$FINAL_ROOT/runs/pi05_max8000_q5" \
    --run "FastWAM-LIBERO-q16=$FINAL_ROOT/runs/fastwam_max8000_q16" \
    --output-dir "$FINAL_ROOT/tables/main" \
    >"$FINAL_ROOT/logs/build-main-tables-full-models.log"
  "$PYTHON" scripts/build_paper_tables.py \
    --run "Cosmos-Base-q16=$FINAL_ROOT/runs/cosmos_base_5600" \
    --run "Cosmos-PRO-q16=$FINAL_ROOT/runs/cosmos_pro_2400" \
    --run "pi0.5-Base-q5=$FINAL_ROOT/runs/pi05_base_5600_q5" \
    --run "pi0.5-PRO-q5=$FINAL_ROOT/runs/pi05_pro_2400_q5" \
    --run "FastWAM-Base-q16=$FINAL_ROOT/runs/fastwam_base_5600_q16" \
    --run "FastWAM-PRO-q16=$FINAL_ROOT/runs/fastwam_pro_2400_q16" \
    --output-dir "$FINAL_ROOT/tables/tracks" \
    >"$FINAL_ROOT/logs/build-track-tables-full-models.log"
  "$PYTHON" scripts/build_paper_tables.py \
    --run "Cosmos-Policy=$FINAL_ROOT/runs/cosmos_max8000" \
    --run "pi0.5-LIBERO-q5=$FINAL_ROOT/runs/pi05_max8000_q5" \
    --run "FastWAM-LIBERO=$FINAL_ROOT/runs/fastwam_max8000_q16" \
    --output-dir "$FINAL_ROOT/tables/model_comparison" \
    >"$FINAL_ROOT/logs/build-model-comparison-full8000.log"
  "$PYTHON" scripts/build_paper_tables.py \
    --run "Cosmos-Policy-Intent=results/v1/runs/cosmos2_intent" \
    --run "pi0.5-LIBERO-Intent=results/v1/runs/pi05_intent" \
    --output-dir "$FINAL_ROOT/tables/intent" \
    >"$FINAL_ROOT/logs/build-intent-tables-full-models.log"
  "$PYTHON" scripts/build_paper_tables.py \
    --run "Cosmos-q16=results/v1/runs/cosmos2_q16_subset" \
    --run "Cosmos-q5=results/v1/runs/cosmos2_q5_subset" \
    --run "Cosmos-notified-q16=results/v1/runs/cosmos2_notified_q16" \
    --output-dir "$FINAL_ROOT/tables/ablation" \
    >"$FINAL_ROOT/logs/build-ablation-tables-full-models.log"
  "$PYTHON" scripts/build_paper_figures.py \
    --run "Cosmos-Policy-q16=$FINAL_ROOT/runs/cosmos_max8000" \
    --run "pi0.5-LIBERO-q5=$FINAL_ROOT/runs/pi05_max8000_q5" \
    --run "FastWAM-LIBERO-q16=$FINAL_ROOT/runs/fastwam_max8000_q16" \
    --output-dir "$FINAL_ROOT/figures/main" \
    >"$FINAL_ROOT/logs/build-main-figures-full-models.log"
  "$PYTHON" scripts/build_paper_figures.py \
    --run "Cosmos-Policy=$FINAL_ROOT/runs/cosmos_max8000" \
    --run "pi0.5-LIBERO-q5=$FINAL_ROOT/runs/pi05_max8000_q5" \
    --run "FastWAM-LIBERO=$FINAL_ROOT/runs/fastwam_max8000_q16" \
    --output-dir "$FINAL_ROOT/figures/model_comparison" \
    >"$FINAL_ROOT/logs/build-model-comparison-figures-full8000.log"
  "$PYTHON" scripts/build_human_review_queue.py "$COMBINED_MANIFEST" \
    --preflight "$BASE_PREFLIGHT" --preflight "$PRO_PREFLIGHT" \
    --run "Cosmos-Policy=$FINAL_ROOT/runs/cosmos_max8000" \
    --run "pi0.5-LIBERO=$FINAL_ROOT/runs/pi05_max8000_q5" \
    --run "FastWAM-LIBERO=$FINAL_ROOT/runs/fastwam_max8000_q16" \
    --limit 300 --minimum-score 5 --output-dir "$FINAL_ROOT/human_review" \
    >"$FINAL_ROOT/logs/build-human-review-full-models.log"
  "$PYTHON" scripts/build_experiment_status.py "$FINAL_ROOT" \
    --expected-runs 15 \
    --external-run "frozen/intent/cosmos=results/v1/runs/cosmos2_intent/benchmark_summary.json" \
    --external-run "frozen/intent/pi05=results/v1/runs/pi05_intent/benchmark_summary.json" \
    --external-run "frozen/ablation/cosmos-q16=results/v1/runs/cosmos2_q16_subset/benchmark_summary.json" \
    --external-run "frozen/ablation/cosmos-q5=results/v1/runs/cosmos2_q5_subset/benchmark_summary.json" \
    --external-run "frozen/ablation/cosmos-notified-q16=results/v1/runs/cosmos2_notified_q16/benchmark_summary.json" \
    >"$FINAL_ROOT/logs/build-experiment-status-full-models.log"
  "$PYTHON" scripts/build_paper_analysis.py "$FINAL_ROOT" \
    "$FINAL_ROOT/paper/MAX8000_ANALYSIS.md"
  "$PYTHON" scripts/build_paper_appendix.py "$FINAL_ROOT" \
    "$FINAL_ROOT/paper/MAX8000_RESULTS.md"
  "$PYTHON" scripts/package_paper_results.py "$FINAL_ROOT" \
    "$PROJECT_DIR/results/max8000" --media-output-dir "$PROJECT_DIR/assets/media"
  "$PYTHON" scripts/update_readme_paper_results.py "$FINAL_ROOT" \
    "$PROJECT_DIR/README.md"
  touch "$FINAL_ROOT/PAPER_QUEUE_FULL_MODELS_DONE" "$FINAL_ROOT/PAPER_QUEUE_DONE"
  mv "$FINAL_ROOT/FULL_MODELS_QUEUE_PENDING" \
    "$FINAL_ROOT/FULL_MODELS_QUEUE_COMPLETE"
  log "FULL THREE-MODEL PAPER QUEUE COMPLETE"
}

wait_for_pid_file "$WAIT_FOR_PID_FILE"
if [[ "$FINALIZE_PUBLICATION" == "1" && -f "$FINAL_ROOT/PAPER_QUEUE_DONE" ]]; then
  mv "$FINAL_ROOT/PAPER_QUEUE_DONE" "$FINAL_ROOT/PAPER_QUEUE_SUBSET_DONE"
  log "ARCHIVED subset-only completion marker before full-model extension"
fi
"$PYTHON" scripts/build_manifest_complement.py \
  "$PRO_MANIFEST" "$COMPARISON_MANIFEST" "$PRO_REMAINDER" \
  >"$FINAL_ROOT/logs/build-pro-remainder.log"

IFS=',' read -r -a requested_models <<<"$MODELS"
for model in "${requested_models[@]}"; do
  run_full_model "$model"
done
if [[ "$FINALIZE_PUBLICATION" == "1" ]]; then
  rebuild_publication
fi
