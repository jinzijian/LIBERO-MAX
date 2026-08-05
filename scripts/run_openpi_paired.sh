#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
COSMOS_DEPS="${COSMOS_DEPS:-/vepfs/zijian/alter-wam-deps}"
OPENPI_DIR="${OPENPI_DIR:-$COSMOS_DEPS/openpi}"
OPENPI_CLIENT_OVERLAY="${OPENPI_CLIENT_OVERLAY:-$COSMOS_DEPS/openpi-client-overlay}"
CLIENT_PYTHON="${CLIENT_PYTHON:-$COSMOS_DEPS/cosmos-policy/.venv/bin/python}"
GPU_ID="${GPU_ID:?set GPU_ID}"
PORT_BASE="${PORT_BASE:-8100}"
PORT="$((PORT_BASE + GPU_ID))"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SCENARIO_FILE="${SCENARIO_FILE:?set SCENARIO_FILE}"
SUITE="${SUITE:?set SUITE}"
TASK_INDEX="${TASK_INDEX:?set TASK_INDEX}"
INIT_STATE_INDEX="${INIT_STATE_INDEX:?set INIT_STATE_INDEX}"
SEED="${SEED:?set SEED}"
REPLAN_STEPS="${REPLAN_STEPS:-${QUERY_INTERVAL:-5}}"
POLICY_NOTIFICATION="${POLICY_NOTIFICATION:-}"

site_packages="$COSMOS_DEPS/cosmos-policy/.venv/lib/python3.10/site-packages"
legacy_site_packages="$COSMOS_DEPS/.venv-libero/lib/python3.10/site-packages"
pythonpath="$PROJECT_DIR/src:$OPENPI_DIR/packages/openpi-client/src:$OPENPI_CLIENT_OVERLAY:$COSMOS_DEPS/robosuite-1.4.0:$site_packages:$legacy_site_packages:$COSMOS_DEPS/LIBERO:$COSMOS_DEPS/cosmos-policy"

mkdir -p "$OUTPUT_ROOT"
for arm in control intervention; do
  arm_dir="$OUTPUT_ROOT/$arm"
  control_trace_args=()
  if [[ "$arm" == intervention ]]; then
    control_trace_args=(--control-trace "$OUTPUT_ROOT/control/trace.jsonl")
  fi
  notification_args=()
  if [[ -n "$POLICY_NOTIFICATION" ]]; then
    notification_args=(--policy-notification "$POLICY_NOTIFICATION")
  fi
  mkdir -p "$arm_dir"
  : > "$arm_dir/trace.jsonl"
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  LIBERO_CONFIG_PATH="$COSMOS_DEPS/libero-config" \
  PYTHONPATH="$pythonpath" \
  "$CLIENT_PYTHON" "$PROJECT_DIR/scripts/run_openpi_libero_max.py" \
    --scenario "$SCENARIO_FILE" \
    --arm "$arm" \
    --suite "$SUITE" \
    --task-index "$TASK_INDEX" \
    --init-state-index "$INIT_STATE_INDEX" \
    --policy-seed "$SEED" \
    --port "$PORT" \
    --replan-steps "$REPLAN_STEPS" \
    --trace "$arm_dir/trace.jsonl" \
    "${control_trace_args[@]}" \
    "${notification_args[@]}" \
    > "$arm_dir/console.log" 2>&1
done

PYTHONPATH="$PROJECT_DIR/src" "$CLIENT_PYTHON" \
  "$PROJECT_DIR/scripts/summarize_cosmos_paired_smoke.py" "$OUTPUT_ROOT"
