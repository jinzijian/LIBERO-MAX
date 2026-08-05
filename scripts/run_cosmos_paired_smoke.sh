#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
COSMOS_POLICY_DIR="${COSMOS_POLICY_DIR:-/vepfs/zijian/alter-wam-deps/cosmos-policy}"
LIBERO_DIR="${LIBERO_DIR:-/vepfs/zijian/alter-wam-deps/LIBERO}"
ROBOSUITE_DIR="${ROBOSUITE_DIR:-/vepfs/zijian/alter-wam-deps/robosuite-1.4.0}"
LEGACY_SITE_PACKAGES="${LEGACY_SITE_PACKAGES:-/vepfs/zijian/alter-wam-deps/.venv-libero/lib/python3.10/site-packages}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-/vepfs/zijian/alter-wam-deps/libero-config}"
ASSET_DIR="${ASSET_DIR:-/vepfs/zijian/alter-wam-deps/cosmos-assets/Cosmos-Policy-LIBERO-Predict2-2B}"
T5_EMBEDDINGS_PATH="${T5_EMBEDDINGS_PATH:-$ASSET_DIR/libero_t5_embeddings.pkl}"
HF_HOME_DIR="${HF_HOME_DIR:-/vepfs/zijian/alter-wam-deps/hf-cache-cosmos}"
GPU_ID="${GPU_ID:?set GPU_ID to one free physical GPU}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/artifacts/cosmos_paired_smoke}"
SCENARIO_FILE="${SCENARIO_FILE:-$PROJECT_DIR/examples/scenarios/cosmos_camera_shift_chunk16.json}"
SUITE="${SUITE:-libero_object}"
TASK_INDEX="${TASK_INDEX:-0}"
SEED="${SEED:-195}"
INIT_STATE_INDEX="${INIT_STATE_INDEX:-0}"

cosmos_site_packages="$COSMOS_POLICY_DIR/.venv/lib/python3.10/site-packages"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
SCENARIO_FILE="$(cd "$(dirname "$SCENARIO_FILE")" && pwd)/$(basename "$SCENARIO_FILE")"

for arm in control intervention; do
  arm_dir="$OUTPUT_ROOT/$arm"
  control_trace_path=""
  if [[ "$arm" == intervention ]]; then
    control_trace_path="$OUTPUT_ROOT/control/trace.jsonl"
  fi
  mkdir -p "$arm_dir/eval"
  : > "$arm_dir/trace.jsonl"
  (
    cd "$arm_dir"
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    HF_HOME="$HF_HOME_DIR" \
    LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR" \
    LIBERO_MAX_ARM="$arm" \
    LIBERO_MAX_SUITE="$SUITE" \
    LIBERO_MAX_TASK_INDEX="$TASK_INDEX" \
    LIBERO_MAX_INIT_STATE_INDEX="$INIT_STATE_INDEX" \
    LIBERO_MAX_SCENARIO_FILE="$SCENARIO_FILE" \
    LIBERO_MAX_TRACE_PATH="$arm_dir/trace.jsonl" \
    LIBERO_MAX_CONTROL_TRACE_PATH="$control_trace_path" \
    PYTHONPATH="$PROJECT_DIR/src:$ROBOSUITE_DIR:$cosmos_site_packages:$LEGACY_SITE_PACKAGES:$LIBERO_DIR:$COSMOS_POLICY_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$COSMOS_POLICY_DIR/.venv/bin/python" \
      "$PROJECT_DIR/scripts/run_cosmos_libero_max.py" \
      --config cosmos_predict2_2b_480p_libero__inference_only \
      --ckpt_path "$ASSET_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt" \
      --config_file cosmos_policy/config/config.py \
      --use_wrist_image True \
      --use_proprio True \
      --normalize_proprio True \
      --unnormalize_actions True \
      --dataset_stats_path "$ASSET_DIR/libero_dataset_statistics.json" \
      --t5_text_embeddings_path "$T5_EMBEDDINGS_PATH" \
      --trained_with_image_aug True \
      --chunk_size 16 \
      --num_open_loop_steps 16 \
      --task_suite_name "$SUITE" \
      --num_trials_per_task 1 \
      --local_log_dir "$arm_dir/eval" \
      --randomize_seed False \
      --data_collection False \
      --available_gpus 0 \
      --seed "$SEED" \
      --use_variance_scale False \
      --deterministic True \
      --run_id_note "libero-max-${arm}-seed${SEED}" \
      --ar_future_prediction False \
      --ar_value_prediction False \
      --use_jpeg_compression True \
      --flip_images True \
      --num_denoising_steps_action 5 \
      --num_denoising_steps_future_state 1 \
      --num_denoising_steps_value 1
  ) > "$arm_dir/console.log" 2>&1
done

PYTHONPATH="$PROJECT_DIR/src" \
  "$COSMOS_POLICY_DIR/.venv/bin/python" \
  "$PROJECT_DIR/scripts/summarize_cosmos_paired_smoke.py" "$OUTPUT_ROOT"
