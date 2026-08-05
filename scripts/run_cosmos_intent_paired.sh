#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/vepfs/zijian/LIBERO-MAX}"
ASSET_DIR="${ASSET_DIR:-/vepfs/zijian/alter-wam-deps/cosmos-assets/Cosmos-Policy-LIBERO-Predict2-2B}"
export T5_EMBEDDINGS_PATH="${T5_EMBEDDINGS_PATH:-$PROJECT_DIR/artifacts/intent/libero_max_intent_t5_embeddings.pkl}"

exec bash "$PROJECT_DIR/scripts/run_cosmos_paired_smoke.sh"
