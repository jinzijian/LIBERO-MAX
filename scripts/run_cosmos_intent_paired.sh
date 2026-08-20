#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEPS_DIR="${DEPS_DIR:-$PROJECT_DIR/.deps}"
ASSET_DIR="${ASSET_DIR:-$DEPS_DIR/cosmos-assets/Cosmos-Policy-LIBERO-Predict2-2B}"
export T5_EMBEDDINGS_PATH="${T5_EMBEDDINGS_PATH:-$PROJECT_DIR/artifacts/intent/libero_max_intent_t5_embeddings.pkl}"

exec bash "$PROJECT_DIR/scripts/run_cosmos_paired_smoke.sh"
