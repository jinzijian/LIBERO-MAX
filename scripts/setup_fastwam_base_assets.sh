#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${DEPS_DIR:-$PROJECT_DIR/.deps}"
ASSET_ROOT="${FASTWAM_BASE_ASSETS:-$DEPS_DIR/model-assets/fastwam-base}"
WAN22_REVISION="921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
TOKENIZER_REVISION="37ec512624d61f7aa208f7ea8140a131f93afc9a"

hf download Wan-AI/Wan2.2-TI2V-5B \
  --revision "$WAN22_REVISION" \
  --include Wan2.2_VAE.pth \
  --include models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir "$ASSET_ROOT/Wan-AI/Wan2.2-TI2V-5B"

hf download Wan-AI/Wan2.1-T2V-1.3B \
  --revision "$TOKENIZER_REVISION" \
  --include 'google/umt5-xxl/**' \
  --local-dir "$ASSET_ROOT/Wan-AI/Wan2.1-T2V-1.3B"

python3 - "$ASSET_ROOT" "$WAN22_REVISION" "$TOKENIZER_REVISION" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
files = [
    root / "Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    root / "Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    root / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model",
    root / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json",
]
missing = [str(path) for path in files if not path.is_file()]
if missing:
    raise SystemExit("missing FastWAM base assets: " + ", ".join(missing))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "wan22_repo": "Wan-AI/Wan2.2-TI2V-5B",
    "wan22_revision": sys.argv[2],
    "tokenizer_repo": "Wan-AI/Wan2.1-T2V-1.3B",
    "tokenizer_revision": sys.argv[3],
    "files": {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    },
}
(root / "source_lock.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
