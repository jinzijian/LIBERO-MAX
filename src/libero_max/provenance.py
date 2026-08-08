"""Credential-safe provenance records for composed and derived runs."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "auth",
    "cookie",
    "credential",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
)
_URL_CREDENTIALS = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)([^/@\s:]+):([^/@\s]+)@")
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scrub_credentials(value: Any, key: str = "") -> Any:
    """Recursively redact likely credentials from JSON-compatible data."""

    lowered = key.lower()
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(child_key): scrub_credentials(child_value, str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [scrub_credentials(item) for item in value]
    if isinstance(value, str):
        value = _URL_CREDENTIALS.sub(r"\1<redacted>@", value)
        return _BEARER_TOKEN.sub("Bearer <redacted>", value)
    return value


def source_run_configs(source_roots: Iterable[Path]) -> List[Dict[str, Any]]:
    """Load and scrub the run configuration attached to each source root."""

    records: List[Dict[str, Any]] = []
    for source_root in source_roots:
        resolved_root = source_root.resolve()
        config_path = resolved_root / "run_config.json"
        record: Dict[str, Any] = {
            "source_root": str(resolved_root),
            "run_config_present": config_path.is_file(),
        }
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            record.update(
                {
                    "run_config_path": str(config_path),
                    "run_config_sha256": sha256_file(config_path),
                    "run_config": scrub_credentials(config),
                }
            )
        else:
            record["absence_reason"] = "source root has no run_config.json"
        records.append(record)
    return records


def write_run_config(path: Path, config: Dict[str, Any]) -> None:
    """Write a deterministic, credential-scrubbed run configuration."""

    path.write_text(
        json.dumps(scrub_credentials(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
