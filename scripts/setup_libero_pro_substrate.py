#!/usr/bin/env python3
"""Download and configure the pinned LIBERO-PRO substrate without vendoring it."""

import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-pro-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    subprocess.run(
        [
            "hf",
            "download",
            "zhouxueyang/LIBERO-Pro",
            "--repo-type",
            "dataset",
            "--revision",
            args.revision,
            "--include",
            "bddl_files/**",
            "--include",
            "init_files/**",
            "--include",
            "metadata/**",
            "--include",
            "README.md",
            "--include",
            "SHA256SUMS.txt",
            "--local-dir",
            str(args.dataset_root),
        ],
        check=True,
    )
    package_root = args.libero_pro_root / "libero" / "libero"
    required = [
        package_root / "assets",
        args.dataset_root / "bddl_files",
        args.dataset_root / "init_files",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing LIBERO-PRO paths: %s" % ", ".join(missing))
    config = {
        "benchmark_root": str(package_root.resolve()),
        "bddl_files": str((args.dataset_root / "bddl_files").resolve()),
        "init_states": str((args.dataset_root / "init_files").resolve()),
        "datasets": str((args.libero_pro_root / "libero" / "datasets").resolve()),
        "assets": str((package_root / "assets").resolve()),
    }
    args.config_dir.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML and avoids an extra setup-time dependency.
    (args.config_dir / "config.yaml").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lock = {
        "dataset": "zhouxueyang/LIBERO-Pro",
        "revision": args.revision,
        "libero_pro_root": str(args.libero_pro_root.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "config_dir": str(args.config_dir.resolve()),
    }
    (args.config_dir / "source_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
