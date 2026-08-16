#!/usr/bin/env python3
"""Launch one persistent VLA-JEPA MAX shard per GPU."""

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from libero_max.manifest import load_manifest


def _git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _versions(*packages: str) -> dict:
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--vlajepa-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-interval", type=int, default=7)
    parser.add_argument("--max-cases-per-shard", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_config = {
        "model": "VLA-JEPA-LIBERO",
        "source_revision": _git_revision(args.vlajepa_root),
        "checkpoint": str(args.checkpoint.absolute()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "physical_gpus": gpus,
        "workers": len(gpus),
        "query_interval": args.query_interval,
        "deterministic": True,
        "control_replay_before_event": True,
        "rollout_videos_disabled": True,
        "egl_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "egl_vendor_file": os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"),
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime_versions": _versions(
            "torch", "numpy", "transformers", "opencv-python", "mujoco", "libero"
        ),
    }
    config_path = args.output_root / "run_config.json"
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runner = Path(__file__).resolve().parent / "run_vlajepa_persistent_shard.py"
    processes = []
    handles = []
    for shard_index, gpu in enumerate(gpus):
        command = [
            sys.executable,
            str(runner),
            str(args.manifest.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
            "--shard-index",
            str(shard_index),
            "--num-shards",
            str(len(gpus)),
            "--vlajepa-root",
            str(args.vlajepa_root.resolve()),
            "--checkpoint",
            str(args.checkpoint.absolute()),
            "--query-interval",
            str(args.query_interval),
        ]
        if args.max_cases_per_shard is not None:
            command.extend(["--max-cases", str(args.max_cases_per_shard)])
        if args.resume:
            command.append("--resume")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "MUJOCO_EGL_DEVICE_ID": gpu,
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        log_path = args.output_root / ("worker-shard%d-gpu%s.log" % (shard_index, gpu))
        handle = log_path.open("a" if args.resume else "w", encoding="utf-8")
        handles.append(handle)
        processes.append(
            (
                shard_index,
                gpu,
                subprocess.Popen(
                    command, env=environment, stdout=handle, stderr=subprocess.STDOUT
                ),
            )
        )
    statuses = {}
    try:
        for shard_index, gpu, process in processes:
            statuses["shard%d-gpu%s" % (shard_index, gpu)] = process.wait()
    finally:
        for handle in handles:
            handle.close()
    run_config["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / "worker_status.json").write_text(
        json.dumps(statuses, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if any(code != 0 for code in statuses.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
