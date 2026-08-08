#!/usr/bin/env python3
"""Launch one persistent LingBot-VA shard per GPU."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

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
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--master-port-base", type=int, default=29600)
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
    checkpoint_bytes = sum(
        path.stat().st_size for path in args.checkpoint.rglob("*") if path.is_file()
    )
    run_config = {
        "model": "LingBot-VA-LIBERO-LONG",
        "source_revision": _git_revision(args.lingbot_root),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": checkpoint_bytes,
        "physical_gpus": gpus,
        "workers": len(gpus),
        "query_interval": manifest["protocol"]["query_interval"],
        "deterministic": False,
        "cuda_recomputation_bitwise_deterministic": False,
        "control_replay_before_event": True,
        "native_cache_replay_before_event": True,
        "exact_runtime_state_snapshot_at_trigger_query": True,
        "paired_input_contract": (
            "byte-identical policy images and MuJoCo state at every pre-event "
            "query, with control action replay and exact control runtime-state "
            "restore before the first post-event response query"
        ),
        "native_cache_chunk": "4 video frames x 4 actions",
        "attention_backend": "torch SDPA",
        "bootstrap": "first conditioned frame replaced by four LIBERO no-ops",
        "rollout_videos_disabled": True,
        "runtime_versions": _versions(
            "torch", "numpy", "numba", "diffusers", "transformers", "mujoco"
        ),
    }
    (args.output_root / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runner = Path(__file__).resolve().parent / "run_lingbot_persistent_shard.py"
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
            "--lingbot-root",
            str(args.lingbot_root.resolve()),
            "--checkpoint",
            str(args.checkpoint.resolve()),
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
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(args.master_port_base + shard_index),
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
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
                    command,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
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
    (args.output_root / "worker_status.json").write_text(
        json.dumps(statuses, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if any(code != 0 for code in statuses.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
