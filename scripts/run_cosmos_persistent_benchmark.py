#!/usr/bin/env python3
"""Launch one persistent Cosmos model process per GPU for a MAX manifest."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from libero_max.manifest import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--shard-indices")
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--t5-embeddings", type=Path, required=True)
    parser.add_argument("--query-interval", type=int)
    parser.add_argument("--max-cases-per-shard", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU")
    shard_indices = (
        list(range(len(gpus)))
        if args.shard_indices is None
        else [
            int(item.strip())
            for item in args.shard_indices.split(",")
            if item.strip()
        ]
    )
    num_shards = args.num_shards or len(gpus)
    if len(shard_indices) != len(gpus):
        parser.error("--shard-indices must contain one index per GPU")
    if (
        num_shards < 1
        or len(set(shard_indices)) != len(shard_indices)
        or any(index < 0 or index >= num_shards for index in shard_indices)
    ):
        parser.error("invalid logical shard configuration")
    if args.query_interval is not None and args.query_interval < 1:
        parser.error("--query-interval must be positive")
    if args.max_cases_per_shard is not None and args.max_cases_per_shard < 1:
        parser.error("--max-cases-per-shard must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint = (
        args.t5_embeddings.resolve().parent
        / "Cosmos-Policy-LIBERO-Predict2-2B.pt"
    )
    run_config = {
        "model": "Cosmos-Policy-LIBERO-Predict2-2B",
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "t5_embeddings": str(args.t5_embeddings.resolve()),
        "t5_embeddings_bytes": args.t5_embeddings.stat().st_size,
        "physical_gpus": gpus,
        "workers": len(gpus),
        "logical_shard_indices": shard_indices,
        "logical_num_shards": num_shards,
        "query_interval": (
            args.query_interval
            if args.query_interval is not None
            else manifest["protocol"]["query_interval"]
        ),
        "policy_seed_contract": sorted(
            {case["policy_seed"] for case in manifest["cases"]}
        ),
        "deterministic": True,
        "control_replay_before_event": True,
        "rollout_videos_disabled": True,
    }
    (args.output_root / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runner = Path(__file__).resolve().parent / "run_cosmos_persistent_shard.py"
    processes = []
    log_handles = []
    for shard_index, gpu in zip(shard_indices, gpus):
        command = [
            sys.executable,
            str(runner),
            str(args.manifest.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
            "--shard-index",
            str(shard_index),
            "--num-shards",
            str(num_shards),
            "--t5-embeddings",
            str(args.t5_embeddings.resolve()),
        ]
        if args.query_interval is not None:
            command.extend(["--query-interval", str(args.query_interval)])
        if args.max_cases_per_shard is not None:
            command.extend(
                ["--max-cases", str(args.max_cases_per_shard)]
            )
        if args.resume:
            command.append("--resume")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "MUJOCO_EGL_DEVICE_ID": gpu,
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            }
        )
        log_path = args.output_root / (
            "worker-shard%d-gpu%s.log" % (shard_index, gpu)
        )
        log_handle = log_path.open("a" if args.resume else "w", encoding="utf-8")
        log_handles.append(log_handle)
        processes.append(
            (
                gpu,
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                ),
            )
        )
    statuses = {}
    try:
        for gpu, process in processes:
            statuses[gpu] = process.wait()
    finally:
        for handle in log_handles:
            handle.close()
    shard_tag = "-".join(str(index) for index in shard_indices)
    status_path = args.output_root / ("worker_status_%s.json" % shard_tag)
    status_path.write_text(
        json.dumps(statuses, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failures = {gpu: code for gpu, code in statuses.items() if code != 0}
    print(json.dumps({"workers": statuses, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
