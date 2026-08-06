#!/usr/bin/env python3
"""Evaluate a manifest shard against one persistent pi0.5 policy server."""

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path

from libero_max.manifest import load_manifest


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port-base", type=int, default=8100)
    parser.add_argument("--query-interval", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard")

    manifest = load_manifest(args.manifest)
    selected = manifest["cases"][args.shard_index :: args.num_shards]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    failures = []
    for ordinal, case in enumerate(selected, start=1):
        case_id = case["case_id"]
        case_dir = args.output_root / "cases" / case_id
        summary_path = case_dir / "paired_summary.json"
        done_path = case_dir / "DONE"
        if args.resume and done_path.exists() and summary_path.exists():
            print("[%d/%d] %s skipped-complete" % (ordinal, len(selected), case_id))
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        for stale in (done_path, case_dir / "FAILED", summary_path):
            if stale.exists():
                stale.unlink()
        scenario_path = case_dir / "scenario.json"
        _write_json(scenario_path, case["scenario"])
        environment = dict(os.environ)
        environment.update(
            {
                "GPU_ID": str(args.gpu_id),
                "OUTPUT_ROOT": str(case_dir),
                "SCENARIO_FILE": str(scenario_path),
                "SUITE": case["task_suite_name"],
                "TASK_INDEX": str(case["task_index"]),
                "INIT_STATE_INDEX": str(case["init_state_index"]),
                "SEED": str(case["policy_seed"]),
                "REPLAN_STEPS": str(args.query_interval),
                "PORT_BASE": str(args.port_base),
            }
        )
        try:
            subprocess.run(
                ["bash", "scripts/run_openpi_paired.sh"],
                check=True,
                env=environment,
            )
            if not summary_path.exists():
                raise RuntimeError("paired runner returned without paired_summary.json")
            _write_json(
                case_dir / "status.json",
                {
                    "case_id": case_id,
                    "shard_index": args.shard_index,
                    "summary_exists": summary_path.exists(),
                },
            )
            done_path.touch()
            print("[%d/%d] %s completed" % (ordinal, len(selected), case_id), flush=True)
        except Exception as exc:
            failures.append(case_id)
            _write_json(
                case_dir / "status.json",
                {
                    "case_id": case_id,
                    "shard_index": args.shard_index,
                    "summary_exists": summary_path.exists(),
                    "error": "%s: %s" % (type(exc).__name__, exc),
                    "traceback": traceback.format_exc(),
                },
            )
            (case_dir / "FAILED").touch()
            print("[%d/%d] %s failed: %s" % (ordinal, len(selected), case_id, exc), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
