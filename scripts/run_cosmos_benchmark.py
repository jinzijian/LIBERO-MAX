#!/usr/bin/env python3
"""Run a manifest of matched Cosmos control/intervention cases."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from libero_max.manifest import load_manifest


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_case(
    case: Dict[str, Any],
    gpu: str,
    output_root: Path,
    launcher: Path,
    resume: bool,
) -> Tuple[str, bool, str]:
    case_dir = output_root / "cases" / case["case_id"]
    done_path = case_dir / "DONE"
    summary_path = case_dir / "paired_summary.json"
    if resume and done_path.exists() and summary_path.exists():
        return case["case_id"], True, "skipped-complete"
    case_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = case_dir / "scenario.json"
    _write_json(scenario_path, case["scenario"])
    for marker in (done_path, case_dir / "FAILED"):
        if marker.exists():
            marker.unlink()

    env = os.environ.copy()
    env.update(
        {
            "GPU_ID": gpu,
            "OUTPUT_ROOT": str(case_dir.resolve()),
            "SCENARIO_FILE": str(scenario_path.resolve()),
            "SUITE": case["task_suite_name"],
            "TASK_INDEX": str(case["task_index"]),
            "INIT_STATE_INDEX": str(case["init_state_index"]),
            "SEED": str(case["policy_seed"]),
        }
    )
    log_path = case_dir / "launcher.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            ["bash", str(launcher.resolve())],
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = {
        "case_id": case["case_id"],
        "gpu": gpu,
        "returncode": process.returncode,
        "summary_exists": summary_path.exists(),
    }
    _write_json(case_dir / "status.json", status)
    if process.returncode == 0 and summary_path.exists():
        done_path.touch()
        return case["case_id"], True, "completed"
    (case_dir / "FAILED").touch()
    return case["case_id"], False, "returncode=%d" % process.returncode


def _run_gpu_queue(
    gpu: str,
    cases: List[Dict[str, Any]],
    output_root: Path,
    launcher: Path,
    resume: bool,
) -> List[Tuple[str, bool, str]]:
    """Run one sequential queue so a physical GPU is never oversubscribed."""

    return [
        _run_case(case, gpu, output_root, launcher, resume) for case in cases
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--gpus", default="0", help="comma-separated physical GPU IDs"
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=Path(__file__).resolve().parent / "run_cosmos_paired_smoke.sh",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU ID")
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "manifest.json", manifest)

    failures: List[str] = []
    queues = {
        gpu: manifest["cases"][index :: len(gpus)] for index, gpu in enumerate(gpus)
    }
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(
                _run_gpu_queue,
                gpu,
                cases,
                args.output_root,
                args.launcher,
                args.resume,
            ): gpu
            for gpu, cases in queues.items()
            if cases
        }
        for future in as_completed(futures):
            for case_id, ok, detail in future.result():
                print(
                    "%s\t%s\t%s"
                    % (case_id, "ok" if ok else "failed", detail),
                    flush=True,
                )
                if not ok:
                    failures.append(case_id)
    if failures:
        print("failed cases: %s" % ", ".join(sorted(failures)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
