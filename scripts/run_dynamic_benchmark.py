#!/usr/bin/env python3
"""Run any LIBERO-MAX shard adapter with dynamic GPU work stealing.

The scheduler owns only resource allocation. The model adapter remains
responsible for loading a checkpoint, executing one manifest shard, writing
``DONE`` or ``FAILED`` markers, and honoring ``--resume``. This separation
lets a new policy use LIBERO-MAX without adding model-specific logic here.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from libero_max.manifest import load_manifest


SUITE_PRIORITY = {
    "libero_10": 0,
    "libero_goal": 1,
    "libero_object": 2,
    "libero_spatial": 3,
}


@dataclass(frozen=True)
class WorkUnit:
    suite: str
    manifest: Path
    shard_index: int
    num_shards: int


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def visible_gpus(specification: str, environ: dict[str, str] | None = None) -> list[str]:
    """Resolve explicit GPUs or every GPU visible to the current process."""

    if specification != "auto":
        gpus = _split_csv(specification)
    else:
        environment = os.environ if environ is None else environ
        constrained = environment.get("CUDA_VISIBLE_DEVICES")
        if constrained is not None:
            gpus = _split_csv(constrained)
        else:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            gpus = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not gpus or gpus == ["-1"]:
        raise ValueError("no visible GPU was found; pass --gpus explicitly")
    if len(gpus) != len(set(gpus)):
        raise ValueError("GPU identifiers must be unique")
    return gpus


def _suite_order(name: str) -> tuple[int, str]:
    return SUITE_PRIORITY.get(name, len(SUITE_PRIORITY)), name


def materialize_work_units(
    manifest: dict,
    output_root: Path,
    shards_per_suite: int,
) -> list[WorkUnit]:
    """Write suite-local manifests and return deterministic work units."""

    if shards_per_suite < 1:
        raise ValueError("shards_per_suite must be positive")
    grouped: dict[str, list[dict]] = {}
    for case in manifest["cases"]:
        suite = case.get("task_suite_name")
        if not suite:
            raise ValueError("every case must define task_suite_name")
        grouped.setdefault(str(suite), []).append(case)
    if not grouped:
        raise ValueError("manifest contains no cases")

    manifest_root = output_root / "scheduler" / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    units_by_suite: dict[str, list[WorkUnit]] = {}
    for suite in sorted(grouped, key=_suite_order):
        cases = grouped[suite]
        subset = dict(manifest)
        subset["cases"] = cases
        path = manifest_root / (suite + ".json")
        path.write_text(
            json.dumps(subset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        count = min(shards_per_suite, len(cases))
        units_by_suite[suite] = [
            WorkUnit(suite, path, shard_index, count)
            for shard_index in range(count)
        ]

    # Round-robin across suites starts long LIBERO-10 work early while keeping
    # enough small units available for GPUs that finish sooner.
    units: list[WorkUnit] = []
    max_shards = max(len(items) for items in units_by_suite.values())
    for shard_index in range(max_shards):
        for suite in sorted(units_by_suite, key=_suite_order):
            suite_units = units_by_suite[suite]
            if shard_index < len(suite_units):
                units.append(suite_units[shard_index])
    return units


def build_runner_command(
    python: str,
    runner: Path,
    unit: WorkUnit,
    output_root: Path,
    runner_args: Sequence[str],
) -> list[str]:
    reserved = {"--output-root", "--shard-index", "--num-shards", "--resume"}
    if any(argument.split("=", 1)[0] in reserved for argument in runner_args):
        raise ValueError(
            "runner arguments must not override scheduler-owned shard arguments"
        )
    return [
        python,
        str(runner),
        str(unit.manifest),
        "--output-root",
        str(output_root),
        "--shard-index",
        str(unit.shard_index),
        "--num-shards",
        str(unit.num_shards),
        "--resume",
        *runner_args,
    ]


def _count_markers(root: Path, marker: str) -> int:
    return sum(1 for _ in (root / "cases").glob("*/%s" % marker))


def _run_round(
    *,
    round_index: int,
    units: Iterable[WorkUnit],
    gpus: Sequence[str],
    runner: Path,
    output_root: Path,
    runner_args: Sequence[str],
    status_path: Path,
) -> None:
    work_queue: queue.Queue[WorkUnit] = queue.Queue()
    for unit in units:
        work_queue.put(unit)
    status_lock = threading.Lock()
    log_root = output_root / "scheduler" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    def worker(gpu: str) -> None:
        while True:
            try:
                unit = work_queue.get_nowait()
            except queue.Empty:
                return
            label = "round%d-%s-shard%d-gpu%s" % (
                round_index,
                unit.suite,
                unit.shard_index,
                gpu.replace("/", "_"),
            )
            command = build_runner_command(
                sys.executable, runner, unit, output_root, runner_args
            )
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
            started = _timestamp()
            log_path = log_root / (label + ".log")
            with log_path.open("a", encoding="utf-8") as handle:
                return_code = subprocess.run(
                    command,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
            record = {
                "finished_at_utc": _timestamp(),
                "gpu": gpu,
                "label": label,
                "log": str(log_path.relative_to(output_root)),
                "return_code": return_code,
                "round": round_index,
                "shard_index": unit.shard_index,
                "started_at_utc": started,
                "suite": unit.suite,
            }
            with status_lock:
                with status_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
            work_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamically schedule a LIBERO-MAX shard adapter on all visible GPUs. "
            "Place model-specific runner arguments after '--'."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        default="auto",
        help="comma-separated GPU identifiers, or 'auto' for all visible GPUs",
    )
    parser.add_argument("--shards-per-suite", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.max_rounds < 1:
        parser.error("--max-rounds must be positive")
    if not args.runner.is_file():
        parser.error("--runner must point to a Python shard adapter")
    runner_args = list(args.runner_args)
    if runner_args and runner_args[0] == "--":
        runner_args.pop(0)
    try:
        gpus = visible_gpus(args.gpus)
        manifest = load_manifest(args.manifest)
        args.output_root.mkdir(parents=True, exist_ok=True)
        units = materialize_work_units(
            manifest, args.output_root, args.shards_per_suite
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))

    expected = len(manifest["cases"])
    scheduler_root = args.output_root / "scheduler"
    status_path = scheduler_root / "work_status.jsonl"
    config_path = scheduler_root / "scheduler_config.json"
    config = {
        "expected_cases": expected,
        "gpus": gpus,
        "manifest": str(args.manifest),
        "max_rounds": args.max_rounds,
        "runner": str(args.runner),
        "runner_args": runner_args,
        "scheduling": "suite_round_robin_dynamic_work_stealing",
        "shards_per_suite": args.shards_per_suite,
        "started_at_utc": _timestamp(),
    }
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for round_index in range(1, args.max_rounds + 1):
        _run_round(
            round_index=round_index,
            units=units,
            gpus=gpus,
            runner=args.runner.resolve(),
            output_root=args.output_root.resolve(),
            runner_args=runner_args,
            status_path=status_path,
        )
        done = _count_markers(args.output_root, "DONE")
        failed = _count_markers(args.output_root, "FAILED")
        print(
            "%s round=%d done=%d/%d failed=%d"
            % (_timestamp(), round_index, done, expected, failed),
            flush=True,
        )
        if done == expected and failed == 0:
            config.update(
                {"complete": True, "done": done, "failed": 0, "finished_at_utc": _timestamp()}
            )
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return 0

    config.update(
        {
            "complete": False,
            "done": _count_markers(args.output_root, "DONE"),
            "failed": _count_markers(args.output_root, "FAILED"),
            "finished_at_utc": _timestamp(),
        }
    )
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
