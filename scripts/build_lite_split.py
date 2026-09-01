#!/usr/bin/env python3
"""Build the deterministic LIBERO-MAX Lite release from Max."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


SELECTION_SEED = 20260830
LITE_PLUS_PER_EVENT = 70
LITE_PRO_PER_EVENT = 30


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_track(case: dict) -> str:
    return "pro" if case["case_id"].startswith("pro-") else "plus"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_split(
    source_path: Path,
    output_dir: Path,
    selection_seed: int,
) -> None:
    source = json.loads(source_path.read_text())
    cases = source["cases"]
    events = sorted({case["scenario"]["change_type"] for case in cases})
    if len(events) != 8:
        raise ValueError("expected eight event types in the Max manifest")

    selection_rng = random.Random(selection_seed)
    selected = []
    for event in events:
        for track, quota in (
            ("plus", LITE_PLUS_PER_EVENT),
            ("pro", LITE_PRO_PER_EVENT),
        ):
            pool = sorted(
                (
                    case
                    for case in cases
                    if case["scenario"]["change_type"] == event
                    and _source_track(case) == track
                ),
                key=lambda case: case["case_id"],
            )
            if len(pool) < quota:
                raise ValueError(
                    "insufficient %s cases for %s: %d" % (track, event, len(pool))
                )
            selected.extend(selection_rng.sample(pool, quota))

    selected.sort(
        key=lambda case: (
            case["scenario"]["change_type"],
            _source_track(case),
            case["case_id"],
        )
    )
    if len(selected) != 800 or len({case["case_id"] for case in selected}) != 800:
        raise ValueError("Lite split must contain 800 unique case IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "libero_max_lite.json"
    case_index_path = output_dir / "case_index.csv"
    summary_path = output_dir / "selection_summary.json"

    protocol = dict(source["protocol"])
    protocol.update(
        {
            "profile": "lite",
            "selection_contract": (
                "100 cases per event: 70 LIBERO-Plus + 30 LIBERO-PRO"
            ),
        }
    )
    manifest = {
        "benchmark_id": "libero-max-lite",
        "benchmark_version": source["benchmark_version"],
        "protocol": protocol,
        "cases": selected,
    }
    _write_json(manifest_path, manifest)

    with case_index_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=(
                "case_id",
                "event",
                "source_track",
                "task_suite",
                "substrate_category",
                "substrate_difficulty",
            ),
        )
        writer.writeheader()
        for case in selected:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "event": case["scenario"]["change_type"],
                    "source_track": _source_track(case),
                    "task_suite": case["task_suite_name"],
                    "substrate_category": case.get("substrate_category", ""),
                    "substrate_difficulty": case.get("substrate_difficulty", ""),
                }
            )

    event_counts = Counter(
        case["scenario"]["change_type"] for case in selected
    )
    source_counts = Counter(_source_track(case) for case in selected)
    summary = {
        "benchmark_id": "libero-max-lite",
        "benchmark_version": source["benchmark_version"],
        "status": "released",
        "matched_pairs": 800,
        "rollouts_per_checkpoint": 1600,
        "selection_seed": selection_seed,
        "selection_method": (
            "outcome-blind stratified sample with 70 Plus and 30 PRO cases "
            "per event; the same fixed pool is used for cadence analysis"
        ),
        "source_manifest": "../max8000/libero_max_8000.json",
        "source_manifest_sha256": _sha256(source_path),
        "manifest_sha256": _sha256(manifest_path),
        "case_index_sha256": _sha256(case_index_path),
        "event_counts": dict(sorted(event_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }
    _write_json(summary_path, summary)

    checksums = output_dir / "SHA256SUMS"
    lines = [
        "%s  %s" % (_sha256(path), path.name)
        for path in (manifest_path, case_index_path, summary_path)
    ]
    checksums.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("benchmark/max8000/libero_max_8000.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark/lite")
    )
    parser.add_argument(
        "--selection-seed", type=int, default=SELECTION_SEED
    )
    args = parser.parse_args()
    build_split(args.source, args.output_dir, args.selection_seed)


if __name__ == "__main__":
    main()
