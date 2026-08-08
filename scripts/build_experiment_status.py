#!/usr/bin/env python3
"""Build the publication gate over every new and frozen paper run."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("external runs must use NAME=SUMMARY")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("external runs must use NAME=SUMMARY")
    return name, Path(raw_path)


def _run_status(summary_path: Path) -> Dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    coverage = summary["coverage"]
    end_to_end = summary.get("end_to_end_metrics")
    if end_to_end is None:
        result_path = summary_path.parent / "end_to_end_results.jsonl"
        if not result_path.is_file():
            result_path = summary_path.parent / "paired_results.jsonl"
        rows = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != coverage["planned"]:
            raise ValueError(
                "%s has %d/%d full-denominator rows"
                % (summary_path, len(rows), coverage["planned"])
            )
        control = sum(bool(row["control_correct"]) for row in rows) / len(rows)
        intervention = sum(
            bool(row["intervention_correct"]) for row in rows
        ) / len(rows)
        end_to_end = {
            "control": {"accuracy_on_planned": control},
            "intervention": {"accuracy_on_planned": intervention},
            "paired_robustness_delta_on_planned": intervention - control,
        }
    return {
        "planned": coverage["planned"],
        "triggered": coverage.get("trigger_reached", coverage["completed"]),
        "response_evaluable": coverage["completed"],
        "trigger_unreached": coverage.get("trigger_unreached", 0),
        "response_query_unreached": coverage.get("response_query_unreached", 0),
        "execution_complete": coverage.get(
            "execution_complete", coverage.get("complete", False)
        ),
        "control_accuracy": end_to_end["control"]["accuracy_on_planned"],
        "intervention_accuracy": end_to_end["intervention"][
            "accuracy_on_planned"
        ],
        "paired_delta": end_to_end["paired_robustness_delta_on_planned"],
        "summary_path": str(summary_path),
    }


def build_status(
    paper_root: Path,
    external_runs: Iterable[Tuple[str, Path]],
    expected_runs: int,
) -> Dict[str, Any]:
    runs = {}
    for summary_path in sorted(
        (paper_root / "runs").glob("**/benchmark_summary.json")
    ):
        name = str(summary_path.parent.relative_to(paper_root))
        runs[name] = _run_status(summary_path)
    for name, summary_path in external_runs:
        if name in runs:
            raise ValueError("duplicate paper run name: %s" % name)
        runs[name] = _run_status(summary_path)
    if len(runs) != expected_runs:
        raise ValueError(
            "expected %d paper runs, found %d" % (expected_runs, len(runs))
        )
    return {
        "paper_experiments_complete": all(
            row["execution_complete"] for row in runs.values()
        ),
        "expected_runs": expected_runs,
        "registered_runs": len(runs),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_root", type=Path)
    parser.add_argument("--external-run", action="append", type=_parse_run, default=[])
    parser.add_argument("--expected-runs", type=int, required=True)
    args = parser.parse_args()
    payload = build_status(args.paper_root, args.external_run, args.expected_runs)
    output = args.paper_root / "experiment_status.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["paper_experiments_complete"]:
        raise SystemExit("at least one registered paper run is incomplete")


if __name__ == "__main__":
    main()
