#!/usr/bin/env python3
"""Build evidence-gated Markdown/JSON tables from complete benchmark runs."""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _percent(value: Any) -> str:
    return "--" if value is None else "%.1f" % (100.0 * float(value))


def _interval(value: Any) -> str:
    if value is None:
        return "--"
    return "[%s, %s]" % (_percent(value[0]), _percent(value[1]))


def _mcnemar(left_only: int, right_only: int) -> Any:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _paired_delta_ci(differences: List[int], samples: int = 5000) -> Any:
    if not differences:
        return None
    rng = random.Random(0)
    estimates = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(samples)
    )
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def _parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must use MODEL=ROOT")
    name, root = value.split("=", 1)
    if not name or not root:
        raise argparse.ArgumentTypeError("runs must use MODEL=ROOT")
    return name, Path(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for name, root in args.run:
        summary = _load_json(root / "benchmark_summary.json")
        coverage = summary["coverage"]
        if not coverage.get("execution_complete", coverage.get("complete", False)):
            raise ValueError("%s is not execution-complete" % name)
        records = _load_jsonl(root / "paired_results.jsonl")
        runs.append({"name": name, "root": root, "summary": summary, "records": records})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_rows = []
    type_names = sorted(
        {
            change_type
            for run in runs
            for change_type in run["summary"]["metrics"]["by_change_type"]
        }
    )
    for run in runs:
        summary = run["summary"]
        coverage = summary["coverage"]
        metrics = summary["metrics"]["overall"]
        planned = coverage["planned"]
        valid = coverage["completed"]
        triggered = planned - int(coverage.get("trigger_unreached", 0))
        main_rows.append(
            {
                "model": run["name"],
                "track": summary["protocol"]["scoring_track"],
                "valid_pairs": valid,
                "planned_pairs": planned,
                "trigger_coverage": triggered / planned,
                "control_accuracy": metrics["control_accuracy"],
                "intervention_accuracy": metrics["scenario_aware_outcome_accuracy"],
                "paired_delta": metrics["paired_robustness_delta"],
                "paired_delta_95ci": metrics[
                    "paired_robustness_delta_95ci_bootstrap"
                ],
                "regressions": metrics["outcome_table"]["regression_under_change"],
                "recoveries": metrics["outcome_table"]["intervention_side_gain"],
                "mcnemar_p": metrics["mcnemar_exact_two_sided_p"],
            }
        )

    lines = [
        "| Model | Track | Valid / planned | Trigger | Control | Change | Delta (95% CI) | Regressions | Recoveries | McNemar p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in main_rows:
        p_value = row["mcnemar_p"]
        lines.append(
            "| {model} | {track} | {valid_pairs}/{planned_pairs} | {trigger} | {control} | {change} | {delta} {ci} | {regressions} | {recoveries} | {p} |".format(
                **row,
                trigger=_percent(row["trigger_coverage"]),
                control=_percent(row["control_accuracy"]),
                change=_percent(row["intervention_accuracy"]),
                delta=_percent(row["paired_delta"]),
                ci=_interval(row["paired_delta_95ci"]),
                p="--" if p_value is None else "%.4g" % p_value,
            )
        )
    (args.output_dir / "main_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    type_lines = [
        "| Model | " + " | ".join(type_names) + " |",
        "| --- | " + " | ".join("---:" for _ in type_names) + " |",
    ]
    for run in runs:
        blocks = run["summary"]["metrics"]["by_change_type"]
        type_lines.append(
            "| %s | %s |"
            % (
                run["name"],
                " | ".join(
                    _percent(blocks.get(name, {}).get("scenario_aware_outcome_accuracy"))
                    for name in type_names
                ),
            )
        )
    (args.output_dir / "by_change_type.md").write_text(
        "\n".join(type_lines) + "\n", encoding="utf-8"
    )

    comparisons = []
    for left_index, left in enumerate(runs):
        left_records = {record["pair_id"]: record for record in left["records"]}
        for right in runs[left_index + 1 :]:
            right_records = {record["pair_id"]: record for record in right["records"]}
            common = sorted(set(left_records) & set(right_records))
            differences = [
                int(right_records[key]["intervention_correct"])
                - int(left_records[key]["intervention_correct"])
                for key in common
            ]
            left_only = sum(value == -1 for value in differences)
            right_only = sum(value == 1 for value in differences)
            comparisons.append(
                {
                    "left": left["name"],
                    "right": right["name"],
                    "common_pairs": len(common),
                    "left_only_success": left_only,
                    "right_only_success": right_only,
                    "right_minus_left": (
                        None if not differences else sum(differences) / len(differences)
                    ),
                    "right_minus_left_95ci": _paired_delta_ci(differences),
                    "mcnemar_p": _mcnemar(left_only, right_only),
                }
            )
    comparison_lines = [
        "| Comparison | Common pairs | Right - left (95% CI) | Left-only | Right-only | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        p_value = row["mcnemar_p"]
        comparison_lines.append(
            "| {left} vs {right} | {common_pairs} | {delta} {ci} | {left_only_success} | {right_only_success} | {p} |".format(
                **row,
                delta=_percent(row["right_minus_left"]),
                ci=_interval(row["right_minus_left_95ci"]),
                p="--" if p_value is None else "%.4g" % p_value,
            )
        )
    (args.output_dir / "model_comparison.md").write_text(
        "\n".join(comparison_lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {"main": main_rows, "model_comparisons": comparisons},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
