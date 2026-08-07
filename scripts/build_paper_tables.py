#!/usr/bin/env python3
"""Build evidence-gated Markdown/JSON tables from complete benchmark runs."""

import argparse
import json
import math
import random
import statistics
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


def _binary_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    if not total:
        return {"n": 0, "control": None, "intervention": None, "delta": None}
    control = sum(bool(record["control_correct"]) for record in records) / total
    intervention = (
        sum(bool(record["intervention_correct"]) for record in records) / total
    )
    return {
        "n": total,
        "control": control,
        "intervention": intervention,
        "delta": intervention - control,
    }


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
        end_to_end = summary.get("end_to_end_metrics")
        if end_to_end is None:
            raise ValueError("%s lacks all-case end-to-end metrics" % run["name"])
        planned = coverage["planned"]
        valid = coverage["completed"]
        triggered = valid
        main_rows.append(
            {
                "model": run["name"],
                "track": summary["protocol"]["scoring_track"],
                "valid_pairs": valid,
                "planned_pairs": planned,
                "trigger_coverage": triggered / planned,
                "control_accuracy": end_to_end["control"]["accuracy_on_planned"],
                "intervention_accuracy": end_to_end["intervention"][
                    "accuracy_on_planned"
                ],
                "paired_delta": end_to_end[
                    "paired_robustness_delta_on_planned"
                ],
                "conditional_paired_delta": metrics["paired_robustness_delta"],
                "paired_delta_95ci": metrics[
                    "paired_robustness_delta_95ci_bootstrap"
                ],
                "regressions": end_to_end["outcome_table"][
                    "regression_under_change"
                ],
                "recoveries": end_to_end["outcome_table"][
                    "intervention_side_gain"
                ],
                "mcnemar_p": metrics["mcnemar_exact_two_sided_p"],
            }
        )

    lines = [
        "| Model | Track | Triggered / planned | Trigger | Full control | Full change | Full delta | Triggered-only delta (95% CI) | Regressions | Recoveries | McNemar p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in main_rows:
        p_value = row["mcnemar_p"]
        lines.append(
            "| {model} | {track} | {valid_pairs}/{planned_pairs} | {trigger} | {control} | {change} | {delta} | {conditional_delta} {ci} | {regressions} | {recoveries} | {p} |".format(
                **row,
                trigger=_percent(row["trigger_coverage"]),
                control=_percent(row["control_accuracy"]),
                change=_percent(row["intervention_accuracy"]),
                delta=_percent(row["paired_delta"]),
                conditional_delta=_percent(row["conditional_paired_delta"]),
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

    severity_names = ["low", "medium", "high"]
    severity_lines = [
        "| Model | " + " | ".join(severity_names) + " |",
        "| --- | " + " | ".join("---:" for _ in severity_names) + " |",
    ]
    for run in runs:
        blocks = run["summary"]["metrics"]["by_severity"]
        severity_lines.append(
            "| %s | %s |"
            % (
                run["name"],
                " | ".join(
                    _percent(blocks.get(name, {}).get("scenario_aware_outcome_accuracy"))
                    for name in severity_names
                ),
            )
        )
    (args.output_dir / "by_severity.md").write_text(
        "\n".join(severity_lines) + "\n", encoding="utf-8"
    )

    stratified_lines = [
        "| Model | Change type | Severity | n | Control | Change | Delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        records = [
            record
            for record in run["records"]
            if record.get("change_type") is not None
            and record.get("severity") is not None
        ]
        groups = sorted(
            {(record["change_type"], record["severity"]) for record in records}
        )
        for change_type, severity in groups:
            metrics = _binary_metrics(
                [
                    record
                    for record in records
                    if record["change_type"] == change_type
                    and record["severity"] == severity
                ]
            )
            stratified_lines.append(
                "| %s | %s | %s | %d | %s | %s | %s |"
                % (
                    run["name"],
                    change_type,
                    severity,
                    metrics["n"],
                    _percent(metrics["control"]),
                    _percent(metrics["intervention"]),
                    _percent(metrics["delta"]),
                )
            )
    (args.output_dir / "by_type_severity.md").write_text(
        "\n".join(stratified_lines) + "\n", encoding="utf-8"
    )

    draw_lines = [
        "| Model | Change type | Draw | n | Control | Change | Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        records = [
            record
            for record in run["records"]
            if record.get("change_type") is not None
            and record.get("intervention_draw_id") is not None
        ]
        groups = sorted(
            {
                (record["change_type"], record["intervention_draw_id"])
                for record in records
            }
        )
        for change_type, draw_id in groups:
            metrics = _binary_metrics(
                [
                    record
                    for record in records
                    if record["change_type"] == change_type
                    and record["intervention_draw_id"] == draw_id
                ]
            )
            draw_lines.append(
                "| %s | %s | %s | %d | %s | %s | %s |"
                % (
                    run["name"],
                    change_type,
                    draw_id,
                    metrics["n"],
                    _percent(metrics["control"]),
                    _percent(metrics["intervention"]),
                    _percent(metrics["delta"]),
                )
            )
    (args.output_dir / "by_type_draw.md").write_text(
        "\n".join(draw_lines) + "\n", encoding="utf-8"
    )

    suite_lines = [
        "| Model | Suite | n | Control | Change | Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        records = [
            record
            for record in run["records"]
            if record.get("task_suite_name") is not None
        ]
        for suite in sorted({record["task_suite_name"] for record in records}):
            metrics = _binary_metrics(
                [record for record in records if record["task_suite_name"] == suite]
            )
            suite_lines.append(
                "| %s | %s | %d | %s | %s | %s |"
                % (
                    run["name"],
                    suite,
                    metrics["n"],
                    _percent(metrics["control"]),
                    _percent(metrics["intervention"]),
                    _percent(metrics["delta"]),
                )
            )
    (args.output_dir / "by_suite.md").write_text(
        "\n".join(suite_lines) + "\n", encoding="utf-8"
    )

    diagnostic_lines = [
        "| Model | Exposure n | Exposure mean / median | Changed post-event chunk | Safety measured | Safety violation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        records = run["records"]
        exposures = [
            record["open_loop_exposure_steps"]
            for record in records
            if record.get("open_loop_exposure_steps") is not None
        ]
        action_mads = [
            record["post_event_action_chunk_mad"]
            for record in records
            if record.get("post_event_action_chunk_mad") is not None
        ]
        safety = run["summary"]["metrics"]["overall"]
        coverage = safety["safety_measurement_coverage"]
        diagnostic_lines.append(
            "| %s | %d | %s | %s | %d/%d | %s |"
            % (
                run["name"],
                len(exposures),
                (
                    "--"
                    if not exposures
                    else "%.2f / %.1f"
                    % (statistics.mean(exposures), statistics.median(exposures))
                ),
                (
                    "--"
                    if not action_mads
                    else _percent(
                        sum(value > 1e-8 for value in action_mads)
                        / len(action_mads)
                    )
                ),
                coverage["measured"],
                coverage["total"],
                _percent(safety["safety_violation_rate"]),
            )
        )
    (args.output_dir / "diagnostics.md").write_text(
        "\n".join(diagnostic_lines) + "\n", encoding="utf-8"
    )

    comparisons = []
    for left_index, left in enumerate(runs):
        left_records = {record["pair_id"]: record for record in left["records"]}
        for right in runs[left_index + 1 :]:
            if (
                left["summary"]["protocol"]["scoring_track"]
                != right["summary"]["protocol"]["scoring_track"]
            ):
                continue
            right_records = {record["pair_id"]: record for record in right["records"]}
            common = sorted(set(left_records) & set(right_records))
            differences = [
                int(right_records[key]["intervention_correct"])
                - int(left_records[key]["intervention_correct"])
                for key in common
            ]
            control_differences = [
                int(right_records[key]["control_correct"])
                - int(left_records[key]["control_correct"])
                for key in common
            ]
            robustness_differences = [
                (
                    int(right_records[key]["intervention_correct"])
                    - int(right_records[key]["control_correct"])
                )
                - (
                    int(left_records[key]["intervention_correct"])
                    - int(left_records[key]["control_correct"])
                )
                for key in common
            ]
            left_only = sum(value == -1 for value in differences)
            right_only = sum(value == 1 for value in differences)
            control_left_only = sum(value == -1 for value in control_differences)
            control_right_only = sum(value == 1 for value in control_differences)
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
                    "control_left_only_success": control_left_only,
                    "control_right_only_success": control_right_only,
                    "control_right_minus_left": (
                        None
                        if not control_differences
                        else sum(control_differences) / len(control_differences)
                    ),
                    "control_right_minus_left_95ci": _paired_delta_ci(
                        control_differences
                    ),
                    "control_mcnemar_p": _mcnemar(
                        control_left_only, control_right_only
                    ),
                    "robustness_right_minus_left": (
                        None
                        if not robustness_differences
                        else sum(robustness_differences)
                        / len(robustness_differences)
                    ),
                    "robustness_right_minus_left_95ci": _paired_delta_ci(
                        robustness_differences
                    ),
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
    control_comparison_lines = [
        "| Comparison | Common pairs | Control right - left (95% CI) | Left-only | Right-only | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        p_value = row["control_mcnemar_p"]
        control_comparison_lines.append(
            "| {left} vs {right} | {common_pairs} | {delta} {ci} | {control_left_only_success} | {control_right_only_success} | {p} |".format(
                **row,
                delta=_percent(row["control_right_minus_left"]),
                ci=_interval(row["control_right_minus_left_95ci"]),
                p="--" if p_value is None else "%.4g" % p_value,
            )
        )
    (args.output_dir / "control_repeatability.md").write_text(
        "\n".join(control_comparison_lines) + "\n", encoding="utf-8"
    )
    robustness_lines = [
        "| Comparison | Common pairs | Difference in paired robustness delta (95% CI) |",
        "| --- | ---: | ---: |",
    ]
    for row in comparisons:
        robustness_lines.append(
            "| {left} vs {right} | {common_pairs} | {delta} {ci} |".format(
                **row,
                delta=_percent(row["robustness_right_minus_left"]),
                ci=_interval(row["robustness_right_minus_left_95ci"]),
            )
        )
    (args.output_dir / "robustness_comparison.md").write_text(
        "\n".join(robustness_lines) + "\n", encoding="utf-8"
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
