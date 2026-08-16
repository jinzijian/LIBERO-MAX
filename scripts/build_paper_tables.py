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


def _wilson_interval(successes: int, total: int) -> Any:
    if not total:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _mcnemar(left_only: int, right_only: int) -> Any:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    tail = sum(
        math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _paired_delta_ci(differences: List[int], samples: int = 5000) -> Any:
    if not differences:
        return None
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        # A nonparametric bootstrap draw is exactly a multinomial draw over the
        # empirical {-1, 0, +1} paired-outcome distribution. Sampling the three
        # counts in native code preserves that definition while avoiding tens
        # of millions of Python-level random-index operations on MAX-8000.
        values = (-1, 0, 1)
        probabilities = np.asarray(
            [differences.count(value) / len(differences) for value in values],
            dtype=np.float64,
        )
        counts = np.random.default_rng(0).multinomial(
            len(differences), probabilities, size=samples
        )
        estimates = np.sort(
            (counts[:, 2] - counts[:, 0]) / float(len(differences))
        )
        return [
            float(estimates[int(0.025 * (samples - 1))]),
            float(estimates[int(0.975 * (samples - 1))]),
        ]
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


def _holm_adjust(p_values: List[Any]) -> List[Any]:
    """Return Holm-Bonferroni adjusted p-values, preserving missing entries."""

    indexed = sorted(
        (float(value), index)
        for index, value in enumerate(p_values)
        if value is not None
    )
    adjusted: List[Any] = [None] * len(p_values)
    running = 0.0
    total = len(indexed)
    for rank, (value, index) in enumerate(indexed):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def _response_coverage(records: List[Dict[str, Any]]) -> float:
    return sum(
        bool(record.get("response_query_reached", record.get("trigger_reached", True)))
        for record in records
    ) / len(records)


def _parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must use MODEL=ROOT")
    name, root = value.split("=", 1)
    if not name or not root:
        raise argparse.ArgumentTypeError("runs must use MODEL=ROOT")
    return name, Path(root)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _markdown_table_to_latex(path: Path) -> None:
    """Write a paper-ready booktabs fragment next to a generated Markdown table."""

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(lines) < 2:
        return

    def cells(line: str) -> List[str]:
        return [cell.strip() for cell in line.strip("|").split("|")]

    header = cells(lines[0])
    separators = cells(lines[1])
    if len(header) != len(separators):
        raise ValueError("malformed Markdown table: %s" % path)
    alignment = "".join(
        "r" if separator.endswith(":") else "l" for separator in separators
    )
    body = [cells(line) for line in lines[2:]]
    if any(len(row) != len(header) for row in body):
        raise ValueError("inconsistent Markdown table width: %s" % path)
    label = "tab:libero-max-%s" % path.stem.replace("_", "-")
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{%s}" % alignment,
        r"\toprule",
        " & ".join(_latex_escape(cell) for cell in header) + r" \\",
        r"\midrule",
    ]
    latex.extend(
        " & ".join(_latex_escape(cell.replace("`", "")) for cell in row) + r" \\"
        for row in body
    )
    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\caption{LIBERO-MAX %s.}" % _latex_escape(path.stem.replace("_", " ")),
            r"\label{%s}" % label,
            r"\end{table*}",
            "",
        ]
    )
    path.with_suffix(".tex").write_text("\n".join(latex), encoding="utf-8")


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


def _end_to_end_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = _binary_metrics(records)
    differences = [
        int(record["intervention_correct"]) - int(record["control_correct"])
        for record in records
    ]
    return {
        "control": {"accuracy_on_planned": metrics["control"]},
        "intervention": {"accuracy_on_planned": metrics["intervention"]},
        "paired_robustness_delta_on_planned": metrics["delta"],
        "outcome_table": {
            "regression_under_change": sum(value == -1 for value in differences),
            "intervention_side_gain": sum(value == 1 for value in differences),
        },
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
        paired_path = root / "paired_results.jsonl"
        # Compact publication bundles may retain the validated aggregate and
        # full-denominator rows while omitting response-conditioned trace rows.
        # The headline and factorized tables use end_to_end_results.jsonl, so
        # keep those builds available and leave trace-only diagnostics blank.
        records = _load_jsonl(paired_path) if paired_path.exists() else []
        end_to_end_path = root / "end_to_end_results.jsonl"
        end_to_end = (
            _load_jsonl(end_to_end_path)
            if end_to_end_path.exists()
            else [{**record, "trigger_reached": True} for record in records]
        )
        if len(end_to_end) != coverage["planned"]:
            raise ValueError("%s lacks complete end-to-end result rows" % name)
        runs.append(
            {
                "name": name,
                "root": root,
                "summary": summary,
                "records": records,
                "end_to_end": end_to_end,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_rows = []
    type_names = sorted(
        {
            change_type
            for run in runs
            for record in run["end_to_end"]
            for change_type in [record.get("change_type")]
            if change_type is not None
        }
    )
    for run in runs:
        summary = run["summary"]
        coverage = summary["coverage"]
        metrics = summary["metrics"]["overall"]
        end_to_end = summary.get("end_to_end_metrics")
        if end_to_end is None:
            end_to_end = _end_to_end_metrics(run["end_to_end"])
        planned = coverage["planned"]
        response_evaluable = coverage["completed"]
        triggered = sum(
            bool(record.get("trigger_reached", True)) for record in run["end_to_end"]
        )
        full_differences = [
            int(record["intervention_correct"]) - int(record["control_correct"])
            for record in run["end_to_end"]
        ]
        full_regressions = sum(value == -1 for value in full_differences)
        full_recoveries = sum(value == 1 for value in full_differences)
        preserved_capability = sum(
            bool(record["control_correct"]) and bool(record["intervention_correct"])
            for record in run["end_to_end"]
        )
        persistent_failure = sum(
            not bool(record["control_correct"])
            and not bool(record["intervention_correct"])
            for record in run["end_to_end"]
        )
        main_rows.append(
            {
                "model": run["name"],
                "track": summary["protocol"]["scoring_track"],
                "triggered_pairs": triggered,
                "response_evaluable_pairs": response_evaluable,
                "planned_pairs": planned,
                "trigger_coverage": triggered / planned,
                "response_coverage": response_evaluable / planned,
                "control_accuracy": end_to_end["control"]["accuracy_on_planned"],
                "control_accuracy_95ci": _wilson_interval(
                    sum(
                        bool(record["control_correct"]) for record in run["end_to_end"]
                    ),
                    planned,
                ),
                "intervention_accuracy": end_to_end["intervention"][
                    "accuracy_on_planned"
                ],
                "intervention_accuracy_95ci": _wilson_interval(
                    sum(
                        bool(record["intervention_correct"])
                        for record in run["end_to_end"]
                    ),
                    planned,
                ),
                "paired_delta": end_to_end["paired_robustness_delta_on_planned"],
                "paired_delta_95ci_full": _paired_delta_ci(full_differences),
                "conditional_paired_delta": metrics["paired_robustness_delta"],
                "paired_delta_95ci": metrics["paired_robustness_delta_95ci_bootstrap"],
                "regressions": full_regressions,
                "recoveries": full_recoveries,
                "preserved_capability": preserved_capability,
                "persistent_failure": persistent_failure,
                "mcnemar_p": _mcnemar(full_regressions, full_recoveries),
            }
        )

    lines = [
        "| Model | Track | Trigger reached | Response-evaluable | Full control (95% CI) | Full change (95% CI) | Full delta (95% CI) | Response-conditioned delta (95% CI) | Regressions | Recoveries | McNemar p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in main_rows:
        p_value = row["mcnemar_p"]
        lines.append(
            "| {model} | {track} | {triggered_pairs}/{planned_pairs} ({trigger}) | {response_evaluable_pairs}/{planned_pairs} ({response}) | {control} {control_ci} | {change} {change_ci} | {delta} {full_delta_ci} | {conditional_delta} {ci} | {regressions} | {recoveries} | {p} |".format(
                **row,
                trigger=_percent(row["trigger_coverage"]),
                response=_percent(row["response_coverage"]),
                control=_percent(row["control_accuracy"]),
                control_ci=_interval(row["control_accuracy_95ci"]),
                change=_percent(row["intervention_accuracy"]),
                change_ci=_interval(row["intervention_accuracy_95ci"]),
                delta=_percent(row["paired_delta"]),
                full_delta_ci=_interval(row["paired_delta_95ci_full"]),
                conditional_delta=_percent(row["conditional_paired_delta"]),
                ci=_interval(row["paired_delta_95ci"]),
                p="--" if p_value is None else "%.4g" % p_value,
            )
        )
    (args.output_dir / "main_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    outcome_lines = [
        "| Model | n | Preserved capability (C=1, I=1) | Intervention-side gain (C=0, I=1) | Regression under change (C=1, I=0) | Persistent failure (C=0, I=0) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in main_rows:
        outcome_lines.append(
            "| {model} | {planned_pairs} | {preserved_capability} | {recoveries} | {regressions} | {persistent_failure} |".format(
                **row
            )
        )
    (args.output_dir / "paired_outcomes.md").write_text(
        "\n".join(outcome_lines) + "\n", encoding="utf-8"
    )

    type_rows = []
    type_lines = [
        "| Model | Change type | n | Trigger | Response | Control | Change | Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        for change_type in type_names:
            rows = [
                record
                for record in run["end_to_end"]
                if record.get("change_type") == change_type
            ]
            if not rows:
                continue
            metrics = _binary_metrics(rows)
            type_rows.append(
                {
                    "model": run["name"],
                    "change_type": change_type,
                    "n": metrics["n"],
                    "trigger_coverage": sum(
                        record["trigger_reached"] for record in rows
                    )
                    / len(rows),
                    "response_coverage": _response_coverage(rows),
                    "control_accuracy": metrics["control"],
                    "intervention_accuracy": metrics["intervention"],
                    "paired_delta": metrics["delta"],
                }
            )
            type_lines.append(
                "| %s | %s | %d | %s | %s | %s | %s | %s |"
                % (
                    run["name"],
                    change_type,
                    metrics["n"],
                    _percent(
                        sum(record["trigger_reached"] for record in rows) / len(rows)
                    ),
                    _percent(_response_coverage(rows)),
                    _percent(metrics["control"]),
                    _percent(metrics["intervention"]),
                    _percent(metrics["delta"]),
                )
            )
    (args.output_dir / "by_change_type.md").write_text(
        "\n".join(type_lines) + "\n", encoding="utf-8"
    )

    severity_names = ["low", "medium", "high"]
    severity_lines = [
        "| Model | Severity | n | Trigger | Response | Control | Change | Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        for severity in severity_names:
            rows = [
                record
                for record in run["end_to_end"]
                if record.get("severity") == severity
            ]
            if not rows:
                continue
            metrics = _binary_metrics(rows)
            severity_lines.append(
                "| %s | %s | %d | %s | %s | %s | %s | %s |"
                % (
                    run["name"],
                    severity,
                    metrics["n"],
                    _percent(
                        sum(record["trigger_reached"] for record in rows) / len(rows)
                    ),
                    _percent(_response_coverage(rows)),
                    _percent(metrics["control"]),
                    _percent(metrics["intervention"]),
                    _percent(metrics["delta"]),
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
            for record in run["end_to_end"]
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

    draw_rows = []
    draw_lines = [
        "| Model | Change type | Draw | n | Control | Change | Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        records = [
            record
            for record in run["end_to_end"]
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
            draw_rows.append(
                {
                    "model": run["name"],
                    "change_type": change_type,
                    "draw_id": draw_id,
                    "n": metrics["n"],
                    "control_accuracy": metrics["control"],
                    "intervention_accuracy": metrics["intervention"],
                    "paired_delta": metrics["delta"],
                }
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
            for record in run["end_to_end"]
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

    substrate_lines = [
        "| Model | Substrate category | n | Trigger | Response | Control | Change | Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        records = [
            record
            for record in run["end_to_end"]
            if record.get("substrate_category") is not None
        ]
        for category in sorted({record["substrate_category"] for record in records}):
            rows = [
                record for record in records if record["substrate_category"] == category
            ]
            metrics = _binary_metrics(rows)
            substrate_lines.append(
                "| %s | %s | %d | %s | %s | %s | %s | %s |"
                % (
                    run["name"],
                    category,
                    metrics["n"],
                    _percent(
                        sum(record["trigger_reached"] for record in rows) / len(rows)
                    ),
                    _percent(_response_coverage(rows)),
                    _percent(metrics["control"]),
                    _percent(metrics["intervention"]),
                    _percent(metrics["delta"]),
                )
            )
    (args.output_dir / "by_substrate_category.md").write_text(
        "\n".join(substrate_lines) + "\n", encoding="utf-8"
    )

    macro_rows = []
    macro_lines = [
        "| Model | Categories | Macro trigger | Macro response | Macro control | Macro change | Macro delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        records = [
            record
            for record in run["end_to_end"]
            if record.get("substrate_category") is not None
        ]
        categories = sorted({record["substrate_category"] for record in records})
        groups = [
            [record for record in records if record["substrate_category"] == category]
            for category in categories
        ]
        if not groups:
            continue
        group_metrics = [_binary_metrics(group) for group in groups]
        row = {
            "model": run["name"],
            "categories": len(groups),
            "trigger": sum(
                sum(bool(record.get("trigger_reached", True)) for record in group)
                / len(group)
                for group in groups
            )
            / len(groups),
            "response": sum(_response_coverage(group) for group in groups)
            / len(groups),
            "control": sum(metric["control"] for metric in group_metrics) / len(groups),
            "intervention": sum(metric["intervention"] for metric in group_metrics)
            / len(groups),
            "delta": sum(metric["delta"] for metric in group_metrics) / len(groups),
        }
        macro_rows.append(row)
        macro_lines.append(
            "| {model} | {categories} | {trigger} | {response} | {control} | {intervention} | {delta} |".format(
                model=row["model"],
                categories=row["categories"],
                trigger=_percent(row["trigger"]),
                response=_percent(row["response"]),
                control=_percent(row["control"]),
                intervention=_percent(row["intervention"]),
                delta=_percent(row["delta"]),
            )
        )
    (args.output_dir / "category_macro.md").write_text(
        "\n".join(macro_lines) + "\n", encoding="utf-8"
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
                        sum(value > 1e-8 for value in action_mads) / len(action_mads)
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
        left_records = {record["pair_id"]: record for record in left["end_to_end"]}
        for right in runs[left_index + 1 :]:
            if (
                left["summary"]["protocol"]["scoring_track"]
                != right["summary"]["protocol"]["scoring_track"]
            ):
                continue
            right_records = {
                record["pair_id"]: record for record in right["end_to_end"]
            }
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
                        else sum(robustness_differences) / len(robustness_differences)
                    ),
                    "robustness_right_minus_left_95ci": _paired_delta_ci(
                        robustness_differences
                    ),
                }
            )
    for key in ("mcnemar_p", "control_mcnemar_p"):
        adjusted = _holm_adjust([row[key] for row in comparisons])
        for row, value in zip(comparisons, adjusted):
            row[key + "_holm"] = value
    comparison_lines = [
        "| Comparison | Common pairs | Right - left (95% CI) | Left-only | Right-only | McNemar p | Holm p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        p_value = row["mcnemar_p"]
        comparison_lines.append(
            "| {left} vs {right} | {common_pairs} | {delta} {ci} | {left_only_success} | {right_only_success} | {p} | {p_holm} |".format(
                **row,
                delta=_percent(row["right_minus_left"]),
                ci=_interval(row["right_minus_left_95ci"]),
                p="--" if p_value is None else "%.4g" % p_value,
                p_holm=(
                    "--"
                    if row["mcnemar_p_holm"] is None
                    else "%.4g" % row["mcnemar_p_holm"]
                ),
            )
        )
    (args.output_dir / "model_comparison.md").write_text(
        "\n".join(comparison_lines) + "\n", encoding="utf-8"
    )
    control_comparison_lines = [
        "| Comparison | Common pairs | Control right - left (95% CI) | Left-only | Right-only | McNemar p | Holm p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        p_value = row["control_mcnemar_p"]
        control_comparison_lines.append(
            "| {left} vs {right} | {common_pairs} | {delta} {ci} | {control_left_only_success} | {control_right_only_success} | {p} | {p_holm} |".format(
                **row,
                delta=_percent(row["control_right_minus_left"]),
                ci=_interval(row["control_right_minus_left_95ci"]),
                p="--" if p_value is None else "%.4g" % p_value,
                p_holm=(
                    "--"
                    if row["control_mcnemar_p_holm"] is None
                    else "%.4g" % row["control_mcnemar_p_holm"]
                ),
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
            {
                "main": main_rows,
                "by_change_type": type_rows,
                "by_type_draw": draw_rows,
                "category_macro": macro_rows,
                "model_comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for table_path in sorted(args.output_dir.glob("*.md")):
        _markdown_table_to_latex(table_path)


if __name__ == "__main__":
    main()
