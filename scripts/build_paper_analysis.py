#!/usr/bin/env python3
"""Generate evidence-bound analysis prose from complete paper tables."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percent(value: float, signed: bool = False) -> str:
    pattern = "%+.1f%%" if signed else "%.1f%%"
    return pattern % (100.0 * value)


def _points(value: float, signed: bool = True) -> str:
    pattern = "%+.1f points" if signed else "%.1f points"
    return pattern % (100.0 * value)


def _interval(value: List[float], signed: bool = True) -> str:
    if signed:
        return "[%+.1f, %+.1f] points" % (100.0 * value[0], 100.0 * value[1])
    return "[%.1f%%, %.1f%%]" % (100.0 * value[0], 100.0 * value[1])


def _main_result_lines(rows: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for row in rows:
        lines.append(
            "- **{model}:** full control {control} (95% CI {control_ci}), "
            "full changed {change} (95% CI {change_ci}), paired delta {delta} "
            "(95% CI {delta_ci}); trigger/response coverage {trigger}/{response}. "
            "The full 2x2 outcome counts are {preserved} preserved, {recovery} "
            "intervention-side gains, {regression} regressions, and {persistent} "
            "persistent failures.".format(
                model=row["model"],
                control=_percent(row["control_accuracy"]),
                control_ci=_interval(row["control_accuracy_95ci"], signed=False),
                change=_percent(row["intervention_accuracy"]),
                change_ci=_interval(row["intervention_accuracy_95ci"], signed=False),
                delta=_points(row["paired_delta"]),
                delta_ci=_interval(row["paired_delta_95ci_full"]),
                trigger=_percent(row["trigger_coverage"]),
                response=_percent(row["response_coverage"]),
                preserved=row["preserved_capability"],
                recovery=row["recoveries"],
                regression=row["regressions"],
                persistent=row["persistent_failure"],
            )
        )
    return lines


def _track_lines(rows: List[Dict[str, Any]]) -> List[str]:
    by_name = {row["model"]: row for row in rows}
    lines = []
    for label, base_name, pro_name in (
        ("Cosmos Policy", "Cosmos-Base-q16", "Cosmos-PRO-q16"),
        ("pi0.5-LIBERO", "pi0.5-Base-q5", "pi0.5-PRO-q5"),
    ):
        base = by_name[base_name]
        pro = by_name[pro_name]
        lines.append(
            "- **%s:** PRO-Hard changed success is %s versus %s on Base "
            "(%s, descriptive); the paired robustness delta is %s on "
            "PRO-Hard versus %s on Base."
            % (
                label,
                _percent(pro["intervention_accuracy"]),
                _percent(base["intervention_accuracy"]),
                _points(
                    pro["intervention_accuracy"] - base["intervention_accuracy"],
                ),
                _points(pro["paired_delta"]),
                _points(base["paired_delta"]),
            )
        )
    return lines


def _event_lines(table: Dict[str, Any]) -> List[str]:
    grouped = defaultdict(list)
    for row in table["by_change_type"]:
        grouped[row["model"]].append(row)
    lines = []
    for model, rows in grouped.items():
        hardest = min(
            rows,
            key=lambda row: (row["intervention_accuracy"], row["change_type"]),
        )
        largest_gap = min(
            rows, key=lambda row: (row["paired_delta"], row["change_type"])
        )
        lines.append(
            "- **%s:** lowest changed success is `%s` at %s; the largest "
            "control-to-change drop is `%s` at %s."
            % (
                model,
                hardest["change_type"],
                _percent(hardest["intervention_accuracy"]),
                largest_gap["change_type"],
                _points(largest_gap["paired_delta"]),
            )
        )
    return lines


def _draw_lines(table: Dict[str, Any]) -> List[str]:
    grouped = defaultdict(list)
    for row in table["by_type_draw"]:
        grouped[(row["model"], row["change_type"])].append(row)
    spreads = defaultdict(list)
    for (model, change_type), rows in grouped.items():
        if len(rows) != 2:
            continue
        spread = abs(
            rows[0]["intervention_accuracy"] - rows[1]["intervention_accuracy"]
        )
        spreads[model].append((spread, change_type))
    lines = []
    for model, values in spreads.items():
        spread, change_type = max(values)
        lines.append(
            "- **%s:** the largest absolute changed-success difference between "
            "the two frozen draws is %s for `%s`; this is a descriptive "
            "randomization-sensitivity diagnostic, not a significance test."
            % (model, _points(spread, signed=False), change_type)
        )
    return lines


def _comparison_lines(table: Dict[str, Any]) -> List[str]:
    ranked = sorted(
        table["main"], key=lambda row: (-row["intervention_accuracy"], row["model"])
    )
    lines = [
        "- Changed-success ranking: %s."
        % ", ".join(
            "%s %s" % (row["model"], _percent(row["intervention_accuracy"]))
            for row in ranked
        )
    ]
    significant = [
        row
        for row in table["model_comparisons"]
        if row.get("mcnemar_p_holm") is not None and row["mcnemar_p_holm"] < 0.05
    ]
    if not significant:
        lines.append(
            "- No pairwise changed-success difference survives Holm correction "
            "at alpha=0.05."
        )
    else:
        for row in significant:
            lines.append(
                "- **%s vs %s:** right-minus-left changed success %s "
                "(95%% CI %s), Holm-adjusted exact McNemar p=%.4g."
                % (
                    row["left"],
                    row["right"],
                    _points(row["right_minus_left"]),
                    _interval(row["right_minus_left_95ci"]),
                    row["mcnemar_p_holm"],
                )
            )
    return lines


def build_analysis(paper_root: Path) -> str:
    status = _load(paper_root / "experiment_status.json")
    if not status.get("paper_experiments_complete"):
        raise ValueError("paper experiments are not complete")
    main = _load(paper_root / "tables/main/results.json")
    tracks = _load(paper_root / "tables/tracks/results.json")
    comparison = _load(paper_root / "tables/model_comparison/results.json")
    intent = _load(paper_root / "tables/intent/results.json")
    ablation = _load(paper_root / "tables/ablation/results.json")
    review = _load(paper_root / "human_review/human_review_queue.json")

    lines = [
        "# Validated MAX-8000 result analysis",
        "",
        "This summary is generated only after all 15 registered runs are "
        "execution-complete. Percentages below use full frozen denominators "
        "unless explicitly described as response-conditioned. Base and "
        "PRO-Hard contain disjoint task instances, so their difference is "
        "reported descriptively rather than as a paired significance test.",
        "",
        "## Headline end-to-end result",
        "",
        *_main_result_lines(main["main"]),
        "",
        "## Does PRO-Hard increase difficulty?",
        "",
        *_track_lines(tracks["main"]),
        "",
        "## Hardest mid-execution changes",
        "",
        *_event_lines(main),
        "",
        "## Frozen intervention-draw stability",
        "",
        *_draw_lines(main),
        "",
        "## Fair q16 four-model comparison",
        "",
        *_comparison_lines(comparison),
        "",
        "## Intent-response track",
        "",
        *_main_result_lines(intent["main"]),
        "",
        "## Query-interval and event-notification ablations",
        "",
        *_main_result_lines(ablation["main"]),
        "",
        "## Human feasibility follow-up",
        "",
        "- %d cases meet the deterministic risk threshold and %d are queued "
        "for three-attempt expert teleoperation. This list is triage only and "
        "does not change any reported denominator. Every 0/3 defect candidate "
        "is forwarded to an independent second reviewer."
        % (review["candidate_count"], review["selected_count"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_analysis(args.paper_root), encoding="utf-8")


if __name__ == "__main__":
    main()
