#!/usr/bin/env python3
"""Render publication-ready figures from complete full-denominator runs."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


CHANGE_ORDER = [
    "illumination_switch",
    "camera_shift",
    "visual_theme_switch",
    "sensor_noise_onset",
    "target_relocation",
    "receptacle_relocation",
    "distractor_burst",
    "obstacle_insertion",
]


def _parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must use MODEL=ROOT")
    name, root = value.split("=", 1)
    return name, Path(root)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_run(name: str, root: Path) -> Dict[str, Any]:
    summary = json.loads((root / "benchmark_summary.json").read_text(encoding="utf-8"))
    coverage = summary["coverage"]
    if not coverage.get("execution_complete"):
        raise ValueError("%s is not execution-complete" % name)
    path = root / "end_to_end_results.jsonl"
    records = _load_jsonl(path if path.exists() else root / "paired_results.jsonl")
    if len(records) != coverage["planned"]:
        raise ValueError("%s does not contain its full planned denominator" % name)

    by_type = {}
    for change_type in CHANGE_ORDER:
        rows = [row for row in records if row.get("change_type") == change_type]
        if not rows:
            continue
        total = len(rows)
        control = sum(bool(row["control_correct"]) for row in rows) / total
        intervention = sum(bool(row["intervention_correct"]) for row in rows) / total
        trigger = sum(bool(row.get("trigger_reached", True)) for row in rows) / total
        by_type[change_type] = {
            "n": total,
            "control": control,
            "intervention": intervention,
            "delta": intervention - control,
            "trigger_coverage": trigger,
        }
    total = len(records)
    control = sum(bool(row["control_correct"]) for row in records) / total
    intervention = sum(bool(row["intervention_correct"]) for row in records) / total
    return {
        "model": name,
        "planned": total,
        "control": control,
        "intervention": intervention,
        "delta": intervention - control,
        "trigger_coverage": sum(
            bool(row.get("trigger_reached", True)) for row in records
        )
        / total,
        "by_change_type": by_type,
    }


def _label(value: str) -> str:
    return value.replace("_", " ").replace("receptacle", "recept.").title()


def _write_figure(fig: Any, root: Path, stem: str) -> None:
    fig.savefig(root / (stem + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(root / (stem + ".pdf"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    runs = [summarize_run(name, root) for name, root in args.run]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure_data.json").write_text(
        json.dumps({"runs": runs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    model_names = [run["model"] for run in runs]
    x = np.arange(len(runs))
    width = 0.34
    fig, axis = plt.subplots(figsize=(max(6.2, 1.65 * len(runs)), 3.8))
    control = np.asarray([run["control"] for run in runs]) * 100
    changed = np.asarray([run["intervention"] for run in runs]) * 100
    bars_control = axis.bar(
        x - width / 2, control, width, label="Control", color="#4C78A8"
    )
    bars_changed = axis.bar(
        x + width / 2, changed, width, label="Changed", color="#E45756"
    )
    for bars in (bars_control, bars_changed):
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.set_ylabel("Success on full planned denominator (%)")
    axis.set_xticks(x, model_names)
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.set_title("Static competence versus mid-execution change")
    fig.tight_layout()
    _write_figure(fig, args.output_dir, "overall_success")
    plt.close(fig)

    types = [
        change_type
        for change_type in CHANGE_ORDER
        if any(change_type in run["by_change_type"] for run in runs)
    ]
    delta = np.full((len(runs), len(types)), np.nan)
    trigger = np.full((len(runs), len(types)), np.nan)
    for row_index, run in enumerate(runs):
        for column_index, change_type in enumerate(types):
            metrics = run["by_change_type"].get(change_type)
            if metrics:
                delta[row_index, column_index] = metrics["delta"] * 100
                trigger[row_index, column_index] = metrics["trigger_coverage"] * 100

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(7.8, 1.15 * len(types)), 2.25 + 0.72 * len(runs)),
        constrained_layout=True,
    )
    maximum = max(1.0, float(np.nanmax(np.abs(delta))))
    delta_image = axes[0].imshow(
        delta, aspect="auto", cmap="RdBu", vmin=-maximum, vmax=maximum
    )
    trigger_image = axes[1].imshow(
        trigger, aspect="auto", cmap="Blues", vmin=0, vmax=100
    )
    for axis, matrix, percent in (
        (axes[0], delta, False),
        (axes[1], trigger, True),
    ):
        axis.set_yticks(range(len(runs)), model_names)
        axis.set_xticks(range(len(types)))
        if percent:
            axis.set_xticklabels([_label(value) for value in types], rotation=28)
        else:
            axis.set_xticklabels([])
        for row_index in range(len(runs)):
            for column_index in range(len(types)):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    text = "%.1f%%" % value if percent else "%+.1f" % value
                    color = (
                        "white"
                        if (percent and value >= 60)
                        or (not percent and abs(value) >= 0.55 * maximum)
                        else "black"
                    )
                    axis.text(
                        column_index,
                        row_index,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=color,
                    )
    axes[0].set_title("Paired robustness delta by change type (percentage points)")
    axes[1].set_title("Trigger coverage by change type")
    fig.colorbar(delta_image, ax=axes[0], label="Changed minus control (pp)")
    fig.colorbar(trigger_image, ax=axes[1], label="Coverage (%)")
    _write_figure(fig, args.output_dir, "change_type_breakdown")
    plt.close(fig)

    print(
        json.dumps(
            {"runs": len(runs), "change_types": types, "output": str(args.output_dir)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
