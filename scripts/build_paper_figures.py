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

# Shared LIBERO-MAX paper palette. Keep these values synchronized with the
# LaTeX definitions and the benchmark SVG figures.
LAB_BLUE = "#3B82F6"
LAB_RED = "#EF4444"
LAB_GRAY = "#6B7280"
LAB_DARK = "#2C2C2C"
LAB_LIGHT_GRAY = "#E5E7EB"
AWARD_ORANGE = "#C2410C"
LAB_GREEN = "#14532D"
LAB_PURPLE = "#41007F"


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
    control_count = sum(bool(row["control_correct"]) for row in records)
    intervention_count = sum(bool(row["intervention_correct"]) for row in records)
    outcomes = {
        "preserved": sum(
            bool(row["control_correct"]) and bool(row["intervention_correct"])
            for row in records
        ),
        "gain": sum(
            not bool(row["control_correct"]) and bool(row["intervention_correct"])
            for row in records
        ),
        "regression": sum(
            bool(row["control_correct"]) and not bool(row["intervention_correct"])
            for row in records
        ),
        "persistent_failure": sum(
            not bool(row["control_correct"])
            and not bool(row["intervention_correct"])
            for row in records
        ),
    }
    control = control_count / total
    intervention = intervention_count / total
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
        "outcomes": outcomes,
        "regression_given_control": (
            outcomes["regression"] / control_count if control_count else None
        ),
        "by_change_type": by_type,
    }


def _label(value: str) -> str:
    labels = {
        "illumination_switch": "Light\nswitch",
        "camera_shift": "Camera\nshift",
        "visual_theme_switch": "Theme\nswitch",
        "sensor_noise_onset": "Sensor\nnoise",
        "target_relocation": "Target\nmove",
        "receptacle_relocation": "Receptacle\nmove",
        "distractor_burst": "Distractor\nburst",
        "obstacle_insertion": "Obstacle\ninsert",
    }
    return labels.get(value, value.replace("_", " ").title())


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
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": LAB_DARK,
            "text.color": LAB_DARK,
            "axes.labelcolor": LAB_DARK,
            "xtick.color": LAB_DARK,
            "ytick.color": LAB_DARK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    control_color = LAB_BLUE
    changed_color = LAB_RED
    gain_color = LAB_GREEN
    failure_color = LAB_LIGHT_GRAY
    regression_color = LAB_RED
    model_names = [run["model"] for run in runs]
    y = np.arange(len(runs))[::-1]
    fig, axis = plt.subplots(figsize=(6.7, 2.25))
    control = np.asarray([run["control"] for run in runs]) * 100
    changed = np.asarray([run["intervention"] for run in runs]) * 100
    for row, c_value, i_value in zip(y, control, changed):
        axis.plot(
            [i_value, c_value], [row, row], color=LAB_GRAY, linewidth=2.1,
            solid_capstyle="round", zorder=1,
        )
    axis.scatter(control, y, s=46, color=control_color, edgecolor="white",
                 linewidth=0.7, label="No change", zorder=3)
    axis.scatter(changed, y, s=46, color=changed_color, edgecolor="white",
                 linewidth=0.7, label="Dynamic change (MAX)", zorder=3)
    for row, c_value, i_value in zip(y, control, changed):
        axis.text(c_value + 1.0, row, f"{c_value:.1f}", va="center", color=control_color)
        axis.text(i_value - 1.0, row, f"{i_value:.1f}", va="center", ha="right",
                  color=changed_color)
        axis.text((c_value + i_value) / 2, row + 0.19,
                  f"{i_value - c_value:+.1f} pp", ha="center", va="bottom",
                  color=LAB_GRAY, fontsize=8)
    axis.set_yticks(y, model_names)
    axis.set_xlim(0, 100)
    axis.set_ylim(-0.55, len(runs) - 0.25)
    axis.set_xlabel("Success rate on the full 8,000-pair denominator (%)")
    axis.grid(axis="x", color=LAB_LIGHT_GRAY, linewidth=0.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.legend(frameon=False, ncol=2, loc="lower center",
                bbox_to_anchor=(0.5, 1.01), handletextpad=0.5, columnspacing=1.5)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.25)
    _write_figure(fig, args.output_dir, "overall_success")
    plt.close(fig)

    # A normalized paired-outcome view separates adaptation failures from the
    # control-floor effect that is hidden by a single changed-success number.
    outcome_order = ["preserved", "gain", "regression", "persistent_failure"]
    outcome_labels = [
        "Both (1,1)", "Dynamic only (0,1)", "No change only (1,0)", "Neither (0,0)"
    ]
    outcome_colors = [control_color, gain_color, regression_color, failure_color]
    outcome_matrix = np.asarray(
        [[run["outcomes"][key] / run["planned"] * 100 for key in outcome_order]
         for run in runs]
    )
    fig, axis = plt.subplots(figsize=(6.7, 2.35))
    left = np.zeros(len(runs))
    for column, (label, color) in enumerate(zip(outcome_labels, outcome_colors)):
        values = outcome_matrix[:, column]
        bars = axis.barh(y, values, left=left, height=0.52, color=color,
                         edgecolor="white", linewidth=0.5, label=label)
        for bar, value in zip(bars, values):
            if value >= 6.5:
                text_color = "white" if color != failure_color else LAB_DARK
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}", ha="center", va="center", color=text_color,
                    fontsize=8,
                )
        left += values
    axis.set_yticks(y, model_names)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of episode IDs (%)")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.grid(axis="x", color=LAB_LIGHT_GRAY, linewidth=0.55, zorder=0)
    axis.legend(frameon=False, ncol=4, loc="lower center",
                bbox_to_anchor=(0.5, 1.01), handlelength=1.2,
                handletextpad=0.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.77, bottom=0.25)
    _write_figure(fig, args.output_dir, "paired_outcomes")
    plt.close(fig)

    types = [
        change_type
        for change_type in CHANGE_ORDER
        if any(change_type in run["by_change_type"] for run in runs)
    ]
    delta = np.full((len(runs), len(types)), np.nan)
    for row_index, run in enumerate(runs):
        for column_index, change_type in enumerate(types):
            metrics = run["by_change_type"].get(change_type)
            if metrics:
                delta[row_index, column_index] = metrics["delta"] * 100

    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    loss = -delta
    red_rgb = to_rgb(LAB_RED)
    loss_map = LinearSegmentedColormap.from_list(
        "max_loss",
        [(*red_rgb, alpha) for alpha in (0.08, 0.35, 0.65, 1.0)],
    )
    fig, axis = plt.subplots(figsize=(6.9, 2.55))
    image = axis.imshow(loss, aspect="auto", cmap=loss_map, vmin=0, vmax=50)
    axis.set_yticks(range(len(runs)), model_names)
    axis.set_xticks(range(len(types)), [_label(value) for value in types])
    axis.tick_params(axis="x", length=0, pad=5)
    axis.tick_params(axis="y", length=0, pad=7)
    for row_index in range(len(runs)):
        for column_index in range(len(types)):
            value = delta[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index, row_index, f"{value:+.1f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if value <= -28 else LAB_DARK,
                    fontweight="bold" if value <= -30 else "normal",
                )
    for boundary in (3.5, 5.5, 6.5):
        axis.axvline(boundary, color="white", linewidth=2.0)
    for x_position, family in ((1.5, "Observation"), (4.5, "Geometry"),
                               (6.0, "Clutter"), (7.0, "Obstacle")):
        axis.text(x_position, 1.09, family, transform=axis.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=8, color=LAB_GRAY)
    axis.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", pad=0.24,
                            fraction=0.10, aspect=45)
    colorbar.set_label("Drop from no change (percentage points)")
    colorbar.outline.set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.80, bottom=0.31)
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
