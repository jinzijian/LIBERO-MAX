#!/usr/bin/env python3
"""Render the public ten-model LIBERO-MAX comparison from frozen JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAB_BLUE = "#3B82F6"
LAB_RED = "#EF4444"
LAB_GRAY = "#6B7280"
LAB_DARK = "#2C2C2C"
LAB_LIGHT_GRAY = "#E5E7EB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path("assets/figures/main_results.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("assets/figures/main_results.png")
    )
    args = parser.parse_args()

    rows = json.loads(args.data.read_text(encoding="utf-8"))["models"]
    labels = ["π0.5" if row["model"] == "pi0.5" else row["model"] for row in rows]
    base = np.asarray([row["base_sr"] for row in rows])
    dynamic = np.asarray([row["dynamic_sr"] for row in rows])
    families = [row["family"] for row in rows]
    y = np.arange(len(rows))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(12.8, 7.6), facecolor="white")
    for row_index in range(len(rows)):
        if row_index % 2:
            ax.axhspan(row_index - 0.5, row_index + 0.5, color="#F7F7F5", zorder=0)
    ax.hlines(y, dynamic, base, color=LAB_GRAY, linewidth=3.2, zorder=1)
    ax.scatter(base, y, s=90, color=LAB_BLUE, edgecolor="white", linewidth=1.5,
               label="Base: no change", zorder=3)
    ax.scatter(dynamic, y, s=84, marker="s", color=LAB_RED, edgecolor="white",
               linewidth=1.5, label="Dynamic change (MAX)", zorder=3)

    for index, row in enumerate(rows):
        ax.text(base[index] + 1.0, index, f"{base[index]:.1f}", va="center",
                color=LAB_BLUE, fontsize=11, fontweight="bold")
        ax.text(dynamic[index] - 1.0, index, f"{dynamic[index]:.1f}", va="center",
                ha="right", color=LAB_RED, fontsize=11, fontweight="bold")
        midpoint = (base[index] + dynamic[index]) / 2
        ax.text(midpoint, index - 0.27, f"{row['gap_pp']:.1f} pp", ha="center",
                va="bottom", color=LAB_GRAY, fontsize=9.5)

    family_starts = []
    previous = None
    for index, family in enumerate(families):
        if family != previous:
            family_starts.append((index, family))
            if index:
                ax.axhline(index - 0.5, color=LAB_LIGHT_GRAY, linewidth=1.2)
            previous = family
    for start, family in family_starts:
        ax.text(-0.305, start, family.upper(), transform=ax.get_yaxis_transform(),
                ha="left", va="center", color=LAB_GRAY, fontsize=8.5,
                fontweight="bold", clip_on=False)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Success rate over the full 8,000-pair denominator (%)",
                  color=LAB_DARK, labelpad=12)
    ax.grid(axis="x", color=LAB_LIGHT_GRAY, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=13, colors=LAB_DARK)
    ax.tick_params(axis="x", length=0, colors=LAB_GRAY)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LAB_DARK)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=2,
              frameon=False, fontsize=11, handletextpad=0.5, columnspacing=2.2)
    fig.subplots_adjust(left=0.25, right=0.96, top=0.90, bottom=0.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
