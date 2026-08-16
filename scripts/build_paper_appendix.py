#!/usr/bin/env python3
"""Assemble all evidence-gated experiment tables into one paper appendix."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple


TABLES: List[Tuple[str, str, str]] = [
    ("Headline MAX-8000 evaluation", "tables/main/main_results.md", "main"),
    (
        "Complete paired outcome decomposition",
        "tables/main/paired_outcomes.md",
        "main",
    ),
    ("Base versus PRO-Hard tracks", "tables/tracks/main_results.md", "main"),
    ("Substrate-category macro average", "tables/main/category_macro.md", "main"),
    ("Change-type breakdown", "tables/main/by_change_type.md", "main"),
    ("Severity breakdown", "tables/main/by_severity.md", "main"),
    ("Intervention-draw stability", "tables/main/by_type_draw.md", "main"),
    ("Task-suite breakdown", "tables/main/by_suite.md", "main"),
    (
        "Substrate-category breakdown",
        "tables/main/by_substrate_category.md",
        "main",
    ),
    ("Response diagnostics", "tables/main/diagnostics.md", "main"),
    (
        "Full native-protocol four-model result",
        "tables/model_comparison/main_results.md",
        "comparison",
    ),
    (
        "Four-model paired intervention comparison",
        "tables/model_comparison/model_comparison.md",
        "comparison",
    ),
    (
        "Four-model robustness-gap comparison",
        "tables/model_comparison/robustness_comparison.md",
        "comparison",
    ),
    (
        "Cross-run control repeatability",
        "tables/model_comparison/control_repeatability.md",
        "comparison",
    ),
    ("Intent revision result", "tables/intent/main_results.md", "intent"),
    ("Intent-type breakdown", "tables/intent/by_change_type.md", "intent"),
    ("Intent response diagnostics", "tables/intent/diagnostics.md", "intent"),
    ("Query/notification ablations", "tables/ablation/main_results.md", "ablation"),
    (
        "Paired ablation comparisons",
        "tables/ablation/model_comparison.md",
        "ablation",
    ),
    ("Ablation diagnostics", "tables/ablation/diagnostics.md", "ablation"),
]


def build_appendix(paper_root: Path) -> str:
    status = json.loads(
        (paper_root / "experiment_status.json").read_text(encoding="utf-8")
    )
    if not status.get("paper_experiments_complete"):
        raise ValueError("paper experiments are not complete")
    review = json.loads(
        (paper_root / "human_review/human_review_queue.json").read_text(
            encoding="utf-8"
        )
    )
    analysis_path = paper_root / "paper/MAX8000_ANALYSIS.md"
    if not analysis_path.is_file():
        raise FileNotFoundError(
            "required paper analysis is missing: %s" % analysis_path
        )
    analysis = analysis_path.read_text(encoding="utf-8").strip()
    analysis_lines = analysis.splitlines()
    if analysis_lines and analysis_lines[0].startswith("# "):
        analysis_lines = analysis_lines[1:]
    analysis = "\n".join(
        "### " + line[3:] if line.startswith("## ") else line for line in analysis_lines
    ).lstrip()
    grouped = {
        "main": [],
        "comparison": [],
        "intent": [],
        "ablation": [],
    }
    for title, relative, group in TABLES:
        path = paper_root / relative
        if not path.is_file():
            raise FileNotFoundError("required paper table is missing: %s" % path)
        grouped[group].append((title, path.read_text(encoding="utf-8").strip()))

    lines = [
        "# LIBERO-MAX-8000 paper experiment appendix",
        "",
        "This appendix is generated only after every planned run is execution-"
        "complete. Headline control/change rates use each track's full frozen "
        "denominator. `trigger_unreached` and late-trigger/no-response-query "
        "episodes remain end-to-end model outcomes; infrastructure gaps block "
        "generation rather than being charged to a policy.",
        "",
        "## Validated analysis",
        "",
        analysis,
        "",
        "## A. Main physical evaluation",
        "",
    ]
    for title, table in grouped["main"]:
        lines.extend(["### %s" % title, "", table, ""])
    lines.extend(
        [
            "## B. Full MAX-8000 native-protocol cross-model comparison",
            "",
            "All four models use the exact same 8,000-case Base + PRO union at "
            "their native query intervals. The frozen 800-case PRO subset is reused within these runs, "
            "not substituted for the full denominator. Raw and Holm-adjusted "
            "exact McNemar p-values are reported for the six pairwise comparisons.",
            "",
        ]
    )
    for title, table in grouped["comparison"]:
        lines.extend(["### %s" % title, "", table, ""])
    lines.extend(["## C. Intent-response track", ""])
    for title, table in grouped["intent"]:
        lines.extend(["### %s" % title, "", table, ""])
    lines.extend(["## D. Query interval and event-notification ablations", ""])
    for title, table in grouped["ablation"]:
        lines.extend(["### %s" % title, "", table, ""])
    protocol = review["human_review_protocol"]
    lines.extend(
        [
            "## E. Human feasibility secondary review",
            "",
            "%d cases meet the deterministic risk threshold; the diverse top "
            "%d are queued for expert teleoperation. The ranking is triage only "
            "and changes no benchmark denominator. Each case receives %d attempts; "
            "%s. If all attempts fail, %s. %s."
            % (
                review["candidate_count"],
                review["selected_count"],
                protocol["attempts_per_case"],
                protocol["feasible_rule"],
                protocol["failure_rule"],
                protocol["secondary_review_rule"],
            ),
            "",
            "The machine-readable run summaries, full-denominator rows, LaTeX "
            "versions of every table, figures, review CSV, provenance, and "
            "SHA-256 checksums are distributed alongside this appendix.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_appendix(args.paper_root), encoding="utf-8")


if __name__ == "__main__":
    main()
