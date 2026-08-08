#!/usr/bin/env python3
"""Replace the README paper-results block from validated final artifacts."""

import argparse
import json
from pathlib import Path


START = "<!-- PAPER_RESULTS_START -->"
END = "<!-- PAPER_RESULTS_END -->"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_section(paper_root: Path) -> str:
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
    main_table = _read(paper_root / "tables/main/main_results.md")
    comparison_table = _read(paper_root / "tables/model_comparison/main_results.md")
    return "\n".join(
        [
            START,
            "## Paper-scale results",
            "",
            "All headline rates below use the full frozen denominator. A policy "
            "that never reaches the proximity trigger remains an end-to-end "
            "failure. A trigger reached too late for the next policy query is "
            "also retained in that denominator but is not imputed into the "
            "response-conditioned diagnostic. Trigger and response-query "
            "coverage are therefore reported separately. "
            "Infrastructure gaps block publication and are never charged to a model.",
            "",
            "### MAX-8000 main evaluation",
            "",
            "Cosmos uses its native 16-step commitment and pi0.5 its native "
            "5-step replanning interval. Their static control arms are the matched "
            "LIBERO-Plus/LIBERO-PRO tasks before the mid-execution change.",
            "",
            main_table,
            "",
            "![Full-denominator control and changed success](results/max8000/figures/main/overall_success.png)",
            "",
            "### Frozen q16 cross-model comparison",
            "",
            "Cosmos Policy, pi0.5, FastWAM, and LingBot-VA are compared on the "
            "same outcome-independent 800-case MAX-PRO subset with a 16-step "
            "evaluator commitment.",
            "",
            comparison_table,
            "",
            "![Cross-model robustness and trigger coverage](results/max8000/figures/model_comparison/change_type_breakdown.png)",
            "",
            "### Simulator intervention gallery",
            "",
            "These are deterministic real-MuJoCo before/after previews, not model "
            "success claims.",
            "",
            "| Lighting | Camera | Visual theme | Sensor noise |",
            "| --- | --- | --- | --- |",
            "| ![Lighting switch](assets/media/illumination-switch.gif) | ![Camera shift](assets/media/camera-shift.gif) | ![Visual theme](assets/media/visual-theme-switch.gif) | ![Sensor noise](assets/media/sensor-noise-onset.gif) |",
            "| Target relocation | Receptacle relocation | Five distractors | Obstacle |",
            "| --- | --- | --- | --- |",
            "| ![Target relocation](assets/media/target-relocation.gif) | ![Receptacle relocation](assets/media/receptacle-relocation.gif) | ![Distractor burst](assets/media/distractor-burst.gif) | ![Obstacle insertion](assets/media/obstacle-insertion.gif) |",
            "",
            "The following GIF reconstructs a real Cosmos intervention trajectory "
            "from its recorded action trace and verifies the exact event step.",
            "",
            "![Audited Cosmos rollout replay](assets/media/cosmos-rollout-replay.gif)",
            "",
            "### Human feasibility secondary review",
            "",
            "%d cases met the risk threshold; the diverse top %d are queued for "
            "three-attempt expert teleoperation. This queue is triage only and "
            "does not remove any case from the reported denominator. See "
            "[`human_review_queue.csv`](results/max8000/human_review/human_review_queue.csv) "
            "and [`docs/HUMAN_FEASIBILITY_REVIEW.md`](docs/HUMAN_FEASIBILITY_REVIEW.md)."
            % (review["candidate_count"], review["selected_count"]),
            "",
            "Machine-readable summaries, full-denominator rows, Markdown and "
            "LaTeX confidence-interval tables, pairwise tests, figures, and "
            "SHA-256 checksums are "
            "under [`results/max8000`](results/max8000).",
            END,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_root", type=Path)
    parser.add_argument("readme", type=Path)
    args = parser.parse_args()
    source = args.readme.read_text(encoding="utf-8")
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError("README must contain exactly one paper-results marker block")
    before, remainder = source.split(START, 1)
    _, after = remainder.split(END, 1)
    args.readme.write_text(
        before.rstrip() + "\n\n" + build_section(args.paper_root) + after,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
