#!/usr/bin/env python3
"""Build a diverse, evidence-backed queue for human feasibility review."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CHANGE_RISK = {
    "obstacle_insertion": (3.0, "new path obstacle"),
    "distractor_burst": (2.5, "five-object clutter burst"),
    "target_relocation": (2.0, "target relocated after approach"),
    "receptacle_relocation": (2.0, "receptacle relocated after approach"),
    "sensor_noise_onset": (2.0, "sensor corruption after approach"),
    "camera_shift": (1.5, "camera pose or field-of-view shift"),
    "illumination_switch": (1.0, "large illumination switch"),
    "visual_theme_switch": (1.0, "visual theme switch"),
}


def _substrate_risk(category: str) -> Tuple[float, str]:
    normalized = category.lower().replace("-", "_").replace(" ", "_")
    for token, score, reason in (
        ("view_occlusion", 3.0, "static view occlusion"),
        ("visual_noise", 2.5, "static visual noise or glare"),
        ("camera", 1.5, "static camera perturbation"),
        ("initial_pose", 1.5, "perturbed initial pose"),
        ("object_shape", 1.0, "perturbed object shape"),
        ("object_texture", 1.0, "perturbed object texture"),
    ):
        if token in normalized:
            return score, reason
    return 0.0, ""


def _load_preflight(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for row in report.get("cases", []):
            rows[row["case_id"]] = row
    return rows


def _load_model_controls(specs: Iterable[str]) -> Dict[str, Dict[str, bool]]:
    controls: Dict[str, Dict[str, bool]] = defaultdict(dict)
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--run must use MODEL=RUN_ROOT")
        model, raw_root = spec.split("=", 1)
        root = Path(raw_root)
        end_to_end_path = root / "end_to_end_results.jsonl"
        path = (
            end_to_end_path
            if end_to_end_path.exists()
            else root / "paired_results.jsonl"
        )
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            controls[row["pair_id"]][model] = bool(row["control_correct"])
    return controls


def _score_case(
    case: Dict[str, Any],
    preflight: Dict[str, Any],
    controls: Dict[str, bool],
) -> Dict[str, Any]:
    scenario = case["scenario"]
    change_type = scenario["change_type"]
    category = case.get("substrate_category", "base")
    task = preflight.get("task_description") or (
        case.get("substrate_variant") or {}
    ).get("language", case.get("task_name", ""))
    score = 0.0
    reasons: List[str] = []

    change_score, change_reason = CHANGE_RISK.get(change_type, (0.0, ""))
    score += change_score
    if change_reason:
        reasons.append(change_reason)
    substrate_score, substrate_reason = _substrate_risk(category)
    score += substrate_score
    if substrate_reason:
        reasons.append(substrate_reason)

    lowered = " %s " % task.lower()
    words = task.split()
    if " both " in lowered:
        score += 2.0
        reasons.append("multi-object goal")
    if " and " in lowered or " then " in lowered:
        score += 1.5
        reasons.append("multi-stage instruction")
    if len(words) >= 14:
        score += 1.0
        reasons.append("long instruction")

    minimum_mean = min(
        float(preflight.get("pre_image_mean", 255.0)),
        float(preflight.get("post_image_mean", 255.0)),
    )
    minimum_std = min(
        float(preflight.get("pre_image_std", 255.0)),
        float(preflight.get("post_image_std", 255.0)),
    )
    if minimum_mean < 16.0:
        score += 3.0
        reasons.append("near-dark rendered view")
    elif minimum_mean < 24.0:
        score += 2.0
        reasons.append("low-brightness rendered view")
    if minimum_std < 8.0:
        score += 3.0
        reasons.append("near-threshold image contrast")
    elif minimum_std < 12.0:
        score += 2.0
        reasons.append("low image contrast")
    if (preflight.get("substrate_runtime") or {}).get("state_adapter") != "identity":
        score += 2.0
        reasons.append("topology-extended state adapter")
    if float(preflight.get("max_excess_settle_displacement_m", 0.0)) > 0.01:
        score += 2.0
        reasons.append("large post-event settling margin")

    failed_models = sorted(model for model, success in controls.items() if not success)
    successful_models = sorted(model for model, success in controls.items() if success)
    score += 1.5 * len(failed_models)
    if len(controls) >= 2 and not successful_models:
        score += 4.0
        reasons.append("all evaluated model controls failed")
    if successful_models:
        score -= 4.0
        reasons.append("at least one model completed the matched control")

    return {
        "case_id": case["case_id"],
        "risk_score": round(score, 2),
        "review_status": "pending_human_teleoperation",
        "task_description": task,
        "task_suite_name": case["task_suite_name"],
        "task_index": case["task_index"],
        "init_state_index": case["init_state_index"],
        "substrate_category": category,
        "change_type": change_type,
        "intervention_draw_id": scenario.get("randomization", {}).get("draw_id"),
        "severity": scenario.get("severity"),
        "minimum_image_mean": round(minimum_mean, 3),
        "minimum_image_std": round(minimum_std, 3),
        "failed_model_controls": failed_models,
        "successful_model_controls": successful_models,
        "risk_signals": reasons,
    }


def _diverse_top(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-row["risk_score"], row["case_id"]))
    selected: List[Dict[str, Any]] = []
    seen = set()
    # Reserve one slot for every represented substrate/change cell first.
    for row in ranked:
        cell = (row["substrate_category"], row["change_type"])
        if cell not in seen:
            selected.append(row)
            seen.add(cell)
            if len(selected) == limit:
                return selected
    selected_ids = {row["case_id"] for row in selected}
    selected.extend(row for row in ranked if row["case_id"] not in selected_ids)
    return selected[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--preflight", type=Path, action="append", required=True)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--minimum-score", type=float, default=5.0)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    preflight = _load_preflight(args.preflight)
    controls = _load_model_controls(args.run)
    rows = [
        _score_case(case, preflight.get(case["case_id"], {}), controls[case["case_id"]])
        for case in manifest["cases"]
    ]
    candidates = [row for row in rows if row["risk_score"] >= args.minimum_score]
    selected = _diverse_top(candidates, args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_id": manifest["benchmark_id"],
        "interpretation": (
            "Prioritization signals only. Inclusion is not evidence that a task "
            "is infeasible; a human teleoperator must validate it."
        ),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "model_runs": args.run,
        "cases": selected,
    }
    (args.output_dir / "human_review_queue.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = [
        "case_id",
        "risk_score",
        "review_status",
        "task_description",
        "task_suite_name",
        "task_index",
        "init_state_index",
        "substrate_category",
        "change_type",
        "intervention_draw_id",
        "severity",
        "minimum_image_mean",
        "minimum_image_std",
        "failed_model_controls",
        "successful_model_controls",
        "risk_signals",
    ]
    with (args.output_dir / "human_review_queue.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    **row,
                    "failed_model_controls": "; ".join(row["failed_model_controls"]),
                    "successful_model_controls": "; ".join(
                        row["successful_model_controls"]
                    ),
                    "risk_signals": "; ".join(row["risk_signals"]),
                }
            )
    markdown = [
        "# Human feasibility review queue",
        "",
        payload["interpretation"],
        "",
        "| Rank | Case | Score | Substrate | Change | Risk signals |",
        "| ---: | --- | ---: | --- | --- | --- |",
    ]
    for index, row in enumerate(selected, start=1):
        markdown.append(
            "| %d | `%s` | %.2f | %s | %s | %s |"
            % (
                index,
                row["case_id"],
                row["risk_score"],
                row["substrate_category"],
                row["change_type"],
                "; ".join(row["risk_signals"]),
            )
        )
    (args.output_dir / "human_review_queue.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in payload if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
