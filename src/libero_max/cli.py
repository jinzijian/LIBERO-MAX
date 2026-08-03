"""Command-line interface for the LIBERO-MAX benchmark core."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .results import ResultLoadError, load_results_jsonl, summarize_results
from .manifest import ManifestLoadError, load_manifest
from .scenario import (
    ScenarioLoadError,
    load_scenarios,
    validate_scenario_collection,
)


def _write_json(payload: Dict[str, Any], output: Optional[Path]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def _validate_command(args: argparse.Namespace) -> int:
    try:
        scenarios = load_scenarios(Path(path) for path in args.paths)
    except ScenarioLoadError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    errors = validate_scenario_collection(scenarios)
    payload: Dict[str, Any] = {
        "valid": not errors,
        "scenario_count": len(scenarios),
        "errors": errors,
    }
    _write_json(payload, args.output)
    return 0 if not errors else 1


def _summarize_command(args: argparse.Namespace) -> int:
    try:
        records = load_results_jsonl(args.results)
        scenarios = None
        if args.scenarios:
            scenarios = load_scenarios(Path(path) for path in args.scenarios)
            errors = validate_scenario_collection(scenarios)
            if errors:
                raise ScenarioLoadError("; ".join(errors))
        summary = summarize_results(records, scenarios)
    except (ResultLoadError, ScenarioLoadError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    _write_json(summary, args.output)
    return 0


def _validate_manifest_command(args: argparse.Namespace) -> int:
    try:
        manifest = load_manifest(args.path)
        payload = {
            "valid": True,
            "benchmark_id": manifest["benchmark_id"],
            "case_count": len(manifest["cases"]),
            "errors": [],
        }
    except ManifestLoadError as exc:
        payload = {"valid": False, "errors": [str(exc)]}
    _write_json(payload, args.output)
    return 0 if payload["valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="libero-max")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate scenario JSON")
    validate.add_argument("paths", nargs="+", help="scenario files or directories")
    validate.add_argument("--output", type=Path, help="write the JSON report")
    validate.set_defaults(handler=_validate_command)

    summarize = subparsers.add_parser(
        "summarize", help="summarize matched paired-result JSONL"
    )
    summarize.add_argument("results", type=Path, help="paired-result JSONL")
    summarize.add_argument(
        "--scenarios", nargs="+", help="planned scenario files or directories"
    )
    summarize.add_argument("--output", type=Path, help="write the JSON report")
    summarize.set_defaults(handler=_summarize_command)

    validate_manifest_parser = subparsers.add_parser(
        "validate-manifest", help="validate a benchmark execution manifest"
    )
    validate_manifest_parser.add_argument("path", type=Path)
    validate_manifest_parser.add_argument("--output", type=Path)
    validate_manifest_parser.set_defaults(handler=_validate_manifest_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
