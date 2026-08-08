#!/usr/bin/env python3
"""Create a compact, auditable GitHub bundle from completed paper artifacts."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable


REQUIRED_MEDIA = {
    "camera-shift-after.png",
    "camera-shift.gif",
    "cosmos-rollout-replay.gif",
    "cosmos-rollout-replay.json",
    "distractor-burst-after.png",
    "distractor-burst.gif",
    "illumination-switch-after.png",
    "illumination-switch.gif",
    "intervention-overview.png",
    "media_manifest.json",
    "obstacle-insertion-after.png",
    "obstacle-insertion.gif",
    "receptacle-relocation-after.png",
    "receptacle-relocation.gif",
    "sensor-noise-onset-after.png",
    "sensor-noise-onset.gif",
    "target-relocation-after.png",
    "target-relocation.gif",
    "visual-theme-switch-after.png",
    "visual-theme-switch.gif",
}


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError("required paper artifact is missing: %s" % source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _files(root: Path, patterns: Iterable[str]) -> Iterable[Path]:
    for pattern in patterns:
        yield from sorted(root.glob(pattern))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--media-output-dir", type=Path)
    args = parser.parse_args()

    status_path = args.paper_root / "experiment_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not status.get("paper_experiments_complete"):
        raise ValueError("paper experiment bundle is not complete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _copy(status_path, args.output_dir / "experiment_status.json")
    for source in _files(
        args.paper_root,
        (
            "tables/**/*.md",
            "tables/**/*.tex",
            "tables/**/*.json",
            "paper/*.md",
            "human_review/*.md",
            "human_review/*.json",
            "human_review/*.csv",
            "figures/**/*.png",
            "figures/**/*.pdf",
            "figures/**/*.json",
        ),
    ):
        _copy(source, args.output_dir / source.relative_to(args.paper_root))

    for summary_path in sorted(
        (args.paper_root / "runs").glob("**/benchmark_summary.json")
    ):
        run_root = summary_path.parent
        relative = run_root.relative_to(args.paper_root)
        for name in (
            "benchmark_summary.json",
            "end_to_end_results.jsonl",
            "paired_results.jsonl",
            "manifest.json",
            "run_config.json",
        ):
            _copy_required(run_root / name, args.output_dir / relative / name)

    if args.media_output_dir:
        media_source_dir = args.paper_root / "media"
        missing_media = sorted(
            name for name in REQUIRED_MEDIA if not (media_source_dir / name).is_file()
        )
        if missing_media:
            raise FileNotFoundError(
                "required paper media are missing: %s" % ", ".join(missing_media)
            )
        args.media_output_dir.mkdir(parents=True, exist_ok=True)
        media_release_files = []
        for source in sorted(media_source_dir.glob("*")):
            if source.suffix in {".gif", ".json", ".png"}:
                destination = args.media_output_dir / source.name
                _copy(source, destination)
                media_release_files.append(destination)
        (args.media_output_dir / "SHA256SUMS").write_text(
            "".join(
                "%s  %s\n" % (_sha256(path), path.name) for path in media_release_files
            ),
            encoding="utf-8",
        )

    release_files = [
        path
        for path in sorted(args.output_dir.glob("**/*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(
            "%s  %s\n" % (_sha256(path), path.relative_to(args.output_dir))
            for path in release_files
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "release_files": len(release_files),
                "media_output_dir": (
                    str(args.media_output_dir) if args.media_output_dir else None
                ),
                "media_release_files": (
                    len(media_release_files) if args.media_output_dir else 0
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
