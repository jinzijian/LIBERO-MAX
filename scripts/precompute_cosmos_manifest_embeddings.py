#!/usr/bin/env python3
"""Create a Cosmos T5 cache covering every instruction in a MAX manifest."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Set


def manifest_prompts(manifest: Dict[str, Any]) -> Set[str]:
    prompts: Set[str] = set()
    for case in manifest["cases"]:
        variant = case.get("substrate_variant") or {}
        if variant.get("language"):
            prompts.add(variant["language"])
        change = case.get("scenario", {}).get("change", {})
        if change.get("instruction"):
            prompts.add(change["instruction"])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with args.source.open("rb") as handle:
        embeddings = pickle.load(handle)
    prompts = sorted(manifest_prompts(manifest))
    missing = [prompt for prompt in prompts if prompt not in embeddings]
    if missing:
        from cosmos_policy._src.predict2.inference.get_t5_emb import (
            get_text_embedding,
        )

        encoded = get_text_embedding(missing)
        for prompt, embedding in zip(missing, encoded):
            embeddings[prompt] = embedding[None].cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(embeddings, handle)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "source_entries": len(embeddings) - len(missing),
                "manifest_prompts": len(prompts),
                "computed_count": len(missing),
                "computed": missing,
                "output_entries": len(embeddings),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
