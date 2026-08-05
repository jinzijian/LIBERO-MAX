#!/usr/bin/env python3
"""Extend the Cosmos T5 cache with oracle-notification instructions."""

import argparse
import json
import pickle
from pathlib import Path


def notified_instruction(instruction: str, notification: str) -> str:
    return "%s %s" % (instruction.rstrip(" ."), notification.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--notification", required=True)
    args = parser.parse_args()
    if not args.notification.strip():
        parser.error("--notification must be non-empty")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    prompts = sorted(
        {
            notified_instruction(task["language"], args.notification)
            for task in catalog["tasks"]
        }
    )
    with args.source.open("rb") as handle:
        embeddings = pickle.load(handle)
    source_entries = len(embeddings)
    missing = [prompt for prompt in prompts if prompt not in embeddings]
    if missing:
        from cosmos_policy._src.predict2.inference.get_t5_emb import get_text_embedding

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
                "source_entries": source_entries,
                "prompts": len(prompts),
                "computed": len(missing),
                "output_entries": len(embeddings),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
