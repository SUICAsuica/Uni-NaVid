#!/usr/bin/env python3
"""Add explicit goal metadata to a stock Uni-NaVid Task 4 manifest."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--stock-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = json.loads(args.samples_json.read_text())
    stock = json.loads(args.stock_json.read_text())
    metadata_by_id = {sample["id"]: sample for sample in samples}

    if len(metadata_by_id) != len(samples):
        raise ValueError("Sample metadata contains duplicate IDs")

    enriched = []
    for item in stock:
        metadata = metadata_by_id.get(item["id"])
        if metadata is None:
            raise ValueError(f"Missing metadata for {item['id']}")
        enriched.append({
            **item,
            "goal": metadata["goal"],
            "instruction": metadata["instruction"],
            "episode": metadata["episode"],
            "time_index": metadata["time_index"],
            "history_frame_count": metadata["history_frame_count"],
            "scene_id": metadata["scene_id"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(enriched, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "samples": len(enriched),
        "goals": sorted({item["goal"] for item in enriched}),
    }, indent=2))


if __name__ == "__main__":
    main()
