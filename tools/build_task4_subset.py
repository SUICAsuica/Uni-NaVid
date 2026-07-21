#!/usr/bin/env python3
"""Build a history-length-balanced stock-loader manifest for Task 4."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


BINS = (
    ("short", 0, 64),
    ("medium", 64, 128),
    ("long", 128, 256),
    ("extra_long", 256, None),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--stock-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-bin", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help="Exclude samples whose episode appears in this stock-loader manifest",
    )
    parser.add_argument(
        "--all-eligible",
        action="store_true",
        help="Write every eligible sample instead of sampling --per-bin items",
    )
    return parser.parse_args()


def bin_name(time_index):
    for name, lower, upper in BINS:
        if time_index >= lower and (upper is None or time_index < upper):
            return name
    raise ValueError(f"Invalid time_index: {time_index}")


def main():
    args = parse_args()
    if args.per_bin < 1:
        raise ValueError("--per-bin must be positive")

    samples = json.loads(args.samples_json.read_text())
    stock = json.loads(args.stock_json.read_text())
    stock_by_id = {item["id"]: item for item in stock}
    if len(stock_by_id) != len(stock):
        raise ValueError("Stock manifest contains duplicate IDs")

    excluded_episodes = set()
    if args.exclude_manifest:
        excluded = json.loads(args.exclude_manifest.read_text())
        missing_episode = [
            item.get("id", "<unknown>")
            for item in excluded
            if "episode" not in item
        ]
        if missing_episode:
            raise ValueError(
                "Exclusion manifest entries must contain episode; missing for "
                f"{missing_episode[:5]}"
            )
        excluded_episodes = {item["episode"] for item in excluded}

    eligible = [
        sample for sample in samples if sample["episode"] not in excluded_episodes
    ]
    grouped = defaultdict(list)
    for sample in eligible:
        grouped[bin_name(sample["time_index"])].append(sample)

    rng = random.Random(args.seed)
    selected = []
    selected_episodes = set()
    summary = {}
    for name, _, _ in BINS:
        candidates = grouped[name]
        if not args.all_eligible and len(candidates) < args.per_bin:
            raise ValueError(
                f"Bin {name} has {len(candidates)} samples, fewer than {args.per_bin}"
            )
        chosen = (
            candidates
            if args.all_eligible
            else rng.sample(candidates, args.per_bin)
        )
        missing = [sample["id"] for sample in chosen if sample["id"] not in stock_by_id]
        if missing:
            raise ValueError(f"IDs missing from stock manifest: {missing[:5]}")
        selected.extend(stock_by_id[sample["id"]] for sample in chosen)
        selected_episodes.update(sample["episode"] for sample in chosen)
        summary[name] = {
            "samples": len(chosen),
            "trajectories": len({sample["episode"] for sample in chosen}),
            "scenes": len({sample["scene_id"] for sample in chosen}),
        }

    rng.shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "total": len(selected),
                "episodes": len(selected_episodes),
                "excluded_episodes": len(excluded_episodes),
                **summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
