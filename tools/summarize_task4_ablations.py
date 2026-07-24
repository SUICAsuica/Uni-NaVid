#!/usr/bin/env python3
"""Combine Task 4 history-ablation evaluations into one comparison."""

import argparse
import json
from pathlib import Path


ABLATIONS = (
    "full",
    "current_only",
    "shuffled",
    "wrong",
    "recent_only",
    "old_only",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(metrics):
    return {
        **metrics,
        "mean_action_accuracy": (
            sum(metrics["action_accuracy"]) / len(metrics["action_accuracy"])
        ),
    }


def delta(candidate, baseline):
    return {
        "loss": candidate["loss"] - baseline["loss"],
        "action_accuracy": [
            candidate_value - baseline_value
            for candidate_value, baseline_value in zip(
                candidate["action_accuracy"],
                baseline["action_accuracy"],
            )
        ],
        "mean_action_accuracy": (
            candidate["mean_action_accuracy"]
            - baseline["mean_action_accuracy"]
        ),
        "first_action_accuracy": (
            candidate["first_action_accuracy"]
            - baseline["first_action_accuracy"]
        ),
        "four_action_exact_match": (
            candidate["four_action_exact_match"]
            - baseline["four_action_exact_match"]
        ),
    }


def build_summary(input_dir):
    result = {}
    for split in ("train", "heldout"):
        evaluations = {
            ablation: json.loads(
                (input_dir / f"{split}_{ablation}.json").read_text()
            )
            for ablation in ABLATIONS
        }
        reference = evaluations["full"]
        for ablation, evaluation in evaluations.items():
            if evaluation["manifest"] != reference["manifest"]:
                raise ValueError(f"{split}/{ablation} manifest does not match")
            if evaluation["mode"] != reference["mode"]:
                raise ValueError(f"{split}/{ablation} mode does not match")
            if evaluation["teacher_forced"] != reference["teacher_forced"]:
                raise ValueError(f"{split}/{ablation} protocol does not match")
            if evaluation["history_ablation"] != ablation:
                raise ValueError(f"{split}/{ablation} is mislabeled")

        baseline = summarize(reference["overall"])
        split_result = {
            "manifest": reference["manifest"],
            "mode": reference["mode"],
            "teacher_forced": reference["teacher_forced"],
            "ablation_seed": reference["ablation_seed"],
            "conditions": {},
        }
        for ablation, evaluation in evaluations.items():
            overall = summarize(evaluation["overall"])
            split_result["conditions"][ablation] = {
                "overall": overall,
                "delta_from_full": delta(overall, baseline),
                "by_history_bin": {
                    name: summarize(metrics)
                    for name, metrics in evaluation["by_history_bin"].items()
                },
            }
        result[split] = split_result
    return result


def main():
    args = parse_args()
    result = build_summary(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
