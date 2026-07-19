#!/usr/bin/env python3
"""Compare two Task 4 evaluation JSON files."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(metrics):
    accuracies = metrics["action_accuracy"]
    return {
        **metrics,
        "mean_action_accuracy": sum(accuracies) / len(accuracies),
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


def main():
    args = parse_args()
    baseline_raw = json.loads(args.baseline.read_text())
    candidate_raw = json.loads(args.candidate.read_text())
    if baseline_raw["manifest"] != candidate_raw["manifest"]:
        raise ValueError("Evaluation manifests do not match")
    if baseline_raw["teacher_forced"] != candidate_raw["teacher_forced"]:
        raise ValueError("Evaluation protocols do not match")

    baseline = summarize(baseline_raw["overall"])
    candidate = summarize(candidate_raw["overall"])
    result = {
        "manifest": baseline_raw["manifest"],
        "teacher_forced": baseline_raw["teacher_forced"],
        "baseline_mode": baseline_raw["mode"],
        "candidate_mode": candidate_raw["mode"],
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": delta(candidate, baseline),
        "by_history_bin": {},
    }
    for name in sorted(baseline_raw["by_history_bin"]):
        baseline_bin = summarize(baseline_raw["by_history_bin"][name])
        candidate_bin = summarize(candidate_raw["by_history_bin"][name])
        result["by_history_bin"][name] = {
            "baseline": baseline_bin,
            "candidate": candidate_bin,
            "candidate_minus_baseline": delta(candidate_bin, baseline_bin),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
