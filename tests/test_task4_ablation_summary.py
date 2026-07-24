import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_task4_ablations import ABLATIONS, build_summary


class Task4AblationSummaryTest(unittest.TestCase):
    def test_builds_deltas_from_full_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "heldout"):
                for index, ablation in enumerate(ABLATIONS):
                    accuracy = 0.8 - index * 0.01
                    evaluation = {
                        "mode": "goal_cross_attention",
                        "manifest": f"{split}.json",
                        "history_ablation": ablation,
                        "ablation_seed": 42,
                        "overall": {
                            "samples": 2,
                            "loss": 0.1 + index * 0.01,
                            "action_accuracy": [accuracy] * 4,
                            "first_action_accuracy": accuracy,
                            "four_action_exact_match": accuracy,
                        },
                        "by_history_bin": {
                            "short": {
                                "samples": 2,
                                "loss": 0.1 + index * 0.01,
                                "action_accuracy": [accuracy] * 4,
                                "first_action_accuracy": accuracy,
                                "four_action_exact_match": accuracy,
                            }
                        },
                        "teacher_forced": True,
                    }
                    (root / f"{split}_{ablation}.json").write_text(
                        json.dumps(evaluation)
                    )

            summary = build_summary(root)

        current = summary["train"]["conditions"]["current_only"]
        self.assertAlmostEqual(
            current["delta_from_full"]["mean_action_accuracy"],
            -0.01,
        )
        self.assertAlmostEqual(
            current["delta_from_full"]["four_action_exact_match"],
            -0.01,
        )
