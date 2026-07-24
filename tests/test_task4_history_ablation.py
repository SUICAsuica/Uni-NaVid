import unittest

import torch

from tools.evaluate_task4 import (
    apply_history_ablation,
    build_wrong_history_indices,
)


class Task4HistoryAblationTest(unittest.TestCase):
    def setUp(self):
        self.images = torch.arange(5).reshape(5, 1, 1, 1)

    def values(self, images):
        return images.flatten().tolist()

    def test_current_frame_is_preserved_for_all_ablations(self):
        wrong = torch.arange(10, 13).reshape(3, 1, 1, 1)
        for ablation in (
            "full",
            "current_only",
            "shuffled",
            "wrong",
            "recent_only",
            "old_only",
        ):
            result = apply_history_ablation(
                self.images,
                ablation,
                generator=torch.Generator().manual_seed(42),
                wrong_images=wrong,
                recent_window=2,
            )
            self.assertEqual(result[-1].item(), 4)

    def test_history_slices_and_wrong_history(self):
        wrong = torch.arange(10, 13).reshape(3, 1, 1, 1)
        self.assertEqual(
            self.values(
                apply_history_ablation(
                    self.images,
                    "current_only",
                    recent_window=2,
                )
            ),
            [4],
        )
        self.assertEqual(
            self.values(
                apply_history_ablation(
                    self.images,
                    "recent_only",
                    recent_window=2,
                )
            ),
            [2, 3, 4],
        )
        self.assertEqual(
            self.values(
                apply_history_ablation(
                    self.images,
                    "old_only",
                    recent_window=2,
                )
            ),
            [0, 1, 4],
        )
        self.assertEqual(
            self.values(
                apply_history_ablation(
                    self.images,
                    "wrong",
                    wrong_images=wrong,
                    recent_window=2,
                )
            ),
            [10, 11, 4],
        )

    def test_shuffle_changes_only_history_order(self):
        result = apply_history_ablation(
            self.images,
            "shuffled",
            generator=torch.Generator().manual_seed(42),
        )
        self.assertEqual(result[-1].item(), 4)
        self.assertEqual(sorted(self.values(result[:-1])), [0, 1, 2, 3])
        self.assertNotEqual(self.values(result[:-1]), [0, 1, 2, 3])

    def test_wrong_history_prefers_same_goal_bin_and_length(self):
        manifest = [
            {
                "id": "target",
                "episode": "a",
                "goal": "chair",
                "time_index": 70,
            },
            {
                "id": "best",
                "episode": "b",
                "goal": "chair",
                "time_index": 72,
            },
            {
                "id": "wrong-goal",
                "episode": "c",
                "goal": "toilet",
                "time_index": 70,
            },
            {
                "id": "wrong-bin",
                "episode": "d",
                "goal": "chair",
                "time_index": 10,
            },
        ]

        indices = build_wrong_history_indices(manifest)

        self.assertEqual(indices[0], 1)
        for index, wrong_index in enumerate(indices):
            self.assertNotEqual(
                manifest[index]["episode"],
                manifest[wrong_index]["episode"],
            )


if __name__ == "__main__":
    unittest.main()
