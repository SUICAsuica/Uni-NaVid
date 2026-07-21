import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value))


class BuildTask4SubsetTest(unittest.TestCase):
    def test_excludes_entire_episodes_and_keeps_all_eligible(self):
        samples = [
            {
                "id": "train-0",
                "episode": "train",
                "scene_id": "a",
                "time_index": 0,
            },
            {
                "id": "train-1",
                "episode": "train",
                "scene_id": "a",
                "time_index": 70,
            },
            {
                "id": "held-0",
                "episode": "held",
                "scene_id": "b",
                "time_index": 1,
            },
            {
                "id": "held-1",
                "episode": "held",
                "scene_id": "b",
                "time_index": 130,
            },
        ]
        stock = [{**sample, "video": f"{sample['id']}.mp4"} for sample in samples]
        excluded = [{"id": "train-0", "episode": "train"}]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            samples_path = tmp_path / "samples.json"
            stock_path = tmp_path / "stock.json"
            excluded_path = tmp_path / "excluded.json"
            output_path = tmp_path / "heldout.json"
            write_json(samples_path, samples)
            write_json(stock_path, stock)
            write_json(excluded_path, excluded)

            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parents[1] / "tools" / "build_task4_subset.py"),
                    "--samples-json",
                    str(samples_path),
                    "--stock-json",
                    str(stock_path),
                    "--exclude-manifest",
                    str(excluded_path),
                    "--all-eligible",
                    "--output",
                    str(output_path),
                ],
                check=True,
            )

            heldout = json.loads(output_path.read_text())
            self.assertEqual(
                {item["id"] for item in heldout},
                {"held-0", "held-1"},
            )
            self.assertEqual({item["episode"] for item in heldout}, {"held"})


if __name__ == "__main__":
    unittest.main()
