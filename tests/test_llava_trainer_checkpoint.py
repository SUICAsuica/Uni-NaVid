from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn
from transformers import PretrainedConfig, TrainingArguments

from uninavid.train.llava_trainer import LLaVATrainer


class CheckpointModel(nn.Module):
    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.history_compressor = nn.Linear(3, 4)
        self.mm_projector = nn.Linear(4, 5)
        self.frozen = nn.Linear(5, 6)


class AdapterCheckpointTest(unittest.TestCase):
    def test_checkpoint_round_trip_only_saves_trainable_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(output_dir=tmp, report_to=[])
            args.tune_mm_mlp_adapter = True
            args.tune_history_compressor = True
            args.use_im_start_end = False

            source = CheckpointModel()
            trainer = LLaVATrainer(model=source, args=args)
            checkpoint = Path(tmp) / "checkpoint-test"
            trainer._save(str(checkpoint))

            state = torch.load(
                checkpoint / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(state)
            self.assertTrue(
                all(
                    "history_compressor" in key or "mm_projector" in key
                    for key in state
                )
            )
            self.assertFalse(any("frozen" in key for key in state))

            target = CheckpointModel()
            with torch.no_grad():
                for parameter in target.parameters():
                    parameter.zero_()
            target_trainer = LLaVATrainer(model=target, args=args)
            target_trainer._load_from_checkpoint(str(checkpoint))

            self.assertTrue(
                torch.equal(
                    target.history_compressor.weight,
                    source.history_compressor.weight,
                )
            )
            self.assertTrue(
                torch.equal(target.mm_projector.weight, source.mm_projector.weight)
            )
            self.assertEqual(torch.count_nonzero(target.frozen.weight), 0)


if __name__ == "__main__":
    unittest.main()
