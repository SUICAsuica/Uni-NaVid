from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from uninavid.constants import NAVIGATION_IDENTIFIER
from uninavid.model.history_compressor import LearnedHistoryCompressor
from uninavid.model.uninavid_arch import UniNaVIDMetaForCausalLM


class HistoryCompressorTest(unittest.TestCase):
    def build_compressor(self):
        return LearnedHistoryCompressor(
            vision_dim=32,
            hidden_dim=16,
            num_queries=8,
            num_heads=4,
            num_layers=2,
            ffn_dim=32,
            dropout=0.0,
        )

    def test_output_shape_for_variable_and_empty_history(self):
        compressor = self.build_compressor()
        for frame_count in (0, 1, 65):
            history = torch.randn(2, frame_count, 64, 32)
            output = compressor(history)
            self.assertEqual(output.shape, (2, 8, 32))

    def test_gradients_reach_compressor(self):
        compressor = self.build_compressor()
        output = compressor(torch.randn(1, 3, 64, 32))
        output.square().mean().backward()

        self.assertIsNotNone(compressor.query_tokens.grad)
        self.assertIsNotNone(compressor.input_projection.weight.grad)
        self.assertIsNotNone(compressor.output_projection.weight.grad)
        self.assertIsNotNone(compressor.spatial_embedding.grad)
        self.assertIsNotNone(compressor.temporal_embedding.weight.grad)

    def test_output_changes_when_frame_order_changes(self):
        compressor = self.build_compressor().eval()
        history = torch.randn(1, 4, 64, 32)

        original = compressor(history)
        reversed_history = compressor(history.flip(1))

        self.assertFalse(torch.allclose(original, reversed_history, atol=1e-6))

    def test_output_changes_when_spatial_positions_change(self):
        compressor = self.build_compressor().eval()
        history = torch.randn(1, 1, 64, 32)

        original = compressor(history)
        spatially_reversed = compressor(history.flip(2))

        self.assertFalse(torch.allclose(original, spatially_reversed, atol=1e-6))

    def test_rejects_non_8x8_frame_tokens(self):
        compressor = self.build_compressor()
        with self.assertRaisesRegex(ValueError, "expected 64 tokens per frame"):
            compressor(torch.randn(1, 2, 16, 32))

    def test_goal_conditioning_changes_output_and_receives_gradients(self):
        compressor = LearnedHistoryCompressor(
            vision_dim=32,
            hidden_dim=16,
            num_queries=8,
            num_heads=4,
            num_layers=1,
            ffn_dim=32,
            dropout=0.0,
            goal_dim=24,
        ).eval()
        history = torch.randn(1, 3, 64, 32)
        goal_mask = torch.ones(1, 4, dtype=torch.long)
        chair_goal = torch.randn(1, 4, 24)
        toilet_goal = torch.randn(1, 4, 24)

        chair_output = compressor(history, chair_goal, goal_mask)
        toilet_output = compressor(history, toilet_goal, goal_mask)

        self.assertFalse(torch.allclose(chair_output, toilet_output, atol=1e-6))
        chair_output.square().mean().backward()
        self.assertIsNotNone(compressor.goal_projection[1].weight.grad)

    def test_goal_conditioning_rejects_missing_goal(self):
        compressor = LearnedHistoryCompressor(
            vision_dim=32,
            hidden_dim=16,
            num_queries=8,
            num_heads=4,
            num_layers=1,
            ffn_dim=32,
            dropout=0.0,
            goal_dim=24,
        )
        with self.assertRaisesRegex(ValueError, "requires goal features"):
            compressor(torch.randn(1, 2, 64, 32))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_projector = nn.Linear(32, 48)
        self.history_compressor = LearnedHistoryCompressor(
            vision_dim=32,
            hidden_dim=16,
            num_queries=8,
            num_heads=4,
            num_layers=1,
            ffn_dim=32,
            dropout=0.0,
        )


class ArchitectureHarness(UniNaVIDMetaForCausalLM):
    def __init__(self):
        self.model = DummyModel()
        self.config = SimpleNamespace(
            compress_type="grid:2",
            history_compressor_type="cross_attention",
            mm_vision_select_feature="patch",
            run_type="train",
        )

    def get_model(self):
        return self.model


class HistoryCompressorIntegrationTest(unittest.TestCase):
    def test_navigation_path_keeps_current_frame_separate(self):
        harness = ArchitectureHarness()
        prompt = [[f"{NAVIGATION_IDENTIFIER}. Search for a chair."]]
        features = torch.randn(3, 256, 32)

        history, video_flags, current, token_lengths = harness.vlm_attention(
            features,
            prompts=prompt,
            image_counts=[3],
        )

        self.assertEqual(history[0].shape, (1, 8, 48))
        self.assertEqual(current[0].shape, (1, 64, 48))
        self.assertEqual(token_lengths, [[8]])
        self.assertEqual(video_flags, [True])


if __name__ == "__main__":
    unittest.main()
