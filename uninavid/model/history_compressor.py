import torch
import torch.nn as nn


class LearnedHistoryCompressor(nn.Module):
    """Compress variable-length 8x8 frame tokens into fixed learned queries."""

    def __init__(
        self,
        vision_dim=1408,
        hidden_dim=512,
        num_queries=64,
        num_heads=8,
        num_layers=2,
        ffn_dim=2048,
        dropout=0.1,
        tokens_per_frame=64,
        max_history_frames=512,
        goal_dim=None,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if max_history_frames < 1:
            raise ValueError("max_history_frames must be positive")

        self.num_queries = num_queries
        self.tokens_per_frame = tokens_per_frame
        self.vision_dim = vision_dim
        self.max_history_frames = max_history_frames
        self.goal_dim = goal_dim

        self.input_norm = nn.LayerNorm(vision_dim)
        self.input_projection = nn.Linear(vision_dim, hidden_dim)
        self.spatial_embedding = nn.Parameter(
            torch.empty(tokens_per_frame, hidden_dim)
        )
        self.temporal_embedding = nn.Embedding(max_history_frames, hidden_dim)
        if goal_dim is not None:
            self.goal_projection = nn.Sequential(
                nn.LayerNorm(goal_dim),
                nn.Linear(goal_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.goal_projection = None
        self.query_tokens = nn.Parameter(torch.empty(num_queries, hidden_dim))
        self.null_history_token = nn.Parameter(torch.empty(1, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vision_dim)

        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.null_history_token, std=0.02)
        nn.init.normal_(self.spatial_embedding, std=0.02)
        nn.init.normal_(self.temporal_embedding.weight, std=0.02)

    def forward(self, history, goal_features=None, goal_attention_mask=None):
        """
        Args:
            history: EVA features shaped [batch, frames, 64, vision_dim].
            goal_features: Frozen LLM embeddings shaped [batch, tokens, goal_dim].
            goal_attention_mask: Valid goal tokens shaped [batch, tokens].

        Returns:
            Fixed history tokens shaped [batch, num_queries, vision_dim].
        """
        if history.ndim != 4:
            raise ValueError(
                "history must have shape [batch, frames, tokens, vision_dim]"
            )

        batch_size, frame_count, token_count, vision_dim = history.shape
        if token_count != self.tokens_per_frame:
            raise ValueError(
                f"expected {self.tokens_per_frame} tokens per frame, got {token_count}"
            )
        if vision_dim != self.vision_dim:
            raise ValueError(f"expected vision_dim={self.vision_dim}, got {vision_dim}")

        if frame_count == 0:
            memory = self.null_history_token.view(1, 1, -1).expand(
                batch_size, -1, -1
            )
        else:
            memory = self.input_projection(self.input_norm(history))
            relative_age = torch.arange(
                frame_count - 1,
                -1,
                -1,
                device=history.device,
            ).clamp(max=self.max_history_frames - 1)
            temporal_embedding = self.temporal_embedding(relative_age)
            memory = (
                memory
                + self.spatial_embedding.view(1, 1, token_count, -1)
                + temporal_embedding.view(1, frame_count, 1, -1)
            )
            memory = memory.flatten(1, 2)

        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        if self.goal_projection is not None:
            if goal_features is None or goal_attention_mask is None:
                raise ValueError("goal-conditioned compression requires goal features")
            if goal_features.shape[0] != batch_size:
                raise ValueError("goal batch size must match history batch size")
            mask = goal_attention_mask.to(goal_features.dtype).unsqueeze(-1)
            pooled_goal = (goal_features * mask).sum(dim=1)
            pooled_goal = pooled_goal / mask.sum(dim=1).clamp_min(1)
            queries = queries + self.goal_projection(pooled_goal).unsqueeze(1)
        compressed = self.decoder(tgt=queries, memory=memory)
        return self.output_projection(self.output_norm(compressed))


def build_history_compressor(config):
    compressor_type = getattr(config, "history_compressor_type", "heuristic")
    if compressor_type != "cross_attention":
        raise ValueError(f"Unsupported history compressor: {compressor_type}")

    goal_conditioned = getattr(config, "history_goal_conditioned", False)
    return LearnedHistoryCompressor(
        vision_dim=config.mm_hidden_size,
        hidden_dim=getattr(config, "history_hidden_size", 512),
        num_queries=getattr(config, "history_num_queries", 64),
        num_heads=getattr(config, "history_num_heads", 8),
        num_layers=getattr(config, "history_num_layers", 2),
        ffn_dim=getattr(config, "history_ffn_dim", 2048),
        dropout=getattr(config, "history_dropout", 0.1),
        max_history_frames=getattr(config, "history_max_frames", 512),
        goal_dim=config.hidden_size if goal_conditioned else None,
    )
