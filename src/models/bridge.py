"""The bridge: the only trained component, connecting a frozen encoder to a frozen LLM.

A Q-Former-style connector. A small set of learnable "query" tokens attend (via
cross-attention) to the encoder's patch features and summarise them into a fixed
number of vectors, independent of how many patches the encoder produced. Those
vectors are projected to the language model's embedding width so they can be fed to
the LLM as if they were word embeddings ("visual tokens").

Design notes:
- Fixed-length output (num_query_tokens) decouples the LLM interface from the
  encoder, so swapping encoders for RQ2 only changes `encoder_dim`.
- This is the sole trainable module; the encoder and LLM are frozen.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QFormerBridge(nn.Module):
    """Map encoder patch features [B, P, encoder_dim] to visual tokens [B, Q, llm_dim]."""

    def __init__(self, encoder_dim: int, llm_dim: int, num_query_tokens: int = 32,
                 hidden_size: int = 768, num_layers: int = 6, num_heads: int = 12,
                 drop_cls: bool = False, dropout: float = 0.1,
                 norm_first: bool = False):
        """Build the projection-in, cross-attention stack, and projection-out."""
        super().__init__()
        # RQ2 [CLS] ablation: drop memory token 0 before cross-attention. Only valid
        # when the encoder puts a global-summary token first (CLIP does; I-JEPA has
        # no such token, so its features must never be sliced). Adds no parameters,
        # so the state_dict is identical either way — the flag must therefore travel
        # in bridge_config, or a checkpoint would silently load with the wrong memory.
        self.drop_cls = drop_cls
        # Learnable query tokens — the "questions" the bridge asks of the image.
        self.queries = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size) * 0.02)
        # Project encoder features into the bridge's working width.
        self.encoder_proj = nn.Linear(encoder_dim, hidden_size)
        self.encoder_norm = nn.LayerNorm(hidden_size)
        # Transformer decoder layers: queries self-attend, then cross-attend to the
        # image features, then a feed-forward — repeated num_layers times.
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            batch_first=True, norm_first=norm_first)
        final_norm = nn.LayerNorm(hidden_size) if norm_first else None
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=num_layers, norm=final_norm,
        )
        # Project the summarised queries to the LLM's embedding dimension.
        self.llm_proj = nn.Linear(hidden_size, llm_dim)

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """encoder_features [B, P, encoder_dim] -> visual tokens [B, Q, llm_dim]."""
        if self.drop_cls:
            encoder_features = encoder_features[:, 1:, :]
        memory = self.encoder_norm(self.encoder_proj(encoder_features))  # [B, P, H]
        queries = self.queries.expand(encoder_features.size(0), -1, -1)  # [B, Q, H]
        summarised = self.decoder(tgt=queries, memory=memory)           # [B, Q, H]
        return self.llm_proj(summarised)                                # [B, Q, llm_dim]
