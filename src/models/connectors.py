"""Connector zoo for the B2 ablation: alternatives to the Q-Former bridge.

The Q-Former compresses 256+ patch tokens into 32 learned queries — a lossy
bottleneck that may discard the spatial detail the `relate` category needs. The
literature's cheaper connectors preserve more: an MLP projector (LLaVA-style)
hands the LLM every patch token; an average-pooling compressor keeps the token
budget but replaces learned attention with fixed pooling. All connectors share
the QFormerBridge contract: features [B, P, encoder_dim] -> tokens [B, Q, llm_dim],
and accept drop_cls (memory token 0 sliced first) so the [CLS] control composes.

`build_bridge` is the single factory every script uses; a checkpoint's
`bridge_config["type"]` reconstructs the exact trained connector at eval time.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.bridge import QFormerBridge


class _PatchMLP(nn.Module):
    """Per-patch MLP: encoder_dim -> hidden x(depth-1) -> llm_dim, GELU between."""

    def __init__(self, encoder_dim: int, llm_dim: int, hidden: int, depth: int):
        """Build depth linear layers (depth >= 2) applied independently per patch."""
        super().__init__()
        dims = [encoder_dim] + [hidden] * (depth - 1) + [llm_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, P, encoder_dim] -> [B, P, llm_dim]."""
        return self.net(x)


class MLPBridge(nn.Module):
    """LLaVA-style projector: every patch token goes to the LLM (no compression)."""

    def __init__(self, encoder_dim: int, llm_dim: int, hidden: int = 4096,
                 depth: int = 3, drop_cls: bool = False):
        """Per-patch MLP; the LLM sees all P visual tokens (256/257, not 32)."""
        super().__init__()
        self.drop_cls = drop_cls
        self.proj = _PatchMLP(encoder_dim, llm_dim, hidden, depth)

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """[B, P, encoder_dim] -> [B, P(-1 if drop_cls), llm_dim]."""
        if self.drop_cls:
            encoder_features = encoder_features[:, 1:, :]
        return self.proj(encoder_features)


class PoolBridge(nn.Module):
    """Average-pooling compressor: fixed pooling to Q tokens, then the same MLP."""

    def __init__(self, encoder_dim: int, llm_dim: int, num_query_tokens: int = 32,
                 hidden: int = 4096, depth: int = 3, drop_cls: bool = False):
        """Adaptive average-pool the patch axis to num_query_tokens, then project."""
        super().__init__()
        self.drop_cls = drop_cls
        self.num_query_tokens = num_query_tokens
        self.proj = _PatchMLP(encoder_dim, llm_dim, hidden, depth)

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """[B, P, encoder_dim] -> [B, num_query_tokens, llm_dim]."""
        if self.drop_cls:
            encoder_features = encoder_features[:, 1:, :]
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            encoder_features.transpose(1, 2), self.num_query_tokens).transpose(1, 2)
        return self.proj(pooled)


def build_bridge(bridge_config: dict, encoder_dim: int, llm_dim: int) -> nn.Module:
    """Construct the connector a bridge_config describes (checkpoint-reproducible).

    type "qformer" (the default, and every checkpoint that predates this factory)
    reconstructs QFormerBridge with exactly the arguments the scripts passed before
    the factory existed, so old checkpoints load bit-identically.
    """
    btype = bridge_config.get("type", "qformer")
    drop_cls = bridge_config.get("drop_cls", False)
    if btype == "qformer":
        return QFormerBridge(
            encoder_dim=encoder_dim, llm_dim=llm_dim,
            num_query_tokens=bridge_config["num_query_tokens"],
            hidden_size=bridge_config["hidden_size"],
            num_layers=bridge_config["num_hidden_layers"],
            num_heads=bridge_config["num_attention_heads"],
            drop_cls=drop_cls,
            dropout=bridge_config.get("dropout", 0.1),
            norm_first=bridge_config.get("norm_first", False),
        )
    if btype == "mlp":
        return MLPBridge(encoder_dim=encoder_dim, llm_dim=llm_dim,
                         hidden=bridge_config.get("mlp_hidden", 4096),
                         depth=bridge_config.get("mlp_depth", 3), drop_cls=drop_cls)
    if btype == "pool":
        return PoolBridge(encoder_dim=encoder_dim, llm_dim=llm_dim,
                          num_query_tokens=bridge_config.get("num_query_tokens", 32),
                          hidden=bridge_config.get("mlp_hidden", 4096),
                          depth=bridge_config.get("mlp_depth", 3), drop_cls=drop_cls)
    raise ValueError(f"unknown bridge type {btype!r} (expected qformer | mlp | pool)")
