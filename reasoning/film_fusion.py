"""Phase 3: FiLM-modulated multi-chain fusion.

Architecture:
    continuous:     (1, d_model)  — from ContinuousEncoder (Phase 1)
    event:          (1, d_model)  — from EventEncoder       (Phase 2)
    wallet:         (1, d_model)  — from WalletGNN          (Phase 2)
    chain_metadata: (1, 4)        — ChainConfig.as_metadata_vector()

Step 1 — Fusion MLP:
    concat([continuous, event, wallet]) → (1, 3*d_model)
    → Linear(3*d_model, d_model) + GELU → fused (1, d_model)

Step 2 — FiLM modulation (Feature-wise Linear Modulation):
    chain_metadata → Linear(4, hidden_dim) + ReLU → h (1, hidden_dim)
    γ = gamma_proj(h)   (1, d_model)   — initialized to 1
    β = beta_proj(h)    (1, d_model)   — initialized to 0
    modulated = γ ⊙ fused + β

Step 3 — RMSNorm on the modulated embedding → (1, d_model)

FiLM init (γ=1, β=0): at startup the layer is identity w.r.t. fused,
so training begins from a neutral modulation and learns per-chain adjustments.

The chain metadata vector encodes four properties that differ significantly
across Solana / Base / BSC / Ethereum: block time, MEV intensity,
gas mechanic, and finality depth.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FiLMConfig
from .ssm import RMSNorm


# ── FiLM layer ────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """Generates per-channel (γ, β) from chain metadata and applies them."""

    def __init__(self, metadata_dim: int, hidden_dim: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gamma_proj = nn.Linear(hidden_dim, d_model)
        self.beta_proj = nn.Linear(hidden_dim, d_model)

        # Identity init: γ=1, β=0 so FiLM starts as a pass-through.
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self, x: torch.Tensor, chain_meta: torch.Tensor
    ) -> torch.Tensor:
        """
        x:          (batch, d_model)
        chain_meta: (batch, metadata_dim)
        Returns:    (batch, d_model)
        """
        h = self.net(chain_meta)            # (batch, hidden_dim)
        gamma = self.gamma_proj(h)          # (batch, d_model)
        beta = self.beta_proj(h)            # (batch, d_model)
        return gamma * x + beta


# ── Multi-stream fusion ───────────────────────────────────────────────────────

class MultiChainFusion(nn.Module):
    """Fuse continuous + event + wallet embeddings with FiLM chain conditioning.

    Critical-path interface: fuse() satisfies FusionProto.
    """

    def __init__(self, cfg: FiLMConfig, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

        # Concat three d_model streams → single d_model vector
        self.fusion_proj = nn.Linear(3 * d_model, d_model, bias=True)

        # FiLM conditioned on chain metadata
        self.film = FiLMLayer(cfg.metadata_dim, cfg.hidden_dim, d_model)

        # Post-modulation normalisation
        self.norm = RMSNorm(d_model)

    # ── FusionProto interface ─────────────────────────────────────────────────

    def fuse(
        self,
        continuous: torch.Tensor,       # (1, d_model)
        event: torch.Tensor,            # (1, d_model)
        wallet: torch.Tensor,           # (1, d_model)
        chain_metadata: torch.Tensor,   # (1, metadata_dim)
    ) -> torch.Tensor:
        """Returns fused embedding (1, d_model)."""
        combined = torch.cat([continuous, event, wallet], dim=-1)   # (1, 3*d_model)
        fused = F.gelu(self.fusion_proj(combined))                  # (1, d_model)
        modulated = self.film(fused, chain_metadata)                # (1, d_model)
        return self.norm(modulated)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_fusion(
    cfg: FiLMConfig,
    d_model: int = 128,
    device: str = "cpu",
) -> MultiChainFusion:
    return MultiChainFusion(cfg, d_model).to(device)
