"""Phase 4: Mixture-of-Experts (MoE) router + experts.

Architecture:
    x (batch, d_model)
    → Router: Linear(d_model, n_experts, bias=False)
    → [training] add Gaussian noise ~ N(0, noise_std) before softmax
    → softmax → route_gates (batch, n_experts)
    → top-k selection → (topk_indices, topk_gates), renormalised
    → dispatch each token to its k experts; weighted-sum → moe_out (batch, d_model)
    → output = RMSNorm(x + moe_out)   ← residual keeps gradients flowing at init

Auxiliary losses (returned every forward pass; caller decides weight schedule):
    balance_loss — Switch-style load-balancing:
        f_i = fraction of tokens routed to expert i  (hard, from top-k)
        P_i = mean soft gate probability for expert i (differentiable, clean logits)
        loss = n_experts * Σ_i (f_i * P_i)   ×  balance_loss_coeff
    z_loss — router z-loss (penalises large logits, prevents collapse):
        loss = mean(logsumexp(clean_logits)²)  ×  z_loss_coeff

Routing design notes:
    • Noise is added to logits (not gates) before softmax, following the
      original Shazeer et al. noisy top-k recipe.
    • Balance loss uses *clean* gates for P_i so it is differentiable
      and not contaminated by training noise.
    • Dispatch uses a per-token loop over the batch, which is exact and
      autograd-safe for all batch sizes. Critical path is batch=1.
    • Expert d_ff = 4 × d_model (standard transformer expansion ratio).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoEConfig
from .ssm import RMSNorm


# ── Expert FFN ────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """Single expert: 2-layer FFN with GELU activation."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        d_ff = 4 * d_model
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(F.gelu(self.ff1(x)))


# ── MoE module ────────────────────────────────────────────────────────────────

class MoE(nn.Module):
    """Noisy top-k Mixture-of-Experts layer satisfying MoEProto."""

    def __init__(self, cfg: MoEConfig, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.noise_std = cfg.noise_std
        self.balance_loss_coeff = cfg.balance_loss_coeff
        self.z_loss_coeff = cfg.z_loss_coeff

        # No bias: prevents trivial routing collapse to a constant offset.
        self.router = nn.Linear(d_model, cfg.n_experts, bias=False)

        self.experts = nn.ModuleList([Expert(d_model) for _ in range(cfg.n_experts)])
        self.norm = RMSNorm(d_model)

    # ── MoEProto interface ────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Route x through top-k experts.

        Args:
            x: (batch, d_model)
        Returns:
            output:     (batch, d_model)  — RMSNorm(x + weighted expert outputs)
            aux_losses: dict with keys "balance_loss" and "z_loss" (both scalar)
        """
        batch = x.shape[0]

        # ── Router ────────────────────────────────────────────────────────────
        clean_logits = self.router(x)   # (batch, n_experts); used for losses

        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(clean_logits) * self.noise_std
            route_logits = clean_logits + noise
        else:
            route_logits = clean_logits

        # Soft gates for loss (clean) and routing (potentially noisy)
        clean_gates = F.softmax(clean_logits, dim=-1)   # (batch, n_experts)
        route_gates = F.softmax(route_logits, dim=-1)   # (batch, n_experts)

        # Top-k from noisy gates; renormalise so selected gates sum to 1
        topk_vals, topk_idx = route_gates.topk(self.top_k, dim=-1)  # (batch, k)
        topk_gates = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # ── Expert dispatch (per-token; autograd-safe for all batch sizes) ────
        token_outputs: list[torch.Tensor] = []
        for b in range(batch):
            token_out = torch.zeros(1, self.d_model, device=x.device, dtype=x.dtype)
            for k in range(self.top_k):
                ei = int(topk_idx[b, k].item())
                gate = topk_gates[b, k]             # scalar — participates in grad
                expert_out = self.experts[ei](x[b : b + 1])  # (1, d_model)
                token_out = token_out + gate * expert_out
            token_outputs.append(token_out)
        moe_out = torch.cat(token_outputs, dim=0)   # (batch, d_model)

        # Residual + normalisation
        output = self.norm(x + moe_out)

        # ── Auxiliary losses ──────────────────────────────────────────────────
        # f_i: fraction of token-assignments routed to expert i (non-differentiable)
        # Each token distributes 1/top_k assignment weight across its k experts.
        topk_onehot = torch.zeros(
            batch, self.n_experts, device=x.device, dtype=x.dtype
        )
        topk_onehot.scatter_(1, topk_idx, 1.0 / self.top_k)
        f_i = topk_onehot.mean(dim=0)  # (n_experts,)

        # P_i: mean clean gate probability per expert (differentiable via clean_gates)
        P_i = clean_gates.mean(dim=0)  # (n_experts,)

        balance_loss = (
            self.n_experts * (f_i * P_i).sum() * self.balance_loss_coeff
        )
        z_loss = (
            torch.logsumexp(clean_logits, dim=-1).pow(2).mean() * self.z_loss_coeff
        )

        return output, {"balance_loss": balance_loss, "z_loss": z_loss}


# ── Factory ───────────────────────────────────────────────────────────────────

def build_moe(
    cfg: MoEConfig,
    d_model: int = 128,
    device: str = "cpu",
) -> MoE:
    return MoE(cfg, d_model).to(device)
