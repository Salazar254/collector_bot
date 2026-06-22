"""Phase 5: Output heads — regime classifier, position sizing, survival, uncertainty.

All six outputs come from a shared input x (batch, d_model):

    regime_logits  (batch, n_regimes)  — raw logits for regime classification
    size_mu        (batch, 1)          — log-normal position-size mean (unconstrained)
    size_sigma     (batch, 1)          — log-normal std; softplus + ε → strictly > 0
    hazard         (batch, 1)          — survival hazard rate; sigmoid → (0, 1)
    epistemic_var  (batch, 1)          — MC-dropout variance of regime_logits
    ood_score      (batch, 1)          — Mahalanobis distance to nearest class mean

Epistemic uncertainty (MC dropout):
    In eval mode with n_mc_passes > 1, dropout is temporarily enabled and the
    heads are run n_mc_passes times.  Output values are the mean across passes;
    epistemic_var is the mean per-regime variance of regime_logits.
    In training mode epistemic_var is returned as zeros (MC is expensive to run
    every training step and the variance estimate is unreliable with one batch).

OOD detection (Mahalanobis):
    class_means (n_regimes, d_model) and precision_diag (d_model,) are registered
    buffers (not learned parameters).  Call update_class_means() after each
    training epoch to keep them current.  At init they are zeros / ones so the
    score is just the squared Euclidean distance to the origin.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import UncertaintyConfig

_SIGMA_EPS = 1e-5   # floor on size_sigma to prevent division-by-zero in NLL


class OutputHeads(nn.Module):
    """All output heads + uncertainty.  Satisfies OutputHeadsProto."""

    def __init__(self, cfg: UncertaintyConfig, d_model: int = 128) -> None:
        super().__init__()
        self.n_regimes = cfg.n_regimes
        self.n_mc_passes = cfg.n_mc_passes
        self.d_model = d_model

        self.dropout = nn.Dropout(p=cfg.dropout_rate)

        # Prediction heads — all linear from the shared embedding
        self.regime_head     = nn.Linear(d_model, cfg.n_regimes)
        self.size_mu_head    = nn.Linear(d_model, 1)
        self.size_sigma_head = nn.Linear(d_model, 1)   # softplus-constrained in forward
        self.hazard_head     = nn.Linear(d_model, 1)   # sigmoid-constrained in forward

        # OOD buffers — updated post-training; non-trainable
        self.register_buffer(
            "class_means",
            torch.zeros(cfg.n_regimes, cfg.mahalanobis_feature_dim),
        )
        self.register_buffer(
            "precision_diag",
            torch.ones(cfg.mahalanobis_feature_dim),
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _heads_pass(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Single forward pass through all prediction heads (dropout applied)."""
        h = self.dropout(x)
        return {
            "regime_logits": self.regime_head(h),
            "size_mu":       self.size_mu_head(h),
            "size_sigma":    F.softplus(self.size_sigma_head(h)) + _SIGMA_EPS,
            "hazard":        torch.sigmoid(self.hazard_head(h)),
        }

    def _mahalanobis_ood(self, x: torch.Tensor) -> torch.Tensor:
        """Minimum diagonal Mahalanobis distance to any class mean.

        Returns (batch, 1).  High score → far from all training-distribution classes.
        """
        # diff: (batch, n_regimes, d_model)
        diff = x.unsqueeze(1) - self.class_means.unsqueeze(0)
        # Weighted squared distance per regime: (batch, n_regimes)
        dist = (diff.pow(2) * self.precision_diag).sum(dim=-1)
        return dist.min(dim=-1, keepdim=True)[0]   # (batch, 1)

    # ── OutputHeadsProto interface ────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute all six outputs.

        epistemic_var is computed via MC dropout only in eval mode with
        n_mc_passes > 1; otherwise it is returned as zeros.
        """
        batch = x.shape[0]

        if (not self.training) and (self.n_mc_passes > 1):
            # MC dropout: enable dropout temporarily for epistemic estimation
            self.dropout.train()
            mc = [self._heads_pass(x) for _ in range(self.n_mc_passes)]
            self.dropout.eval()

            regime_stack = torch.stack([r["regime_logits"] for r in mc])  # (K, batch, n_reg)
            epistemic_var = regime_stack.var(dim=0).mean(dim=-1, keepdim=True)  # (batch, 1)

            result: dict[str, torch.Tensor] = {
                "regime_logits": regime_stack.mean(dim=0),
                "size_mu":       torch.stack([r["size_mu"]    for r in mc]).mean(dim=0),
                "size_sigma":    torch.stack([r["size_sigma"] for r in mc]).mean(dim=0),
                "hazard":        torch.stack([r["hazard"]     for r in mc]).mean(dim=0),
                "epistemic_var": epistemic_var,
            }
        else:
            result = self._heads_pass(x)
            result["epistemic_var"] = torch.zeros(
                batch, 1, device=x.device, dtype=x.dtype
            )

        result["ood_score"] = self._mahalanobis_ood(x)
        return result

    # ── utility ───────────────────────────────────────────────────────────────

    def update_class_means(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        momentum: float = 0.9,
    ) -> None:
        """EMA update of class-conditional means from a labelled batch.

        Args:
            features: (N, d_model)
            labels:   (N,) int64 — regime indices in [0, n_regimes)
            momentum: EMA decay (0 = full replacement, 1 = no update)
        """
        with torch.no_grad():
            for c in range(self.n_regimes):
                mask = labels == c
                if mask.any():
                    class_feat = features[mask].mean(dim=0)
                    self.class_means[c] = (
                        momentum * self.class_means[c]
                        + (1.0 - momentum) * class_feat
                    )


# ── Factory ───────────────────────────────────────────────────────────────────

def build_output_heads(
    cfg: UncertaintyConfig,
    d_model: int = 128,
    device: str = "cpu",
) -> OutputHeads:
    return OutputHeads(cfg, d_model).to(device)
