"""Phase 8: Multi-task training loop for ReasoningAgent.

Loss components
───────────────
regime_loss     CrossEntropy(regime_logits, regime_labels)         — classification
size_loss       Log-normal NLL on non-zero size_labels             — position sizing
survival_loss   BCE(hazard, is_event.float())                      — survival
balance_loss    Switch-style MoE load-balance (from agent.forward) — routing
z_loss          Router z-loss              (from agent.forward)    — routing

Adversarial augmentation (optional, controlled by TrainingConfig)
──────────────────────────────────────────────────────────────────
PGD on tick_features:
    x_adv = x + ε-ball PGD(steps, alpha) maximising regime_loss + size_loss
    adv_loss = regime+size at x_adv
    total += pgd_loss_coeff * adv_loss

Chain adversarial (FiLM):
    Perturb chain_meta by ±δ (uniform), measure KL of regime_logits
    total += chain_adv_loss_coeff * kl_loss

TrainingBatch (plain dict) keys
────────────────────────────────
    tick_features   (B, T, 4)   float32
    tick_dts        (T,)        float32
    event_features  (B, 2)      float32
    event_dts       (B,)        float32
    wallet_embs     (B, d)      float32
    chain_meta      (B, 4)      float32
    regime_labels   (B,)        int64
    size_labels     (B,)        float32   — 0 if no trade this block
    is_event        (B,)        float32   — 1 if event occurred
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import ReasoningAgent
from .config import TrainingConfig

_LOG_NORMAL_EPS = 1e-8   # guard log(0)


# ── Loss helpers ──────────────────────────────────────────────────────────────

def _size_nll_loss(
    size_mu: torch.Tensor,     # (B, 1)
    size_sigma: torch.Tensor,  # (B, 1)
    size_labels: torch.Tensor, # (B,)
) -> torch.Tensor:
    """Log-normal NLL for non-zero position sizes (mean over active samples)."""
    labels = size_labels.unsqueeze(-1)          # (B, 1)
    mask = labels > 0                           # only compute where trade occurred
    if not mask.any():
        return torch.zeros(1, device=size_mu.device)
    log_y = torch.log(labels.clamp(min=_LOG_NORMAL_EPS))
    nll = 0.5 * ((log_y - size_mu) / size_sigma).pow(2) + torch.log(size_sigma)
    return nll[mask].mean()


def _pgd_perturb(
    agent: ReasoningAgent,
    batch: dict,
    cfg: TrainingConfig,
    device: str,
) -> torch.Tensor:
    """Return adversarially perturbed tick_features via PGD (L-inf ball).

    Only the regime and size losses are maximised during the attack.
    Gradients do NOT propagate back to model parameters.
    """
    x_orig = batch["tick_features"].detach().to(device)
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-cfg.pgd_epsilon, cfg.pgd_epsilon)

    for _ in range(cfg.pgd_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        out = agent(
            x_adv,
            batch["tick_dts"].to(device),
            batch["event_features"].to(device),
            batch["event_dts"].to(device),
            batch["wallet_embs"].to(device),
            batch["chain_meta"].to(device),
        )
        loss = F.cross_entropy(out["regime_logits"], batch["regime_labels"].to(device))
        loss = loss + _size_nll_loss(
            out["size_mu"], out["size_sigma"], batch["size_labels"].to(device)
        )
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + cfg.pgd_alpha * x_adv.grad.sign()
            delta = (x_adv - x_orig).clamp(-cfg.pgd_epsilon, cfg.pgd_epsilon)
            x_adv = (x_orig + delta).detach()

    return x_adv


# ── Training loop ─────────────────────────────────────────────────────────────

class TrainingLoop:
    """Single-epoch + multi-epoch training loop for ReasoningAgent.

    Usage::

        loop = TrainingLoop(agent, cfg.training, optimizer)
        metrics = loop.step(batch)          # single gradient step
        epoch_metrics = loop.run_epoch(data_iter)
    """

    def __init__(
        self,
        agent: ReasoningAgent,
        cfg: TrainingConfig,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        use_pgd: bool = True,
        use_chain_adv: bool = True,
    ) -> None:
        self.agent = agent
        self.cfg = cfg
        self.optimizer = optimizer
        self.device = device
        self.use_pgd = use_pgd
        self.use_chain_adv = use_chain_adv

    # ── loss ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        out: dict[str, torch.Tensor],
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss components from a forward-pass output dict.

        Returns a dict of named scalar tensors. 'total' is the one to backprop.
        """
        dev = self.device
        regime_labels = batch["regime_labels"].to(dev)
        size_labels   = batch["size_labels"].to(dev)
        is_event      = batch["is_event"].to(dev)

        regime_loss   = F.cross_entropy(out["regime_logits"], regime_labels)
        size_loss     = _size_nll_loss(out["size_mu"], out["size_sigma"], size_labels)
        survival_loss = F.binary_cross_entropy(
            out["hazard"].squeeze(-1).clamp(1e-6, 1 - 1e-6), is_event
        )
        balance_loss = out["balance_loss"].mean()
        z_loss       = out["z_loss"].mean()

        total = (
            regime_loss
            + size_loss
            + survival_loss
            + balance_loss
            + z_loss
        )
        return {
            "total":         total,
            "regime_loss":   regime_loss,
            "size_loss":     size_loss,
            "survival_loss": survival_loss,
            "balance_loss":  balance_loss,
            "z_loss":        z_loss,
        }

    # ── single gradient step ──────────────────────────────────────────────────

    def step(self, batch: dict) -> dict[str, float]:
        """One forward + backward + optimiser step.

        Returns a dict of float loss values (detached).
        """
        dev = self.device
        self.agent.train()
        self.optimizer.zero_grad()

        out = self.agent(
            batch["tick_features"].to(dev),
            batch["tick_dts"].to(dev),
            batch["event_features"].to(dev),
            batch["event_dts"].to(dev),
            batch["wallet_embs"].to(dev),
            batch["chain_meta"].to(dev),
        )
        losses = self.compute_loss(out, batch)
        total = losses["total"]

        # PGD adversarial augmentation
        if self.use_pgd and self.cfg.pgd_loss_coeff > 0:
            x_adv = _pgd_perturb(self.agent, batch, self.cfg, dev)
            out_adv = self.agent(
                x_adv,
                batch["tick_dts"].to(dev),
                batch["event_features"].to(dev),
                batch["event_dts"].to(dev),
                batch["wallet_embs"].to(dev),
                batch["chain_meta"].to(dev),
            )
            losses_adv = self.compute_loss(out_adv, batch)
            adv_total = losses_adv["total"]
            total = total + self.cfg.pgd_loss_coeff * adv_total
            losses["adv_loss"] = adv_total

        # Chain adversarial (FiLM consistency)
        if self.use_chain_adv and self.cfg.chain_adv_loss_coeff > 0:
            meta_orig = batch["chain_meta"].to(dev)
            meta_perturbed = meta_orig + torch.randn_like(meta_orig) * 0.1
            out_adv_chain = self.agent(
                batch["tick_features"].to(dev),
                batch["tick_dts"].to(dev),
                batch["event_features"].to(dev),
                batch["event_dts"].to(dev),
                batch["wallet_embs"].to(dev),
                meta_perturbed,
            )
            # KL divergence between original and perturbed regime distributions
            log_p = F.log_softmax(out["regime_logits"].detach(), dim=-1)
            log_q = F.log_softmax(out_adv_chain["regime_logits"], dim=-1)
            kl_loss = F.kl_div(log_q, log_p.exp(), reduction="batchmean")
            total = total + self.cfg.chain_adv_loss_coeff * kl_loss
            losses["chain_adv_kl"] = kl_loss

        total.backward()
        self.optimizer.step()

        return {k: float(v.detach().cpu()) for k, v in losses.items()}

    # ── epoch ─────────────────────────────────────────────────────────────────

    def run_epoch(
        self, data_iter: Iterator[dict], max_steps: Optional[int] = None
    ) -> dict[str, float]:
        """Run one full epoch. Returns mean of each loss component."""
        from collections import defaultdict
        sums: dict[str, float] = defaultdict(float)
        n = 0

        t0 = time.perf_counter()
        for i, batch in enumerate(data_iter):
            if max_steps is not None and i >= max_steps:
                break
            step_losses = self.step(batch)
            for k, v in step_losses.items():
                sums[k] += v
            n += 1

        elapsed = time.perf_counter() - t0
        result = {k: v / max(n, 1) for k, v in sums.items()}
        result["steps"] = float(n)
        result["elapsed_sec"] = elapsed
        return result
