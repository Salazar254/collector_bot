"""Phase 9: Walk-forward evaluation harness + performance metrics.

EvalMetrics — immutable result dataclass covering four categories:

    Classification: regime_accuracy, ece (expected calibration error)
    Uncertainty:    mean_epistemic_var, mean_ood_score
    Safety:         ood_halt_count, epistemic_halt_count   (threshold exceedances)
    Trading PnL:    sharpe, max_drawdown, total_return     (from realized_return)

Evaluator — stateless evaluation runner:

    run_episode(data_iter) → EvalMetrics
        Runs agent.forward() in eval mode on each batch.  Accumulates regime
        preds/labels/confidences for classification + calibration metrics;
        epistemic / OOD tensors for uncertainty metrics; realized_return (if
        present in batch) for PnL metrics.  Never modifies agent weights.

    walk_forward_eval(batches, n_splits=3) → list[EvalMetrics]
        Splits a time-ordered list of batches into n_splits non-overlapping
        eval windows and returns one EvalMetrics per window.  Call order is
        chronological so index 0 is the earliest window.

Batch format (same as train.py + optional PnL field):
    tick_features   (B, T, 4)  float32
    tick_dts        (T,)       float32
    event_features  (B, 2)     float32
    event_dts       (B,)       float32
    wallet_embs     (B, d)     float32
    chain_meta      (B, 4)     float32
    regime_labels   (B,)       int64
    realized_return (B,)       float32   OPTIONAL — per-sample period return
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from .agent import ReasoningAgent
from .config import KillSwitchConfig


# ── Metric helpers ────────────────────────────────────────────────────────────

def _ece(
    confidences: np.ndarray,  # (N,) max-softmax probability per sample
    correct: np.ndarray,      # (N,) 1.0 if top-1 correct, 0.0 otherwise
    n_bins: int = 10,
) -> float:
    """Expected calibration error (equal-width bins over [0,1])."""
    if len(confidences) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        cnt = mask.sum()
        if cnt == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = correct[mask].mean()
        ece += cnt / n * abs(bin_conf - bin_acc)
    return float(ece)


def _sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe ratio (annualization = sqrt(N)).

    Returns 0 if fewer than 2 samples or std ≈ 0.
    """
    if len(returns) < 2:
        return 0.0
    std = float(returns.std())
    if std < 1e-10:
        return 0.0
    return float(returns.mean() / std * math.sqrt(len(returns)))


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from a sequence of per-step simple returns.

    Returns value in [0, 1].  0 if returns is empty or all non-negative.
    """
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (peak - cumulative) / peak.clip(min=1e-10)
    return float(drawdowns.max())


# ── EvalMetrics ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvalMetrics:
    """Immutable result of one evaluation episode."""

    # Classification
    regime_accuracy: float
    ece: float

    # Uncertainty
    mean_epistemic_var: float
    mean_ood_score: float

    # Safety (count of threshold exceedances)
    ood_halt_count: int
    epistemic_halt_count: int

    # Trading PnL (all 0 if batch has no realized_return)
    sharpe: float
    max_drawdown: float
    total_return: float

    # Bookkeeping
    n_steps: int

    def to_dict(self) -> dict[str, float]:
        return {
            "regime_accuracy":      self.regime_accuracy,
            "ece":                  self.ece,
            "mean_epistemic_var":   self.mean_epistemic_var,
            "mean_ood_score":       self.mean_ood_score,
            "ood_halt_count":       float(self.ood_halt_count),
            "epistemic_halt_count": float(self.epistemic_halt_count),
            "sharpe":               self.sharpe,
            "max_drawdown":         self.max_drawdown,
            "total_return":         self.total_return,
            "n_steps":              float(self.n_steps),
        }


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """Stateless evaluation runner for ReasoningAgent.

    Does NOT modify agent weights. Safe to call during training for val metrics.
    """

    def __init__(
        self,
        agent: ReasoningAgent,
        ks_cfg: KillSwitchConfig,
        device: str = "cpu",
    ) -> None:
        self.agent = agent
        self.ks_cfg = ks_cfg
        self.device = device

    # ── core episode ──────────────────────────────────────────────────────────

    def run_episode(self, data_iter: Iterator[dict]) -> EvalMetrics:
        """Evaluate on one iterator of batches.

        Returns EvalMetrics with all fields populated.
        """
        dev = self.device
        self.agent.eval()

        regime_preds:  list[int]   = []
        regime_labels: list[int]   = []
        confidences:   list[float] = []
        epistemic_vals: list[float] = []
        ood_vals:       list[float] = []
        step_returns:   list[float] = []

        with torch.no_grad():
            for batch in data_iter:
                out = self.agent(
                    batch["tick_features"].to(dev),
                    batch["tick_dts"].to(dev),
                    batch["event_features"].to(dev),
                    batch["event_dts"].to(dev),
                    batch["wallet_embs"].to(dev),
                    batch["chain_meta"].to(dev),
                )

                # ── classification metrics ─────────────────────────────────
                preds = out["regime_logits"].argmax(dim=-1).cpu()
                labels = batch["regime_labels"]
                regime_preds.extend(preds.tolist())
                regime_labels.extend(labels.tolist())

                probs = F.softmax(out["regime_logits"], dim=-1).cpu()
                confs = probs.max(dim=-1).values
                confidences.extend(confs.tolist())

                # ── uncertainty metrics ────────────────────────────────────
                epi = out["epistemic_var"].squeeze(-1).cpu()
                ood = out["ood_score"].squeeze(-1).cpu()
                epistemic_vals.extend(epi.tolist())
                ood_vals.extend(ood.tolist())

                # ── PnL simulation (optional) ──────────────────────────────
                if "realized_return" in batch:
                    pos = torch.sigmoid(out["size_mu"].squeeze(-1)).cpu()
                    r = batch["realized_return"].float() * pos
                    step_returns.extend(r.tolist())

        # ── aggregate ─────────────────────────────────────────────────────────
        n = len(regime_preds)
        if n == 0:
            return EvalMetrics(
                regime_accuracy=0.0, ece=0.0,
                mean_epistemic_var=0.0, mean_ood_score=0.0,
                ood_halt_count=0, epistemic_halt_count=0,
                sharpe=0.0, max_drawdown=0.0, total_return=0.0,
                n_steps=0,
            )

        preds_arr  = np.array(regime_preds, dtype=int)
        labels_arr = np.array(regime_labels, dtype=int)
        correct    = (preds_arr == labels_arr).astype(float)
        conf_arr   = np.array(confidences, dtype=float)
        epi_arr    = np.array(epistemic_vals, dtype=float)
        ood_arr    = np.array(ood_vals, dtype=float)

        # Threshold exceedances (independent per sample — not a halt state machine)
        ood_halt_count = int((ood_arr > self.ks_cfg.ood_threshold).sum())
        epi_halt_count = int((epi_arr > self.ks_cfg.epistemic_threshold).sum())

        ret_arr = np.array(step_returns, dtype=float) if step_returns else np.zeros(0)

        return EvalMetrics(
            regime_accuracy=float(correct.mean()),
            ece=_ece(conf_arr, correct),
            mean_epistemic_var=float(epi_arr.mean()),
            mean_ood_score=float(ood_arr.mean()),
            ood_halt_count=ood_halt_count,
            epistemic_halt_count=epi_halt_count,
            sharpe=_sharpe(ret_arr),
            max_drawdown=_max_drawdown(ret_arr),
            total_return=float(ret_arr.sum()),
            n_steps=n,
        )

    # ── walk-forward evaluation ───────────────────────────────────────────────

    def walk_forward_eval(
        self,
        batches: list[dict],
        n_splits: int = 3,
    ) -> list[EvalMetrics]:
        """Evaluate on n_splits non-overlapping temporal windows.

        batches must be in chronological order. Each window is an equal slice
        of the list. Returns one EvalMetrics per window (index 0 = earliest).
        """
        if not batches or n_splits < 1:
            return []

        n = len(batches)
        window_size = max(1, n // n_splits)
        results: list[EvalMetrics] = []

        for i in range(n_splits):
            start = i * window_size
            end   = start + window_size if i < n_splits - 1 else n
            window = batches[start:end]
            if window:
                results.append(self.run_episode(iter(window)))

        return results


# ── Factory ───────────────────────────────────────────────────────────────────

def build_evaluator(
    agent: ReasoningAgent,
    ks_cfg: KillSwitchConfig,
    device: str = "cpu",
) -> Evaluator:
    return Evaluator(agent, ks_cfg, device=device)
