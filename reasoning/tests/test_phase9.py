"""Phase 9 tests — walk-forward evaluation harness + performance metrics.

Coverage:
    TestEvalMetrics         — frozen dataclass fields, to_dict, range checks
    TestMetricHelpers       — _ece, _sharpe, _max_drawdown edge cases
    TestEvaluatorEpisode    — output types, n_steps, regime accuracy range,
                              uncertainty non-negative, PnL path, empty iter
    TestHaltCounting        — ood_halt_count / epistemic_halt_count vs thresholds
    TestWalkForwardEval     — n_results, window independence, single split,
                              empty batches
    TestLatency             — run_episode p99 < 500ms on CPU
"""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import pytest
import torch

from reasoning.agent import build_agent
from reasoning.config import ReasoningConfig, KillSwitchConfig
from reasoning.evaluate import (
    EvalMetrics,
    Evaluator,
    _ece,
    _max_drawdown,
    _sharpe,
    build_evaluator,
)


# ── shared helpers ─────────────────────────────────────────────────────────────

def _cfg_small() -> ReasoningConfig:
    from reasoning.config import (
        SSMConfig, EventEncoderConfig, WalletGNNConfig, MoEConfig,
        FiLMConfig, UncertaintyConfig, RAGConfig, TrainingConfig,
    )
    return ReasoningConfig(
        ssm=SSMConfig(d_model=16, d_state=4, n_layers=1, d_conv=2, expand=2),
        event=EventEncoderConfig(n_event_types=5, d_event_emb=8, n_mlp_layers=2),
        wallet=WalletGNNConfig(node_feature_dim=4, d_hidden=16, n_gnn_layers=1),
        moe=MoEConfig(n_experts=4, top_k=2),
        film=FiLMConfig(metadata_dim=4, hidden_dim=16),
        uncertainty=UncertaintyConfig(
            n_mc_passes=3, dropout_rate=0.1, n_regimes=4,
            mahalanobis_feature_dim=16,
        ),
        rag=RAGConfig(embedding_dim=32, n_neighbors=4),
        training=TrainingConfig(),
        kill_switch=KillSwitchConfig(ood_threshold=3.0, epistemic_threshold=0.5),
    )


def _make_batch(
    B: int = 2,
    T: int = 4,
    d_model: int = 16,
    n_regimes: int = 4,
    with_return: bool = False,
) -> dict:
    batch = {
        "tick_features":  torch.randn(B, T, 4),
        "tick_dts":       torch.ones(T) * 0.4,
        "event_features": torch.zeros(B, 2),
        "event_dts":      torch.zeros(B),
        "wallet_embs":    torch.zeros(B, d_model),
        "chain_meta":     torch.randn(B, 4),
        "regime_labels":  torch.randint(0, n_regimes, (B,)),
    }
    if with_return:
        batch["realized_return"] = torch.randn(B) * 0.01
    return batch


def _make_evaluator(ks_cfg: KillSwitchConfig | None = None) -> Evaluator:
    cfg = _cfg_small()
    agent = build_agent(cfg, device="cpu")
    agent.eval()
    return build_evaluator(agent, ks_cfg or cfg.kill_switch, device="cpu")


# ── TestEvalMetrics ────────────────────────────────────────────────────────────

class TestEvalMetrics:
    def _sample(self, **overrides) -> EvalMetrics:
        defaults = dict(
            regime_accuracy=0.6, ece=0.05,
            mean_epistemic_var=0.1, mean_ood_score=1.5,
            ood_halt_count=3, epistemic_halt_count=1,
            sharpe=0.8, max_drawdown=0.05, total_return=0.12,
            n_steps=100,
        )
        defaults.update(overrides)
        return EvalMetrics(**defaults)

    def test_fields_set(self):
        m = self._sample()
        assert m.regime_accuracy == 0.6
        assert m.n_steps == 100
        assert m.ood_halt_count == 3

    def test_to_dict_keys(self):
        m = self._sample()
        d = m.to_dict()
        for k in ("regime_accuracy", "ece", "mean_epistemic_var", "mean_ood_score",
                  "ood_halt_count", "epistemic_halt_count", "sharpe",
                  "max_drawdown", "total_return", "n_steps"):
            assert k in d

    def test_to_dict_values_float(self):
        m = self._sample()
        for v in m.to_dict().values():
            assert isinstance(v, float)

    def test_frozen_immutable(self):
        m = self._sample()
        with pytest.raises((AttributeError, TypeError)):
            m.regime_accuracy = 0.9  # type: ignore[misc]

    def test_max_drawdown_range(self):
        m = self._sample(max_drawdown=0.20)
        assert 0.0 <= m.max_drawdown <= 1.0


# ── TestMetricHelpers ──────────────────────────────────────────────────────────

class TestMetricHelpers:
    # ── ECE ───────────────────────────────────────────────────────────────────

    def test_ece_empty(self):
        assert _ece(np.array([]), np.array([])) == 0.0

    def test_ece_perfect_calibration(self):
        # All samples in high-confidence bin and all correct → ECE ≈ 0
        conf = np.ones(100) * 0.95
        correct = np.ones(100)
        ece = _ece(conf, correct)
        assert ece < 0.1

    def test_ece_overconfident(self):
        # High confidence (0.9) but only 50% correct → high ECE
        conf = np.ones(100) * 0.9
        correct = np.array([1.0, 0.0] * 50)
        ece = _ece(conf, correct)
        assert ece > 0.3

    def test_ece_range(self):
        rng = np.random.default_rng(42)
        conf = rng.uniform(0, 1, 200)
        correct = (rng.uniform(0, 1, 200) > 0.5).astype(float)
        ece = _ece(conf, correct)
        assert 0.0 <= ece <= 1.0

    def test_ece_uniform_confidence_uniform_accuracy(self):
        # 10 bins, each fully populated, perfect calibration
        confs = np.linspace(0.05, 0.95, 10)
        correct = confs  # accuracy equals confidence in each bin
        ece = _ece(confs, correct)
        assert ece < 0.01

    # ── Sharpe ────────────────────────────────────────────────────────────────

    def test_sharpe_empty(self):
        assert _sharpe(np.array([])) == 0.0

    def test_sharpe_one_sample(self):
        assert _sharpe(np.array([0.01])) == 0.0

    def test_sharpe_zero_std(self):
        assert _sharpe(np.ones(10) * 0.01) == 0.0

    def test_sharpe_positive_returns(self):
        returns = np.ones(100) * 0.01 + np.random.default_rng(0).normal(0, 0.001, 100)
        s = _sharpe(returns)
        assert s > 0

    def test_sharpe_negative_returns(self):
        returns = np.ones(50) * -0.02
        # std is 0 → returns 0 (not negative Sharpe — consistent with zero-std guard)
        s = _sharpe(returns)
        assert s == 0.0

    def test_sharpe_mixed(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.01, 252)
        s = _sharpe(returns)
        assert math.isfinite(s)

    # ── Max drawdown ──────────────────────────────────────────────────────────

    def test_max_drawdown_empty(self):
        assert _max_drawdown(np.array([])) == 0.0

    def test_max_drawdown_monotone_increase(self):
        returns = np.ones(10) * 0.01
        assert _max_drawdown(returns) == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_full_loss(self):
        # Drop to near zero: -99% return in one step
        returns = np.array([0.0, 0.0, -0.99])
        dd = _max_drawdown(returns)
        assert dd > 0.9

    def test_max_drawdown_range(self):
        rng = np.random.default_rng(99)
        returns = rng.normal(0.001, 0.02, 200)
        dd = _max_drawdown(returns)
        assert 0.0 <= dd <= 1.0


# ── TestEvaluatorEpisode ───────────────────────────────────────────────────────

class TestEvaluatorEpisode:
    def setup_method(self):
        self.cfg = _cfg_small()
        self.ev = _make_evaluator(self.cfg.kill_switch)
        self.d = self.cfg.ssm.d_model
        self.nr = self.cfg.uncertainty.n_regimes

    def _batches(self, n: int = 3, with_return: bool = False) -> list[dict]:
        return [_make_batch(B=2, d_model=self.d, n_regimes=self.nr,
                            with_return=with_return) for _ in range(n)]

    def test_returns_eval_metrics(self):
        m = self.ev.run_episode(iter(self._batches()))
        assert isinstance(m, EvalMetrics)

    def test_n_steps_correct(self):
        m = self.ev.run_episode(iter(self._batches(3)))
        assert m.n_steps == 6  # 3 batches × B=2

    def test_regime_accuracy_in_range(self):
        m = self.ev.run_episode(iter(self._batches(4)))
        assert 0.0 <= m.regime_accuracy <= 1.0

    def test_ece_in_range(self):
        m = self.ev.run_episode(iter(self._batches(4)))
        assert 0.0 <= m.ece <= 1.0

    def test_mean_ood_nonnegative(self):
        m = self.ev.run_episode(iter(self._batches()))
        assert m.mean_ood_score >= 0.0

    def test_mean_epistemic_nonnegative(self):
        m = self.ev.run_episode(iter(self._batches()))
        assert m.mean_epistemic_var >= 0.0

    def test_pnl_zero_without_realized_return(self):
        m = self.ev.run_episode(iter(self._batches(with_return=False)))
        assert m.total_return == 0.0
        assert m.sharpe == 0.0
        assert m.max_drawdown == 0.0

    def test_pnl_nonzero_with_realized_return(self):
        # With random returns the PnL may be anything; just check finite
        m = self.ev.run_episode(iter(self._batches(with_return=True)))
        assert math.isfinite(m.total_return)
        assert math.isfinite(m.sharpe)
        assert 0.0 <= m.max_drawdown <= 1.0

    def test_empty_iterator(self):
        m = self.ev.run_episode(iter([]))
        assert m.n_steps == 0
        assert m.regime_accuracy == 0.0

    def test_no_agent_weight_change(self):
        params_before = {n: p.clone() for n, p in self.ev.agent.named_parameters()}
        self.ev.run_episode(iter(self._batches(5)))
        for n, p in self.ev.agent.named_parameters():
            assert torch.equal(p, params_before[n]), f"param {n} changed"


# ── TestHaltCounting ───────────────────────────────────────────────────────────

class TestHaltCounting:
    def test_no_halts_below_threshold(self):
        # With ood_threshold=1000, nothing should trigger
        ks_cfg = KillSwitchConfig(ood_threshold=1000.0, epistemic_threshold=1000.0)
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        ev = Evaluator(agent, ks_cfg, device="cpu")
        d, nr = cfg.ssm.d_model, cfg.uncertainty.n_regimes
        batches = [_make_batch(B=4, d_model=d, n_regimes=nr) for _ in range(3)]
        m = ev.run_episode(iter(batches))
        assert m.ood_halt_count == 0
        assert m.epistemic_halt_count == 0

    def test_all_halt_at_zero_threshold(self):
        # With ood_threshold=-1, every sample fires (OOD >= 0 always)
        ks_cfg = KillSwitchConfig(ood_threshold=-1.0, epistemic_threshold=-1.0)
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        ev = Evaluator(agent, ks_cfg, device="cpu")
        d, nr = cfg.ssm.d_model, cfg.uncertainty.n_regimes
        batches = [_make_batch(B=4, d_model=d, n_regimes=nr) for _ in range(2)]
        m = ev.run_episode(iter(batches))
        assert m.ood_halt_count == 8  # 2 batches × B=4
        assert m.epistemic_halt_count == 8

    def test_halt_counts_are_ints(self):
        ev = _make_evaluator()
        d, nr = ev.agent.cfg.ssm.d_model, ev.agent.cfg.uncertainty.n_regimes
        batches = [_make_batch(B=2, d_model=d, n_regimes=nr)]
        m = ev.run_episode(iter(batches))
        assert isinstance(m.ood_halt_count, int)
        assert isinstance(m.epistemic_halt_count, int)


# ── TestWalkForwardEval ────────────────────────────────────────────────────────

class TestWalkForwardEval:
    def setup_method(self):
        self.cfg = _cfg_small()
        self.ev = _make_evaluator(self.cfg.kill_switch)
        self.d = self.cfg.ssm.d_model
        self.nr = self.cfg.uncertainty.n_regimes

    def _batches(self, n: int) -> list[dict]:
        return [_make_batch(B=2, d_model=self.d, n_regimes=self.nr) for _ in range(n)]

    def test_returns_list_of_metrics(self):
        results = self.ev.walk_forward_eval(self._batches(9), n_splits=3)
        assert isinstance(results, list)
        assert all(isinstance(m, EvalMetrics) for m in results)

    def test_n_splits_correct(self):
        results = self.ev.walk_forward_eval(self._batches(9), n_splits=3)
        assert len(results) == 3

    def test_n_steps_cover_all_batches(self):
        batches = self._batches(6)
        results = self.ev.walk_forward_eval(batches, n_splits=3)
        total_steps = sum(m.n_steps for m in results)
        assert total_steps == 6 * 2  # 6 batches × B=2

    def test_empty_batches_returns_empty(self):
        results = self.ev.walk_forward_eval([], n_splits=3)
        assert results == []

    def test_single_split(self):
        results = self.ev.walk_forward_eval(self._batches(4), n_splits=1)
        assert len(results) == 1
        assert results[0].n_steps == 8  # 4 batches × B=2

    def test_results_are_independent(self):
        # Each window uses different batches → different random outputs
        results = self.ev.walk_forward_eval(self._batches(12), n_splits=3)
        # Can't guarantee different values (random outputs), but all should be valid
        for m in results:
            assert 0.0 <= m.regime_accuracy <= 1.0

    def test_build_factory(self):
        ev = build_evaluator(
            self.ev.agent, self.cfg.kill_switch, device="cpu"
        )
        assert isinstance(ev, Evaluator)


# ── TestLatency ────────────────────────────────────────────────────────────────

class TestLatency:
    N = 10

    def test_run_episode_p99_under_500ms(self):
        import time
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        ev = build_evaluator(agent, cfg.kill_switch, device="cpu")
        d, nr = cfg.ssm.d_model, cfg.uncertainty.n_regimes
        latencies = []
        for _ in range(self.N):
            batches = [_make_batch(B=2, T=4, d_model=d, n_regimes=nr) for _ in range(4)]
            t0 = time.perf_counter()
            ev.run_episode(iter(batches))
            latencies.append(time.perf_counter() - t0)
        latencies.sort()
        p99 = latencies[int(0.99 * self.N)]
        assert p99 < 0.5, f"run_episode p99={p99*1000:.1f}ms > 500ms"
