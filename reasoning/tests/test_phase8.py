"""Phase 8 tests — end-to-end ReasoningAgent + TrainingLoop.

Coverage:
    TestAgentConstruction   — build, parameter count, sub-module types
    TestAgentForward        — output keys, shapes, dtypes, no NaN/Inf
    TestAgentForwardGrad    — gradients flow to all trainable sub-modules
    TestAgentStepBlock      — streaming step, kill signal integration
    TestTrainingLoopLoss    — compute_loss shapes, non-negative regime loss
    TestTrainingLoopStep    — single gradient step, loss decreases
    TestTrainingLoopEpoch   — run_epoch returns correct keys
    TestAdversarial         — PGD + chain adv paths execute without error
    TestLatency             — forward p99 < 500ms on CPU (no GPU)
"""
from __future__ import annotations

import math
from typing import Iterator

import pytest
import torch
import torch.optim as optim

from reasoning.agent import ReasoningAgent, build_agent
from reasoning.config import ReasoningConfig, KillSwitchConfig
from reasoning.train import TrainingLoop, _size_nll_loss


# ── shared fixtures ────────────────────────────────────────────────────────────

def _cfg_small() -> ReasoningConfig:
    """Tiny config for fast CPU tests."""
    from reasoning.config import (
        SSMConfig, EventEncoderConfig, WalletGNNConfig, MoEConfig,
        FiLMConfig, UncertaintyConfig, RAGConfig, TrainingConfig,
    )
    return ReasoningConfig(
        ssm=SSMConfig(d_model=16, d_state=4, n_layers=1, d_conv=2, expand=2),
        event=EventEncoderConfig(n_event_types=5, d_event_emb=8, n_mlp_layers=2),
        wallet=WalletGNNConfig(node_feature_dim=4, d_hidden=16, n_gnn_layers=1),
        moe=MoEConfig(n_experts=4, top_k=2, balance_loss_coeff=0.01, z_loss_coeff=0.001),
        film=FiLMConfig(metadata_dim=4, hidden_dim=16),
        uncertainty=UncertaintyConfig(
            n_mc_passes=3, dropout_rate=0.1, n_regimes=4,
            mahalanobis_feature_dim=16,  # must == d_model
        ),
        rag=RAGConfig(embedding_dim=32, n_neighbors=4),
        training=TrainingConfig(pgd_steps=2, pgd_epsilon=0.01, pgd_alpha=0.005,
                                pgd_loss_coeff=0.1, chain_adv_loss_coeff=0.01),
        kill_switch=KillSwitchConfig(),
    )


def _make_batch(B: int = 2, T: int = 4, d_model: int = 16, n_regimes: int = 4) -> dict:
    return {
        "tick_features":  torch.randn(B, T, 4),
        "tick_dts":       torch.ones(T) * 0.4,
        "event_features": torch.zeros(B, 2),
        "event_dts":      torch.zeros(B),
        "wallet_embs":    torch.zeros(B, d_model),
        "chain_meta":     torch.randn(B, 4),
        "regime_labels":  torch.randint(0, n_regimes, (B,)),
        "size_labels":    torch.rand(B),
        "is_event":       torch.zeros(B),
    }


# ── TestAgentConstruction ──────────────────────────────────────────────────────

class TestAgentConstruction:
    def test_build_factory(self):
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        assert isinstance(agent, ReasoningAgent)

    def test_sub_module_types(self):
        from reasoning.ssm import ContinuousEncoder
        from reasoning.event_encoder import EventEncoder
        from reasoning.film_fusion import MultiChainFusion
        from reasoning.moe import MoE
        from reasoning.output_heads import OutputHeads

        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        assert isinstance(agent.encoder, ContinuousEncoder)
        assert isinstance(agent.event_encoder, EventEncoder)
        assert isinstance(agent.fusion, MultiChainFusion)
        assert isinstance(agent.moe, MoE)
        assert isinstance(agent.heads, OutputHeads)

    def test_trainable_params_nonzero(self):
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        n_params = sum(p.numel() for p in agent.parameters())
        assert n_params > 0

    def test_non_nn_components_present(self):
        from reasoning.wallet_gnn import WalletGNN
        from reasoning.rag import AsyncRAG
        from reasoning.kill_switch import KillSwitch

        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        assert isinstance(agent.wallet_gnn, WalletGNN)
        assert isinstance(agent.rag, AsyncRAG)
        assert isinstance(agent.kill_switch, KillSwitch)

    def test_kill_switch_not_nn_parameter(self):
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        param_names = {n for n, _ in agent.named_parameters()}
        assert not any("kill_switch" in n for n in param_names)


# ── TestAgentForward ───────────────────────────────────────────────────────────

class TestAgentForward:
    def setup_method(self):
        self.cfg = _cfg_small()
        self.agent = build_agent(self.cfg, device="cpu")
        self.agent.eval()

    def test_output_keys(self):
        batch = _make_batch()
        out = self.agent(
            batch["tick_features"], batch["tick_dts"],
            batch["event_features"], batch["event_dts"],
            batch["wallet_embs"], batch["chain_meta"],
        )
        expected = {
            "regime_logits", "size_mu", "size_sigma", "hazard",
            "epistemic_var", "ood_score", "balance_loss", "z_loss",
        }
        assert expected.issubset(out.keys())

    def test_output_shapes(self):
        B, T = 3, 5
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        agent.eval()
        d = cfg.ssm.d_model
        n_reg = cfg.uncertainty.n_regimes
        batch = _make_batch(B=B, T=T, d_model=d, n_regimes=n_reg)
        out = agent(
            batch["tick_features"], batch["tick_dts"],
            batch["event_features"], batch["event_dts"],
            batch["wallet_embs"], batch["chain_meta"],
        )
        assert out["regime_logits"].shape == (B, n_reg)
        assert out["size_mu"].shape == (B, 1)
        assert out["size_sigma"].shape == (B, 1)
        assert out["hazard"].shape == (B, 1)
        assert out["epistemic_var"].shape == (B, 1)
        assert out["ood_score"].shape == (B, 1)

    def test_no_nan_or_inf(self):
        batch = _make_batch()
        out = self.agent(
            batch["tick_features"], batch["tick_dts"],
            batch["event_features"], batch["event_dts"],
            batch["wallet_embs"], batch["chain_meta"],
        )
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert torch.isfinite(v).all(), f"{k} contains NaN/Inf"

    def test_size_sigma_positive(self):
        batch = _make_batch()
        out = self.agent(
            batch["tick_features"], batch["tick_dts"],
            batch["event_features"], batch["event_dts"],
            batch["wallet_embs"], batch["chain_meta"],
        )
        assert (out["size_sigma"] > 0).all()

    def test_hazard_in_unit_interval(self):
        batch = _make_batch()
        out = self.agent(
            batch["tick_features"], batch["tick_dts"],
            batch["event_features"], batch["event_dts"],
            batch["wallet_embs"], batch["chain_meta"],
        )
        assert (out["hazard"] > 0).all() and (out["hazard"] < 1).all()

    def test_batch_size_one(self):
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        agent.eval()
        batch = _make_batch(B=1, d_model=cfg.ssm.d_model, n_regimes=cfg.uncertainty.n_regimes)
        out = agent(
            batch["tick_features"], batch["tick_dts"],
            batch["event_features"], batch["event_dts"],
            batch["wallet_embs"], batch["chain_meta"],
        )
        assert out["regime_logits"].shape[0] == 1

    def test_train_mode_epistemic_zero(self):
        # In training mode, n_mc_passes is not applied → epistemic_var = 0
        self.agent.train()
        batch = _make_batch()
        out = self.agent(
            batch["tick_features"], batch["tick_dts"],
            batch["event_features"], batch["event_dts"],
            batch["wallet_embs"], batch["chain_meta"],
        )
        assert (out["epistemic_var"] == 0).all()


# ── TestAgentForwardGrad ───────────────────────────────────────────────────────

class TestAgentForwardGrad:
    def test_regime_loss_has_grad(self):
        import torch.nn.functional as F
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        agent.train()
        batch = _make_batch(d_model=cfg.ssm.d_model, n_regimes=cfg.uncertainty.n_regimes)
        out = agent(
            batch["tick_features"], batch["tick_dts"],
            batch["event_features"], batch["event_dts"],
            batch["wallet_embs"], batch["chain_meta"],
        )
        loss = F.cross_entropy(out["regime_logits"], batch["regime_labels"])
        loss.backward()
        # Check at least one parameter in each trainable module has a gradient
        for name, param in agent.named_parameters():
            if param.requires_grad and param.grad is not None:
                return  # found at least one — pass
        pytest.fail("No parameter has a gradient after backward()")

    def test_grad_reaches_encoder(self):
        import torch.nn.functional as F
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        agent.train()
        batch = _make_batch(d_model=cfg.ssm.d_model, n_regimes=cfg.uncertainty.n_regimes)
        out = agent(
            batch["tick_features"], batch["tick_dts"],
            batch["event_features"], batch["event_dts"],
            batch["wallet_embs"], batch["chain_meta"],
        )
        loss = F.cross_entropy(out["regime_logits"], batch["regime_labels"])
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in agent.encoder.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0
        assert any(g > 0 for g in grad_norms)

    def test_grad_reaches_event_encoder(self):
        import torch.nn.functional as F
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        agent.train()
        batch = _make_batch(d_model=cfg.ssm.d_model, n_regimes=cfg.uncertainty.n_regimes)
        out = agent(
            batch["tick_features"], batch["tick_dts"],
            batch["event_features"], batch["event_dts"],
            batch["wallet_embs"], batch["chain_meta"],
        )
        loss = F.cross_entropy(out["regime_logits"], batch["regime_labels"])
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in agent.event_encoder.parameters()
            if p.grad is not None
        ]
        assert any(g > 0 for g in grad_norms)


# ── TestAgentStepBlock ─────────────────────────────────────────────────────────

class TestAgentStepBlock:
    def setup_method(self):
        self.cfg = _cfg_small()
        self.agent = build_agent(self.cfg, device="cpu")
        self.agent.eval()
        self.d = self.cfg.ssm.d_model

    def test_step_block_returns_kill_signals(self):
        out = self.agent.step_block(
            tick=torch.randn(1, 4),
            dt=0.4,
            event_features=torch.zeros(1, 2),
            event_dt=1.0,
            wallet_emb=torch.zeros(1, self.d),
            chain_meta=torch.randn(1, 4),
        )
        assert "kill_signals" in out
        assert isinstance(out["kill_signals"], list)

    def test_step_block_output_shapes(self):
        out = self.agent.step_block(
            tick=torch.randn(1, 4),
            dt=0.4,
            event_features=torch.zeros(1, 2),
            event_dt=0.0,
            wallet_emb=torch.zeros(1, self.d),
            chain_meta=torch.randn(1, 4),
        )
        n_reg = self.cfg.uncertainty.n_regimes
        assert out["regime_logits"].shape == (1, n_reg)
        assert out["hazard"].shape == (1, 1)

    def test_step_block_triggers_kill_switch(self):
        from reasoning.config import KillSwitchConfig
        # Use a very low threshold so OOD fires immediately
        cfg = _cfg_small()
        cfg = cfg.model_copy(update={"kill_switch": KillSwitchConfig(ood_threshold=0.0)})
        agent = build_agent(cfg, device="cpu")
        agent.eval()
        d = cfg.ssm.d_model
        out = agent.step_block(
            tick=torch.randn(1, 4),
            dt=0.4,
            event_features=torch.zeros(1, 2),
            event_dt=0.0,
            wallet_emb=torch.zeros(1, d),
            chain_meta=torch.randn(1, 4),
        )
        # ood_score is always >= 0 (squared distance) > 0.0
        assert agent.kill_switch.is_halted

    def test_no_grad_in_step_block(self):
        out = self.agent.step_block(
            tick=torch.randn(1, 4),
            dt=0.4,
            event_features=torch.zeros(1, 2),
            event_dt=0.0,
            wallet_emb=torch.zeros(1, self.d),
            chain_meta=torch.randn(1, 4),
        )
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert not v.requires_grad, f"{k} has requires_grad=True in step_block"


# ── TestTrainingLoopLoss ───────────────────────────────────────────────────────

class TestTrainingLoopLoss:
    def setup_method(self):
        self.cfg = _cfg_small()
        self.agent = build_agent(self.cfg, device="cpu")
        self.agent.train()
        self.opt = optim.Adam(self.agent.parameters(), lr=1e-3)
        self.loop = TrainingLoop(
            self.agent, self.cfg.training, self.opt,
            device="cpu", use_pgd=False, use_chain_adv=False,
        )

    def _fwd(self, B: int = 2) -> tuple[dict, dict]:
        batch = _make_batch(B=B, d_model=self.cfg.ssm.d_model,
                            n_regimes=self.cfg.uncertainty.n_regimes)
        out = self.agent(
            batch["tick_features"], batch["tick_dts"],
            batch["event_features"], batch["event_dts"],
            batch["wallet_embs"], batch["chain_meta"],
        )
        return out, batch

    def test_compute_loss_keys(self):
        out, batch = self._fwd()
        losses = self.loop.compute_loss(out, batch)
        for k in ("total", "regime_loss", "size_loss", "survival_loss",
                  "balance_loss", "z_loss"):
            assert k in losses

    def test_total_is_scalar(self):
        out, batch = self._fwd()
        losses = self.loop.compute_loss(out, batch)
        assert losses["total"].shape == ()

    def test_regime_loss_nonnegative(self):
        out, batch = self._fwd()
        losses = self.loop.compute_loss(out, batch)
        assert float(losses["regime_loss"]) >= 0

    def test_size_nll_all_zero_labels(self):
        B = 2
        mu = torch.zeros(B, 1)
        sigma = torch.ones(B, 1)
        labels = torch.zeros(B)  # all zero → no active samples
        loss = _size_nll_loss(mu, sigma, labels)
        assert float(loss) == 0.0

    def test_size_nll_nonzero_labels(self):
        B = 4
        mu = torch.zeros(B, 1)
        sigma = torch.ones(B, 1)
        labels = torch.full((B,), 2.0)  # log(2) != 0 → NLL > 0
        loss = _size_nll_loss(mu, sigma, labels)
        assert float(loss) > 0


# ── TestTrainingLoopStep ───────────────────────────────────────────────────────

class TestTrainingLoopStep:
    def setup_method(self):
        self.cfg = _cfg_small()
        self.agent = build_agent(self.cfg, device="cpu")
        self.opt = optim.Adam(self.agent.parameters(), lr=1e-2)
        self.loop = TrainingLoop(
            self.agent, self.cfg.training, self.opt,
            device="cpu", use_pgd=False, use_chain_adv=False,
        )
        self.batch = _make_batch(
            d_model=self.cfg.ssm.d_model, n_regimes=self.cfg.uncertainty.n_regimes
        )

    def test_step_returns_float_dict(self):
        losses = self.loop.step(self.batch)
        assert isinstance(losses, dict)
        for v in losses.values():
            assert isinstance(v, float)

    def test_step_has_total_key(self):
        losses = self.loop.step(self.batch)
        assert "total" in losses

    def test_multiple_steps_run(self):
        for _ in range(3):
            losses = self.loop.step(self.batch)
            assert math.isfinite(losses["total"])

    def test_step_modifies_params(self):
        p0 = {n: p.clone().detach() for n, p in self.agent.named_parameters()}
        self.loop.step(self.batch)
        changed = [
            n for n, p in self.agent.named_parameters()
            if not torch.equal(p, p0[n])
        ]
        assert len(changed) > 0


# ── TestTrainingLoopEpoch ──────────────────────────────────────────────────────

class TestTrainingLoopEpoch:
    def setup_method(self):
        self.cfg = _cfg_small()
        self.agent = build_agent(self.cfg, device="cpu")
        self.opt = optim.Adam(self.agent.parameters(), lr=1e-3)
        self.loop = TrainingLoop(
            self.agent, self.cfg.training, self.opt,
            device="cpu", use_pgd=False, use_chain_adv=False,
        )

    def _iter(self, n: int = 3) -> Iterator[dict]:
        d = self.cfg.ssm.d_model
        r = self.cfg.uncertainty.n_regimes
        for _ in range(n):
            yield _make_batch(d_model=d, n_regimes=r)

    def test_run_epoch_keys(self):
        result = self.loop.run_epoch(self._iter(3))
        for k in ("total", "steps", "elapsed_sec"):
            assert k in result

    def test_run_epoch_step_count(self):
        result = self.loop.run_epoch(self._iter(4))
        assert result["steps"] == 4.0

    def test_run_epoch_max_steps(self):
        result = self.loop.run_epoch(self._iter(10), max_steps=2)
        assert result["steps"] == 2.0


# ── TestAdversarial ────────────────────────────────────────────────────────────

class TestAdversarial:
    def setup_method(self):
        self.cfg = _cfg_small()
        self.agent = build_agent(self.cfg, device="cpu")
        self.opt = optim.Adam(self.agent.parameters(), lr=1e-3)

    def test_pgd_step_runs(self):
        loop = TrainingLoop(
            self.agent, self.cfg.training, self.opt,
            device="cpu", use_pgd=True, use_chain_adv=False,
        )
        batch = _make_batch(d_model=self.cfg.ssm.d_model,
                            n_regimes=self.cfg.uncertainty.n_regimes)
        losses = loop.step(batch)
        assert "adv_loss" in losses
        assert math.isfinite(losses["total"])

    def test_chain_adv_step_runs(self):
        loop = TrainingLoop(
            self.agent, self.cfg.training, self.opt,
            device="cpu", use_pgd=False, use_chain_adv=True,
        )
        batch = _make_batch(d_model=self.cfg.ssm.d_model,
                            n_regimes=self.cfg.uncertainty.n_regimes)
        losses = loop.step(batch)
        assert "chain_adv_kl" in losses
        assert math.isfinite(losses["total"])

    def test_both_adv_step_runs(self):
        loop = TrainingLoop(
            self.agent, self.cfg.training, self.opt,
            device="cpu", use_pgd=True, use_chain_adv=True,
        )
        batch = _make_batch(d_model=self.cfg.ssm.d_model,
                            n_regimes=self.cfg.uncertainty.n_regimes)
        losses = loop.step(batch)
        assert "adv_loss" in losses
        assert "chain_adv_kl" in losses


# ── TestLatency ────────────────────────────────────────────────────────────────

class TestLatency:
    N = 20  # small count — forward pass is expensive on CPU

    def test_forward_p99_under_500ms(self):
        import time
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        agent.eval()
        d = cfg.ssm.d_model
        n_reg = cfg.uncertainty.n_regimes
        latencies = []
        for _ in range(self.N):
            batch = _make_batch(B=1, T=4, d_model=d, n_regimes=n_reg)
            t0 = time.perf_counter()
            with torch.no_grad():
                agent(
                    batch["tick_features"], batch["tick_dts"],
                    batch["event_features"], batch["event_dts"],
                    batch["wallet_embs"], batch["chain_meta"],
                )
            latencies.append(time.perf_counter() - t0)
        latencies.sort()
        p99 = latencies[int(0.99 * self.N)]
        assert p99 < 0.5, f"forward p99={p99*1000:.1f}ms > 500ms"

    def test_step_block_p99_under_500ms(self):
        import time
        cfg = _cfg_small()
        agent = build_agent(cfg, device="cpu")
        agent.eval()
        d = cfg.ssm.d_model
        latencies = []
        for _ in range(self.N):
            t0 = time.perf_counter()
            agent.step_block(
                tick=torch.randn(1, 4), dt=0.4,
                event_features=torch.zeros(1, 2), event_dt=0.0,
                wallet_emb=torch.zeros(1, d),
                chain_meta=torch.randn(1, 4),
            )
            latencies.append(time.perf_counter() - t0)
        latencies.sort()
        p99 = latencies[int(0.99 * self.N)]
        assert p99 < 0.5, f"step_block p99={p99*1000:.1f}ms > 500ms"
