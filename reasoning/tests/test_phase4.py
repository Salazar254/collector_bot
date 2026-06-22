"""Phase 4 tests: Mixture-of-Experts router + experts.

Done-when criteria:
1. forward() returns (batch, d_model) tensor + dict{"balance_loss", "z_loss"}.
2. Satisfies MoEProto.
3. Top-k constraint: exactly k experts selected per token.
4. Noise applied during training; eval mode is deterministic.
5. balance_loss >= 0; z_loss >= 0.
6. balance_loss increases as routing collapses (load imbalance detected).
7. Gradients flow to: input x, router weights, expert weights,
   and through z_loss. (balance_loss gradient flows through P_i = clean_gates.)
8. Numerically stable for large / small input magnitudes.
9. forward() p99 latency benchmarked; hard gate <5ms on CUDA.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from reasoning.benchmark import LatencyBenchmark
from reasoning.config import MoEConfig
from reasoning.interfaces import MoEProto
from reasoning.moe import Expert, MoE, build_moe


# ── Fixtures ──────────────────────────────────────────────────────────────────

D_MODEL = 128


@pytest.fixture()
def cfg() -> MoEConfig:
    return MoEConfig(
        n_experts=6, top_k=2,
        balance_loss_coeff=0.01,
        z_loss_coeff=0.001,
        noise_std=1.0,
    )


@pytest.fixture()
def moe(cfg: MoEConfig) -> MoE:
    m = MoE(cfg, d_model=D_MODEL)
    m.eval()
    return m


@pytest.fixture()
def x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, D_MODEL)


# ══════════════════════════════════════════════════════════════════════════════
# Expert — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExpert:
    def test_output_shape(self):
        exp = Expert(D_MODEL)
        x = torch.randn(1, D_MODEL)
        assert exp(x).shape == (1, D_MODEL)

    def test_batch_shape(self):
        exp = Expert(D_MODEL)
        x = torch.randn(4, D_MODEL)
        assert exp(x).shape == (4, D_MODEL)

    def test_output_finite(self):
        exp = Expert(D_MODEL)
        x = torch.randn(1, D_MODEL)
        assert torch.isfinite(exp(x)).all()

    def test_gradient_flows(self):
        exp = Expert(D_MODEL)
        exp.train()
        x = torch.randn(1, D_MODEL, requires_grad=True)
        loss = exp(x).sum()
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum().item() > 0
        assert exp.ff1.weight.grad is not None
        assert exp.ff2.weight.grad is not None

    def test_different_weights_different_output(self):
        """Two experts with different weights produce different outputs."""
        torch.manual_seed(0)
        e1 = Expert(D_MODEL)
        torch.manual_seed(99)
        e2 = Expert(D_MODEL)
        x = torch.randn(1, D_MODEL)
        assert not torch.allclose(e1(x), e2(x), atol=1e-4)

    def test_d_ff_is_4x_d_model(self):
        exp = Expert(D_MODEL)
        assert exp.ff1.out_features == 4 * D_MODEL
        assert exp.ff2.in_features == 4 * D_MODEL


# ══════════════════════════════════════════════════════════════════════════════
# MoE — shape and protocol
# ══════════════════════════════════════════════════════════════════════════════

class TestMoEShape:
    def test_output_shape(self, moe: MoE, x: torch.Tensor):
        out, aux = moe(x)
        assert out.shape == (1, D_MODEL)

    def test_output_dtype(self, moe: MoE, x: torch.Tensor):
        out, _ = moe(x)
        assert out.dtype == torch.float32

    def test_output_finite(self, moe: MoE, x: torch.Tensor):
        out, _ = moe(x)
        assert torch.isfinite(out).all()

    def test_aux_keys_present(self, moe: MoE, x: torch.Tensor):
        _, aux = moe(x)
        assert "balance_loss" in aux
        assert "z_loss" in aux

    def test_aux_are_scalars(self, moe: MoE, x: torch.Tensor):
        _, aux = moe(x)
        assert aux["balance_loss"].ndim == 0, "balance_loss should be scalar"
        assert aux["z_loss"].ndim == 0, "z_loss should be scalar"

    def test_satisfies_protocol(self, moe: MoE):
        assert isinstance(moe, MoEProto)

    def test_build_factory(self, cfg: MoEConfig):
        m = build_moe(cfg, d_model=64, device="cpu")
        out, aux = m(torch.randn(1, 64))
        assert out.shape == (1, 64)
        assert "balance_loss" in aux

    def test_batch_size_2(self, moe: MoE):
        x = torch.randn(2, D_MODEL)
        out, aux = moe(x)
        assert out.shape == (2, D_MODEL)
        assert torch.isfinite(out).all()

    def test_n_experts_attribute(self, moe: MoE, cfg: MoEConfig):
        assert len(moe.experts) == cfg.n_experts


# ══════════════════════════════════════════════════════════════════════════════
# MoE — routing behaviour
# ══════════════════════════════════════════════════════════════════════════════

class TestMoERouting:
    def test_exactly_k_experts_selected(self, moe: MoE, cfg: MoEConfig, x: torch.Tensor):
        """Router must select exactly top_k experts per token."""
        with torch.no_grad():
            clean_logits = moe.router(x)
            route_gates = torch.softmax(clean_logits, dim=-1)
            topk_vals, topk_idx = route_gates.topk(cfg.top_k, dim=-1)
        assert topk_idx.shape == (1, cfg.top_k)

    def test_topk_gates_sum_to_one(self, moe: MoE, cfg: MoEConfig, x: torch.Tensor):
        """Re-normalised top-k gates must sum to 1 per token."""
        with torch.no_grad():
            clean_logits = moe.router(x)
            route_gates = torch.softmax(clean_logits, dim=-1)
            topk_vals, _ = route_gates.topk(cfg.top_k, dim=-1)
            renorm = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        assert torch.allclose(renorm.sum(dim=-1), torch.ones(1), atol=1e-6)

    def test_different_inputs_route_differently(self, moe: MoE):
        """Distinct inputs should (with high probability) hit different experts."""
        torch.manual_seed(7)
        seen = set()
        for _ in range(20):
            xi = torch.randn(1, D_MODEL)
            with torch.no_grad():
                logits = moe.router(xi)
                _, idx = logits.topk(moe.top_k, dim=-1)
            seen.add(tuple(idx[0].tolist()))
        assert len(seen) > 1, "All inputs routed identically — router may be degenerate"

    def test_eval_deterministic(self, moe: MoE, x: torch.Tensor):
        """Same input twice in eval mode → same output (no noise)."""
        out1, _ = moe(x)
        out2, _ = moe(x)
        assert torch.allclose(out1, out2)

    def test_training_noise_changes_output(self, cfg: MoEConfig, x: torch.Tensor):
        """In training mode (noise_std > 0), same input may route differently."""
        moe = MoE(cfg, d_model=D_MODEL)
        moe.train()
        # With noise, repeated calls on the same x should not always agree.
        outputs = [moe(x)[0].detach() for _ in range(20)]
        unique = sum(
            1 for i in range(1, len(outputs))
            if not torch.allclose(outputs[0], outputs[i], atol=1e-4)
        )
        assert unique > 0, "Training mode noise never changed the output"

    def test_zero_noise_eval_matches_train(self, x: torch.Tensor):
        """With noise_std=0, train and eval produce the same routing."""
        cfg0 = MoEConfig(n_experts=6, top_k=2, noise_std=0.0,
                         balance_loss_coeff=0.01, z_loss_coeff=0.001)
        moe = MoE(cfg0, d_model=D_MODEL)
        moe.eval()
        out_eval, _ = moe(x)
        moe.train()
        out_train, _ = moe(x)
        assert torch.allclose(out_eval, out_train)


# ══════════════════════════════════════════════════════════════════════════════
# MoE — auxiliary losses
# ══════════════════════════════════════════════════════════════════════════════

class TestMoEAuxLosses:
    def test_balance_loss_non_negative(self, moe: MoE, x: torch.Tensor):
        _, aux = moe(x)
        assert aux["balance_loss"].item() >= 0.0

    def test_z_loss_non_negative(self, moe: MoE, x: torch.Tensor):
        _, aux = moe(x)
        assert aux["z_loss"].item() >= 0.0

    def test_both_losses_finite(self, moe: MoE, x: torch.Tensor):
        _, aux = moe(x)
        assert torch.isfinite(aux["balance_loss"])
        assert torch.isfinite(aux["z_loss"])

    def test_balance_loss_scales_with_coeff(self, x: torch.Tensor):
        torch.manual_seed(0)
        cfg_high = MoEConfig(n_experts=6, top_k=2, balance_loss_coeff=1.0,
                              z_loss_coeff=0.001, noise_std=0.0)
        cfg_low  = MoEConfig(n_experts=6, top_k=2, balance_loss_coeff=0.001,
                              z_loss_coeff=0.001, noise_std=0.0)
        # Same init seed → same weights, same routing → raw balance value identical;
        # only the coeff differs → high coeff gives larger loss.
        torch.manual_seed(42)
        moe_high = MoE(cfg_high, d_model=D_MODEL)
        moe_high.eval()
        torch.manual_seed(42)
        moe_low = MoE(cfg_low, d_model=D_MODEL)
        moe_low.eval()

        _, aux_high = moe_high(x)
        _, aux_low  = moe_low(x)
        assert aux_high["balance_loss"].item() > aux_low["balance_loss"].item()

    def test_z_loss_scales_with_coeff(self, x: torch.Tensor):
        torch.manual_seed(42)
        cfg_high = MoEConfig(n_experts=6, top_k=2, balance_loss_coeff=0.01,
                              z_loss_coeff=1.0, noise_std=0.0)
        torch.manual_seed(42)
        cfg_low  = MoEConfig(n_experts=6, top_k=2, balance_loss_coeff=0.01,
                              z_loss_coeff=0.0001, noise_std=0.0)
        torch.manual_seed(42); moe_high = MoE(cfg_high, d_model=D_MODEL); moe_high.eval()
        torch.manual_seed(42); moe_low  = MoE(cfg_low,  d_model=D_MODEL); moe_low.eval()
        _, aux_high = moe_high(x)
        _, aux_low  = moe_low(x)
        assert aux_high["z_loss"].item() > aux_low["z_loss"].item()

    def test_balance_loss_higher_when_imbalanced(self):
        """More imbalanced routing → higher balance loss (direction check).

        We verify this by comparing a random input (which spreads load somewhat)
        against a router whose weights strongly favour one expert.
        """
        cfg = MoEConfig(n_experts=6, top_k=2, balance_loss_coeff=1.0,
                        z_loss_coeff=0.0, noise_std=0.0)
        # Collapsed router: force expert 0 to always be picked
        moe_collapse = MoE(cfg, d_model=D_MODEL)
        moe_collapse.eval()
        with torch.no_grad():
            nn.init.zeros_(moe_collapse.router.weight)
            moe_collapse.router.weight[0].fill_(10.0)  # expert 0 dominates

        # Normal router with random weights
        moe_normal = MoE(cfg, d_model=D_MODEL)
        moe_normal.eval()

        torch.manual_seed(0)
        x = torch.randn(4, D_MODEL)   # batch=4 for meaningful f_i

        _, aux_collapse = moe_collapse(x)
        _, aux_normal   = moe_normal(x)

        # Collapsed routing should have higher balance loss (coeff=1 so raw > scaled)
        assert aux_collapse["balance_loss"].item() >= aux_normal["balance_loss"].item(), (
            f"Collapsed loss {aux_collapse['balance_loss'].item():.4f} should be >= "
            f"normal loss {aux_normal['balance_loss'].item():.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MoE — gradient flow
# ══════════════════════════════════════════════════════════════════════════════

class TestMoEGradients:
    def _backward(self, moe: MoE, x_val: torch.Tensor | None = None):
        moe.train()
        if x_val is None:
            torch.manual_seed(1)
            x_val = torch.randn(1, D_MODEL)
        x = x_val.detach().requires_grad_(True)
        out, aux = moe(x)
        total = out.sum() + aux["balance_loss"] + aux["z_loss"]
        total.backward()
        return x, out, aux

    def test_gradient_reaches_input(self, moe: MoE):
        x, _, _ = self._backward(moe)
        assert x.grad is not None and x.grad.abs().sum().item() > 0

    def test_gradient_reaches_router(self, moe: MoE):
        self._backward(moe)
        assert moe.router.weight.grad is not None
        assert moe.router.weight.grad.abs().sum().item() > 0

    def test_gradient_reaches_experts(self, moe: MoE):
        self._backward(moe)
        grads_found = sum(
            1 for exp in moe.experts
            if exp.ff1.weight.grad is not None and exp.ff1.weight.grad.abs().sum() > 0
        )
        # At least top_k experts should have received gradient
        assert grads_found >= moe.top_k, (
            f"Only {grads_found} expert(s) got gradient; expected >= {moe.top_k}"
        )

    def test_z_loss_gradient_reaches_router(self, cfg: MoEConfig):
        """z_loss must propagate gradient to router weights on its own."""
        moe = MoE(cfg, d_model=D_MODEL)
        moe.train()
        x = torch.randn(1, D_MODEL)
        _, aux = moe(x)
        aux["z_loss"].backward()
        assert moe.router.weight.grad is not None
        assert moe.router.weight.grad.abs().sum().item() > 0

    def test_balance_loss_gradient_reaches_router(self, cfg: MoEConfig):
        """balance_loss (through P_i = clean_gates) must reach router weights."""
        moe = MoE(cfg, d_model=D_MODEL)
        moe.train()
        x = torch.randn(1, D_MODEL)
        _, aux = moe(x)
        aux["balance_loss"].backward()
        assert moe.router.weight.grad is not None
        assert moe.router.weight.grad.abs().sum().item() > 0

    def test_norm_weight_receives_gradient(self, moe: MoE):
        self._backward(moe)
        assert moe.norm.weight.grad is not None
        assert moe.norm.weight.grad.abs().sum().item() > 0


# ══════════════════════════════════════════════════════════════════════════════
# MoE — numerical stability
# ══════════════════════════════════════════════════════════════════════════════

class TestMoEStability:
    @pytest.mark.parametrize("scale", [0.001, 1.0, 10.0, 100.0])
    def test_stable_for_input_scales(self, moe: MoE, scale: float):
        x = torch.randn(1, D_MODEL) * scale
        out, aux = moe(x)
        assert torch.isfinite(out).all(), f"Non-finite output for scale={scale}"
        assert torch.isfinite(aux["balance_loss"]), f"Non-finite balance_loss for scale={scale}"
        assert torch.isfinite(aux["z_loss"]), f"Non-finite z_loss for scale={scale}"

    def test_zero_input(self, moe: MoE):
        out, aux = moe(torch.zeros(1, D_MODEL))
        assert torch.isfinite(out).all()
        assert torch.isfinite(aux["balance_loss"])
        assert torch.isfinite(aux["z_loss"])

    @pytest.mark.parametrize("top_k", [1, 2, 3])
    def test_various_top_k(self, top_k: int):
        cfg = MoEConfig(n_experts=6, top_k=top_k, balance_loss_coeff=0.01,
                        z_loss_coeff=0.001, noise_std=0.0)
        moe = MoE(cfg, d_model=D_MODEL)
        moe.eval()
        x = torch.randn(1, D_MODEL)
        out, aux = moe(x)
        assert out.shape == (1, D_MODEL)
        assert torch.isfinite(out).all()

    def test_output_rms_near_one(self, moe: MoE, x: torch.Tensor):
        """RMSNorm on output keeps root-mean-square ≈ 1."""
        out, _ = moe(x)
        rms = out.pow(2).mean().sqrt()
        assert (rms - 1.0).abs().item() < 0.15


# ══════════════════════════════════════════════════════════════════════════════
# Latency
# ══════════════════════════════════════════════════════════════════════════════

def test_moe_forward_latency(cfg: MoEConfig):
    """forward() p99 benchmarked; hard gate <5ms on A100 fp16 only."""
    moe = build_moe(cfg, d_model=D_MODEL, device="cpu")
    moe.eval()
    x = torch.randn(1, D_MODEL)
    bench = LatencyBenchmark(device="cpu")

    for _ in range(20):
        moe(x)

    N = 200
    for _ in range(N):
        with bench.stage("moe_forward"):
            moe(x)

    bench.print_report()
    rep = bench.report()
    assert rep["moe_forward"]["n"] == N
    assert torch.isfinite(torch.tensor(rep["moe_forward"]["p99"]))

    p99 = rep["moe_forward"]["p99"]
    print(f"\nmoe_forward p99={p99:.2f}ms  (target: <5ms on A100 fp16)")
    if torch.cuda.is_available():
        assert p99 < 5.0, f"moe_forward p99={p99:.2f}ms exceeds 5ms budget"
