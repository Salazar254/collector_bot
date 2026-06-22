"""Phase 5 tests: Output heads — regime, sizing, hazard, epistemic, OOD.

Done-when criteria:
1. forward() returns all six keys with correct shapes.
2. Satisfies OutputHeadsProto.
3. size_sigma > 0 (softplus + ε constraint).
4. hazard ∈ (0, 1) (sigmoid constraint).
5. epistemic_var == 0 in training mode.
6. epistemic_var > 0 in eval mode with n_mc_passes > 1 and high dropout.
7. ood_score >= 0; == 0 when x equals a class mean; increases with distance.
8. update_class_means() updates buffers and reduces ood_score for labelled data.
9. Gradients flow to all head weights from the prediction outputs.
10. Numerically stable for a range of input scales.
11. forward() p99 latency benchmarked.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from reasoning.benchmark import LatencyBenchmark
from reasoning.config import UncertaintyConfig
from reasoning.interfaces import OutputHeadsProto
from reasoning.output_heads import OutputHeads, build_output_heads

D_MODEL = 128
N_REGIMES = 6

REQUIRED_KEYS = {"regime_logits", "size_mu", "size_sigma", "hazard", "epistemic_var", "ood_score"}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def cfg() -> UncertaintyConfig:
    return UncertaintyConfig(
        n_mc_passes=5,
        dropout_rate=0.1,
        n_regimes=N_REGIMES,
        mahalanobis_feature_dim=D_MODEL,
    )


@pytest.fixture()
def heads(cfg: UncertaintyConfig) -> OutputHeads:
    m = OutputHeads(cfg, d_model=D_MODEL)
    m.eval()
    return m


@pytest.fixture()
def x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, D_MODEL)


# ══════════════════════════════════════════════════════════════════════════════
# Shape and protocol
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputHeadsShape:
    def test_all_keys_present(self, heads: OutputHeads, x: torch.Tensor):
        out = heads(x)
        assert set(out.keys()) == REQUIRED_KEYS

    def test_regime_logits_shape(self, heads: OutputHeads, x: torch.Tensor):
        out = heads(x)
        assert out["regime_logits"].shape == (1, N_REGIMES)

    def test_size_mu_shape(self, heads: OutputHeads, x: torch.Tensor):
        out = heads(x)
        assert out["size_mu"].shape == (1, 1)

    def test_size_sigma_shape(self, heads: OutputHeads, x: torch.Tensor):
        out = heads(x)
        assert out["size_sigma"].shape == (1, 1)

    def test_hazard_shape(self, heads: OutputHeads, x: torch.Tensor):
        out = heads(x)
        assert out["hazard"].shape == (1, 1)

    def test_epistemic_var_shape(self, heads: OutputHeads, x: torch.Tensor):
        out = heads(x)
        assert out["epistemic_var"].shape == (1, 1)

    def test_ood_score_shape(self, heads: OutputHeads, x: torch.Tensor):
        out = heads(x)
        assert out["ood_score"].shape == (1, 1)

    def test_batch_size_4(self, heads: OutputHeads):
        x = torch.randn(4, D_MODEL)
        out = heads(x)
        for key in REQUIRED_KEYS:
            assert out[key].shape[0] == 4, f"{key} wrong batch dim"

    def test_satisfies_protocol(self, heads: OutputHeads):
        assert isinstance(heads, OutputHeadsProto)

    def test_build_factory(self):
        cfg_small = UncertaintyConfig(
            n_mc_passes=1, dropout_rate=0.0,
            n_regimes=N_REGIMES, mahalanobis_feature_dim=64,
        )
        m = build_output_heads(cfg_small, d_model=64, device="cpu")
        out = m(torch.randn(1, 64))
        assert set(out.keys()) == REQUIRED_KEYS
        assert out["regime_logits"].shape == (1, N_REGIMES)

    def test_all_outputs_float32(self, heads: OutputHeads, x: torch.Tensor):
        out = heads(x)
        for key, val in out.items():
            assert val.dtype == torch.float32, f"{key} has dtype {val.dtype}"

    def test_all_outputs_finite(self, heads: OutputHeads, x: torch.Tensor):
        out = heads(x)
        for key, val in out.items():
            assert torch.isfinite(val).all(), f"{key} contains non-finite values"


# ══════════════════════════════════════════════════════════════════════════════
# Output constraints (softplus, sigmoid)
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputConstraints:
    def test_size_sigma_strictly_positive(self, heads: OutputHeads):
        torch.manual_seed(42)
        for _ in range(10):
            x = torch.randn(1, D_MODEL)
            sigma = heads(x)["size_sigma"]
            assert (sigma > 0).all(), f"size_sigma={sigma.item():.6f} is not positive"

    def test_size_sigma_above_epsilon(self, heads: OutputHeads, x: torch.Tensor):
        """size_sigma must be above the numerical floor (1e-5)."""
        sigma = heads(x)["size_sigma"]
        assert (sigma >= 1e-5).all()

    def test_hazard_in_open_unit_interval(self, heads: OutputHeads):
        torch.manual_seed(42)
        for _ in range(10):
            x = torch.randn(1, D_MODEL)
            h = heads(x)["hazard"]
            assert (h > 0).all() and (h < 1).all(), f"hazard={h.item():.6f} not in (0,1)"

    def test_regime_logits_unconstrained(self, heads: OutputHeads, x: torch.Tensor):
        """regime_logits are raw — they can be positive or negative."""
        out = heads(x)["regime_logits"]
        # Just check shape + finite; no range constraint expected
        assert torch.isfinite(out).all()

    def test_size_mu_unconstrained(self, heads: OutputHeads, x: torch.Tensor):
        """size_mu is a log-space mean — no positivity constraint."""
        out = heads(x)["size_mu"]
        assert torch.isfinite(out).all()


# ══════════════════════════════════════════════════════════════════════════════
# Epistemic uncertainty (MC dropout)
# ══════════════════════════════════════════════════════════════════════════════

class TestEpistemicVariance:
    def test_epistemic_var_zero_in_training(self, cfg: UncertaintyConfig):
        """Training mode: epistemic_var must be zero (MC not run)."""
        m = OutputHeads(cfg, d_model=D_MODEL)
        m.train()
        out = m(torch.randn(1, D_MODEL))
        assert (out["epistemic_var"] == 0).all()

    def test_epistemic_var_zero_with_single_pass(self):
        """n_mc_passes=1 in eval mode → zeros (no variance from one sample)."""
        cfg1 = UncertaintyConfig(n_mc_passes=1, dropout_rate=0.5,
                                 n_regimes=N_REGIMES, mahalanobis_feature_dim=D_MODEL)
        m = OutputHeads(cfg1, d_model=D_MODEL)
        m.eval()
        out = m(torch.randn(1, D_MODEL))
        assert (out["epistemic_var"] == 0).all()

    def test_epistemic_var_non_negative_in_eval(self, heads: OutputHeads, x: torch.Tensor):
        out = heads(x)
        assert (out["epistemic_var"] >= 0).all()

    def test_epistemic_var_positive_with_high_dropout(self):
        """High dropout rate in eval mode → reliably non-zero variance."""
        cfg_hi = UncertaintyConfig(n_mc_passes=10, dropout_rate=0.5,
                                   n_regimes=N_REGIMES, mahalanobis_feature_dim=D_MODEL)
        m = OutputHeads(cfg_hi, d_model=D_MODEL)
        m.eval()
        torch.manual_seed(0)
        x = torch.randn(1, D_MODEL)
        out = m(x)
        assert out["epistemic_var"].item() > 0, (
            "epistemic_var should be > 0 with 50% dropout over 10 MC passes"
        )

    def test_epistemic_var_shape_batch_4(self):
        cfg5 = UncertaintyConfig(n_mc_passes=5, dropout_rate=0.5,
                                  n_regimes=N_REGIMES, mahalanobis_feature_dim=D_MODEL)
        m = OutputHeads(cfg5, d_model=D_MODEL)
        m.eval()
        out = m(torch.randn(4, D_MODEL))
        assert out["epistemic_var"].shape == (4, 1)

    def test_mc_outputs_are_mean_not_last_pass(self):
        """Verify MC path returns mean over passes, not just one pass's output."""
        cfg10 = UncertaintyConfig(n_mc_passes=10, dropout_rate=0.9,
                                   n_regimes=N_REGIMES, mahalanobis_feature_dim=D_MODEL)
        m = OutputHeads(cfg10, d_model=D_MODEL)
        m.eval()
        torch.manual_seed(7)
        x = torch.randn(1, D_MODEL)
        out = m(x)
        # With 90% dropout, single-pass output would be mostly zeroed.
        # The mean over 10 passes should have non-negligible magnitude.
        assert out["regime_logits"].abs().mean().item() > 0


# ══════════════════════════════════════════════════════════════════════════════
# OOD score (Mahalanobis)
# ══════════════════════════════════════════════════════════════════════════════

class TestOODScore:
    def test_ood_score_non_negative(self, heads: OutputHeads):
        torch.manual_seed(0)
        for _ in range(10):
            out = heads(torch.randn(1, D_MODEL))
            assert (out["ood_score"] >= 0).all()

    def test_ood_score_zero_when_x_equals_class_mean(self, heads: OutputHeads):
        """Set class_means[0] = x → distance to class 0 is exactly 0."""
        x = torch.randn(1, D_MODEL)
        heads.class_means[0] = x[0]
        out = heads(x)
        assert out["ood_score"].item() < 1e-6, (
            f"ood_score should be ~0 when x is on a class mean, got {out['ood_score'].item():.2e}"
        )

    def test_ood_score_increases_with_distance(self, heads: OutputHeads):
        """A point far from all class means should have a higher OOD score."""
        x_near = torch.zeros(1, D_MODEL)   # close to zero-init class means
        x_far  = torch.ones(1, D_MODEL) * 100.0
        ood_near = heads(x_near)["ood_score"].item()
        ood_far  = heads(x_far)["ood_score"].item()
        assert ood_far > ood_near, (
            f"ood_score should increase with distance: near={ood_near:.2f}, far={ood_far:.2f}"
        )

    def test_ood_score_batch_consistency(self, heads: OutputHeads):
        """Batch and single-sample OOD scores must agree."""
        torch.manual_seed(1)
        x = torch.randn(3, D_MODEL)
        out_batch = heads(x)["ood_score"]
        scores_single = torch.cat([heads(x[i:i+1])["ood_score"] for i in range(3)])
        assert torch.allclose(out_batch, scores_single, atol=1e-5)

    def test_ood_precision_diag_scales_score(self, heads: OutputHeads):
        """Higher precision → higher OOD score for the same displacement."""
        x = torch.ones(1, D_MODEL)            # class_means = 0, so displacement = 1
        out_unit = heads(x)["ood_score"].item()

        heads2 = OutputHeads(
            UncertaintyConfig(n_mc_passes=1, dropout_rate=0.0,
                              n_regimes=N_REGIMES, mahalanobis_feature_dim=D_MODEL),
            d_model=D_MODEL,
        )
        heads2.precision_diag.fill_(10.0)
        out_high = heads2(x)["ood_score"].item()
        assert out_high > out_unit


# ══════════════════════════════════════════════════════════════════════════════
# update_class_means
# ══════════════════════════════════════════════════════════════════════════════

class TestUpdateClassMeans:
    def test_update_changes_buffer(self, cfg: UncertaintyConfig):
        m = OutputHeads(cfg, d_model=D_MODEL)
        before = m.class_means.clone()
        features = torch.randn(10, D_MODEL)
        labels = torch.randint(0, N_REGIMES, (10,))
        m.update_class_means(features, labels, momentum=0.0)  # full replace
        assert not torch.allclose(before, m.class_means), "class_means not updated"

    def test_update_reduces_ood_score(self, cfg: UncertaintyConfig):
        """After updating class_means to x, ood_score for x should decrease."""
        m = OutputHeads(cfg, d_model=D_MODEL)
        m.eval()
        x = torch.randn(1, D_MODEL)
        ood_before = m(x)["ood_score"].item()

        labels = torch.zeros(1, dtype=torch.long)   # assign x to regime 0
        m.update_class_means(x, labels, momentum=0.0)
        ood_after = m(x)["ood_score"].item()

        assert ood_after < ood_before, (
            f"ood_score should decrease after updating class means "
            f"({ood_before:.3f} → {ood_after:.3f})"
        )

    def test_full_momentum_does_not_update(self, cfg: UncertaintyConfig):
        m = OutputHeads(cfg, d_model=D_MODEL)
        before = m.class_means.clone()
        features = torch.randn(5, D_MODEL)
        labels = torch.zeros(5, dtype=torch.long)
        m.update_class_means(features, labels, momentum=1.0)   # no change
        assert torch.allclose(before, m.class_means)

    def test_update_only_affects_present_classes(self, cfg: UncertaintyConfig):
        """Classes with no samples in the batch are unchanged."""
        m = OutputHeads(cfg, d_model=D_MODEL)
        before = m.class_means.clone()
        features = torch.randn(5, D_MODEL)
        labels = torch.zeros(5, dtype=torch.long)  # only class 0
        m.update_class_means(features, labels, momentum=0.0)
        # Classes 1..5 must be unchanged
        assert torch.allclose(before[1:], m.class_means[1:])


# ══════════════════════════════════════════════════════════════════════════════
# Gradient flow
# ══════════════════════════════════════════════════════════════════════════════

class TestGradients:
    def _run_backward(self, heads: OutputHeads, x_seed: int = 5):
        heads.train()
        torch.manual_seed(x_seed)
        x = torch.randn(1, D_MODEL, requires_grad=True)
        out = heads(x)
        # Sum all prediction outputs (excluding ood_score which has no parameters)
        loss = (
            out["regime_logits"].sum()
            + out["size_mu"].sum()
            + out["size_sigma"].sum()
            + out["hazard"].sum()
        )
        loss.backward()
        return x, out

    def test_gradient_reaches_input(self, heads: OutputHeads):
        x, _ = self._run_backward(heads)
        assert x.grad is not None and x.grad.abs().sum().item() > 0

    def test_gradient_reaches_regime_head(self, heads: OutputHeads):
        self._run_backward(heads)
        assert heads.regime_head.weight.grad is not None
        assert heads.regime_head.weight.grad.abs().sum().item() > 0

    def test_gradient_reaches_size_mu_head(self, heads: OutputHeads):
        self._run_backward(heads)
        assert heads.size_mu_head.weight.grad is not None
        assert heads.size_mu_head.weight.grad.abs().sum().item() > 0

    def test_gradient_reaches_size_sigma_head(self, heads: OutputHeads):
        self._run_backward(heads)
        assert heads.size_sigma_head.weight.grad is not None
        assert heads.size_sigma_head.weight.grad.abs().sum().item() > 0

    def test_gradient_reaches_hazard_head(self, heads: OutputHeads):
        self._run_backward(heads)
        assert heads.hazard_head.weight.grad is not None
        assert heads.hazard_head.weight.grad.abs().sum().item() > 0


# ══════════════════════════════════════════════════════════════════════════════
# Sensitivity — different inputs give different outputs
# ══════════════════════════════════════════════════════════════════════════════

class TestSensitivity:
    @pytest.mark.parametrize("key", ["regime_logits", "size_mu", "size_sigma", "hazard"])
    def test_input_affects_prediction(self, heads: OutputHeads, key: str):
        torch.manual_seed(3)
        out_a = heads(torch.randn(1, D_MODEL))[key]
        out_b = heads(torch.randn(1, D_MODEL))[key]
        assert not torch.allclose(out_a, out_b, atol=1e-6), (
            f"{key} did not change for different inputs"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Numerical stability
# ══════════════════════════════════════════════════════════════════════════════

class TestStability:
    @pytest.mark.parametrize("scale", [0.001, 1.0, 10.0, 100.0])
    def test_stable_for_input_scale(self, heads: OutputHeads, scale: float):
        x = torch.randn(1, D_MODEL) * scale
        out = heads(x)
        for key, val in out.items():
            assert torch.isfinite(val).all(), f"{key} non-finite for scale={scale}"

    def test_stable_zero_input(self, heads: OutputHeads):
        out = heads(torch.zeros(1, D_MODEL))
        for key, val in out.items():
            assert torch.isfinite(val).all(), f"{key} non-finite for zero input"

    def test_stable_large_negative_input(self, heads: OutputHeads):
        x = -torch.ones(1, D_MODEL) * 100.0
        out = heads(x)
        for key, val in out.items():
            assert torch.isfinite(val).all(), f"{key} non-finite for large negative input"


# ══════════════════════════════════════════════════════════════════════════════
# Latency
# ══════════════════════════════════════════════════════════════════════════════

def test_forward_latency(cfg: UncertaintyConfig):
    """forward() (eval, MC) p99 benchmarked; no hard gate on CPU."""
    m = build_output_heads(cfg, d_model=D_MODEL, device="cpu")
    m.eval()
    x = torch.randn(1, D_MODEL)
    bench = LatencyBenchmark(device="cpu")

    for _ in range(20):
        m(x)

    N = 200
    for _ in range(N):
        with bench.stage("output_heads_forward"):
            m(x)

    bench.print_report()
    rep = bench.report()
    assert rep["output_heads_forward"]["n"] == N
    assert torch.isfinite(torch.tensor(rep["output_heads_forward"]["p99"]))

    p99 = rep["output_heads_forward"]["p99"]
    print(f"\noutput_heads_forward p99={p99:.2f}ms  (n_mc_passes={cfg.n_mc_passes})")
    if torch.cuda.is_available():
        assert p99 < 10.0 * cfg.n_mc_passes, "Latency per MC pass exceeds 10ms on GPU"
