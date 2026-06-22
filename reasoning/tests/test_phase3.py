"""Phase 3 tests: FiLM-modulated multi-chain fusion.

Done-when criteria:
1. fuse() output shape (1, d_model), satisfies FusionProto.
2. FiLM init: at startup γ≈1 and β≈0 for any chain_meta input.
3. chain_metadata differences produce different outputs.
4. Each of continuous / event / wallet affects the output independently.
5. Numerically stable for all four real chain configs (Solana/Base/BSC/ETH).
6. Stable with zero event or wallet embedding (common case: no event this block).
7. Gradients flow from output back to all four inputs.
8. fuse() p99 latency benchmarked; hard gate <2ms on CUDA only.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from reasoning.benchmark import LatencyBenchmark
from reasoning.config import FiLMConfig, ReasoningConfig
from reasoning.film_fusion import FiLMLayer, MultiChainFusion, build_fusion
from reasoning.interfaces import FusionProto


# ── Fixtures ──────────────────────────────────────────────────────────────────

D_MODEL = 128

_CHAIN_METAS: dict[str, list[float]] = {
    "solana":   [0.4,  0.3, 1.0, 32.0],
    "base":     [2.0,  0.5, 0.0,  1.0],
    "bsc":      [3.0,  0.8, 1.0, 15.0],
    "ethereum": [12.0, 0.9, 0.0, 64.0],
}


def _meta(chain: str) -> torch.Tensor:
    return torch.tensor([_CHAIN_METAS[chain]], dtype=torch.float32)


def _rand(*shape: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape)


@pytest.fixture()
def cfg() -> FiLMConfig:
    return FiLMConfig(metadata_dim=4, hidden_dim=64)


@pytest.fixture()
def fusion(cfg: FiLMConfig) -> MultiChainFusion:
    m = MultiChainFusion(cfg, d_model=D_MODEL)
    m.eval()
    return m


@pytest.fixture()
def inputs():
    """Canonical set of (continuous, event, wallet, chain_meta) tensors."""
    torch.manual_seed(7)
    return (
        torch.randn(1, D_MODEL),   # continuous
        torch.randn(1, D_MODEL),   # event
        torch.randn(1, D_MODEL),   # wallet
        _meta("solana"),           # chain_metadata
    )


# ══════════════════════════════════════════════════════════════════════════════
# FiLMLayer — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFiLMLayer:
    def test_output_shape(self, cfg: FiLMConfig):
        layer = FiLMLayer(cfg.metadata_dim, cfg.hidden_dim, D_MODEL)
        x = torch.randn(1, D_MODEL)
        meta = _meta("solana")
        out = layer(x, meta)
        assert out.shape == (1, D_MODEL)

    def test_init_gamma_is_one(self, cfg: FiLMConfig):
        """At init, gamma_proj.bias == 1 and weight == 0 → γ = 1 for any input."""
        layer = FiLMLayer(cfg.metadata_dim, cfg.hidden_dim, D_MODEL)
        assert torch.allclose(layer.gamma_proj.bias, torch.ones(D_MODEL))
        assert torch.allclose(layer.gamma_proj.weight, torch.zeros_like(layer.gamma_proj.weight))

    def test_init_beta_is_zero(self, cfg: FiLMConfig):
        """At init, beta_proj.weight == 0 and bias == 0 → β = 0 for any input."""
        layer = FiLMLayer(cfg.metadata_dim, cfg.hidden_dim, D_MODEL)
        assert torch.allclose(layer.beta_proj.bias, torch.zeros(D_MODEL))
        assert torch.allclose(layer.beta_proj.weight, torch.zeros_like(layer.beta_proj.weight))

    def test_init_is_identity(self, cfg: FiLMConfig):
        """At init, FiLM output equals x (γ=1, β=0)."""
        layer = FiLMLayer(cfg.metadata_dim, cfg.hidden_dim, D_MODEL)
        layer.eval()
        x = torch.randn(1, D_MODEL)
        out = layer(x, _meta("ethereum"))
        assert torch.allclose(out, x, atol=1e-5)

    def test_different_meta_yields_different_output_after_training(self, cfg: FiLMConfig):
        """After random-init of net weights, different chain_meta → different γ,β."""
        layer = FiLMLayer(cfg.metadata_dim, cfg.hidden_dim, D_MODEL)
        # Randomise net weights to break identity init
        with torch.no_grad():
            for p in layer.net.parameters():
                nn.init.normal_(p)
            nn.init.normal_(layer.gamma_proj.weight)
            nn.init.normal_(layer.beta_proj.weight)
        layer.eval()
        x = torch.randn(1, D_MODEL)
        out_sol = layer(x, _meta("solana"))
        out_eth = layer(x, _meta("ethereum"))
        assert not torch.allclose(out_sol, out_eth, atol=1e-4)

    def test_output_finite(self, cfg: FiLMConfig):
        layer = FiLMLayer(cfg.metadata_dim, cfg.hidden_dim, D_MODEL)
        for chain in _CHAIN_METAS:
            out = layer(torch.randn(1, D_MODEL), _meta(chain))
            assert torch.isfinite(out).all(), f"Non-finite for chain={chain}"

    def test_batch_dim_supported(self, cfg: FiLMConfig):
        layer = FiLMLayer(cfg.metadata_dim, cfg.hidden_dim, D_MODEL)
        x = torch.randn(4, D_MODEL)
        meta = torch.randn(4, cfg.metadata_dim)
        out = layer(x, meta)
        assert out.shape == (4, D_MODEL)


# ══════════════════════════════════════════════════════════════════════════════
# MultiChainFusion — shape and protocol
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiChainFusionShape:
    def test_output_shape(self, fusion: MultiChainFusion, inputs):
        continuous, event, wallet, meta = inputs
        out = fusion.fuse(continuous, event, wallet, meta)
        assert out.shape == (1, D_MODEL)

    def test_output_dtype_float32(self, fusion: MultiChainFusion, inputs):
        continuous, event, wallet, meta = inputs
        out = fusion.fuse(continuous, event, wallet, meta)
        assert out.dtype == torch.float32

    def test_output_finite(self, fusion: MultiChainFusion, inputs):
        continuous, event, wallet, meta = inputs
        out = fusion.fuse(continuous, event, wallet, meta)
        assert torch.isfinite(out).all()

    def test_satisfies_protocol(self, fusion: MultiChainFusion):
        assert isinstance(fusion, FusionProto)

    def test_build_factory(self, cfg: FiLMConfig):
        m = build_fusion(cfg, d_model=64, device="cpu")
        c = torch.randn(1, 64)
        out = m.fuse(c, c, c, _meta("base"))
        assert out.shape == (1, 64)


# ══════════════════════════════════════════════════════════════════════════════
# MultiChainFusion — sensitivity
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiChainFusionSensitivity:
    def test_chain_meta_affects_output(self, fusion: MultiChainFusion):
        """Different chain metadata must produce different fused embeddings."""
        torch.manual_seed(1)
        c, e, w = torch.randn(1, D_MODEL), torch.randn(1, D_MODEL), torch.randn(1, D_MODEL)
        # Break FiLM identity init so chain_meta actually matters
        with torch.no_grad():
            for p in fusion.film.net.parameters():
                nn.init.normal_(p)
            nn.init.normal_(fusion.film.gamma_proj.weight)
            nn.init.normal_(fusion.film.beta_proj.weight)
        out_sol = fusion.fuse(c, e, w, _meta("solana"))
        out_eth = fusion.fuse(c, e, w, _meta("ethereum"))
        assert not torch.allclose(out_sol, out_eth, atol=1e-4)

    def test_continuous_affects_output(self, fusion: MultiChainFusion):
        torch.manual_seed(2)
        e, w, meta = torch.randn(1, D_MODEL), torch.randn(1, D_MODEL), _meta("base")
        out_a = fusion.fuse(torch.randn(1, D_MODEL), e, w, meta)
        out_b = fusion.fuse(torch.randn(1, D_MODEL), e, w, meta)
        assert not torch.allclose(out_a, out_b, atol=1e-6)

    def test_event_affects_output(self, fusion: MultiChainFusion):
        torch.manual_seed(3)
        c, w, meta = torch.randn(1, D_MODEL), torch.randn(1, D_MODEL), _meta("bsc")
        out_a = fusion.fuse(c, torch.randn(1, D_MODEL), w, meta)
        out_b = fusion.fuse(c, torch.randn(1, D_MODEL), w, meta)
        assert not torch.allclose(out_a, out_b, atol=1e-6)

    def test_wallet_affects_output(self, fusion: MultiChainFusion):
        torch.manual_seed(4)
        c, e, meta = torch.randn(1, D_MODEL), torch.randn(1, D_MODEL), _meta("ethereum")
        out_a = fusion.fuse(c, e, torch.randn(1, D_MODEL), meta)
        out_b = fusion.fuse(c, e, torch.randn(1, D_MODEL), meta)
        assert not torch.allclose(out_a, out_b, atol=1e-6)

    def test_zero_event_stable(self, fusion: MultiChainFusion):
        """Zero event embedding (no event this block) must not crash."""
        c = torch.randn(1, D_MODEL)
        e = torch.zeros(1, D_MODEL)   # stub output when no event fired
        w = torch.randn(1, D_MODEL)
        out = fusion.fuse(c, e, w, _meta("solana"))
        assert out.shape == (1, D_MODEL)
        assert torch.isfinite(out).all()

    def test_zero_wallet_stable(self, fusion: MultiChainFusion):
        """Zero wallet embedding (cache miss) must not crash."""
        c = torch.randn(1, D_MODEL)
        e = torch.randn(1, D_MODEL)
        w = torch.zeros(1, D_MODEL)   # stub/cache-miss output
        out = fusion.fuse(c, e, w, _meta("base"))
        assert out.shape == (1, D_MODEL)
        assert torch.isfinite(out).all()

    def test_all_zeros_stable(self, fusion: MultiChainFusion):
        z = torch.zeros(1, D_MODEL)
        out = fusion.fuse(z, z, z, _meta("ethereum"))
        assert torch.isfinite(out).all()


# ══════════════════════════════════════════════════════════════════════════════
# MultiChainFusion — real chain configs
# ══════════════════════════════════════════════════════════════════════════════

class TestRealChainConfigs:
    @pytest.mark.parametrize("chain", list(_CHAIN_METAS.keys()))
    def test_stable_for_all_chains(self, fusion: MultiChainFusion, chain: str):
        torch.manual_seed(0)
        c, e, w = torch.randn(1, D_MODEL), torch.randn(1, D_MODEL), torch.randn(1, D_MODEL)
        out = fusion.fuse(c, e, w, _meta(chain))
        assert out.shape == (1, D_MODEL)
        assert torch.isfinite(out).all(), f"Non-finite output for chain={chain}"

    def test_chain_config_as_metadata_vector(self, cfg: FiLMConfig):
        """ChainConfig.as_metadata_vector() produces a valid (1, 4) input."""
        from reasoning.config import ReasoningConfig
        rc = ReasoningConfig()
        m = build_fusion(cfg, d_model=D_MODEL)
        m.eval()
        torch.manual_seed(0)
        c, e, w = torch.randn(1, D_MODEL), torch.randn(1, D_MODEL), torch.randn(1, D_MODEL)
        for chain_cfg in rc.chains:
            meta = torch.tensor([chain_cfg.as_metadata_vector()], dtype=torch.float32)
            assert meta.shape == (1, 4)
            out = m.fuse(c, e, w, meta)
            assert out.shape == (1, D_MODEL)
            assert torch.isfinite(out).all(), f"Non-finite for chain_id={chain_cfg.chain_id}"


# ══════════════════════════════════════════════════════════════════════════════
# MultiChainFusion — gradient flow
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiChainFusionGradients:
    def _run_backward(
        self, fusion: MultiChainFusion
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fusion.train()
        # Break identity init so the chain_meta path is live
        with torch.no_grad():
            nn.init.normal_(fusion.film.gamma_proj.weight)
            nn.init.normal_(fusion.film.beta_proj.weight)
        torch.manual_seed(5)
        c = torch.randn(1, D_MODEL, requires_grad=True)
        e = torch.randn(1, D_MODEL, requires_grad=True)
        w = torch.randn(1, D_MODEL, requires_grad=True)
        meta = _meta("bsc").requires_grad_(True)
        out = fusion.fuse(c, e, w, meta)
        out.sum().backward()
        return c, e, w, meta

    def test_gradient_reaches_continuous(self, fusion: MultiChainFusion):
        c, _, _, _ = self._run_backward(fusion)
        assert c.grad is not None and c.grad.abs().sum().item() > 0

    def test_gradient_reaches_event(self, fusion: MultiChainFusion):
        _, e, _, _ = self._run_backward(fusion)
        assert e.grad is not None and e.grad.abs().sum().item() > 0

    def test_gradient_reaches_wallet(self, fusion: MultiChainFusion):
        _, _, w, _ = self._run_backward(fusion)
        assert w.grad is not None and w.grad.abs().sum().item() > 0

    def test_gradient_reaches_chain_metadata(self, fusion: MultiChainFusion):
        _, _, _, meta = self._run_backward(fusion)
        assert meta.grad is not None and meta.grad.abs().sum().item() > 0

    def test_fusion_proj_receives_gradient(self, fusion: MultiChainFusion):
        self._run_backward(fusion)
        assert fusion.fusion_proj.weight.grad is not None
        assert fusion.fusion_proj.weight.grad.abs().sum().item() > 0

    def test_film_net_receives_gradient(self, fusion: MultiChainFusion):
        self._run_backward(fusion)
        first_layer = fusion.film.net[0]
        assert first_layer.weight.grad is not None
        assert first_layer.weight.grad.abs().sum().item() > 0


# ══════════════════════════════════════════════════════════════════════════════
# MultiChainFusion — numerical stability
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiChainFusionStability:
    @pytest.mark.parametrize("scale", [0.001, 1.0, 100.0, 1e4])
    def test_large_input_magnitudes(self, fusion: MultiChainFusion, scale: float):
        torch.manual_seed(0)
        c = torch.randn(1, D_MODEL) * scale
        e = torch.randn(1, D_MODEL) * scale
        w = torch.randn(1, D_MODEL) * scale
        out = fusion.fuse(c, e, w, _meta("solana"))
        assert torch.isfinite(out).all(), f"Non-finite for input scale={scale}"

    def test_output_unit_rms_after_rms_norm(self, fusion: MultiChainFusion):
        """RMSNorm ensures output has unit root-mean-square (≈1 per row)."""
        torch.manual_seed(0)
        c, e, w = torch.randn(1, D_MODEL), torch.randn(1, D_MODEL), torch.randn(1, D_MODEL)
        out = fusion.fuse(c, e, w, _meta("base"))
        rms = out.pow(2).mean(-1).sqrt()
        # RMSNorm weight starts at 1 so RMS ≈ 1; allow 10% slack
        assert (rms - 1.0).abs().item() < 0.1, f"RMS={rms.item():.4f} not near 1"


# ══════════════════════════════════════════════════════════════════════════════
# Latency
# ══════════════════════════════════════════════════════════════════════════════

def test_fuse_latency(cfg: FiLMConfig):
    """fuse() p99 benchmarked; hard gate <2ms on A100 fp16 only."""
    m = build_fusion(cfg, d_model=D_MODEL, device="cpu")
    m.eval()
    torch.manual_seed(0)
    c, e, w = torch.randn(1, D_MODEL), torch.randn(1, D_MODEL), torch.randn(1, D_MODEL)
    meta = _meta("solana")
    bench = LatencyBenchmark(device="cpu")

    for _ in range(20):
        m.fuse(c, e, w, meta)

    N = 300
    for _ in range(N):
        with bench.stage("fuse"):
            m.fuse(c, e, w, meta)

    bench.print_report()
    rep = bench.report()
    assert rep["fuse"]["n"] == N
    assert torch.isfinite(torch.tensor(rep["fuse"]["p99"]))

    p99 = rep["fuse"]["p99"]
    print(f"\nfuse p99={p99:.2f}ms  (target: <2ms on A100 fp16)")
    if torch.cuda.is_available():
        assert p99 < 2.0, f"fuse p99={p99:.2f}ms exceeds 2ms budget"
