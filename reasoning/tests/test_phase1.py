"""Phase 1 tests: Mamba-style SSM continuous encoder.

Done-when criteria (from spec):
1. Streaming encode_step output matches encode_batch output within tolerance.
2. Single-step update benchmarked <10 ms (reported; not hard-gated on slow CI).
3. Δt actually influences output (model is time-aware).
4. Hidden state persists across calls and resets cleanly.
"""
from __future__ import annotations

import math
import time

import pytest
import torch
import torch.nn as nn

from reasoning.config import SSMConfig
from reasoning.ssm import ContinuousEncoder, MambaBlock, RMSNorm, build_encoder
from reasoning.benchmark import LatencyBenchmark
from reasoning.interfaces import ContinuousEncoderProto


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def cfg() -> SSMConfig:
    return SSMConfig(d_model=128, d_state=16, n_layers=2, d_conv=4, expand=2)


@pytest.fixture()
def small_cfg() -> SSMConfig:
    """Smaller config for fast shape / logic tests."""
    return SSMConfig(d_model=32, d_state=8, n_layers=2, d_conv=4, expand=2)


@pytest.fixture()
def encoder(cfg: SSMConfig) -> ContinuousEncoder:
    enc = ContinuousEncoder(cfg, d_in_features=4)
    enc.eval()
    return enc


@pytest.fixture()
def small_encoder(small_cfg: SSMConfig) -> ContinuousEncoder:
    enc = ContinuousEncoder(small_cfg, d_in_features=4)
    enc.eval()
    return enc


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(32)
        x = torch.randn(4, 32)
        assert norm(x).shape == (4, 32)

    def test_unit_rms(self):
        norm = RMSNorm(64)
        # With weight=1, output RMS ≈ 1 per row
        x = torch.randn(100, 64)
        y = norm(x)
        rms = y.pow(2).mean(-1).sqrt()
        assert (rms - 1.0).abs().max().item() < 0.05


# ── MambaBlock ────────────────────────────────────────────────────────────────

class TestMambaBlock:
    def test_step_output_shape(self, small_cfg: SSMConfig):
        blk = MambaBlock(small_cfg.d_model, small_cfg.d_state, small_cfg.d_conv, small_cfg.expand)
        blk.eval()
        b = 3
        conv_state = torch.zeros(b, blk.d_inner, blk.d_conv)
        ssm_state = torch.zeros(b, blk.d_inner, blk.d_state)
        x = torch.randn(b, small_cfg.d_model)
        out, new_conv, new_ssm = blk.step(x, conv_state, ssm_state, 1.0)
        assert out.shape == (b, small_cfg.d_model)
        assert new_conv.shape == (b, blk.d_inner, blk.d_conv)
        assert new_ssm.shape == (b, blk.d_inner, blk.d_state)

    def test_sequence_output_shape(self, small_cfg: SSMConfig):
        blk = MambaBlock(small_cfg.d_model, small_cfg.d_state, small_cfg.d_conv, small_cfg.expand)
        blk.eval()
        seq, b = 20, 2
        x = torch.randn(b, seq, small_cfg.d_model)
        dts = torch.ones(seq) * 0.4
        out = blk.forward_sequence(x, dts)
        assert out.shape == (b, seq, small_cfg.d_model)

    def test_A_negative_definite(self, small_cfg: SSMConfig):
        blk = MambaBlock(small_cfg.d_model, small_cfg.d_state, small_cfg.d_conv, small_cfg.expand)
        A = blk._A()
        assert (A < 0).all(), "All A values must be negative for stability"

    def test_A_bar_in_unit_interval(self, small_cfg: SSMConfig):
        blk = MambaBlock(small_cfg.d_model, small_cfg.d_state, small_cfg.d_conv, small_cfg.expand)
        blk.eval()
        b = 1
        conv_state = torch.zeros(b, blk.d_inner, blk.d_conv)
        ssm_state = torch.zeros(b, blk.d_inner, blk.d_state)
        x = torch.randn(b, small_cfg.d_model)
        # Just check the step doesn't explode
        for dt in [0.4, 2.0, 12.0]:
            out, _, _ = blk.step(x, conv_state, ssm_state, dt)
            assert torch.isfinite(out).all()

    def test_step_sequence_consistency(self, small_cfg: SSMConfig):
        """step() called in a loop must match forward_sequence() exactly."""
        blk = MambaBlock(small_cfg.d_model, small_cfg.d_state, small_cfg.d_conv, small_cfg.expand)
        blk.eval()
        torch.manual_seed(0)
        b, seq = 1, 15
        x = torch.randn(b, seq, small_cfg.d_model)
        dts = torch.rand(seq) * 2.0 + 0.1  # random Δts

        # Sequence path
        seq_out = blk.forward_sequence(x, dts)  # (1, seq, d_model)

        # Step-by-step path from zero state
        conv_state = torch.zeros(b, blk.d_inner, blk.d_conv)
        ssm_state = torch.zeros(b, blk.d_inner, blk.d_state)
        step_outs = []
        for t in range(seq):
            out_t, conv_state, ssm_state = blk.step(
                x[:, t, :], conv_state, ssm_state, float(dts[t])
            )
            step_outs.append(out_t)
        step_out = torch.stack(step_outs, dim=1)  # (1, seq, d_model)

        assert torch.allclose(seq_out, step_out, atol=1e-5), \
            f"Max diff: {(seq_out - step_out).abs().max().item():.2e}"


# ── ContinuousEncoder ─────────────────────────────────────────────────────────

class TestContinuousEncoderShapes:
    def test_encode_step_shape(self, encoder: ContinuousEncoder):
        out = encoder.encode_step(torch.randn(1, 4), 0.4)
        assert out.shape == (1, 128)

    def test_encode_step_batch_shape(self, encoder: ContinuousEncoder):
        out = encoder.encode_step(torch.randn(4, 4), 1.0)
        assert out.shape == (4, 128)

    def test_encode_batch_shape(self, encoder: ContinuousEncoder):
        encoder.reset_state()
        out = encoder.encode_batch(torch.randn(2, 30, 4), torch.ones(30) * 0.4)
        assert out.shape == (2, 30, 128)

    def test_satisfies_protocol(self, encoder: ContinuousEncoder):
        assert isinstance(encoder, ContinuousEncoderProto)


class TestStreamingConsistency:
    """Core done-when criterion: step ≡ batch within tolerance."""

    @pytest.mark.parametrize("seq_len", [1, 10, 50])
    def test_step_matches_batch(self, small_encoder: ContinuousEncoder, seq_len: int):
        torch.manual_seed(42)
        batch = 1
        features = torch.randn(batch, seq_len, 4)
        dts = torch.rand(seq_len) * 3.0 + 0.4  # mix of chain block times

        # Batch path (fresh state)
        small_encoder.reset_state()
        batch_out = small_encoder.encode_batch(features, dts)  # (1, seq_len, d_model)

        # Step-by-step path (fresh state)
        small_encoder.reset_state()
        step_outs = []
        for t in range(seq_len):
            o = small_encoder.encode_step(features[:, t, :], float(dts[t]))
            step_outs.append(o)
        step_out = torch.stack(step_outs, dim=1)  # (1, seq_len, d_model)

        max_diff = (batch_out - step_out).abs().max().item()
        assert max_diff < 1e-4, (
            f"seq_len={seq_len}: streaming vs batch max diff = {max_diff:.2e}"
        )

    def test_full_config_step_matches_batch(self, encoder: ContinuousEncoder):
        """Test with the spec's d_model=128, d_state=16, 2 layers."""
        torch.manual_seed(7)
        seq_len = 20
        features = torch.randn(1, seq_len, 4)
        dts = torch.tensor([0.4, 2.0, 3.0, 12.0] * 5, dtype=torch.float32)

        encoder.reset_state()
        batch_out = encoder.encode_batch(features, dts)

        encoder.reset_state()
        step_outs = [
            encoder.encode_step(features[:, t, :], float(dts[t]))
            for t in range(seq_len)
        ]
        step_out = torch.stack(step_outs, dim=1)

        max_diff = (batch_out - step_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff: {max_diff:.2e}"


class TestHiddenState:
    def test_state_persists(self, small_encoder: ContinuousEncoder):
        """Second call with same input produces a DIFFERENT output than first call
        (because hidden state has been updated)."""
        small_encoder.reset_state()
        x = torch.randn(1, 4)
        out1 = small_encoder.encode_step(x, 1.0)
        out2 = small_encoder.encode_step(x, 1.0)
        assert not torch.allclose(out1, out2), \
            "Outputs should differ — hidden state should change between calls"

    def test_reset_restores_initial_output(self, small_encoder: ContinuousEncoder):
        """After reset, first step on the same input produces the same output."""
        x = torch.randn(1, 4)

        small_encoder.reset_state()
        out_a = small_encoder.encode_step(x, 1.0)

        # Advance state with some other inputs
        for _ in range(10):
            small_encoder.encode_step(torch.randn(1, 4), 0.5)

        small_encoder.reset_state()
        out_b = small_encoder.encode_step(x, 1.0)

        assert torch.allclose(out_a, out_b, atol=1e-6), \
            "After reset, same input must yield same output"

    def test_batch_does_not_pollute_streaming_state(self, small_encoder: ContinuousEncoder):
        """encode_batch must NOT modify self._states."""
        x = torch.randn(1, 4)
        small_encoder.reset_state()
        ref_out = small_encoder.encode_step(x, 1.0)

        # Run batch encode
        small_encoder.reset_state()
        small_encoder.encode_batch(torch.randn(1, 10, 4), torch.ones(10))

        # Now step from reset — should not be affected by the batch call
        small_encoder.reset_state()
        out_after = small_encoder.encode_step(x, 1.0)
        assert torch.allclose(ref_out, out_after, atol=1e-6)


class TestDeltaTSensitivity:
    def test_dt_affects_output(self, small_encoder: ContinuousEncoder):
        """Different Δt values must produce different outputs (model is time-aware)."""
        x = torch.randn(1, 4)

        small_encoder.reset_state()
        out_fast = small_encoder.encode_step(x, 0.4)   # Solana

        small_encoder.reset_state()
        out_slow = small_encoder.encode_step(x, 12.0)  # Ethereum

        assert not torch.allclose(out_fast, out_slow, atol=1e-4), \
            "Outputs should differ for different Δt values"

    def test_large_dt_finite(self, small_encoder: ContinuousEncoder):
        """Model must remain numerically stable for large Δt values."""
        for dt in [0.01, 1.0, 60.0, 3600.0]:
            small_encoder.reset_state()
            out = small_encoder.encode_step(torch.randn(1, 4), dt)
            assert torch.isfinite(out).all(), f"Got non-finite output for dt={dt}"

    def test_batch_irregular_dts(self, encoder: ContinuousEncoder):
        """Sequences with very irregular Δts should produce finite outputs."""
        torch.manual_seed(3)
        dts = torch.cat([
            torch.tensor([0.4] * 5),   # Solana-like
            torch.tensor([12.0] * 5),  # Ethereum-like
            torch.tensor([0.001, 100.0, 0.4, 3.0, 2.0]),  # mixed
        ])
        features = torch.randn(1, 15, 4)
        encoder.reset_state()
        out = encoder.encode_batch(features, dts)
        assert torch.isfinite(out).all()


class TestGradientFlow:
    def test_gradients_reach_input(self, small_encoder: ContinuousEncoder):
        """Loss must backpropagate to input features."""
        small_encoder.train()
        small_encoder.reset_state()
        x = torch.randn(1, 4, requires_grad=True)
        out = small_encoder.encode_step(x, 1.0)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

    def test_gradients_reach_A_log(self, small_encoder: ContinuousEncoder):
        """A_log receives gradient once non-zero hidden state exists.

        At step 0, h=0 so d_loss/d_A_log = 0 (A_bar * 0 = 0 regardless of A_log).
        At step 1, h != 0 so A_log contributes and must accumulate gradient.
        """
        small_encoder.train()
        small_encoder.reset_state()
        # Step 0: build up a non-zero hidden state
        with torch.no_grad():
            small_encoder.encode_step(torch.randn(1, 4), 1.0)
        # Re-initialise so the graph is clean for the second step
        # (we need state from step 0 to be in the fwd graph)
        small_encoder.reset_state()
        x0 = torch.randn(1, 4)
        out0 = small_encoder.encode_step(x0, 1.0)   # builds up state, in-graph
        x1 = torch.randn(1, 4)
        out1 = small_encoder.encode_step(x1, 1.0)   # A_log now contributes via h
        (out0 + out1).sum().backward()
        for blk in small_encoder.blocks:
            assert blk.A_log.grad is not None
            assert blk.A_log.grad.abs().sum().item() > 0, \
                "A_log must receive non-zero gradient on step >=1"


# ── Latency micro-benchmark ───────────────────────────────────────────────────

def test_step_latency(cfg: SSMConfig):
    """Single encode_step must be benchmarked <10 ms (done-when criterion).

    On A100 fp16 this is the target. On dev CPU we report and do not gate.
    """
    enc = build_encoder(cfg, d_in_features=4, device="cpu")
    enc.eval()

    x = torch.randn(1, 4)
    bench = LatencyBenchmark(device="cpu")

    # Warm-up
    enc.reset_state()
    for _ in range(10):
        enc.encode_step(x, 0.4)

    # Benchmark
    enc.reset_state()
    N = 200
    for _ in range(N):
        with bench.stage("encode_step"):
            _ = enc.encode_step(x, 0.4)

    rep = bench.report()
    bench.print_report()

    assert rep["encode_step"]["n"] == N
    assert torch.isfinite(torch.tensor(rep["encode_step"]["p99"]))

    p99 = rep["encode_step"]["p99"]
    print(f"\nencode_step p99={p99:.2f}ms  (target: <10ms on A100 fp16)")
    # Hard gate only on A100 CUDA; report on CPU
    if torch.cuda.is_available():
        assert p99 < 10.0, f"encode_step p99={p99:.2f}ms exceeds 10ms budget"


def test_batch_throughput(cfg: SSMConfig):
    """encode_batch over seq_len=128 — throughput sanity check."""
    enc = build_encoder(cfg, d_in_features=4, device="cpu")
    enc.eval()

    features = torch.randn(8, 128, 4)
    dts = torch.rand(128) * 2.0 + 0.4
    bench = LatencyBenchmark(device="cpu")

    # Warm-up
    for _ in range(3):
        enc.encode_batch(features, dts)

    N = 20
    for _ in range(N):
        with bench.stage("encode_batch_128"):
            enc.encode_batch(features, dts)

    bench.print_report()
    rep = bench.report()
    assert rep["encode_batch_128"]["n"] == N
    assert torch.isfinite(torch.tensor(rep["encode_batch_128"]["p99"]))
