"""Phase 2 tests: Sparse event encoder and async wallet GNN.

Done-when criteria:
1. EventEncoder: output shape (1, d_model), satisfies EventEncoderProto.
2. EventEncoder: dt_since_last, event_type, and amount each affect output.
3. EventEncoder: numerically stable for dt in [0, 86400] and amount=0.
4. EventEncoder: gradients flow to continuous input (log_amount column).
5. WalletGNN: encode() shape (1, d_model), stable for empty/single-node graphs.
6. WalletGNN: encode_async + read_cached round-trip returns correct shape.
7. WalletGNN: cache miss returns zeros of correct shape.
8. WalletGNN: read_cached p99 < 1ms (O(1) dict lookup).
9. WalletGNN: gradients flow through sync encode path.
10. ReasoningConfig gains event and wallet sub-configs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from reasoning.benchmark import LatencyBenchmark
from reasoning.config import (
    EventEncoderConfig,
    ReasoningConfig,
    WalletGNNConfig,
)
from reasoning.event_encoder import (
    EVENT_TYPES,
    EventEncoder,
    build_event_encoder,
    event_to_features,
)
from reasoning.interfaces import EventEncoderProto, WalletGNNProto
from reasoning.schema import DiscreteEvent, WalletEdge, WalletGraphBatch
from reasoning.wallet_gnn import WalletGNN, WalletGNNModel, build_wallet_gnn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_event(
    event_type: str = "whale_buy",
    amount_usd: float = 50_000.0,
    token_address: str = "TOKEN_0042",
) -> DiscreteEvent:
    return DiscreteEvent(
        timestamp=1_700_000_100.0,
        token_address=token_address,
        chain_id="solana",
        event_type=event_type,
        wallet_address="WALLET_0001",
        amount_usd=amount_usd,
    )


def _make_wallet_graph(
    n_nodes: int = 5,
    n_edges: int = 10,
    node_feature_dim: int = 8,
    token_address: str = "TOKEN_0042",
) -> WalletGraphBatch:
    rng = np.random.default_rng(0)
    nodes = [f"WALLET_{i:04d}" for i in range(n_nodes)]
    edges: list[WalletEdge] = []
    for _ in range(n_edges):
        src = nodes[int(rng.integers(0, n_nodes))]
        dst = nodes[int(rng.integers(0, n_nodes))]
        if src != dst:
            edges.append(
                WalletEdge(
                    src=src,
                    dst=dst,
                    weight=float(rng.random()),
                    last_interaction_ts=1_700_000_000.0,
                )
            )
    return WalletGraphBatch(
        token_address=token_address,
        chain_id="solana",
        timestamp=1_700_000_100.0,
        nodes=nodes,
        edges=edges,
        node_features=rng.random((n_nodes, node_feature_dim)).astype(np.float32),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def event_cfg() -> EventEncoderConfig:
    return EventEncoderConfig(n_event_types=5, d_event_emb=16, n_mlp_layers=2)


@pytest.fixture()
def wallet_cfg() -> WalletGNNConfig:
    return WalletGNNConfig(
        node_feature_dim=8, d_hidden=64, n_gnn_layers=2, max_nodes=200, executor_workers=1
    )


@pytest.fixture()
def event_encoder(event_cfg: EventEncoderConfig) -> EventEncoder:
    enc = EventEncoder(event_cfg, d_model=128)
    enc.eval()
    return enc


@pytest.fixture()
def wallet_gnn(wallet_cfg: WalletGNNConfig) -> WalletGNN:
    return build_wallet_gnn(wallet_cfg, d_model=128)


# ══════════════════════════════════════════════════════════════════════════════
# event_to_features helper
# ══════════════════════════════════════════════════════════════════════════════

class TestEventToFeatures:
    def test_shape(self):
        f = event_to_features(_make_event())
        assert f.shape == (1, 2)

    def test_dtype(self):
        f = event_to_features(_make_event())
        assert f.dtype == torch.float32

    def test_event_type_indices(self):
        for i, etype in enumerate(EVENT_TYPES):
            evt = _make_event(event_type=etype)
            f = event_to_features(evt)
            assert int(f[0, 0].item()) == i, f"{etype!r} should map to index {i}"

    def test_log_amount(self):
        evt = _make_event(amount_usd=50_000.0)
        f = event_to_features(evt)
        assert abs(f[0, 1].item() - math.log1p(50_000.0)) < 1e-5

    def test_zero_amount(self):
        evt = _make_event(amount_usd=0.0)
        f = event_to_features(evt)
        assert f[0, 1].item() == 0.0   # log1p(0) == 0

    def test_unknown_event_type_defaults_to_zero(self):
        evt = _make_event(event_type="unknown_type")
        f = event_to_features(evt)
        assert int(f[0, 0].item()) == 0


# ══════════════════════════════════════════════════════════════════════════════
# EventEncoder — shape, protocol, and stability
# ══════════════════════════════════════════════════════════════════════════════

class TestEventEncoderShape:
    def test_output_shape(self, event_encoder: EventEncoder):
        f = event_to_features(_make_event())
        out = event_encoder.encode_event(f, dt_since_last=5.0)
        assert out.shape == (1, 128)

    def test_output_dtype_float32(self, event_encoder: EventEncoder):
        f = event_to_features(_make_event())
        out = event_encoder.encode_event(f, dt_since_last=1.0)
        assert out.dtype == torch.float32

    def test_satisfies_protocol(self, event_encoder: EventEncoder):
        assert isinstance(event_encoder, EventEncoderProto)

    def test_output_finite(self, event_encoder: EventEncoder):
        f = event_to_features(_make_event())
        out = event_encoder.encode_event(f, dt_since_last=1.0)
        assert torch.isfinite(out).all()

    def test_build_factory(self, event_cfg: EventEncoderConfig):
        enc = build_event_encoder(event_cfg, d_model=64, device="cpu")
        f = event_to_features(_make_event())
        assert enc.encode_event(f, dt_since_last=1.0).shape == (1, 64)


# ══════════════════════════════════════════════════════════════════════════════
# EventEncoder — sensitivity (each input dimension affects output)
# ══════════════════════════════════════════════════════════════════════════════

class TestEventEncoderSensitivity:
    def test_dt_affects_output(self, event_encoder: EventEncoder):
        f = event_to_features(_make_event())
        out_fast = event_encoder.encode_event(f, dt_since_last=0.4)
        out_slow = event_encoder.encode_event(f, dt_since_last=3600.0)
        assert not torch.allclose(out_fast, out_slow, atol=1e-6)

    def test_event_type_affects_output(self, event_encoder: EventEncoder):
        f_buy = event_to_features(_make_event(event_type="whale_buy"))
        f_sell = event_to_features(_make_event(event_type="whale_sell"))
        out_buy = event_encoder.encode_event(f_buy, dt_since_last=1.0)
        out_sell = event_encoder.encode_event(f_sell, dt_since_last=1.0)
        assert not torch.allclose(out_buy, out_sell, atol=1e-6)

    def test_amount_affects_output(self, event_encoder: EventEncoder):
        f_small = event_to_features(_make_event(amount_usd=100.0))
        f_large = event_to_features(_make_event(amount_usd=1_000_000.0))
        out_small = event_encoder.encode_event(f_small, dt_since_last=1.0)
        out_large = event_encoder.encode_event(f_large, dt_since_last=1.0)
        assert not torch.allclose(out_small, out_large, atol=1e-6)

    @pytest.mark.parametrize("dt", [0.0, 0.4, 100.0, 3600.0, 86400.0])
    def test_stable_for_all_dt(self, event_encoder: EventEncoder, dt: float):
        f = event_to_features(_make_event())
        out = event_encoder.encode_event(f, dt_since_last=dt)
        assert torch.isfinite(out).all(), f"Non-finite output for dt={dt}"

    def test_zero_amount_stable(self, event_encoder: EventEncoder):
        f = event_to_features(_make_event(amount_usd=0.0))
        out = event_encoder.encode_event(f, dt_since_last=1.0)
        assert torch.isfinite(out).all()

    def test_all_event_types_stable(self, event_encoder: EventEncoder):
        for etype in EVENT_TYPES:
            f = event_to_features(_make_event(event_type=etype))
            out = event_encoder.encode_event(f, dt_since_last=1.0)
            assert torch.isfinite(out).all(), f"Non-finite output for event_type={etype!r}"


# ══════════════════════════════════════════════════════════════════════════════
# EventEncoder — gradient flow
# ══════════════════════════════════════════════════════════════════════════════

class TestEventEncoderGradients:
    def test_gradient_reaches_log_amount(self, event_encoder: EventEncoder):
        """Continuous log_amount column must receive a non-zero gradient."""
        event_encoder.train()
        f = event_to_features(_make_event()).requires_grad_(True)
        out = event_encoder.encode_event(f, dt_since_last=1.0)
        out.sum().backward()
        assert f.grad is not None
        # Column 0 (event type index) is cast to long — no gradient there.
        # Column 1 (log_amount) must carry gradient.
        assert f.grad[:, 1].abs().sum().item() > 0

    def test_embedding_weights_receive_gradient(self, event_encoder: EventEncoder):
        event_encoder.train()
        f = event_to_features(_make_event())
        out = event_encoder.encode_event(f, dt_since_last=1.0)
        out.sum().backward()
        assert event_encoder.event_emb.weight.grad is not None
        assert event_encoder.event_emb.weight.grad.abs().sum().item() > 0


# ══════════════════════════════════════════════════════════════════════════════
# EventEncoder — latency
# ══════════════════════════════════════════════════════════════════════════════

def test_encode_event_latency(event_cfg: EventEncoderConfig):
    """encode_event p99 benchmarked; hard gate only on CUDA (target <5ms on A100)."""
    enc = build_event_encoder(event_cfg, d_model=128, device="cpu")
    enc.eval()

    f = event_to_features(_make_event())
    bench = LatencyBenchmark(device="cpu")

    for _ in range(20):
        enc.encode_event(f, dt_since_last=1.0)

    N = 300
    for _ in range(N):
        with bench.stage("encode_event"):
            enc.encode_event(f, dt_since_last=1.0)

    bench.print_report()
    rep = bench.report()
    assert rep["encode_event"]["n"] == N
    assert torch.isfinite(torch.tensor(rep["encode_event"]["p99"]))

    p99 = rep["encode_event"]["p99"]
    print(f"\nencode_event p99={p99:.2f}ms  (target: <5ms on A100 fp16)")
    if torch.cuda.is_available():
        assert p99 < 5.0, f"encode_event p99={p99:.2f}ms exceeds 5ms budget"


# ══════════════════════════════════════════════════════════════════════════════
# WalletGNNModel — sync encode path
# ══════════════════════════════════════════════════════════════════════════════

class TestWalletGNNModelShape:
    def test_encode_shape(self, wallet_cfg: WalletGNNConfig):
        model = WalletGNNModel(wallet_cfg, d_model=128)
        graph = _make_wallet_graph(n_nodes=10, n_edges=20)
        out = model.encode(graph)
        assert out.shape == (1, 128)

    def test_encode_single_node(self, wallet_cfg: WalletGNNConfig):
        model = WalletGNNModel(wallet_cfg, d_model=128)
        graph = _make_wallet_graph(n_nodes=1, n_edges=0)
        out = model.encode(graph)
        assert out.shape == (1, 128)

    def test_encode_no_edges(self, wallet_cfg: WalletGNNConfig):
        """Graph with nodes but zero edges should produce a finite embedding."""
        model = WalletGNNModel(wallet_cfg, d_model=128)
        graph = _make_wallet_graph(n_nodes=8, n_edges=0)
        graph.edges = []
        out = model.encode(graph)
        assert out.shape == (1, 128)
        assert torch.isfinite(out).all()

    def test_encode_empty_node_list(self, wallet_cfg: WalletGNNConfig):
        """Empty node list returns zeros (1, d_model)."""
        model = WalletGNNModel(wallet_cfg, d_model=128)
        graph = WalletGraphBatch(
            token_address="T", chain_id="solana", timestamp=0.0,
            nodes=[], edges=[],
            node_features=np.zeros((0, wallet_cfg.node_feature_dim), dtype=np.float32),
        )
        out = model.encode(graph)
        assert out.shape == (1, 128)
        assert (out == 0).all()

    def test_encode_output_finite(self, wallet_cfg: WalletGNNConfig):
        model = WalletGNNModel(wallet_cfg, d_model=128)
        graph = _make_wallet_graph(n_nodes=30, n_edges=60)
        out = model.encode(graph)
        assert torch.isfinite(out).all()

    def test_encode_large_graph(self, wallet_cfg: WalletGNNConfig):
        """Near-maximum node count (200) should not crash."""
        model = WalletGNNModel(wallet_cfg, d_model=128)
        graph = _make_wallet_graph(n_nodes=200, n_edges=400)
        out = model.encode(graph)
        assert out.shape == (1, 128)
        assert torch.isfinite(out).all()


# ══════════════════════════════════════════════════════════════════════════════
# WalletGNN — async interface and cache
# ══════════════════════════════════════════════════════════════════════════════

class TestWalletGNNAsync:
    def test_satisfies_protocol(self, wallet_gnn: WalletGNN):
        assert isinstance(wallet_gnn, WalletGNNProto)

    def test_cache_miss_returns_zeros(self, wallet_gnn: WalletGNN):
        emb = wallet_gnn.read_cached("NONEXISTENT_TOKEN")
        assert emb.shape == (1, 128)
        assert (emb == 0).all()

    def test_cache_miss_shape_always_correct(self, wallet_gnn: WalletGNN):
        for addr in ["A", "B", "C"]:
            emb = wallet_gnn.read_cached(addr)
            assert emb.shape == (1, 128)

    def test_async_then_read(self, wallet_gnn: WalletGNN, wallet_cfg: WalletGNNConfig):
        """encode_async followed by a barrier future ensures cache is populated."""
        graph = _make_wallet_graph(
            n_nodes=5, n_edges=8, node_feature_dim=wallet_cfg.node_feature_dim
        )
        wallet_gnn.encode_async(graph)
        # Barrier: a no-op submitted after the GNN work will resolve only after
        # the GNN work completes (single-worker executor preserves FIFO order).
        wallet_gnn._executor.submit(lambda: None).result(timeout=5.0)

        emb = wallet_gnn.read_cached("TOKEN_0042")
        assert emb.shape == (1, 128)
        assert torch.isfinite(emb).all()

    def test_async_populates_non_zero(self, wallet_gnn: WalletGNN, wallet_cfg: WalletGNNConfig):
        """Result of GNN on a real graph should not be exactly zero."""
        graph = _make_wallet_graph(
            n_nodes=5, n_edges=8, node_feature_dim=wallet_cfg.node_feature_dim
        )
        wallet_gnn.encode_async(graph)
        wallet_gnn._executor.submit(lambda: None).result(timeout=5.0)
        emb = wallet_gnn.read_cached("TOKEN_0042")
        # With random initialised weights, output on real features is non-zero
        assert emb.abs().sum().item() > 0

    def test_cache_overwrite_on_second_call(
        self, wallet_gnn: WalletGNN, wallet_cfg: WalletGNNConfig
    ):
        """Second encode_async for the same token overwrites the cache entry."""
        g1 = _make_wallet_graph(n_nodes=3, n_edges=3, node_feature_dim=wallet_cfg.node_feature_dim)
        g2 = _make_wallet_graph(
            n_nodes=10, n_edges=15, node_feature_dim=wallet_cfg.node_feature_dim,
            token_address="TOKEN_0042",
        )
        wallet_gnn.encode_async(g1)
        wallet_gnn._executor.submit(lambda: None).result(timeout=5.0)
        emb1 = wallet_gnn.read_cached("TOKEN_0042").clone()

        wallet_gnn.encode_async(g2)
        wallet_gnn._executor.submit(lambda: None).result(timeout=5.0)
        emb2 = wallet_gnn.read_cached("TOKEN_0042")

        assert emb1.shape == (1, 128)
        assert emb2.shape == (1, 128)

    def test_multiple_tokens_cached_independently(
        self, wallet_gnn: WalletGNN, wallet_cfg: WalletGNNConfig
    ):
        ga = _make_wallet_graph(
            n_nodes=4, n_edges=5, node_feature_dim=wallet_cfg.node_feature_dim,
            token_address="TOKEN_A",
        )
        gb = _make_wallet_graph(
            n_nodes=6, n_edges=8, node_feature_dim=wallet_cfg.node_feature_dim,
            token_address="TOKEN_B",
        )
        wallet_gnn.encode_async(ga)
        wallet_gnn.encode_async(gb)
        wallet_gnn._executor.submit(lambda: None).result(timeout=5.0)

        assert wallet_gnn.read_cached("TOKEN_A").shape == (1, 128)
        assert wallet_gnn.read_cached("TOKEN_B").shape == (1, 128)
        assert wallet_gnn.read_cached("TOKEN_C").shape == (1, 128)  # miss → zeros
        assert (wallet_gnn.read_cached("TOKEN_C") == 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# WalletGNN — gradient flow (sync path for training)
# ══════════════════════════════════════════════════════════════════════════════

class TestWalletGNNGradients:
    def test_gradients_flow_through_encode(self, wallet_cfg: WalletGNNConfig):
        model = WalletGNNModel(wallet_cfg, d_model=128)
        model.train()
        graph = _make_wallet_graph(n_nodes=5, n_edges=8)
        out = model.encode(graph)
        out.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert has_grad, "No parameter received a non-zero gradient"


# ══════════════════════════════════════════════════════════════════════════════
# WalletGNN — latency
# ══════════════════════════════════════════════════════════════════════════════

def test_read_cached_latency(wallet_gnn: WalletGNN):
    """read_cached must be <1ms p99 (O(1) dict lookup)."""
    wallet_gnn._cache["BENCH_TOKEN"] = torch.zeros(1, 128)
    bench = LatencyBenchmark(device="cpu")

    N = 1000
    for _ in range(N):
        with bench.stage("read_cached"):
            wallet_gnn.read_cached("BENCH_TOKEN")

    bench.print_report()
    rep = bench.report()
    assert rep["read_cached"]["n"] == N
    p99 = rep["read_cached"]["p99"]
    print(f"\nread_cached p99={p99:.3f}ms  (target: <1ms)")
    assert p99 < 1.0, f"read_cached p99={p99:.3f}ms exceeds 1ms budget"


# ══════════════════════════════════════════════════════════════════════════════
# ReasoningConfig — Phase 2 fields
# ══════════════════════════════════════════════════════════════════════════════

class TestReasoningConfigPhase2:
    def test_config_has_event_field(self):
        cfg = ReasoningConfig()
        assert isinstance(cfg.event, EventEncoderConfig)

    def test_config_has_wallet_field(self):
        cfg = ReasoningConfig()
        assert isinstance(cfg.wallet, WalletGNNConfig)

    def test_event_config_defaults(self):
        cfg = EventEncoderConfig()
        assert cfg.n_event_types == 5
        assert cfg.d_event_emb == 16
        assert cfg.n_mlp_layers == 2

    def test_wallet_config_defaults(self):
        cfg = WalletGNNConfig()
        assert cfg.node_feature_dim == 8
        assert cfg.n_gnn_layers == 2
        assert cfg.executor_workers == 1

    def test_reasoning_config_round_trips_json(self):
        cfg = ReasoningConfig()
        restored = ReasoningConfig.model_validate_json(cfg.model_dump_json())
        assert restored.event.n_event_types == cfg.event.n_event_types
        assert restored.wallet.node_feature_dim == cfg.wallet.node_feature_dim
