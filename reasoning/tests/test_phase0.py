"""Phase 0 test suite: config, schema, synthetic data, benchmark, temporal split, stubs."""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from reasoning.benchmark import LatencyBenchmark
from reasoning.config import ChainConfig, MoEConfig, ReasoningConfig, SSMConfig
from reasoning.interfaces import (
    ContinuousEncoderProto,
    EventEncoderProto,
    StubContinuousEncoder,
    StubEventEncoder,
    StubFusion,
    StubMoE,
    StubOutputHeads,
    StubWalletGNN,
    WalletGNNProto,
)
from reasoning.schema import (
    BlockData,
    ContinuousTick,
    DiscreteEvent,
    TransitionRecord,
    WalletEdge,
    WalletGraphBatch,
)
from reasoning.synthetic import synthetic_block_stream, synthetic_dataset
from reasoning.temporal_split import (
    TemporalSplit,
    temporal_split,
    verify_no_leakage,
    walk_forward_splits,
)


# ── Config ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        cfg = ReasoningConfig()
        assert cfg.ssm.d_model == 128
        assert cfg.ssm.d_state == 16
        assert cfg.ssm.n_layers == 2
        assert cfg.moe.n_experts == 6
        assert cfg.moe.top_k == 2
        assert cfg.moe.balance_loss_coeff == pytest.approx(0.01)
        assert cfg.moe.z_loss_coeff == pytest.approx(0.001)
        assert len(cfg.chains) == 4

    def test_override(self):
        cfg = ReasoningConfig(ssm=SSMConfig(d_model=256))
        assert cfg.ssm.d_model == 256
        assert cfg.ssm.d_state == 16  # unchanged

    def test_chain_ids(self):
        cfg = ReasoningConfig()
        ids = {c.chain_id for c in cfg.chains}
        assert ids == {"solana", "base", "bsc", "ethereum"}

    def test_chain_by_id(self):
        cfg = ReasoningConfig()
        sol = cfg.chain_by_id("solana")
        assert sol.block_time_sec == pytest.approx(0.4)
        assert sol.finality_blocks == 32

    def test_chain_by_id_missing(self):
        cfg = ReasoningConfig()
        with pytest.raises(KeyError):
            cfg.chain_by_id("nonexistent")

    def test_metadata_vector_length(self):
        cfg = ReasoningConfig()
        for chain in cfg.chains:
            vec = chain.as_metadata_vector()
            assert len(vec) == cfg.film.metadata_dim

    def test_serialization_roundtrip(self):
        cfg = ReasoningConfig()
        cfg2 = ReasoningConfig.model_validate(cfg.model_dump())
        assert cfg == cfg2

    def test_kill_switch_thresholds(self):
        cfg = ReasoningConfig()
        assert cfg.kill_switch.rug_rate_threshold == pytest.approx(0.5)
        assert cfg.kill_switch.lp_depth_drop_threshold == pytest.approx(0.30)


# ── Schema ────────────────────────────────────────────────────────────────────

class TestSchema:
    def test_continuous_tick_feature_vector(self):
        tick = ContinuousTick(1.0, "T1", "solana", 1.5, 10_000.0, 0.001, 50_000.0)
        fv = tick.as_feature_vector()
        assert fv.dtype == np.float32
        assert fv.shape == (4,)
        assert fv[0] == pytest.approx(1.5)

    def test_block_data_no_event(self):
        tick = ContinuousTick(1.0, "T1", "solana", 1.5, 10_000.0, 0.001, 50_000.0)
        block = BlockData(0, 1.0, "solana", "T1", [tick], [], 100.0)
        assert not block.has_event
        assert block.dt_since_last_event == pytest.approx(100.0)

    def test_block_data_with_event(self):
        tick = ContinuousTick(1.0, "T1", "solana", 1.5, 10_000.0, 0.001, 50_000.0)
        evt = DiscreteEvent(1.0, "T1", "solana", "whale_buy", "W1", 5_000.0)
        block = BlockData(0, 1.0, "solana", "T1", [tick], [evt], 0.0)
        assert block.has_event
        assert block.events[0].event_type == "whale_buy"

    def test_transition_record(self):
        rec = TransitionRecord(
            embedding=np.zeros(256, dtype=np.float32),
            chain_id="base",
            regime_from=0,
            regime_to=2,
            trajectory=np.linspace(1.0, 2.0, 20),
            outcome=0.42,
            is_rug=False,
            whale_signature="deadbeef",
            liquidity_path=np.ones(20),
            timestamp=1_700_000_000.0,
        )
        assert rec.embedding.shape == (256,)
        assert not rec.is_rug
        assert rec.outcome == pytest.approx(0.42)

    def test_wallet_graph_batch(self):
        nodes = [f"W{i}" for i in range(10)]
        edges = [WalletEdge("W0", "W1", 1.0, 1.0)]
        features = np.zeros((10, 8), dtype=np.float32)
        graph = WalletGraphBatch("TOKEN_0001", "solana", 1.0, nodes, edges, features)
        assert graph.node_features.shape == (10, 8)


# ── Synthetic Data ────────────────────────────────────────────────────────────

class TestSyntheticStream:
    def test_monotone_timestamps(self):
        blocks = list(synthetic_block_stream(n_blocks=500, seed=0))
        ts = [b.timestamp for b in blocks]
        assert all(a < b for a, b in zip(ts, ts[1:]))

    def test_sparse_events(self):
        n = 5_000
        rate = 0.01
        blocks = list(synthetic_block_stream(n_blocks=n, event_rate=rate, seed=42))
        n_events = sum(1 for b in blocks if b.has_event)
        expected = n * rate
        std = np.sqrt(n * rate * (1 - rate))
        # allow 4-sigma slack
        assert abs(n_events - expected) < 4 * std

    def test_99pct_blocks_have_no_event(self):
        n = 5_000
        blocks = list(synthetic_block_stream(n_blocks=n, event_rate=0.01, seed=1))
        frac_empty = sum(1 for b in blocks if not b.has_event) / n
        assert frac_empty > 0.90  # should be ~99%; >90% is a safe lower bound

    def test_dt_since_last_event_nonnegative(self):
        blocks = list(synthetic_block_stream(n_blocks=500, seed=2))
        assert all(b.dt_since_last_event >= 0.0 for b in blocks)

    def test_multiple_chains_present(self):
        blocks = list(synthetic_block_stream(n_blocks=2_000, seed=3))
        chains = {b.chain_id for b in blocks}
        assert len(chains) >= 3  # at least 3 of 4 chains appear in 2 000 blocks

    def test_price_always_positive(self):
        blocks = list(synthetic_block_stream(n_blocks=500, seed=4))
        assert all(b.ticks[0].price_usd > 0 for b in blocks)

    def test_correct_schema_types(self):
        for block in synthetic_block_stream(n_blocks=50, seed=5):
            assert isinstance(block, BlockData)
            assert len(block.ticks) == 1
            assert isinstance(block.ticks[0], ContinuousTick)
            assert isinstance(block.chain_id, str)

    def test_flows_through_stub_interfaces(self):
        """Synthetic blocks can be processed end-to-end through all stubs."""
        enc = StubContinuousEncoder(d_model=128)
        evt_enc = StubEventEncoder(d_model=128)
        gnn = StubWalletGNN(d_model=128)
        fusion = StubFusion(d_model=128)
        moe = StubMoE(d_model=128)
        heads = StubOutputHeads(n_regimes=6)

        cfg = ReasoningConfig()

        for block in synthetic_block_stream(n_blocks=100, seed=6):
            tick = block.ticks[0]
            feat = torch.tensor(tick.as_feature_vector(), dtype=torch.float32).unsqueeze(0)

            h = enc.encode_step(feat, block.ticks[0].timestamp)
            assert h.shape == (1, 128)

            if block.has_event:
                ef = torch.tensor([[block.events[0].amount_usd]], dtype=torch.float32)
                ev = evt_enc.encode_event(ef, block.dt_since_last_event)
                assert ev.shape == (1, 128)
            else:
                ev = torch.zeros(1, 128)

            gnn_emb = gnn.read_cached(block.token_address)
            assert gnn_emb.shape == (1, 128)

            chain_meta = torch.tensor(
                cfg.chain_by_id(block.chain_id).as_metadata_vector(),
                dtype=torch.float32,
            ).unsqueeze(0)

            fused = fusion.fuse(h, ev, gnn_emb, chain_meta)
            assert fused.shape == (1, 128)

            out, aux = moe.forward(fused)
            assert out.shape == (1, 128)
            assert "balance_loss" in aux

            preds = heads.forward(out)
            assert preds["regime_logits"].shape == (1, 6)
            assert preds["ood_score"].shape == (1, 1)


class TestSyntheticDataset:
    def test_dataset_size(self):
        blocks = synthetic_dataset(n_months=3, n_blocks_per_month=100)
        assert len(blocks) == 300

    def test_dataset_spans_time(self):
        blocks = synthetic_dataset(n_months=6, n_blocks_per_month=200, seed=99)
        ts = [b.timestamp for b in blocks]
        # 6 months * ~2s avg block time * 200/month is tiny; just verify monotone
        assert all(a < b for a, b in zip(ts, ts[1:]))


# ── Temporal Split ────────────────────────────────────────────────────────────

def _make_monthly_records(
    n_months: int = 24, n_per_month: int = 100, seed: int = 0
) -> list[dict]:
    """Records with timestamps uniformly distributed within each calendar month."""
    rng = np.random.default_rng(seed)
    start_ts = 1_700_000_000.0
    month_sec = 30 * 24 * 3600
    records = []
    for month in range(1, n_months + 1):
        for _ in range(n_per_month):
            ts = start_ts + (month - 1) * month_sec + rng.uniform(0, month_sec * 0.99)
            records.append({"ts": float(ts), "month": month})
    return records


_get_ts = lambda r: r["ts"]  # noqa: E731


class TestTemporalSplit:
    def setup_method(self):
        self.records = _make_monthly_records(n_months=24, n_per_month=100)

    def test_split_sizes(self):
        split = temporal_split(self.records, _get_ts)
        assert len(split.train) == 18 * 100
        assert len(split.val) == 3 * 100
        assert len(split.test) == 3 * 100

    def test_no_leakage_passes(self):
        split = temporal_split(self.records, _get_ts)
        verify_no_leakage(split, _get_ts)  # must not raise

    def test_month_assignment_train(self):
        split = temporal_split(self.records, _get_ts)
        assert {r["month"] for r in split.train}.issubset(set(range(1, 19)))

    def test_month_assignment_val(self):
        split = temporal_split(self.records, _get_ts)
        assert {r["month"] for r in split.val}.issubset(set(range(19, 22)))

    def test_month_assignment_test(self):
        split = temporal_split(self.records, _get_ts)
        assert {r["month"] for r in split.test}.issubset(set(range(22, 25)))

    def test_no_overlap(self):
        split = temporal_split(self.records, _get_ts)
        train_ts = {r["ts"] for r in split.train}
        val_ts = {r["ts"] for r in split.val}
        test_ts = {r["ts"] for r in split.test}
        assert train_ts.isdisjoint(val_ts)
        assert train_ts.isdisjoint(test_ts)
        assert val_ts.isdisjoint(test_ts)

    def test_leakage_detected(self):
        split = temporal_split(self.records, _get_ts)
        # Inject a future record into train
        corrupt = TemporalSplit(
            train=split.train + split.test[:1],
            val=split.val,
            test=split.test[1:],
        )
        with pytest.raises(AssertionError, match="Leakage"):
            verify_no_leakage(corrupt, _get_ts)

    def test_empty_split_no_error(self):
        split = temporal_split([], _get_ts)
        verify_no_leakage(split, _get_ts)  # should not raise

    def test_custom_ranges(self):
        split = temporal_split(
            self.records, _get_ts,
            train_range=(1, 12), val_range=(13, 18), test_range=(19, 24),
        )
        verify_no_leakage(split, _get_ts)
        assert {r["month"] for r in split.train}.issubset(set(range(1, 13)))


class TestWalkForward:
    def setup_method(self):
        self.records = _make_monthly_records(n_months=24, n_per_month=50)

    def test_produces_at_least_one_fold(self):
        splits = walk_forward_splits(self.records, _get_ts)
        assert len(splits) >= 1

    def test_all_folds_no_leakage(self):
        splits = walk_forward_splits(self.records, _get_ts)
        for s in splits:
            verify_no_leakage(s, _get_ts)

    def test_train_grows_each_fold(self):
        splits = walk_forward_splits(
            self.records, _get_ts,
            initial_train_months=12, val_months=3, test_months=3,
            step_months=3, total_months=24,
        )
        train_sizes = [len(s.train) for s in splits]
        assert all(a <= b for a, b in zip(train_sizes, train_sizes[1:]))

    def test_test_windows_non_overlapping(self):
        splits = walk_forward_splits(self.records, _get_ts)
        test_month_sets = [
            frozenset(r["month"] for r in s.test) for s in splits
        ]
        for i in range(len(test_month_sets)):
            for j in range(i + 1, len(test_month_sets)):
                assert test_month_sets[i].isdisjoint(test_month_sets[j])


# ── Benchmark Harness ─────────────────────────────────────────────────────────

class TestBenchmark:
    def test_stage_records_positive_time(self):
        bench = LatencyBenchmark()
        with bench.stage("work"):
            _ = np.zeros((500, 500)).sum()
        rep = bench.report()
        assert "work" in rep
        assert rep["work"]["mean"] >= 0.0
        assert rep["work"]["n"] == 1

    def test_multiple_calls_accumulate(self):
        bench = LatencyBenchmark()
        for _ in range(10):
            with bench.stage("x"):
                pass
        rep = bench.report()
        assert rep["x"]["n"] == 10

    def test_two_stages_independent(self):
        bench = LatencyBenchmark()
        for _ in range(5):
            with bench.stage("fast"):
                pass
            with bench.stage("slow"):
                time.sleep(0.005)
        rep = bench.report()
        assert rep["slow"]["mean"] > rep["fast"]["mean"]

    def test_percentile_keys_present(self):
        bench = LatencyBenchmark()
        for _ in range(20):
            with bench.stage("s"):
                pass
        rep = bench.report()
        assert all(k in rep["s"] for k in ("p50", "p90", "p99", "mean", "min", "max", "n"))

    def test_reset_clears_all(self):
        bench = LatencyBenchmark()
        with bench.stage("a"):
            pass
        bench.reset()
        assert bench.report() == {}

    def test_sleep_accuracy(self):
        """Benchmark measures real wall-clock time."""
        bench = LatencyBenchmark()
        for _ in range(5):
            with bench.stage("sleep"):
                time.sleep(0.010)
        rep = bench.report()
        # Should be at least 8 ms (generous lower bound for CI jitter)
        assert rep["sleep"]["mean"] > 8.0

    def test_assert_stage_under_passes(self):
        bench = LatencyBenchmark()
        with bench.stage("fast"):
            pass
        bench.assert_stage_under("fast", max_ms=200.0)  # should not raise

    def test_assert_stage_under_fails(self):
        bench = LatencyBenchmark()
        with bench.stage("slow"):
            time.sleep(0.050)
        with pytest.raises(AssertionError, match="budget"):
            bench.assert_stage_under("slow", max_ms=1.0)


# ── Interface Protocols ───────────────────────────────────────────────────────

class TestStubInterfaces:
    def test_stub_encoder_satisfies_protocol(self):
        assert isinstance(StubContinuousEncoder(), ContinuousEncoderProto)

    def test_stub_event_encoder_satisfies_protocol(self):
        assert isinstance(StubEventEncoder(), EventEncoderProto)

    def test_stub_gnn_satisfies_protocol(self):
        assert isinstance(StubWalletGNN(), WalletGNNProto)

    def test_stub_encoder_output_shape(self):
        enc = StubContinuousEncoder(d_model=64)
        out = enc.encode_step(torch.zeros(3, 4), 1.0)
        assert out.shape == (3, 64)

    def test_stub_encoder_batch_shape(self):
        enc = StubContinuousEncoder(d_model=64)
        # features: (batch=8, seq_len=10, d_in=4), dts: (seq_len=10,)
        out = enc.encode_batch(torch.zeros(8, 10, 4), torch.zeros(10))
        assert out.shape == (8, 10, 64)

    def test_stub_event_encoder_shape(self):
        enc = StubEventEncoder(d_model=32)
        out = enc.encode_event(torch.zeros(1, 5), 10.0)
        assert out.shape == (1, 32)

    def test_stub_gnn_cached_shape(self):
        gnn = StubWalletGNN(d_model=128)
        out = gnn.read_cached("TOKEN_0001")
        assert out.shape == (1, 128)

    def test_stub_moe_output_and_aux(self):
        moe = StubMoE(d_model=128)
        out, aux = moe.forward(torch.zeros(1, 128))
        assert out.shape == (1, 128)
        assert "balance_loss" in aux and "z_loss" in aux

    def test_stub_output_heads_keys(self):
        heads = StubOutputHeads(n_regimes=6)
        preds = heads.forward(torch.zeros(1, 128))
        required_keys = {"regime_logits", "size_mu", "size_sigma", "hazard",
                         "epistemic_var", "ood_score"}
        assert required_keys.issubset(preds.keys())

    def test_stub_output_heads_regime_shape(self):
        heads = StubOutputHeads(n_regimes=6)
        preds = heads.forward(torch.zeros(1, 128))
        assert preds["regime_logits"].shape == (1, 6)


# ── Latency Micro-Benchmark (reported, not gated) ────────────────────────────

def test_benchmark_harness_report(capsys):
    """Print a latency table; verify stages are populated."""
    bench = LatencyBenchmark()
    n = 50

    for _ in range(n):
        with bench.stage("synthetic_stream_100_blocks"):
            list(synthetic_block_stream(n_blocks=100, seed=0))

    for _ in range(n):
        with bench.stage("temporal_split_2400_records"):
            recs = _make_monthly_records(n_months=24, n_per_month=100)
            temporal_split(recs, _get_ts)

    bench.print_report()
    rep = bench.report()

    assert rep["synthetic_stream_100_blocks"]["n"] == n
    assert rep["temporal_split_2400_records"]["n"] == n

    # Nothing on a dev CPU should be pathologically slow for these ops
    assert rep["synthetic_stream_100_blocks"]["p99"] < 500.0
    assert rep["temporal_split_2400_records"]["p99"] < 500.0
