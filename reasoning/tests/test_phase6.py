"""Phase 6 tests: Async RAG retrieval over TransitionRecord history.

Done-when criteria:
1. FlatIndex: query returns correct nearest neighbours (exact L2 match).
2. FlatIndex: results sorted by ascending distance.
3. FlatIndex: query on empty index returns [].
4. FlatIndex: query returns at most min(k, n_records) results.
5. FlatIndex: wrong embedding dim raises ValueError.
6. AsyncRAG: add / add_batch correctly increase __len__.
7. AsyncRAG: query_async returns immediately (non-blocking).
8. AsyncRAG: read_result returns [] while query is pending.
9. AsyncRAG: read_result returns correct TransitionRecords after completion.
10. AsyncRAG: multiple independent query IDs work simultaneously.
11. AsyncRAG: should_refresh fires every refresh_every_n_blocks blocks.
12. AsyncRAG: clear_cache discards stored results.
13. read_result p99 < 1ms (O(1) dict lookup).
14. query_async returns in < 2ms (non-blocking submit).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from reasoning.benchmark import LatencyBenchmark
from reasoning.config import RAGConfig
from reasoning.rag import AsyncRAG, FlatIndex, build_rag
from reasoning.schema import TransitionRecord

DIM = 256


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_record(
    embedding: np.ndarray,
    outcome: float = 0.1,
    is_rug: bool = False,
    chain_id: str = "solana",
    regime_from: int = 0,
    regime_to: int = 1,
) -> TransitionRecord:
    return TransitionRecord(
        embedding=embedding.astype(np.float32),
        chain_id=chain_id,
        regime_from=regime_from,
        regime_to=regime_to,
        trajectory=np.linspace(1.0, 1.1, 10, dtype=np.float32),
        outcome=outcome,
        is_rug=is_rug,
        whale_signature="abc123",
        liquidity_path=np.linspace(50_000.0, 48_000.0, 10, dtype=np.float32),
        timestamp=1_700_000_000.0,
    )


def _rng_record(seed: int, dim: int = DIM, **kw) -> TransitionRecord:
    rng = np.random.default_rng(seed)
    return _make_record(rng.random(dim).astype(np.float32), **kw)


@pytest.fixture()
def cfg() -> RAGConfig:
    return RAGConfig(
        embedding_dim=DIM,
        n_neighbors=5,
        refresh_every_n_blocks=3,
        index_type="hnsw",
    )


@pytest.fixture()
def flat(cfg: RAGConfig) -> FlatIndex:
    return FlatIndex(cfg.embedding_dim)


@pytest.fixture()
def rag(cfg: RAGConfig) -> AsyncRAG:
    r = build_rag(cfg)
    yield r
    r.shutdown()


# ══════════════════════════════════════════════════════════════════════════════
# FlatIndex — basic operations
# ══════════════════════════════════════════════════════════════════════════════

class TestFlatIndexBasics:
    def test_empty_len(self, flat: FlatIndex):
        assert len(flat) == 0

    def test_add_increments_len(self, flat: FlatIndex):
        flat.add(_rng_record(0))
        assert len(flat) == 1
        flat.add(_rng_record(1))
        assert len(flat) == 2

    def test_query_empty_returns_empty(self, flat: FlatIndex):
        q = np.zeros(DIM, dtype=np.float32)
        assert flat.query(q, k=5) == []

    def test_query_returns_list_of_transition_records(self, flat: FlatIndex):
        flat.add(_rng_record(0))
        results = flat.query(np.zeros(DIM, dtype=np.float32), k=1)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], TransitionRecord)

    def test_query_k_capped_at_n_records(self, flat: FlatIndex):
        for i in range(3):
            flat.add(_rng_record(i))
        results = flat.query(np.zeros(DIM, dtype=np.float32), k=100)
        assert len(results) == 3

    def test_wrong_dim_raises(self, flat: FlatIndex):
        bad_emb = np.zeros(DIM + 1, dtype=np.float32)
        bad_rec = _make_record(bad_emb)
        with pytest.raises(ValueError, match="dim"):
            flat.add(bad_rec)

    def test_query_returns_k_results_when_sufficient(self, flat: FlatIndex):
        for i in range(10):
            flat.add(_rng_record(i))
        results = flat.query(np.zeros(DIM, dtype=np.float32), k=5)
        assert len(results) == 5


# ══════════════════════════════════════════════════════════════════════════════
# FlatIndex — correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestFlatIndexCorrectness:
    def test_exact_match_is_nearest(self, flat: FlatIndex):
        """A query equal to one record's embedding should return that record first."""
        target_emb = np.random.default_rng(42).random(DIM).astype(np.float32)
        target = _make_record(target_emb, outcome=9.9)
        # Add decoys first
        for i in range(5):
            flat.add(_rng_record(i))
        flat.add(target)
        for i in range(5, 10):
            flat.add(_rng_record(i))

        results = flat.query(target_emb, k=1)
        assert len(results) == 1
        assert results[0].outcome == pytest.approx(9.9)

    def test_results_sorted_by_ascending_distance(self, flat: FlatIndex):
        """Verify strict distance ordering across 3 records at known positions."""
        origin = np.zeros(DIM, dtype=np.float32)
        near  = _make_record(origin + 1.0,  outcome=1.0)
        mid   = _make_record(origin + 5.0,  outcome=5.0)
        far   = _make_record(origin + 20.0, outcome=20.0)
        flat.add(far)
        flat.add(near)
        flat.add(mid)

        results = flat.query(origin, k=3)
        assert results[0].outcome == pytest.approx(1.0)
        assert results[1].outcome == pytest.approx(5.0)
        assert results[2].outcome == pytest.approx(20.0)

    def test_k1_always_returns_nearest(self, flat: FlatIndex):
        rng = np.random.default_rng(7)
        embeddings = [rng.random(DIM).astype(np.float32) for _ in range(20)]
        records = [_make_record(e, outcome=float(i)) for i, e in enumerate(embeddings)]
        for r in records:
            flat.add(r)

        query = embeddings[11]
        result = flat.query(query, k=1)
        assert result[0].outcome == pytest.approx(11.0)

    def test_records_with_matching_fields_preserved(self, flat: FlatIndex):
        """All TransitionRecord fields survive the add → query round-trip."""
        emb = np.ones(DIM, dtype=np.float32)
        rec = _make_record(
            emb, outcome=-0.5, is_rug=True,
            chain_id="ethereum", regime_from=3, regime_to=5,
        )
        flat.add(rec)
        result = flat.query(emb, k=1)[0]
        assert result.outcome == pytest.approx(-0.5)
        assert result.is_rug is True
        assert result.chain_id == "ethereum"
        assert result.regime_from == 3
        assert result.regime_to == 5

    def test_rug_records_retrievable(self, flat: FlatIndex):
        """Rug-pull records can be retrieved when their embedding is the nearest."""
        rug_emb = np.ones(DIM, dtype=np.float32) * 99.0
        rug = _make_record(rug_emb, is_rug=True)
        for i in range(8):
            flat.add(_rng_record(i))
        flat.add(rug)

        results = flat.query(rug_emb, k=1)
        assert results[0].is_rug is True

    def test_duplicate_embeddings_both_returned(self, flat: FlatIndex):
        """Identical embeddings are stored as separate records."""
        emb = np.ones(DIM, dtype=np.float32)
        flat.add(_make_record(emb, outcome=1.0))
        flat.add(_make_record(emb, outcome=2.0))
        results = flat.query(emb, k=2)
        outcomes = sorted(r.outcome for r in results)
        assert outcomes == pytest.approx([1.0, 2.0])

    def test_brute_force_agrees_with_manual(self, flat: FlatIndex):
        """Verify results against a manual NumPy reference implementation."""
        rng = np.random.default_rng(99)
        embeddings = [rng.random(DIM).astype(np.float32) for _ in range(20)]
        for i, e in enumerate(embeddings):
            flat.add(_make_record(e, outcome=float(i)))

        q = rng.random(DIM).astype(np.float32)
        k = 5
        dists = [float(np.sum((e - q) ** 2)) for e in embeddings]
        expected_order = sorted(range(20), key=lambda i: dists[i])[:k]
        expected_outcomes = [float(j) for j in expected_order]

        results = flat.query(q, k=k)
        got_outcomes = [r.outcome for r in results]
        assert got_outcomes == pytest.approx(expected_outcomes, abs=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# AsyncRAG — basic add / len
# ══════════════════════════════════════════════════════════════════════════════

class TestAsyncRAGAdd:
    def test_empty_len(self, rag: AsyncRAG):
        assert len(rag) == 0

    def test_add_increments_len(self, rag: AsyncRAG):
        rag.add(_rng_record(0))
        assert len(rag) == 1

    def test_add_batch_increments_len(self, rag: AsyncRAG):
        records = [_rng_record(i) for i in range(5)]
        rag.add_batch(records)
        assert len(rag) == 5

    def test_mixed_add_and_add_batch(self, rag: AsyncRAG):
        rag.add(_rng_record(0))
        rag.add_batch([_rng_record(i) for i in range(1, 4)])
        assert len(rag) == 4


# ══════════════════════════════════════════════════════════════════════════════
# AsyncRAG — async query interface
# ══════════════════════════════════════════════════════════════════════════════

class TestAsyncRAGQuery:
    def test_read_result_returns_empty_on_miss(self, rag: AsyncRAG):
        assert rag.read_result("nonexistent") == []

    def test_is_ready_false_before_query(self, rag: AsyncRAG):
        assert not rag.is_ready("q0")

    def test_query_async_then_wait(self, rag: AsyncRAG, cfg: RAGConfig):
        for i in range(10):
            rag.add(_rng_record(i))
        q = np.zeros(DIM, dtype=np.float32)
        rag.query_async(q, "q0")
        results = rag.wait_for("q0", timeout=5.0)
        assert isinstance(results, list)
        assert len(results) <= cfg.n_neighbors
        for r in results:
            assert isinstance(r, TransitionRecord)

    def test_is_ready_after_wait(self, rag: AsyncRAG):
        rag.add(_rng_record(0))
        rag.query_async(np.zeros(DIM, np.float32), "q1")
        rag.wait_for("q1", timeout=5.0)
        assert rag.is_ready("q1")

    def test_read_result_consistent_with_wait(self, rag: AsyncRAG):
        for i in range(10):
            rag.add(_rng_record(i))
        q = np.zeros(DIM, dtype=np.float32)
        rag.query_async(q, "q2")
        via_wait = rag.wait_for("q2", timeout=5.0)
        via_read = rag.read_result("q2")
        assert via_wait == via_read

    def test_multiple_query_ids_independent(self, rag: AsyncRAG):
        for i in range(10):
            rag.add(_rng_record(i))
        rng = np.random.default_rng(0)
        for qid in ["a", "b", "c"]:
            rag.query_async(rng.random(DIM).astype(np.float32), qid)
        for qid in ["a", "b", "c"]:
            results = rag.wait_for(qid, timeout=5.0)
            assert isinstance(results, list)

    def test_empty_index_query_returns_empty(self, rag: AsyncRAG):
        rag.query_async(np.zeros(DIM, np.float32), "qempty")
        results = rag.wait_for("qempty", timeout=5.0)
        assert results == []

    def test_nearest_neighbour_correctness_via_async(self, rag: AsyncRAG):
        """Async path must return the same nearest record as a sync query would."""
        target_emb = np.ones(DIM, dtype=np.float32)
        target = _make_record(target_emb, outcome=77.7)
        for i in range(8):
            rag.add(_rng_record(i))
        rag.add(target)

        rag.query_async(target_emb, "exact")
        results = rag.wait_for("exact", timeout=5.0)
        assert results[0].outcome == pytest.approx(77.7)


# ══════════════════════════════════════════════════════════════════════════════
# AsyncRAG — operational helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestAsyncRAGOperational:
    def test_should_refresh_fires_every_n_blocks(self, rag: AsyncRAG, cfg: RAGConfig):
        n = cfg.refresh_every_n_blocks
        assert rag.should_refresh(0)
        assert rag.should_refresh(n)
        assert rag.should_refresh(2 * n)
        assert not rag.should_refresh(1)
        assert not rag.should_refresh(n - 1)

    def test_clear_cache_removes_results(self, rag: AsyncRAG):
        rag.add(_rng_record(0))
        rag.query_async(np.zeros(DIM, np.float32), "qc")
        rag.wait_for("qc", timeout=5.0)
        assert rag.is_ready("qc")
        rag.clear_cache()
        assert not rag.is_ready("qc")
        assert rag.read_result("qc") == []

    def test_clear_cache_does_not_affect_index(self, rag: AsyncRAG):
        rag.add(_rng_record(0))
        rag.clear_cache()
        assert len(rag) == 1

    def test_shutdown_allows_graceful_exit(self, cfg: RAGConfig):
        r = build_rag(cfg)
        r.add(_rng_record(0))
        r.query_async(np.zeros(DIM, np.float32), "qs")
        r.shutdown()   # must not hang

    def test_build_factory_returns_async_rag(self, cfg: RAGConfig):
        r = build_rag(cfg)
        assert isinstance(r, AsyncRAG)
        assert len(r) == 0
        r.shutdown()


# ══════════════════════════════════════════════════════════════════════════════
# Latency
# ══════════════════════════════════════════════════════════════════════════════

def test_read_result_latency(cfg: RAGConfig):
    """read_result must be < 1ms p99 (O(1) dict lookup under a lock)."""
    r = build_rag(cfg)
    r._results["bench_qid"] = []   # pre-populate to avoid empty-miss path

    bench = LatencyBenchmark(device="cpu")
    N = 1000
    for _ in range(N):
        with bench.stage("read_result"):
            r.read_result("bench_qid")

    bench.print_report()
    rep = bench.report()
    assert rep["read_result"]["n"] == N
    p99 = rep["read_result"]["p99"]
    print(f"\nread_result p99={p99:.3f}ms  (target: <1ms)")
    assert p99 < 1.0, f"read_result p99={p99:.3f}ms exceeds 1ms"
    r.shutdown()


def test_query_async_returns_immediately(cfg: RAGConfig):
    """query_async (the submit call itself) must return in < 2ms p99."""
    r = build_rag(cfg)
    for i in range(100):
        r.add(_rng_record(i))
    q = np.zeros(DIM, dtype=np.float32)

    bench = LatencyBenchmark(device="cpu")
    N = 100
    for i in range(N):
        with bench.stage("query_async_submit"):
            r.query_async(q, f"bq_{i}")

    # Drain all pending queries
    for i in range(N):
        r.wait_for(f"bq_{i}", timeout=30.0)

    bench.print_report()
    rep = bench.report()
    p99 = rep["query_async_submit"]["p99"]
    print(f"\nquery_async submit p99={p99:.3f}ms  (target: <2ms)")
    assert p99 < 2.0, f"query_async submit p99={p99:.3f}ms exceeds 2ms"
    r.shutdown()
