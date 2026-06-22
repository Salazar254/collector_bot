"""Phase 6: Async RAG retrieval over TransitionRecord history.

Architecture:

    FlatIndex — pure-NumPy brute-force L2 KNN.
        No external dependencies.  Correct for any N; O(N·d) per query.
        Used for both "hnsw" and "ivf" index_type config values until an
        optional hnswlib/faiss dependency is added in a later pass.

    AsyncRAG — non-blocking query interface.
        add(record)              — append a TransitionRecord to the index
        add_batch(records)       — bulk append
        query_async(emb, qid)    — submit KNN to background thread; returns immediately
        read_result(qid)         — O(1) dict lookup; [] while query is pending
        is_ready(qid)            — True once the query result is available
        wait_for(qid, timeout)   — block until done (for tests / debug)
        should_refresh(block_no) — True every refresh_every_n_blocks blocks
        clear_cache()            — discard stored results (prevent unbounded growth)
        shutdown()               — join the executor thread cleanly

Critical-path latency contract:
    read_result() is a plain dict.get() under a lock — sub-millisecond.
    query_async() is a non-blocking submit — sub-millisecond on the hot path.
    The KNN query itself runs in a background thread and does not block inference.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np

from .config import RAGConfig
from .schema import TransitionRecord


# ── Flat index ────────────────────────────────────────────────────────────────

class FlatIndex:
    """Exact brute-force L2 nearest-neighbour index over float32 embeddings.

    Query complexity: O(N · dim) — suitable up to ~10 k records on CPU.
    Top-k selection uses np.argpartition (O(N)) followed by O(k log k) sort.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._embeddings: list[np.ndarray] = []
        self._records: list[TransitionRecord] = []

    # ── write path ────────────────────────────────────────────────────────────

    def add(self, record: TransitionRecord) -> None:
        emb = np.asarray(record.embedding, dtype=np.float32).ravel()
        if emb.shape != (self.dim,):
            raise ValueError(
                f"Expected embedding dim {self.dim}, got {emb.shape[0]}"
            )
        self._embeddings.append(emb)
        self._records.append(record)

    # ── read path ─────────────────────────────────────────────────────────────

    def query(self, query: np.ndarray, k: int) -> list[TransitionRecord]:
        """Return up to k nearest TransitionRecords sorted by ascending L2²."""
        n = len(self._records)
        if n == 0:
            return []
        k = min(k, n)
        q = np.asarray(query, dtype=np.float32).ravel()
        matrix = np.stack(self._embeddings)          # (N, dim)
        diffs = matrix - q                           # (N, dim)
        dists = (diffs * diffs).sum(axis=1)          # (N,) — squared L2

        if k == n:
            # All records: just sort fully
            order = np.argsort(dists)
        else:
            # Partial sort: O(N) partition + O(k log k) sort
            part = np.argpartition(dists, k)[:k]
            order = part[np.argsort(dists[part])]

        return [self._records[int(i)] for i in order]

    def __len__(self) -> int:
        return len(self._records)


# ── Async RAG wrapper ─────────────────────────────────────────────────────────

class AsyncRAG:
    """Non-blocking RAG retrieval for TransitionRecord history.

    Thread model: one background worker (single-worker ThreadPoolExecutor).
    Queries are FIFO-ordered; results are stored in self._results until cleared.
    """

    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        self._index = FlatIndex(cfg.embedding_dim)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._results: dict[str, list[TransitionRecord]] = {}
        self._pending: dict[str, Future[None]] = {}
        self._lock = threading.Lock()

    # ── write path ────────────────────────────────────────────────────────────

    def add(self, record: TransitionRecord) -> None:
        """Append a single TransitionRecord to the index."""
        self._index.add(record)

    def add_batch(self, records: list[TransitionRecord]) -> None:
        """Append a list of TransitionRecords."""
        for r in records:
            self._index.add(r)

    # ── async query interface ─────────────────────────────────────────────────

    def query_async(self, query_embedding: np.ndarray, query_id: str) -> None:
        """Submit a KNN query.  Returns immediately; result available via read_result."""
        future = self._executor.submit(self._run_query, query_embedding, query_id)
        with self._lock:
            self._pending[query_id] = future

    def _run_query(self, query_embedding: np.ndarray, query_id: str) -> None:
        results = self._index.query(query_embedding, self.cfg.n_neighbors)
        with self._lock:
            self._results[query_id] = results
            self._pending.pop(query_id, None)

    def read_result(self, query_id: str) -> list[TransitionRecord]:
        """Return cached results or [] if query is still pending or not submitted."""
        with self._lock:
            return self._results.get(query_id, [])

    def is_ready(self, query_id: str) -> bool:
        """True once the query result has been written to the cache."""
        with self._lock:
            return query_id in self._results

    def is_pending(self, query_id: str) -> bool:
        """True while the background query has not yet completed."""
        with self._lock:
            return query_id in self._pending

    def wait_for(
        self, query_id: str, timeout: float = 5.0
    ) -> list[TransitionRecord]:
        """Block until the query result is ready, then return it."""
        with self._lock:
            future: Optional[Future[None]] = self._pending.get(query_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.read_result(query_id)

    # ── operational helpers ───────────────────────────────────────────────────

    def should_refresh(self, block_number: int) -> bool:
        """True every refresh_every_n_blocks blocks (use to trigger index rebuild)."""
        return block_number % self.cfg.refresh_every_n_blocks == 0

    def clear_cache(self) -> None:
        """Discard all stored query results (prevents unbounded memory growth)."""
        with self._lock:
            self._results.clear()

    def shutdown(self) -> None:
        """Shut down the background executor; waits for in-flight queries."""
        self._executor.shutdown(wait=True)

    def __len__(self) -> int:
        return len(self._index)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_rag(cfg: RAGConfig) -> AsyncRAG:
    return AsyncRAG(cfg)
