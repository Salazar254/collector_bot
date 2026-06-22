"""Phase 2b: Async wallet-graph GNN encoder.

Architecture:
    WalletGraphBatch (≤200 nodes, sparse edges)
    → node_features (N, node_feature_dim) via input_proj → (N, d_hidden)
    → n_gnn_layers × GraphSAGE (mean aggregation, concat self+neigh → project)
    → mean pool over nodes → (1, d_model)

Async path (critical path):
    encode_async(graph)  — submits GNN to a single-worker ThreadPoolExecutor; returns immediately
    read_cached(addr)    — O(1) dict lookup; returns zeros until the future completes

No PyTorch Geometric dependency — adjacency is built as a dense (N, N) tensor,
which is acceptable for N ≤ 200.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WalletGNNConfig
from .schema import WalletGraphBatch


class GraphSAGELayer(nn.Module):
    """Mean-aggregation GraphSAGE layer: concat(self, mean_neigh) → Linear."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.W = nn.Linear(d_in * 2, d_out, bias=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   (N, d_in)
        adj: (N, N) raw weight matrix (not pre-normalised)
        Returns: (N, d_out)
        """
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        neigh = (adj @ x) / deg          # (N, d_in) mean-aggregated neighbours
        combined = torch.cat([x, neigh], dim=-1)   # (N, 2*d_in)
        return F.relu(self.W(combined))


class WalletGNNModel(nn.Module):
    """Multi-layer GraphSAGE that reduces a wallet graph to (1, d_model)."""

    def __init__(self, cfg: WalletGNNConfig, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(cfg.node_feature_dim, cfg.d_hidden)

        self.layers: nn.ModuleList = nn.ModuleList()
        for _ in range(cfg.n_gnn_layers - 1):
            self.layers.append(GraphSAGELayer(cfg.d_hidden, cfg.d_hidden))
        self.layers.append(GraphSAGELayer(cfg.d_hidden, d_model))

    def forward(
        self, node_features: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        node_features: (N, node_feature_dim)
        adj:           (N, N) edge weight matrix
        Returns:       (1, d_model)
        """
        x = F.relu(self.input_proj(node_features))   # (N, d_hidden)
        for layer in self.layers:
            x = layer(x, adj)                         # (N, d_hidden or d_model)
        return x.mean(dim=0, keepdim=True)            # (1, d_model)

    def encode(self, graph: WalletGraphBatch) -> torch.Tensor:
        """Convert a WalletGraphBatch to tensors and run the GNN."""
        n = len(graph.nodes)
        if n == 0:
            return torch.zeros(1, self.d_model)

        node_feat = torch.as_tensor(
            graph.node_features, dtype=torch.float32
        )  # (N, node_feature_dim)

        node_idx: dict[str, int] = {addr: i for i, addr in enumerate(graph.nodes)}
        adj = torch.zeros(n, n)
        for edge in graph.edges:
            i = node_idx.get(edge.src, -1)
            j = node_idx.get(edge.dst, -1)
            if i >= 0 and j >= 0:
                adj[i, j] += edge.weight

        return self(node_feat, adj)


class WalletGNN:
    """Async wallet-graph encoder satisfying WalletGNNProto.

    encode_async() is non-blocking: it submits GNN work to a background thread.
    read_cached()  is O(1): plain dict lookup; returns zeros on a cache miss.
    """

    def __init__(self, cfg: WalletGNNConfig, d_model: int = 128) -> None:
        self.d_model = d_model
        self._model = WalletGNNModel(cfg, d_model)
        self._model.eval()
        self._executor = ThreadPoolExecutor(max_workers=cfg.executor_workers)
        self._cache: dict[str, torch.Tensor] = {}
        self._zeros = torch.zeros(1, d_model)

    # ── WalletGNNProto interface ───────────────────────────────────────────────

    def encode_async(self, graph: WalletGraphBatch) -> None:
        """Submit GNN computation; result stored in cache upon completion."""
        self._executor.submit(self._run, graph)

    def read_cached(self, token_address: str) -> torch.Tensor:
        """Return cached embedding or zeros if not yet computed."""
        return self._cache.get(token_address, self._zeros)

    # ── internals ─────────────────────────────────────────────────────────────

    def _run(self, graph: WalletGraphBatch) -> None:
        with torch.no_grad():
            emb = self._model.encode(graph)
        self._cache[graph.token_address] = emb.detach()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def build_wallet_gnn(cfg: WalletGNNConfig, d_model: int = 128) -> WalletGNN:
    return WalletGNN(cfg, d_model)
