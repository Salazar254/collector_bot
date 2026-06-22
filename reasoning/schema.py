"""Data schema definitions for the reasoning layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ContinuousTick:
    """One price/volume sample at a point in time."""
    timestamp: float           # unix seconds
    token_address: str
    chain_id: str
    price_usd: float
    volume_24h_usd: float
    funding_rate: float
    lp_depth_usd: float

    def as_feature_vector(self) -> np.ndarray:
        """[price, volume, funding_rate, lp_depth] — 4 features."""
        return np.array(
            [self.price_usd, self.volume_24h_usd, self.funding_rate, self.lp_depth_usd],
            dtype=np.float32,
        )


@dataclass
class DiscreteEvent:
    """Sparse on-chain event (Helius SPL webhook / Alchemy EVM log)."""
    timestamp: float
    token_address: str
    chain_id: str
    # "whale_buy" | "whale_sell" | "lp_add" | "lp_remove" | "rug_pull"
    event_type: str
    wallet_address: str
    amount_usd: float
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class WalletEdge:
    src: str
    dst: str
    weight: float
    last_interaction_ts: float


@dataclass
class WalletGraphBatch:
    """2-hop wallet graph for GNN encoding (async path)."""
    token_address: str
    chain_id: str
    timestamp: float
    nodes: list[str]           # wallet addresses, capped at ~200
    edges: list[WalletEdge]
    node_features: np.ndarray  # (n_nodes, node_feature_dim)


@dataclass
class BlockData:
    """All data available at a single block for one token."""
    block_number: int
    timestamp: float
    chain_id: str
    token_address: str
    ticks: list[ContinuousTick]
    events: list[DiscreteEvent]        # empty ~99% of the time
    dt_since_last_event: float         # seconds; always >= 0

    @property
    def has_event(self) -> bool:
        return len(self.events) > 0


@dataclass
class TransitionRecord:
    """Schema for RAG retrieval store (Phase 6).

    Stored per regime-transition event; queried async.
    """
    embedding: np.ndarray       # float32, shape (256,)
    chain_id: str
    regime_from: int
    regime_to: int
    trajectory: np.ndarray      # price sequence over the transition window
    outcome: float              # realized PnL (positive = gain)
    is_rug: bool
    whale_signature: str        # hash of dominant whale activity pattern
    liquidity_path: np.ndarray  # LP-depth sequence over the transition window
    timestamp: float            # when the transition started
