"""Phase-interface Protocols and zero-output stub implementations.

Each later phase must implement these Protocols. Stubs return correct shapes
so Phase 0 tests can exercise the full pipeline without real model weights.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from .schema import WalletGraphBatch


# ── Phase 1 ──────────────────────────────────────────────────────────────────

@runtime_checkable
class ContinuousEncoderProto(Protocol):
    """Streaming SSM encoder (Phase 1)."""

    def encode_step(self, features: torch.Tensor, dt: float) -> torch.Tensor:
        """Single-step update; features (batch, d_in) → (batch, d_model)."""
        ...

    def encode_batch(
        self, features: torch.Tensor, dts: torch.Tensor
    ) -> torch.Tensor:
        """Sequence encode; features (batch, seq, d_in), dts (seq,) → (batch, seq, d_model)."""
        ...

    def reset_state(self) -> None:
        """Reset recurrent hidden state to zeros."""
        ...


class StubContinuousEncoder:
    """Zero-output stub — correct shape, no learned parameters."""

    def __init__(self, d_model: int = 128) -> None:
        self.d_model = d_model

    def encode_step(self, features: torch.Tensor, dt: float) -> torch.Tensor:
        return torch.zeros(features.shape[0], self.d_model)

    def encode_batch(
        self, features: torch.Tensor, dts: torch.Tensor
    ) -> torch.Tensor:
        # features: (batch, seq_len, d_in) → (batch, seq_len, d_model)
        return torch.zeros(features.shape[0], features.shape[1], self.d_model)

    def reset_state(self) -> None:
        pass


# ── Phase 2 ──────────────────────────────────────────────────────────────────

@runtime_checkable
class EventEncoderProto(Protocol):
    """Sparse discrete-event encoder (Phase 2)."""

    def encode_event(
        self, event_features: torch.Tensor, dt_since_last: float
    ) -> torch.Tensor:
        """Returns (1, d_model)."""
        ...


class StubEventEncoder:
    def __init__(self, d_model: int = 128) -> None:
        self.d_model = d_model

    def encode_event(
        self, event_features: torch.Tensor, dt_since_last: float
    ) -> torch.Tensor:
        return torch.zeros(1, self.d_model)


@runtime_checkable
class WalletGNNProto(Protocol):
    """Async wallet-graph encoder (Phase 2)."""

    def encode_async(self, graph: WalletGraphBatch) -> None:
        """Kick off async GNN computation; result stored in cache."""
        ...

    def read_cached(self, token_address: str) -> torch.Tensor:
        """Read cached embedding (1, d_model); <1 ms on critical path."""
        ...


class StubWalletGNN:
    def __init__(self, d_model: int = 128) -> None:
        self.d_model = d_model

    def encode_async(self, graph: WalletGraphBatch) -> None:
        pass

    def read_cached(self, token_address: str) -> torch.Tensor:
        return torch.zeros(1, self.d_model)


# ── Phase 3 ──────────────────────────────────────────────────────────────────

@runtime_checkable
class FusionProto(Protocol):
    """FiLM-modulated multi-chain fusion (Phase 3)."""

    def fuse(
        self,
        continuous: torch.Tensor,
        event: torch.Tensor,
        wallet: torch.Tensor,
        chain_metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Returns fused embedding (1, d_model)."""
        ...


class StubFusion:
    def __init__(self, d_model: int = 128) -> None:
        self.d_model = d_model

    def fuse(
        self,
        continuous: torch.Tensor,
        event: torch.Tensor,
        wallet: torch.Tensor,
        chain_metadata: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(1, self.d_model)


# ── Phase 4 ──────────────────────────────────────────────────────────────────

@runtime_checkable
class MoEProto(Protocol):
    """Mixture-of-Experts router + experts (Phase 4)."""

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (output (1, d_model), aux_losses dict)."""
        ...


class StubMoE:
    def __init__(self, d_model: int = 128) -> None:
        self.d_model = d_model

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux = {
            "balance_loss": torch.zeros(1),
            "z_loss": torch.zeros(1),
        }
        return torch.zeros(1, self.d_model), aux


# ── Phase 5 ──────────────────────────────────────────────────────────────────

@runtime_checkable
class OutputHeadsProto(Protocol):
    """All output heads + uncertainty (Phase 5)."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns dict with keys: regime_logits, size_mu, size_sigma,
        hazard, epistemic_var, ood_score."""
        ...


class StubOutputHeads:
    def __init__(self, n_regimes: int = 6) -> None:
        self.n_regimes = n_regimes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "regime_logits": torch.zeros(1, self.n_regimes),
            "size_mu": torch.zeros(1, 1),
            "size_sigma": torch.ones(1, 1),
            "hazard": torch.zeros(1, 1),
            "epistemic_var": torch.zeros(1, 1),
            "ood_score": torch.zeros(1, 1),
        }
