"""Phase 2a: Sparse discrete-event encoder.

Architecture:
    event_features (1, 2) = [event_type_idx_float, log1p(amount_usd)]
    dt_since_last: float  (seconds since last discrete event)

    event_type_idx → Embedding(n_event_types, d_event_emb) → (1, d_event_emb)
    log1p(amount_usd)                                       → (1, 1)
    log1p(dt_since_last)                                    → (1, 1)
    concat → (1, d_event_emb + 2) → MLP → (1, d_model)

The log1p transform maps both dollar amounts and elapsed times to a
numerically stable range regardless of chain block time.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EventEncoderConfig
from .schema import DiscreteEvent

EVENT_TYPES = ["whale_buy", "whale_sell", "lp_add", "lp_remove", "rug_pull"]
_EVENT_TYPE_INDEX: dict[str, int] = {t: i for i, t in enumerate(EVENT_TYPES)}


def event_to_features(event: DiscreteEvent) -> torch.Tensor:
    """Convert a DiscreteEvent to a (1, 2) feature tensor.

    Returns: [[event_type_index (float), log1p(amount_usd)]]
    """
    type_idx = float(_EVENT_TYPE_INDEX.get(event.event_type, 0))
    log_amount = math.log1p(max(0.0, event.amount_usd))
    return torch.tensor([[type_idx, log_amount]], dtype=torch.float32)


class EventEncoder(nn.Module):
    """Sparse discrete-event encoder satisfying EventEncoderProto."""

    def __init__(self, cfg: EventEncoderConfig, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_event_types = cfg.n_event_types

        self.event_emb = nn.Embedding(cfg.n_event_types, cfg.d_event_emb)

        d_in = cfg.d_event_emb + 2  # emb + log_amount + log_dt
        layers: list[nn.Module] = []
        d_curr = d_in
        for _ in range(cfg.n_mlp_layers - 1):
            layers += [nn.Linear(d_curr, d_model), nn.ReLU()]
            d_curr = d_model
        layers.append(nn.Linear(d_curr, d_model))
        self.mlp = nn.Sequential(*layers)

    def encode_event(
        self, event_features: torch.Tensor, dt_since_last: float
    ) -> torch.Tensor:
        """Encode a single event.

        Args:
            event_features: (1, 2) = [event_type_idx_float, log1p(amount_usd)]
            dt_since_last:  seconds elapsed since the previous discrete event
        Returns:
            (1, d_model)
        """
        type_idx = event_features[:, 0].long().clamp(0, self.n_event_types - 1)
        log_amount = event_features[:, 1:2]          # (1, 1) — continuous

        emb = self.event_emb(type_idx)               # (1, d_event_emb)

        log_dt = torch.tensor(
            [[math.log1p(max(0.0, dt_since_last))]],
            dtype=event_features.dtype,
            device=event_features.device,
        )

        combined = torch.cat([emb, log_amount, log_dt], dim=-1)  # (1, d_emb+2)
        return self.mlp(combined)


def build_event_encoder(
    cfg: EventEncoderConfig,
    d_model: int = 128,
    device: str = "cpu",
) -> EventEncoder:
    return EventEncoder(cfg, d_model).to(device)
