"""Synthetic data generator for sparse irregular meme-coin event streams."""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .schema import BlockData, ContinuousTick, DiscreteEvent

# Approximate real block times (seconds)
_CHAIN_BLOCK_TIMES: dict[str, float] = {
    "solana":   0.4,
    "base":     2.0,
    "bsc":      3.0,
    "ethereum": 12.0,
}

_EVENT_TYPES = ["whale_buy", "whale_sell", "lp_add", "lp_remove", "rug_pull"]
_DEFAULT_CHAINS = list(_CHAIN_BLOCK_TIMES.keys())


def synthetic_block_stream(
    n_blocks: int,
    n_tokens: int = 50,
    chains: list[str] | None = None,
    event_rate: float = 0.01,       # ~1% blocks carry a whale event
    start_timestamp: float = 1_700_000_000.0,  # Nov 2023
    seed: int = 42,
) -> Iterator[BlockData]:
    """Yield BlockData objects with realistic irregular timestamps.

    Key properties:
    - Timestamps are strictly monotone.
    - ~event_rate fraction of blocks carry a DiscreteEvent.
    - dt_since_last_event tracks actual time since last event per token.
    - Prices follow geometric Brownian motion with per-token state.
    """
    if chains is None:
        chains = _DEFAULT_CHAINS

    rng = np.random.default_rng(seed)

    # Per-token state
    token_prices: dict[str, float] = {
        f"TOKEN_{i:04d}": float(np.exp(rng.normal(0, 1))) for i in range(n_tokens)
    }
    token_last_event_ts: dict[str, float] = {k: start_timestamp for k in token_prices}

    t = start_timestamp

    for block_num in range(n_blocks):
        chain_id: str = str(rng.choice(chains))  # type: ignore[arg-type]
        # Log-normal jitter on block time (std ~15%)
        block_dt = float(_CHAIN_BLOCK_TIMES.get(chain_id, 2.0) * np.exp(rng.normal(0, 0.15)))
        t += block_dt

        token_address = f"TOKEN_{int(rng.integers(0, n_tokens)):04d}"

        # GBM price step
        mu, sigma = 0.0, 0.03
        prev = token_prices[token_address]
        token_prices[token_address] = float(
            prev * np.exp(
                (mu - 0.5 * sigma ** 2) * block_dt
                + sigma * np.sqrt(block_dt) * float(rng.standard_normal())
            )
        )
        price = token_prices[token_address]

        tick = ContinuousTick(
            timestamp=t,
            token_address=token_address,
            chain_id=chain_id,
            price_usd=price,
            volume_24h_usd=float(rng.exponential(10_000.0)),
            funding_rate=float(rng.normal(0.0, 0.001)),
            lp_depth_usd=float(rng.exponential(50_000.0)),
        )

        events: list[DiscreteEvent] = []
        if float(rng.random()) < event_rate:
            evt_type: str = str(rng.choice(_EVENT_TYPES))  # type: ignore[arg-type]
            events.append(
                DiscreteEvent(
                    timestamp=t,
                    token_address=token_address,
                    chain_id=chain_id,
                    event_type=evt_type,
                    wallet_address=f"WALLET_{int(rng.integers(0, 1000)):04d}",
                    amount_usd=float(rng.exponential(50_000.0)),
                    raw_payload={"block": block_num},
                )
            )
            token_last_event_ts[token_address] = t

        dt_since = max(0.0, t - token_last_event_ts[token_address])

        yield BlockData(
            block_number=block_num,
            timestamp=t,
            chain_id=chain_id,
            token_address=token_address,
            ticks=[tick],
            events=events,
            dt_since_last_event=dt_since,
        )


def synthetic_dataset(
    n_months: int = 24,
    n_blocks_per_month: int = 1_000,
    n_tokens: int = 50,
    chains: list[str] | None = None,
    event_rate: float = 0.01,
    seed: int = 42,
) -> list[BlockData]:
    """Return a list of BlockData spanning n_months.

    Uses n_blocks_per_month (default 1 000) to keep memory manageable.
    Timestamps advance naturally via synthetic block times.
    """
    total = n_months * n_blocks_per_month
    return list(
        synthetic_block_stream(
            n_blocks=total,
            n_tokens=n_tokens,
            chains=chains,
            event_rate=event_rate,
            seed=seed,
        )
    )
