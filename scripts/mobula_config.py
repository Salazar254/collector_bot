"""
scripts/mobula_config.py — Configuration management for Mobula GraphQL API.

Manages all wallet-intelligence settings via environment variables.
Integration is disabled if MOBULA_API_KEY is not set.
Replaces axiom_config.py — Mobula is the real data source for Axiom-style signals.
"""

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Cost estimate per GraphQL query (USD) — single filterTokenWallets call.
# Adjust based on actual Mobula pricing when known.
# ---------------------------------------------------------------------------

MOBULA_COST_PER_QUERY: float = 0.005

# Smart money / quality wallet labels from Mobula's walletLabelTypes vocabulary.
# These labels are applied by Mobula's classification engine.
DEFAULT_SMART_MONEY_LABELS: frozenset[str] = frozenset({
    "smart_money", "pro_trader", "elite",
})

DEFAULT_RISK_LABELS: frozenset[str] = frozenset({
    "scammer", "bot", "sniper", "bundler",
})

DEFAULT_NEW_WALLET_LABELS: frozenset[str] = frozenset({
    "fresh_wallet", "new_wallet",
})


@dataclass
class MobulaConfig:
    """Mobula GraphQL API configuration — loaded from environment variables."""

    api_key: str = field(
        default_factory=lambda: os.environ.get("MOBULA_API_KEY", "")
    )
    graphql_url: str = field(
        default_factory=lambda: os.environ.get(
            "MOBULA_GRAPHQL_URL", "https://graphql.mobula.io/graphql"
        )
    )
    enabled: bool = field(default=False)  # set in __post_init__

    # Solana network ID used in tokenId format: "mint:networkId"
    # Mobula uses integer network IDs internally.
    solana_network_id: int = field(
        default_factory=lambda: int(
            os.environ.get("MOBULA_SOLANA_NETWORK_ID", "1399811149")
        )
    )

    # Rate limiting
    rate_limit_rps: float = field(
        default_factory=lambda: float(os.environ.get("MOBULA_RATE_LIMIT_RPS", "5"))
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(
            os.environ.get("MOBULA_CACHE_TTL_SECONDS", "300")
        )
    )
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("MOBULA_MAX_RETRIES", "3"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.environ.get("MOBULA_REQUEST_TIMEOUT", "15"))
    )

    # Pagination
    max_wallets_per_page: int = 200  # Mobula filterTokenWallets limit

    # Whale thresholds in USD
    whale_thresholds: list[float] = field(
        default_factory=lambda: [1000.0, 5000.0, 10000.0]
    )

    # Label sets for wallet classification (from Mobula's walletLabelTypes)
    smart_money_labels: frozenset[str] = field(
        default_factory=lambda: DEFAULT_SMART_MONEY_LABELS
    )
    risk_labels: frozenset[str] = field(
        default_factory=lambda: DEFAULT_RISK_LABELS
    )
    new_wallet_labels: frozenset[str] = field(
        default_factory=lambda: DEFAULT_NEW_WALLET_LABELS
    )

    # Feature category toggles — all enabled by default
    enable_smart_money: bool = True
    enable_wallet_quality: bool = True
    enable_pnl: bool = True
    enable_roi: bool = True
    enable_profitable_trader: bool = True
    enable_whale_axiom: bool = True
    enable_buyer_quality: bool = True
    enable_conviction: bool = True
    enable_early_strength: bool = True
    enable_distribution: bool = True
    enable_risk_signals: bool = True
    enable_smart_vs_retail: bool = True
    enable_composite: bool = True

    # Cost thresholds for warnings
    cost_warning_per_token_usd: float = 0.01
    cost_warning_monthly_usd: float = 50.0

    def __post_init__(self) -> None:
        """Derive enabled flag from API key presence."""
        self.enabled = bool(self.api_key)

        # Parse custom label sets from env (comma-separated)
        for attr, env_var in [
            ("smart_money_labels", "MOBULA_SMART_MONEY_LABELS"),
            ("risk_labels", "MOBULA_RISK_LABELS"),
            ("new_wallet_labels", "MOBULA_NEW_WALLET_LABELS"),
        ]:
            env_val = os.environ.get(env_var, "")
            if env_val:
                labels = {l.strip() for l in env_val.split(",") if l.strip()}
                if labels:
                    setattr(self, attr, frozenset(labels))

    @property
    def is_enabled(self) -> bool:
        """Check if Mobula integration should be active."""
        return self.enabled and self.api_key != ""

    @staticmethod
    def cost_for_query() -> float:
        """Return estimated cost for a single filterTokenWallets query."""
        return MOBULA_COST_PER_QUERY


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_config: MobulaConfig | None = None


def get_mobula_config() -> MobulaConfig:
    """Return the global MobulaConfig singleton."""
    global _config
    if _config is None:
        _config = MobulaConfig()
    return _config


def reset_mobula_config() -> None:
    """Reset the cached config (useful for testing)."""
    global _config
    _config = None
