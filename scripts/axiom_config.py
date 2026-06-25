"""
scripts/axiom_config.py — Configuration management for Axiom API integration.

Manages all Axiom-related settings via environment variables.
Axiom is optional — integration is disabled if AXIOM_API_KEY is not set.
"""

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Cost estimates per endpoint (USD) — placeholder values, adjust based on
# actual Axiom pricing when known.
# ---------------------------------------------------------------------------

AXIOM_COST_PER_REQUEST: dict[str, float] = {
    "token_wallet_stats": 0.001,
    "smart_money_activity": 0.002,
    "wallet_profiles": 0.0005,
    "whale_transactions": 0.001,
}


@dataclass
class AxiomConfig:
    """Axiom API configuration — loaded from environment variables."""

    api_key: str = field(default_factory=lambda: os.environ.get("AXIOM_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "AXIOM_BASE_URL", "https://api.axiom.trade/v1"
        )
    )
    enabled: bool = field(default=False)  # set in __post_init__

    # Rate limiting
    rate_limit_rps: float = field(
        default_factory=lambda: float(os.environ.get("AXIOM_RATE_LIMIT_RPS", "5"))
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.environ.get("AXIOM_CACHE_TTL_SECONDS", "300"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("AXIOM_MAX_RETRIES", "3"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.environ.get("AXIOM_REQUEST_TIMEOUT", "15"))
    )

    # Whale thresholds in USD
    whale_thresholds: list[float] = field(
        default_factory=lambda: [1000.0, 5000.0, 10000.0]
    )

    # Smart money wallet addresses (configurable list)
    smart_money_wallets: list[str] = field(default_factory=list)

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
        """Derive enabled flag and parse wallet list."""
        self.enabled = bool(self.api_key)

        # Parse smart money wallet list from env var (comma-separated)
        wallet_str = os.environ.get("AXIOM_SMART_MONEY_WALLETS", "")
        if wallet_str and not self.smart_money_wallets:
            self.smart_money_wallets = [
                w.strip() for w in wallet_str.split(",") if w.strip()
            ]

    @property
    def is_enabled(self) -> bool:
        """Check if Axiom integration should be active."""
        return self.enabled and self.api_key != ""

    @staticmethod
    def cost_for_request(request_type: str) -> float:
        """Return estimated cost for a given request type."""
        return AXIOM_COST_PER_REQUEST.get(request_type, 0.001)


# Global singleton — lazy-initialized
_config: AxiomConfig | None = None


def get_axiom_config() -> AxiomConfig:
    """Return the global AxiomConfig singleton."""
    global _config
    if _config is None:
        _config = AxiomConfig()
    return _config


def reset_axiom_config() -> None:
    """Reset the cached config (useful for testing)."""
    global _config
    _config = None
