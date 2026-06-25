"""
scripts/axiom_service.py — Axiom service layer orchestrating Mobula API calls and feature extraction.

The AxiomService is the single entry point for wallet-intelligence data collection.
It handles:
  - Checking if Mobula is enabled
  - Fetching token wallet data via MobulaClient
  - Computing features via axiom_features
  - Tracking API costs
  - Returning a flat feature dict for Supabase upsert

Mobula is OPTIONAL — returns empty features if disabled or on failure.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from mobula_client import MobulaClient, get_mobula_client, reset_mobula_client
from mobula_config import MobulaConfig, get_mobula_config, reset_mobula_config
from axiom_features import compute_axiom_features

log = logging.getLogger(__name__)


class AxiomService:
    """
    Orchestrates the full wallet-intelligence collection flow for a single token.

    Usage:
        service = AxiomService()
        features = service.collect_for_token(mint, t0_ts, swaps_by_window)
        # features is a flat dict ready for Supabase upsert
    """

    def __init__(self, config: Optional[MobulaConfig] = None) -> None:
        self._config = config or get_mobula_config()
        self._client: Optional[MobulaClient] = None

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def collect_for_token(
        self,
        mint: str,
        t0_ts: int,
        swaps_by_window: dict[str, list[dict]],
    ) -> dict[str, Any]:
        """
        Collect all wallet-intelligence features for a single token.

        Calls Mobula filterTokenWallets once, then cross-references with
        swap data to compute ~80 features across 13 categories.

        Args:
            mint: Token mint address
            t0_ts: Unix timestamp of token graduation (T0)
            swaps_by_window: {"1m": [parsed_swaps], "5m": [...], "15m": [...]}

        Returns:
            Flat dict with:
              - All Axiom feature values
              - axiom_collected: bool (True if data was fetched)
              - axiom_cost_usd: float (estimated API cost)
            Returns empty defaults if disabled or fails.
        """
        if not self._config.is_enabled:
            return self._empty_features()

        cost_usd = 0.0
        collected = False

        try:
            client = self._get_client()

            # Single GraphQL query replaces 4 fake REST calls
            token_wallets = client.fetch_token_wallets(mint)
            cost_usd = MobulaConfig.cost_for_query() * max(
                1, (len(token_wallets) // self._config.max_wallets_per_page) + 1
            )

            if not token_wallets:
                log.debug(
                    "Mobula: no wallet data for %s (token may be too new)",
                    mint[:12],
                )
                return self._empty_features(cost_usd=cost_usd)

            collected = True

            # Compute features from wallet data + swap cross-reference
            features = compute_axiom_features(
                token_wallets=token_wallets,
                swaps_by_window=swaps_by_window,
                t0_ts=t0_ts,
                smart_labels=self._config.smart_money_labels,
                risk_labels=self._config.risk_labels,
                new_labels=self._config.new_wallet_labels,
            )

            # Add metadata
            features["axiom_collected"] = True
            features["axiom_cost_usd"] = round(cost_usd, 6)

            log.debug(
                "Mobula: %d features collected for %s (cost: $%.6f)",
                len(features) - 2, mint[:12], cost_usd,
            )

            return features

        except Exception:
            log.exception(
                "Mobula collection failed for %s — returning empty features",
                mint[:12],
            )
            return self._empty_features(cost_usd=cost_usd)

    def collect_for_mints_batch(
        self,
        mints: list[str],
        swaps_by_mint: dict[str, dict[str, list[dict]]],
        t0_ts_by_mint: dict[str, int],
    ) -> dict[str, dict[str, Any]]:
        """
        Collect wallet-intelligence features for multiple tokens in batch.
        Useful for testing or backfill operations.

        Returns: {mint: features_dict}
        """
        results: dict[str, dict[str, Any]] = {}
        for mint in mints:
            swaps = swaps_by_mint.get(mint, {})
            t0 = t0_ts_by_mint.get(mint, 0)
            results[mint] = self.collect_for_token(mint, t0, swaps)
        return results

    # ---------------------------------------------------------------
    # Health Check
    # ---------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """
        Check if Mobula integration is healthy.
        Returns dict with status and diagnostic info.
        """
        if not self._config.is_enabled:
            return {
                "status": "disabled",
                "reason": "MOBULA_API_KEY not set",
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

        client = self._get_client()
        return {
            "status": "healthy",
            "queries_made": client.query_count,
            "total_cost_usd": round(client.total_cost_usd, 6),
            "avg_cost_per_query": round(client.avg_cost_per_query, 6),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    def _get_client(self) -> MobulaClient:
        """Lazy-initialize the MobulaClient."""
        if self._client is None:
            self._client = MobulaClient(self._config)
        return self._client

    @staticmethod
    def _empty_features(cost_usd: float = 0.0) -> dict[str, Any]:
        """Return a dict with axiom_collected=False and zero cost."""
        return {
            "axiom_collected": False,
            "axiom_cost_usd": round(cost_usd, 6),
        }


# ===================================================================
# Module-level convenience
# ===================================================================

_service: Optional[AxiomService] = None


def get_axiom_service() -> AxiomService:
    """Return the global AxiomService singleton."""
    global _service
    if _service is None:
        _service = AxiomService()
    return _service


def reset_axiom_service() -> None:
    """Reset the cached service and client (useful for testing)."""
    global _service
    _service = None
    reset_mobula_client()
    reset_mobula_config()
