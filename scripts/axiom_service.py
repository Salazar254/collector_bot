"""
scripts/axiom_service.py — Axiom service layer orchestrating API calls and feature extraction.

The AxiomService is the single entry point for Axiom data collection.
It handles:
  - Checking if Axiom is enabled
  - Fetching raw data via AxiomClient
  - Computing features via axiom_features
  - Tracking API costs
  - Returning a flat feature dict for Supabase upsert

Axiom is OPTIONAL — returns empty features if disabled or on failure.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from axiom_client import AxiomClient, get_axiom_client, reset_axiom_client
from axiom_config import AxiomConfig, get_axiom_config, reset_axiom_config
from axiom_features import compute_axiom_features

log = logging.getLogger(__name__)


class AxiomService:
    """
    Orchestrates the full Axiom collection flow for a single token.

    Usage:
        service = AxiomService()
        axiom_features = service.collect_for_token(mint, t0_ts, swaps_by_window)
        # axiom_features is a flat dict ready for Supabase upsert
    """

    def __init__(self, config: Optional[AxiomConfig] = None) -> None:
        self._config = config or get_axiom_config()
        self._client: Optional[AxiomClient] = None

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
        Collect all Axiom features for a single token.

        Args:
            mint: Token mint address
            t0_ts: Unix timestamp of token graduation (T0)
            swaps_by_window: {"1m": [parsed_swaps], "5m": [...], "15m": [...]}

        Returns:
            Flat dict with:
              - All Axiom feature values
              - axiom_collected: bool (True if Axiom data was fetched)
              - axiom_cost_usd: float (estimated API cost)
            Returns empty defaults if Axiom is disabled or fails.
        """
        if not self._config.is_enabled:
            return self._empty_features()

        cost_usd = 0.0
        axiom_data: dict[str, Any] = {}
        collected = False

        try:
            client = self._get_client()

            # Fetch smart money activity
            if self._config.enable_smart_money:
                sm_data = client.get_smart_money_activity(mint, window_seconds=900)
                if sm_data:
                    axiom_data["smart_money_activity"] = sm_data
                    collected = True
                cost_usd += AxiomConfig.cost_for_request("smart_money_activity")

            # Fetch wallet profiles for all buyers
            if self._config.enable_wallet_quality:
                buyer_wallets = self._extract_buyer_wallets(swaps_by_window)
                profiles = client.get_wallet_profiles(buyer_wallets)
                if profiles:
                    axiom_data["wallet_profiles"] = profiles
                    collected = True
                cost_usd += AxiomConfig.cost_for_request("wallet_profiles")

            # If nothing was collected, return empty
            if not collected or not axiom_data:
                log.debug("Axiom: no data collected for %s (all endpoints returned empty)", mint[:12])
                return self._empty_features()

            # Compute features from raw data
            features = compute_axiom_features(
                axiom_data=axiom_data,
                swaps_by_window=swaps_by_window,
                t0_ts=t0_ts,
                smart_wallets=self._config.smart_money_wallets,
            )

            # Add metadata
            features["axiom_collected"] = True
            features["axiom_cost_usd"] = round(cost_usd, 6)

            log.debug(
                "Axiom: %d features collected for %s (cost: $%.6f)",
                len(features) - 2, mint[:12], cost_usd,
            )

            return features

        except Exception:
            log.exception("Axiom collection failed for %s — returning empty features", mint[:12])
            return self._empty_features(cost_usd=cost_usd)

    def collect_for_mints_batch(
        self,
        mints: list[str],
        swaps_by_mint: dict[str, dict[str, list[dict]]],
        t0_ts_by_mint: dict[str, int],
    ) -> dict[str, dict[str, Any]]:
        """
        Collect Axiom features for multiple tokens in batch.
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
        Check if Axiom integration is healthy.
        Returns dict with status and diagnostic info.
        """
        if not self._config.is_enabled:
            return {
                "status": "disabled",
                "reason": "AXIOM_API_KEY not set",
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

        client = self._get_client()
        return {
            "status": "healthy",
            "requests_made": client.request_count,
            "total_cost_usd": round(client.total_cost_usd, 6),
            "avg_cost_per_request": round(client.avg_cost_per_request, 6),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    def _get_client(self) -> AxiomClient:
        """Lazy-initialize the AxiomClient."""
        if self._client is None:
            self._client = AxiomClient(self._config)
        return self._client

    @staticmethod
    def _extract_buyer_wallets(
        swaps_by_window: dict[str, list[dict]],
    ) -> list[str]:
        """Extract unique buyer wallet addresses from swap data."""
        all_swaps = swaps_by_window.get("15m", [])
        wallets = {
            s.get("fee_payer", "")
            for s in all_swaps
            if s.get("is_buy", True) and s.get("fee_payer", "")
        }
        return list(wallets)[:50]  # API limit

    @staticmethod
    def _empty_features(cost_usd: float = 0.0) -> dict[str, Any]:
        """Return a dict with axiom_collected=False and all features set to defaults."""
        features: dict[str, Any] = {
            "axiom_collected": False,
            "axiom_cost_usd": round(cost_usd, 6),
        }
        return features


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
    reset_axiom_client()
    reset_axiom_config()
