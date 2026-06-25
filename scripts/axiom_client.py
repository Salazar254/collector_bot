"""
scripts/axiom_client.py — Axiom API client with caching, rate limiting, and retry.

Provides wallet intelligence, smart money, whale behavior, and trader quality
signals from the Axiom API. Designed to be swappable for future providers.

Architecture:
  - In-memory TTL cache to avoid redundant API calls within a collection window
  - Token bucket rate limiter (configurable RPS)
  - Exponential backoff retry with jitter
  - Raw response storage for cost tracking and debugging
  - Automatic cost estimation per endpoint
  - Clean error handling — never crashes, returns empty defaults
"""

import json
import logging
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional

import requests

from axiom_config import AxiomConfig, get_axiom_config, reset_axiom_config

log = logging.getLogger(__name__)

# ===================================================================
# Token Bucket Rate Limiter
# ===================================================================


class TokenBucket:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate_per_second: float, burst: int = 5) -> None:
        self._rate = rate_per_second
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = Lock()

    def acquire(self) -> float:
        """
        Wait until a token is available and return the wait time.
        Returns 0.0 if no wait was needed.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0

            wait = (1.0 - self._tokens) / self._rate
            self._tokens = 0.0
            return wait


# ===================================================================
# TTL Cache
# ===================================================================


class TTLCache:
    """Simple in-memory TTL cache for API responses."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()

    def _make_key(self, *parts: str) -> str:
        return ":".join(str(p) for p in parts)

    def get(self, *key_parts: str) -> Optional[Any]:
        """Return cached value if not expired, else None."""
        key = self._make_key(*key_parts)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                return None
            return value

    def set(self, value: Any, *key_parts: str) -> None:
        """Store a value with current timestamp."""
        key = self._make_key(*key_parts)
        with self._lock:
            self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ===================================================================
# Axiom API Client
# ===================================================================


class AxiomClient:
    """
    Axiom API client with caching, rate limiting, retry, and cost tracking.

    All public methods return dicts — never raise on API errors.
    Missing data returns empty dicts with sensible defaults.

    Usage:
        client = AxiomClient(config)
        stats = client.get_token_wallet_stats(mint)
        smart = client.get_smart_money_activity(mint, window_seconds=900)
    """

    def __init__(
        self,
        config: Optional[AxiomConfig] = None,
        cache_ttl: Optional[int] = None,
    ) -> None:
        self._config = config or get_axiom_config()
        self._cache = TTLCache(cache_ttl or self._config.cache_ttl_seconds)
        self._bucket = TokenBucket(self._config.rate_limit_rps)
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._total_cost: float = 0.0
        self._request_count: int = 0
        self._cost_lock = Lock()

        # Default auth headers for all requests
        self._auth_params: dict[str, str] = {}
        if self._config.api_key:
            self._auth_params["api_key"] = self._config.api_key

    # ---------------------------------------------------------------
    # Public API Methods
    # ---------------------------------------------------------------

    def get_token_wallet_stats(self, mint: str) -> dict[str, Any]:
        """
        Fetch wallet intelligence for all traders of a token.
        Returns wallet-level stats: age, trade count, win rate, PnL, ROI.
        """
        return self._cached_request(
            "token_wallet_stats",
            f"{self._config.base_url}/token/{mint}/wallet-stats",
            params={**self._auth_params},
        )

    def get_smart_money_activity(
        self, mint: str, window_seconds: int = 900
    ) -> dict[str, Any]:
        """
        Fetch smart money buy/sell activity for a token within a time window.
        Uses configured smart money wallet list for classification.
        """
        params: dict[str, Any] = {
            **self._auth_params,
            "window_seconds": window_seconds,
        }
        if self._config.smart_money_wallets:
            params["wallets"] = ",".join(self._config.smart_money_wallets[:50])

        return self._cached_request(
            "smart_money_activity",
            f"{self._config.base_url}/token/{mint}/smart-money",
            params=params,
            cache_key_extra=str(window_seconds),
        )

    def get_wallet_profiles(self, wallets: list[str]) -> dict[str, Any]:
        """
        Batch-fetch wallet profiles: age, trade count, win rate, PnL, ROI.
        Max 50 wallets per request.
        """
        wallets_batch = wallets[:50]
        return self._cached_request(
            "wallet_profiles",
            f"{self._config.base_url}/wallets/profiles",
            params={**self._auth_params, "wallets": ",".join(wallets_batch)},
            cache_key_extra=":".join(sorted(wallets_batch[:10])),
        )

    def get_whale_transactions(
        self, mint: str, threshold_usd: float = 1000.0
    ) -> dict[str, Any]:
        """
        Fetch whale-sized transactions (>= threshold USD) for a token.
        Used at multiple thresholds: $1K, $5K, $10K.
        """
        params: dict[str, Any] = {
            **self._auth_params,
            "min_value_usd": threshold_usd,
        }
        return self._cached_request(
            "whale_transactions",
            f"{self._config.base_url}/token/{mint}/whale-transactions",
            params=params,
            cache_key_extra=str(int(threshold_usd)),
        )

    # ---------------------------------------------------------------
    # Cost & Stats
    # ---------------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        with self._cost_lock:
            return self._total_cost

    @property
    def request_count(self) -> int:
        with self._cost_lock:
            return self._request_count

    @property
    def avg_cost_per_request(self) -> float:
        with self._cost_lock:
            if self._request_count == 0:
                return 0.0
            return self._total_cost / self._request_count

    # ---------------------------------------------------------------
    # Internal — Request Execution
    # ---------------------------------------------------------------

    def _cached_request(
        self,
        request_type: str,
        url: str,
        params: Optional[dict[str, Any]] = None,
        cache_key_extra: str = "",
    ) -> dict[str, Any]:
        """
        Execute a cached API request with rate limiting and retry.

        Returns parsed JSON dict on success, empty dict on failure.
        Never raises — all errors are logged.
        """
        cache_key = f"{request_type}:{url}:{cache_key_extra}"

        # Check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            log.debug("Axiom cache HIT for %s", request_type)
            return cached

        # Rate limit
        wait = self._bucket.acquire()
        if wait > 0:
            log.debug("Axiom rate limit: waiting %.2fs", wait)
            time.sleep(wait)

        # Execute with retry
        result = self._retry_request(request_type, url, params)

        # Cache successful results
        if result:
            self._cache.set(result, cache_key)

        return result

    def _retry_request(
        self,
        request_type: str,
        url: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute request with exponential backoff retry."""
        max_retries = self._config.max_retries
        base_wait = 1.0

        for attempt in range(max_retries):
            start = time.monotonic()
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    timeout=self._config.request_timeout,
                )
                latency_ms = int((time.monotonic() - start) * 1000)

                # Track cost
                cost = AxiomConfig.cost_for_request(request_type)
                self._track_cost(cost)

                # Store raw response (fire-and-forget for performance)
                self._store_raw_response(
                    mint=_extract_mint_from_url(url),
                    request_type=request_type,
                    response_data=resp.text if resp.status_code < 500 else None,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    status_code=resp.status_code,
                )

                if resp.status_code == 429:
                    log.warning(
                        "Axiom rate limited (429) on %s (attempt %d)",
                        request_type, attempt + 1,
                    )
                    wait = base_wait * (2 ** attempt)
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    log.warning(
                        "Axiom server error %d on %s (attempt %d)",
                        resp.status_code, request_type, attempt + 1,
                    )
                    wait = base_wait * (2 ** attempt)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()

                data = resp.json()
                if isinstance(data, dict):
                    return data
                return {"data": data}

            except requests.exceptions.Timeout:
                log.warning(
                    "Axiom timeout on %s (attempt %d/%d)",
                    request_type, attempt + 1, max_retries,
                )
            except requests.exceptions.ConnectionError:
                log.warning(
                    "Axiom connection error on %s (attempt %d/%d)",
                    request_type, attempt + 1, max_retries,
                )
            except requests.exceptions.RequestException as e:
                log.warning(
                    "Axiom request error on %s (attempt %d/%d): %s",
                    request_type, attempt + 1, max_retries, e,
                )
            except Exception:
                log.exception(
                    "Unexpected error on %s (attempt %d/%d)",
                    request_type, attempt + 1, max_retries,
                )

            if attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt) + (0.1 * attempt)
                time.sleep(wait)

        log.error(
            "Axiom %s failed after %d retries — returning empty result",
            request_type, max_retries,
        )
        return {}

    def _track_cost(self, cost_usd: float) -> None:
        """Increment cost tracking counters."""
        with self._cost_lock:
            self._total_cost += cost_usd
            self._request_count += 1

    def _store_raw_response(
        self,
        mint: str,
        request_type: str,
        response_data: Optional[str],
        cost_usd: float,
        latency_ms: int,
        status_code: int,
    ) -> None:
        """
        Store raw API response in Supabase axiom_raw_responses table.
        Non-blocking — errors are silently logged.
        """
        try:
            from supabase import create_client

            supabase_url = __import__("os").environ.get("SUPABASE_URL", "")
            supabase_key = __import__("os").environ.get("SUPABASE_KEY", "")

            if not supabase_url or not supabase_key:
                return

            client = create_client(supabase_url, supabase_key)
            client.table("axiom_raw_responses").insert({
                "mint": mint,
                "request_type": request_type,
                "response_json": response_data,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "status_code": status_code,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        except Exception:
            # Raw response storage is best-effort — never crash the collector
            pass

    def reset_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()


# ===================================================================
# Helpers
# ===================================================================


def _extract_mint_from_url(url: str) -> str:
    """Extract mint address from Axiom API URL path."""
    parts = url.split("/")
    for i, part in enumerate(parts):
        if part in ("token", "tokens") and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


# ===================================================================
# Module-level convenience
# ===================================================================

_client: Optional[AxiomClient] = None
_client_lock = Lock()


def get_axiom_client() -> Optional[AxiomClient]:
    """
    Return the global AxiomClient singleton, or None if Axiom is disabled.
    """
    global _client
    config = get_axiom_config()
    if not config.is_enabled:
        return None

    with _client_lock:
        if _client is None:
            _client = AxiomClient(config)
        return _client


def reset_axiom_client() -> None:
    """Reset the cached client (useful for testing)."""
    global _client
    with _client_lock:
        if _client:
            _client.close()
            _client = None
    reset_axiom_config()
