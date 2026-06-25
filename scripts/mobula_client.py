"""
scripts/mobula_client.py — Mobula GraphQL API client with caching, rate limiting, and retry.

Provides wallet-intelligence signals via Mobula's filterTokenWallets GraphQL query.
A single query replaces the 4 fake REST endpoints previously in axiom_client.py.

Architecture:
  - In-memory TTL cache to avoid redundant API calls within a collection window
  - Token bucket rate limiter (configurable RPS)
  - Exponential backoff retry with jitter
  - Raw response storage for cost tracking and debugging
  - Automatic cost estimation per query
  - Clean error handling — never crashes, returns empty defaults

GraphQL endpoint: https://graphql.mobula.io/graphql
Auth: Authorization header with API key
"""

import json
import logging
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional

import requests

from mobula_config import MobulaConfig, get_mobula_config, reset_mobula_config

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
# GraphQL Query Templates
# ===================================================================

FILTER_TOKEN_WALLETS_QUERY = """
query FilterTokenWallets($input: FilterTokenWalletsInput!) {
  filterTokenWallets(input: $input) {
    results {
      address
      labels
      firstTransactionAt
      lastTransactionAt
      buys1d
      sells1d
      sellsAll1d
      amountBoughtUsd1d
      amountSoldUsd1d
      amountSoldUsdAll1d
      realizedProfitUsd1d
      realizedProfitPercentage1d
      buys1w
      sells1w
      sellsAll1w
      amountBoughtUsd1w
      amountSoldUsd1w
      amountSoldUsdAll1w
      realizedProfitUsd1w
      realizedProfitPercentage1w
      buys30d
      sells30d
      sellsAll30d
      amountBoughtUsd30d
      amountSoldUsd30d
      amountSoldUsdAll30d
      realizedProfitUsd30d
      realizedProfitPercentage30d
      buys1y
      sells1y
      sellsAll1y
      amountBoughtUsd1y
      amountSoldUsd1y
      amountSoldUsdAll1y
      realizedProfitUsd1y
      realizedProfitPercentage1y
      tokenBalance
      tokenBalanceLive
      tokenBalanceLiveUsd
      scammerScore
      botScore
    }
    count
    offset
  }
}
"""


# ===================================================================
# Mobula GraphQL Client
# ===================================================================


class MobulaClient:
    """
    Mobula GraphQL API client with caching, rate limiting, retry, and cost tracking.

    All public methods return lists/dicts — never raise on API errors.
    Missing data returns empty lists with sensible defaults.

    Usage:
        client = MobulaClient(config)
        wallets = client.fetch_token_wallets(mint)
    """

    REQUEST_TYPE = "filterTokenWallets"

    def __init__(
        self,
        config: Optional[MobulaConfig] = None,
        cache_ttl: Optional[int] = None,
    ) -> None:
        self._config = config or get_mobula_config()
        self._cache = TTLCache(cache_ttl or self._config.cache_ttl_seconds)
        self._bucket = TokenBucket(self._config.rate_limit_rps)
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": self._config.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._total_cost: float = 0.0
        self._query_count: int = 0
        self._cost_lock = Lock()

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def fetch_token_wallets(
        self,
        mint: str,
        max_pages: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Fetch all wallets that traded a given token via filterTokenWallets.

        Paginates through results (max 200 per page) to collect all wallets.
        Returns list of TokenWalletResult dicts with labels, windowed stats,
        PnL, volume, scammer/bot scores.

        Args:
            mint: Solana token mint address
            max_pages: Maximum number of paginated pages to fetch (5 * 200 = 1000 wallets)

        Returns:
            List of wallet result dicts. Empty list on failure.
        """
        token_id = f"{mint}:{self._config.solana_network_id}"
        all_wallets: list[dict[str, Any]] = []
        offset = 0
        limit = self._config.max_wallets_per_page

        for page in range(max_pages):
            variables = {
                "input": {
                    "tokenIds": [token_id],
                    "limit": limit,
                    "offset": offset,
                }
            }

            result = self._cached_graphql_request(
                FILTER_TOKEN_WALLETS_QUERY,
                variables,
                cache_key_extra=f"{mint}:{offset}",
            )

            if not result:
                break

            data = result.get("data", {})
            ftw = data.get("filterTokenWallets", {})
            results = ftw.get("results", []) or []
            count = ftw.get("count", 0)

            all_wallets.extend(results)

            # Stop if we've fetched all wallets
            offset += len(results)
            if offset >= count:
                break

        return all_wallets

    # ---------------------------------------------------------------
    # Cost & Stats
    # ---------------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        with self._cost_lock:
            return self._total_cost

    @property
    def query_count(self) -> int:
        with self._cost_lock:
            return self._query_count

    @property
    def avg_cost_per_query(self) -> float:
        with self._cost_lock:
            if self._query_count == 0:
                return 0.0
            return self._total_cost / self._query_count

    # ---------------------------------------------------------------
    # Internal — GraphQL Request Execution
    # ---------------------------------------------------------------

    def _cached_graphql_request(
        self,
        query: str,
        variables: dict[str, Any],
        cache_key_extra: str = "",
    ) -> dict[str, Any]:
        """
        Execute a cached GraphQL request with rate limiting and retry.

        Returns parsed JSON response dict on success, empty dict on failure.
        Never raises — all errors are logged.
        """
        cache_key = f"{self.REQUEST_TYPE}:{cache_key_extra}"

        # Check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            log.debug("Mobula cache HIT for %s", cache_key_extra)
            return cached

        # Rate limit
        wait = self._bucket.acquire()
        if wait > 0:
            log.debug("Mobula rate limit: waiting %.2fs", wait)
            time.sleep(wait)

        # Execute with retry
        result = self._retry_graphql_request(query, variables, cache_key_extra)

        # Cache successful results
        if result:
            self._cache.set(result, cache_key)

        return result

    def _retry_graphql_request(
        self,
        query: str,
        variables: dict[str, Any],
        context: str = "",
    ) -> dict[str, Any]:
        """Execute GraphQL POST with exponential backoff retry."""
        max_retries = self._config.max_retries
        base_wait = 1.0

        for attempt in range(max_retries):
            start = time.monotonic()
            try:
                resp = self._session.post(
                    self._config.graphql_url,
                    json={"query": query, "variables": variables},
                    timeout=self._config.request_timeout,
                )
                latency_ms = int((time.monotonic() - start) * 1000)

                # Track cost
                cost = MobulaConfig.cost_for_query()
                self._track_cost(cost)

                # Store raw response (best-effort)
                self._store_raw_response(
                    mint=_extract_mint_from_context(context),
                    response_data=(
                        resp.text if resp.status_code < 500 else None
                    ),
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    status_code=resp.status_code,
                )

                if resp.status_code == 429:
                    log.warning(
                        "Mobula rate limited (429) on %s (attempt %d)",
                        context, attempt + 1,
                    )
                    wait = base_wait * (2 ** attempt)
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    log.warning(
                        "Mobula server error %d on %s (attempt %d)",
                        resp.status_code, context, attempt + 1,
                    )
                    wait = base_wait * (2 ** attempt)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                body = resp.json()

                # Check for GraphQL-level errors
                gql_errors = body.get("errors", [])
                if gql_errors:
                    error_msgs = [
                        e.get("message", str(e)) for e in gql_errors
                    ]
                    log.warning(
                        "Mobula GraphQL errors on %s: %s",
                        context, "; ".join(error_msgs[:3]),
                    )
                    # Non-fatal GraphQL errors — return partial data if available
                    if body.get("data"):
                        return body
                    if attempt < max_retries - 1:
                        wait = base_wait * (2 ** attempt)
                        time.sleep(wait)
                        continue
                    return {}

                return body

            except requests.exceptions.Timeout:
                log.warning(
                    "Mobula timeout on %s (attempt %d/%d)",
                    context, attempt + 1, max_retries,
                )
            except requests.exceptions.ConnectionError:
                log.warning(
                    "Mobula connection error on %s (attempt %d/%d)",
                    context, attempt + 1, max_retries,
                )
            except requests.exceptions.RequestException as e:
                log.warning(
                    "Mobula request error on %s (attempt %d/%d): %s",
                    context, attempt + 1, max_retries, e,
                )
            except Exception:
                log.exception(
                    "Unexpected error on %s (attempt %d/%d)",
                    context, attempt + 1, max_retries,
                )

            if attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt) + (0.1 * attempt)
                time.sleep(wait)

        log.error(
            "Mobula %s failed after %d retries — returning empty result",
            context, max_retries,
        )
        return {}

    def _track_cost(self, cost_usd: float) -> None:
        """Increment cost tracking counters."""
        with self._cost_lock:
            self._total_cost += cost_usd
            self._query_count += 1

    def _store_raw_response(
        self,
        mint: str,
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
                "request_type": self.REQUEST_TYPE,
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


def _extract_mint_from_context(context: str) -> str:
    """Extract mint address from cache key context (format: 'mint:offset')."""
    if ":" in context:
        return context.split(":")[0]
    return "unknown"


# ===================================================================
# Module-level convenience
# ===================================================================

_client: Optional[MobulaClient] = None
_client_lock = Lock()


def get_mobula_client() -> Optional[MobulaClient]:
    """
    Return the global MobulaClient singleton, or None if Mobula is disabled.
    """
    global _client
    config = get_mobula_config()
    if not config.is_enabled:
        return None

    with _client_lock:
        if _client is None:
            _client = MobulaClient(config)
        return _client


def reset_mobula_client() -> None:
    """Reset the cached client (useful for testing)."""
    global _client
    with _client_lock:
        if _client:
            _client.close()
            _client = None
    reset_mobula_config()
