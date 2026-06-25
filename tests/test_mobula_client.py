"""
tests/test_mobula_client.py — Unit tests for MobulaClient (GraphQL API).

Tests: caching, rate limiting, retry, error handling, cost tracking,
GraphQL error responses, pagination, and disabled state.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from mobula_client import (
    MobulaClient,
    TokenBucket,
    TTLCache,
    get_mobula_client,
    reset_mobula_client,
    _extract_mint_from_context,
)
from mobula_config import MobulaConfig, reset_mobula_config


# ---------------------------------------------------------------------------
# Mock GraphQL response helpers
# ---------------------------------------------------------------------------

def _make_graphql_response(wallets=None, count=0, offset=0):
    """Build a valid Mobula filterTokenWallets GraphQL response."""
    return {
        "data": {
            "filterTokenWallets": {
                "results": wallets or [],
                "count": count,
                "offset": offset,
            }
        }
    }


def _make_wallet(address="wallet1", **overrides):
    """Build a single TokenWalletResult dict with realistic defaults."""
    wallet = {
        "address": address,
        "labels": [],
        "firstTransactionAt": 1700000000,
        "lastTransactionAt": 1710000000,
        "buys1d": 5,
        "sells1d": 2,
        "sellsAll1d": 2,
        "amountBoughtUsd1d": 500.0,
        "amountSoldUsd1d": 200.0,
        "amountSoldUsdAll1d": 200.0,
        "realizedProfitUsd1d": 50.0,
        "realizedProfitPercentage1d": 0.15,
        "buys1w": 15,
        "sells1w": 8,
        "sellsAll1w": 8,
        "amountBoughtUsd1w": 1500.0,
        "amountSoldUsd1w": 600.0,
        "amountSoldUsdAll1w": 600.0,
        "realizedProfitUsd1w": 200.0,
        "realizedProfitPercentage1w": 0.25,
        "buys30d": 50,
        "sells30d": 20,
        "sellsAll30d": 20,
        "amountBoughtUsd30d": 5000.0,
        "amountSoldUsd30d": 2000.0,
        "amountSoldUsdAll30d": 2000.0,
        "realizedProfitUsd30d": 800.0,
        "realizedProfitPercentage30d": 0.40,
        "buys1y": 200,
        "sells1y": 150,
        "sellsAll1y": 150,
        "amountBoughtUsd1y": 25000.0,
        "amountSoldUsd1y": 20000.0,
        "amountSoldUsdAll1y": 20000.0,
        "realizedProfitUsd1y": 5000.0,
        "realizedProfitPercentage1y": 1.20,
        "tokenBalance": 10000.0,
        "tokenBalanceLive": 10000.0,
        "tokenBalanceLiveUsd": 500.0,
        "scammerScore": 0,
        "botScore": 0,
    }
    wallet.update(overrides)
    return wallet


# ===================================================================
# Token Bucket Tests
# ===================================================================


class TestTokenBucket(unittest.TestCase):
    """Token bucket rate limiter tests."""

    def test_acquire_when_full(self):
        bucket = TokenBucket(rate_per_second=10.0, burst=5)
        wait = bucket.acquire()
        self.assertEqual(wait, 0.0)

    def test_acquire_depletes_tokens(self):
        bucket = TokenBucket(rate_per_second=1.0, burst=2)
        self.assertEqual(bucket.acquire(), 0.0)
        self.assertEqual(bucket.acquire(), 0.0)
        wait = bucket.acquire()
        self.assertGreater(wait, 0.0)

    def test_refill_over_time(self):
        bucket = TokenBucket(rate_per_second=100.0, burst=5)
        for _ in range(5):
            bucket.acquire()
        time.sleep(0.05)
        wait = bucket.acquire()
        self.assertEqual(wait, 0.0)


# ===================================================================
# TTL Cache Tests
# ===================================================================


class TestTTLCache(unittest.TestCase):
    """TTL cache tests."""

    def test_set_and_get(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set({"data": "test"}, "key1")
        result = cache.get("key1")
        self.assertEqual(result, {"data": "test"})

    def test_expired_entry(self):
        cache = TTLCache(ttl_seconds=0.01)
        cache.set({"data": "test"}, "key1")
        time.sleep(0.02)
        result = cache.get("key1")
        self.assertIsNone(result)

    def test_miss_returns_none(self):
        cache = TTLCache(ttl_seconds=60)
        result = cache.get("nonexistent")
        self.assertIsNone(result)

    def test_clear(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set({"data": "test"}, "key1")
        cache.clear()
        self.assertEqual(len(cache), 0)

    def test_multi_part_key(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set({"data": "test"}, "ns", "key", "extra")
        result = cache.get("ns", "key", "extra")
        self.assertEqual(result, {"data": "test"})


# ===================================================================
# MobulaClient Initialization Tests
# ===================================================================


class TestMobulaClientInit(unittest.TestCase):
    """MobulaClient initialization tests."""

    def setUp(self):
        reset_mobula_config()
        reset_mobula_client()
        os.environ["MOBULA_API_KEY"] = "test-mobula-key"

    def tearDown(self):
        reset_mobula_config()
        reset_mobula_client()
        os.environ.pop("MOBULA_API_KEY", None)

    def test_client_initializes_with_config(self):
        config = MobulaConfig()
        client = MobulaClient(config)
        self.assertIsNotNone(client)
        self.assertEqual(client._config.api_key, "test-mobula-key")

    def test_get_mobula_client_returns_none_when_disabled(self):
        os.environ.pop("MOBULA_API_KEY", None)
        reset_mobula_config()
        client = get_mobula_client()
        self.assertIsNone(client)


# ===================================================================
# MobulaClient GraphQL Request Tests
# ===================================================================


class TestMobulaClientRequests(unittest.TestCase):
    """MobulaClient GraphQL request tests."""

    def setUp(self):
        reset_mobula_config()
        reset_mobula_client()
        os.environ["MOBULA_API_KEY"] = "test-mobula-key"
        self.config = MobulaConfig()
        self.config.rate_limit_rps = 1000.0  # effectively no rate limit
        self.config.max_retries = 2
        self.config.request_timeout = 5

    def tearDown(self):
        reset_mobula_config()
        reset_mobula_client()
        os.environ.pop("MOBULA_API_KEY", None)

    @patch("mobula_client.requests.Session.post")
    def test_fetch_token_wallets_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_graphql_response(
            wallets=[_make_wallet("w1"), _make_wallet("w2")],
            count=2,
            offset=0,
        )
        mock_response.text = '{"data": {"filterTokenWallets": {"results": [], "count": 2}}}'
        mock_post.return_value = mock_response

        client = MobulaClient(self.config)
        wallets = client.fetch_token_wallets("mintABC123")

        self.assertEqual(len(wallets), 2)
        self.assertEqual(wallets[0]["address"], "w1")

    @patch("mobula_client.requests.Session.post")
    def test_cached_request_hits_cache(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_graphql_response(
            wallets=[_make_wallet("w_cached")],
            count=1,
            offset=0,
        )
        mock_response.text = "{}"
        mock_post.return_value = mock_response

        client = MobulaClient(self.config)
        result1 = client.fetch_token_wallets("mint_cache")
        result2 = client.fetch_token_wallets("mint_cache")

        # Should only call API once (second call hits cache for offset 0)
        self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(result1, result2)

    @patch("mobula_client.requests.Session.post")
    def test_retry_on_500(self, mock_post):
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = _make_graphql_response(
            wallets=[_make_wallet("w_retry")],
            count=1,
            offset=0,
        )
        mock_ok.text = "{}"
        mock_post.side_effect = [mock_fail, mock_ok]

        client = MobulaClient(self.config)
        wallets = client.fetch_token_wallets("mint_retry")

        self.assertEqual(len(wallets), 1)
        self.assertGreaterEqual(mock_post.call_count, 2)

    @patch("mobula_client.requests.Session.post")
    def test_all_retries_exhausted(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        client = MobulaClient(self.config)
        wallets = client.fetch_token_wallets("mint_fail")

        # Should return empty list after exhausting retries
        self.assertEqual(wallets, [])

    @patch("mobula_client.requests.Session.post")
    def test_cost_tracking(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_graphql_response(
            wallets=[_make_wallet("w_cost")],
            count=1,
            offset=0,
        )
        mock_response.text = "{}"
        mock_post.return_value = mock_response

        client = MobulaClient(self.config)
        client.fetch_token_wallets("mint_cost")

        self.assertGreater(client.total_cost_usd, 0)
        self.assertEqual(client.query_count, 1)

    @patch("mobula_client.requests.Session.post")
    def test_timeout_handling(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()

        client = MobulaClient(self.config)
        wallets = client.fetch_token_wallets("mint_timeout")

        self.assertEqual(wallets, [])

    @patch("mobula_client.requests.Session.post")
    def test_rate_limit_429_retry(self, mock_post):
        mock_rate_limited = MagicMock()
        mock_rate_limited.status_code = 429
        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = _make_graphql_response(
            wallets=[_make_wallet("w_rl")],
            count=1,
            offset=0,
        )
        mock_ok.text = "{}"
        mock_post.side_effect = [mock_rate_limited, mock_ok]

        client = MobulaClient(self.config)
        wallets = client.fetch_token_wallets("mint_ratelimit")

        self.assertEqual(len(wallets), 1)
        self.assertGreaterEqual(mock_post.call_count, 2)

    @patch("mobula_client.requests.Session.post")
    def test_graphql_errors_in_response(self, mock_post):
        """GraphQL-level errors (not HTTP errors) are handled gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "errors": [{"message": "Invalid token ID format"}],
            "data": None,
        }
        mock_post.return_value = mock_response

        client = MobulaClient(self.config)
        wallets = client.fetch_token_wallets("bad_mint")

        # Should return empty since no data + errors
        self.assertEqual(wallets, [])

    @patch("mobula_client.requests.Session.post")
    def test_graphql_partial_data_with_errors(self, mock_post):
        """When errors exist but data is still present, return the data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "errors": [{"message": "Some wallets could not be resolved"}],
            "data": {
                "filterTokenWallets": {
                    "results": [_make_wallet("w_partial")],
                    "count": 1,
                    "offset": 0,
                }
            },
        }
        mock_post.return_value = mock_response

        client = MobulaClient(self.config)
        wallets = client.fetch_token_wallets("mint_partial")

        self.assertEqual(len(wallets), 1)
        self.assertEqual(wallets[0]["address"], "w_partial")

    @patch("mobula_client.requests.Session.post")
    def test_pagination_single_page(self, mock_post):
        """When count <= page size, only one request is made."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_graphql_response(
            wallets=[_make_wallet(f"w{i}") for i in range(3)],
            count=3,
            offset=0,
        )
        mock_response.text = "{}"
        mock_post.return_value = mock_response

        client = MobulaClient(self.config)
        wallets = client.fetch_token_wallets("mint_small")

        self.assertEqual(len(wallets), 3)
        self.assertEqual(mock_post.call_count, 1)

    @patch("mobula_client.requests.Session.post")
    def test_pagination_multi_page(self, mock_post):
        """When count > page size, paginate until all fetched."""
        page1 = MagicMock()
        page1.status_code = 200
        page1.json.return_value = _make_graphql_response(
            wallets=[_make_wallet(f"p1_w{i}") for i in range(200)],
            count=350,
            offset=0,
        )
        page1.text = "{}"

        page2 = MagicMock()
        page2.status_code = 200
        page2.json.return_value = _make_graphql_response(
            wallets=[_make_wallet(f"p2_w{i}") for i in range(150)],
            count=350,
            offset=200,
        )
        page2.text = "{}"

        mock_post.side_effect = [page1, page2]

        # Need a client with smaller cache to not cache page1
        config = MobulaConfig()
        config.rate_limit_rps = 1000.0
        config.max_retries = 2
        client = MobulaClient(config)
        client._cache.clear()

        wallets = client.fetch_token_wallets("mint_large", max_pages=3)

        self.assertEqual(len(wallets), 350)


# ===================================================================
# Context Extraction Tests
# ===================================================================


class TestExtractMintFromContext(unittest.TestCase):
    """Mint extraction from cache key context tests."""

    def test_extract_mint_from_context(self):
        context = "mintABC123:0"
        mint = _extract_mint_from_context(context)
        self.assertEqual(mint, "mintABC123")

    def test_extract_unknown_returns_unknown(self):
        context = ""
        mint = _extract_mint_from_context(context)
        self.assertEqual(mint, "unknown")


if __name__ == "__main__":
    unittest.main()
