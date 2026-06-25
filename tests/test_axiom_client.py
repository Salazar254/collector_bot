"""
tests/test_axiom_client.py — Unit tests for AxiomClient.

Tests: caching, rate limiting, retry, error handling, cost tracking,
and disabled state.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

# Ensure scripts is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from axiom_client import (
    AxiomClient,
    TokenBucket,
    TTLCache,
    get_axiom_client,
    reset_axiom_client,
    _extract_mint_from_url,
)
from axiom_config import AxiomConfig, reset_axiom_config


class TestTokenBucket(unittest.TestCase):
    """Token bucket rate limiter tests."""

    def test_acquire_when_full(self):
        bucket = TokenBucket(rate_per_second=10.0, burst=5)
        # Should return 0.0 when tokens are available
        wait = bucket.acquire()
        self.assertEqual(wait, 0.0)

    def test_acquire_depletes_tokens(self):
        bucket = TokenBucket(rate_per_second=1.0, burst=2)
        self.assertEqual(bucket.acquire(), 0.0)
        self.assertEqual(bucket.acquire(), 0.0)
        # Third acquire should require waiting
        wait = bucket.acquire()
        self.assertGreater(wait, 0.0)

    def test_refill_over_time(self):
        bucket = TokenBucket(rate_per_second=100.0, burst=5)
        # Exhaust all tokens
        for _ in range(5):
            bucket.acquire()
        time.sleep(0.05)
        wait = bucket.acquire()
        self.assertEqual(wait, 0.0)


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


class TestAxiomClientInit(unittest.TestCase):
    """AxiomClient initialization tests."""

    def setUp(self):
        reset_axiom_config()
        reset_axiom_client()
        os.environ["AXIOM_API_KEY"] = "test-key-123"

    def tearDown(self):
        reset_axiom_config()
        reset_axiom_client()
        os.environ.pop("AXIOM_API_KEY", None)

    def test_client_initializes_with_config(self):
        config = AxiomConfig()
        client = AxiomClient(config)
        self.assertIsNotNone(client)
        self.assertEqual(client._config.api_key, "test-key-123")

    def test_get_axiom_client_returns_none_when_disabled(self):
        os.environ.pop("AXIOM_API_KEY", None)
        reset_axiom_config()
        client = get_axiom_client()
        self.assertIsNone(client)


class TestAxiomClientRequests(unittest.TestCase):
    """AxiomClient HTTP request tests."""

    def setUp(self):
        reset_axiom_config()
        reset_axiom_client()
        os.environ["AXIOM_API_KEY"] = "test-key-123"
        # Create config with very fast rate limit for testing
        self.config = AxiomConfig()
        self.config.rate_limit_rps = 1000.0  # effectively no rate limit

    def tearDown(self):
        reset_axiom_config()
        reset_axiom_client()
        os.environ.pop("AXIOM_API_KEY", None)

    @patch("axiom_client.requests.Session.get")
    def test_get_token_wallet_stats_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "profiles": {"wallet1": {"trade_count": 10, "win_rate": 0.6}}
        }
        mock_response.text = '{"profiles": {}}'
        mock_get.return_value = mock_response

        client = AxiomClient(self.config)
        result = client.get_token_wallet_stats("mint123")

        self.assertIsInstance(result, dict)
        self.assertIn("profiles", result)

    @patch("axiom_client.requests.Session.get")
    def test_cached_request_returns_cached(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cached": True}
        mock_response.text = '{"cached": true}'
        mock_get.return_value = mock_response

        client = AxiomClient(self.config)
        result1 = client.get_token_wallet_stats("mint456")
        result2 = client.get_token_wallet_stats("mint456")

        # Should only call API once
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(result1, result2)

    @patch("axiom_client.requests.Session.get")
    def test_retry_on_500(self, mock_get):
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {"ok": True}
        mock_ok.text = '{"ok": true}'

        mock_get.side_effect = [mock_fail, mock_ok]

        client = AxiomClient(self.config)
        result = client.get_token_wallet_stats("mint789")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_get.call_count, 2)

    @patch("axiom_client.requests.Session.get")
    def test_all_retries_exhausted(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        client = AxiomClient(self.config)
        result = client.get_token_wallet_stats("mint999")

        # Should return empty dict after exhausting retries
        self.assertEqual(result, {})

    @patch("axiom_client.requests.Session.get")
    def test_cost_tracking(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "ok"}
        mock_response.text = '{"data": "ok"}'
        mock_get.return_value = mock_response

        client = AxiomClient(self.config)
        client.get_token_wallet_stats("mint1")
        client.get_smart_money_activity("mint1")

        self.assertGreater(client.total_cost_usd, 0)
        self.assertEqual(client.request_count, 2)

    @patch("axiom_client.requests.Session.get")
    def test_timeout_handling(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        client = AxiomClient(self.config)
        result = client.get_token_wallet_stats("mint_timeout")

        self.assertEqual(result, {})


class TestExtractMintFromUrl(unittest.TestCase):
    """URL mint extraction tests."""

    def test_extract_mint_from_token_url(self):
        url = "https://api.axiom.trade/v1/token/AbC123DeF456/wallet-stats"
        mint = _extract_mint_from_url(url)
        self.assertEqual(mint, "AbC123DeF456")

    def test_extract_unknown_returns_unknown(self):
        url = "https://example.com/other"
        mint = _extract_mint_from_url(url)
        self.assertEqual(mint, "unknown")


if __name__ == "__main__":
    unittest.main()
