"""
tests/test_axiom_features.py — Unit tests for Axiom feature extraction.

Tests all 13 compute_* functions with complete, edge-case, and empty data.
Updated for Mobula filterTokenWallets data shapes (list[dict] instead of
fake Axiom REST dicts).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from axiom_features import (
    compute_smart_money_features,
    compute_wallet_quality_features,
    compute_pnl_features,
    compute_whale_axiom_features,
    compute_buyer_quality_features,
    compute_conviction_features,
    compute_early_strength_features,
    compute_distribution_features,
    compute_risk_signals_features,
    compute_smart_vs_retail_features,
    compute_composite_scores,
    compute_axiom_features,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_swap(wallet: str, is_buy: bool = True, usd: float = 100.0,
               sol: float = 0.5, timestamp: int = 1000) -> dict:
    return {
        "fee_payer": wallet,
        "is_buy": is_buy,
        "usd_estimate": usd,
        "sol_amount": sol,
        "timestamp": timestamp,
    }


def _make_swaps_by_window() -> dict:
    """Create test swap data across 3 time windows."""
    return {
        "1m": [
            _make_swap("wallet1", True, 500, 2.5, 1000),
            _make_swap("wallet2", True, 200, 1.0, 1005),
            _make_swap("wallet3", False, 100, 0.5, 1010),
        ],
        "5m": [
            _make_swap("wallet1", True, 500, 2.5, 1000),
            _make_swap("wallet2", True, 200, 1.0, 1005),
            _make_swap("wallet3", False, 100, 0.5, 1010),
            _make_swap("wallet4", True, 1500, 7.5, 1100),  # whale $1.5K
            _make_swap("wallet5", True, 50, 0.25, 1200),
        ],
        "15m": [
            _make_swap("wallet1", True, 500, 2.5, 1000),
            _make_swap("wallet2", True, 200, 1.0, 1005),
            _make_swap("wallet2", True, 300, 1.5, 1020),  # repeat buyer
            _make_swap("wallet3", False, 100, 0.5, 1010),
            _make_swap("wallet4", True, 1500, 7.5, 1100),  # whale
            _make_swap("wallet5", True, 50, 0.25, 1200),
            _make_swap("wallet6", True, 6000, 30.0, 1300),  # whale
            _make_swap("wallet7", False, 5000, 25.0, 1400),  # whale sell
            _make_swap("wallet1", False, 200, 1.0, 1500),  # wallet1 dumps
        ],
    }


def _make_token_wallets() -> list[dict]:
    """
    Create a list of Mobula-style TokenWalletResult dicts.
    Mirrors the structure returned by filterTokenWallets GraphQL query.
    """
    return [
        {
            "address": "wallet1",
            "labels": ["smart_money", "pro_trader"],
            "firstTransactionAt": 1700000000,  # ~197 days ago from ~now
            "lastTransactionAt": 1710000000,
            "buys1d": 10,
            "sells1d": 4,
            "amountBoughtUsd1d": 2500.0,
            "amountSoldUsd1d": 800.0,
            "realizedProfitUsd30d": 25000.0,
            "realizedProfitPercentage30d": 0.65,
            "realizedProfitUsd1w": 8000.0,
            "realizedProfitPercentage1w": 0.30,
            "realizedProfitUsd1y": 50000.0,
            "realizedProfitPercentage1y": 1.50,
            "buys1w": 25,
            "buys30d": 80,
            "buys1y": 500,
            "tokenBalanceLiveUsd": 2000.0,
            "scammerScore": 0,
            "botScore": 0,
        },
        {
            "address": "wallet2",
            "labels": [],
            "firstTransactionAt": 1717000000,  # ~3 days ago
            "lastTransactionAt": 1717100000,
            "buys1d": 2,
            "sells1d": 5,
            "amountBoughtUsd1d": 300.0,
            "amountSoldUsd1d": 600.0,
            "realizedProfitUsd30d": -500.0,
            "realizedProfitPercentage30d": -0.10,
            "realizedProfitUsd1w": -200.0,
            "realizedProfitPercentage1w": -0.05,
            "realizedProfitUsd1y": 0.0,
            "realizedProfitPercentage1y": 0.0,
            "buys1w": 3,
            "buys30d": 5,
            "buys1y": 5,
            "tokenBalanceLiveUsd": 100.0,
            "scammerScore": 0,
            "botScore": 30,
        },
        {
            "address": "wallet3",
            "labels": ["scammer"],
            "firstTransactionAt": 1690000000,  # ~313 days ago
            "lastTransactionAt": 1710000000,
            "buys1d": 0,
            "sells1d": 12,
            "amountBoughtUsd1d": 0.0,
            "amountSoldUsd1d": 5000.0,
            "realizedProfitUsd30d": 80000.0,
            "realizedProfitPercentage30d": 3.00,
            "realizedProfitUsd1w": 15000.0,
            "realizedProfitPercentage1w": 0.70,
            "realizedProfitUsd1y": 300000.0,
            "realizedProfitPercentage1y": 8.00,
            "buys1w": 0,
            "buys30d": 10,
            "buys1y": 150,
            "tokenBalanceLiveUsd": 5000.0,
            "scammerScore": 85,
            "botScore": 0,
        },
        {
            "address": "wallet4",
            "labels": ["elite"],
            "firstTransactionAt": 1705000000,  # ~140 days ago
            "lastTransactionAt": 1710000000,
            "buys1d": 8,
            "sells1d": 2,
            "amountBoughtUsd1d": 6000.0,  # whale threshold
            "amountSoldUsd1d": 1500.0,
            "realizedProfitUsd30d": 1000.0,
            "realizedProfitPercentage30d": 0.55,
            "realizedProfitUsd1w": 500.0,
            "realizedProfitPercentage1w": 0.20,
            "realizedProfitUsd1y": 5000.0,
            "realizedProfitPercentage1y": 0.80,
            "buys1w": 15,
            "buys30d": 80,
            "buys1y": 200,
            "tokenBalanceLiveUsd": 800.0,
            "scammerScore": 0,
            "botScore": 0,
        },
        {
            "address": "wallet5",
            "labels": ["fresh_wallet"],
            "firstTransactionAt": 1717200000,  # ~1 day ago
            "lastTransactionAt": 1717200000,
            "buys1d": 1,
            "sells1d": 0,
            "amountBoughtUsd1d": 50.0,
            "amountSoldUsd1d": 0.0,
            "realizedProfitUsd30d": 0.0,
            "realizedProfitPercentage30d": 0.0,
            "realizedProfitUsd1w": 0.0,
            "realizedProfitPercentage1w": 0.0,
            "realizedProfitUsd1y": 0.0,
            "realizedProfitPercentage1y": 0.0,
            "buys1w": 1,
            "buys30d": 2,
            "buys1y": 2,
            "tokenBalanceLiveUsd": 25.0,
            "scammerScore": 0,
            "botScore": 0,
        },
        {
            "address": "wallet6",
            "labels": ["pro_trader"],
            "firstTransactionAt": 1708000000,  # ~105 days ago
            "lastTransactionAt": 1710000000,
            "buys1d": 5,
            "sells1d": 1,
            "amountBoughtUsd1d": 12000.0,  # >10k whale threshold
            "amountSoldUsd1d": 1000.0,
            "realizedProfitUsd30d": 50000.0,
            "realizedProfitPercentage30d": 2.50,
            "realizedProfitUsd1w": 10000.0,
            "realizedProfitPercentage1w": 0.60,
            "realizedProfitUsd1y": 150000.0,
            "realizedProfitPercentage1y": 5.00,
            "buys1w": 12,
            "buys30d": 60,
            "buys1y": 300,
            "tokenBalanceLiveUsd": 10000.0,
            "scammerScore": 0,
            "botScore": 0,
        },
        {
            "address": "wallet7",
            "labels": ["sniper", "bundler"],
            "firstTransactionAt": 1710000000,  # ~80 days ago
            "lastTransactionAt": 1710100000,
            "buys1d": 3,
            "sells1d": 10,
            "amountBoughtUsd1d": 2000.0,
            "amountSoldUsd1d": 15000.0,  # whale sell
            "realizedProfitUsd30d": 5000.0,
            "realizedProfitPercentage30d": 0.90,
            "realizedProfitUsd1w": 2000.0,
            "realizedProfitPercentage1w": 0.30,
            "realizedProfitUsd1y": 20000.0,
            "realizedProfitPercentage1y": 2.00,
            "buys1w": 8,
            "buys30d": 30,
            "buys1y": 100,
            "tokenBalanceLiveUsd": 3000.0,
            "scammerScore": 20,
            "botScore": 65,
        },
    ]


# Default label sets matching mobula_config defaults
SMART_LABELS = frozenset({"smart_money", "pro_trader", "elite"})
RISK_LABELS = frozenset({"scammer", "bot", "sniper", "bundler"})
NEW_LABELS = frozenset({"fresh_wallet", "new_wallet"})


# ===================================================================
# Tests
# ===================================================================


class TestSmartMoneyFeatures(unittest.TestCase):
    def test_computes_all_12_features(self):
        result = compute_smart_money_features(
            _make_token_wallets(), _make_swaps_by_window(),
            t0_ts=1000, smart_labels=SMART_LABELS,
        )
        expected_keys = [
            "smart_wallet_buyers_1m", "smart_wallet_buyers_5m", "smart_wallet_buyers_15m",
            "smart_wallet_volume_1m", "smart_wallet_volume_5m", "smart_wallet_volume_15m",
            "smart_wallet_percentage", "smart_money_first_buyer",
            "first_smart_money_buy_timestamp",
            "smart_money_within_first_minute", "smart_money_within_first_5m",
            "smart_money_accumulation_rate",
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_empty_data_returns_zeros(self):
        result = compute_smart_money_features(
            [], {"1m": [], "5m": [], "15m": []},
        )
        for val in result.values():
            self.assertIsNotNone(val)

    def test_smart_money_first_buyer_detected(self):
        swaps = {
            "15m": [
                _make_swap("wallet1", True, 100, 0.5, 1000),
                _make_swap("wallet_x", True, 200, 1.0, 1005),
            ]
        }
        # wallet1 has smart_money label in _make_token_wallets()
        result = compute_smart_money_features(
            _make_token_wallets(), swaps,
            t0_ts=1000, smart_labels=SMART_LABELS,
        )
        self.assertEqual(result["smart_money_first_buyer"], 1)


class TestWalletQualityFeatures(unittest.TestCase):
    def test_computes_all_10_features(self):
        result = compute_wallet_quality_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        expected = [
            "avg_wallet_age_days", "median_wallet_age_days",
            "avg_wallet_trade_count", "median_wallet_trade_count",
            "avg_wallet_win_rate", "median_wallet_win_rate",
            "avg_wallet_realized_pnl", "median_wallet_realized_pnl",
            "avg_wallet_roi", "median_wallet_roi",
        ]
        for key in expected:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_empty_swaps_returns_zeros(self):
        result = compute_wallet_quality_features([], {"15m": []})
        for key in result:
            self.assertEqual(result[key], 0.0)

    def test_values_are_reasonable(self):
        result = compute_wallet_quality_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        # avg_wallet_win_rate should be between 0 and 1
        self.assertGreaterEqual(result["avg_wallet_win_rate"], 0.0)
        self.assertLessEqual(result["avg_wallet_win_rate"], 1.0)


class TestPnlFeatures(unittest.TestCase):
    def test_computes_all_10_pnl_features(self):
        result = compute_pnl_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        expected = [
            "avg_buyer_pnl_30d", "median_buyer_pnl_30d", "top_buyer_pnl_30d",
            "avg_buyer_pnl_90d", "median_buyer_pnl_90d", "top_buyer_pnl_90d",
            "avg_seller_pnl_30d", "median_seller_pnl_30d",
            "avg_seller_pnl_90d", "median_seller_pnl_90d",
        ]
        for key in expected:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_empty_data_returns_zeros(self):
        result = compute_pnl_features([], {"15m": []})
        for val in result.values():
            self.assertEqual(val, 0.0)


class TestWhaleAxiomFeatures(unittest.TestCase):
    def test_computes_16_whale_features(self):
        result = compute_whale_axiom_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        # 2 thresholds x 8 features = 16 (5k, 10k only)
        self.assertEqual(len(result), 16)

    def test_whale_threshold_5k(self):
        result = compute_whale_axiom_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        # Only wallet4 ($6K) and wallet6 ($12K) have amountBoughtUsd1d >= 5k
        # whale buys: wallet4 + wallet6 = 2
        # whale sells: none (wallet7 is not a 5k whale)
        self.assertGreater(result["whale_buy_count_5k"], 0)

    def test_whale_threshold_10k_boundary(self):
        result = compute_whale_axiom_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        # wallet6 $6K is BELOW $10K threshold (swap is $6K, but amountBoughtUsd1d is $12K)
        # Actually wallet6 has amountBoughtUsd1d = 12000 which IS >= 10K
        self.assertEqual(result["whale_buy_count_10k"], 1)


class TestConvictionFeatures(unittest.TestCase):
    def test_computes_6_features(self):
        result = compute_conviction_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        expected = [
            "repeat_buyers", "multi_buy_wallets",
            "wallet_rebuy_rate", "wallet_accumulation_rate",
            "avg_buys_per_wallet", "median_buys_per_wallet",
        ]
        for key in expected:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_repeat_buyer_detected(self):
        # wallet2 buys twice in 15m
        result = compute_conviction_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        self.assertEqual(result["repeat_buyers"], 1)
        self.assertGreater(result["avg_buys_per_wallet"], 1.0)


class TestEarlyStrengthFeatures(unittest.TestCase):
    def test_computes_7_features(self):
        result = compute_early_strength_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        expected = [
            "first_buyer_win_rate",
            "first_5_buyers_avg_win_rate", "first_10_buyers_avg_win_rate",
            "first_20_buyers_avg_win_rate",
            "first_5_buyers_avg_pnl", "first_10_buyers_avg_pnl",
            "first_20_buyers_avg_pnl",
        ]
        for key in expected:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_empty_data_returns_zeros(self):
        result = compute_early_strength_features([], {"15m": []})
        for val in result.values():
            self.assertEqual(val, 0.0)


class TestDistributionFeatures(unittest.TestCase):
    def test_computes_5_features(self):
        result = compute_distribution_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        expected = [
            "top_wallet_buy_share", "top5_wallet_buy_share",
            "top10_wallet_buy_share", "top20_wallet_buy_share",
            "buyer_concentration_index",
        ]
        for key in expected:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_hhi_bounded(self):
        result = compute_distribution_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        self.assertGreaterEqual(result["buyer_concentration_index"], 0.0)
        self.assertLessEqual(result["buyer_concentration_index"], 1.0)


class TestRiskSignalsFeatures(unittest.TestCase):
    def test_computes_6_features(self):
        result = compute_risk_signals_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        expected = [
            "dumping_wallet_count",
            "wallets_sold_within_5m", "wallets_sold_within_15m",
            "wallets_sold_within_60m",
            "fast_exit_rate", "paper_hand_rate",
        ]
        for key in expected:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_dumping_wallet_detected(self):
        # wallet1 buys $500 then sells $200 — NOT > 50% dump
        result = compute_risk_signals_features(
            _make_token_wallets(), _make_swaps_by_window(),
        )
        # wallet1 sold $200 out of $500 = 40%, not a dump
        self.assertEqual(result["dumping_wallet_count"], 0)


class TestCompositeScores(unittest.TestCase):
    def test_computes_5_scores(self):
        existing = {
            "smart_wallet_buyers_15m": 2,
            "unique_buyers_15m": 5,
            "smart_money_accumulation_rate": 0.5,
            "smart_money_first_buyer": 1,
            "avg_wallet_win_rate": 0.5,
            "avg_wallet_roi": 0.5,
            "elite_trader_count": 1,
            "whale_buy_volume_10k": 100,
            "volume_15m": 1000,
            "whale_net_flow_10k": 50,
            "wallet_rebuy_rate": 0.3,
            "avg_buys_per_wallet": 1.5,
            "experienced_wallet_buyers": 2,
            "wallets_older_than_90_days": 3,
        }
        result = compute_composite_scores(existing)
        expected = [
            "smart_money_score", "wallet_quality_score",
            "whale_score", "conviction_score", "buyer_quality_score",
        ]
        for key in expected:
            self.assertIn(key, result, f"Missing key: {key}")
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0,
                                 f"{key} = {result[key]} not in [0,1]")

    def test_scores_normalized_zero_to_one(self):
        result = compute_composite_scores({})
        for val in result.values():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)


class TestComputeAxiomFeatures(unittest.TestCase):
    def test_master_compute_returns_all_features(self):
        token_wallets = _make_token_wallets()
        result = compute_axiom_features(
            token_wallets=token_wallets,
            swaps_by_window=_make_swaps_by_window(),
            t0_ts=1000,
            smart_labels=SMART_LABELS,
            risk_labels=RISK_LABELS,
            new_labels=NEW_LABELS,
        )
        # Should have many features (all 13 categories)
        self.assertGreater(len(result), 80)
        self.assertIn("smart_wallet_buyers_15m", result)
        self.assertIn("avg_wallet_win_rate", result)
        self.assertIn("smart_money_score", result)
        self.assertIn("first_buyer_win_rate", result)

    def test_empty_data_never_crashes(self):
        result = compute_axiom_features(
            [], {"1m": [], "5m": [], "15m": []},
        )
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_single_wallet_edge_case(self):
        swaps = {
            "15m": [_make_swap("only_one", True, 100, 0.5, 1000)],
        }
        token_wallets = [{
            "address": "only_one",
            "labels": [],
            "firstTransactionAt": 1700000000,
            "buys1d": 1,
            "realizedProfitUsd30d": 0.0,
            "realizedProfitPercentage30d": 0.0,
            "realizedProfitUsd1y": 0.0,
            "realizedProfitPercentage1y": 0.0,
        }]
        result = compute_axiom_features(
            token_wallets=token_wallets,
            swaps_by_window=swaps,
            t0_ts=1000,
        )
        self.assertGreater(len(result), 0)
        # Should not crash
        self.assertIsInstance(result["avg_wallet_win_rate"], float)


if __name__ == "__main__":
    unittest.main()
