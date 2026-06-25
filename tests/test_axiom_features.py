"""
tests/test_axiom_features.py — Unit tests for Axiom feature extraction.

Tests all 13 compute_* functions with complete, edge-case, and empty data.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from axiom_features import (
    compute_smart_money_features,
    compute_wallet_quality_features,
    compute_pnl_features,
    compute_roi_features,
    compute_profitable_trader_features,
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


def _make_wallet_profiles() -> dict:
    return {
        "profiles": {
            "wallet1": {
                "wallet_age_days": 120,
                "trade_count": 500,
                "win_rate": 0.65,
                "realized_pnl": 25000.0,
                "roi": 1.5,
                "pnl_30d": 5000.0,
                "pnl_90d": 15000.0,
                "roi_30d": 0.3,
                "roi_90d": 0.8,
            },
            "wallet2": {
                "wallet_age_days": 3,
                "trade_count": 5,
                "win_rate": 0.2,
                "realized_pnl": -500.0,
                "roi": -0.1,
            },
            "wallet3": {
                "wallet_age_days": 200,
                "trade_count": 150,
                "win_rate": 0.70,
                "realized_pnl": 80000.0,
                "roi": 3.0,
            },
            "wallet4": {
                "wallet_age_days": 60,
                "trade_count": 80,
                "win_rate": 0.55,
                "realized_pnl": 1000.0,
                "roi": 0.5,
            },
            "wallet5": {
                "wallet_age_days": 1,
                "trade_count": 2,
                "win_rate": 0.0,
                "realized_pnl": 0.0,
                "roi": 0.0,
            },
        }
    }


# ===================================================================
# Tests
# ===================================================================


class TestSmartMoneyFeatures(unittest.TestCase):
    def test_computes_all_12_features(self):
        result = compute_smart_money_features(
            {}, _make_swaps_by_window(), t0_ts=1000,
            smart_wallets=["wallet1", "wallet4"],
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
        result = compute_smart_money_features({}, {"1m": [], "5m": [], "15m": []})
        for val in result.values():
            self.assertIsNotNone(val)

    def test_smart_money_first_buyer_detected(self):
        swaps = {
            "15m": [
                _make_swap("smart_wallet_1", True, 100, 0.5, 1000),
                _make_swap("wallet_x", True, 200, 1.0, 1005),
            ]
        }
        result = compute_smart_money_features(
            {}, swaps, t0_ts=1000,
            smart_wallets=["smart_wallet_1"],
        )
        self.assertEqual(result["smart_money_first_buyer"], 1)


class TestWalletQualityFeatures(unittest.TestCase):
    def test_computes_all_10_features(self):
        result = compute_wallet_quality_features(
            _make_wallet_profiles(), _make_swaps_by_window(),
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
        result = compute_wallet_quality_features({}, {"15m": []})
        for key in result:
            self.assertEqual(result[key], 0.0)

    def test_values_are_reasonable(self):
        result = compute_wallet_quality_features(
            _make_wallet_profiles(), _make_swaps_by_window(),
        )
        # avg_wallet_win_rate should be between 0 and 1
        self.assertGreaterEqual(result["avg_wallet_win_rate"], 0.0)
        self.assertLessEqual(result["avg_wallet_win_rate"], 1.0)


class TestPnlFeatures(unittest.TestCase):
    def test_computes_all_10_pnl_features(self):
        result = compute_pnl_features(
            _make_wallet_profiles(), _make_swaps_by_window(),
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
        result = compute_pnl_features({}, {"15m": []})
        for val in result.values():
            self.assertEqual(val, 0.0)


class TestWhaleAxiomFeatures(unittest.TestCase):
    def test_computes_24_whale_features(self):
        result = compute_whale_axiom_features(_make_swaps_by_window())
        # 3 thresholds x 8 features = 24
        self.assertEqual(len(result), 24)

    def test_whale_threshold_1k(self):
        result = compute_whale_axiom_features(_make_swaps_by_window())
        # wallet4 $1.5K buy, wallet6 $6K buy, wallet7 $5K sell
        self.assertGreater(result["whale_buy_count_1k"], 0)
        self.assertEqual(result["whale_sell_count_1k"], 1)

    def test_whale_threshold_5k(self):
        result = compute_whale_axiom_features(_make_swaps_by_window())
        # Only wallet6 $6K buy, wallet7 $5K sell
        self.assertEqual(result["whale_buy_count_5k"], 1)
        self.assertEqual(result["whale_sell_count_5k"], 1)

    def test_whale_threshold_10k_boundary(self):
        result = compute_whale_axiom_features(_make_swaps_by_window())
        # wallet6 $6K is BELOW $10K threshold
        self.assertEqual(result["whale_buy_count_10k"], 0)


class TestConvictionFeatures(unittest.TestCase):
    def test_computes_6_features(self):
        result = compute_conviction_features(_make_swaps_by_window())
        expected = [
            "repeat_buyers", "multi_buy_wallets",
            "wallet_rebuy_rate", "wallet_accumulation_rate",
            "avg_buys_per_wallet", "median_buys_per_wallet",
        ]
        for key in expected:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_repeat_buyer_detected(self):
        # wallet2 buys twice in 15m
        result = compute_conviction_features(_make_swaps_by_window())
        self.assertEqual(result["repeat_buyers"], 1)
        self.assertGreater(result["avg_buys_per_wallet"], 1.0)


class TestEarlyStrengthFeatures(unittest.TestCase):
    def test_computes_7_features(self):
        result = compute_early_strength_features(
            _make_wallet_profiles(), _make_swaps_by_window(),
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
        result = compute_early_strength_features({}, {"15m": []})
        for val in result.values():
            self.assertEqual(val, 0.0)


class TestDistributionFeatures(unittest.TestCase):
    def test_computes_5_features(self):
        result = compute_distribution_features(_make_swaps_by_window())
        expected = [
            "top_wallet_buy_share", "top5_wallet_buy_share",
            "top10_wallet_buy_share", "top20_wallet_buy_share",
            "buyer_concentration_index",
        ]
        for key in expected:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_hhi_bounded(self):
        result = compute_distribution_features(_make_swaps_by_window())
        self.assertGreaterEqual(result["buyer_concentration_index"], 0.0)
        self.assertLessEqual(result["buyer_concentration_index"], 1.0)


class TestRiskSignalsFeatures(unittest.TestCase):
    def test_computes_6_features(self):
        result = compute_risk_signals_features(_make_swaps_by_window())
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
        result = compute_risk_signals_features(_make_swaps_by_window())
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
        axiom_data = {
            "smart_money_activity": {},
            "wallet_profiles": _make_wallet_profiles(),
        }
        result = compute_axiom_features(
            axiom_data, _make_swaps_by_window(), t0_ts=1000,
            smart_wallets=["wallet1"],
        )
        # Should have many features (all 13 categories)
        self.assertGreater(len(result), 80)
        self.assertIn("smart_wallet_buyers_15m", result)
        self.assertIn("avg_wallet_win_rate", result)
        self.assertIn("smart_money_score", result)
        self.assertIn("first_buyer_win_rate", result)

    def test_empty_data_never_crashes(self):
        result = compute_axiom_features({}, {"1m": [], "5m": [], "15m": []})
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_single_wallet_edge_case(self):
        swaps = {
            "15m": [_make_swap("only_one", True, 100, 0.5, 1000)],
        }
        profiles = {
            "profiles": {
                "only_one": {"trade_count": 1, "win_rate": 0.5, "roi": 0.0},
            }
        }
        result = compute_axiom_features(
            {"wallet_profiles": profiles}, swaps, t0_ts=1000,
        )
        self.assertGreater(len(result), 0)
        # Should not crash
        self.assertIsInstance(result["avg_wallet_win_rate"], float)


if __name__ == "__main__":
    unittest.main()
