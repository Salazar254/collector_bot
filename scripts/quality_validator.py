"""
scripts/quality_validator.py — Automated data quality validation for snapshot features.

For every collection cycle (batch of N tokens), computes per-feature:
  - missing percentage (null/zero prevalence)
  - unique value count and ratio
  - population variance

Flags features where unique_ratio < 5% or variance ≈ 0.
Generates a JSON-formatted quality report logged to stdout.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature category definitions (mirrors TypeScript schema + DB columns)
# Matches the column names in training_tokens v2.
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: list[str] = [
    # PRICE (9)
    "price_usd_t0", "price_usd_1m", "price_usd_5m", "price_usd_15m",
    "price_change_1m_pct", "price_change_5m_pct", "price_change_15m_pct",
    "max_price_first_15m", "min_price_first_15m",
    # LIQUIDITY (6)
    "liquidity_usd_t0", "liquidity_usd_1m", "liquidity_usd_5m", "liquidity_usd_15m",
    "liquidity_growth_5m", "liquidity_growth_15m",
    # VOLUME (3)
    "volume_1m", "volume_5m", "volume_15m",
    # BUYERS (4)
    "unique_buyers_1m", "unique_buyers_5m", "unique_buyers_15m",
    "buyer_growth_rate",
    # SELLERS (4)
    "unique_sellers_1m", "unique_sellers_5m", "unique_sellers_15m",
    "seller_growth_rate",
    # ORDER FLOW (10)
    "buy_count_1m", "buy_count_5m", "buy_count_15m",
    "sell_count_1m", "sell_count_5m", "sell_count_15m",
    "buy_sell_ratio_1m", "buy_sell_ratio_5m", "buy_sell_ratio_15m",
    "net_flow_usd",
    # WHALES (5)
    "largest_buy_usd", "largest_sell_usd",
    "whale_buy_count", "whale_sell_count",
    "whale_net_flow",
    # VOLATILITY (4)
    "volatility_1m", "volatility_5m", "volatility_15m",
    "drawdown_first_15m",
    # SAFETY (7 numeric — excludes sequence_b64 TEXT, has_sequence BOOLEAN)
    "mint_authority_active", "freeze_authority_active",
    "mutable_metadata", "lp_burn_pct",
    "initial_liquidity_sol", "migration_speed_seconds",
    "avg_transaction_size_sol",
    # SMART_MONEY (12)
    "smart_wallet_buyers_1m", "smart_wallet_buyers_5m", "smart_wallet_buyers_15m",
    "smart_wallet_volume_1m", "smart_wallet_volume_5m", "smart_wallet_volume_15m",
    "smart_wallet_percentage", "smart_money_first_buyer",
    "first_smart_money_buy_timestamp",
    "smart_money_within_first_minute", "smart_money_within_first_5m",
    "smart_money_accumulation_rate",
    # WALLET_QUALITY (10)
    "avg_wallet_age_days", "median_wallet_age_days",
    "avg_wallet_trade_count", "median_wallet_trade_count",
    "avg_wallet_win_rate", "median_wallet_win_rate",
    "avg_wallet_realized_pnl", "median_wallet_realized_pnl",
    "avg_wallet_roi", "median_wallet_roi",
    # PNL (10)
    "avg_buyer_pnl_30d", "median_buyer_pnl_30d", "top_buyer_pnl_30d",
    "avg_buyer_pnl_90d", "median_buyer_pnl_90d", "top_buyer_pnl_90d",
    "avg_seller_pnl_30d", "median_seller_pnl_30d",
    "avg_seller_pnl_90d", "median_seller_pnl_90d",
    # WHALE_AXIOM — 5K (8)
    "largest_buy_usd_5k", "largest_sell_usd_5k",
    "whale_buy_count_5k", "whale_sell_count_5k",
    "whale_buy_volume_5k", "whale_sell_volume_5k",
    "whale_net_flow_5k", "whale_accumulation_rate_5k",
    # WHALE_AXIOM — 10K (8)
    "largest_buy_usd_10k", "largest_sell_usd_10k",
    "whale_buy_count_10k", "whale_sell_count_10k",
    "whale_buy_volume_10k", "whale_sell_volume_10k",
    "whale_net_flow_10k", "whale_accumulation_rate_10k",
    # BUYER_QUALITY (5)
    "new_wallet_buyers", "experienced_wallet_buyers",
    "wallets_older_than_30_days", "wallets_older_than_90_days",
    "wallets_older_than_180_days",
    # CONVICTION (6)
    "repeat_buyers", "multi_buy_wallets",
    "wallet_rebuy_rate", "wallet_accumulation_rate",
    "avg_buys_per_wallet", "median_buys_per_wallet",
    # EARLY_STRENGTH (7)
    "first_buyer_win_rate",
    "first_5_buyers_avg_win_rate", "first_10_buyers_avg_win_rate",
    "first_20_buyers_avg_win_rate",
    "first_5_buyers_avg_pnl", "first_10_buyers_avg_pnl",
    "first_20_buyers_avg_pnl",
    # DISTRIBUTION (5)
    "top_wallet_buy_share", "top5_wallet_buy_share",
    "top10_wallet_buy_share", "top20_wallet_buy_share",
    "buyer_concentration_index",
    # RISK_SIGNALS (6)
    "dumping_wallet_count",
    "wallets_sold_within_5m", "wallets_sold_within_15m",
    "wallets_sold_within_60m",
    "fast_exit_rate", "paper_hand_rate",
    # SMART_VS_RETAIL (5)
    "smart_money_volume_share", "smart_money_buy_share",
    "retail_buy_share", "retail_sell_share",
    "smart_money_net_flow",
    # COMPOSITE (5)
    "smart_money_score", "wallet_quality_score",
    "whale_score", "conviction_score", "buyer_quality_score",
]

FEATURE_CATEGORY_MAP: dict[str, str] = {}
for f in NUMERIC_FEATURES:
    if f.startswith("smart_wallet") or f.startswith("smart_money") or f.startswith("first_smart_money"):
        FEATURE_CATEGORY_MAP[f] = "SMART_MONEY"
    elif f.startswith("avg_wallet") or f.startswith("median_wallet"):
        FEATURE_CATEGORY_MAP[f] = "WALLET_QUALITY"
    elif (f.startswith("avg_buyer_pnl") or f.startswith("median_buyer_pnl") or
          f.startswith("top_buyer_pnl") or f.startswith("avg_seller_pnl") or
          f.startswith("median_seller_pnl")):
        FEATURE_CATEGORY_MAP[f] = "PNL"
    elif (f.startswith("avg_buyer_roi") or f.startswith("median_buyer_roi") or
          f.startswith("top_buyer_roi")):
        FEATURE_CATEGORY_MAP[f] = "ROI"
    elif f.startswith("profitable") or f.startswith("high_roi") or \
         f.startswith("elite") or f.startswith("wallets_"):
        FEATURE_CATEGORY_MAP[f] = "PROFITABLE_TRADER"
    elif f.endswith("_1k") or f.endswith("_5k") or f.endswith("_10k"):
        FEATURE_CATEGORY_MAP[f] = "WHALE_AXIOM"
    elif f.startswith("new_wallet") or f.startswith("experienced") or \
         f.startswith("wallets_older"):
        FEATURE_CATEGORY_MAP[f] = "BUYER_QUALITY"
    elif f.startswith("repeat") or f.startswith("multi_buy") or \
         f.startswith("wallet_rebuy") or f.startswith("wallet_accumulation") or \
         f.endswith("buys_per_wallet"):
        FEATURE_CATEGORY_MAP[f] = "CONVICTION"
    elif f.startswith("first_") or f.startswith("first_5") or \
         f.startswith("first_10") or f.startswith("first_20"):
        FEATURE_CATEGORY_MAP[f] = "EARLY_STRENGTH"
    elif f.startswith("top") or f.startswith("top5") or f.startswith("top10") or \
         f.startswith("top20") or f.startswith("buyer_concentration"):
        FEATURE_CATEGORY_MAP[f] = "DISTRIBUTION"
    elif f.startswith("dumping") or f.startswith("wallets_sold") or \
         f.startswith("fast_exit") or f.startswith("paper_hand"):
        FEATURE_CATEGORY_MAP[f] = "RISK_SIGNALS"
    elif f.startswith("smart_money_") and (f.endswith("_share") or f.endswith("_flow")):
        FEATURE_CATEGORY_MAP[f] = "SMART_VS_RETAIL"
    elif f.startswith("retail_"):
        FEATURE_CATEGORY_MAP[f] = "SMART_VS_RETAIL"
    elif f.endswith("_score"):
        FEATURE_CATEGORY_MAP[f] = "COMPOSITE"
    elif f.startswith("price"):
        FEATURE_CATEGORY_MAP[f] = "PRICE"
    elif f.startswith("liquidity"):
        FEATURE_CATEGORY_MAP[f] = "LIQUIDITY"
    elif f.startswith("volume"):
        FEATURE_CATEGORY_MAP[f] = "VOLUME"
    elif f.startswith("unique_buyers") or f.startswith("buyer"):
        FEATURE_CATEGORY_MAP[f] = "BUYERS"
    elif f.startswith("unique_sellers") or f.startswith("seller"):
        FEATURE_CATEGORY_MAP[f] = "SELLERS"
    elif any(f.startswith(p) for p in ("buy_count", "sell_count", "buy_sell", "net_flow")):
        FEATURE_CATEGORY_MAP[f] = "ORDER_FLOW"
    elif f.startswith("holder"):
        FEATURE_CATEGORY_MAP[f] = "HOLDERS"
    elif any(f.startswith(p) for p in ("largest", "whale")):
        FEATURE_CATEGORY_MAP[f] = "WHALES"
    elif f.startswith("volatility") or f.startswith("drawdown"):
        FEATURE_CATEGORY_MAP[f] = "VOLATILITY"
    elif f.startswith("mint_authority") or f.startswith("freeze_authority") or \
         f.startswith("mutable_") or f.startswith("lp_burn") or \
         f.startswith("initial_liquidity") or f.startswith("migration_speed") or \
         f.startswith("avg_transaction_size"):
        FEATURE_CATEGORY_MAP[f] = "SAFETY"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

UNIQUE_RATIO_THRESHOLD = 0.05   # flag if < 5%
VARIANCE_EPSILON = 1e-8         # flag if variance ≈ 0
MISSING_RATE_THRESHOLD = 50.0   # flag if missing_rate > 50%


# ===================================================================
# QualityValidator
# ===================================================================


class QualityValidator:
    """
    Computes data quality metrics for snapshot features across a batch
    of collected token records.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add_record(self, record: dict[str, Any]) -> None:
        """Accumulate a single token record for batch validation."""
        self.records.append(record)

    def add_records(self, records: list[dict[str, Any]]) -> None:
        """Accumulate multiple token records."""
        self.records.extend(records)

    def compute_missing_pct(self, field: str) -> float:
        """
        Percentage of records where the field is missing (None, null) or zero.
        A zero value for counts (e.g. unique_buyers) is semantically missing data.
        """
        if not self.records:
            return 0.0

        count_missing = 0
        for r in self.records:
            val = r.get(field)
            if val is None or val == 0 or val == 0.0:
                count_missing += 1

        return round(count_missing / len(self.records) * 100, 2)

    def compute_unique_ratio(self, field: str) -> tuple[int, float]:
        """Return (unique_count, unique_ratio) for a field."""
        if not self.records:
            return 0, 0.0

        values = [r.get(field) for r in self.records if r.get(field) is not None]
        unique = len(set(values))
        ratio = unique / len(self.records) if self.records else 0.0
        return unique, round(ratio, 4)

    def compute_variance(self, field: str) -> float:
        """Population variance of a field across all records."""
        if not self.records:
            return 0.0

        values = [
            float(r.get(field, 0) or 0)
            for r in self.records
        ]
        if len(values) < 2:
            return 0.0

        arr = np.array(values, dtype=np.float64)
        return float(np.var(arr))

    def flag_features(self) -> list[dict[str, Any]]:
        """
        Evaluate all numeric features and return flagged results.

        Returns list of dicts with:
            feature_name, category, missing_pct, unique_count,
            unique_ratio, variance, flagged, flag_reason
        """
        flags: list[dict[str, Any]] = []
        total = len(self.records)

        for field in NUMERIC_FEATURES:
            missing_pct = self.compute_missing_pct(field)
            unique_count, unique_ratio = self.compute_unique_ratio(field)
            variance = self.compute_variance(field)

            flagged = False
            flag_reason: Optional[str] = None

            if missing_pct > MISSING_RATE_THRESHOLD and total >= 10:
                flagged = True
                flag_reason = "high_missing_rate"
            elif unique_ratio < UNIQUE_RATIO_THRESHOLD and total >= 10:
                flagged = True
                flag_reason = "low_uniqueness"
            elif variance < VARIANCE_EPSILON and total >= 10:
                flagged = True
                flag_reason = "zero_variance"

            flags.append({
                "feature_name": field,
                "category": FEATURE_CATEGORY_MAP.get(field, "UNKNOWN"),
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
                "variance": round(variance, 8),
                "flagged": flagged,
                "flag_reason": flag_reason,
            })

        return flags

    def generate_report(self) -> str:
        """
        Generate a JSON-formatted quality report.

        Returns JSON string with: timestamp, total_rows, features_checked,
        features_flagged, flags[]
        """
        flags = self.flag_features()
        features_flagged = sum(1 for f in flags if f["flagged"])

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_rows": len(self.records),
            "features_checked": len(flags),
            "features_flagged": features_flagged,
            "thresholds": {
                "unique_ratio_min": UNIQUE_RATIO_THRESHOLD,
                "variance_epsilon": VARIANCE_EPSILON,
                "min_rows_for_flag": 10,
            },
            "flags": flags,
        }

        return json.dumps(report, indent=2)

    def reset(self) -> None:
        """Clear accumulated records for next batch."""
        self.records.clear()


# ===================================================================
# Convenience function
# ===================================================================


def run_quality_check(
    records: list[dict[str, Any]],
    write_report: bool = True,
    report_dir: str = "reports",
) -> str:
    """
    Run quality validation on a batch of records and return the report JSON.

    When write_report=True, also writes the report to reports/quality_report.json.
    """
    validator = QualityValidator()
    validator.add_records(records)
    report_json = validator.generate_report()

    flags = json.loads(report_json)
    flagged_count = flags["features_flagged"]
    log.info(
        "Quality check: %d rows, %d features checked, %d flagged",
        flags["total_rows"], flags["features_checked"], flagged_count,
    )

    if flagged_count > 0:
        flagged_names = [f["feature_name"] for f in flags["flags"] if f["flagged"]]
        log.warning("Flagged features: %s", ", ".join(flagged_names))

    if write_report:
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "quality_report.json")
        with open(report_path, "w") as f:
            f.write(report_json)
        log.info("Quality report written to %s", report_path)

    return report_json
