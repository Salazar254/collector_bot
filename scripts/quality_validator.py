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
    # HOLDERS (5)
    "holder_count_1m", "holder_count_5m", "holder_count_15m",
    "holder_growth_5m", "holder_growth_15m",
    # WHALES (5)
    "largest_buy_usd", "largest_sell_usd",
    "whale_buy_count", "whale_sell_count",
    "whale_net_flow",
    # VOLATILITY (4)
    "volatility_1m", "volatility_5m", "volatility_15m",
    "drawdown_first_15m",
]

FEATURE_CATEGORY_MAP: dict[str, str] = {}
for f in NUMERIC_FEATURES:
    if f.startswith("price"):
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

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

UNIQUE_RATIO_THRESHOLD = 0.05   # flag if < 5%
VARIANCE_EPSILON = 1e-8         # flag if variance ≈ 0


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

            if unique_ratio < UNIQUE_RATIO_THRESHOLD and total >= 10:
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
