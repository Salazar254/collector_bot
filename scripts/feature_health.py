"""
scripts/feature_health.py — Feature-health reporting system.

Runs after every collection cycle (every QUALITY_CHECK_INTERVAL tokens).
Computes missing_pct, unique_values, variance for ALL features (existing + Axiom).
Flags: unique_values <= 1, variance ≈ 0, missing_rate > 50%.
Generates JSON report and stores in Supabase axiom_feature_health table.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MISSING_RATE_THRESHOLD = 80.0  # FAIL if > 80% of rows are null/zero
VARIANCE_EPSILON = 1e-8        # FAIL if variance below this (effectively zero)
UNIQUE_MIN_THRESHOLD = 1       # FAIL if unique_values <= 1


class FeatureHealthReporter:
    """
    Generates feature-health reports after every N-token collection cycle.
    Computes metrics for ALL numeric features in training_tokens and stores
    results in axiom_feature_health table + JSON report file.
    """

    def __init__(self, report_dir: str = "reports") -> None:
        self.report_dir = report_dir
        self.records: list[dict[str, Any]] = []
        self._cycle_count = 0

    def add_record(self, record: dict[str, Any]) -> None:
        """Accumulate a single token record for batch health analysis."""
        self.records.append(record)

    def add_records(self, records: list[dict[str, Any]]) -> None:
        """Accumulate multiple token records."""
        self.records.extend(records)

    def reset(self) -> None:
        """Clear accumulated records for next cycle."""
        self.records.clear()

    def generate_report(
        self,
        feature_names: Optional[list[str]] = None,
        feature_categories: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Generate a JSON feature-health report for all accumulated records.

        Args:
            feature_names: List of numeric feature column names to check.
                           If None, auto-discovers from record keys.
            feature_categories: Optional {feature_name: category} mapping.

        Returns:
            JSON string with timestamp, total_rows, features_checked,
            flags array with per-feature metrics.
        """
        total = len(self.records)

        if total == 0:
            return json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_rows": 0,
                "features_checked": 0,
                "features_flagged": 0,
                "flags": [],
            }, indent=2)

        # Auto-discover numeric feature columns
        if feature_names is None:
            feature_names = self._discover_numeric_features()

        category_map = feature_categories or {}

        flags: list[dict[str, Any]] = []
        features_flagged = 0

        for field in feature_names:
            missing_pct = self._compute_missing_pct(field)
            unique_count, unique_ratio = self._compute_uniqueness(field)
            variance = self._compute_variance(field)

            flagged = False
            flag_reason: Optional[str] = None

            if missing_pct > MISSING_RATE_THRESHOLD and total >= 10:
                flagged = True
                flag_reason = "high_missing_rate"
            elif unique_count <= UNIQUE_MIN_THRESHOLD and total >= 10:
                flagged = True
                flag_reason = "single_unique_value"
            elif variance < VARIANCE_EPSILON and total >= 10:
                flagged = True
                flag_reason = "zero_variance"

            if flagged:
                features_flagged += 1

            flags.append({
                "feature_name": field,
                "category": category_map.get(field, "UNKNOWN"),
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
                "variance": round(variance, 8),
                "flagged": flagged,
                "flag_reason": flag_reason,
            })

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_rows": total,
            "features_checked": len(flags),
            "features_flagged": features_flagged,
            "thresholds": {
                "missing_rate_pct": MISSING_RATE_THRESHOLD,
                "variance_epsilon": VARIANCE_EPSILON,
                "unique_min_threshold": UNIQUE_MIN_THRESHOLD,
                "min_rows_for_flag": 10,
            },
            "flags": flags,
        }

        report_json = json.dumps(report, indent=2)

        # Log summary
        log.info(
            "Feature health: %d rows, %d features, %d flagged",
            total, len(flags), features_flagged,
        )
        if features_flagged > 0:
            worst = [f for f in flags if f["flagged"]]
            worst_names = [f["feature_name"] for f in worst[:5]]
            log.warning("Top flagged features: %s", ", ".join(worst_names))

        return report_json

    def generate_and_store(
        self,
        feature_names: Optional[list[str]] = None,
        feature_categories: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Generate report, write to file, and store in Supabase.
        Returns the JSON string.
        """
        report_json = self.generate_report(feature_names, feature_categories)

        # Write to file
        self._write_report(report_json)

        # Store in Supabase
        self._store_in_supabase(report_json)

        return report_json

    # ---------------------------------------------------------------
    # Internal — Metrics Computation
    # ---------------------------------------------------------------

    def _discover_numeric_features(self) -> list[str]:
        """Auto-discover numeric feature column names from records."""
        if not self.records:
            return []

        # Collect all keys that appear to be numeric features
        skip_keys = {
            "mint", "symbol", "collected_at", "deployer_address",
            "inferred_label", "axiom_collected",
        }

        feature_keys: set[str] = set()
        for record in self.records:
            for key, val in record.items():
                if key in skip_keys:
                    continue
                if isinstance(val, (int, float)):
                    feature_keys.add(key)

        return sorted(feature_keys)

    def _compute_missing_pct(self, field: str) -> float:
        """Percentage of records where field is null or zero."""
        if not self.records:
            return 0.0

        count_missing = 0
        for r in self.records:
            val = r.get(field)
            if val is None or val == 0 or val == 0.0:
                count_missing += 1

        return round(count_missing / len(self.records) * 100, 2)

    def _compute_uniqueness(self, field: str) -> tuple[int, float]:
        """Return (unique_count, unique_ratio)."""
        if not self.records:
            return 0, 0.0

        values = [r.get(field) for r in self.records if r.get(field) is not None]
        unique = len(set(values))
        ratio = unique / len(self.records) if self.records else 0.0
        return unique, round(ratio, 4)

    def _compute_variance(self, field: str) -> float:
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

    # ---------------------------------------------------------------
    # Internal — I/O
    # ---------------------------------------------------------------

    def _write_report(self, report_json: str) -> None:
        """Write health report to a timestamped JSON file."""
        try:
            os.makedirs(self.report_dir, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                self.report_dir, f"feature_health_{ts}.json"
            )
            with open(report_path, "w") as f:
                f.write(report_json)
            log.info("Feature health report written to %s", report_path)
        except OSError as e:
            log.warning("Failed to write feature health report: %s", e)

    def _store_in_supabase(self, report_json: str) -> None:
        """
        Store each flagged feature entry in axiom_feature_health table.
        Best-effort — errors are silently logged.
        """
        try:
            from supabase import create_client

            supabase_url = os.environ.get("SUPABASE_URL", "")
            supabase_key = os.environ.get("SUPABASE_KEY", "")

            if not supabase_url or not supabase_key:
                return

            report = json.loads(report_json)
            total_rows = report.get("total_rows", 0)
            flags = report.get("flags", [])

            if not flags:
                return

            client = create_client(supabase_url, supabase_key)
            now = datetime.now(timezone.utc).isoformat()

            rows = [
                {
                    "timestamp": now,
                    "feature_name": f["feature_name"],
                    "category": f["category"],
                    "missing_pct": f["missing_pct"],
                    "unique_count": f["unique_count"],
                    "variance": f["variance"],
                    "flagged": 1 if f["flagged"] else 0,
                    "flag_reason": f["flag_reason"],
                    "total_rows_sampled": total_rows,
                }
                for f in flags
            ]

            # Insert in batches of 50 to avoid large payloads
            batch_size = 50
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                client.table("axiom_feature_health").insert(batch).execute()

            log.info("Stored %d feature health entries in Supabase", len(rows))

        except Exception:
            # Best-effort — never crash the collector
            log.debug("Failed to store feature health in Supabase", exc_info=True)


# ===================================================================
# Convenience function
# ===================================================================


def run_feature_health_check(
    records: list[dict[str, Any]],
    feature_names: Optional[list[str]] = None,
    feature_categories: Optional[dict[str, str]] = None,
    report_dir: str = "reports",
    store_in_db: bool = True,
) -> str:
    """
    Run a feature health check on a batch of records.

    Args:
        records: List of token record dicts
        feature_names: Optional list of feature column names
        feature_categories: Optional {name: category} mapping
        report_dir: Directory for JSON report files
        store_in_db: Whether to also store in axiom_feature_health table

    Returns:
        JSON report string
    """
    reporter = FeatureHealthReporter(report_dir=report_dir)
    reporter.add_records(records)

    if store_in_db:
        return reporter.generate_and_store(feature_names, feature_categories)
    else:
        return reporter.generate_report(feature_names, feature_categories)
