"""
scripts/cost_monitor.py — Axiom API cost monitoring and reporting.

Tracks cumulative Axiom API cost per token, per cycle, and total.
Reads from axiom_raw_responses table and generates cost summary reports.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

DEFAULT_COST_WARNING_PER_TOKEN = 0.01   # USD
DEFAULT_COST_WARNING_MONTHLY = 50.0     # USD


class CostMonitor:
    """
    Tracks and reports Axiom API costs.

    Reads from axiom_raw_responses Supabase table and in-memory counters.
    Generates cost summary reports with per-token and monthly estimates.
    """

    def __init__(
        self,
        report_dir: str = "reports",
        warning_per_token: float = DEFAULT_COST_WARNING_PER_TOKEN,
        warning_monthly: float = DEFAULT_COST_WARNING_MONTHLY,
    ) -> None:
        self.report_dir = report_dir
        self.warning_per_token = warning_per_token
        self.warning_monthly = warning_monthly

        # In-memory counters (reset per reporting cycle)
        self._cycle_tokens: int = 0
        self._cycle_cost: float = 0.0

        # Cumulative (lifetime of this process)
        self._total_tokens: int = 0
        self._total_cost: float = 0.0

        # Timestamps for rate calculations
        self._start_time: Optional[datetime] = None

    def start_cycle(self) -> None:
        """Reset per-cycle counters at the start of a collection cycle."""
        self._cycle_tokens = 0
        self._cycle_cost = 0.0

        if self._start_time is None:
            self._start_time = datetime.now(timezone.utc)

    def record_token(self, cost_usd: float, axiom_collected: bool = False) -> None:
        """
        Record cost for one token collected.

        Args:
            cost_usd: Estimated cost for this token's Axiom API calls
            axiom_collected: Whether Axiom data was actually collected
        """
        if axiom_collected:
            self._cycle_tokens += 1
            self._cycle_cost += cost_usd
            self._total_tokens += 1
            self._total_cost += cost_usd

    def get_cycle_summary(self) -> dict[str, Any]:
        """Return per-cycle cost summary."""
        cost_per_token = (
            self._cycle_cost / max(self._cycle_tokens, 1)
            if self._cycle_tokens > 0
            else 0.0
        )

        return {
            "cycle_tokens": self._cycle_tokens,
            "cycle_cost_usd": round(self._cycle_cost, 6),
            "cost_per_token_usd": round(cost_per_token, 6),
        }

    def get_cumulative_summary(self) -> dict[str, Any]:
        """Return cumulative cost summary with monthly estimates."""
        cost_per_token = (
            self._total_cost / max(self._total_tokens, 1)
            if self._total_tokens > 0
            else 0.0
        )

        # Monthly estimate based on elapsed time
        monthly_estimate = 0.0
        if self._start_time is not None:
            elapsed_hours = (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds() / 3600
            if elapsed_hours > 0:
                tokens_per_hour = self._total_tokens / elapsed_hours
                monthly_tokens = tokens_per_hour * 24 * 30
                monthly_estimate = monthly_tokens * cost_per_token

        return {
            "total_tokens_collected": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "cost_per_token_usd": round(cost_per_token, 6),
            "estimated_monthly_cost_usd": round(monthly_estimate, 2),
            "elapsed_hours": (
                round(
                    (datetime.now(timezone.utc) - self._start_time).total_seconds()
                    / 3600,
                    1,
                )
                if self._start_time
                else 0.0
            ),
        }

    def generate_report(self, include_db_snapshot: bool = False) -> str:
        """
        Generate a JSON cost report.

        Args:
            include_db_snapshot: If True, also queries axiom_raw_responses
                                 for a comprehensive snapshot.

        Returns:
            JSON report string
        """
        cycle = self.get_cycle_summary()
        cumulative = self.get_cumulative_summary()

        report: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "per_cycle": cycle,
            "cumulative": cumulative,
            "thresholds": {
                "warning_per_token_usd": self.warning_per_token,
                "warning_monthly_usd": self.warning_monthly,
            },
            "warnings": [],
        }

        # Check thresholds
        if cycle.get("cost_per_token_usd", 0) > self.warning_per_token:
            report["warnings"].append(
                f"Per-token cost ${cycle['cost_per_token_usd']:.4f} exceeds "
                f"threshold ${self.warning_per_token:.4f}"
            )

        if cumulative.get("estimated_monthly_cost_usd", 0) > self.warning_monthly:
            report["warnings"].append(
                f"Estimated monthly cost ${cumulative['estimated_monthly_cost_usd']:.2f} "
                f"exceeds threshold ${self.warning_monthly:.2f}"
            )

        # Optionally include DB snapshot
        if include_db_snapshot:
            report["db_snapshot"] = self._fetch_db_cost_snapshot()

        report_json = json.dumps(report, indent=2)

        # Log warnings
        for warning in report["warnings"]:
            log.warning("Cost warning: %s", warning)

        return report_json

    def generate_and_store(self, include_db_snapshot: bool = False) -> str:
        """
        Generate report and write to file.
        Returns the JSON report string.
        """
        report_json = self.generate_report(include_db_snapshot)

        try:
            os.makedirs(self.report_dir, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                self.report_dir, f"cost_report_{ts}.json"
            )
            with open(report_path, "w") as f:
                f.write(report_json)
            log.info("Cost report written to %s", report_path)
        except OSError as e:
            log.warning("Failed to write cost report: %s", e)

        return report_json

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    def _fetch_db_cost_snapshot(self) -> dict[str, Any]:
        """
        Query axiom_raw_responses for a comprehensive cost snapshot.
        Returns empty dict on failure.
        """
        try:
            from supabase import create_client

            supabase_url = os.environ.get("SUPABASE_URL", "")
            supabase_key = os.environ.get("SUPABASE_KEY", "")

            if not supabase_url or not supabase_key:
                return {"error": "Supabase not configured"}

            client = create_client(supabase_url, supabase_key)

            # Count and sum costs
            result = (
                client.table("axiom_raw_responses")
                .select("cost_usd, request_type")
                .limit(10000)
                .execute()
            )

            rows = result.data if result.data else []
            total_cost = sum(r.get("cost_usd", 0) for r in rows)
            total_requests = len(rows)

            # Per request type breakdown
            type_costs: dict[str, dict[str, Any]] = {}
            for row in rows:
                rt = row.get("request_type", "unknown")
                if rt not in type_costs:
                    type_costs[rt] = {"count": 0, "cost_usd": 0.0}
                type_costs[rt]["count"] += 1
                type_costs[rt]["cost_usd"] += row.get("cost_usd", 0)

            return {
                "total_requests": total_requests,
                "total_cost_usd": round(total_cost, 6),
                "by_request_type": type_costs,
            }

        except Exception as e:
            return {"error": str(e)}


# ===================================================================
# Module-level convenience
# ===================================================================

_monitor: Optional[CostMonitor] = None


def get_cost_monitor() -> CostMonitor:
    """Return the global CostMonitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = CostMonitor()
    return _monitor


def reset_cost_monitor() -> None:
    """Reset the cached monitor (useful for testing)."""
    global _monitor
    _monitor = None
