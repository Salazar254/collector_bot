"""
scripts/export_csv.py — ML-optimized CSV export from Supabase training_tokens.

Exports all snapshot features + labels in column order matching the
TypeScript CSVExportRow interface. Supports XGBoost, LightGBM, PyTorch GRU,
and ONNX deployment pipelines.

Usage:
    python scripts/export_csv.py --output data/training.csv --days 30
    python scripts/export_csv.py --output data/training.csv --all
    python scripts/export_csv.py --output data/onnx_export.csv --onnx
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from supabase import create_client, Client

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column order — MUST match src/types/features.ts CSVExportRow interface
# ---------------------------------------------------------------------------

CSV_COLUMNS: list[str] = [
    # TOKEN identity
    "mint_address",
    "symbol",
    "migration_timestamp",
    "collected_at",

    # PRICE (9)
    "price_usd_t0",
    "price_usd_1m",
    "price_usd_5m",
    "price_usd_15m",
    "price_change_1m_pct",
    "price_change_5m_pct",
    "price_change_15m_pct",
    "max_price_first_15m",
    "min_price_first_15m",

    # LIQUIDITY (6)
    "liquidity_usd_t0",
    "liquidity_usd_1m",
    "liquidity_usd_5m",
    "liquidity_usd_15m",
    "liquidity_growth_5m",
    "liquidity_growth_15m",

    # VOLUME (3)
    "volume_1m",
    "volume_5m",
    "volume_15m",

    # BUYERS (4)
    "unique_buyers_1m",
    "unique_buyers_5m",
    "unique_buyers_15m",
    "buyer_growth_rate",

    # SELLERS (4)
    "unique_sellers_1m",
    "unique_sellers_5m",
    "unique_sellers_15m",
    "seller_growth_rate",

    # ORDER FLOW (10)
    "buy_count_1m",
    "buy_count_5m",
    "buy_count_15m",
    "sell_count_1m",
    "sell_count_5m",
    "sell_count_15m",
    "buy_sell_ratio_1m",
    "buy_sell_ratio_5m",
    "buy_sell_ratio_15m",
    "net_flow_usd",

    # HOLDERS (5)
    "holder_count_1m",
    "holder_count_5m",
    "holder_count_15m",
    "holder_growth_5m",
    "holder_growth_15m",

    # WHALES (5)
    "largest_buy_usd",
    "largest_sell_usd",
    "whale_buy_count",
    "whale_sell_count",
    "whale_net_flow",

    # VOLATILITY (4)
    "volatility_1m",
    "volatility_5m",
    "volatility_15m",
    "drawdown_first_15m",

    # LABELS (5)
    "did_2x",
    "did_5x",
    "did_10x",
    "max_drawdown_pct",
    "inferred_label",
]

# DB column name → CSV column name mapping
# (DB uses "mint" for address, CSV uses "mint_address")
DB_TO_CSV_MAP: dict[str, str] = {
    "mint": "mint_address",
}
for col in CSV_COLUMNS:
    if col not in DB_TO_CSV_MAP.values():
        DB_TO_CSV_MAP[col] = col  # same name in DB

# Reverse: CSV → DB
CSV_TO_DB_MAP: dict[str, str] = {v: k for k, v in DB_TO_CSV_MAP.items()}

# ---------------------------------------------------------------------------
# ONNX dtype metadata
# ---------------------------------------------------------------------------

ONNX_DTYPE_MAP: dict[str, str] = {}
for col in CSV_COLUMNS:
    if col in ("mint_address", "symbol", "collected_at", "migration_timestamp"):
        ONNX_DTYPE_MAP[col] = "string"
    elif col.endswith("_count") or col.startswith("unique_") or col.startswith("holder_count"):
        ONNX_DTYPE_MAP[col] = "int32"
    elif col.startswith("did_"):
        ONNX_DTYPE_MAP[col] = "int32"
    elif col == "inferred_label":
        ONNX_DTYPE_MAP[col] = "int32"
    else:
        ONNX_DTYPE_MAP[col] = "float32"


# ===================================================================
# Export functions
# ===================================================================


def fetch_rows(
    supabase: Client,
    days: Optional[int] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Fetch rows from training_tokens with optional time cutoff.

    Args:
        supabase: Supabase client
        days: Only fetch rows collected in last N days
        limit: Max rows to fetch

    Returns:
        List of row dicts
    """
    # Build the column list for SELECT
    db_columns = [CSV_TO_DB_MAP.get(c, c) for c in CSV_COLUMNS]
    # Ensure mint is included (maps to mint_address)
    if "mint" not in db_columns:
        db_columns.insert(0, "mint")
    # Add deployer_address for completeness
    if "deployer_address" not in db_columns:
        db_columns.append("deployer_address")

    select_str = ", ".join(db_columns)

    query = supabase.table("training_tokens").select(select_str)

    if days is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        query = query.gte("collected_at", cutoff)

    if limit is not None:
        query = query.limit(limit)

    query = query.order("collected_at", desc=False)

    result = query.execute()
    return result.data if result.data else []


def row_to_csv(row: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a Supabase row dict to CSV column format.
    Maps DB column names to CSV column names and fills missing with defaults.
    """
    csv_row: dict[str, Any] = {}
    for csv_col in CSV_COLUMNS:
        db_col = CSV_TO_DB_MAP.get(csv_col, csv_col)
        val = row.get(db_col)

        # Defaults for missing values
        if val is None:
            if csv_col in ("mint_address", "symbol", "collected_at"):
                val = ""
            elif csv_col in ("migration_timestamp",):
                val = 0
            elif any(csv_col.startswith(p) for p in (
                "did_", "unique_", "buy_count", "sell_count",
                "holder_count", "whale_buy_count", "whale_sell_count",
            )):
                val = 0
            elif csv_col == "inferred_label":
                val = 0
            else:
                val = 0.0

        csv_row[csv_col] = val

    return csv_row


def export_training_csv(
    output_path: str,
    supabase_url: str,
    supabase_key: str,
    days: Optional[int] = None,
    limit: Optional[int] = None,
) -> int:
    """
    Export training data as CSV.

    Returns: number of rows exported.
    """
    supabase: Client = create_client(supabase_url, supabase_key)
    rows = fetch_rows(supabase, days=days, limit=limit)

    if not rows:
        log.warning("No rows fetched — export aborted.")
        return 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_csv(row))

    log.info("Exported %d rows to %s", len(rows), output_path)
    return len(rows)


def export_for_onnx(
    output_path: str,
    supabase_url: str,
    supabase_key: str,
    days: Optional[int] = None,
    limit: Optional[int] = None,
) -> int:
    """
    Export training data as CSV + companion ONNX .meta.json with dtype annotations.

    Returns: number of rows exported.
    """
    count = export_training_csv(output_path, supabase_url, supabase_key, days=days, limit=limit)

    meta_path = output_path.replace(".csv", ".meta.json")

    # Build label columns list (targets for ONNX)
    label_cols = ["did_2x", "did_5x", "did_10x", "max_drawdown_pct"]
    feature_cols = [c for c in CSV_COLUMNS if c not in label_cols
                    and c not in ("mint_address", "symbol", "migration_timestamp",
                                  "collected_at", "inferred_label")]

    meta = {
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "rows": count,
        "columns": CSV_COLUMNS,
        "dtypes": ONNX_DTYPE_MAP,
        "label_columns": label_cols,
        "feature_columns": feature_cols,
        "onnx_opset": 15,
        "notes": (
            "String columns (mint_address, symbol, collected_at, migration_timestamp) "
            "should be excluded from feature tensors. "
            "Use feature_columns for X (float32/int32) and label_columns for y."
        ),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("ONNX metadata written to %s", meta_path)
    return count


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export training_tokens as ML-ready CSV"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/training.csv",
        help="Output CSV file path (default: data/training.csv)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only export rows collected in last N days",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all rows (no time cutoff)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to export",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Also generate ONNX-compatible .meta.json",
    )
    args = parser.parse_args()

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")

    if not supabase_url or not supabase_key:
        log.error("SUPABASE_URL and SUPABASE_KEY environment variables are required.")
        sys.exit(1)

    days = None if args.all else (args.days or 30)

    log.info("Exporting from Supabase: days=%s, limit=%s, onnx=%s",
             days, args.limit, args.onnx)

    if args.onnx:
        count = export_for_onnx(args.output, supabase_url, supabase_key,
                                days=days, limit=args.limit)
    else:
        count = export_training_csv(args.output, supabase_url, supabase_key,
                                    days=days, limit=args.limit)

    log.info("Done. %d rows exported.", count)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    main()
