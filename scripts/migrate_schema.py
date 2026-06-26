"""
scripts/migrate_schema.py — Unified schema migration for training_tokens.

Consolidates tokens from multiple incompatible schema versions into a
single clean training_tokens table with the full unified schema including
safety features and new label columns.

Migration phases:
  1. --prepare  → Output CREATE TABLE SQL for training_tokens_v2
  2. --migrate  → Copy all rows from training_tokens → training_tokens_v2
  3. --finalize → Verify counts, print rename SQL, trigger label backfill

Usage:
  python scripts/migrate_schema.py --prepare
  python scripts/migrate_schema.py --migrate --batch-size 50
  python scripts/migrate_schema.py --finalize

Environment variables required:
  SUPABASE_URL, SUPABASE_KEY
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

from supabase import create_client, Client

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Full unified schema column list (order matters for INSERT)
# ---------------------------------------------------------------------------

UNIFIED_COLUMNS: list[str] = [
    "mint",
    "symbol",
    "graduation_timestamp",
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
    # ORDER_FLOW (10)
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
    # SAFETY (9) — NEW in unified schema
    "mint_authority_active",
    "freeze_authority_active",
    "mutable_metadata",
    "lp_burn_pct",
    "initial_liquidity_sol",
    "migration_speed_seconds",
    "avg_transaction_size_sol",
    "sequence_b64",
    "has_sequence",
    # SMART_MONEY (12)
    "smart_wallet_buyers_1m",
    "smart_wallet_buyers_5m",
    "smart_wallet_buyers_15m",
    "smart_wallet_volume_1m",
    "smart_wallet_volume_5m",
    "smart_wallet_volume_15m",
    "smart_wallet_percentage",
    "smart_money_first_buyer",
    "first_smart_money_buy_timestamp",
    "smart_money_within_first_minute",
    "smart_money_within_first_5m",
    "smart_money_accumulation_rate",
    # WALLET_QUALITY (10)
    "avg_wallet_age_days",
    "median_wallet_age_days",
    "avg_wallet_trade_count",
    "median_wallet_trade_count",
    "avg_wallet_win_rate",
    "median_wallet_win_rate",
    "avg_wallet_realized_pnl",
    "median_wallet_realized_pnl",
    "avg_wallet_roi",
    "median_wallet_roi",
    # PNL (10)
    "avg_buyer_pnl_30d",
    "median_buyer_pnl_30d",
    "top_buyer_pnl_30d",
    "avg_buyer_pnl_90d",
    "median_buyer_pnl_90d",
    "top_buyer_pnl_90d",
    "avg_seller_pnl_30d",
    "median_seller_pnl_30d",
    "avg_seller_pnl_90d",
    "median_seller_pnl_90d",
    # WHALE_AXIOM_5K (8)
    "largest_buy_usd_5k",
    "largest_sell_usd_5k",
    "whale_buy_count_5k",
    "whale_sell_count_5k",
    "whale_buy_volume_5k",
    "whale_sell_volume_5k",
    "whale_net_flow_5k",
    "whale_accumulation_rate_5k",
    # WHALE_AXIOM_10K (8)
    "largest_buy_usd_10k",
    "largest_sell_usd_10k",
    "whale_buy_count_10k",
    "whale_sell_count_10k",
    "whale_buy_volume_10k",
    "whale_sell_volume_10k",
    "whale_net_flow_10k",
    "whale_accumulation_rate_10k",
    # BUYER_QUALITY (5)
    "new_wallet_buyers",
    "experienced_wallet_buyers",
    "wallets_older_than_30_days",
    "wallets_older_than_90_days",
    "wallets_older_than_180_days",
    # CONVICTION (6)
    "repeat_buyers",
    "multi_buy_wallets",
    "wallet_rebuy_rate",
    "wallet_accumulation_rate",
    "avg_buys_per_wallet",
    "median_buys_per_wallet",
    # EARLY_STRENGTH (7)
    "first_buyer_win_rate",
    "first_5_buyers_avg_win_rate",
    "first_10_buyers_avg_win_rate",
    "first_20_buyers_avg_win_rate",
    "first_5_buyers_avg_pnl",
    "first_10_buyers_avg_pnl",
    "first_20_buyers_avg_pnl",
    # DISTRIBUTION (5)
    "top_wallet_buy_share",
    "top5_wallet_buy_share",
    "top10_wallet_buy_share",
    "top20_wallet_buy_share",
    "buyer_concentration_index",
    # RISK_SIGNALS (6)
    "dumping_wallet_count",
    "wallets_sold_within_5m",
    "wallets_sold_within_15m",
    "wallets_sold_within_60m",
    "fast_exit_rate",
    "paper_hand_rate",
    # SMART_VS_RETAIL (5)
    "smart_money_volume_share",
    "smart_money_buy_share",
    "retail_buy_share",
    "retail_sell_share",
    "smart_money_net_flow",
    # COMPOSITE (5)
    "smart_money_score",
    "wallet_quality_score",
    "whale_score",
    "conviction_score",
    "buyer_quality_score",
    # AXIOM_META (2)
    "axiom_collected",
    "axiom_cost_usd",
    # LABELS (13)
    "did_1_25x",
    "did_1_5x",
    "did_2x",
    "did_3x",
    "did_5x",
    "did_10x",
    "rugged",
    "survived_24h",
    "max_drawdown_pct",
    "inferred_label",
    "labels_ready",
    "time_to_peak_minutes",
    "peak_multiplier",
    # METADATA
    "deployer_address",
]


def get_create_v2_sql() -> str:
    """Return the SQL to create training_tokens_v2 with the full unified schema."""
    sql = """-- ===================================================================
-- Create training_tokens_v2 with the full unified schema (v2).
-- Run this in the Supabase SQL Editor before running --migrate.
-- ===================================================================

CREATE TABLE IF NOT EXISTS training_tokens_v2 (LIKE training_tokens INCLUDING ALL);

-- Drop v2 indexes (they reference the wrong table name)
DROP INDEX IF EXISTS idx_graduation_ts_v2;
DROP INDEX IF EXISTS idx_collected_at_v2;
DROP INDEX IF EXISTS idx_axiom_collected_v2;

-- Add NEW columns that don't exist in the old table (SAFETY + backfill labels)
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS mint_authority_active   FLOAT4;
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS freeze_authority_active FLOAT4;
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS mutable_metadata        FLOAT4;
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS lp_burn_pct             FLOAT4;
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS initial_liquidity_sol   FLOAT4;
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS migration_speed_seconds FLOAT4;
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS avg_transaction_size_sol FLOAT4;
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS sequence_b64            TEXT;
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS has_sequence            BOOLEAN DEFAULT FALSE;
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS labels_ready            BOOLEAN DEFAULT FALSE;
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS time_to_peak_minutes    FLOAT4;
ALTER TABLE training_tokens_v2 ADD COLUMN IF NOT EXISTS peak_multiplier         FLOAT4;

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_graduation_ts_v2 ON training_tokens_v2(graduation_timestamp);
CREATE INDEX IF NOT EXISTS idx_collected_at_v2 ON training_tokens_v2(collected_at);
CREATE INDEX IF NOT EXISTS idx_axiom_collected_v2 ON training_tokens_v2(axiom_collected);
CREATE INDEX IF NOT EXISTS idx_labels_ready_v2 ON training_tokens_v2(labels_ready);

-- RLS
ALTER TABLE training_tokens_v2 ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_allow_all_v2 ON training_tokens_v2;
CREATE POLICY rls_allow_all_v2 ON training_tokens_v2 FOR ALL USING (true) WITH CHECK (true);
"""
    return sql


def get_rename_sql() -> str:
    """Return the SQL to swap old and new tables."""
    return """-- ===================================================================
-- FINALIZE: Swap training_tokens → training_tokens_v1, v2 → training_tokens
-- Run this in the Supabase SQL Editor AFTER verifying counts match.
-- WARNING: This is a DESTRUCTIVE rename. Backup your data first.
-- ===================================================================

BEGIN;
ALTER TABLE IF EXISTS training_tokens RENAME TO training_tokens_v1;
ALTER TABLE IF EXISTS training_tokens_v2 RENAME TO training_tokens;
COMMIT;

-- After rename, verify:
-- SELECT COUNT(*) FROM training_tokens;
-- SELECT COUNT(*) FROM training_tokens_v1;
"""


# ===================================================================
# Copy data from old table to v2
# ===================================================================


def _sanitize_row(row: dict, known_columns: set[str]) -> dict:
    """Keep only columns that exist in the unified schema. Set missing to None."""
    clean: dict[str, Any] = {}
    for col in UNIFIED_COLUMNS:
        if col in row and row[col] is not None:
            clean[col] = row[col]
        # else: omit — let the DB apply its DEFAULT (NULL for most)
    return clean


def copy_data(
    supabase: Client,
    batch_size: int = 100,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Copy all rows from training_tokens to training_tokens_v2.

    Returns (source_count, copied_count).
    """
    # 1. Count source rows
    try:
        count_result = supabase.table("training_tokens").select("mint", count="exact").execute()
        source_count = count_result.count if count_result.count is not None else 0
    except Exception as exc:
        log.error("Failed to count source rows: %s", exc)
        return 0, 0

    if source_count == 0:
        log.warning("Source table training_tokens has 0 rows — nothing to copy.")
        return 0, 0

    log.info("Source table has %d rows. Starting copy to v2...", source_count)

    # 2. Get the columns that actually exist in the source table
    # Fetch one row to discover columns
    try:
        sample = supabase.table("training_tokens").select("*").limit(1).execute()
        if sample.data:
            known = set(sample.data[0].keys())
        else:
            known = set()
    except Exception:
        known = set()

    # 3. Fetch and copy in batches
    copied = 0
    start = 0

    while start < source_count:
        try:
            result = (
                supabase.table("training_tokens")
                .select("*")
                .range(start, start + batch_size - 1)
                .order("id", desc=False)
                .execute()
            )
            rows = result.data if result.data else []
        except Exception as exc:
            log.error("Failed to fetch batch at offset %d: %s", start, exc)
            break

        if not rows:
            break

        for row in rows:
            clean = _sanitize_row(row, known)
            # Remove 'id' — let v2 auto-generate its own SERIAL id
            clean.pop("id", None)

            if not clean.get("mint"):
                continue

            if dry_run:
                copied += 1
                continue

            try:
                supabase.table("training_tokens_v2").upsert(
                    clean, on_conflict="mint",
                ).execute()
                copied += 1
            except Exception as exc:
                log.warning("Failed to copy mint=%s: %s", clean.get("mint", "?")[:12], exc)

        start += batch_size
        if start % 500 == 0 and start > 0:
            log.info("  Progress: %d / %d rows copied", copied, source_count)

        # Rate-limit: brief pause between batches
        time.sleep(0.3)

    log.info("Copy complete: %d / %d rows copied to v2.", copied, source_count)
    return source_count, copied


def verify_counts(supabase: Client) -> tuple[int, int]:
    """Return (old_count, v2_count)."""
    try:
        old = supabase.table("training_tokens").select("mint", count="exact").execute()
        old_count = old.count if old.count is not None else 0
    except Exception as exc:
        log.error("Failed to count old table: %s", exc)
        old_count = -1

    try:
        new = supabase.table("training_tokens_v2").select("mint", count="exact").execute()
        new_count = new.count if new.count is not None else 0
    except Exception as exc:
        log.error("Failed to count v2 table: %s", exc)
        new_count = -1

    return old_count, new_count


def trigger_backfill(supabase: Client, batch_size: int = 50) -> int:
    """Run label backfill on all tokens >24h old in v2."""
    from backfill_labels import backfill_labels

    total = 0
    while True:
        count = backfill_labels(supabase, batch_size=batch_size, dry_run=False)
        if count == 0:
            break
        total += count
        log.info("Backfill batch: %d tokens (total: %d)", count, total)

    log.info("Backfill complete: %d total tokens updated.", total)
    return total


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified schema migration for training_tokens"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Print CREATE TABLE SQL for training_tokens_v2 and exit.",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Copy all data from training_tokens → training_tokens_v2.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Verify counts, print rename SQL, trigger backfill.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Rows per batch during copy (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log operations without writing to DB.",
    )
    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="Skip confirmation prompts (for automation).",
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help="Skip label backfill during finalize.",
    )
    args = parser.parse_args()

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")

    if not supabase_url or not supabase_key:
        log.error("SUPABASE_URL and SUPABASE_KEY environment variables are required.")
        sys.exit(1)

    supabase = create_client(supabase_url, supabase_key)

    # -- Phase 1: Prepare (print SQL)
    if args.prepare:
        print(get_create_v2_sql())
        print("\n-- After running the above SQL, run:\n")
        print("  python scripts/migrate_schema.py --migrate")
        return

    # -- Phase 2: Migrate (copy data)
    if args.migrate:
        log.info("=== Phase 2: Copy data from training_tokens → training_tokens_v2 ===")
        source_count, copied = copy_data(supabase, batch_size=args.batch_size, dry_run=args.dry_run)

        if source_count == copied:
            log.info("✓ All %d rows copied successfully.", copied)
        else:
            log.warning("⚠ Mismatch: %d source rows, %d copied rows.", source_count, copied)
        return

    # -- Phase 3: Finalize
    if args.finalize:
        log.info("=== Phase 3: Finalize migration ===")

        # Verify
        old_count, new_count = verify_counts(supabase)
        log.info("Row counts: old=%d, v2=%d", old_count, new_count)

        if old_count != new_count or old_count <= 0:
            log.error(
                "Count mismatch! old=%d, v2=%d. Aborting finalization. "
                "Re-run --migrate to retry the copy.",
                old_count, new_count,
            )
            sys.exit(1)

        log.info("✓ Counts match: %d rows in both tables.", old_count)

        # Print rename SQL
        print("\n" + get_rename_sql())

        # Confirmation
        if not args.skip_confirm:
            print("\n⚠  The rename SQL above will REPLACE training_tokens with v2.")
            print("   This is IRREVERSIBLE without a backup.")
            response = input("\nProceed with rename? Type 'yes' to continue: ")
            if response.strip().lower() != "yes":
                log.info("Aborted by user.")
                return

        # Trigger backfill
        if not args.skip_backfill:
            log.info("Starting label backfill on migrated tokens...")
            updated = trigger_backfill(supabase, batch_size=50)
            log.info("Backfill updated %d tokens.", updated)

        log.info("=== Migration complete ===")
        log.info("Next step: Run the rename SQL printed above in the Supabase SQL Editor.")
        return

    # No mode selected
    parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    main()
