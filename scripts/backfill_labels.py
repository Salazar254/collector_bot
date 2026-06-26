"""
scripts/backfill_labels.py — 24h label backfill for existing tokens.

Queries Supabase for tokens collected >24h ago whose labels are still
inferred (not yet confirmed with real 24h price data). Calls DexScreener
to get actual price_change_24h, then computes profit-tier labels and
updates the database row.

Intended to run as a daemon thread inside collector_bot v2, or standalone:

    python scripts/backfill_labels.py --once

Environment variables required:
    SUPABASE_URL, SUPABASE_KEY
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

from supabase import create_client, Client

from features import compute_labels

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DexScreener 24h fetch (self-contained, no collect_service import needed)
# ---------------------------------------------------------------------------

import requests

DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens/{}"


def _fetch_price_24h(mint: str, timeout: int = 10) -> Optional[dict]:
    """
    Fetch 24h price change + liquidity from DexScreener for a single mint.

    Returns dict with: price_change_24h, liquidity_usd, volume_24h
    Returns None on any failure.
    """
    url = DEXSCREENER_TOKEN_URL.format(mint)
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 429:
                wait = 0.5 * (2 ** attempt)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            pairs = (data.get("pairs") or [])
            if not pairs:
                return None
            pair = pairs[0]
            return {
                "price_change_24h": float(
                    (pair.get("priceChange") or {}).get("h24", 0) or 0
                ),
                "liquidity_usd": float(
                    (pair.get("liquidity") or {}).get("usd", 0) or 0
                ),
                "volume_24h": float(
                    (pair.get("volume") or {}).get("h24", 0) or 0
                ),
            }
        except Exception:
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
            continue
    return None


# ---------------------------------------------------------------------------
# Backfill logic
# ---------------------------------------------------------------------------

def _compute_peak_metrics(price_change_24h: float) -> tuple[Optional[float], float]:
    """
    Best-effort peak detection from a single 24h price-change datapoint.

    DexScreener free-tier does not provide intraday tick data, so true
    peak time is unknowable from a single snapshot.  We set:
      - peak_multiplier = 1 + max(0, price_change_24h) / 100
      - time_to_peak_minutes = None  (unknown)
    """
    if price_change_24h > 0:
        peak_multiplier = 1.0 + (price_change_24h / 100.0)
    else:
        peak_multiplier = 1.0
    return None, round(peak_multiplier, 4)


def backfill_labels(
    supabase: Client,
    batch_size: int = 50,
    dry_run: bool = False,
) -> int:
    """
    Backfill real labels for tokens whose graduation was >24h ago.

    For each eligible token:
      1. Fetch current 24h price data from DexScreener.
      2. Compute profit-tier labels via features.compute_labels().
      3. Compute peak_multiplier and time_to_peak_minutes.
      4. UPDATE the row in training_tokens with real labels + labels_ready=True.

    Args:
        supabase: Authenticated Supabase client.
        batch_size: Max tokens to process per call.
        dry_run: If True, log what would be updated without writing.

    Returns:
        Number of tokens updated.
    """
    # Find tokens that graduated >24h ago and haven't been backfilled yet
    cutoff_epoch = int(time.time() - 86400)  # 24h ago

    try:
        result = (
            supabase.table("training_tokens")
            .select("mint, symbol, graduation_timestamp")
            .lt("graduation_timestamp", cutoff_epoch)
            .or_("labels_ready.is.null,labels_ready.eq.false")
            .order("graduation_timestamp", desc=False)
            .limit(batch_size)
            .execute()
        )
        candidates = result.data if result.data else []
    except Exception:
        log.exception("Failed to query candidates for backfill")
        return 0

    if not candidates:
        return 0

    updated = 0
    for row in candidates:
        mint = row.get("mint", "")
        if not mint:
            continue

        symbol = row.get("symbol", mint[:8])

        # Fetch real 24h data
        price_24h = _fetch_price_24h(mint)
        if price_24h is None:
            log.debug("Backfill: no DexScreener data for %s", symbol)
            continue

        # Compute labels
        labels = compute_labels(price_24h)

        # Compute peak metrics
        pc = price_24h.get("price_change_24h", 0)
        time_to_peak, peak_mult = _compute_peak_metrics(pc)

        update_payload = {
            **labels,
            "labels_ready": True,
            "inferred_label": False,  # now backed by real 24h data
            "time_to_peak_minutes": time_to_peak,
            "peak_multiplier": peak_mult,
        }

        if dry_run:
            log.info(
                "DRY-RUN: %s | change_24h=%+.1f%% | rugged=%d survived=%d | peak=%.2fx",
                symbol,
                pc,
                labels.get("rugged", 0),
                labels.get("survived_24h", 0),
                peak_mult,
            )
            updated += 1
            continue

        try:
            (
                supabase.table("training_tokens")
                .update(update_payload)
                .eq("mint", mint)
                .execute()
            )
            updated += 1
            log.debug(
                "Backfill: %s change_24h=%+.1f%% rugged=%d survived=%d",
                symbol,
                pc,
                labels.get("rugged", 0),
                labels.get("survived_24h", 0),
            )
        except Exception:
            log.exception("Backfill: update failed for %s", mint)

        # Respect DexScreener rate limit (1 req/s)
        time.sleep(1.0)

    return updated


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill 24h labels for tokens >24h old"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one backfill pass and exit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Max tokens per pass (default: 50)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between passes in daemon mode (default: 300)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log updates without writing to DB.",
    )
    args = parser.parse_args()

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")

    if not supabase_url or not supabase_key:
        log.error("SUPABASE_URL and SUPABASE_KEY are required.")
        sys.exit(1)

    supabase = create_client(supabase_url, supabase_key)

    if args.once:
        count = backfill_labels(supabase, batch_size=args.batch_size, dry_run=args.dry_run)
        log.info("Backfill complete: %d tokens updated.", count)
        return

    # Daemon mode
    log.info(
        "Backfill daemon started — checking every %ds, batch size %d.",
        args.interval,
        args.batch_size,
    )
    while True:
        try:
            count = backfill_labels(supabase, batch_size=args.batch_size)
            if count > 0:
                log.info("Backfill: updated %d token labels.", count)
        except Exception:
            log.exception("Backfill daemon error")
        time.sleep(args.interval)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    main()
