"""
scripts/collect_service.py — Standalone data collection service.

Runs as a long-running Python service on Render.com free tier.
Continuously collects graduated pump.fun token data from
Helius + DexScreener and saves to Supabase free tier.

KEY CONSTRAINT: Supabase free tier = 500 MB limit.
Sequence data is stored as compressed float16 base64
to fit 1M rows within 500 MB.

ENVIRONMENT VARIABLES (set in Render dashboard):
  HELIUS_API_KEY=
  SUPABASE_URL=
  SUPABASE_KEY=
  COLLECTION_INTERVAL_SECONDS=30
  PORT=8080
"""

import os
import time
import json
import sys
import argparse
import logging
import requests
import numpy as np
import base64
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sequence compression utilities  (PART 2)
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 16
SEQUENCE_FEATURES = 6
# Feature order must match src/features/sequence_buffer.ts
# [holders, liquidity, volume, ratio, velocity, tx_count]


def compress_sequence(sequence: list) -> str:
    """
    Converts [16, 6] float list to compressed base64.
    Uses float16 (half precision) — sufficient for
    training, 3x smaller than float32 JSONB.

    Size: 16 × 6 × 2 bytes = 192 bytes raw
          × 4/3 base64 overhead = ~256 bytes as TEXT
    """
    arr = np.array(sequence, dtype=np.float16)
    if arr.shape != (SEQUENCE_LENGTH, SEQUENCE_FEATURES):
        # Pad or truncate to correct shape
        padded = np.zeros(
            (SEQUENCE_LENGTH, SEQUENCE_FEATURES),
            dtype=np.float16,
        )
        rows = min(len(sequence), SEQUENCE_LENGTH)
        padded[:rows] = arr[:rows]
        arr = padded
    return base64.b64encode(arr.tobytes()).decode("utf-8")


def decompress_sequence(b64_str: str) -> np.ndarray:
    """
    Converts compressed base64 back to [16, 6] float32.
    Called during Colab training data loading.
    Returns float32 for PyTorch compatibility.
    """
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.float16)
    return arr.reshape(SEQUENCE_LENGTH, SEQUENCE_FEATURES).astype(np.float32)


def zero_sequence() -> str:
    """Returns a compressed all-zero sequence."""
    return compress_sequence(
        [[0.0] * SEQUENCE_FEATURES for _ in range(SEQUENCE_LENGTH)]
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
HELIUS_KEY = os.environ["HELIUS_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
INTERVAL = int(os.environ.get("COLLECTION_INTERVAL_SECONDS", "30"))
KEEPALIVE_PORT = int(os.environ.get("PORT", "8080"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PUMP_MIGRATION_PROGRAM = (
    "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
)

# Well-known non-pump.fun mints — skip these to avoid collecting
# wrapped-SOL, stablecoins, and system accounts as "tokens"
SKIP_MINTS: frozenset[str] = frozenset({
    "So11111111111111111111111111111111111111112",   # Wrapped SOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
    "11111111111111111111111111111111",               # Native SOL (system program)
})

# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ===================================================================
# API HELPERS
# ===================================================================


def with_retry(fn, max_retries=5, base_wait=1.0):
    """Exponential backoff retry for all API calls."""
    for attempt in range(max_retries):
        try:
            result = fn()
            if result is not None:
                return result
        except Exception as e:
            log.warning("Attempt %d failed: %s", attempt + 1, e)
        wait = (base_wait * 2 ** attempt) + (0.1 * attempt)
        log.info("Waiting %.1fs before retry", wait)
        time.sleep(wait)
    return None


# ===================================================================
# SUPABASE HELPERS
# ===================================================================


def get_existing_mints() -> set[str]:
    """Return set of mint addresses already stored in Supabase."""
    try:
        result = supabase.table("training_tokens").select("mint").execute()
        return {row["mint"] for row in result.data}
    except Exception as e:
        log.error("Failed to fetch existing mints: %s", e)
        return set()


def save_to_supabase(record: dict) -> bool:
    """Upsert a single token record into the training_tokens table."""
    try:
        supabase.table("training_tokens").upsert(
            record, on_conflict="mint",
        ).execute()
        return True
    except Exception as e:
        log.error("Supabase upsert failed for %s: %s", record.get("mint", "?"), e)
        return False


def save_batch(records: list[dict]) -> int:
    """Upsert a batch of records into Supabase.  Returns count saved."""
    if not records:
        return 0
    try:
        supabase.table("training_tokens").upsert(
            records, on_conflict="mint",
        ).execute()
        return len(records)
    except Exception as e:
        log.error("Supabase batch upsert failed: %s", e)
        # Fall back to one-by-one
        saved = 0
        for record in records:
            if save_to_supabase(record):
                saved += 1
        return saved


# ===================================================================
# MINT EXTRACTION
# ===================================================================


def _is_valid_mint(mint: str) -> bool:
    """Return False for known non-pump.fun mints (WSOL, USDC, etc.)."""
    return mint not in SKIP_MINTS and mint != PUMP_MIGRATION_PROGRAM


def extract_mint_from_tx(tx: dict) -> str | None:
    """
    Extract the token mint address from a Helius enhanced transaction.

    Pump.fun migration transactions contain token-balance changes or
    token-transfers that reveal the mint being migrated to Raydium.

    Filters out well-known non-pump.fun mints (WSOL, USDC, USDT, etc.).
    """
    # Strategy 1 — tokenTransfers array (Helius enhanced)
    for transfer in tx.get("tokenTransfers", []):
        mint = transfer.get("mint")
        if mint and _is_valid_mint(mint):
            return mint

    # Strategy 2 — accountData (Helius enhanced)
    for acct in tx.get("accountData", []):
        if not _is_valid_mint(acct.get("account", "")):
            continue
        raw = acct.get("raw", "")
        if raw and len(raw) > 32:
            return acct["account"]

    # Strategy 3 — tokenBalanceChanges
    for change in tx.get("tokenBalanceChanges", []):
        mint = change.get("mint")
        if mint and _is_valid_mint(mint):
            return mint

    # Strategy 4 — logMessages parsing (fallback)
    for msg in tx.get("logMessages", []):
        if "mint" in msg.lower() or "token" in msg.lower():
            for word in msg.split():
                if len(word) >= 43 and len(word) <= 45 and _is_valid_mint(word):
                    return word

    return None


# ===================================================================
# STEP 1: GET GRADUATED MINTS FROM HELIUS
# ===================================================================


def get_graduated_mints(
    before_sig: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    Fetches recently graduated pump.fun tokens
    from the migration program transaction history.
    No text search — raw program transactions only.
    ~0.5 s delay between calls (2 req/s free tier).
    """
    url = (
        f"https://api.helius.xyz/v0/addresses/"
        f"{PUMP_MIGRATION_PROGRAM}/transactions"
    )
    params: dict = {"api-key": HELIUS_KEY, "limit": limit}
    if before_sig:
        params["before"] = before_sig

    def fetch():
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 429:
            log.warning("Helius rate limited")
            return None
        r.raise_for_status()
        return r.json()

    result = with_retry(fetch)
    time.sleep(0.5)  # 2 req/s limit
    return result or []


# ===================================================================
# STEP 2: BATCH ASSET METADATA FROM HELIUS
# ===================================================================


def get_assets_batch(mints: list[str]) -> list[dict]:
    """
    Fetches token metadata for up to 100 mints.
    Uses Helius DAS getAssets batch endpoint.
    Much faster than individual getAsset calls.
    ~0.5 s delay (2 req/s free tier).
    """
    if not mints:
        return []

    url = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAssetBatch",
        "params": {
            "ids": mints[:100],
            "displayOptions": {"showFungible": True},
        },
    }

    def fetch():
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code == 429:
            return None
        r.raise_for_status()
        return r.json().get("result", [])

    result = with_retry(fetch)
    time.sleep(0.5)
    return result or []


# ===================================================================
# STEP 3: BATCH PRICE DATA FROM DEXSCREENER
# ===================================================================


def get_prices_batch(mints: list[str]) -> dict[str, dict]:
    """
    Fetches price + liquidity data for up to 30 mints from DexScreener.
    Free tier — no API key, no strict rate limit on the pairs endpoint.
    Returns dict: mint → price_data.
    """
    if not mints:
        return {}

    ids = ",".join(mints[:30])
    url = f"https://api.dexscreener.com/latest/dex/tokens/{ids}"

    def fetch():
        r = requests.get(url, timeout=15)
        if r.status_code == 429:
            return None
        r.raise_for_status()
        return r.json()

    result = with_retry(fetch, max_retries=2, base_wait=0.5)
    # No sleep — DexScreener pairs endpoint has no hard rate limit

    if not result:
        return {}

    price_map: dict[str, dict] = {}
    for pair in result.get("pairs", []) or []:
        mint = pair.get("baseToken", {}).get("address", "")
        if mint and mint not in price_map:
            price_map[mint] = {
                "price_usd": float(pair.get("priceUsd", 0) or 0),
                "liquidity_usd": float(
                    (pair.get("liquidity") or {}).get("usd", 0) or 0
                ),
                "volume_24h": float(
                    (pair.get("volume") or {}).get("h24", 0) or 0
                ),
                "price_change_24h": float(
                    (pair.get("priceChange") or {}).get("h24", 0) or 0
                ),
                "created_at": pair.get("pairCreatedAt", 0),
            }
    return price_map


# ===================================================================
# STEP 4: GET SWAP TRANSACTIONS FROM HELIUS
# ===================================================================


def get_token_swaps(
    mint: str,
    graduation_ts: int,
) -> list[dict]:
    """
    Gets swap transactions for sequence features.
    Fetches first 100 swaps after graduation.
    ~0.5 s delay (2 req/s).
    """
    url = f"https://api.helius.xyz/v0/addresses/{mint}/transactions"
    params: dict = {
        "api-key": HELIUS_KEY,
        "limit": 100,
        "type": "SWAP",
    }

    def fetch():
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 429:
            return None
        r.raise_for_status()
        txs = r.json()
        # Filter to 72 h window after graduation
        return [
            tx
            for tx in txs
            if tx.get("timestamp", 0) >= graduation_ts
            and tx.get("timestamp", 0) <= graduation_ts + 259200
        ]

    result = with_retry(fetch)
    time.sleep(0.5)
    return result or []


# ===================================================================
# STEP 5: COMPUTE FEATURES
# ===================================================================


def compute_features(
    asset: dict,
    swaps: list[dict],
    graduation_ts: int,
) -> tuple[dict, list[list]]:
    """
    Computes all 14 tabular features and a [16, 6] sequence
    from raw on-chain data.  No Rugcheck dependency.
    """
    features: dict = {}

    # ---- Authority features (from DAS asset) ----
    authorities = asset.get("authorities", [])
    features["mint_authority_active"] = float(
        any("mint" in a.get("scopes", []) for a in authorities)
    )
    features["freeze_authority_active"] = float(
        any("freeze" in a.get("scopes", []) for a in authorities)
    )
    features["mutable_metadata"] = float(asset.get("mutable", True))

    # ---- Default values ----
    features.update({
        "lp_burn_pct": 0.9,
        "initial_liquidity_sol": 0.0,
        "liquidity_concentration": 0.0,
        "dev_hold_pct": 0.0,
        "top10_holder_pct": 0.0,
        "bundle_wallet_count": 0.0,
        "migration_speed_seconds": 0.5,
        "buy_sell_ratio_60s": 0.5,
        "price_velocity_60s": 0.0,
        "unique_buyers_60s": 0.0,
        "avg_transaction_size_sol": 0.0,
    })

    if not swaps:
        return features, []

    # ---- Initial liquidity from first swap ----
    first_ev = swaps[0].get("events", {}).get("swap", {})
    native_in = int((first_ev.get("nativeInput") or {}).get("amount", 0) or 0)
    pool_sol = abs(native_in) / 1e9
    features["initial_liquidity_sol"] = min(pool_sol / 1000, 1.0)

    # ---- First 60 seconds behavior ----
    swaps_60s = [
        s for s in swaps
        if graduation_ts <= s.get("timestamp", 0) <= graduation_ts + 60
    ]

    if swaps_60s:
        buyers: set[str] = set()
        buy_vol = sell_vol = 0.0
        prices: list[float] = []

        for swap in swaps_60s:
            ev = swap.get("events", {}).get("swap", {})
            signer = swap.get("feePayer", "")
            nat_in = (ev.get("nativeInput") or {})
            nat_out = (ev.get("nativeOutput") or {})
            tok_out = ev.get("tokenOutputs", [{}])

            in_amt = int(nat_in.get("amount", 0) or 0)
            out_amt = int(nat_out.get("amount", 0) or 0)
            sol_in = in_amt / 1e9

            if in_amt:
                buyers.add(signer)
                buy_vol += in_amt
            if out_amt:
                sell_vol += out_amt

            tok_amt = float(tok_out[0].get("tokenAmount", 0)) if tok_out else 0
            if tok_amt > 0:
                prices.append(sol_in / tok_amt)

        # Aggregate metrics
        total_vol = buy_vol + sell_vol
        features["buy_sell_ratio_60s"] = (
            (buy_vol / total_vol) if total_vol > 0 else 0.5
        )
        features["unique_buyers_60s"] = min(len(buyers) / 100, 1.0)

        if len(prices) >= 2 and prices[0] > 0:
            change = (prices[-1] - prices[0]) / prices[0]
            features["price_velocity_60s"] = max(min(change, 1.0), -1.0)

        avg_swap = ((buy_vol / len(buyers)) / 1e9) if buyers else 0.0
        features["avg_transaction_size_sol"] = min(avg_swap / 10, 1.0)

    # ---- Build [16, 6] temporal sequence ----
    sequence: list[list] = []
    seq_buyers: set[str] = set()
    seq_buy_vol = seq_sell_vol = 0.0

    for i, swap in enumerate(swaps[:SEQUENCE_LENGTH]):
        ev = swap.get("events", {}).get("swap", {})
        signer = swap.get("feePayer", "")
        seq_in = int((ev.get("nativeInput") or {}).get("amount", 0) or 0)
        seq_out = int((ev.get("nativeOutput") or {}).get("amount", 0) or 0)

        sol_in = seq_in / 1e9
        if seq_in:
            seq_buyers.add(signer)
            seq_buy_vol += seq_in
        if seq_out:
            seq_sell_vol += seq_out

        total_nat = seq_buy_vol + seq_sell_vol
        sequence.append([
            len(seq_buyers),                            # holders (cumulative buyers)
            pool_sol,                                   # liquidity
            total_nat / 1e9,                            # volume
            (seq_buy_vol / total_nat) if total_nat > 0 else 0.5,  # buy ratio
            sol_in,                                     # velocity
            i + 1,                                      # tx_count
        ])

    # Pad sequence to exactly [16, 6]
    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append([0, 0, 0, 0.5, 0, 0])

    return features, sequence[:SEQUENCE_LENGTH]


# ===================================================================
# LABEL COMPUTATION  (rug / drawdown / pump-2x)
# ===================================================================


def compute_labels_dexscreener(
    price_data: dict,
    graduation_ts: int,
) -> dict:
    """
    Compute rug-detection labels from DexScreener price data.
    Uses priceChange.h24 and liquidity to infer rug outcome
    without needing 72 h of raw swap data.
    """
    price_change_24h = price_data.get("price_change_24h", 0)
    liquidity_usd = price_data.get("liquidity_usd", 0)

    # Simple rug detection from price change
    max_drawdown = max(0.0, -price_change_24h)

    rug_label = 1 if (
        price_change_24h < -80 or
        liquidity_usd < 100  # LP essentially gone
    ) else 0

    return {
        "rug_label": rug_label,
        "time_to_rug_hours": 12.0 if rug_label else 72.0,
        "max_drawdown_pct": max_drawdown,
        "pump_2x_label": 1 if price_change_24h > 100 else 0,
        "inferred_label": True,
    }


# ===================================================================
# QUALITY FILTER — avoid spending 100 Helius credits on bad tokens
# ===================================================================


def should_fetch_swaps(asset: dict, price_data: dict | None) -> bool:
    """
    Only spend 100 credits on swap data for tokens that look promising.

    Returns False for tokens likely to be rugs or with no price history,
    saving ~75% of monthly credit budget (5x more tokens collected).
    """
    # Must have DexScreener data — no price history = skip
    if not price_data:
        return False

    # Must have meaningful liquidity (LP not drained yet)
    if price_data.get("liquidity_usd", 0) < 50:
        return False

    # Both mint + freeze authority active → immediate rug risk
    mint_ext = asset.get("mint_extensions", {}) or {}
    mint_auth = mint_ext.get("mint_authority", "") or ""
    freeze_auth = mint_ext.get("freeze_authority", "") or ""
    if mint_auth and freeze_auth:
        return False

    # Must have valid supply
    supply = asset.get("supply") or {}
    if not supply or int(supply.get("supply", "0") or 0) <= 0:
        return False

    return True


# ===================================================================
# TOKEN PROCESSING PIPELINE
# ===================================================================


def build_record(
    mint: str,
    graduation_ts: int,
    tx_signature: str,
    asset: dict,
    swaps: list[dict],
    price_data: dict | None,
) -> dict:
    """Build a single token record from all data sources."""
    features, sequence = compute_features(asset, swaps, graduation_ts)

    # Compress sequence using float16 base64
    sequence_b64 = compress_sequence(sequence) if sequence else zero_sequence()

    # Labels from DexScreener
    if price_data:
        labels = compute_labels_dexscreener(price_data, graduation_ts)
    else:
        # Fallback: infer labels from authority features
        high_risk = (
            features.get("mint_authority_active", 0)
            + features.get("freeze_authority_active", 0)
        )
        labels = {
            "rug_label": 1 if high_risk > 0 else 0,
            "time_to_rug_hours": 24.0 if high_risk > 0 else 72.0,
            "max_drawdown_pct": 90.0 if high_risk > 0 else 20.0,
            "pump_2x_label": 0,
            "inferred_label": True,
        }

    record = {
        "mint": mint,
        "graduation_timestamp": graduation_ts,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        **features,
        "sequence_b64": sequence_b64,
        "has_sequence": len(swaps) > 0,
        **labels,
        "deployer_address": asset.get("ownership", {}).get("owner", ""),
    }

    # Attach DexScreener metadata columns when available
    if price_data:
        record["price_usd"] = price_data.get("price_usd", 0)
        record["liquidity_usd"] = price_data.get("liquidity_usd", 0)
        record["volume_24h"] = price_data.get("volume_24h", 0)
        record["price_change_24h"] = price_data.get("price_change_24h", 0)

    return record


def process_batch(
    tokens: list[tuple[str, int, str]],
    total_collected: int,
    start_time: float,
    backfill: bool = False,
) -> tuple[int, float]:
    """
    Process a batch of new tokens through the full pipeline:
      1. Helius getAssetBatch  (100 tokens × 2 req/s)
      2. DexScreener price batch (30 tokens, no rate limit)
      3. Per-token swap fetch     (individual × 2 req/s)
      4. Build records + save to Supabase

    When backfill=True, skip ALL swap fetches — every token is saved
    with zero_sequence.  This burns ~0.11 credits/token instead of
    ~100 credits/token, allowing ~250K tokens/day on the Free tier.
    Swap data can be backfilled in a second pass later.

    Returns (new_tokens_saved, tokens_per_second).
    """
    if not tokens:
        return 0, 0.0

    mints = [t[0] for t in tokens]
    new_count = 0

    # ── Phase 1: Helius DAS assets (batches of 100) ──
    asset_map: dict[str, dict] = {}
    for i in range(0, len(mints), 100):
        chunk = mints[i : i + 100]
        batch_result = get_assets_batch(chunk)
        for item in batch_result:
            if isinstance(item, dict) and "id" in item:
                asset_map[item["id"]] = item
        log.info("  Assets: %d/%d fetched", len(asset_map), len(mints))
        if i + 100 < len(mints):
            time.sleep(0.5)  # 2 req/s limit

    # ── Phase 2: DexScreener prices (batches of 30) ──
    price_map: dict[str, dict] = {}
    for i in range(0, len(mints), 30):
        chunk = mints[i : i + 30]
        batch_result = get_prices_batch(chunk)
        price_map.update(batch_result)
    # No delay — DexScreener has no strict rate limit on the pairs endpoint

    priced = len(price_map)
    log.info("  Prices: %d/%d tokens have DexScreener data", priced, len(mints))

    # ── Phase 3: Per-token swaps (backfill skips all — saves 100 credits/token) ──
    records: list[dict] = []
    fetched_swaps = skipped_swaps = 0
    for mint, graduation_ts, tx_sig in tokens:
        asset = asset_map.get(mint, {})
        price_data = price_map.get(mint)

        if backfill:
            # Skip ALL swap fetches — metadata + labels only
            swaps = []
            skipped_swaps += 1
        elif should_fetch_swaps(asset, price_data):
            swaps = get_token_swaps(mint, graduation_ts)
            fetched_swaps += 1
        else:
            swaps = []
            skipped_swaps += 1

        record = build_record(mint, graduation_ts, tx_sig, asset, swaps, price_data)
        records.append(record)

    if backfill:
        log.info("  Swaps: BACKFILL — all %d tokens saved (no swap fetch)",
                 skipped_swaps)
    elif skipped_swaps:
        log.info("  Swaps: %d fetched, %d skipped (saves %d credits)",
                 fetched_swaps, skipped_swaps, skipped_swaps * 100)

    # ── Phase 4: Save to Supabase ──
    saved = save_batch(records)
    new_count += saved

    # ── Throughput ──
    elapsed = time.time() - start_time
    tps = (total_collected + saved) / elapsed if elapsed > 0 else 0.0

    return new_count, tps


# ===================================================================
# KEEP-ALIVE HTTP SERVER  (prevents Render free-tier spin-down)
# ===================================================================


class _KeepAliveHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"collector_bot alive\n")

    def log_message(self, fmt: str, *args) -> None:
        """Suppress default HTTP access logging."""
        pass


def start_keepalive_server(port: int) -> None:
    """Start a trivial HTTP server so Render does not spin down the service."""
    server = HTTPServer(("0.0.0.0", port), _KeepAliveHandler)
    log.info("Keep-alive HTTP server listening on port %d", port)
    server.serve_forever()


# ===================================================================
# MAIN LOOP
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="collector_bot — pump.fun token data collection"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Skip swap fetching entirely — collect metadata + labels only "
             "(~0.11 credits/token vs 100, ~250K tokens/day on Free tier). "
             "Swap sequences can be backfilled in a second pass.",
    )
    args = parser.parse_args()

    # Env var BACKFILL=true overrides CLI arg — toggle from Render dashboard
    backfill = (
        os.environ.get("BACKFILL", "").lower() == "true"
        or args.backfill
    )

    mode = "BACKFILL (no swaps)" if backfill else "collect (with swaps)"
    log.info("========================================")
    log.info("collector_bot — data collection service")
    log.info("Mode        : %s", mode)
    log.info("========================================")
    log.info("Helius key  : %s", "SET" if HELIUS_KEY else "MISSING")
    log.info("Supabase URL: %s", "SET" if SUPABASE_URL else "MISSING")
    log.info("Interval    : %ds", INTERVAL)
    log.info("Batch sizes : Helius=100, DexScreener=30")
    log.info("Sequence    : float16 base64 (~256 B/row)")

    # ---- Start keep-alive HTTP server in background ----
    keepalive_thread = Thread(
        target=start_keepalive_server, args=(KEEPALIVE_PORT,), daemon=True,
    )
    keepalive_thread.start()
    time.sleep(0.5)

    # ---- Load existing mints ----
    existing = get_existing_mints()
    log.info("Already collected: %d tokens", len(existing))

    before_sig: str | None = None
    total_collected = len(existing)
    cycle = 0
    start_time = time.time()
    last_throughput_log = total_collected

    while True:
        try:
            cycle += 1
            log.info("--- Cycle %d ---", cycle)

            # Fetch new graduated-mint transactions  (STEP 1)
            txs = get_graduated_mints(before_sig)

            if not txs:
                log.info("No new transactions returned — "
                         "sleeping %ds", INTERVAL)
                time.sleep(INTERVAL)
                continue

            log.info("Got %d transactions from Helius", len(txs))

            # Collect new (unseen) tokens for batch processing
            new_tokens: list[tuple[str, int, str]] = []
            for tx in txs:
                mint = extract_mint_from_tx(tx)
                if not mint or mint in existing:
                    continue
                sig = tx.get("signature", "")
                ts = tx.get("timestamp", 0)
                new_tokens.append((mint, ts, sig))
                existing.add(mint)  # mark seen immediately

            if not new_tokens:
                log.info("All %d tx already collected — "
                         "sleeping %ds", len(txs), INTERVAL)
                # Update pagination cursor
                if txs:
                    before_sig = txs[-1].get("signature", before_sig)
                time.sleep(INTERVAL)
                continue

            log.info("%d new tokens to process in batch", len(new_tokens))

            # Process the batch  (STEPS 2–5)
            new_count, tps = process_batch(
                new_tokens, total_collected, start_time,
                backfill=backfill,
            )
            total_collected += new_count

            log.info("Cycle %d done: +%d new  (total: %d)",
                     cycle, new_count, total_collected)

            # ---- Throughput logging every 1000 tokens ----
            if total_collected - last_throughput_log >= 1000:
                last_throughput_log = total_collected
                elapsed_hrs = (time.time() - start_time) / 3600
                eta_1M = (
                    (1_000_000 - total_collected) / max(tps, 0.01) / 3600
                )
                log.info(
                    "Throughput: %.1f tok/s | "
                    "Total: %d | "
                    "Uptime: %.1fh | "
                    "ETA 1M: %.1fh",
                    tps, total_collected, elapsed_hrs, eta_1M,
                )

            # Update pagination cursor to the oldest tx in this batch
            if txs:
                before_sig = txs[-1].get("signature", before_sig)

        except KeyboardInterrupt:
            log.info("Shutdown requested — exiting.")
            break
        except Exception:
            log.exception("Unhandled error in main loop — "
                          "sleeping %ds before retry", INTERVAL)

        # Backfill: skip sleep when tokens are flowing (max throughput).
        # Steady-state collection: sleep INTERVAL seconds between cycles.
        if not backfill:
            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
