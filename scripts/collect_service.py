"""
scripts/collect_service.py — Standalone data collection service.

Runs as a long-running Python service on Render.com free tier.
Continuously collects graduated pump.fun token data from Helius
and saves to Supabase.  Runs for days uninterrupted.

ENVIRONMENT VARIABLES (set in Render dashboard):
  HELIUS_API_KEY=
  SUPABASE_URL=
  SUPABASE_KEY=
  COLLECTION_INTERVAL_SECONDS=30
  PORT=8080                          # keep-alive HTTP server
"""

import os
import time
import json
import sys
import requests
import logging
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
# Environment
# ---------------------------------------------------------------------------
HELIUS_KEY = os.environ["HELIUS_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
INTERVAL = int(os.environ.get("COLLECTION_INTERVAL_SECONDS", 30))
KEEPALIVE_PORT = int(os.environ.get("PORT", 8080))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PUMP_MIGRATION_PROGRAM = (
    "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
)

# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ===================================================================
# HELIUS API HELPERS
# ===================================================================


def _helius_get(url: str, params: dict | None = None) -> dict | list | None:
    """GET request to Helius REST API with retry + rate-limit handling."""
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = (2 ** attempt) + 1
                log.warning("Rate limited, waiting %ds", wait)
                time.sleep(wait)
            else:
                log.error("Helius GET %s → HTTP %d", url[:80], r.status_code)
                return None
        except requests.RequestException as e:
            log.error("Helius GET failed: %s", e)
            time.sleep(5)
    return None


def _helius_rpc(method: str, params: dict) -> dict | None:
    """JSON-RPC call to Helius RPC endpoint."""
    url = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    for attempt in range(5):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = (2 ** attempt) + 1
                log.warning("RPC rate limited, waiting %ds", wait)
                time.sleep(wait)
            else:
                log.error("Helius RPC %s → HTTP %d", method, r.status_code)
                return None
        except requests.RequestException as e:
            log.error("Helius RPC failed: %s", e)
            time.sleep(5)
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


# ===================================================================
# MINT EXTRACTION
# ===================================================================


def extract_mint_from_tx(tx: dict) -> str | None:
    """
    Extract the token mint address from a Helius enhanced transaction.

    Pump.fun migration transactions contain token-balance changes or
    token-transfers that reveal the mint being migrated to Raydium.
    """
    # Strategy 1 — tokenTransfers array (Helius enhanced)
    for transfer in tx.get("tokenTransfers", []):
        mint = transfer.get("mint")
        if mint and mint != PUMP_MIGRATION_PROGRAM:
            return mint

    # Strategy 2 — accountData (Helius enhanced)
    for acct in tx.get("accountData", []):
        if acct.get("account") == PUMP_MIGRATION_PROGRAM:
            continue
        # Native SOL mint is all-zeroes or a well-known constant
        raw = acct.get("raw", "")
        if raw and len(raw) > 32:
            return acct["account"]

    # Strategy 3 — tokenBalanceChanges
    for change in tx.get("tokenBalanceChanges", []):
        mint = change.get("mint")
        if mint and mint != PUMP_MIGRATION_PROGRAM:
            return mint

    # Strategy 4 — logMessages parsing (fallback)
    for msg in tx.get("logMessages", []):
        if "mint" in msg.lower() or "token" in msg.lower():
            for word in msg.split():
                if len(word) >= 43 and len(word) <= 45:
                    return word

    return None


# ===================================================================
# DATA FETCHING
# ===================================================================


def get_new_graduated_mints(
    before_sig: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    Fetch recently graduated pump.fun tokens from the Helius
    migration-program address-transactions endpoint.
    """
    url = (
        f"https://api.helius.xyz/v0/addresses/"
        f"{PUMP_MIGRATION_PROGRAM}/transactions"
    )
    params: dict = {"api-key": HELIUS_KEY, "limit": limit}
    if before_sig:
        params["before"] = before_sig

    result = _helius_get(url, params)
    if isinstance(result, list):
        return result
    return []


def get_token_asset(mint: str) -> dict:
    """Get token metadata from Helius Digital Asset Standard API."""
    resp = _helius_rpc("getAsset", {"id": mint})
    if resp and "result" in resp:
        return resp["result"]
    return {}


def get_token_swaps(
    mint: str,
    from_ts: int,
    to_ts: int,
) -> list[dict]:
    """
    Get SWAP-type transactions for a token mint within a time window.
    Uses Helius enhanced-transactions endpoint filtered by type=SWAP.
    """
    url = f"https://api.helius.xyz/v0/addresses/{mint}/transactions"
    params: dict = {"api-key": HELIUS_KEY, "limit": 100, "type": "SWAP"}

    result = _helius_get(url, params)
    if not isinstance(result, list):
        return []

    return [
        tx for tx in result
        if from_ts <= tx.get("timestamp", 0) <= to_ts
    ]


# ===================================================================
# FEATURE ENGINEERING  (14 features)
# ===================================================================


def compute_features(
    mint: str,
    graduation_ts: int,
    asset: dict,
    swaps: list[dict],
) -> tuple[dict, list[list]]:
    """Compute 14 normalised features and a 16-row temporal sequence."""
    features: dict = {}

    # ---- authority features (from DAS asset) ----
    authorities = asset.get("authorities", [])
    features["mint_authority_active"] = float(
        any("mint" in a.get("scopes", []) for a in authorities)
    )
    features["freeze_authority_active"] = float(
        any("freeze" in a.get("scopes", []) for a in authorities)
    )
    features["mutable_metadata"] = float(asset.get("mutable", True))

    # ---- default liquidity features ----
    features["lp_burn_pct"] = 0.9
    features["initial_liquidity_sol"] = 0.0
    features["liquidity_concentration"] = 0.0
    features["dev_hold_pct"] = 0.0
    features["top10_holder_pct"] = 0.0
    features["bundle_wallet_count"] = 0.0
    features["migration_speed_seconds"] = 0.5

    if not swaps:
        features["buy_sell_ratio_60s"] = 0.5
        features["price_velocity_60s"] = 0.0
        features["unique_buyers_60s"] = 0.0
        features["avg_transaction_size_sol"] = 0.0
        return features, []

    # ---- first-swap liquidity ----
    first = swaps[0]
    swap_ev = first.get("events", {}).get("swap", {})
    native_in = (swap_ev.get("nativeInput") or {}).get("amount", 0)
    pool_sol = abs(native_in) / 1e9 if native_in else 0.0
    features["initial_liquidity_sol"] = min(pool_sol / 1000, 1.0)

    # ---- 60-second window metrics ----
    swaps_60s = [s for s in swaps if s.get("timestamp", 0) <= graduation_ts + 60]

    buyers: set[str] = set()
    buy_vol = sell_vol = 0.0
    prices: list[float] = []
    sequence: list[list] = []

    for i, swap in enumerate(swaps):
        ev = swap.get("events", {}).get("swap", {})
        signer = swap.get("feePayer", "")
        nat_in = (ev.get("nativeInput") or {}).get("amount", 0) or 0
        nat_out = (ev.get("nativeOutput") or {}).get("amount", 0) or 0
        tok_out = ev.get("tokenOutputs", [{}])

        # Price per token  (SOL spent ÷ tokens received)
        sol_in = nat_in / 1e9
        tok_amt = float(tok_out[0].get("tokenAmount", 0)) if tok_out else 0
        price = (sol_in / tok_amt) if tok_amt > 0 else 0.0

        if nat_in:
            buyers.add(signer)
            buy_vol += nat_in
        if nat_out:
            sell_vol += nat_out

        if price > 0:
            prices.append(price)

        # Build 16-row sequence (padded)
        if i < 16:
            total_nat = buy_vol + sell_vol
            sequence.append([
                len(buyers),                              # holders (cumulative buyers)
                pool_sol,                                 # liquidity
                total_nat / 1e9,                          # volume
                (buy_vol / total_nat) if total_nat > 0 else 0.5,  # buy ratio
                sol_in,                                   # velocity
                i + 1,                                    # tx_count
            ])

    # Pad sequence to exactly 16 rows
    while len(sequence) < 16:
        sequence.append([0, 0, 0, 0.5, 0, 0])

    # ---- aggregate metrics ----
    total_vol = buy_vol + sell_vol
    features["buy_sell_ratio_60s"] = (
        (buy_vol / total_vol) if total_vol > 0 else 0.5
    )
    features["unique_buyers_60s"] = min(len(buyers) / 100, 1.0)

    if len(prices) >= 2 and prices[0] > 0:
        change = (prices[-1] - prices[0]) / prices[0]
        features["price_velocity_60s"] = max(min(change, 1.0), -1.0)
    else:
        features["price_velocity_60s"] = 0.0

    avg_swap = ((buy_vol / len(buyers)) / 1e9) if buyers else 0.0
    features["avg_transaction_size_sol"] = min(avg_swap / 10, 1.0)

    return features, sequence[:16]


# ===================================================================
# LABEL COMPUTATION  (rug / drawdown / pump-2x)
# ===================================================================


def compute_labels(
    swaps: list[dict],
    graduation_ts: int,
) -> dict | None:
    """
    Compute rug-detection labels from 72-hour post-graduation swap history.

    Returns dict with:
      rug_label         0 | 1
      time_to_rug_hours float (≤72)
      max_drawdown_pct  float
      pump_2x_label     0 | 1
    """
    if not swaps:
        return None

    # Aggregate hourly prices
    prices_by_hour: dict[int, list[float]] = {}
    for swap in swaps:
        ts = swap.get("timestamp", 0)
        hour = int((ts - graduation_ts) / 3600)
        if not (0 <= hour <= 72):
            continue

        ev = swap.get("events", {}).get("swap", {})
        nat_in = (ev.get("nativeInput") or {}).get("amount", 0)
        tok_out = ev.get("tokenOutputs", [{}])
        sol = nat_in / 1e9
        tok = float(tok_out[0].get("tokenAmount", 0)) if tok_out else 0

        if tok > 0 and sol > 0:
            prices_by_hour.setdefault(hour, []).append(sol / tok)

    if not prices_by_hour:
        return None

    hours = sorted(prices_by_hour.keys())
    prices = [sum(prices_by_hour[h]) / len(prices_by_hour[h]) for h in hours]

    if len(prices) < 2:
        return None

    entry = prices[0]
    peak = max(prices)
    low = min(prices)

    max_dd = ((peak - low) / peak * 100) if peak > 0 else 0.0
    pump_2x = 1 if peak >= entry * 2 else 0

    # Rug detection: ≥80 % drawdown from running peak
    rug = 0
    rug_time = 72.0
    running_peak = entry

    for i, p in enumerate(prices):
        running_peak = max(running_peak, p)
        if running_peak > 0:
            dd = (running_peak - p) / running_peak * 100
            if dd >= 80:
                rug = 1
                rug_time = float(hours[i])
                break

    return {
        "rug_label": rug,
        "time_to_rug_hours": rug_time,
        "max_drawdown_pct": max_dd,
        "pump_2x_label": pump_2x,
    }


# ===================================================================
# PERSISTENCE
# ===================================================================


def save_to_supabase(record: dict) -> bool:
    """Upsert a token record into the training_tokens table."""
    try:
        supabase.table("training_tokens").upsert(
            record, on_conflict="mint",
        ).execute()
        return True
    except Exception as e:
        log.error("Supabase upsert failed for %s: %s", record.get("mint", "?"), e)
        return False


# ===================================================================
# TOKEN PROCESSING PIPELINE
# ===================================================================


def process_token(mint: str, graduation_ts: int, tx_signature: str) -> bool:
    """Full pipeline: fetch data → compute features & labels → save."""
    log.info("Processing %s...", mint[:12])

    # 1. Get DAS asset metadata
    asset = get_token_asset(mint)
    time.sleep(0.3)

    # 2. Get 72-hour swap history  (259 200 seconds)
    swaps = get_token_swaps(mint, graduation_ts, graduation_ts + 259200)
    time.sleep(0.3)

    # 3. Compute features + temporal sequence
    features, sequence = compute_features(mint, graduation_ts, asset, swaps)

    # 4. Compute labels
    labels = compute_labels(swaps, graduation_ts)
    inferred = False
    if labels is None:
        # Infer labels from authority features when no price data exists
        high_risk = (
            features.get("mint_authority_active", 0)
            + features.get("freeze_authority_active", 0)
        )
        labels = {
            "rug_label": 1 if high_risk > 0 else 0,
            "time_to_rug_hours": 24.0 if high_risk > 0 else 72.0,
            "max_drawdown_pct": 90.0 if high_risk > 0 else 20.0,
            "pump_2x_label": 0,
        }
        inferred = True

    # 5. Build record
    record = {
        "mint": mint,
        "graduation_timestamp": graduation_ts,
        "tx_signature": tx_signature,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        **features,
        "sequence": json.dumps(sequence),
        "has_sequence": len(swaps) > 0,
        **labels,
        "inferred_label": inferred,
        "deployer_address": asset.get("ownership", {}).get("owner", ""),
    }

    return save_to_supabase(record)


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
    log.info("========================================")
    log.info("collector_bot — data collection service")
    log.info("========================================")
    log.info("Helius key  : %s", "SET" if HELIUS_KEY else "MISSING")
    log.info("Supabase URL: %s", "SET" if SUPABASE_URL else "MISSING")
    log.info("Interval    : %ds", INTERVAL)

    # ---- Start keep-alive HTTP server in background ----
    keepalive_thread = Thread(target=start_keepalive_server, args=(KEEPALIVE_PORT,), daemon=True)
    keepalive_thread.start()
    time.sleep(0.5)

    # ---- Load existing mints ----
    existing = get_existing_mints()
    log.info("Already collected: %d tokens", len(existing))

    before_sig: str | None = None
    total_collected = len(existing)
    cycle = 0

    while True:
        try:
            cycle += 1
            log.info("--- Cycle %d ---", cycle)

            # Fetch new graduated-mint transactions
            txs = get_new_graduated_mints(before_sig)

            if not txs:
                log.info("No new transactions returned — "
                         "sleeping %ds", INTERVAL)
                time.sleep(INTERVAL)
                continue

            log.info("Got %d transactions from Helius", len(txs))

            new_count = 0
            for tx in txs:
                mint = extract_mint_from_tx(tx)
                if not mint:
                    continue
                if mint in existing:
                    continue

                sig = tx.get("signature", "")
                ts = tx.get("timestamp", 0)

                if process_token(mint, ts, sig):
                    existing.add(mint)
                    new_count += 1
                    total_collected += 1

                # Small delay between tokens to respect rate limits
                time.sleep(0.8)

            log.info("Cycle %d done: +%d new  (total: %d)",
                     cycle, new_count, total_collected)

            # Update pagination cursor to the oldest tx in this batch
            if txs:
                before_sig = txs[-1].get("signature", before_sig)

        except KeyboardInterrupt:
            log.info("Shutdown requested — exiting.")
            break
        except Exception:
            log.exception("Unhandled error in main loop — "
                          "sleeping %ds before retry", INTERVAL)

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
