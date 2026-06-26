"""
scripts/collect_service.py — Snapshot-based pump.fun token data collection service.

Runs as a long-running Python service on Render.com free tier.
Detects pump.fun graduations and collects time-series snapshots at
T0, T0+1m, T0+5m, T0+15m with 50 post-migration trading features.

DATA SOURCES:
  - Helius API    (migration detection, swap transactions, DAS metadata)
  - DexScreener   (price, liquidity)

COLLECTION STRATEGY:
  When a token graduates (T0 = migration detected):
    T0+1m   →  DexScreener price + Helius swaps [T0, T0+60s]
    T0+5m   →  DexScreener price + Helius swaps [T0, T0+300s]
    T0+15m  →  DexScreener price + Helius swaps [T0, T0+900s]
                →  compute all features → save to Supabase

ENVIRONMENT VARIABLES (set in Render dashboard):
  HELIUS_API_KEY=
  SUPABASE_URL=
  SUPABASE_KEY=
  COLLECTION_INTERVAL_SECONDS=10
  QUALITY_CHECK_INTERVAL=100
  PORT=8080
"""

import os
import socket
import time
import json
import sys
import argparse
import logging
import requests
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Lock
from typing import Optional

from supabase import create_client, Client

from features import compute_all_features, parse_swaps_for_window
from quality_validator import QualityValidator, run_quality_check
from axiom_service import AxiomService, get_axiom_service  # uses Mobula under the hood

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
MOBULA_API_KEY = os.environ.get("MOBULA_API_KEY", "")
AXIOM_ENABLED = os.environ.get("MOBULA_ENABLED", "true").lower() == "true" and bool(MOBULA_API_KEY)
INTERVAL = int(os.environ.get("COLLECTION_INTERVAL_SECONDS", "10"))
QUALITY_CHECK_INTERVAL = int(os.environ.get("QUALITY_CHECK_INTERVAL", "100"))
KEEPALIVE_PORT = int(os.environ.get("PORT", "8080"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PUMP_MIGRATION_PROGRAM = (
    "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
)

SKIP_MINTS: frozenset[str] = frozenset({
    "So11111111111111111111111111111111111111112",   # Wrapped SOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
    "11111111111111111111111111111111",               # Native SOL
})

SNAPSHOT_WINDOWS = {
    "t1m": 60,
    "t5m": 300,
    "t15m": 900,
}

# Maps scheduler window labels → features.py window keys
WINDOW_KEY_MAP = {"t1m": "1m", "t5m": "5m", "t15m": "15m"}

# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ===================================================================
# SNAPSHOT SCHEDULER
# ===================================================================


class SnapshotScheduler:
    """
    In-memory scheduler tracking tokens from T0 detection through
    T0+1m, T0+5m, T0+15m snapshot collection.

    State machine: pending → collecting → complete
    """

    def __init__(self) -> None:
        self._lock = Lock()
        # {mint: {"t0": ts, "symbol": str, "tx_sig": str,
        #         "snapshots": {"t0": {...}, "t1m": {...}, ...},
        #         "state": "pending"|"collecting"|"complete",
        #         "created_at": float}}
        self._tokens: dict[str, dict] = {}

    def schedule(self, mint: str, t0_ts: int, tx_sig: str, symbol: str = "") -> None:
        """Register a newly detected migration for snapshot tracking."""
        with self._lock:
            if mint in self._tokens:
                return
            self._tokens[mint] = {
                "t0": t0_ts,
                "symbol": symbol,
                "tx_sig": tx_sig,
                "snapshots": {},
                "state": "pending",
                "created_at": time.time(),
            }

    def get_due_snapshots(self, now_ts: int) -> list[tuple[str, str]]:
        """
        Return list of (mint, window_label) that are due for collection.
        window_label is one of "t1m", "t5m", "t15m".
        """
        due: list[tuple[str, str]] = []
        with self._lock:
            for mint, data in list(self._tokens.items()):
                if data["state"] == "complete":
                    continue
                t0 = data["t0"]
                elapsed = now_ts - t0
                for window_label, window_sec in SNAPSHOT_WINDOWS.items():
                    if window_label not in data["snapshots"] and elapsed >= window_sec:
                        due.append((mint, window_label))
        return due

    def record_snapshot(self, mint: str, window_label: str, data: dict) -> None:
        """Store collected snapshot data."""
        with self._lock:
            token = self._tokens.get(mint)
            if not token:
                return
            token["snapshots"][window_label] = data
            token["state"] = "collecting"

    def finalize(self, mint: str) -> Optional[dict]:
        """
        Mark a token as complete and return all accumulated data.
        Returns None if the token was not found or not ready.
        """
        with self._lock:
            token = self._tokens.get(mint)
            if not token:
                return None
            if "t15m" not in token["snapshots"]:
                return None  # not ready yet
            token["state"] = "complete"
            return dict(token)  # return a copy

    def remove(self, mint: str) -> None:
        """Remove a token from the scheduler after it's been saved."""
        with self._lock:
            self._tokens.pop(mint, None)

    def cleanup_stale(self, max_age_seconds: int = 1200) -> int:
        """
        Remove tokens that never completed within max_age_seconds.
        Returns count of removed tokens.
        """
        now = time.time()
        removed = 0
        with self._lock:
            for mint, data in list(self._tokens.items()):
                if data["state"] == "complete":
                    self._tokens.pop(mint, None)
                    removed += 1
                elif now - data["created_at"] > max_age_seconds:
                    log.warning("Removing stale token %s (age: %.0fs, state: %s)",
                                mint[:12], now - data["created_at"], data["state"])
                    self._tokens.pop(mint, None)
                    removed += 1
        return removed

    def get_t0(self, mint: str) -> int:
        """Return the T0 timestamp for a tracked mint, or 0 if not found."""
        with self._lock:
            token = self._tokens.get(mint)
            return token["t0"] if token else 0

    def pending_count(self) -> int:
        """Number of tokens currently being tracked."""
        with self._lock:
            return sum(1 for d in self._tokens.values() if d["state"] != "complete")

    def __len__(self) -> int:
        with self._lock:
            return len(self._tokens)


# ===================================================================
# API HELPERS
# ===================================================================

DNS_WARMUP_HOSTS = [
    "api.helius.xyz",
    "mainnet.helius-rpc.com",
]


def wait_for_dns(
    hosts: list[str] | None = None,
    timeout_seconds: int = 180,
    check_interval: float = 10.0,
) -> bool:
    """
    Block until DNS resolution succeeds for all required hosts, or timeout.

    Render free-tier cold-starts delay DNS resolver initialization by 1-3 minutes.
    This gate prevents the main loop from burning retry budgets on infrastructure-
    level name-resolution failures that no amount of exponential backoff can fix.

    Args:
        hosts: Hostnames to check. Defaults to DNS_WARMUP_HOSTS.
        timeout_seconds: Maximum time to wait for all hosts to resolve.
        check_interval: Seconds between resolution attempts.

    Returns True if all hosts resolved, False if timeout elapsed.
    """
    if hosts is None:
        hosts = list(DNS_WARMUP_HOSTS)

    deadline = time.time() + timeout_seconds
    unresolved = set(hosts)

    while unresolved and time.time() < deadline:
        still_down = set()
        for host in unresolved:
            try:
                socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
                log.info("DNS resolved: %s", host)
            except socket.gaierror:
                still_down.add(host)

        unresolved = still_down

        if not unresolved:
            log.info("All DNS hosts resolved — entering main loop")
            return True

        remaining = max(0.0, deadline - time.time())
        log.info(
            "DNS pending for %d host(s): %s — retrying in %.0fs (%.0fs remaining)",
            len(unresolved),
            ", ".join(sorted(unresolved)),
            check_interval,
            remaining,
        )
        time.sleep(min(check_interval, remaining))

    if unresolved:
        log.error(
            "DNS warmup timed out after %ds — unresolvable: %s",
            timeout_seconds,
            ", ".join(sorted(unresolved)),
        )
        return False
    return True


def with_retry(fn, max_retries=5, base_wait=1.0, name: str = ""):
    """Exponential backoff retry for all API calls.

    DNS-resolution failures (socket.gaierror) are logged at INFO level
    rather than WARNING because they are expected during Render cold-starts
    and are handled by the pre-flight wait_for_dns() gate.

    Args:
        fn: Callable that returns a result or None on soft failure.
        max_retries: Maximum number of attempts.
        base_wait: Base wait time in seconds for exponential backoff.
        name: Human-readable label for log messages (e.g. "DexScreener:price").
    """
    tag = f"[{name}] " if name else ""
    for attempt in range(max_retries):
        try:
            result = fn()
            if result is not None:
                return result
            # Soft failure — fn() returned None (rate-limited, no data, etc.)
            if attempt < max_retries - 1:
                log.info("%sAttempt %d returned empty (rate limit / no data)", tag, attempt + 1)
        except Exception as e:
            # Distinguish DNS failures from application errors
            if _is_dns_error(e):
                log.info("%sAttempt %d DNS not ready: %s", tag, attempt + 1, e)
            else:
                log.warning("%sAttempt %d failed: %s", tag, attempt + 1, e)
        if attempt < max_retries - 1:
            wait = (base_wait * 2 ** attempt) + (0.1 * attempt)
            log.info("%sWaiting %.1fs before retry", tag, wait)
            time.sleep(wait)
    return None


def _is_dns_error(exc: Exception) -> bool:
    """Return True if the exception (or its cause chain) is a DNS failure."""
    cur: BaseException | None = exc
    while cur is not None:
        if isinstance(cur, socket.gaierror):
            return True
        # requests wraps DNS errors in various exception types — check the message
        msg = str(cur).lower()
        if any(phrase in msg for phrase in (
            "name resolution",
            "failed to resolve",
            "temporary failure in name resolution",
            "nodata",
            "nxdomain",
        )):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


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


# ===================================================================
# MINT EXTRACTION
# ===================================================================


def _is_valid_mint(mint: str) -> bool:
    """Return False for known non-pump.fun mints (WSOL, USDC, etc.)."""
    return mint not in SKIP_MINTS and mint != PUMP_MIGRATION_PROGRAM


def extract_mint_from_tx(tx: dict) -> Optional[str]:
    """
    Extract the token mint address from a Helius enhanced transaction.
    """
    # Strategy 1 — tokenTransfers
    for transfer in tx.get("tokenTransfers", []):
        mint = transfer.get("mint")
        if mint and _is_valid_mint(mint):
            return mint

    # Strategy 2 — accountData
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

    # Strategy 4 — logMessages parsing
    for msg in tx.get("logMessages", []):
        if "mint" in msg.lower() or "token" in msg.lower():
            for word in msg.split():
                if len(word) >= 43 and len(word) <= 45 and _is_valid_mint(word):
                    return word

    return None


# ===================================================================
# HELIUS API — MIGRATION DETECTION
# ===================================================================


def get_graduated_mints(
    before_sig: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """
    Fetches recently graduated pump.fun tokens from migration program
    transaction history. ~0.5s delay between calls (2 req/s).
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
            log.warning("Helius rate limited (migration detection)")
            return None
        r.raise_for_status()
        return r.json()

    result = with_retry(fetch, name="Helius:migration")
    time.sleep(0.5)  # 2 req/s limit
    return result or []


# ===================================================================
# HELIUS API — DAS ASSET METADATA (for T0 symbol + holders)
# ===================================================================


def get_asset(mint: str) -> dict:
    """
    Fetch DAS asset metadata for a single mint.
    Used at T0 for symbol and initial holder count.
    """
    url = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAsset",
        "params": {
            "id": mint,
            "displayOptions": {"showFungible": True},
        },
    }

    def fetch():
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 429:
            return None
        r.raise_for_status()
        return r.json().get("result", {})

    result = with_retry(fetch, max_retries=3, base_wait=0.5, name="Helius:getAsset")
    time.sleep(0.3)
    return result or {}


# ===================================================================
# HELIUS API — SWAP TRANSACTIONS PER WINDOW
# ===================================================================


def get_token_swaps_window(
    mint: str,
    from_ts: int,
    to_ts: int,
    limit: int = 50,
) -> list[dict]:
    """
    Fetch swap transactions for a token within a time window.
    Filters swaps where timestamp is in [from_ts, to_ts].

    Uses Helius enhanced transactions endpoint with type=SWAP.
    ~0.5s delay between calls (2 req/s).
    """
    url = f"https://api.helius.xyz/v0/addresses/{mint}/transactions"
    params: dict = {
        "api-key": HELIUS_KEY,
        "limit": limit,
        "type": "SWAP",
    }

    def fetch():
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 429:
            return None
        r.raise_for_status()
        txs = r.json()
        # Filter to time window
        return [
            tx for tx in txs
            if from_ts <= tx.get("timestamp", 0) <= to_ts
        ]

    result = with_retry(fetch, max_retries=3, base_wait=0.5, name="Helius:swaps")
    time.sleep(0.5)  # 2 req/s limit
    return result or []


# ===================================================================
# DEXSCREENER API — SINGLE-TOKEN PRICE SNAPSHOT
# ===================================================================


def get_price_snapshot(mint: str) -> dict:
    """
    Fetch current price + liquidity for a single mint from DexScreener.

    Returns dict with: price_usd, liquidity_usd, fdv, pair_created_at
    """
    url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"

    def fetch():
        r = requests.get(url, timeout=10)
        if r.status_code == 429:
            return None
        r.raise_for_status()
        return r.json()

    result = with_retry(fetch, max_retries=2, base_wait=0.5, name="DexScreener:price")

    if not result:
        return {}

    pairs = result.get("pairs", []) or []
    if not pairs:
        return {}

    # Use the first Solana/Raydium pair
    pair = pairs[0]
    return {
        "price_usd": float(pair.get("priceUsd", 0) or 0),
        "liquidity_usd": float(
            (pair.get("liquidity") or {}).get("usd", 0) or 0
        ),
        "fdv": float(pair.get("fdv", 0) or 0),
        "pair_created_at": pair.get("pairCreatedAt", 0),
    }


def get_price_data_24h(mint: str) -> Optional[dict]:
    """
    Fetch 24h price change + liquidity from DexScreener for label computation.
    Returns dict with: price_change_24h, liquidity_usd, volume_24h
    """
    url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"

    def fetch():
        r = requests.get(url, timeout=10)
        if r.status_code == 429:
            return None
        r.raise_for_status()
        return r.json()

    result = with_retry(fetch, max_retries=2, base_wait=0.5, name="DexScreener:24h")

    if not result:
        return None

    pairs = result.get("pairs", []) or []
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


# ===================================================================
# RECORD BUILDER
# ===================================================================


def build_record(
    mint: str,
    t0_ts: int,
    scheduler_data: dict,
) -> dict:
    """
    Build a complete training_tokens record from scheduler-accumulated
    snapshot data using the features.py compute_all_features() function.
    """
    snapshots_raw = scheduler_data.get("snapshots", {})

    # Build snapshots dict for features.py (price + liquidity per timestamp)
    snapshots: dict[str, dict] = {}
    for window_label in ("t0", "t1m", "t5m", "t15m"):
        snap = snapshots_raw.get(window_label, {})
        price = snap.get("price", {})
        snapshots[window_label] = {
            "price_usd": price.get("price_usd", 0) if price else 0,
            "liquidity_usd": price.get("liquidity_usd", 0) if price else 0,
            "timestamp": t0_ts + SNAPSHOT_WINDOWS.get(window_label, 0) if window_label != "t0" else t0_ts,
        }

    # Build swaps_by_window dict for features.py
    swaps_by_window: dict[str, list] = {}
    for window_label in ("t1m", "t5m", "t15m"):
        snap = snapshots_raw.get(window_label, {})
        raw_swaps = snap.get("swaps", [])
        window_start = t0_ts
        window_end = t0_ts + SNAPSHOT_WINDOWS[window_label]
        parsed = parse_swaps_for_window(raw_swaps, window_start, window_end, token_mint=mint)
        feature_key = WINDOW_KEY_MAP[window_label]
        swaps_by_window[feature_key] = parsed

    # 24h data for labels (fetched after 15m when we have mature data)
    price_24h = get_price_data_24h(mint)

    # Compute all features
    features = compute_all_features(
        snapshots=snapshots,
        swaps_by_window=swaps_by_window,
        price_data_24h=price_24h,
    )

    # ---- Axiom: wallet-intelligence features (optional) ----
    axiom_features: dict = {}
    if AXIOM_ENABLED:
        try:
            axiom_svc = get_axiom_service()
            axiom_features = axiom_svc.collect_for_token(mint, t0_ts, swaps_by_window)
            if axiom_features.get("axiom_collected"):
                log.debug("  Axiom: %d features for %s (cost: $%.6f)",
                         len(axiom_features) - 2, mint[:10],
                         axiom_features.get("axiom_cost_usd", 0))
        except Exception:
            log.exception("Axiom collection failed for %s", mint[:10])
            axiom_features = {"axiom_collected": False, "axiom_cost_usd": 0.0}
    else:
        axiom_features = {"axiom_collected": False, "axiom_cost_usd": 0.0}

    # Build record
    record = {
        "mint": mint,
        "symbol": scheduler_data.get("symbol", ""),
        "graduation_timestamp": t0_ts,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "deployer_address": "",  # populated from DAS if available
        **features,
        **axiom_features,
    }

    # Debug: log swap classification breakdown per window
    total_buys = sum(
        features.get(f"buy_count_{w}", 0) for w in ("1m", "5m", "15m")
    )
    total_sells = sum(
        features.get(f"sell_count_{w}", 0) for w in ("1m", "5m", "15m")
    )
    log.info("  Record %s: buys=%d sells=%d vol=$%.0f liq=$%.0f",
             mint[:10], total_buys, total_sells,
             features.get("volume_15m", 0),
             features.get("liquidity_usd_15m", 0))

    return record


# ===================================================================
# SNAPSHOT COLLECTION ORCHESTRATOR
# ===================================================================


def collect_snapshot_for_token(
    mint: str,
    t0_ts: int,
    window_label: str,
) -> dict:
    """
    Collect all data needed for a single snapshot window.

    For each window, fetches:
      - DexScreener price/liquidity
      - Helius swap transactions within [T0, T0+window_seconds]

    Returns a dict to store in the scheduler.
    """
    window_sec = SNAPSHOT_WINDOWS[window_label]
    from_ts = t0_ts
    to_ts = t0_ts + window_sec

    # Fetch price from DexScreener
    price = get_price_snapshot(mint)

    # Fetch swaps from Helius
    swaps = get_token_swaps_window(mint, from_ts, to_ts)

    return {
        "price": price,
        "swaps": swaps,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def collect_t0_snapshot(mint: str, t0_ts: int) -> dict:
    """
    Collect T0 baseline snapshot immediately after migration detection.
    Tries to get DexScreener price (may not exist yet for brand-new pairs)
    and DAS metadata for symbol.
    """
    price = get_price_snapshot(mint)
    asset = get_asset(mint)

    symbol = ""
    holders = 0
    deployer = ""
    if asset:
        content = asset.get("content", {}) or {}
        metadata = content.get("metadata", {}) or {}
        symbol = metadata.get("symbol", "")
        supply_info = asset.get("supply", {}) or {}
        holders = int(supply_info.get("holderCount", 0) or 0)
        ownership = asset.get("ownership", {}) or {}
        deployer = ownership.get("owner", "")

    return {
        "price": price,
        "asset": {
            "holder_count": holders,
            "symbol": symbol,
            "deployer": deployer,
        },
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


# ===================================================================
# KEEP-ALIVE HTTP SERVER
# ===================================================================


class _KeepAliveHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"collector_bot alive\n")

    def log_message(self, fmt: str, *args) -> None:
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
        description="collector_bot — snapshot-based pump.fun token data collection"
    )
    parser.add_argument(
        "--quality-report-only",
        action="store_true",
        help="Run quality validation on existing data and exit.",
    )
    args = parser.parse_args()

    log.info("========================================")
    log.info("collector_bot v2 — snapshot data collection")
    log.info("========================================")
    log.info("Helius key   : %s", "SET" if HELIUS_KEY else "MISSING")
    log.info("Supabase URL : %s", "SET" if SUPABASE_URL else "MISSING")
    log.info("Mobula API   : %s", "ENABLED" if AXIOM_ENABLED else "DISABLED")
    log.info("Poll interval: %ds", INTERVAL)
    log.info("Quality check: every %d tokens", QUALITY_CHECK_INTERVAL)
    log.info("Snapshots    : T0, T0+1m, T0+5m, T0+15m")
    log.info("Features     : 50 post-migration trading features")

    # ---- Quality-report-only mode ----
    if args.quality_report_only:
        log.info("Running quality report on existing data...")
        try:
            result = supabase.table("training_tokens").select("*").limit(500).execute()
            rows = result.data if result.data else []
            if rows:
                report = run_quality_check(rows, write_report=True)
                print(report)
            else:
                log.warning("No rows found for quality check.")
        except Exception as e:
            log.error("Quality report failed: %s", e)
        return

    # ---- Start keep-alive HTTP server in background ----
    keepalive_thread = Thread(
        target=start_keepalive_server, args=(KEEPALIVE_PORT,), daemon=True,
    )
    keepalive_thread.start()
    time.sleep(0.5)

    # ---- DNS warmup (Render free-tier cold-start mitigations) ----
    _hosts = list(DNS_WARMUP_HOSTS)
    if SUPABASE_URL:
        try:
            from urllib.parse import urlparse
            _supa_host = urlparse(SUPABASE_URL).hostname
            if _supa_host:
                _hosts.append(_supa_host)
        except Exception:
            pass
    if AXIOM_ENABLED:
        _hosts.append("graphql.mobula.io")

    log.info("DNS warmup: checking %d host(s) before starting collection", len(_hosts))
    wait_for_dns(hosts=_hosts, timeout_seconds=180, check_interval=5.0)

    # ---- Load existing mints ----
    existing = get_existing_mints()
    log.info("Already collected: %d tokens", len(existing))

    # ---- Initialize scheduler + quality validator ----
    scheduler = SnapshotScheduler()
    quality = QualityValidator()

    before_sig: Optional[str] = None
    total_collected = len(existing)
    last_quality_check = total_collected
    cycle = 0
    start_time = time.time()

    while True:
        try:
            cycle += 1
            now_ts = int(time.time())

            # =====================================================
            # STEP 1: Detect new migrations
            # =====================================================
            txs = get_graduated_mints(before_sig)

            if txs:
                new_count = 0
                for tx in txs:
                    mint = extract_mint_from_tx(tx)
                    if not mint or mint in existing:
                        continue
                    sig = tx.get("signature", "")
                    t0_ts = tx.get("timestamp", 0)

                    # Collect T0 baseline immediately
                    t0_data = collect_t0_snapshot(mint, t0_ts)
                    symbol = ""
                    if t0_data.get("asset", {}).get("symbol"):
                        symbol = t0_data["asset"]["symbol"]

                    # Schedule future snapshots
                    scheduler.schedule(mint, t0_ts, sig, symbol)
                    scheduler.record_snapshot(mint, "t0", t0_data)
                    existing.add(mint)
                    new_count += 1

                if new_count:
                    log.info("Cycle %d: detected %d new migrations (tracking: %d pending)",
                             cycle, new_count, scheduler.pending_count())

                # Update pagination cursor
                before_sig = txs[-1].get("signature", before_sig)

            # =====================================================
            # STEP 2: Collect due snapshots
            # =====================================================
            due = scheduler.get_due_snapshots(now_ts)

            for mint, window_label in due:
                t0_ts = scheduler.get_t0(mint)

                if not t0_ts:
                    continue

                log.info("  Snapshot: %s @ %s (%ds after T0)",
                         mint[:12], window_label, now_ts - t0_ts)

                snap_data = collect_snapshot_for_token(mint, t0_ts, window_label)
                scheduler.record_snapshot(mint, window_label, snap_data)

                # If this is the final (t15m) snapshot, finalize and save
                if window_label == "t15m":
                    finalized = scheduler.finalize(mint)
                    if finalized:
                        record = build_record(mint, t0_ts, finalized)
                        saved = save_to_supabase(record)
                        if saved:
                            total_collected += 1
                            quality.add_record(record)
                            scheduler.remove(mint)

            # =====================================================
            # STEP 3: Quality check
            # =====================================================
            if total_collected - last_quality_check >= QUALITY_CHECK_INTERVAL:
                last_quality_check = total_collected
                if quality.records:
                    report_json = quality.generate_report()
                    flags = json.loads(report_json)
                    flagged_count = flags["features_flagged"]
                    log.info("Quality check: %d rows, %d features, %d flagged",
                             flags["total_rows"], flags["features_checked"], flagged_count)
                    if flagged_count > 0:
                        flagged_names = [f["feature_name"] for f in flags["flags"] if f["flagged"]]
                        log.warning("Flagged features: %s", ", ".join(flagged_names))
                    quality.reset()

            # =====================================================
            # STEP 4: Cleanup stale entries
            # =====================================================
            stale_removed = scheduler.cleanup_stale(max_age_seconds=1200)
            if stale_removed > 0:
                log.info("Cleaned up %d stale/completed scheduler entries", stale_removed)

            # =====================================================
            # STEP 5: Throughput logging (every 100 tokens)
            # =====================================================
            if total_collected > 0 and total_collected % 100 == 0:
                elapsed_hrs = (time.time() - start_time) / 3600
                tps = total_collected / max(time.time() - start_time, 1)
                eta_1M = (1_000_000 - total_collected) / max(tps, 0.001) / 3600
                log.info(
                    "Throughput: %.3f tok/s | Total: %d | Uptime: %.1fh | ETA 1M: %.1fh | Pending: %d",
                    tps, total_collected, elapsed_hrs, eta_1M, scheduler.pending_count(),
                )

        except KeyboardInterrupt:
            log.info("Shutdown requested — exiting.")
            break
        except Exception:
            log.exception("Unhandled error in main loop — sleeping %ds before retry", INTERVAL)

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
