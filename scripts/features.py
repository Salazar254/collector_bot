"""
scripts/features.py — Feature computation module for snapshot-based collector.

Computes all 50 post-migration features organized into 9 categories:
PRICE, LIQUIDITY, VOLUME, BUYERS, SELLERS, ORDER_FLOW, HOLDERS,
WHALES, VOLATILITY.

Inputs: snapshot price/liquidity data + swap transaction lists per window.
Output: flat dict of all features + labels ready for Supabase upsert.
"""

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHALE_THRESHOLD_SOL = 10.0  # swaps >= 10 SOL classified as whale activity
WINDOWS = ("1m", "5m", "15m")
WINDOW_SECONDS = {"1m": 60, "5m": 300, "15m": 900}


# ===================================================================
# SWAP PARSING HELPERS
# ===================================================================


def _safe_int(value) -> int:
    """Helius amounts may be strings or ints — always coerce to int."""
    try:
        return int(value or 0)
    except (ValueError, TypeError):
        return 0


def parse_swap(tx: dict, token_mint: str = "") -> dict:
    """
    Parse a Helius enhanced swap transaction into a normalized dict.

    Direction detection (priority order):
      1. tokenTransfers: fee-payer receives token_mint → buy, sends → sell
      2. nativeTransfers:  fee-payer sends SOL → buy, receives SOL → sell
      3. nativeInput.amount > 0 → buy
      4. nativeOutput.amount > 0 → sell
      5. Fallback: assume buy (most post-graduation swaps are buys)

    SOL amount: nativeTransfers > nativeInput/nativeOutput max

    Returns dict with:
        timestamp, fee_payer, is_buy, sol_amount, usd_estimate
    """
    fee_payer = tx.get("feePayer", "")
    ev = tx.get("events", {}).get("swap", {}) or {}
    native_in = ev.get("nativeInput") or {}
    native_out = ev.get("nativeOutput") or {}

    in_amt = _safe_int(native_in.get("amount", 0))
    out_amt = _safe_int(native_out.get("amount", 0))

    # ── Priority 1: tokenTransfers (most reliable for Raydium) ──
    is_buy = _detect_direction_from_token_transfers(tx, token_mint, fee_payer)

    # ── Priority 2: nativeTransfers (SOL movement) ──
    if is_buy is None:
        is_buy = _detect_direction_from_native_transfers(tx, fee_payer)

    # ── Priority 3+4: nativeInput / nativeOutput ──
    if is_buy is None:
        if in_amt > 0:
            is_buy = True
        elif out_amt > 0:
            is_buy = False

    # ── Priority 5: assume buy ──
    if is_buy is None:
        is_buy = True

    # ── SOL amount: nativeTransfers is the ground truth ──
    sol_amount = _extract_sol_amount_from_native_transfers(tx, fee_payer)
    if sol_amount <= 0:
        sol_amount = max(in_amt, out_amt) / 1e9

    # ── USD estimate ──
    sol_price_usd = 150.0
    if is_buy and native_in:
        sol_price_usd = float(native_in.get("usdPrice", 0) or 0) or 150.0
    elif not is_buy and native_out:
        sol_price_usd = float(native_out.get("usdPrice", 0) or 0) or 150.0

    return {
        "timestamp": tx.get("timestamp", 0),
        "fee_payer": fee_payer,
        "is_buy": is_buy,
        "sol_amount": sol_amount,
        "usd_estimate": round(sol_amount * sol_price_usd, 2),
    }


def _detect_direction_from_token_transfers(tx: dict, token_mint: str, fee_payer: str) -> Optional[bool]:
    """
    Determine swap direction from tokenTransfers array.

    If fee_payer RECEIVED token_mint → buy (True)
    If fee_payer SENT token_mint     → sell (False)
    Returns None if direction cannot be determined.
    """
    if not token_mint or not fee_payer:
        return None

    for transfer in tx.get("tokenTransfers", []):
        if transfer.get("mint", "") != token_mint:
            continue
        from_user = transfer.get("fromUserAccount", "")
        to_user = transfer.get("toUserAccount", "")
        token_amount = float(transfer.get("tokenAmount", 0) or 0)
        if token_amount <= 0:
            continue
        if to_user == fee_payer:
            return True   # fee payer received tokens → buy
        if from_user == fee_payer:
            return False  # fee payer sent tokens → sell

    return None


def _detect_direction_from_native_transfers(tx: dict, fee_payer: str) -> Optional[bool]:
    """
    Determine swap direction from nativeTransfers (SOL movement).

    Fee payer sends SOL → buy (True)
    Fee payer receives SOL → sell (False)
    """
    if not fee_payer:
        return None

    for nt in tx.get("nativeTransfers", []):
        amt = _safe_int(nt.get("amount", 0))
        if amt <= 0:
            continue
        if nt.get("fromUserAccount", "") == fee_payer:
            return True   # fee payer sent SOL → buy
        if nt.get("toUserAccount", "") == fee_payer:
            return False  # fee payer received SOL → sell

    return None


def _extract_sol_amount_from_native_transfers(tx: dict, fee_payer: str) -> float:
    """
    Extract the SOL amount involved in this swap from nativeTransfers.
    Returns the max SOL amount sent/received by the fee payer, in SOL units.
    """
    if not fee_payer:
        return 0.0

    max_lamports = 0
    for nt in tx.get("nativeTransfers", []):
        from_user = nt.get("fromUserAccount", "")
        to_user = nt.get("toUserAccount", "")
        if from_user == fee_payer or to_user == fee_payer:
            amt = _safe_int(nt.get("amount", 0))
            if amt > max_lamports:
                max_lamports = amt

    return max_lamports / 1e9


def parse_swaps_for_window(swaps: list[dict], window_start: int, window_end: int, token_mint: str = "") -> list[dict]:
    """Filter and parse swaps within a time window."""
    parsed = []
    for swap in swaps:
        ts = swap.get("timestamp", 0)
        if window_start <= ts <= window_end:
            parsed.append(parse_swap(swap, token_mint))
    return parsed


# ===================================================================
# PRICE FEATURES (9)
# ===================================================================


def compute_price_features(snapshots: dict) -> dict:
    """
    Compute 9 price features from snapshot price data.

    snapshots: {"t0": {"price_usd": ...}, "t1m": {...}, "t5m": {...}, "t15m": {...}}
    """
    p0 = float(snapshots.get("t0", {}).get("price_usd", 0) or 0)
    p1 = float(snapshots.get("t1m", {}).get("price_usd", 0) or 0)
    p5 = float(snapshots.get("t5m", {}).get("price_usd", 0) or 0)
    p15 = float(snapshots.get("t15m", {}).get("price_usd", 0) or 0)

    prices = [p for p in (p0, p1, p5, p15) if p > 0]

    features = {
        "price_usd_t0": p0,
        "price_usd_1m": p1,
        "price_usd_5m": p5,
        "price_usd_15m": p15,
        "price_change_1m_pct": round(((p1 - p0) / p0 * 100), 4) if p0 > 0 else 0.0,
        "price_change_5m_pct": round(((p5 - p0) / p0 * 100), 4) if p0 > 0 else 0.0,
        "price_change_15m_pct": round(((p15 - p0) / p0 * 100), 4) if p0 > 0 else 0.0,
        "max_price_first_15m": max(prices) if prices else 0.0,
        "min_price_first_15m": min(prices) if prices else 0.0,
    }
    return features


# ===================================================================
# LIQUIDITY FEATURES (6)
# ===================================================================


def compute_liquidity_features(snapshots: dict) -> dict:
    """
    Compute 6 liquidity features from snapshot liquidity data.
    """
    l0 = float(snapshots.get("t0", {}).get("liquidity_usd", 0) or 0)
    l1 = float(snapshots.get("t1m", {}).get("liquidity_usd", 0) or 0)
    l5 = float(snapshots.get("t5m", {}).get("liquidity_usd", 0) or 0)
    l15 = float(snapshots.get("t15m", {}).get("liquidity_usd", 0) or 0)

    features = {
        "liquidity_usd_t0": l0,
        "liquidity_usd_1m": l1,
        "liquidity_usd_5m": l5,
        "liquidity_usd_15m": l15,
        "liquidity_growth_5m": round(((l5 - l0) / l0), 4) if l0 > 0 else 0.0,
        "liquidity_growth_15m": round(((l15 - l0) / l0), 4) if l0 > 0 else 0.0,
    }
    return features


# ===================================================================
# VOLUME FEATURES (3)
# ===================================================================


def compute_volume_features(swaps_by_window: dict) -> dict:
    """
    Compute cumulative USD volume for each window from parsed swaps.

    swaps_by_window: {"1m": [parsed_swaps...], "5m": [...], "15m": [...]}
    """
    features = {}
    for window in WINDOWS:
        swaps = swaps_by_window.get(window, [])
        vol = sum(s["usd_estimate"] for s in swaps)
        features[f"volume_{window}"] = round(vol, 2)
    return features


# ===================================================================
# BUYER FEATURES (4)
# ===================================================================


def compute_buyer_features(swaps_by_window: dict) -> dict:
    """
    Count unique buyer addresses per window + growth rate.
    """
    buyers = {}
    for window in WINDOWS:
        swaps = swaps_by_window.get(window, [])
        unique = len({s["fee_payer"] for s in swaps if s["is_buy"]})
        buyers[window] = unique

    b1 = buyers.get("1m", 0)
    b15 = buyers.get("15m", 0)

    features = {
        "unique_buyers_1m": buyers.get("1m", 0),
        "unique_buyers_5m": buyers.get("5m", 0),
        "unique_buyers_15m": buyers.get("15m", 0),
        "buyer_growth_rate": round(((b15 - b1) / max(b1, 1)), 4),
    }
    return features


# ===================================================================
# SELLER FEATURES (4)
# ===================================================================


def compute_seller_features(swaps_by_window: dict) -> dict:
    """
    Count unique seller addresses per window + growth rate.
    """
    sellers = {}
    for window in WINDOWS:
        swaps = swaps_by_window.get(window, [])
        unique = len({s["fee_payer"] for s in swaps if not s["is_buy"]})
        sellers[window] = unique

    s1 = sellers.get("1m", 0)
    s15 = sellers.get("15m", 0)

    features = {
        "unique_sellers_1m": sellers.get("1m", 0),
        "unique_sellers_5m": sellers.get("5m", 0),
        "unique_sellers_15m": sellers.get("15m", 0),
        "seller_growth_rate": round(((s15 - s1) / max(s1, 1)), 4),
    }
    return features


# ===================================================================
# ORDER FLOW FEATURES (10)
# ===================================================================


def compute_order_flow_features(swaps_by_window: dict) -> dict:
    """
    Buy/sell counts, ratios, and net USD flow per window.
    """
    buy_counts = {}
    sell_counts = {}
    buy_vol_total = 0.0
    sell_vol_total = 0.0

    for window in WINDOWS:
        swaps = swaps_by_window.get(window, [])
        bc = sum(1 for s in swaps if s["is_buy"])
        sc = sum(1 for s in swaps if not s["is_buy"])
        buy_counts[window] = bc
        sell_counts[window] = sc

        if window == "15m":
            buy_vol_total = sum(s["usd_estimate"] for s in swaps if s["is_buy"])
            sell_vol_total = sum(s["usd_estimate"] for s in swaps if not s["is_buy"])

    features = {
        "buy_count_1m": buy_counts.get("1m", 0),
        "buy_count_5m": buy_counts.get("5m", 0),
        "buy_count_15m": buy_counts.get("15m", 0),
        "sell_count_1m": sell_counts.get("1m", 0),
        "sell_count_5m": sell_counts.get("5m", 0),
        "sell_count_15m": sell_counts.get("15m", 0),
        "buy_sell_ratio_1m": round(buy_counts.get("1m", 0) / max(sell_counts.get("1m", 0), 1), 4),
        "buy_sell_ratio_5m": round(buy_counts.get("5m", 0) / max(sell_counts.get("5m", 0), 1), 4),
        "buy_sell_ratio_15m": round(buy_counts.get("15m", 0) / max(sell_counts.get("15m", 0), 1), 4),
        "net_flow_usd": round(buy_vol_total - sell_vol_total, 2),
    }
    return features


# ===================================================================
# HOLDER FEATURES (5)
# ===================================================================


def compute_holder_features(holder_snapshots: dict) -> dict:
    """
    Compute holder counts and growth from holder snapshots.

    holder_snapshots: {"1m": count, "5m": count, "15m": count}
    """
    h1 = holder_snapshots.get("1m", 0)
    h5 = holder_snapshots.get("5m", 0)
    h15 = holder_snapshots.get("15m", 0)

    features = {
        "holder_count_1m": h1,
        "holder_count_5m": h5,
        "holder_count_15m": h15,
        "holder_growth_5m": round(((h5 - h1) / max(h1, 1)), 4),
        "holder_growth_15m": round(((h15 - h1) / max(h1, 1)), 4),
    }
    return features


# ===================================================================
# WHALE FEATURES (5)
# ===================================================================


def compute_whale_features(swaps_by_window: dict, threshold_sol: float = WHALE_THRESHOLD_SOL) -> dict:
    """
    Identify whale activity (swaps >= threshold SOL) in the 15m window.
    """
    all_swaps = swaps_by_window.get("15m", [])
    whale_swaps = [s for s in all_swaps if s["sol_amount"] >= threshold_sol]

    whale_buys = [s for s in whale_swaps if s["is_buy"]]
    whale_sells = [s for s in whale_swaps if not s["is_buy"]]

    # Largest individual buy/sell across ALL swaps (not just whales)
    buys = [s for s in all_swaps if s["is_buy"]]
    sells = [s for s in all_swaps if not s["is_buy"]]

    buy_vol = sum(s["sol_amount"] for s in whale_buys)
    sell_vol = sum(s["sol_amount"] for s in whale_sells)

    features = {
        "largest_buy_usd": round(max((s["usd_estimate"] for s in buys), default=0), 2),
        "largest_sell_usd": round(max((s["usd_estimate"] for s in sells), default=0), 2),
        "whale_buy_count": len(whale_buys),
        "whale_sell_count": len(whale_sells),
        "whale_net_flow": round(buy_vol - sell_vol, 4),
    }
    return features


# ===================================================================
# VOLATILITY FEATURES (4)
# ===================================================================


def compute_volatility_features(price_points: list[float]) -> dict:
    """
    Compute volatility (std dev of returns) and max drawdown from price series.

    price_points: ordered list of price observations [p_t0, p_1m, p_5m, p_15m]
    """
    prices = np.array([p for p in price_points if p > 0], dtype=np.float64)

    if len(prices) < 2:
        return {
            "volatility_1m": 0.0,
            "volatility_5m": 0.0,
            "volatility_15m": 0.0,
            "drawdown_first_15m": 0.0,
        }

    # Compute returns
    returns = np.diff(prices) / prices[:-1]

    # Subset returns for each window approximation (fewer points = shorter window)
    # We only have 4 price points, so volatility estimates are coarse
    n = len(returns)
    vol_1m = float(np.std(returns[:1])) if n >= 1 else 0.0
    vol_5m = float(np.std(returns[:min(2, n)])) if n >= 2 else vol_1m
    vol_15m = float(np.std(returns))

    # Drawdown: max percentage drop from peak
    peak = np.maximum.accumulate(prices)
    drawdowns = (peak - prices) / peak * 100
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "volatility_1m": round(vol_1m, 6),
        "volatility_5m": round(vol_5m, 6),
        "volatility_15m": round(vol_15m, 6),
        "drawdown_first_15m": round(max_drawdown, 4),
    }


# ===================================================================
# LABEL COMPUTATION (from DexScreener 24h data)
# ===================================================================


def compute_labels(price_data_24h: Optional[dict]) -> dict:
    """
    Compute profit-tier targets from DexScreener 24h price data.

    Targets are cumulative — a 10x token also sets did_2x and did_5x.
    """
    if not price_data_24h:
        return {
            "did_2x": 0,
            "did_5x": 0,
            "did_10x": 0,
            "max_drawdown_pct": 0.0,
            "inferred_label": False,
        }

    price_change_24h = float(price_data_24h.get("price_change_24h", 0) or 0)
    liquidity_usd = float(price_data_24h.get("liquidity_usd", 0) or 0)

    did_2x = 1 if price_change_24h >= 100 else 0
    did_5x = 1 if price_change_24h >= 400 else 0
    did_10x = 1 if price_change_24h >= 900 else 0

    # Dead LP → no profit tier valid
    if liquidity_usd < 10:
        did_2x = did_5x = did_10x = 0

    max_drawdown = max(0.0, -price_change_24h)

    return {
        "did_2x": did_2x,
        "did_5x": did_5x,
        "did_10x": did_10x,
        "max_drawdown_pct": max_drawdown,
        "inferred_label": True,
    }


# ===================================================================
# MASTER COMPUTE FUNCTION
# ===================================================================


def compute_all_features(
    snapshots: dict,
    swaps_by_window: dict,
    holder_snapshots: Optional[dict] = None,
    price_data_24h: Optional[dict] = None,
) -> dict:
    """
    Compute ALL 50 features + labels from raw data.

    Args:
        snapshots: {"t0": {"price_usd", "liquidity_usd"}, "t1m": {...}, ...}
        swaps_by_window: {"1m": [parsed_swaps], "5m": [...], "15m": [...]}
        holder_snapshots: {"1m": count, "5m": count, "15m": count}
        price_data_24h: DexScreener 24h data for label computation

    Returns:
        Flat dict of all features + labels, ready for Supabase upsert.
    """
    # Ensure swaps_by_window has all keys
    for w in WINDOWS:
        swaps_by_window.setdefault(w, [])

    price_feats = compute_price_features(snapshots)
    liq_feats = compute_liquidity_features(snapshots)
    vol_feats = compute_volume_features(swaps_by_window)
    buyer_feats = compute_buyer_features(swaps_by_window)
    seller_feats = compute_seller_features(swaps_by_window)
    order_feats = compute_order_flow_features(swaps_by_window)
    whale_feats = compute_whale_features(swaps_by_window)

    # Holder features
    if holder_snapshots:
        holder_feats = compute_holder_features(holder_snapshots)
    else:
        holder_feats = {
            "holder_count_1m": 0, "holder_count_5m": 0, "holder_count_15m": 0,
            "holder_growth_5m": 0.0, "holder_growth_15m": 0.0,
        }

    # Volatility from price series
    price_series = [
        snapshots.get("t0", {}).get("price_usd", 0) or 0,
        snapshots.get("t1m", {}).get("price_usd", 0) or 0,
        snapshots.get("t5m", {}).get("price_usd", 0) or 0,
        snapshots.get("t15m", {}).get("price_usd", 0) or 0,
    ]
    vol_feats_volatility = compute_volatility_features(price_series)

    # Labels
    labels = compute_labels(price_data_24h)

    # Merge all
    all_features = {}
    all_features.update(price_feats)
    all_features.update(liq_feats)
    all_features.update(vol_feats)
    all_features.update(buyer_feats)
    all_features.update(seller_feats)
    all_features.update(order_feats)
    all_features.update(holder_feats)
    all_features.update(whale_feats)
    all_features.update(vol_feats_volatility)
    all_features.update(labels)

    return all_features
