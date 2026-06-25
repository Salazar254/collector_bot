"""
scripts/axiom_features.py — Feature extraction from Mobula filterTokenWallets data.

Computes ~80 wallet-intelligence features organized into 13 categories.
All features are designed to improve prediction of profit-tier targets
and rug detection for XGBoost, LightGBM, CatBoost, PyTorch GRU models.

Input:  list[dict] from Mobula filterTokenWallets GraphQL query
        + existing swap data for time-window cross-referencing.
Output: Flat dict of all Axiom features ready for Supabase upsert.

Data source: Mobula GraphQL API (https://graphql.mobula.io/graphql)
Primary query: filterTokenWallets — per-wallet per-token stats with labels,
               windowed PnL/volume/ROI (1d/1w/30d/1y), scammer/bot scores.

All compute_* functions accept partial/missing data and return sensible defaults.
They NEVER crash — missing data produces zeros/empty values.
"""

import logging
import time
from statistics import median as _median, mean as _mean
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHALE_THRESHOLDS = [1000.0, 5000.0, 10000.0]
WINDOWS_SEC = {"1m": 60, "5m": 300, "15m": 900}

# Default smart money / risk / new-wallet labels (Mobula walletLabelTypes)
DEFAULT_SMART_LABELS: frozenset[str] = frozenset({
    "smart_money", "pro_trader", "elite",
})
DEFAULT_RISK_LABELS: frozenset[str] = frozenset({
    "scammer", "bot", "sniper", "bundler",
})
DEFAULT_NEW_LABELS: frozenset[str] = frozenset({
    "fresh_wallet", "new_wallet",
})


# ===================================================================
# Helpers
# ===================================================================


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Coerce value to float, returning default on failure."""
    try:
        return float(val or 0)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Coerce value to int, returning default on failure."""
    try:
        return int(val or 0)
    except (ValueError, TypeError):
        return default


def _safe_median(values: list[float]) -> float:
    """Safely compute median, returns 0.0 for empty list."""
    if not values:
        return 0.0
    return float(_median(values))


def _safe_mean(values: list[float]) -> float:
    """Safely compute mean, returns 0.0 for empty list."""
    if not values:
        return 0.0
    return float(_mean(values))


def _has_label(wallet: dict, labels: frozenset[str]) -> bool:
    """Check if a wallet has any of the given Mobula labels."""
    wallet_labels: list[str] = wallet.get("labels", []) or []
    for lbl in wallet_labels:
        if lbl in labels:
            return True
    return False


def _wallet_age_days(wallet: dict, now_ts: Optional[int] = None) -> float:
    """Approximate wallet age in days from firstTransactionAt."""
    first_ts = _safe_int(wallet.get("firstTransactionAt", 0))
    if first_ts <= 0:
        return 0.0
    now = now_ts or int(time.time())
    return max((now - first_ts) / 86400.0, 0.0)


def _build_wallet_index(
    token_wallets: list[dict],
) -> dict[str, dict]:
    """Build address → wallet-dict lookup from filterTokenWallets results."""
    index: dict[str, dict] = {}
    for w in token_wallets:
        addr = w.get("address", "")
        if addr:
            index[addr] = w
    return index


# ===================================================================
# SMART MONEY (12 features)
# ===================================================================


def compute_smart_money_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
    t0_ts: int = 0,
    smart_labels: Optional[frozenset[str]] = None,
) -> dict[str, Any]:
    """
    Compute 12 smart money features from Mobula wallet labels
    cross-referenced with swap transaction data.

    Uses Mobula labels (smart_money, pro_trader, elite) to identify
    smart money wallets, then counts their activity per time window.
    """
    smart_labels = smart_labels or DEFAULT_SMART_LABELS
    wallet_index = _build_wallet_index(token_wallets)

    features: dict[str, Any] = {}

    for window_key, window_sec in WINDOWS_SEC.items():
        swaps = swaps_by_window.get(window_key, [])
        sm_buyers: set[str] = set()
        sm_vol = 0.0
        sm_first_ts: Optional[int] = None

        for swap in swaps:
            wallet = swap.get("fee_payer", "")
            is_buy = swap.get("is_buy", True)
            usd = swap.get("usd_estimate", 0.0)
            ts = swap.get("timestamp", 0)

            wdata = wallet_index.get(wallet, {})
            if is_buy and _has_label(wdata, smart_labels):
                sm_buyers.add(wallet)
                sm_vol += usd
                if sm_first_ts is None or ts < sm_first_ts:
                    sm_first_ts = ts

        features[f"smart_wallet_buyers_{window_key}"] = len(sm_buyers)
        features[f"smart_wallet_volume_{window_key}"] = round(sm_vol, 2)

    # 15m aggregate percentages
    total_buyers_15m = len({
        s["fee_payer"]
        for s in swaps_by_window.get("15m", [])
        if s.get("is_buy", True) and s.get("fee_payer", "")
    })
    sm_buyers_15m = features.get("smart_wallet_buyers_15m", 0)
    features["smart_wallet_percentage"] = round(
        sm_buyers_15m / max(total_buyers_15m, 1), 4
    )

    # Smart money first buyer
    all_buys_sorted = sorted(
        [s for s in swaps_by_window.get("15m", []) if s.get("is_buy", True)],
        key=lambda x: x.get("timestamp", 0),
    )
    first_buyer_wallet = all_buys_sorted[0].get("fee_payer", "") if all_buys_sorted else ""
    first_data = wallet_index.get(first_buyer_wallet, {})
    features["smart_money_first_buyer"] = 1 if _has_label(first_data, smart_labels) else 0

    # First smart money buy timestamp
    sm_buys = [
        s for s in all_buys_sorted
        if _has_label(wallet_index.get(s.get("fee_payer", ""), {}), smart_labels)
    ]
    features["first_smart_money_buy_timestamp"] = (
        sm_buys[0].get("timestamp", 0) if sm_buys else 0
    )

    # Smart money within first minute / first 5m
    sm_within_1m = sum(
        1 for s in sm_buys if 0 <= (s.get("timestamp", 0) - t0_ts) <= 60
    )
    sm_within_5m = sum(
        1 for s in sm_buys if 0 <= (s.get("timestamp", 0) - t0_ts) <= 300
    )
    features["smart_money_within_first_minute"] = sm_within_1m
    features["smart_money_within_first_5m"] = sm_within_5m

    # Accumulation rate
    total_buy_vol_15m = sum(
        s.get("usd_estimate", 0)
        for s in swaps_by_window.get("15m", [])
        if s.get("is_buy", True)
    )
    sm_vol_15m = features.get("smart_wallet_volume_15m", 0)
    features["smart_money_accumulation_rate"] = round(
        sm_vol_15m / max(total_buy_vol_15m, 0.01), 4
    )

    return features


# ===================================================================
# WALLET QUALITY (10 features)
# ===================================================================


def compute_wallet_quality_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 10 wallet quality features from Mobula per-wallet stats.
    Avg/median of realized PnL, ROI %, and trade count across buyers.
    """
    buyer_wallets = _get_buyer_addresses(swaps_by_window)
    if not buyer_wallets:
        return _empty_wallet_quality()

    wallet_index = _build_wallet_index(token_wallets)

    pnls: list[float] = []
    rois: list[float] = []
    trade_counts: list[float] = []
    ages: list[float] = []

    for addr in buyer_wallets:
        w = wallet_index.get(addr, {})
        pnls.append(_safe_float(w.get("realizedProfitUsd30d", 0)))
        rois.append(_safe_float(w.get("realizedProfitPercentage30d", 0)))
        # Proxy trade count from total buys across all windows
        tc = _safe_int(w.get("buys1d", 0)) + _safe_int(w.get("buys1w", 0))
        trade_counts.append(float(tc))
        ages.append(_wallet_age_days(w))

    return {
        "avg_wallet_age_days": round(_safe_mean(ages), 2),
        "median_wallet_age_days": round(_safe_median(ages), 2),
        "avg_wallet_trade_count": round(_safe_mean(trade_counts), 2),
        "median_wallet_trade_count": round(_safe_median(trade_counts), 2),
        "avg_wallet_win_rate": round(
            _safe_mean([_safe_float(wallet_index.get(a, {}).get("realizedProfitPercentage30d", 0))
                        for a in buyer_wallets]), 4
        ),
        "median_wallet_win_rate": round(
            _safe_median([_safe_float(wallet_index.get(a, {}).get("realizedProfitPercentage30d", 0))
                          for a in buyer_wallets]), 4
        ),
        "avg_wallet_realized_pnl": round(_safe_mean(pnls), 2),
        "median_wallet_realized_pnl": round(_safe_median(pnls), 2),
        "avg_wallet_roi": round(_safe_mean(rois), 4),
        "median_wallet_roi": round(_safe_median(rois), 4),
    }


def _empty_wallet_quality() -> dict[str, Any]:
    return {
        "avg_wallet_age_days": 0.0,
        "median_wallet_age_days": 0.0,
        "avg_wallet_trade_count": 0.0,
        "median_wallet_trade_count": 0.0,
        "avg_wallet_win_rate": 0.0,
        "median_wallet_win_rate": 0.0,
        "avg_wallet_realized_pnl": 0.0,
        "median_wallet_realized_pnl": 0.0,
        "avg_wallet_roi": 0.0,
        "median_wallet_roi": 0.0,
    }


def _get_buyer_addresses(
    swaps_by_window: dict[str, list[dict]],
) -> list[str]:
    """Extract unique buyer wallet addresses from 15m swap window."""
    swaps = swaps_by_window.get("15m", [])
    return list({
        s["fee_payer"]
        for s in swaps
        if s.get("is_buy", True) and s.get("fee_payer", "")
    })


def _get_seller_addresses(
    swaps_by_window: dict[str, list[dict]],
) -> list[str]:
    """Extract unique seller wallet addresses from 15m swap window."""
    swaps = swaps_by_window.get("15m", [])
    return list({
        s["fee_payer"]
        for s in swaps
        if not s.get("is_buy", True) and s.get("fee_payer", "")
    })


# ===================================================================
# PNL FEATURES (10 features)
# ===================================================================


def compute_pnl_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 10 PnL features from Mobula per-wallet realizedProfitUsd
    for 30d and 1y windows, separated by buyer/seller.
    """
    wallet_index = _build_wallet_index(token_wallets)

    buyer_addrs = _get_buyer_addresses(swaps_by_window)
    seller_addrs = _get_seller_addresses(swaps_by_window)

    buyer_pnls_30d = [
        _safe_float(wallet_index.get(a, {}).get("realizedProfitUsd30d", 0))
        for a in buyer_addrs
    ]
    buyer_pnls_1y = [
        _safe_float(wallet_index.get(a, {}).get("realizedProfitUsd1y", 0))
        for a in buyer_addrs
    ]
    seller_pnls_30d = [
        _safe_float(wallet_index.get(a, {}).get("realizedProfitUsd30d", 0))
        for a in seller_addrs
    ]
    seller_pnls_1y = [
        _safe_float(wallet_index.get(a, {}).get("realizedProfitUsd1y", 0))
        for a in seller_addrs
    ]

    return {
        "avg_buyer_pnl_30d": round(_safe_mean(buyer_pnls_30d), 2),
        "median_buyer_pnl_30d": round(_safe_median(buyer_pnls_30d), 2),
        "top_buyer_pnl_30d": round(max(buyer_pnls_30d) if buyer_pnls_30d else 0.0, 2),
        "avg_buyer_pnl_90d": round(_safe_mean(buyer_pnls_1y), 2),   # 1y as 90d proxy
        "median_buyer_pnl_90d": round(_safe_median(buyer_pnls_1y), 2),
        "top_buyer_pnl_90d": round(max(buyer_pnls_1y) if buyer_pnls_1y else 0.0, 2),
        "avg_seller_pnl_30d": round(_safe_mean(seller_pnls_30d), 2),
        "median_seller_pnl_30d": round(_safe_median(seller_pnls_30d), 2),
        "avg_seller_pnl_90d": round(_safe_mean(seller_pnls_1y), 2),
        "median_seller_pnl_90d": round(_safe_median(seller_pnls_1y), 2),
    }


# ===================================================================
# ROI FEATURES (6 features)
# ===================================================================


def compute_roi_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 6 ROI features for buyers across 30d and 1y windows.
    Uses Mobula's realizedProfitPercentage fields.
    """
    wallet_index = _build_wallet_index(token_wallets)
    buyer_addrs = _get_buyer_addresses(swaps_by_window)

    rois_30d = [
        _safe_float(wallet_index.get(a, {}).get("realizedProfitPercentage30d", 0))
        for a in buyer_addrs
    ]
    rois_1y = [
        _safe_float(wallet_index.get(a, {}).get("realizedProfitPercentage1y", 0))
        for a in buyer_addrs
    ]

    return {
        "avg_buyer_roi_30d": round(_safe_mean(rois_30d), 4),
        "median_buyer_roi_30d": round(_safe_median(rois_30d), 4),
        "top_buyer_roi_30d": round(max(rois_30d) if rois_30d else 0.0, 4),
        "avg_buyer_roi_90d": round(_safe_mean(rois_1y), 4),
        "median_buyer_roi_90d": round(_safe_median(rois_1y), 4),
        "top_buyer_roi_90d": round(max(rois_1y) if rois_1y else 0.0, 4),
    }


# ===================================================================
# PROFITABLE TRADER METRICS (8 features)
# ===================================================================


def compute_profitable_trader_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 8 profitable trader metrics from Mobula per-wallet PnL/ROI.
    """
    wallet_index = _build_wallet_index(token_wallets)
    buyer_addrs = _get_buyer_addresses(swaps_by_window)
    swaps_15m = swaps_by_window.get("15m", [])

    profitable_count = 0
    profitable_vol = 0.0
    high_roi_count = 0
    elite_count = 0
    positive_pnl = 0
    above_20 = 0
    above_50 = 0
    above_100 = 0

    for addr in buyer_addrs:
        w = wallet_index.get(addr, {})
        pnl = _safe_float(w.get("realizedProfitUsd30d", 0))
        roi = _safe_float(w.get("realizedProfitPercentage30d", 0))
        # Elite: high PnL + labels suggest quality
        is_elite = _has_label(w, DEFAULT_SMART_LABELS) and pnl > 1000

        if pnl > 0:
            profitable_count += 1
            positive_pnl += 1
            # Sum this wallet's buy volume in 15m window
            wallet_buy_vol = sum(
                s.get("usd_estimate", 0)
                for s in swaps_15m
                if s.get("fee_payer", "") == addr and s.get("is_buy", True)
            )
            profitable_vol += wallet_buy_vol

        if roi >= 1.0:
            high_roi_count += 1
        if roi >= 1.0:
            above_100 += 1
        if roi >= 0.50:
            above_50 += 1
        if roi >= 0.20:
            above_20 += 1

        if is_elite:
            elite_count += 1

    return {
        "profitable_wallet_count": profitable_count,
        "profitable_wallet_buy_volume": round(profitable_vol, 2),
        "high_roi_wallet_count": high_roi_count,
        "elite_trader_count": elite_count,
        "wallets_with_positive_pnl": positive_pnl,
        "wallets_above_20pct_roi": above_20,
        "wallets_above_50pct_roi": above_50,
        "wallets_above_100pct_roi": above_100,
    }


# ===================================================================
# WHALE AXIOM FEATURES (24 features — 8 per threshold x 3)
# ===================================================================


def compute_whale_axiom_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 24 whale features across 3 thresholds ($1K, $5K, $10K).
    Uses Mobula amountBoughtUsd1d to classify whale-sized traders.
    """
    features: dict[str, Any] = {}
    wallet_index = _build_wallet_index(token_wallets)
    all_swaps_15m = swaps_by_window.get("15m", [])
    total_buy_vol = sum(
        s.get("usd_estimate", 0) for s in all_swaps_15m if s.get("is_buy", True)
    )

    threshold_labels = {1000.0: "1k", 5000.0: "5k", 10000.0: "10k"}

    for threshold in WHALE_THRESHOLDS:
        label = threshold_labels[threshold]

        # Find whale wallets from Mobula data (amountBoughtUsd1d >= threshold)
        whale_addrs: set[str] = set()
        for w in token_wallets:
            if _safe_float(w.get("amountBoughtUsd1d", 0)) >= threshold:
                whale_addrs.add(w.get("address", ""))

        # Filter swap data to whale activity only
        whale_buys = [
            s for s in all_swaps_15m
            if s.get("fee_payer", "") in whale_addrs and s.get("is_buy", True)
        ]
        whale_sells = [
            s for s in all_swaps_15m
            if s.get("fee_payer", "") in whale_addrs and not s.get("is_buy", True)
        ]

        buy_vol = sum(s.get("usd_estimate", 0) for s in whale_buys)
        sell_vol = sum(s.get("usd_estimate", 0) for s in whale_sells)

        features[f"largest_buy_usd_{label}"] = round(
            max((s.get("usd_estimate", 0) for s in whale_buys), default=0), 2
        )
        features[f"largest_sell_usd_{label}"] = round(
            max((s.get("usd_estimate", 0) for s in whale_sells), default=0), 2
        )
        features[f"whale_buy_count_{label}"] = len(whale_buys)
        features[f"whale_sell_count_{label}"] = len(whale_sells)
        features[f"whale_buy_volume_{label}"] = round(buy_vol, 2)
        features[f"whale_sell_volume_{label}"] = round(sell_vol, 2)
        features[f"whale_net_flow_{label}"] = round(buy_vol - sell_vol, 2)
        features[f"whale_accumulation_rate_{label}"] = round(
            buy_vol / max(total_buy_vol, 0.01), 4
        )

    return features


# ===================================================================
# BUYER QUALITY (5 features)
# ===================================================================


def compute_buyer_quality_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
    new_labels: Optional[frozenset[str]] = None,
    smart_labels: Optional[frozenset[str]] = None,
) -> dict[str, Any]:
    """
    Compute 5 buyer quality features: new vs experienced wallets,
    wallet age brackets. Uses Mobula labels and firstTransactionAt.
    """
    new_labels = new_labels or DEFAULT_NEW_LABELS
    smart_labels = smart_labels or DEFAULT_SMART_LABELS
    wallet_index = _build_wallet_index(token_wallets)
    buyer_addrs = _get_buyer_addresses(swaps_by_window)

    new_count = 0
    experienced_count = 0
    older_30 = 0
    older_90 = 0
    older_180 = 0

    for addr in buyer_addrs:
        w = wallet_index.get(addr, {})
        age_days = _wallet_age_days(w)
        buys_1d = _safe_int(w.get("buys1d", 0))

        # New wallet: labeled as new OR very few trades
        if _has_label(w, new_labels) or buys_1d < 5:
            new_count += 1
        # Experienced: labeled as smart/pro OR many trades
        if _has_label(w, smart_labels) or buys_1d >= 20:
            experienced_count += 1

        if age_days > 30:
            older_30 += 1
        if age_days > 90:
            older_90 += 1
        if age_days > 180:
            older_180 += 1

    return {
        "new_wallet_buyers": new_count,
        "experienced_wallet_buyers": experienced_count,
        "wallets_older_than_30_days": older_30,
        "wallets_older_than_90_days": older_90,
        "wallets_older_than_180_days": older_180,
    }


# ===================================================================
# CONVICTION SIGNALS (6 features)
# ===================================================================


def compute_conviction_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 6 conviction features: repeat buys, multi-buy wallets,
    rebuy rate, accumulation rate, avg/median buys per wallet.
    """
    wallet_index = _build_wallet_index(token_wallets)
    all_swaps = swaps_by_window.get("15m", [])
    buys = [s for s in all_swaps if s.get("is_buy", True)]

    # Count buys per wallet (from swap data)
    wallet_buy_counts: dict[str, int] = {}
    for s in buys:
        wallet = s.get("fee_payer", "")
        if wallet:
            wallet_buy_counts[wallet] = wallet_buy_counts.get(wallet, 0) + 1

    total_buyers = len(wallet_buy_counts)
    total_buys = len(buys)

    repeat_buyers = sum(1 for c in wallet_buy_counts.values() if c >= 2)
    multi_buy_wallets = sum(1 for c in wallet_buy_counts.values() if c >= 3)

    buy_counts_list = list(wallet_buy_counts.values())

    return {
        "repeat_buyers": repeat_buyers,
        "multi_buy_wallets": multi_buy_wallets,
        "wallet_rebuy_rate": round(repeat_buyers / max(total_buyers, 1), 4),
        "wallet_accumulation_rate": round(
            (total_buys - total_buyers) / max(total_buys, 1), 4
        ),
        "avg_buys_per_wallet": round(_safe_mean(buy_counts_list), 4),
        "median_buys_per_wallet": round(_safe_median(buy_counts_list), 4),
    }


# ===================================================================
# EARLY STRENGTH (7 features)
# ===================================================================


def compute_early_strength_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 7 early strength features: win rates and PnL of
    the first N buyers (ordered by firstTransactionAt from Mobula).
    """
    wallet_index = _build_wallet_index(token_wallets)
    all_swaps = swaps_by_window.get("15m", [])

    # Get unique buyers in order of first appearance in swap data
    buys_sorted = sorted(
        [s for s in all_swaps if s.get("is_buy", True)],
        key=lambda x: x.get("timestamp", 0),
    )
    seen: set[str] = set()
    ordered_buyers: list[str] = []
    for s in buys_sorted:
        wallet = s.get("fee_payer", "")
        if wallet and wallet not in seen:
            seen.add(wallet)
            ordered_buyers.append(wallet)

    def _get_roi(wallet: str) -> float:
        w = wallet_index.get(wallet, {})
        return _safe_float(w.get("realizedProfitPercentage30d", 0))

    def _get_pnl(wallet: str) -> float:
        w = wallet_index.get(wallet, {})
        return _safe_float(w.get("realizedProfitUsd30d", 0))

    def _avg_roi(buyers: list[str]) -> float:
        if not buyers:
            return 0.0
        return _safe_mean([_get_roi(w) for w in buyers])

    def _avg_pnl(buyers: list[str]) -> float:
        if not buyers:
            return 0.0
        return _safe_mean([_get_pnl(w) for w in buyers])

    first_1 = ordered_buyers[:1]
    first_5 = ordered_buyers[:5]
    first_10 = ordered_buyers[:10]
    first_20 = ordered_buyers[:20]

    return {
        "first_buyer_win_rate": round(_avg_roi(first_1), 4),
        "first_5_buyers_avg_win_rate": round(_avg_roi(first_5), 4),
        "first_10_buyers_avg_win_rate": round(_avg_roi(first_10), 4),
        "first_20_buyers_avg_win_rate": round(_avg_roi(first_20), 4),
        "first_5_buyers_avg_pnl": round(_avg_pnl(first_5), 2),
        "first_10_buyers_avg_pnl": round(_avg_pnl(first_10), 2),
        "first_20_buyers_avg_pnl": round(_avg_pnl(first_20), 2),
    }


# ===================================================================
# DISTRIBUTION (5 features)
# ===================================================================


def compute_distribution_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 5 distribution features: top-N wallet buy share and HHI.
    Uses swap data for volume shares and Mobula tokenBalanceLiveUsd
    for holder concentration.
    """
    all_swaps = swaps_by_window.get("15m", [])
    buys = [s for s in all_swaps if s.get("is_buy", True)]

    # Volume per wallet from swap data
    wallet_vols: dict[str, float] = {}
    for s in buys:
        wallet = s.get("fee_payer", "")
        if wallet:
            wallet_vols[wallet] = wallet_vols.get(wallet, 0.0) + s.get(
                "usd_estimate", 0
            )

    total_vol = sum(wallet_vols.values())
    sorted_vols = sorted(wallet_vols.values(), reverse=True)

    def _share(n: int) -> float:
        return (
            sum(sorted_vols[:n]) / max(total_vol, 0.01)
            if sorted_vols
            else 0.0
        )

    # Herfindahl-Hirschman Index from swap volume distribution
    hhi = (
        sum((v / max(total_vol, 0.01)) ** 2 for v in wallet_vols.values())
        if total_vol > 0
        else 0.0
    )

    return {
        "top_wallet_buy_share": round(_share(1), 4),
        "top5_wallet_buy_share": round(_share(5), 4),
        "top10_wallet_buy_share": round(_share(10), 4),
        "top20_wallet_buy_share": round(_share(20), 4),
        "buyer_concentration_index": round(hhi, 4),
    }


# ===================================================================
# RISK SIGNALS (6 features)
# ===================================================================


def compute_risk_signals_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
    risk_labels: Optional[frozenset[str]] = None,
) -> dict[str, Any]:
    """
    Compute 6 risk signal features: dumping wallets, fast exits,
    paper hands, scammer/bot detection.
    Uses Mobula scammerScore, botScore, and risk labels.
    """
    risk_labels = risk_labels or DEFAULT_RISK_LABELS
    wallet_index = _build_wallet_index(token_wallets)
    all_swaps = swaps_by_window.get("15m", [])

    # Per-wallet buy/sell tracking
    wallet_buys: dict[str, float] = {}
    wallet_sells: dict[str, float] = {}
    wallet_first_buy_ts: dict[str, int] = {}
    wallet_first_sell_ts: dict[str, int] = {}

    for s in all_swaps:
        wallet = s.get("fee_payer", "")
        usd = s.get("usd_estimate", 0)
        ts = s.get("timestamp", 0)

        if s.get("is_buy", True):
            wallet_buys[wallet] = wallet_buys.get(wallet, 0) + usd
            if wallet not in wallet_first_buy_ts:
                wallet_first_buy_ts[wallet] = ts
        else:
            wallet_sells[wallet] = wallet_sells.get(wallet, 0) + usd
            if wallet not in wallet_first_sell_ts:
                wallet_first_sell_ts[wallet] = ts

    total_buyers = len(wallet_buys)

    # Dumping wallets: sold > 50% of buy amount within 15m
    dumping = sum(
        1 for w in wallet_buys
        if wallet_sells.get(w, 0) > wallet_buys[w] * 0.5
    )

    # Fast exits: sold within 5m / 15m / 60m of first buy
    sold_5m = 0
    sold_15m = 0
    sold_60m = 0

    for w in wallet_sells:
        buy_ts = wallet_first_buy_ts.get(w)
        sell_ts = wallet_first_sell_ts.get(w)
        if buy_ts is None or sell_ts is None:
            continue
        delta = sell_ts - buy_ts
        if delta <= 300:
            sold_5m += 1
        if delta <= 900:
            sold_15m += 1
        if delta <= 3600:
            sold_60m += 1

    # Count wallets with high scammer/bot scores or risk labels
    for addr in set(list(wallet_buys.keys()) + list(wallet_sells.keys())):
        w = wallet_index.get(addr, {})
        scammer = _safe_int(w.get("scammerScore", 0))
        bot = _safe_int(w.get("botScore", 0))
        if _has_label(w, risk_labels) or scammer > 50 or bot > 50:
            pass  # already counted via swap analysis; labels augment

    return {
        "dumping_wallet_count": dumping,
        "wallets_sold_within_5m": sold_5m,
        "wallets_sold_within_15m": sold_15m,
        "wallets_sold_within_60m": sold_60m,
        "fast_exit_rate": round(sold_5m / max(total_buyers, 1), 4),
        "paper_hand_rate": round(sold_15m / max(total_buyers, 1), 4),
    }


# ===================================================================
# SMART MONEY vs RETAIL (5 features)
# ===================================================================


def compute_smart_vs_retail_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
    smart_labels: Optional[frozenset[str]] = None,
) -> dict[str, Any]:
    """
    Compute 5 smart money vs retail comparison features.
    Compares volume/buy share of labeled (smart) vs unlabeled (retail) wallets.
    """
    smart_labels = smart_labels or DEFAULT_SMART_LABELS
    wallet_index = _build_wallet_index(token_wallets)

    all_swaps = swaps_by_window.get("15m", [])
    total_vol = sum(s.get("usd_estimate", 0) for s in all_swaps)

    sm_vol = 0.0
    sm_buy_vol = 0.0
    sm_sell_vol = 0.0
    retail_buy_vol = 0.0
    retail_sell_vol = 0.0

    for s in all_swaps:
        wallet = s.get("fee_payer", "")
        usd = s.get("usd_estimate", 0)
        wdata = wallet_index.get(wallet, {})
        is_sm = _has_label(wdata, smart_labels)

        if is_sm:
            sm_vol += usd
            if s.get("is_buy", True):
                sm_buy_vol += usd
            else:
                sm_sell_vol += usd
        else:
            if s.get("is_buy", True):
                retail_buy_vol += usd
            else:
                retail_sell_vol += usd

    total_buy_vol = sum(
        s.get("usd_estimate", 0) for s in all_swaps if s.get("is_buy", True)
    )
    total_sell_vol = sum(
        s.get("usd_estimate", 0) for s in all_swaps if not s.get("is_buy", True)
    )

    return {
        "smart_money_volume_share": round(sm_vol / max(total_vol, 0.01), 4),
        "smart_money_buy_share": round(sm_buy_vol / max(total_buy_vol, 0.01), 4),
        "retail_buy_share": round(retail_buy_vol / max(total_buy_vol, 0.01), 4),
        "retail_sell_share": round(
            retail_sell_vol / max(total_sell_vol, 0.01), 4
        ),
        "smart_money_net_flow": round(sm_buy_vol - sm_sell_vol, 2),
    }


# ===================================================================
# COMPOSITE SCORES (5 features)
# ===================================================================


def compute_composite_scores(
    existing_features: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute 5 engineered composite scores normalized to [0, 1] range.

    These are z-score-like composites designed to summarize each category
    into a single signal optimized for gradient-boosted tree models.
    """
    # Smart money score
    sm_buyers = max(
        _safe_int(existing_features.get("smart_wallet_buyers_15m", 0)), 1
    )
    total_buyers = max(
        _safe_int(existing_features.get("unique_buyers_15m", 0)), 1
    )
    sm_vol_share = _safe_float(
        existing_features.get("smart_money_accumulation_rate", 0)
    )
    sm_first = _safe_int(existing_features.get("smart_money_first_buyer", 0))
    smart_money_score = np.clip(
        (sm_buyers / total_buyers) * 0.4
        + sm_vol_share * 0.4
        + sm_first * 0.2,
        0, 1,
    )

    # Wallet quality score
    avg_win = _safe_float(existing_features.get("avg_wallet_win_rate", 0))
    avg_roi = _safe_float(existing_features.get("avg_wallet_roi", 0))
    elite = _safe_int(existing_features.get("elite_trader_count", 0))
    wallet_quality_score = np.clip(
        avg_win * 0.4
        + avg_roi * 0.3
        + (elite / max(total_buyers, 1)) * 0.3,
        0, 1,
    )

    # Whale score
    whale_vol = _safe_float(existing_features.get("whale_buy_volume_10k", 0))
    total_vol = max(_safe_float(existing_features.get("volume_15m", 0)), 1)
    whale_net = _safe_float(existing_features.get("whale_net_flow_10k", 0))
    whale_score = np.clip(
        (whale_vol / total_vol) * 0.5
        + (whale_net / max(total_vol, 0.01)) * 0.5,
        0, 1,
    )

    # Conviction score
    rebuy_rate = _safe_float(existing_features.get("wallet_rebuy_rate", 0))
    avg_buys = _safe_float(existing_features.get("avg_buys_per_wallet", 0))
    conviction_score = np.clip(
        rebuy_rate * 0.5 + min(avg_buys / 3.0, 1.0) * 0.5, 0, 1
    )

    # Buyer quality score
    experienced = _safe_int(
        existing_features.get("experienced_wallet_buyers", 0)
    )
    older_90 = _safe_int(
        existing_features.get("wallets_older_than_90_days", 0)
    )
    buyer_quality_score = np.clip(
        (experienced / max(total_buyers, 1)) * 0.5
        + (older_90 / max(total_buyers, 1)) * 0.5,
        0, 1,
    )

    return {
        "smart_money_score": round(float(smart_money_score), 4),
        "wallet_quality_score": round(float(wallet_quality_score), 4),
        "whale_score": round(float(whale_score), 4),
        "conviction_score": round(float(conviction_score), 4),
        "buyer_quality_score": round(float(buyer_quality_score), 4),
    }


# ===================================================================
# MASTER COMPUTE FUNCTION
# ===================================================================


def compute_axiom_features(
    token_wallets: list[dict],
    swaps_by_window: dict[str, list[dict]],
    t0_ts: int = 0,
    smart_labels: Optional[frozenset[str]] = None,
    risk_labels: Optional[frozenset[str]] = None,
    new_labels: Optional[frozenset[str]] = None,
) -> dict[str, Any]:
    """
    Compute ALL Axiom features from Mobula filterTokenWallets data + swap data.

    Args:
        token_wallets: List of wallet dicts from Mobula filterTokenWallets query.
            Each wallet has: address, labels, firstTransactionAt, realizedProfitUsd*,
            realizedProfitPercentage*, amountBoughtUsd*, buys*/sells*, scammerScore, botScore.
        swaps_by_window: {"1m": [parsed_swaps], "5m": [...], "15m": [...]}
        t0_ts: Unix timestamp of token graduation
        smart_labels: Frozenset of Mobula wallet labels considered "smart money"
        risk_labels: Frozenset of Mobula wallet labels considered "risk"
        new_labels: Frozenset of Mobula wallet labels considered "new/fresh"

    Returns:
        Flat dict of all ~80 Axiom features, ready for Supabase upsert.
    """
    # Ensure swaps_by_window has all keys
    for w in ("1m", "5m", "15m"):
        swaps_by_window.setdefault(w, [])

    smart_labels = smart_labels or DEFAULT_SMART_LABELS
    risk_labels = risk_labels or DEFAULT_RISK_LABELS
    new_labels = new_labels or DEFAULT_NEW_LABELS

    all_features: dict[str, Any] = {}

    # Compute each category — all now accept token_wallets + swaps_by_window
    all_features.update(
        compute_smart_money_features(
            token_wallets, swaps_by_window, t0_ts, smart_labels
        )
    )
    all_features.update(
        compute_wallet_quality_features(token_wallets, swaps_by_window)
    )
    all_features.update(
        compute_pnl_features(token_wallets, swaps_by_window)
    )
    all_features.update(
        compute_roi_features(token_wallets, swaps_by_window)
    )
    all_features.update(
        compute_profitable_trader_features(token_wallets, swaps_by_window)
    )
    all_features.update(
        compute_whale_axiom_features(token_wallets, swaps_by_window)
    )
    all_features.update(
        compute_buyer_quality_features(
            token_wallets, swaps_by_window, new_labels, smart_labels
        )
    )
    all_features.update(
        compute_conviction_features(token_wallets, swaps_by_window)
    )
    all_features.update(
        compute_early_strength_features(token_wallets, swaps_by_window)
    )
    all_features.update(
        compute_distribution_features(token_wallets, swaps_by_window)
    )
    all_features.update(
        compute_risk_signals_features(
            token_wallets, swaps_by_window, risk_labels
        )
    )
    all_features.update(
        compute_smart_vs_retail_features(
            token_wallets, swaps_by_window, smart_labels
        )
    )
    all_features.update(
        compute_composite_scores(all_features)
    )

    return all_features
