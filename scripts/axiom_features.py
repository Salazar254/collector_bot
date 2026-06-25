"""
scripts/axiom_features.py — Feature extraction from Axiom API data.

Computes ~80 wallet-intelligence features organized into 13 categories.
All features are designed to improve prediction of profit-tier targets
and rug detection for XGBoost, LightGBM, CatBoost, PyTorch GRU models.

Input: Raw Axiom API response dicts + existing swap data for cross-referencing.
Output: Flat dict of all Axiom features ready for Supabase upsert.

All compute_* functions accept partial/missing data and return sensible defaults.
They NEVER crash — missing data produces zeros/empty values.
"""

import logging
from statistics import median as _median, mean as _mean
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHALE_THRESHOLDS = [1000.0, 5000.0, 10000.0]
WINDOWS_SEC = {"1m": 60, "5m": 300, "15m": 900}

# Default wallet profile when data is missing
DEFAULT_WALLET = {
    "wallet_age_days": 0,
    "trade_count": 0,
    "win_rate": 0.0,
    "realized_pnl": 0.0,
    "roi": 0.0,
}

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


def _is_smart_money(wallet_addr: str, smart_wallets: list[str]) -> bool:
    """Check if a wallet is classified as smart money."""
    if not smart_wallets:
        return False
    return wallet_addr in smart_wallets


# ===================================================================
# SMART MONEY (12 features)
# ===================================================================


def compute_smart_money_features(
    smart_money_data: dict[str, Any],
    swaps_by_window: dict[str, list[dict]],
    t0_ts: int = 0,
    smart_wallets: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compute 12 smart money features from Axiom smart_money_activity data
    cross-referenced with swap transaction data.
    """
    smart_wallets = smart_wallets or []

    # Extract from Axiom response
    sm_wallets_data = smart_money_data.get("smart_money_wallets", []) or []

    # Build smart money stats per window
    features: dict[str, Any] = {}

    for window_key, window_sec in WINDOWS_SEC.items():
        swaps = swaps_by_window.get(window_key, [])
        sm_buyers = set()
        sm_vol = 0.0
        sm_first_ts = None

        for swap in swaps:
            wallet = swap.get("fee_payer", "")
            is_buy = swap.get("is_buy", True)
            usd = swap.get("usd_estimate", 0.0)
            ts = swap.get("timestamp", 0)

            if is_buy and (
                wallet in smart_wallets
                or _wallet_in_sm_list(wallet, sm_wallets_data)
            ):
                sm_buyers.add(wallet)
                sm_vol += usd
                if sm_first_ts is None or ts < sm_first_ts:
                    sm_first_ts = ts

        features[f"smart_wallet_buyers_{window_key}"] = len(sm_buyers)
        features[f"smart_wallet_volume_{window_key}"] = round(sm_vol, 2)

    # 15m aggregate
    total_buyers_15m = len(
        {s["fee_payer"] for s in swaps_by_window.get("15m", []) if s.get("is_buy", True)}
    )
    sm_buyers_15m = features.get("smart_wallet_buyers_15m", 0)

    features["smart_wallet_percentage"] = round(
        sm_buyers_15m / max(total_buyers_15m, 1), 4
    )

    # Smart money first buyer
    all_buys_sorted = sorted(
        [s for s in swaps_by_window.get("15m", []) if s.get("is_buy", True)],
        key=lambda x: x.get("timestamp", 0),
    )
    first_buyer = all_buys_sorted[0] if all_buys_sorted else {}
    first_wallet = first_buyer.get("fee_payer", "")
    features["smart_money_first_buyer"] = 1 if (
        first_wallet in smart_wallets
        or _wallet_in_sm_list(first_wallet, sm_wallets_data)
    ) else 0

    # First smart money buy timestamp
    sm_buys_sorted = sorted(
        [
            s for s in all_buys_sorted
            if s.get("fee_payer", "") in smart_wallets
            or _wallet_in_sm_list(s.get("fee_payer", ""), sm_wallets_data)
        ],
        key=lambda x: x.get("timestamp", 0),
    )
    features["first_smart_money_buy_timestamp"] = (
        sm_buys_sorted[0].get("timestamp", 0) if sm_buys_sorted else 0
    )

    # Smart money within first minute / first 5m
    sm_within_1m = sum(
        1 for s in sm_buys_sorted
        if 0 <= (s.get("timestamp", 0) - t0_ts) <= 60
    )
    sm_within_5m = sum(
        1 for s in sm_buys_sorted
        if 0 <= (s.get("timestamp", 0) - t0_ts) <= 300
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


def _wallet_in_sm_list(wallet: str, sm_data: list[dict]) -> bool:
    """Check if wallet appears in smart money API response data."""
    if not wallet or not sm_data:
        return False
    for entry in sm_data:
        if isinstance(entry, dict) and entry.get("wallet", "") == wallet:
            return True
    return False


# ===================================================================
# WALLET QUALITY (10 features)
# ===================================================================


def compute_wallet_quality_features(
    wallet_profiles: dict[str, Any],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 10 wallet quality features from Axiom wallet profile data.
    """
    all_swaps = swaps_by_window.get("15m", [])
    buyer_wallets = list({
        s["fee_payer"]
        for s in all_swaps
        if s.get("is_buy", True) and s.get("fee_payer", "")
    })

    if not buyer_wallets:
        return _empty_wallet_quality()

    profiles = wallet_profiles.get("profiles", {}) or wallet_profiles
    if isinstance(profiles, list):
        profiles = {p.get("wallet", ""): p for p in profiles if isinstance(p, dict)}

    # Collect stats
    ages: list[float] = []
    trade_counts: list[float] = []
    win_rates: list[float] = []
    pnls: list[float] = []
    rois: list[float] = []

    for wallet in buyer_wallets:
        profile = profiles.get(wallet, DEFAULT_WALLET) if isinstance(profiles, dict) else DEFAULT_WALLET
        ages.append(_safe_float(profile.get("wallet_age_days", 0)))
        trade_counts.append(_safe_float(profile.get("trade_count", 0)))
        win_rates.append(_safe_float(profile.get("win_rate", 0)))
        pnls.append(_safe_float(profile.get("realized_pnl", 0)))
        rois.append(_safe_float(profile.get("roi", 0)))

    return {
        "avg_wallet_age_days": round(_safe_mean(ages), 2),
        "median_wallet_age_days": round(_safe_median(ages), 2),
        "avg_wallet_trade_count": round(_safe_mean(trade_counts), 2),
        "median_wallet_trade_count": round(_safe_median(trade_counts), 2),
        "avg_wallet_win_rate": round(_safe_mean(win_rates), 4),
        "median_wallet_win_rate": round(_safe_median(win_rates), 4),
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


# ===================================================================
# PNL FEATURES (10 features)
# ===================================================================


def compute_pnl_features(
    wallet_profiles: dict[str, Any],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 10 PnL features from Axiom wallet profile data,
    separated by buyer/seller and 30d/90d windows.
    """
    buyer_pnls_30d, buyer_pnls_90d = _extract_pnls(wallet_profiles, swaps_by_window, is_buyer=True)
    seller_pnls_30d, seller_pnls_90d = _extract_pnls(wallet_profiles, swaps_by_window, is_buyer=False)

    return {
        "avg_buyer_pnl_30d": round(_safe_mean(buyer_pnls_30d), 2),
        "median_buyer_pnl_30d": round(_safe_median(buyer_pnls_30d), 2),
        "top_buyer_pnl_30d": round(max(buyer_pnls_30d) if buyer_pnls_30d else 0.0, 2),
        "avg_buyer_pnl_90d": round(_safe_mean(buyer_pnls_90d), 2),
        "median_buyer_pnl_90d": round(_safe_median(buyer_pnls_90d), 2),
        "top_buyer_pnl_90d": round(max(buyer_pnls_90d) if buyer_pnls_90d else 0.0, 2),
        "avg_seller_pnl_30d": round(_safe_mean(seller_pnls_30d), 2),
        "median_seller_pnl_30d": round(_safe_median(seller_pnls_30d), 2),
        "avg_seller_pnl_90d": round(_safe_mean(seller_pnls_90d), 2),
        "median_seller_pnl_90d": round(_safe_median(seller_pnls_90d), 2),
    }


def _extract_pnls(
    wallet_profiles: dict[str, Any],
    swaps_by_window: dict[str, list[dict]],
    is_buyer: bool,
) -> tuple[list[float], list[float]]:
    """Extract 30d and 90d PnL values for buyer or seller wallets."""
    swaps = swaps_by_window.get("15m", [])
    wallets = {
        s["fee_payer"]
        for s in swaps
        if s.get("is_buy", True) == is_buyer and s.get("fee_payer", "")
    }

    profiles = wallet_profiles.get("profiles", {}) or wallet_profiles
    if isinstance(profiles, list):
        profiles = {p.get("wallet", ""): p for p in profiles if isinstance(p, dict)}

    pnls_30d: list[float] = []
    pnls_90d: list[float] = []

    for wallet in wallets:
        profile = profiles.get(wallet, {}) if isinstance(profiles, dict) else {}
        pnls_30d.append(_safe_float(profile.get("pnl_30d", profile.get("realized_pnl", 0))))
        pnls_90d.append(_safe_float(profile.get("pnl_90d", profile.get("realized_pnl", 0))))

    return pnls_30d, pnls_90d


# ===================================================================
# ROI FEATURES (6 features)
# ===================================================================


def compute_roi_features(
    wallet_profiles: dict[str, Any],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 6 ROI features for buyers across 30d and 90d windows.
    """
    swaps = swaps_by_window.get("15m", [])
    buyer_wallets = {
        s["fee_payer"]
        for s in swaps
        if s.get("is_buy", True) and s.get("fee_payer", "")
    }

    profiles = wallet_profiles.get("profiles", {}) or wallet_profiles
    if isinstance(profiles, list):
        profiles = {p.get("wallet", ""): p for p in profiles if isinstance(p, dict)}

    rois_30d: list[float] = []
    rois_90d: list[float] = []

    for wallet in buyer_wallets:
        profile = profiles.get(wallet, {}) if isinstance(profiles, dict) else {}
        rois_30d.append(_safe_float(profile.get("roi_30d", profile.get("roi", 0))))
        rois_90d.append(_safe_float(profile.get("roi_90d", profile.get("roi", 0))))

    return {
        "avg_buyer_roi_30d": round(_safe_mean(rois_30d), 4),
        "median_buyer_roi_30d": round(_safe_median(rois_30d), 4),
        "top_buyer_roi_30d": round(max(rois_30d) if rois_30d else 0.0, 4),
        "avg_buyer_roi_90d": round(_safe_mean(rois_90d), 4),
        "median_buyer_roi_90d": round(_safe_median(rois_90d), 4),
        "top_buyer_roi_90d": round(max(rois_90d) if rois_90d else 0.0, 4),
    }


# ===================================================================
# PROFITABLE TRADER METRICS (8 features)
# ===================================================================


def compute_profitable_trader_features(
    wallet_profiles: dict[str, Any],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 8 profitable trader metrics from wallet profiles.
    """
    swaps = swaps_by_window.get("15m", [])
    buyer_wallets = {
        s["fee_payer"]
        for s in swaps
        if s.get("is_buy", True) and s.get("fee_payer", "")
    }

    profiles = wallet_profiles.get("profiles", {}) or wallet_profiles
    if isinstance(profiles, list):
        profiles = {p.get("wallet", ""): p for p in profiles if isinstance(p, dict)}

    profitable_count = 0
    profitable_vol = 0.0
    high_roi_count = 0
    elite_count = 0
    positive_pnl = 0
    above_20 = 0
    above_50 = 0
    above_100 = 0

    for wallet in buyer_wallets:
        profile = profiles.get(wallet, {}) if isinstance(profiles, dict) else {}
        pnl = _safe_float(profile.get("realized_pnl", 0))
        roi = _safe_float(profile.get("roi", 0))
        win_rate = _safe_float(profile.get("win_rate", 0))
        trade_count = _safe_int(profile.get("trade_count", 0))

        if pnl > 0:
            profitable_count += 1
            positive_pnl += 1

        if roi >= 1.0:
            high_roi_count += 1
            above_100 += 1
        if roi >= 0.50:
            above_50 += 1
        if roi >= 0.20:
            above_20 += 1

        if win_rate > 0.60 and trade_count > 50:
            elite_count += 1

        # Get buy volume from swap data
        wallet_swaps = [s for s in swaps if s.get("fee_payer", "") == wallet and s.get("is_buy", True)]
        if pnl > 0:
            profitable_vol += sum(s.get("usd_estimate", 0) for s in wallet_swaps)

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
    swaps_by_window: dict[str, list[dict]],
    whale_data_by_threshold: Optional[dict[float, dict]] = None,
) -> dict[str, Any]:
    """
    Compute 24 whale features across 3 thresholds ($1K, $5K, $10K).
    Uses swap data and optionally Axiom whale transaction data.
    """
    features: dict[str, Any] = {}
    all_swaps = swaps_by_window.get("15m", [])

    threshold_labels = {
        1000.0: "1k",
        5000.0: "5k",
        10000.0: "10k",
    }

    for threshold in WHALE_THRESHOLDS:
        label = threshold_labels[threshold]
        whale_swaps = [s for s in all_swaps if s.get("usd_estimate", 0) >= threshold]
        whale_buys = [s for s in whale_swaps if s.get("is_buy", True)]
        whale_sells = [s for s in whale_swaps if not s.get("is_buy", True)]

        total_buy_vol = sum(s.get("usd_estimate", 0) for s in all_swaps if s.get("is_buy", True))

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
    wallet_profiles: dict[str, Any],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 5 buyer quality features: new vs experienced wallets,
    wallet age brackets.
    """
    swaps = swaps_by_window.get("15m", [])
    buyer_wallets = {
        s["fee_payer"]
        for s in swaps
        if s.get("is_buy", True) and s.get("fee_payer", "")
    }

    profiles = wallet_profiles.get("profiles", {}) or wallet_profiles
    if isinstance(profiles, list):
        profiles = {p.get("wallet", ""): p for p in profiles if isinstance(p, dict)}

    new_count = 0
    experienced_count = 0
    older_30 = 0
    older_90 = 0
    older_180 = 0

    for wallet in buyer_wallets:
        profile = profiles.get(wallet, {}) if isinstance(profiles, dict) else {}
        trade_count = _safe_int(profile.get("trade_count", 0))
        age_days = _safe_float(profile.get("wallet_age_days", 0))

        if trade_count < 10:
            new_count += 1
        if trade_count >= 100:
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
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 6 conviction features: repeat buys, multi-buy wallets,
    rebuy rate, accumulation rate, avg/median buys per wallet.
    """
    all_swaps = swaps_by_window.get("15m", [])
    buys = [s for s in all_swaps if s.get("is_buy", True)]

    # Count buys per wallet
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
    wallet_profiles: dict[str, Any],
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 7 early strength features: win rates and PnL of first N buyers.
    """
    all_swaps = swaps_by_window.get("15m", [])
    buys_sorted = sorted(
        [s for s in all_swaps if s.get("is_buy", True)],
        key=lambda x: x.get("timestamp", 0),
    )

    # Get unique buyers in order of first appearance
    seen: set[str] = set()
    ordered_buyers: list[str] = []
    for s in buys_sorted:
        wallet = s.get("fee_payer", "")
        if wallet and wallet not in seen:
            seen.add(wallet)
            ordered_buyers.append(wallet)

    profiles = wallet_profiles.get("profiles", {}) or wallet_profiles
    if isinstance(profiles, list):
        profiles = {p.get("wallet", ""): p for p in profiles if isinstance(p, dict)}

    def _get_win_rate(wallet: str) -> float:
        p = profiles.get(wallet, {}) if isinstance(profiles, dict) else {}
        return _safe_float(p.get("win_rate", 0))

    def _get_pnl(wallet: str) -> float:
        p = profiles.get(wallet, {}) if isinstance(profiles, dict) else {}
        return _safe_float(p.get("realized_pnl", 0))

    def _avg_win_rate(buyers: list[str]) -> float:
        if not buyers:
            return 0.0
        return _safe_mean([_get_win_rate(w) for w in buyers])

    def _avg_pnl(buyers: list[str]) -> float:
        if not buyers:
            return 0.0
        return _safe_mean([_get_pnl(w) for w in buyers])

    first_1 = ordered_buyers[:1]
    first_5 = ordered_buyers[:5]
    first_10 = ordered_buyers[:10]
    first_20 = ordered_buyers[:20]

    return {
        "first_buyer_win_rate": round(_avg_win_rate(first_1), 4),
        "first_5_buyers_avg_win_rate": round(_avg_win_rate(first_5), 4),
        "first_10_buyers_avg_win_rate": round(_avg_win_rate(first_10), 4),
        "first_20_buyers_avg_win_rate": round(_avg_win_rate(first_20), 4),
        "first_5_buyers_avg_pnl": round(_avg_pnl(first_5), 2),
        "first_10_buyers_avg_pnl": round(_avg_pnl(first_10), 2),
        "first_20_buyers_avg_pnl": round(_avg_pnl(first_20), 2),
    }


# ===================================================================
# DISTRIBUTION (5 features)
# ===================================================================


def compute_distribution_features(
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 5 distribution features: top-N wallet buy share and HHI.
    """
    all_swaps = swaps_by_window.get("15m", [])
    buys = [s for s in all_swaps if s.get("is_buy", True)]

    # Volume per wallet
    wallet_vols: dict[str, float] = {}
    for s in buys:
        wallet = s.get("fee_payer", "")
        if wallet:
            wallet_vols[wallet] = wallet_vols.get(wallet, 0.0) + s.get("usd_estimate", 0)

    total_vol = sum(wallet_vols.values())
    sorted_vols = sorted(wallet_vols.values(), reverse=True)

    def _share(n: int) -> float:
        return sum(sorted_vols[:n]) / max(total_vol, 0.01) if sorted_vols else 0.0

    # Herfindahl-Hirschman Index
    hhi = sum((v / max(total_vol, 0.01)) ** 2 for v in wallet_vols.values()) if total_vol > 0 else 0.0

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
    swaps_by_window: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute 6 risk signal features: dumping wallets, fast exits, paper hands.
    """
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

    # Dumping wallets: sold > 50% of their buy amount within 15m
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
    smart_money_data: dict[str, Any],
    swaps_by_window: dict[str, list[dict]],
    smart_wallets: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compute 5 smart money vs retail comparison features.
    """
    smart_wallets = smart_wallets or []
    sm_wallets_data = smart_money_data.get("smart_money_wallets", []) or []

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
        is_sm = wallet in smart_wallets or _wallet_in_sm_list(wallet, sm_wallets_data)

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

    total_buy_vol = sum(s.get("usd_estimate", 0) for s in all_swaps if s.get("is_buy", True))
    total_sell_vol = sum(s.get("usd_estimate", 0) for s in all_swaps if not s.get("is_buy", True))

    return {
        "smart_money_volume_share": round(sm_vol / max(total_vol, 0.01), 4),
        "smart_money_buy_share": round(sm_buy_vol / max(total_buy_vol, 0.01), 4),
        "retail_buy_share": round(retail_buy_vol / max(total_buy_vol, 0.01), 4),
        "retail_sell_share": round(retail_sell_vol / max(total_sell_vol, 0.01), 4),
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
    sm_buyers = max(_safe_int(existing_features.get("smart_wallet_buyers_15m", 0)), 1)
    total_buyers = max(_safe_int(existing_features.get("unique_buyers_15m", 0)), 1)
    sm_vol_share = _safe_float(existing_features.get("smart_money_accumulation_rate", 0))
    sm_first = _safe_int(existing_features.get("smart_money_first_buyer", 0))
    smart_money_score = np.clip(
        (sm_buyers / total_buyers) * 0.4 + sm_vol_share * 0.4 + sm_first * 0.2, 0, 1
    )

    # Wallet quality score
    avg_win = _safe_float(existing_features.get("avg_wallet_win_rate", 0))
    avg_roi = _safe_float(existing_features.get("avg_wallet_roi", 0))
    elite = _safe_int(existing_features.get("elite_trader_count", 0))
    wallet_quality_score = np.clip(
        avg_win * 0.4 + avg_roi * 0.3 + (elite / max(total_buyers, 1)) * 0.3, 0, 1
    )

    # Whale score
    whale_vol = _safe_float(existing_features.get("whale_buy_volume_10k", 0))
    total_vol = max(_safe_float(existing_features.get("volume_15m", 0)), 1)
    whale_net = _safe_float(existing_features.get("whale_net_flow_10k", 0))
    whale_score = np.clip(
        (whale_vol / total_vol) * 0.5 + (whale_net / max(total_vol, 0.01)) * 0.5, 0, 1
    )

    # Conviction score
    rebuy_rate = _safe_float(existing_features.get("wallet_rebuy_rate", 0))
    avg_buys = _safe_float(existing_features.get("avg_buys_per_wallet", 0))
    conviction_score = np.clip(
        rebuy_rate * 0.5 + min(avg_buys / 3.0, 1.0) * 0.5, 0, 1
    )

    # Buyer quality score
    experienced = _safe_int(existing_features.get("experienced_wallet_buyers", 0))
    older_90 = _safe_int(existing_features.get("wallets_older_than_90_days", 0))
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
    axiom_data: dict[str, Any],
    swaps_by_window: dict[str, list[dict]],
    t0_ts: int = 0,
    smart_wallets: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compute ALL Axiom features from raw Axiom API responses and swap data.

    Args:
        axiom_data: Dict with keys matching AxiomClient method names:
            - 'smart_money_activity': raw smart money API response
            - 'wallet_profiles': raw wallet profiles API response
            - 'whale_transactions': optional per-threshold whale data
        swaps_by_window: {"1m": [parsed_swaps], "5m": [...], "15m": [...]}
        t0_ts: Unix timestamp of token graduation
        smart_wallets: Optional list of known smart money wallet addresses

    Returns:
        Flat dict of all ~80 Axiom features, ready for Supabase upsert.
    """
    # Ensure swaps_by_window has all keys
    for w in ("1m", "5m", "15m"):
        swaps_by_window.setdefault(w, [])

    smart_money_data = axiom_data.get("smart_money_activity", {})
    wallet_profiles = axiom_data.get("wallet_profiles", {})
    whale_data = axiom_data.get("whale_transactions", None)

    all_features: dict[str, Any] = {}

    # Compute each category
    all_features.update(
        compute_smart_money_features(smart_money_data, swaps_by_window, t0_ts, smart_wallets)
    )
    all_features.update(
        compute_wallet_quality_features(wallet_profiles, swaps_by_window)
    )
    all_features.update(
        compute_pnl_features(wallet_profiles, swaps_by_window)
    )
    all_features.update(
        compute_roi_features(wallet_profiles, swaps_by_window)
    )
    all_features.update(
        compute_profitable_trader_features(wallet_profiles, swaps_by_window)
    )
    all_features.update(
        compute_whale_axiom_features(swaps_by_window, whale_data)
    )
    all_features.update(
        compute_buyer_quality_features(wallet_profiles, swaps_by_window)
    )
    all_features.update(
        compute_conviction_features(swaps_by_window)
    )
    all_features.update(
        compute_early_strength_features(wallet_profiles, swaps_by_window)
    )
    all_features.update(
        compute_distribution_features(swaps_by_window)
    )
    all_features.update(
        compute_risk_signals_features(swaps_by_window)
    )
    all_features.update(
        compute_smart_vs_retail_features(smart_money_data, swaps_by_window, smart_wallets)
    )
    all_features.update(
        compute_composite_scores(all_features)
    )

    return all_features
