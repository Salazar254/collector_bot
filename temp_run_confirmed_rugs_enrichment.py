from pathlib import Path
import pandas as pd
import numpy as np
import math

CONFIG = {
    "RAW_DATA_PATH": "/kaggle/working/raw_data.parquet",
    "ENTRY_DELAY_SECONDS": 120,
    "LABEL_WINDOW_HOURS": 72,
    "PRICE_DROP_RUG_THRESHOLD": 0.80,
    "LIQUIDITY_DROP_RUG_THRESHOLD": 0.70,
}

RAW_DATA_PATH = Path(CONFIG["RAW_DATA_PATH"])
RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

confirmed_path = Path("confirmed_rugs.csv")
if not confirmed_path.exists():
    raise SystemExit("confirmed_rugs.csv not found")

confirmed = pd.read_csv(confirmed_path, parse_dates=["rug_date"], keep_default_na=False)
if "mint" not in confirmed.columns and "token" in confirmed.columns:
    confirmed["mint"] = confirmed["token"]

expected_cols = {"mint", "rug_date", "amount_stolen_usd", "rug_method", "pump_before_rug", "liquidity_drain_pct"}
missing = expected_cols - set(confirmed.columns)
print("confirmed columns:", list(confirmed.columns))
print("missing expected columns:", missing)
if missing:
    raise SystemExit("confirmed_rugs.csv is missing required columns")

first_mint = str(confirmed.iloc[0]["mint"])
print("first confirmed mint:", first_mint)

if not RAW_DATA_PATH.exists():
    print("RAW_DATA_PATH not found; writing sample raw data with first confirmed mint")
    sample = pd.DataFrame([
        {
            "mint": first_mint,
            "deployer": "test_deployer",
            "graduation_timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
            "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
            "signature": None,
            "buyer": None,
            "seller": None,
            "sol_amount": 100.0,
            "token_amount": 1.0,
            "fee_sol": 0.0,
            "lpBurnPct": 0.0,
            "lpLockedPct": 0.0,
            "initial_liquidity_sol": 100.0,
            "topHolderPct": 0.0,
            "devHoldPct": 0.0,
            "mutableMetadata": False,
            "mintAuthorityRenounced": True,
            "freezeAuthorityRenounced": True,
            "transferTaxPct": 0.0,
            "rugPullRisk": 0.0,
            "honeypotRisk": 0.0,
            "dangerSignals": 0.0,
        }
    ])
    sample.to_parquet(RAW_DATA_PATH, index=False)
    print("sample RAW_DATA_PATH created")
else:
    print("RAW_DATA_PATH exists; using existing file")

raw_df = pd.read_parquet(RAW_DATA_PATH)
print("raw_df rows", len(raw_df))

confirmed["mint"] = confirmed["mint"].astype(str)
confirmed["pump_before_rug"] = confirmed["pump_before_rug"].astype(bool)
raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)

present = confirmed[confirmed["mint"].isin(raw_df["mint"])].copy()
print("matched confirmed rows in raw_df:", len(present))

LABEL_OVERRIDES = {}
enriched_mints = []
deployer_counts = {}

def safe_float(value, fallback=0.0) -> float:
    try:
        if value is None or value == "":
            return fallback
        value = float(value)
        return value if math.isfinite(value) else fallback
    except Exception:
        return fallback

for _, crow in present.iterrows():
    mint = str(crow["mint"])
    token_rows = raw_df[raw_df["mint"] == mint].sort_values("timestamp")
    if token_rows.empty:
        continue
    first_swap = pd.to_datetime(token_rows["timestamp"].min(), utc=True)
    rug_date = pd.to_datetime(crow["rug_date"], utc=True)
    time_to_rug_hours = max(0.0, (rug_date - first_swap).total_seconds() / 3600.0)
    pump_label = 1.0 if bool(crow.get("pump_before_rug", False)) else 0.0
    raw_liq = safe_float(crow.get("liquidity_drain_pct", 0.0), 0.0)
    if 0.0 <= raw_liq <= 1.0:
        liquidity_pct = float(raw_liq * 100.0)
    else:
        liquidity_pct = float(raw_liq)

    LABEL_OVERRIDES[mint] = {
        "rug_label": 1.0,
        "time_to_rug_hours": float(time_to_rug_hours),
        "override_max_drawdown_pct": float(liquidity_pct),
        "pump_2x_label": float(pump_label),
    }
    enriched_mints.append(mint)
    deployer = str(token_rows.iloc[0].get("deployer", ""))
    deployer_counts[deployer] = deployer_counts.get(deployer, 0) + 1

known_deployers = {d for d, c in deployer_counts.items() if c >= 2}
raw_df["known_rugger_deployer"] = raw_df["deployer"].astype(str).isin(known_deployers)
raw_df["override_rug_label"] = raw_df["mint"].map(lambda m: LABEL_OVERRIDES.get(str(m), {}).get("rug_label", np.nan))
raw_df["override_time_to_rug_hours"] = raw_df["mint"].map(lambda m: LABEL_OVERRIDES.get(str(m), {}).get("time_to_rug_hours", np.nan))
raw_df["override_max_drawdown_pct"] = raw_df["mint"].map(lambda m: LABEL_OVERRIDES.get(str(m), {}).get("override_max_drawdown_pct", np.nan))
raw_df["override_pump_2x_label"] = raw_df["mint"].map(lambda m: LABEL_OVERRIDES.get(str(m), {}).get("pump_2x_label", np.nan))

print("enriched_count", len(set(enriched_mints)))
print("total_mints", int(raw_df["mint"].nunique()))
print("label_override_rate", len(set(enriched_mints)) / max(int(raw_df["mint"].nunique()), 1))

raw_df.to_parquet(RAW_DATA_PATH, index=False)
print("wrote enriched RAW_DATA_PATH at", RAW_DATA_PATH)
print(raw_df.head().to_string(index=False))
