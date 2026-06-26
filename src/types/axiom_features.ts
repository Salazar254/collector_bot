/**
 * src/types/axiom_features.ts
 *
 * TypeScript type definitions for Axiom API wallet-intelligence features.
 * Adds ~80 features across 13 categories designed to improve prediction of:
 *   did_1.25x, did_1.5x, did_2x, did_3x, did_5x, did_10x, rugged, survived_24h
 *
 * Targets: XGBoost, LightGBM, CatBoost, PyTorch GRU, ONNX deployment.
 */

// =========================================================================
// SMART MONEY (10 features)
// =========================================================================

export interface SmartMoneyFeatures {
  smart_wallet_buyers_1m: number;          // int32 — smart-money buyers [T0, T0+1m]
  smart_wallet_buyers_5m: number;          // int32 — smart-money buyers [T0, T0+5m]
  smart_wallet_buyers_15m: number;         // int32 — smart-money buyers [T0, T0+15m]

  smart_wallet_volume_1m: number;          // float32 — smart-money buy vol USD [T0, T0+1m]
  smart_wallet_volume_5m: number;          // float32
  smart_wallet_volume_15m: number;         // float32

  smart_wallet_percentage: number;         // float32 — smart wallets / total buyers (15m)

  smart_money_first_buyer: number;         // int32 — 1 if a smart-money wallet was the first buyer
  first_smart_money_buy_timestamp: number; // int32 — unix seconds of earliest SM buy

  smart_money_within_first_minute: number; // int32 — count of SM wallets that bought within 60s
  smart_money_within_first_5m: number;     // int32 — count of SM wallets that bought within 300s

  smart_money_accumulation_rate: number;   // float32 — SM buy vol / total buy vol (15m)
}

// =========================================================================
// WALLET QUALITY (10 features)
// =========================================================================

export interface WalletQualityFeatures {
  avg_wallet_age_days: number;             // float32
  median_wallet_age_days: number;          // float32

  avg_wallet_trade_count: number;          // float32 — lifetime trades per buyer wallet
  median_wallet_trade_count: number;       // float32

  avg_wallet_win_rate: number;             // float32 — fraction of profitable trades
  median_wallet_win_rate: number;          // float32

  avg_wallet_realized_pnl: number;         // float32 — lifetime realized PnL (USD)
  median_wallet_realized_pnl: number;      // float32

  avg_wallet_roi: number;                  // float32 — lifetime ROI (fraction)
  median_wallet_roi: number;               // float32
}

// =========================================================================
// PNL FEATURES (10 features)
// =========================================================================

export interface PnlFeatures {
  avg_buyer_pnl_30d: number;              // float32
  median_buyer_pnl_30d: number;           // float32
  top_buyer_pnl_30d: number;              // float32 — max PnL among buyers (30d)

  avg_buyer_pnl_90d: number;              // float32
  median_buyer_pnl_90d: number;           // float32
  top_buyer_pnl_90d: number;              // float32

  avg_seller_pnl_30d: number;             // float32
  median_seller_pnl_30d: number;          // float32

  avg_seller_pnl_90d: number;             // float32
  median_seller_pnl_90d: number;          // float32
}

// =========================================================================
// WHALE METRICS — per threshold: $5K, $10K (16 features)
// =========================================================================

export interface WhaleAxiomFeatures {
  // --- $5,000 threshold ---
  largest_buy_usd_5k: number;             // float32
  largest_sell_usd_5k: number;            // float32
  whale_buy_count_5k: number;             // int32
  whale_sell_count_5k: number;            // int32
  whale_buy_volume_5k: number;            // float32
  whale_sell_volume_5k: number;           // float32
  whale_net_flow_5k: number;              // float32
  whale_accumulation_rate_5k: number;     // float32

  // --- $10,000 threshold ---
  largest_buy_usd_10k: number;            // float32
  largest_sell_usd_10k: number;           // float32
  whale_buy_count_10k: number;            // int32
  whale_sell_count_10k: number;           // int32
  whale_buy_volume_10k: number;           // float32
  whale_sell_volume_10k: number;          // float32
  whale_net_flow_10k: number;             // float32
  whale_accumulation_rate_10k: number;    // float32
}

// =========================================================================
// BUYER QUALITY (5 features)
// =========================================================================

export interface BuyerQualityFeatures {
  new_wallet_buyers: number;              // int32 — buyers with < 10 lifetime trades
  experienced_wallet_buyers: number;      // int32 — buyers with >= 100 lifetime trades

  wallets_older_than_30_days: number;     // int32 — buyers whose wallet age > 30 days
  wallets_older_than_90_days: number;     // int32
  wallets_older_than_180_days: number;    // int32
}

// =========================================================================
// CONVICTION SIGNALS (6 features)
// =========================================================================

export interface ConvictionFeatures {
  repeat_buyers: number;                  // int32 — wallets that made >= 2 buys of this token
  multi_buy_wallets: number;              // int32 — wallets with >= 3 buys

  wallet_rebuy_rate: number;              // float32 — repeat_buyers / total buyers
  wallet_accumulation_rate: number;       // float32 — (total buys − unique buyers) / total buys

  avg_buys_per_wallet: number;            // float32
  median_buys_per_wallet: number;         // float32
}

// =========================================================================
// EARLY STRENGTH (7 features)
// =========================================================================

export interface EarlyStrengthFeatures {
  first_buyer_win_rate: number;           // float32 — 1st buyer's historical win rate

  first_5_buyers_avg_win_rate: number;    // float32
  first_10_buyers_avg_win_rate: number;   // float32
  first_20_buyers_avg_win_rate: number;   // float32

  first_5_buyers_avg_pnl: number;         // float32
  first_10_buyers_avg_pnl: number;        // float32
  first_20_buyers_avg_pnl: number;        // float32
}

// =========================================================================
// DISTRIBUTION (5 features)
// =========================================================================

export interface DistributionFeatures {
  top_wallet_buy_share: number;           // float32 — largest buyer share of total buy vol
  top5_wallet_buy_share: number;          // float32
  top10_wallet_buy_share: number;         // float32
  top20_wallet_buy_share: number;         // float32

  buyer_concentration_index: number;      // float32 — Herfindahl-Hirschman index of buy distribution
}

// =========================================================================
// RISK SIGNALS (5 features)
// =========================================================================

export interface RiskSignalsFeatures {
  dumping_wallet_count: number;           // int32 — wallets that sold > 50% of their buy within 15m

  wallets_sold_within_5m: number;         // int32 — wallets that sold within 5 min of buying
  wallets_sold_within_15m: number;        // int32
  wallets_sold_within_60m: number;        // int32

  fast_exit_rate: number;                 // float32 — wallets_sold_within_5m / total_buyers
  paper_hand_rate: number;                // float32 — wallets_sold_within_15m / total_buyers
}

// =========================================================================
// SMART MONEY vs RETAIL (5 features)
// =========================================================================

export interface SmartVsRetailFeatures {
  smart_money_volume_share: number;       // float32 — SM volume / total volume
  smart_money_buy_share: number;          // float32 — SM buy volume / total buy volume
  retail_buy_share: number;               // float32 — non-SM buy volume / total buy volume
  retail_sell_share: number;              // float32 — non-SM sell volume / total sell volume
  smart_money_net_flow: number;           // float32 — SM buy vol − SM sell vol (USD)
}

// =========================================================================
// COMPOSITE SCORES (5 features) — engineered z-score composites
// =========================================================================

export interface CompositeFeatures {
  smart_money_score: number;              // float32 — normalized [0,1]
  wallet_quality_score: number;           // float32 — normalized [0,1]
  whale_score: number;                    // float32 — normalized [0,1]
  conviction_score: number;               // float32 — normalized [0,1]
  buyer_quality_score: number;            // float32 — normalized [0,1]
}

// =========================================================================
// All Axiom features composed
// =========================================================================

export interface AxiomFeatures
  extends SmartMoneyFeatures,
    WalletQualityFeatures,
    PnlFeatures,
    WhaleAxiomFeatures,
    BuyerQualityFeatures,
    ConvictionFeatures,
    EarlyStrengthFeatures,
    DistributionFeatures,
    RiskSignalsFeatures,
    SmartVsRetailFeatures,
    CompositeFeatures {}

// =========================================================================
// Axiom collection metadata
// =========================================================================

export interface AxiomMetadata {
  axiom_collected: number;                // int32 — 1 if Axiom data was collected
  axiom_cost_usd: number;                 // float32 — estimated API cost for this token
}

// =========================================================================
// Axiom raw response (stored in axiom_raw_responses table)
// =========================================================================

export interface AxiomRawResponse {
  id: number;
  mint: string;
  request_type: string;                   // e.g. "wallet_stats", "smart_money", "whale_txns"
  response_json: string;                  // raw JSON text
  cost_usd: number;
  latency_ms: number;
  status_code: number;
  created_at: string;                     // ISO 8601 timestamp
}

// =========================================================================
// Axiom feature health report entry
// =========================================================================

export interface AxiomFeatureHealthEntry {
  id: number;
  timestamp: string;                      // ISO 8601
  feature_name: string;
  category: string;
  missing_pct: number;
  unique_count: number;
  variance: number;
  flagged: number;                        // int32 — 1 if flagged
  flag_reason: string | null;
  total_rows_sampled: number;
}

// =========================================================================
// AXIOM FEATURE NAME LIST — for iteration / CSV column ordering
// =========================================================================

export const AXIOM_FEATURE_NAMES: readonly string[] = [
  // SMART_MONEY (10)
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

  // WALLET_QUALITY (10)
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

  // PNL (10)
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

  // ROI (6)
  "avg_buyer_roi_30d",
  "median_buyer_roi_30d",
  "top_buyer_roi_30d",
  "avg_buyer_roi_90d",
  "median_buyer_roi_90d",
  "top_buyer_roi_90d",

  // PROFITABLE_TRADER (8)
  "profitable_wallet_count",
  "profitable_wallet_buy_volume",
  "high_roi_wallet_count",
  "elite_trader_count",
  "wallets_with_positive_pnl",
  "wallets_above_20pct_roi",
  "wallets_above_50pct_roi",
  "wallets_above_100pct_roi",

  // WHALE_AXIOM — 1K (8)
  "largest_buy_usd_1k",
  "largest_sell_usd_1k",
  "whale_buy_count_1k",
  "whale_sell_count_1k",
  "whale_buy_volume_1k",
  "whale_sell_volume_1k",
  "whale_net_flow_1k",
  "whale_accumulation_rate_1k",

  // WHALE_AXIOM — 5K (8)
  "largest_buy_usd_5k",
  "largest_sell_usd_5k",
  "whale_buy_count_5k",
  "whale_sell_count_5k",
  "whale_buy_volume_5k",
  "whale_sell_volume_5k",
  "whale_net_flow_5k",
  "whale_accumulation_rate_5k",

  // WHALE_AXIOM — 10K (8)
  "largest_buy_usd_10k",
  "largest_sell_usd_10k",
  "whale_buy_count_10k",
  "whale_sell_count_10k",
  "whale_buy_volume_10k",
  "whale_sell_volume_10k",
  "whale_net_flow_10k",
  "whale_accumulation_rate_10k",

  // BUYER_QUALITY (5)
  "new_wallet_buyers",
  "experienced_wallet_buyers",
  "wallets_older_than_30_days",
  "wallets_older_than_90_days",
  "wallets_older_than_180_days",

  // CONVICTION (6)
  "repeat_buyers",
  "multi_buy_wallets",
  "wallet_rebuy_rate",
  "wallet_accumulation_rate",
  "avg_buys_per_wallet",
  "median_buys_per_wallet",

  // EARLY_STRENGTH (7)
  "first_buyer_win_rate",
  "first_5_buyers_avg_win_rate",
  "first_10_buyers_avg_win_rate",
  "first_20_buyers_avg_win_rate",
  "first_5_buyers_avg_pnl",
  "first_10_buyers_avg_pnl",
  "first_20_buyers_avg_pnl",

  // DISTRIBUTION (5)
  "top_wallet_buy_share",
  "top5_wallet_buy_share",
  "top10_wallet_buy_share",
  "top20_wallet_buy_share",
  "buyer_concentration_index",

  // RISK_SIGNALS (5)
  "dumping_wallet_count",
  "wallets_sold_within_5m",
  "wallets_sold_within_15m",
  "wallets_sold_within_60m",
  "fast_exit_rate",
  "paper_hand_rate",

  // SMART_VS_RETAIL (5)
  "smart_money_volume_share",
  "smart_money_buy_share",
  "retail_buy_share",
  "retail_sell_share",
  "smart_money_net_flow",

  // COMPOSITE (5)
  "smart_money_score",
  "wallet_quality_score",
  "whale_score",
  "conviction_score",
  "buyer_quality_score",
] as const;

export const AXIOM_METADATA_COLUMNS = [
  "axiom_collected",
  "axiom_cost_usd",
] as const;
