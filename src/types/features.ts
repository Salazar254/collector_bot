/**
 * src/types/features.ts
 *
 * TypeScript type definitions for the snapshot-based meme coin data collector.
 * Consumed by ML pipelines: XGBoost, LightGBM, PyTorch GRU, ONNX deployment.
 *
 * All numeric features are float32 for ONNX compatibility unless noted.
 */

// =========================================================================
// Snapshot timestamps
// =========================================================================

export type SnapshotTimestamp = "t0" | "t1m" | "t5m" | "t15m";

// =========================================================================
// TOKEN — identity fields
// =========================================================================

export interface TokenIdentity {
  mint_address: string;
  symbol: string;
  migration_timestamp: number; // unix seconds (int32-compatible)
}

// =========================================================================
// PRICE — 9 features, source: DexScreener
// =========================================================================

export interface PriceFeatures {
  price_usd_t0: number;    // float32
  price_usd_1m: number;    // float32
  price_usd_5m: number;    // float32
  price_usd_15m: number;   // float32

  price_change_1m_pct: number;   // percentage change T0→T0+1m
  price_change_5m_pct: number;   // percentage change T0→T0+5m
  price_change_15m_pct: number;  // percentage change T0→T0+15m

  max_price_first_15m: number;   // highest price observed in [T0, T0+15m]
  min_price_first_15m: number;   // lowest price observed in [T0, T0+15m]
}

// =========================================================================
// LIQUIDITY — 6 features, source: DexScreener
// =========================================================================

export interface LiquidityFeatures {
  liquidity_usd_t0: number;
  liquidity_usd_1m: number;
  liquidity_usd_5m: number;
  liquidity_usd_15m: number;

  liquidity_growth_5m: number;   // (liq_5m - liq_t0) / liq_t0, 0 if liq_t0=0
  liquidity_growth_15m: number;  // (liq_15m - liq_t0) / liq_t0, 0 if liq_t0=0
}

// =========================================================================
// VOLUME — 3 features, source: Helius swap transactions
// =========================================================================

export interface VolumeFeatures {
  volume_1m: number;   // cumulative USD volume [T0, T0+1m]
  volume_5m: number;   // cumulative USD volume [T0, T0+5m]
  volume_15m: number;  // cumulative USD volume [T0, T0+15m]
}

// =========================================================================
// BUYERS — 4 features, source: Helius swap transactions
// =========================================================================

export interface BuyerFeatures {
  unique_buyers_1m: number;   // int32 — unique buyer addresses [T0, T0+1m]
  unique_buyers_5m: number;   // int32
  unique_buyers_15m: number;  // int32
  buyer_growth_rate: number;  // (buyers_15m - buyers_1m) / max(buyers_1m, 1)
}

// =========================================================================
// SELLERS — 4 features, source: Helius swap transactions
// =========================================================================

export interface SellerFeatures {
  unique_sellers_1m: number;   // int32
  unique_sellers_5m: number;   // int32
  unique_sellers_15m: number;  // int32
  seller_growth_rate: number;  // (sellers_15m - sellers_1m) / max(sellers_1m, 1)
}

// =========================================================================
// ORDER FLOW — 10 features, source: Helius swap transactions
// =========================================================================

export interface OrderFlowFeatures {
  buy_count_1m: number;          // int32 — number of buy swaps [T0, T0+1m]
  buy_count_5m: number;          // int32
  buy_count_15m: number;         // int32

  sell_count_1m: number;         // int32 — number of sell swaps [T0, T0+1m]
  sell_count_5m: number;         // int32
  sell_count_15m: number;        // int32

  buy_sell_ratio_1m: number;     // float32 — buy_count / max(sell_count, 1)
  buy_sell_ratio_5m: number;     // float32
  buy_sell_ratio_15m: number;    // float32

  net_flow_usd: number;          // float32 — (total_buy_vol − total_sell_vol) over 15m
}

// =========================================================================
// WHALES — 5 features, source: Helius swap transactions (≥10 SOL threshold)
// =========================================================================

export interface WhaleFeatures {
  largest_buy_usd: number;       // float32 — largest single buy in first 15m
  largest_sell_usd: number;      // float32 — largest single sell in first 15m
  whale_buy_count: number;       // int32 — buys ≥ 10 SOL
  whale_sell_count: number;      // int32 — sells ≥ 10 SOL
  whale_net_flow: number;        // float32 — whale_buy_vol − whale_sell_vol (SOL)
}

// =========================================================================
// VOLATILITY — 4 features, computed from price snapshots
// =========================================================================

export interface VolatilityFeatures {
  volatility_1m: number;         // float32 — std dev of price returns [T0, T0+1m]
  volatility_5m: number;         // float32
  volatility_15m: number;        // float32
  drawdown_first_15m: number;    // float32 — max(0, (max_price − min_price) / max_price) * 100
}

// =========================================================================
// All token features composed
// =========================================================================

export interface TokenFeatures
  extends PriceFeatures,
    LiquidityFeatures,
    VolumeFeatures,
    BuyerFeatures,
    SellerFeatures,
    OrderFlowFeatures,
    WhaleFeatures,
    VolatilityFeatures {}

// =========================================================================
// Labels (profit-tier targets from DexScreener 24h price change)
// =========================================================================

export interface TokenLabels {
  did_1_25x: number;             // int32 — 1 if price_change_24h ≥ 25%
  did_1_5x: number;              // int32 — 1 if price_change_24h ≥ 50%
  did_2x: number;                // int32 — 1 if price_change_24h ≥ 100%
  did_3x: number;                // int32 — 1 if price_change_24h ≥ 200%
  did_5x: number;                // int32 — 1 if price_change_24h ≥ 400%
  did_10x: number;               // int32 — 1 if price_change_24h ≥ 900%
  rugged: number;                // int32 — 1 if price_change_24h ≤ -80% OR liquidity < $10
  survived_24h: number;          // int32 — 1 if liquidity ≥ $10 AND pair still alive after 24h
  max_drawdown_pct: number;      // float32 — worst observed drawdown
  inferred_label: number;        // int32 — 1 if labels were computed (vs fallback 0)
}

// =========================================================================
// Full snapshot record (stored in Supabase)
// =========================================================================

import type { AxiomFeatures, AxiomMetadata } from "./axiom_features";

export interface TokenSnapshot
  extends TokenIdentity,
    TokenFeatures,
    AxiomFeatures,
    AxiomMetadata,
    TokenLabels {
  collected_at: string;          // ISO 8601 timestamp
  deployer_address: string;
}

// =========================================================================
// Quality validation report
// =========================================================================

export interface FeatureQualityFlag {
  feature_name: string;
  category: string;
  missing_pct: number;           // percentage of rows with null/zero
  unique_count: number;          // number of distinct values
  unique_ratio: number;          // unique_count / total_rows
  variance: number;              // population variance
  flagged: boolean;              // true if unique_ratio < 0.05 or variance ≈ 0 or missing_rate > 50%
  flag_reason: string | null;    // "low_uniqueness" | "zero_variance" | "high_missing_rate" | null
}

export interface QualityReport {
  timestamp: string;             // ISO 8601
  total_rows: number;
  features_checked: number;
  features_flagged: number;
  flags: FeatureQualityFlag[];
}

// =========================================================================
// CSV export row (flattened for ML training)
// =========================================================================

/**
 * Column order for CSV export:
 *   1. mint_address
 *   2. symbol
 *   3. migration_timestamp
 *   4. collected_at
 *   5-13.  PRICE (9 cols)
 *   14-19. LIQUIDITY (6 cols)
 *   20-22. VOLUME (3 cols)
 *   23-26. BUYERS (4 cols)
 *   27-30. SELLERS (4 cols)
 *   31-40. ORDER FLOW (10 cols)
 *   41-45. HOLDERS (5 cols)
 *   46-50. WHALES (5 cols)
 *   51-54. VOLATILITY (4 cols)
 *   55-66.  SMART_MONEY (12 cols)
 *   67-76.  WALLET_QUALITY (10 cols)
 *   77-86.  PNL (10 cols)
 *   87-92.  ROI (6 cols)
 *   93-100. PROFITABLE_TRADER (8 cols)
 *   101-124. WHALE_AXIOM (24 cols)
 *   125-129. BUYER_QUALITY (5 cols)
 *   130-135. CONVICTION (6 cols)
 *   136-142. EARLY_STRENGTH (7 cols)
 *   143-147. DISTRIBUTION (5 cols)
 *   148-153. RISK_SIGNALS (6 cols)
 *   154-158. SMART_VS_RETAIL (5 cols)
 *   159-163. COMPOSITE (5 cols)
 *   164-165. AXIOM META (2 cols)
 *   166-175. LABELS (10 cols)
 */
export interface CSVExportRow extends TokenIdentity, TokenFeatures, AxiomFeatures, AxiomMetadata, TokenLabels {
  collected_at: string;
}
