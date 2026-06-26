-- scripts/create_table.sql
-- Supabase table schema for collector_bot training_tokens (v2 — snapshot-based).
-- Row size: ~382 bytes → 1M rows ≈ 364 MB (well under 500 MB free-tier limit).
-- Replaces the old constant-feature schema with time-series snapshot features
-- collected at T0, T0+1m, T0+5m, T0+15m after pump.fun graduation.

CREATE TABLE IF NOT EXISTS training_tokens (
  id SERIAL PRIMARY KEY,
  mint TEXT UNIQUE NOT NULL,
  symbol TEXT,
  graduation_timestamp BIGINT,
  collected_at TIMESTAMPTZ DEFAULT NOW(),

  -- =======================================================================
  -- PRICE (9 features, source: DexScreener)
  -- =======================================================================
  price_usd_t0           FLOAT4,
  price_usd_1m           FLOAT4,
  price_usd_5m           FLOAT4,
  price_usd_15m          FLOAT4,
  price_change_1m_pct    FLOAT4,   -- (price_1m - price_t0) / price_t0 * 100
  price_change_5m_pct    FLOAT4,   -- (price_5m - price_t0) / price_t0 * 100
  price_change_15m_pct   FLOAT4,   -- (price_15m - price_t0) / price_t0 * 100
  max_price_first_15m    FLOAT4,   -- highest price in [T0, T0+15m]
  min_price_first_15m    FLOAT4,   -- lowest price in [T0, T0+15m]

  -- =======================================================================
  -- LIQUIDITY (6 features, source: DexScreener)
  -- =======================================================================
  liquidity_usd_t0       FLOAT4,
  liquidity_usd_1m       FLOAT4,
  liquidity_usd_5m       FLOAT4,
  liquidity_usd_15m      FLOAT4,
  liquidity_growth_5m    FLOAT4,   -- (liq_5m - liq_t0) / liq_t0, 0 if liq_t0=0
  liquidity_growth_15m   FLOAT4,   -- (liq_15m - liq_t0) / liq_t0, 0 if liq_t0=0

  -- =======================================================================
  -- VOLUME (3 features, source: Helius swap transactions)
  -- =======================================================================
  volume_1m              FLOAT4,   -- cumulative USD volume [T0, T0+1m]
  volume_5m              FLOAT4,   -- cumulative USD volume [T0, T0+5m]
  volume_15m             FLOAT4,   -- cumulative USD volume [T0, T0+15m]

  -- =======================================================================
  -- BUYERS (4 features, source: Helius swap transactions)
  -- =======================================================================
  unique_buyers_1m       INT4,     -- distinct buyer wallet addresses [T0, T0+1m]
  unique_buyers_5m       INT4,     -- distinct buyer wallet addresses [T0, T0+5m]
  unique_buyers_15m      INT4,     -- distinct buyer wallet addresses [T0, T0+15m]
  buyer_growth_rate      FLOAT4,   -- (buyers_15m - buyers_1m) / max(buyers_1m, 1)

  -- =======================================================================
  -- SELLERS (4 features, source: Helius swap transactions)
  -- =======================================================================
  unique_sellers_1m      INT4,     -- distinct seller wallet addresses [T0, T0+1m]
  unique_sellers_5m      INT4,     -- distinct seller wallet addresses [T0, T0+5m]
  unique_sellers_15m     INT4,     -- distinct seller wallet addresses [T0, T0+15m]
  seller_growth_rate     FLOAT4,   -- (sellers_15m - sellers_1m) / max(sellers_1m, 1)

  -- =======================================================================
  -- ORDER FLOW (10 features, source: Helius swap transactions)
  -- =======================================================================
  buy_count_1m           INT4,     -- number of buy swaps [T0, T0+1m]
  buy_count_5m           INT4,     -- number of buy swaps [T0, T0+5m]
  buy_count_15m          INT4,     -- number of buy swaps [T0, T0+15m]
  sell_count_1m          INT4,     -- number of sell swaps [T0, T0+1m]
  sell_count_5m          INT4,     -- number of sell swaps [T0, T0+5m]
  sell_count_15m         INT4,     -- number of sell swaps [T0, T0+15m]
  buy_sell_ratio_1m      FLOAT4,   -- buy_count_1m / max(sell_count_1m, 1)
  buy_sell_ratio_5m      FLOAT4,   -- buy_count_5m / max(sell_count_5m, 1)
  buy_sell_ratio_15m     FLOAT4,   -- buy_count_15m / max(sell_count_15m, 1)
  net_flow_usd           FLOAT4,   -- (total_buy_vol - total_sell_vol) over 15m

  -- =======================================================================
  -- WHALES (5 features, source: Helius swap transactions, >= 10 SOL)
  -- =======================================================================
  largest_buy_usd        FLOAT4,   -- largest single buy in first 15m
  largest_sell_usd       FLOAT4,   -- largest single sell in first 15m
  whale_buy_count        INT4,     -- number of buy swaps >= 10 SOL
  whale_sell_count       INT4,     -- number of sell swaps >= 10 SOL
  whale_net_flow         FLOAT4,   -- whale_buy_vol - whale_sell_vol (SOL)

  -- =======================================================================
  -- VOLATILITY (4 features, computed from price snapshots)
  -- =======================================================================
  volatility_1m          FLOAT4,   -- std dev of price returns [T0, T0+1m]
  volatility_5m          FLOAT4,   -- std dev of price returns [T0, T0+5m]
  volatility_15m         FLOAT4,   -- std dev of price returns [T0, T0+15m]
  drawdown_first_15m     FLOAT4,   -- max(0, (max_price - min_price) / max_price) * 100

  -- =======================================================================
  -- SAFETY (9 features, source: Helius DAS + DexScreener + computed)
  -- =======================================================================
  mint_authority_active   FLOAT4,   -- 1 if mint authority not revoked
  freeze_authority_active FLOAT4,   -- 1 if freeze authority not revoked
  mutable_metadata        FLOAT4,   -- 1 if token metadata is mutable
  lp_burn_pct             FLOAT4,   -- percentage of LP tokens burned (0-100)
  initial_liquidity_sol   FLOAT4,   -- SOL in LP pool at T0 (from DexScreener)
  migration_speed_seconds FLOAT4,   -- graduation_ts - pair_created_at
  avg_transaction_size_sol FLOAT4,  -- average SOL per swap in 15m window
  sequence_b64            TEXT,     -- base64-encoded numpy compressed price+volume series
  has_sequence            BOOLEAN DEFAULT FALSE,

  -- =======================================================================
  -- SMART MONEY (12 features, source: Axiom)
  -- =======================================================================
  smart_wallet_buyers_1m            INT4,
  smart_wallet_buyers_5m            INT4,
  smart_wallet_buyers_15m           INT4,
  smart_wallet_volume_1m            FLOAT4,
  smart_wallet_volume_5m            FLOAT4,
  smart_wallet_volume_15m           FLOAT4,
  smart_wallet_percentage           FLOAT4,
  smart_money_first_buyer           SMALLINT DEFAULT 0,
  first_smart_money_buy_timestamp   BIGINT,
  smart_money_within_first_minute   INT4,
  smart_money_within_first_5m       INT4,
  smart_money_accumulation_rate     FLOAT4,

  -- =======================================================================
  -- WALLET QUALITY (10 features, source: Axiom)
  -- =======================================================================
  avg_wallet_age_days               FLOAT4,
  median_wallet_age_days            FLOAT4,
  avg_wallet_trade_count            FLOAT4,
  median_wallet_trade_count         FLOAT4,
  avg_wallet_win_rate               FLOAT4,
  median_wallet_win_rate            FLOAT4,
  avg_wallet_realized_pnl           FLOAT4,
  median_wallet_realized_pnl        FLOAT4,
  avg_wallet_roi                    FLOAT4,
  median_wallet_roi                 FLOAT4,

  -- =======================================================================
  -- PNL (10 features, source: Axiom)
  -- =======================================================================
  avg_buyer_pnl_30d                 FLOAT4,
  median_buyer_pnl_30d              FLOAT4,
  top_buyer_pnl_30d                 FLOAT4,
  avg_buyer_pnl_90d                 FLOAT4,
  median_buyer_pnl_90d              FLOAT4,
  top_buyer_pnl_90d                 FLOAT4,
  avg_seller_pnl_30d                FLOAT4,
  median_seller_pnl_30d             FLOAT4,
  avg_seller_pnl_90d                FLOAT4,
  median_seller_pnl_90d             FLOAT4,

  -- =======================================================================
  -- WHALE AXIOM — $5,000 threshold (8 features, source: Axiom)
  -- =======================================================================
  largest_buy_usd_5k                FLOAT4,
  largest_sell_usd_5k               FLOAT4,
  whale_buy_count_5k                INT4,
  whale_sell_count_5k               INT4,
  whale_buy_volume_5k               FLOAT4,
  whale_sell_volume_5k              FLOAT4,
  whale_net_flow_5k                 FLOAT4,
  whale_accumulation_rate_5k        FLOAT4,

  -- =======================================================================
  -- WHALE AXIOM — $10,000 threshold (8 features, source: Axiom)
  -- =======================================================================
  largest_buy_usd_10k               FLOAT4,
  largest_sell_usd_10k              FLOAT4,
  whale_buy_count_10k               INT4,
  whale_sell_count_10k              INT4,
  whale_buy_volume_10k              FLOAT4,
  whale_sell_volume_10k             FLOAT4,
  whale_net_flow_10k                FLOAT4,
  whale_accumulation_rate_10k       FLOAT4,

  -- =======================================================================
  -- BUYER QUALITY (5 features, source: Axiom)
  -- =======================================================================
  new_wallet_buyers                 INT4,
  experienced_wallet_buyers         INT4,
  wallets_older_than_30_days        INT4,
  wallets_older_than_90_days        INT4,
  wallets_older_than_180_days       INT4,

  -- =======================================================================
  -- CONVICTION SIGNALS (6 features, source: Axiom)
  -- =======================================================================
  repeat_buyers                     INT4,
  multi_buy_wallets                 INT4,
  wallet_rebuy_rate                 FLOAT4,
  wallet_accumulation_rate          FLOAT4,
  avg_buys_per_wallet               FLOAT4,
  median_buys_per_wallet            FLOAT4,

  -- =======================================================================
  -- EARLY STRENGTH (7 features, source: Axiom)
  -- =======================================================================
  first_buyer_win_rate              FLOAT4,
  first_5_buyers_avg_win_rate       FLOAT4,
  first_10_buyers_avg_win_rate      FLOAT4,
  first_20_buyers_avg_win_rate      FLOAT4,
  first_5_buyers_avg_pnl            FLOAT4,
  first_10_buyers_avg_pnl           FLOAT4,
  first_20_buyers_avg_pnl           FLOAT4,

  -- =======================================================================
  -- DISTRIBUTION (5 features, source: Axiom)
  -- =======================================================================
  top_wallet_buy_share              FLOAT4,
  top5_wallet_buy_share             FLOAT4,
  top10_wallet_buy_share            FLOAT4,
  top20_wallet_buy_share            FLOAT4,
  buyer_concentration_index         FLOAT4,

  -- =======================================================================
  -- RISK SIGNALS (6 features, source: Axiom)
  -- =======================================================================
  dumping_wallet_count              INT4,
  wallets_sold_within_5m            INT4,
  wallets_sold_within_15m           INT4,
  wallets_sold_within_60m           INT4,
  fast_exit_rate                    FLOAT4,
  paper_hand_rate                   FLOAT4,

  -- =======================================================================
  -- SMART MONEY vs RETAIL (5 features, source: Axiom)
  -- =======================================================================
  smart_money_volume_share          FLOAT4,
  smart_money_buy_share             FLOAT4,
  retail_buy_share                  FLOAT4,
  retail_sell_share                 FLOAT4,
  smart_money_net_flow              FLOAT4,

  -- =======================================================================
  -- COMPOSITE SCORES (5 features, source: Axiom — engineered)
  -- =======================================================================
  smart_money_score                 FLOAT4,
  wallet_quality_score              FLOAT4,
  whale_score                       FLOAT4,
  conviction_score                  FLOAT4,
  buyer_quality_score               FLOAT4,

  -- =======================================================================
  -- AXIOM METADATA
  -- =======================================================================
  axiom_collected         BOOLEAN DEFAULT FALSE,  -- true if Axiom data collected
  axiom_cost_usd          FLOAT4 DEFAULT 0.0,     -- estimated Axiom API cost for this token

  -- =======================================================================
  -- LABELS (profit-tier targets from DexScreener price_change_24h)
  -- =======================================================================
  did_1_25x              SMALLINT DEFAULT 0,   -- price_change_24h >= 25%
  did_1_5x               SMALLINT DEFAULT 0,   -- price_change_24h >= 50%
  did_2x                 SMALLINT DEFAULT 0,   -- price_change_24h >= 100%
  did_3x                 SMALLINT DEFAULT 0,   -- price_change_24h >= 200%
  did_5x                 SMALLINT DEFAULT 0,   -- price_change_24h >= 400%
  did_10x                SMALLINT DEFAULT 0,   -- price_change_24h >= 900%
  rugged                 SMALLINT DEFAULT 0,   -- price_change_24h <= -80% OR liquidity < $10
  survived_24h           SMALLINT DEFAULT 0,   -- liquidity >= $10 AND pair still alive after 24h
  max_drawdown_pct       FLOAT4,               -- worst observed drawdown
  inferred_label         BOOLEAN DEFAULT FALSE, -- true if labels were computed
  labels_ready            BOOLEAN DEFAULT FALSE, -- TRUE after backfill completes
  time_to_peak_minutes    FLOAT4,               -- minutes from T0 to peak price (best-effort)
  peak_multiplier         FLOAT4,               -- peak_price / entry_price (best-effort)

  -- =======================================================================
  -- METADATA
  -- =======================================================================
  deployer_address       TEXT
);

-- =======================================================================
-- INDEXES
-- =======================================================================
CREATE INDEX IF NOT EXISTS idx_graduation_ts
  ON training_tokens(graduation_timestamp);
CREATE INDEX IF NOT EXISTS idx_did_1_25x
  ON training_tokens(did_1_25x);
CREATE INDEX IF NOT EXISTS idx_did_1_5x
  ON training_tokens(did_1_5x);
CREATE INDEX IF NOT EXISTS idx_did_2x
  ON training_tokens(did_2x);
CREATE INDEX IF NOT EXISTS idx_did_3x
  ON training_tokens(did_3x);
CREATE INDEX IF NOT EXISTS idx_did_5x
  ON training_tokens(did_5x);
CREATE INDEX IF NOT EXISTS idx_did_10x
  ON training_tokens(did_10x);
CREATE INDEX IF NOT EXISTS idx_rugged
  ON training_tokens(rugged);
CREATE INDEX IF NOT EXISTS idx_survived_24h
  ON training_tokens(survived_24h);
CREATE INDEX IF NOT EXISTS idx_collected_at
  ON training_tokens(collected_at);
CREATE INDEX IF NOT EXISTS idx_axiom_collected
  ON training_tokens(axiom_collected);

-- Allow the service role to insert/update (RLS is on by default)
ALTER TABLE training_tokens ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS rls_allow_all ON training_tokens;
CREATE POLICY rls_allow_all ON training_tokens
  FOR ALL
  USING (true)
  WITH CHECK (true);
