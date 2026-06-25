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
  -- HOLDERS (5 features, source: Helius DAS + swap inference)
  -- =======================================================================
  holder_count_1m        INT4,     -- distinct token-account holders at snapshot
  holder_count_5m        INT4,
  holder_count_15m       INT4,
  holder_growth_5m       FLOAT4,   -- (holders_5m - holders_1m) / max(holders_1m, 1)
  holder_growth_15m      FLOAT4,   -- (holders_15m - holders_1m) / max(holders_1m, 1)

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
  -- LABELS (profit-tier targets from DexScreener price_change_24h)
  -- =======================================================================
  did_2x                 SMALLINT DEFAULT 0,   -- price_change_24h >= 100%
  did_5x                 SMALLINT DEFAULT 0,   -- price_change_24h >= 400%
  did_10x                SMALLINT DEFAULT 0,   -- price_change_24h >= 900%
  max_drawdown_pct       FLOAT4,               -- worst observed drawdown
  inferred_label         BOOLEAN DEFAULT FALSE, -- true if labels were computed

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
CREATE INDEX IF NOT EXISTS idx_did_2x
  ON training_tokens(did_2x);
CREATE INDEX IF NOT EXISTS idx_did_5x
  ON training_tokens(did_5x);
CREATE INDEX IF NOT EXISTS idx_collected_at
  ON training_tokens(collected_at);

-- Allow the service role to insert/update (RLS is on by default)
ALTER TABLE training_tokens ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS rls_allow_all ON training_tokens;
CREATE POLICY rls_allow_all ON training_tokens
  FOR ALL
  USING (true)
  WITH CHECK (true);
