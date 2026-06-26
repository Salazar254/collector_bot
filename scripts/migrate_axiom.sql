-- scripts/migrate_axiom.sql
-- SUPABASE MIGRATION: Add Axiom wallet-intelligence columns + new labels
-- to the existing training_tokens table, plus create Axiom support tables.
-- Run this in the Supabase SQL Editor against your production database.
-- Safe to re-run: all statements use IF NOT EXISTS.

-- =======================================================================
-- 1. NEW LABELS (5 columns — expanded prediction targets)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS did_1_25x              SMALLINT DEFAULT 0;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS did_1_5x               SMALLINT DEFAULT 0;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS did_3x                 SMALLINT DEFAULT 0;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS rugged                 SMALLINT DEFAULT 0;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS survived_24h           SMALLINT DEFAULT 0;

-- =======================================================================
-- 2. SMART MONEY (12 columns, source: Axiom)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_wallet_buyers_1m             INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_wallet_buyers_5m             INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_wallet_buyers_15m            INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_wallet_volume_1m             FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_wallet_volume_5m             FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_wallet_volume_15m            FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_wallet_percentage            FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_money_first_buyer            SMALLINT DEFAULT 0;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS first_smart_money_buy_timestamp    BIGINT;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_money_within_first_minute    INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_money_within_first_5m        INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_money_accumulation_rate      FLOAT4;

-- =======================================================================
-- 3. WALLET QUALITY (10 columns, source: Axiom)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS avg_wallet_age_days              FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS median_wallet_age_days           FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS avg_wallet_trade_count            FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS median_wallet_trade_count         FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS avg_wallet_win_rate                FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS median_wallet_win_rate             FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS avg_wallet_realized_pnl            FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS median_wallet_realized_pnl         FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS avg_wallet_roi                     FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS median_wallet_roi                  FLOAT4;

-- =======================================================================
-- 4. PNL (10 columns, source: Axiom)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS avg_buyer_pnl_30d                FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS median_buyer_pnl_30d             FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS top_buyer_pnl_30d                FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS avg_buyer_pnl_90d                FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS median_buyer_pnl_90d             FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS top_buyer_pnl_90d                FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS avg_seller_pnl_30d               FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS median_seller_pnl_30d            FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS avg_seller_pnl_90d               FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS median_seller_pnl_90d            FLOAT4;

-- =======================================================================
-- 5. WHALE AXIOM — $5,000 threshold (8 columns, source: Axiom)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS largest_buy_usd_5k               FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS largest_sell_usd_5k              FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_buy_count_5k               INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_sell_count_5k              INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_buy_volume_5k              FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_sell_volume_5k             FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_net_flow_5k                FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_accumulation_rate_5k       FLOAT4;

-- =======================================================================
-- 9. WHALE AXIOM — $10,000 threshold (8 columns, source: Axiom)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS largest_buy_usd_10k              FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS largest_sell_usd_10k             FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_buy_count_10k              INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_sell_count_10k             INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_buy_volume_10k             FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_sell_volume_10k            FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_net_flow_10k               FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_accumulation_rate_10k      FLOAT4;

-- =======================================================================
-- 10. BUYER QUALITY (5 columns, source: Axiom)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS new_wallet_buyers                INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS experienced_wallet_buyers        INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS wallets_older_than_30_days       INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS wallets_older_than_90_days       INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS wallets_older_than_180_days      INT4;

-- =======================================================================
-- 11. CONVICTION SIGNALS (6 columns, source: Axiom)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS repeat_buyers                    INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS multi_buy_wallets                INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS wallet_rebuy_rate                FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS wallet_accumulation_rate         FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS avg_buys_per_wallet              FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS median_buys_per_wallet           FLOAT4;

-- =======================================================================
-- 12. EARLY STRENGTH (7 columns, source: Axiom)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS first_buyer_win_rate             FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS first_5_buyers_avg_win_rate      FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS first_10_buyers_avg_win_rate     FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS first_20_buyers_avg_win_rate     FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS first_5_buyers_avg_pnl           FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS first_10_buyers_avg_pnl          FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS first_20_buyers_avg_pnl          FLOAT4;

-- =======================================================================
-- 13. DISTRIBUTION (5 columns, source: Axiom)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS top_wallet_buy_share             FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS top5_wallet_buy_share            FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS top10_wallet_buy_share           FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS top20_wallet_buy_share           FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS buyer_concentration_index        FLOAT4;

-- =======================================================================
-- 14. RISK SIGNALS (6 columns, source: Axiom)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS dumping_wallet_count             INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS wallets_sold_within_5m           INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS wallets_sold_within_15m          INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS wallets_sold_within_60m          INT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS fast_exit_rate                   FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS paper_hand_rate                  FLOAT4;

-- =======================================================================
-- 15. SMART MONEY vs RETAIL (5 columns, source: Axiom)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_money_volume_share         FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_money_buy_share            FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS retail_buy_share                 FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS retail_sell_share                FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_money_net_flow             FLOAT4;

-- =======================================================================
-- 16. COMPOSITE SCORES (5 columns, Axiom — engineered)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS smart_money_score                FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS wallet_quality_score             FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS whale_score                      FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS conviction_score                 FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS buyer_quality_score              FLOAT4;

-- =======================================================================
-- 17. AXIOM METADATA (2 columns)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS axiom_collected                  BOOLEAN DEFAULT FALSE;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS axiom_cost_usd                   FLOAT4 DEFAULT 0.0;

-- =======================================================================
-- 18. SAFETY FEATURES (9 columns, source: Helius DAS + DexScreener)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS mint_authority_active          FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS freeze_authority_active        FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS mutable_metadata               FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS lp_burn_pct                    FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS initial_liquidity_sol          FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS migration_speed_seconds        FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS avg_transaction_size_sol       FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS sequence_b64                   TEXT;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS has_sequence                   BOOLEAN DEFAULT FALSE;

-- =======================================================================
-- 19. BACKFILL LABEL COLUMNS (3 columns)
-- =======================================================================
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS labels_ready                    BOOLEAN DEFAULT FALSE;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS time_to_peak_minutes            FLOAT4;
ALTER TABLE training_tokens ADD COLUMN IF NOT EXISTS peak_multiplier                 FLOAT4;

-- =======================================================================
-- 20. AXIOM SUPPORT TABLES
-- =======================================================================
CREATE TABLE IF NOT EXISTS axiom_raw_responses (
  id BIGSERIAL PRIMARY KEY,
  mint TEXT NOT NULL,
  request_type TEXT NOT NULL,
  response_json JSONB,
  cost_usd FLOAT4 DEFAULT 0.0,
  latency_ms INT4 DEFAULT 0,
  status_code INT2 DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS axiom_feature_health (
  id BIGSERIAL PRIMARY KEY,
  timestamp TIMESTAMPTZ DEFAULT NOW(),
  feature_name TEXT NOT NULL,
  category TEXT NOT NULL,
  missing_pct FLOAT4 DEFAULT 0.0,
  unique_count INT4 DEFAULT 0,
  variance FLOAT4 DEFAULT 0.0,
  flagged SMALLINT DEFAULT 0,
  flag_reason TEXT,
  total_rows_sampled INT4 DEFAULT 0
);

-- =======================================================================
-- 19. NEW INDEXES (safe re-runs via IF NOT EXISTS)
-- =======================================================================
CREATE INDEX IF NOT EXISTS idx_did_1_25x
  ON training_tokens(did_1_25x);
CREATE INDEX IF NOT EXISTS idx_did_1_5x
  ON training_tokens(did_1_5x);
CREATE INDEX IF NOT EXISTS idx_did_3x
  ON training_tokens(did_3x);
CREATE INDEX IF NOT EXISTS idx_did_10x
  ON training_tokens(did_10x);
CREATE INDEX IF NOT EXISTS idx_rugged
  ON training_tokens(rugged);
CREATE INDEX IF NOT EXISTS idx_survived_24h
  ON training_tokens(survived_24h);
CREATE INDEX IF NOT EXISTS idx_axiom_collected
  ON training_tokens(axiom_collected);

-- Axiom support table indexes
CREATE INDEX IF NOT EXISTS idx_axiom_raw_mint
  ON axiom_raw_responses(mint);
CREATE INDEX IF NOT EXISTS idx_axiom_raw_request_type
  ON axiom_raw_responses(request_type);
CREATE INDEX IF NOT EXISTS idx_axiom_raw_created_at
  ON axiom_raw_responses(created_at);
CREATE INDEX IF NOT EXISTS idx_axiom_health_feature
  ON axiom_feature_health(feature_name);
CREATE INDEX IF NOT EXISTS idx_axiom_health_timestamp
  ON axiom_feature_health(timestamp);
CREATE INDEX IF NOT EXISTS idx_axiom_health_flagged
  ON axiom_feature_health(flagged);

-- =======================================================================
-- 20. RLS for Axiom support tables
-- =======================================================================
ALTER TABLE axiom_raw_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE axiom_feature_health ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS rls_allow_all ON axiom_raw_responses;
CREATE POLICY rls_allow_all ON axiom_raw_responses
  FOR ALL
  USING (true)
  WITH CHECK (true);

DROP POLICY IF EXISTS rls_allow_all ON axiom_feature_health;
CREATE POLICY rls_allow_all ON axiom_feature_health
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- =======================================================================
-- VERIFICATION: count new columns (should return 88)
-- =======================================================================
-- SELECT COUNT(*) AS axiom_columns
-- FROM information_schema.columns
-- WHERE table_name = 'training_tokens'
--   AND column_name IN (
--     'did_1_25x','did_1_5x','did_3x','rugged','survived_24h',
--     'smart_wallet_buyers_1m','smart_wallet_buyers_5m','smart_wallet_buyers_15m',
--     'smart_wallet_volume_1m','smart_wallet_volume_5m','smart_wallet_volume_15m',
--     'smart_wallet_percentage','smart_money_first_buyer','first_smart_money_buy_timestamp',
--     'smart_money_within_first_minute','smart_money_within_first_5m','smart_money_accumulation_rate',
--     'avg_wallet_age_days','median_wallet_age_days','avg_wallet_trade_count','median_wallet_trade_count',
--     'avg_wallet_win_rate','median_wallet_win_rate','avg_wallet_realized_pnl','median_wallet_realized_pnl',
--     'avg_wallet_roi','median_wallet_roi',
--     'avg_buyer_pnl_30d','median_buyer_pnl_30d','top_buyer_pnl_30d',
--     'avg_buyer_pnl_90d','median_buyer_pnl_90d','top_buyer_pnl_90d',
--     'avg_seller_pnl_30d','median_seller_pnl_30d','avg_seller_pnl_90d','median_seller_pnl_90d',
--     'avg_buyer_roi_30d','median_buyer_roi_30d','top_buyer_roi_30d',
--     'avg_buyer_roi_90d','median_buyer_roi_90d','top_buyer_roi_90d',
--     'profitable_wallet_count','profitable_wallet_buy_volume','high_roi_wallet_count',
--     'elite_trader_count','wallets_with_positive_pnl',
--     'wallets_above_20pct_roi','wallets_above_50pct_roi','wallets_above_100pct_roi',
--     'largest_buy_usd_1k','largest_sell_usd_1k','whale_buy_count_1k','whale_sell_count_1k',
--     'whale_buy_volume_1k','whale_sell_volume_1k','whale_net_flow_1k','whale_accumulation_rate_1k',
--     'largest_buy_usd_5k','largest_sell_usd_5k','whale_buy_count_5k','whale_sell_count_5k',
--     'whale_buy_volume_5k','whale_sell_volume_5k','whale_net_flow_5k','whale_accumulation_rate_5k',
--     'largest_buy_usd_10k','largest_sell_usd_10k','whale_buy_count_10k','whale_sell_count_10k',
--     'whale_buy_volume_10k','whale_sell_volume_10k','whale_net_flow_10k','whale_accumulation_rate_10k',
--     'new_wallet_buyers','experienced_wallet_buyers',
--     'wallets_older_than_30_days','wallets_older_than_90_days','wallets_older_than_180_days',
--     'repeat_buyers','multi_buy_wallets','wallet_rebuy_rate','wallet_accumulation_rate',
--     'avg_buys_per_wallet','median_buys_per_wallet',
--     'first_buyer_win_rate','first_5_buyers_avg_win_rate','first_10_buyers_avg_win_rate',
--     'first_20_buyers_avg_win_rate','first_5_buyers_avg_pnl','first_10_buyers_avg_pnl','first_20_buyers_avg_pnl',
--     'top_wallet_buy_share','top5_wallet_buy_share','top10_wallet_buy_share','top20_wallet_buy_share',
--     'buyer_concentration_index',
--     'dumping_wallet_count','wallets_sold_within_5m','wallets_sold_within_15m','wallets_sold_within_60m',
--     'fast_exit_rate','paper_hand_rate',
--     'smart_money_volume_share','smart_money_buy_share','retail_buy_share','retail_sell_share',
--     'smart_money_net_flow',
--     'smart_money_score','wallet_quality_score','whale_score','conviction_score','buyer_quality_score',
--     'axiom_collected','axiom_cost_usd'
--   );
