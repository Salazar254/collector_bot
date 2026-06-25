-- scripts/create_axiom_schema.sql
-- Supabase table schema for Axiom API storage.
-- Two tables: raw API response storage + feature health reports.

-- =======================================================================
-- AXIOM RAW RESPONSES — stores raw JSON API responses for cost tracking
-- and debugging. Row size ~2 KB. 1M rows ≈ 1.9 GB.
-- Use retention policy to keep only recent data.
-- =======================================================================

CREATE TABLE IF NOT EXISTS axiom_raw_responses (
  id BIGSERIAL PRIMARY KEY,
  mint TEXT NOT NULL,
  request_type TEXT NOT NULL,       -- e.g. 'wallet_stats', 'smart_money', 'wallet_profiles', 'whale_txns'
  response_json JSONB,              -- raw API response body
  cost_usd FLOAT4 DEFAULT 0.0,     -- estimated cost of this API call
  latency_ms INT4 DEFAULT 0,       -- request latency in milliseconds
  status_code INT2 DEFAULT 0,      -- HTTP status code
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =======================================================================
-- AXIOM FEATURE HEALTH — stores per-cycle feature health reports.
-- Row size ~120 bytes. 10K rows ≈ 1.1 MB.
-- =======================================================================

CREATE TABLE IF NOT EXISTS axiom_feature_health (
  id BIGSERIAL PRIMARY KEY,
  timestamp TIMESTAMPTZ DEFAULT NOW(),
  feature_name TEXT NOT NULL,
  category TEXT NOT NULL,           -- e.g. 'SMART_MONEY', 'WALLET_QUALITY', etc.
  missing_pct FLOAT4 DEFAULT 0.0,  -- percentage of rows with null/zero
  unique_count INT4 DEFAULT 0,     -- number of distinct values
  variance FLOAT4 DEFAULT 0.0,     -- population variance
  flagged SMALLINT DEFAULT 0,      -- 1 if flagged
  flag_reason TEXT,                 -- 'low_uniqueness' | 'zero_variance' | 'high_missing_rate' | null
  total_rows_sampled INT4 DEFAULT 0
);

-- =======================================================================
-- INDEXES
-- =======================================================================

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
-- RLS — Allow service role full access
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
