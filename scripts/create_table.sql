-- scripts/create_table.sql
-- Supabase table schema for collector_bot training_tokens.
-- Row size: ~522 bytes → 1M rows ≈ 497 MB (under 500 MB free-tier limit).

CREATE TABLE IF NOT EXISTS training_tokens (
  id SERIAL PRIMARY KEY,
  mint TEXT UNIQUE NOT NULL,
  graduation_timestamp BIGINT,
  collected_at TIMESTAMPTZ DEFAULT NOW(),

  -- 14 tabular features (raw on-chain, no Rugcheck)
  mint_authority_active FLOAT4,
  freeze_authority_active FLOAT4,
  mutable_metadata FLOAT4,
  lp_burn_pct FLOAT4,
  initial_liquidity_sol FLOAT4,
  liquidity_concentration FLOAT4,
  dev_hold_pct FLOAT4,
  top10_holder_pct FLOAT4,
  bundle_wallet_count FLOAT4,
  migration_speed_seconds FLOAT4,
  buy_sell_ratio_60s FLOAT4,
  price_velocity_60s FLOAT4,
  unique_buyers_60s FLOAT4,
  avg_transaction_size_sol FLOAT4,

  -- Sequence stored as compressed float16 base64
  -- Shape: [16, 6] float16 → base64 string
  -- Size: 16×6×2 bytes × 4/3 base64 ≈ 256 bytes
  -- vs JSONB ≈ 800 bytes (3x smaller)
  sequence_b64 TEXT,
  has_sequence BOOLEAN DEFAULT FALSE,

  -- Labels
  rug_label SMALLINT,
  time_to_rug_hours FLOAT4,
  max_drawdown_pct FLOAT4,
  pump_2x_label SMALLINT,
  inferred_label BOOLEAN DEFAULT FALSE,

  -- Metadata
  deployer_address TEXT,
  price_usd FLOAT4,
  liquidity_usd FLOAT4,
  volume_24h FLOAT4,
  price_change_24h FLOAT4
);

-- Indexes for fast export and dedup queries
CREATE INDEX IF NOT EXISTS idx_graduation_ts
  ON training_tokens(graduation_timestamp);
CREATE INDEX IF NOT EXISTS idx_rug_label
  ON training_tokens(rug_label);
CREATE INDEX IF NOT EXISTS idx_collected_at
  ON training_tokens(collected_at);
