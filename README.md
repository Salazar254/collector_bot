# collector_bot

Standalone data collection service for graduated pump.fun tokens. Runs continuously on Render.com free tier, collecting on-chain data from Helius + DexScreener into a Supabase training dataset.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Helius    │────▶│ collect_     │────▶│   Supabase   │
│  (on-chain) │     │ service.py   │     │ (training_   │
│             │     │              │     │  tokens)     │
│ DexScreener │────▶│ Render.com   │     │ 500 MB free  │
│  (prices)   │     │ free tier    │     │ tier         │
└─────────────┘     └──────────────┘     └──────────────┘
```

- **Helius** — graduated mint discovery + DAS asset metadata + swap transactions
- **DexScreener** — batch price/liquidity/volume data (no API key needed)
- **Supabase** — compressed training dataset (float16 base64 sequences, ~522 B/row)

## Quickstart

```bash
pip install -r requirements.txt

export HELIUS_API_KEY=your_key
export SUPABASE_URL=your_url
export SUPABASE_KEY=your_key
export COLLECTION_INTERVAL_SECONDS=30
export PORT=8080

python scripts/collect_service.py
```

## Schema

Run `scripts/create_table.sql` in the Supabase SQL editor to set up the `training_tokens` table.

**Row size**: ~522 bytes → 1M rows ≈ 497 MB (under the 500 MB free-tier limit).

## Throughput

| Source | Batch size | Rate limit | Throughput |
|---|---|---|---|
| Helius getAssetBatch | 100 tokens | 2 req/s | 200 tok/s |
| DexScreener prices | 30 tokens | 5 req/s | 150 tok/s |
| Helius swaps | 1 token | 2 req/s | 2 tok/s |
| **Bottleneck** | — | — | **~2 tok/s** (swap fetch) |

The per-token swap fetch for sequence data is the bottleneck. Without swap data, throughput would be ~150 tok/s (DexScreener-limited), yielding ~500k–1M tokens in 5 days.
