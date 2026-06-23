#!/bin/bash
set -e

HELIUS_KEY=${1:?"Usage: bash scripts/retrain.sh HELIUS_KEY"}
MIN_AUC=${2:-0.65}

echo "================================================"
echo "Hybrid XGBoost + PyTorch Retrain Pipeline"
echo "Min ensemble AUC required: $MIN_AUC"
echo "Estimated Helius credits: ~880,000 of 1,000,000"
echo "Estimated time: 4-6 hours (mostly data collection)"
echo "================================================"

echo ""
echo "Step 1/4: Collecting training data..."
python scripts/collect_training_data.py \
  --helius-key "$HELIUS_KEY" \
  --target 8000 \
  --output ml/data/raw_data.parquet \
  --checkpoint ml/data/collection_checkpoint.json \
  --start-date 2025-01-01

echo ""
echo "Step 2/4: Training hybrid model..."
python ml/train_rug_model.py \
  --data ml/data/raw_data.parquet \
  --xgb-out models/xgb_model.onnx \
  --rug-out models/rug_model.onnx \
  --meta-out models/rug_model_meta.json \
  --deployer-out models/deployer_lookup.json \
  --feature-importance-out ml/data/feature_importance.json \
  --min-auc "$MIN_AUC"

echo ""
echo "Step 3/4: Validating exported models..."
python scripts/validate_model.py \
  --xgb-model models/xgb_model.onnx \
  --rug-model models/rug_model.onnx \
  --meta models/rug_model_meta.json \
  --data ml/data/raw_data.parquet

echo ""
echo "Step 4/4: Pushing to GitHub..."
git log --oneline -1 -- models/

echo ""
echo "================================================"
echo "Retrain complete. Next steps:"
echo "  1. Set ALLOW_ONNX_RUG_MODEL=true in .env"
echo "  2. Set RUG_MODEL_PATH=models/rug_model.onnx"
echo "  3. Set XGB_MODEL_PATH=models/xgb_model.onnx"
echo "  4. Restart the bot"
echo "  5. Monitor first 100 token scores in logs"
echo "================================================"
