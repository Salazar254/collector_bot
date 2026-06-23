"""
XGBoost ONNX export smoke test.
Run with: python test_xgb_onnx.py
All checks must print PASS before starting Colab collection.
"""
import sys

print("=" * 50)
print("XGBoost + ONNX Pre-Training Smoke Test")
print("=" * 50)

# CHECK 1 — imports
print("\nCHECK 1: Importing dependencies...")
try:
    import numpy as np
    import xgboost as xgb
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
    import onnxruntime as ort
    print(f"  xgboost:      {xgb.__version__}")
    print(f"  onnxruntime:  {ort.__version__}")
    print("  PASS")
except ImportError as e:
    print(f"  FAIL: {e}")
    print("  Run: pip install xgboost onnxmltools onnxruntime numpy")
    sys.exit(1)

# CHECK 2 — train toy XGBoost
print("\nCHECK 2: Training toy XGBClassifier (14 features)...")
try:
    X = np.random.rand(200, 14).astype(np.float32)
    y = np.random.randint(0, 2, 200)
    clf = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        eval_metric='auc'
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X[:1])[:, 1]
    assert 0 <= proba[0] <= 1
    print(f"  sample rug_prob: {proba[0]:.4f}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# CHECK 3 — export to ONNX via onnxmltools
print("\nCHECK 3: Exporting XGBoost to ONNX via onnxmltools...")
try:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType

    onnx_model = convert_xgboost(
        clf,
        initial_types=[
            ('tabular_input', OnnxFloatTensorType([None, 14]))
        ]
    )
    onnx_path = "test_xgb_smoke.onnx"
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Exported to {onnx_path}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)
# CHECK 4 — load and run ONNX model
print("\nCHECK 4: Loading and running ONNX model...")
try:
    sess = ort.InferenceSession(onnx_path)
    test_input = np.random.rand(1, 14).astype(np.float32)
    result = sess.run(None, {'tabular_input': test_input})

    print(f"  Output count: {len(result)}")
    print(f"  result[0] (class): {result[0]}")
    print(f"  result[1] (proba): {result[1]}")

    # onnxmltools output format: flat array [[p0, p1]]
    # result[1] is a numpy array, NOT a dict
    proba_output = result[1]
    if hasattr(proba_output, '__len__') and not isinstance(proba_output, dict):
        # Array format: [[p0, p1]]
        rug_prob = float(proba_output[0][1])
        print(f"  Format: array — rug_prob={rug_prob:.4f}")
    elif isinstance(proba_output[0], dict):
        # Dict format (skl2onnx legacy): [{0: p0, 1: p1}]
        rug_prob = proba_output[0].get(1, proba_output[0].get('1', 0))
        print(f"  Format: dict — rug_prob={rug_prob:.4f}")
    else:
        rug_prob = float(proba_output)
        print(f"  Format: scalar — rug_prob={rug_prob:.4f}")

    assert 0 <= rug_prob <= 1, f"prob out of range: {rug_prob}"
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# CHECK 5 — ensemble computation
print("\nCHECK 5: Ensemble score computation...")
try:
    xgb_prob = rug_prob
    torch_prob = 0.42  # mock pytorch output
    ensemble = (xgb_prob * 0.4) + (torch_prob * 0.6)
    assert 0 <= ensemble <= 1
    print(f"  xgb_prob:      {xgb_prob:.4f}")
    print(f"  torch_prob:    {torch_prob:.4f}")
    print(f"  ensemble:      {ensemble:.4f}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# CHECK 6 — feature importance
print("\nCHECK 6: Feature importance extraction...")
try:
    FEATURE_NAMES = [
        'rugPullRisk', 'honeypotRisk', 'lpBurnGap',
        'transferTaxPct', 'topHolderPct', 'devHoldPct',
        'mutableMetadata', 'mintAuthority', 'freezeAuthority',
        'volatility1m', 'lowLiquidity', 'lowBuyers',
        'rugcheckLpUnlocked', 'rugcheckDangerSignals'
    ]
    importances = dict(zip(
        FEATURE_NAMES,
        clf.feature_importances_
    ))
    top = sorted(
        importances.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    print("  Top 3 features (toy model — random data):")
    for feat, imp in top:
        print(f"    {feat}: {imp:.4f}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Cleanup
import os
if os.path.exists(onnx_path):
    os.remove(onnx_path)

# Final result
print("\n" + "=" * 50)
print("ALL 6 CHECKS PASSED")
print("XGBoost ONNX pipeline is ready for training")
print("You can now open Google Colab and start Cell 3")
print("=" * 50)
