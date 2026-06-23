#!/usr/bin/env python3
"""Validate hybrid XGBoost + PyTorch ONNX rug models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NOISE_FEATURES = {"hour_of_day", "is_weekend"}
EXPECTED_FEATURES = 14
EXPECTED_SEQUENCE_LENGTH = 16
MIN_ENSEMBLE_AUC = 0.65
XGB_WEIGHT = 0.4
TORCH_WEIGHT = 0.6


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"[{status}] {label}{suffix}")
    return passed


def load_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb-model", default="models/xgb_model.onnx")
    parser.add_argument("--rug-model", default="models/rug_model.onnx")
    parser.add_argument("--meta", default="models/rug_model_meta.json")
    parser.add_argument("--data", default="ml/data/raw_data.parquet")
    args = parser.parse_args()

    results: list[bool] = []

    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        print("[FAIL] onnxruntime import")
        return 1

    xgb_path = Path(args.xgb_model)
    rug_path = Path(args.rug_model)
    meta_path = Path(args.meta)
    feature_importance_path = Path("ml/data/feature_importance.json")

    xgb_sess = None
    rug_sess = None

    try:
        xgb_sess = ort.InferenceSession(str(xgb_path))
        results.append(check("xgb_model.onnx loads in onnxruntime", True))
    except Exception as exc:
        results.append(check("xgb_model.onnx loads in onnxruntime", False, str(exc)))

    if xgb_sess is not None:
        try:
            test_tab = np.random.rand(1, EXPECTED_FEATURES).astype(np.float32)
            xgb_out = xgb_sess.run(None, {"tabular_input": test_tab})
            prob = float(xgb_out[1][0][1])
            results.append(check("xgb input [1,14] accepted, output in [0,1]", 0 <= prob <= 1, f"prob={prob:.4f}"))
        except Exception as exc:
            results.append(check("xgb input [1,14] accepted, output in [0,1]", False, str(exc)))

    try:
        rug_sess = ort.InferenceSession(str(rug_path))
        results.append(check("rug_model.onnx loads in onnxruntime", True))
    except Exception as exc:
        results.append(check("rug_model.onnx loads in onnxruntime", False, str(exc)))

    xgb_prob = None
    torch_prob = None
    if rug_sess is not None:
        try:
            tabular = np.random.rand(1, EXPECTED_FEATURES).astype(np.float32)
            sequence = np.random.rand(1, EXPECTED_SEQUENCE_LENGTH, 6).astype(np.float32)
            deployer = np.array([0], dtype=np.int64)
            outputs = rug_sess.run(
                None,
                {"tabular": tabular, "sequence": sequence, "deployer_id": deployer},
            )
            results.append(check("rug tabular [1,14] accepted", True))
            results.append(check("rug sequence [1,16,6] accepted", True))
            results.append(check("rug deployer [1] accepted", True))

            names = [item.name for item in rug_sess.get_outputs()]
            output_map = {name: value for name, value in zip(names, outputs)}
            rug_prob = float(np.asarray(output_map.get("rug_prob", outputs[0])).reshape(-1)[0])
            time_to_rug = float(np.asarray(output_map.get("time_to_rug_hours", output_map.get("time_to_rug", outputs[1]))).reshape(-1)[0])
            max_drawdown = float(np.asarray(output_map.get("max_drawdown_pct", output_map.get("max_drawdown", outputs[2]))).reshape(-1)[0])
            pump_prob = float(np.asarray(output_map.get("pump_2x_prob", output_map.get("pump_2x", outputs[3]))).reshape(-1)[0])
            torch_prob = rug_prob

            valid_outputs = (
                0 <= rug_prob <= 1
                and 0 <= time_to_rug <= 72
                and 0 <= max_drawdown <= 100
                and 0 <= pump_prob <= 1
            )
            results.append(
                check(
                    "rug outputs: 4 values all in valid ranges",
                    valid_outputs,
                    f"rug={rug_prob:.4f}, time={time_to_rug:.2f}, dd={max_drawdown:.2f}, pump={pump_prob:.4f}",
                )
            )
        except Exception as exc:
            results.append(check("rug tabular [1,14] accepted", False, str(exc)))
            results.append(check("rug sequence [1,16,6] accepted", False, str(exc)))
            results.append(check("rug deployer [1] accepted", False, str(exc)))
            results.append(check("rug outputs: 4 values all in valid ranges", False, str(exc)))

    if xgb_sess is not None and torch_prob is not None:
        try:
            test_tab = np.random.rand(1, EXPECTED_FEATURES).astype(np.float32)
            xgb_out = xgb_sess.run(None, {"tabular_input": test_tab})
            xgb_prob = float(xgb_out[1][0][1])
            ensemble = (xgb_prob * XGB_WEIGHT) + (torch_prob * TORCH_WEIGHT)
            results.append(check("ensemble = (xgb*0.4)+(torch*0.6) in [0,1]", 0 <= ensemble <= 1, f"ensemble={ensemble:.4f}"))
        except Exception as exc:
            results.append(check("ensemble = (xgb*0.4)+(torch*0.6) in [0,1]", False, str(exc)))

    if meta_path.exists():
        meta = load_meta(meta_path)
        ensemble_auc = float(meta.get("ensemble_test_auc", meta.get("metrics", {}).get("test", {}).get("auc", 0)))
        results.append(check("meta ensemble_test_auc >= 0.65", ensemble_auc >= MIN_ENSEMBLE_AUC, f"auc={ensemble_auc:.4f}"))
        seq_len = int(meta.get("sequence_length", EXPECTED_SEQUENCE_LENGTH))
        tabular_features = int(meta.get("tabular_features", len(meta.get("feature_names", []))))
        results.append(check("meta sequence_length == 16", seq_len == EXPECTED_SEQUENCE_LENGTH, f"got={seq_len}"))
        results.append(check("meta tabular_features == 14", tabular_features == EXPECTED_FEATURES, f"got={tabular_features}"))
    else:
        results.extend([
            check("meta ensemble_test_auc >= 0.65", False, "meta file missing"),
            check("meta sequence_length == 16", False, "meta file missing"),
            check("meta tabular_features == 14", False, "meta file missing"),
        ])

    if feature_importance_path.exists():
        payload = load_meta(feature_importance_path)
        top_features = payload.get("top_features", [])
        results.append(check("feature_importance.json exists and has 14 entries", len(top_features) == EXPECTED_FEATURES, f"count={len(top_features)}"))
        if top_features:
            top_name = str(top_features[0].get("feature", ""))
            intuitive = top_name not in NOISE_FEATURES
            if not intuitive:
                print(f"[WARN] top feature '{top_name}' may be noise for rug detection")
            results.append(check("top feature makes intuitive sense", intuitive, f"top={top_name}"))
        else:
            results.append(check("top feature makes intuitive sense", False, "no features listed"))
    else:
        results.append(check("feature_importance.json exists and has 14 entries", False, "file missing"))
        results.append(check("top feature makes intuitive sense", False, "file missing"))

    if Path(args.data).exists():
        print(f"[INFO] data file present: {args.data}")
    else:
        print(f"[WARN] data file missing: {args.data}")

    passed = all(results)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
