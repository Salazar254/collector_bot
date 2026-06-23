"""
ml/xgb_model.py - XGBoost model supporting classification or soft-return regression.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("ml.xgb_model")


class XGBModel:
    def __init__(self, params: Optional[Dict[str, Any]] = None, task: str = "regression"):
        import xgboost as xgb

        self.task = task
        self.default_params = {
            "objective": "reg:squarederror" if task == "regression" else "binary:logistic",
            "eval_metric": "rmse" if task == "regression" else "auc",
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 250,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbosity": 0,
        }
        if params:
            self.default_params.update(params)

        estimator_cls = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
        self.model = estimator_cls(**self.default_params)
        self.feature_importance_: Optional[np.ndarray] = None
        self.history: Dict[str, Any] = {}

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> Dict[str, Any]:
        fit_params = {}
        if X_val is not None and len(X_val) > 0:
            fit_params["eval_set"] = [(X_train, y_train), (X_val, y_val)]
        self.model.fit(X_train, y_train, **fit_params)
        self.feature_importance_ = getattr(self.model, "feature_importances_", None)

        train_preds = self.predict_batch(X_train)
        self.history["train_metric"] = self._metric(y_train, train_preds)
        if X_val is not None and len(X_val) > 0:
            val_preds = self.predict_batch(X_val)
            self.history["val_metric"] = self._metric(y_val, val_preds)
        return self.history

    def _metric(self, y_true: np.ndarray, preds: np.ndarray) -> float:
        if self.task == "regression":
            return float(np.sqrt(np.mean((preds - y_true) ** 2)))
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, preds)) if len(np.unique(y_true)) > 1 else 0.5

    def predict_score(self, features: np.ndarray) -> float:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        value = float(self.predict_batch(features)[0])
        if self.task == "regression":
            return float(1.0 / (1.0 + np.exp(-value * 2.0)))
        return value

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        if self.feature_importance_ is None:
            return {}
        if feature_names is None:
            from ml.features import FEATURE_NAMES
            feature_names = FEATURE_NAMES
        return dict(zip(feature_names, self.feature_importance_.tolist()))

    def feature_importance(self, feature_names: list[str]) -> dict:
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))

    def export_onnx(self, output_path: str, n_features: int = 13) -> None:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType

        initial_types = [
            ("tabular_input", OnnxFloatTensorType([None, n_features]))
        ]
        onnx_model = convert_xgboost(
            self.model,
            initial_types=initial_types,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as handle:
            handle.write(onnx_model.SerializeToString())
        print(f"XGBoost ONNX exported to {output_path}")

    def validate_onnx(self, onnx_path: str, n_features: int = 13) -> bool:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path)
        test_input = np.random.rand(1, n_features).astype(np.float32)
        result = sess.run(None, {"tabular_input": test_input})
        prob = float(result[0][0][1]) if len(result[0].shape) > 1 else float(result[1][0][1])
        assert 0 <= prob <= 1, f"XGB ONNX output out of range: {prob}"
        print(f"XGB ONNX validation passed. Sample rug_prob: {prob:.4f}")
        return True

    def save(self, path: str = "ml/saved_models/xgb_model.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        meta_path = path.replace(".json", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump({"history": self.history, "params": self.default_params, "task": self.task}, handle, indent=2)

    @classmethod
    def load(cls, path: str = "ml/saved_models/xgb_model.json") -> "XGBModel":
        import xgboost as xgb

        if not os.path.exists(path):
            raise FileNotFoundError(path)
        meta_path = path.replace(".json", "_meta.json")
        task = "regression"
        params: Dict[str, Any] = {}
        history: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            task = meta.get("task", task)
            params = meta.get("params", {})
            history = meta.get("history", {})
        model = cls(params=params, task=task)
        model.model.load_model(path)
        model.history = history
        return model
