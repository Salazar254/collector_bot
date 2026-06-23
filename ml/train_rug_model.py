"""Train the production rug-risk neural net and export ONNX.

This pipeline intentionally avoids the leaked aggregate rugcheck score. It
uses raw safety signals, strict temporal splits, time-series cross validation,
label-noise injection, permutation importance leakage checks, and multi-task
heads for both risk and trade management.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI envs.
    raise SystemExit(
        "The PyTorch/ONNX training stack is not installed. "
        "Run `python -m pip install -r ml/requirements.txt` first."
    ) from exc

from sequence_encoder import SEQUENCE_LENGTH, SequenceEncoder
from deployer_encoder import EMBEDDING_DIM
from xgb_model import XGBModel


FEATURE_NAMES = [
    "rugPullRisk",
    "honeypotRisk",
    "lpBurnGap",
    "transferTaxPct",
    "topHolderPct",
    "devHoldPct",
    "mutableMetadata",
    "mintAuthority",
    "freezeAuthority",
    "volatility1m",
    "lowLiquidity",
    "lowBuyers",
    "rugcheckLpUnlocked",
    "rugcheckDangerSignals",
]

TRAIN_END = pd.Timestamp("2024-10-01T00:00:00Z")
VAL_END = pd.Timestamp("2025-01-01T00:00:00Z")

PARQUET_TRAIN_START = pd.Timestamp("2025-01-01T00:00:00Z")
PARQUET_TRAIN_END = pd.Timestamp("2025-09-30T23:59:59Z")
PARQUET_VAL_START = pd.Timestamp("2025-10-01T00:00:00Z")
PARQUET_VAL_END = pd.Timestamp("2025-12-31T23:59:59Z")
PARQUET_TEST_START = pd.Timestamp("2026-01-01T00:00:00Z")

XGB_ENSEMBLE_WEIGHT = 0.4
TORCH_ENSEMBLE_WEIGHT = 0.6
GITHUB_REPO = "https://github.com/Salazar254/meme_coin"


@dataclass
class Example:
    timestamp: pd.Timestamp
    deployer: str
    features: list[float]
    sequence: list[list[float]]
    rug_label: float
    time_to_rug_hours: float
    max_drawdown_pct: float
    pump_2x_label: float
    weight: float = 1.0


class MCDropout(nn.Module):
    """Dropout that stays active after ONNX export for uncertainty sampling."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.force_dropout = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.dropout(value, p=self.p, training=self.training or self.force_dropout)


class RugRiskNet(nn.Module):
    def __init__(
        self,
        tabular_dim: int = len(FEATURE_NAMES),
        num_deployers: int = 1,
        deployer_dim: int = EMBEDDING_DIM,
        sequence_dim: int = 16,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ):
        super().__init__()
        input_dim = tabular_dim + deployer_dim + sequence_dim
        self.register_buffer("feature_mean", torch.tensor(feature_mean if feature_mean is not None else np.zeros(tabular_dim), dtype=torch.float32))
        self.register_buffer("feature_std", torch.tensor(feature_std if feature_std is not None else np.ones(tabular_dim), dtype=torch.float32))
        self.deployer_embedding = nn.Embedding(num_deployers, deployer_dim)
        self.sequence_encoder = SequenceEncoder(input_dim=6, hidden_dim=sequence_dim, num_layers=2)

        self.input_skip = nn.Linear(input_dim, 128)
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            MCDropout(0.3),
        )
        self.skip2 = nn.Linear(128, 64)
        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            MCDropout(0.2),
        )
        self.block3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.rug_head = nn.Linear(32, 1)
        self.time_head = nn.Linear(32, 1)
        self.drawdown_head = nn.Linear(32, 1)
        self.pump_head = nn.Linear(32, 1)

    def forward(
        self,
        tabular: torch.Tensor,
        deployer_id: torch.Tensor,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tabular = (tabular - self.feature_mean) / torch.clamp(self.feature_std, min=1e-6)
        deployer = self.deployer_embedding(deployer_id.long())
        temporal = self.sequence_encoder(sequence)
        x = torch.cat([tabular, deployer, temporal], dim=1)
        h1 = self.block1(x) + self.input_skip(x)
        h2 = self.block2(h1) + self.skip2(h1)
        z = self.block3(h2)
        rug_prob = torch.sigmoid(self.rug_head(z))
        time_to_rug = torch.clamp(F.relu(self.time_head(z)), max=72.0)
        max_drawdown = torch.sigmoid(self.drawdown_head(z))
        pump_2x_prob = torch.sigmoid(self.pump_head(z))
        return rug_prob, time_to_rug, max_drawdown, pump_2x_prob


def parse_timestamp(value: Any, fallback_year: int | None = None) -> pd.Timestamp:
    if value is None or value == "":
        if fallback_year:
            return pd.Timestamp(f"{fallback_year}-01-01T00:00:00Z")
        return pd.Timestamp("1970-01-01T00:00:00Z")
    if isinstance(value, (int, float)):
        unit = "ms" if value > 10_000_000_000 else "s"
        return pd.to_datetime(value, unit=unit, utc=True, errors="coerce")
    text = str(value).replace(" ", "T")
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp("1970-01-01T00:00:00Z")
    return parsed


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return fallback
        parsed = float(str(value).replace(",", ""))
        return parsed if math.isfinite(parsed) else fallback
    except (TypeError, ValueError):
        return fallback


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def default_sequence(features: dict[str, float]) -> list[list[float]]:
    holders = max(1.0, (1.0 - features["lowBuyers"]) * 750.0)
    liquidity = max(0.05, (1.0 - features["lowLiquidity"]) * 20.0)
    volume = max(0.0, features["volatility1m"] * liquidity * 2.0)
    ratio = max(0.05, 1.0 - features["honeypotRisk"] + features["rugPullRisk"])
    velocity = features["volatility1m"] * (1.0 - features["rugPullRisk"])
    tx_count = max(1.0, holders * max(ratio, 0.05))
    return [[holders, liquidity, volume, ratio, velocity, tx_count] for _ in range(SEQUENCE_LENGTH)]


def load_generic_jsonl(path: str) -> list[Example]:
    rows: list[Example] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if "futureReturnPct" in item:
                raise ValueError(f"future_return_leakage_in_training_file:{path}")
            timestamp = parse_timestamp(item.get("timestamp") or item.get("timestamp_ms") or item.get("created_at"))
            rugcheck = item.get("rugcheck") or {}
            features = {
                "rugPullRisk": clamp(safe_float(item.get("rugPullRisk"))),
                "honeypotRisk": clamp(safe_float(item.get("honeypotRisk"))),
                "lpBurnGap": clamp(1.0 - safe_float(item.get("lpBurnPct"), 1.0)),
                "transferTaxPct": clamp(safe_float(item.get("transferTaxPct"))),
                "topHolderPct": clamp(safe_float(item.get("topHolderPct"))),
                "devHoldPct": clamp(safe_float(item.get("devHoldPct"))),
                "mutableMetadata": 1.0 if item.get("mutableMetadata") else 0.0,
                "mintAuthority": 0.0 if item.get("mintAuthorityRenounced") else 1.0,
                "freezeAuthority": 0.0 if item.get("freezeAuthorityRenounced") else 1.0,
                "volatility1m": clamp(safe_float(item.get("volatility1m"))),
                "lowLiquidity": clamp(1.0 / max(safe_float(item.get("liquiditySol"), 0.05), 0.05) / 5.0),
                "lowBuyers": clamp(1.0 - safe_float(item.get("uniqueBuyers")) / 40.0),
                "rugcheckLpUnlocked": 0.0 if rugcheck.get("lpLocked") else clamp(1.0 - safe_float(rugcheck.get("lpLockedPct"), 100.0) / 100.0),
                "rugcheckDangerSignals": clamp(sum(1 for risk in rugcheck.get("risks", []) if str(risk.get("level", "")).lower() in {"danger", "critical"}) / 4.0),
            }
            sequence = item.get("sequence24h") or default_sequence(features)
            rows.append(Example(
                timestamp=timestamp,
                deployer=str(item.get("deployer") or "unknown"),
                features=[features[name] for name in FEATURE_NAMES],
                sequence=sequence[-SEQUENCE_LENGTH:],
                rug_label=float(item.get("rug_label") or item.get("is_rug") or 0.0),
                time_to_rug_hours=float(item.get("time_to_rug_hours") or 24.0),
                max_drawdown_pct=float(item.get("max_drawdown_pct") or 0.0),
                pump_2x_label=float(item.get("pump_2x") or item.get("pump_2x_label") or 0.0),
                weight=float(item.get("weight") or 1.0),
            ))
    return rows


def load_solrpds_csv(path: str) -> list[Example]:
    rows: list[Example] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            first = parse_timestamp(row.get("FIRST_POOL_ACTIVITY_TIMESTAMP") or row.get("first_pool_activity_timestamp"))
            last = parse_timestamp(row.get("LAST_POOL_ACTIVITY_TIMESTAMP") or row.get("last_pool_activity_timestamp"))
            added = safe_float(row.get("TOTAL_ADDED_LIQUIDITY"))
            removed = safe_float(row.get("TOTAL_REMOVED_LIQUIDITY"))
            add_count = safe_float(row.get("NUM_LIQUIDITY_ADDS"))
            remove_count = safe_float(row.get("NUM_LIQUIDITY_REMOVES"))
            status = str(row.get("INACTIVITY_STATUS") or "").lower()
            removed_share = removed / max(added + removed, 1e-9)
            remove_to_add = removed / max(added, 1e-9)
            remove_frequency = remove_count / max(add_count + remove_count, 1.0)
            activity_hours = max(0.0, (last - first).total_seconds() / 3600.0)
            is_rug = status != "active" or removed_share > 0.74 or remove_to_add > 1.15
            features = {
                "rugPullRisk": clamp(0.12 + clamp(remove_to_add / 2.0) * 0.76),
                "honeypotRisk": clamp((0.18 if status != "active" else 0.03) + remove_frequency * 0.22),
                "lpBurnGap": clamp(removed_share),
                "transferTaxPct": 0.0,
                "topHolderPct": clamp(0.08 + remove_frequency * 0.24),
                "devHoldPct": clamp(remove_to_add * 0.28),
                "mutableMetadata": 0.0,
                "mintAuthority": 0.15,
                "freezeAuthority": 0.15,
                "volatility1m": clamp(remove_frequency + (0.2 if activity_hours < 1.0 else 0.0)),
                "lowLiquidity": clamp(1.0 / max(math.log10(max(added, 10.0)), 1.0)),
                "lowBuyers": clamp(1.0 - add_count / 24.0),
                "rugcheckLpUnlocked": clamp(removed_share),
                "rugcheckDangerSignals": clamp(remove_frequency * 0.45 + clamp(remove_to_add / 2.0) * 0.35),
            }
            rows.append(Example(
                timestamp=first,
                deployer=str(row.get("CREATOR") or row.get("TOKEN_ADDRESS") or f"solrpds_{len(rows) % 50000}"),
                features=[features[name] for name in FEATURE_NAMES],
                sequence=default_sequence(features),
                rug_label=1.0 if is_rug else 0.0,
                time_to_rug_hours=min(activity_hours, 24.0) if is_rug else 24.0,
                max_drawdown_pct=clamp(removed_share, 0.0, 1.0) * 100.0,
                pump_2x_label=1.0 if not is_rug and add_count >= 3 and removed_share < 0.2 else 0.0,
                weight=1.25 if is_rug else 0.85,
            ))
    return rows


def load_pumpstudio_jsonl(path: str) -> list[Example]:
    rows: list[Example] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("validated") is False:
                continue
            timestamp = parse_timestamp(row.get("timestamp") or row.get("snapshot_at"))
            risk_text = str(row.get("risk_level") or "").lower()
            if "critical" in risk_text:
                label, weight = 1.0, 1.25
            elif "high" in risk_text:
                label, weight = 0.86, 1.1
            elif "medium" in risk_text:
                label, weight = 0.48, 0.7
            elif "low" in risk_text:
                label, weight = 0.08, 1.0
            else:
                continue
            factors = str(row.get("risk_factors") or "").lower()
            top10 = clamp(safe_float(row.get("top10_holder_pct")) / 100.0)
            holders = safe_float(row.get("holder_count"))
            liquidity = safe_float(row.get("liquidity") or row.get("dev_liquidity"))
            market_cap = safe_float(row.get("market_cap") or row.get("dev_market_cap"))
            buys = safe_float(row.get("buys_24h"))
            sells = safe_float(row.get("sells_24h"))
            sell_pressure = sells / max(buys + sells, 1.0)
            liquidity_to_mcap = liquidity / max(market_cap, 1.0)
            features = {
                "rugPullRisk": clamp(top10 * 0.32 + sell_pressure * 0.24 + (0.16 if "rapid_sell_off" in factors else 0.0) + (0.12 if "wash" in factors else 0.0)),
                "honeypotRisk": clamp((0.2 if "single_holder" in factors else 0.0) + sell_pressure * 0.2),
                "lpBurnGap": clamp(0.72 if "no_liquidity_lock" in factors else 0.08),
                "transferTaxPct": 0.0,
                "topHolderPct": top10,
                "devHoldPct": clamp(top10 * 0.5 + (0.08 if "new_deployer" in factors else 0.0)),
                "mutableMetadata": 0.18 if "no_website" in factors or "no_social_presence" in factors else 0.0,
                "mintAuthority": 0.28 if "new_deployer" in factors else 0.08,
                "freezeAuthority": 0.08,
                "volatility1m": clamp(safe_float(row.get("volatility_score")) / 100.0),
                "lowLiquidity": clamp(1.0 - liquidity_to_mcap * 2.0),
                "lowBuyers": clamp(1.0 - holders / 750.0),
                "rugcheckLpUnlocked": clamp(0.72 if "no_liquidity_lock" in factors else 0.08),
                "rugcheckDangerSignals": clamp(sum(token in factors for token in ["rapid_sell_off", "wash", "dead_volume", "single_holder"]) / 4.0),
            }
            rows.append(Example(
                timestamp=timestamp,
                deployer=str(row.get("deployer") or row.get("creator") or f"pump_{len(rows) % 50000}"),
                features=[features[name] for name in FEATURE_NAMES],
                sequence=default_sequence(features),
                rug_label=1.0 if label >= 0.5 else 0.0,
                time_to_rug_hours=2.0 + (1.0 - label) * 22.0,
                max_drawdown_pct=label * 82.0,
                pump_2x_label=1.0 if label < 0.3 and safe_float(row.get("buy_pressure")) > 65 else 0.0,
                weight=weight,
            ))
    return rows


def load_examples(data_dir: str) -> list[Example]:
    examples: list[Example] = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            path = os.path.join(root, filename)
            lower = filename.lower()
            if lower.endswith(".jsonl") and "pumpstudio" in root.lower():
                examples.extend(load_pumpstudio_jsonl(path))
            elif lower.endswith(".jsonl"):
                examples.extend(load_generic_jsonl(path))
            elif lower.endswith(".csv") and "solrpds" in root.lower():
                examples.extend(load_solrpds_csv(path))
    return [item for item in examples if item.timestamp.year >= 2021]


def split_temporal(examples: list[Example]) -> tuple[list[Example], list[Example], list[Example]]:
    ordered = sorted(examples, key=lambda item: item.timestamp)
    train = [item for item in ordered if item.timestamp < TRAIN_END]
    val = [item for item in ordered if TRAIN_END <= item.timestamp < VAL_END]
    test = [item for item in ordered if item.timestamp >= VAL_END]
    if train and val and test:
        return train, val, test
    first = int(len(ordered) * 0.7)
    second = int(len(ordered) * 0.85)
    return ordered[:first], ordered[first:second], ordered[second:]


def build_tensors(examples: list[Example], deployer_to_id: dict[str, int]) -> tuple[torch.Tensor, ...]:
    x = torch.tensor([item.features for item in examples], dtype=torch.float32)
    deployer = torch.tensor([deployer_to_id.get(item.deployer, 0) for item in examples], dtype=torch.long)
    seq = torch.tensor([pad_sequence(item.sequence) for item in examples], dtype=torch.float32)
    y_rug = torch.tensor([[item.rug_label] for item in examples], dtype=torch.float32)
    y_time = torch.tensor([[min(item.time_to_rug_hours, 72.0)] for item in examples], dtype=torch.float32)
    y_dd = torch.tensor([[clamp(item.max_drawdown_pct / 100.0)] for item in examples], dtype=torch.float32)
    y_pump = torch.tensor([[item.pump_2x_label] for item in examples], dtype=torch.float32)
    weights = torch.tensor([[item.weight] for item in examples], dtype=torch.float32)
    return x, deployer, seq, y_rug, y_time, y_dd, y_pump, weights


def pad_sequence(sequence: list[list[float]]) -> list[list[float]]:
    clipped = sequence[-SEQUENCE_LENGTH:]
    if len(clipped) < SEQUENCE_LENGTH:
        clipped = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0] for _ in range(SEQUENCE_LENGTH - len(clipped))] + clipped
    return [[safe_float(value) for value in row[:6]] + [0.0] * max(0, 6 - len(row)) for row in clipped]


def inject_label_noise(examples: list[Example], rate: float, seed: int) -> list[Example]:
    rng = random.Random(seed)
    output: list[Example] = []
    for item in examples:
        label = 1.0 - item.rug_label if rng.random() < rate else item.rug_label
        output.append(Example(
            timestamp=item.timestamp,
            deployer=item.deployer,
            features=item.features,
            sequence=item.sequence,
            rug_label=label,
            time_to_rug_hours=item.time_to_rug_hours,
            max_drawdown_pct=item.max_drawdown_pct,
            pump_2x_label=item.pump_2x_label,
            weight=item.weight,
        ))
    return output


def multitask_loss(outputs: tuple[torch.Tensor, ...], targets: tuple[torch.Tensor, ...], weights: torch.Tensor) -> torch.Tensor:
    rug, time_to_rug, drawdown, pump = outputs
    y_rug, y_time, y_dd, y_pump = targets
    bce_rug = F.binary_cross_entropy(rug, y_rug, reduction="none")
    bce_pump = F.binary_cross_entropy(pump, y_pump, reduction="none")
    time_loss = F.smooth_l1_loss(time_to_rug, y_time, reduction="none") / 72.0
    dd_loss = F.smooth_l1_loss(drawdown, y_dd, reduction="none")
    total = 1.8 * bce_rug + 0.35 * time_loss + 0.35 * dd_loss + 0.6 * bce_pump
    return torch.mean(total * weights)


def evaluate_loss(model: RugRiskNet, tensors: tuple[torch.Tensor, ...], batch_size: int = 1024) -> float:
    model.eval()
    losses: list[float] = []
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, deployer, seq, y_rug, y_time, y_dd, y_pump, weights in loader:
            outputs = model(x, deployer, seq)
            losses.append(float(multitask_loss(outputs, (y_rug, y_time, y_dd, y_pump), weights).detach().cpu()))
    return sum(losses) / max(len(losses), 1)


def predict_rug(model: RugRiskNet, tensors: tuple[torch.Tensor, ...], batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, deployer, seq, y_rug, *_ in loader:
            rug, _, _, _ = model(x, deployer, seq)
            preds.append(rug.detach().cpu().numpy().reshape(-1))
            labels.append(y_rug.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(labels)


def auc_score(preds: np.ndarray, labels: np.ndarray) -> float:
    labels = (labels >= 0.5).astype(np.int32)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(preds)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(preds) + 1)
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_metrics(model: RugRiskNet, tensors: tuple[torch.Tensor, ...]) -> dict[str, float]:
    preds, labels = predict_rug(model, tensors)
    clipped = np.clip(preds, 1e-6, 1 - 1e-6)
    hard = (preds >= 0.5).astype(np.float32)
    labels_hard = (labels >= 0.5).astype(np.float32)
    return {
        "count": float(len(labels)),
        "positive_rate": float(labels_hard.mean()) if len(labels_hard) else 0.0,
        "auc": float(auc_score(preds, labels)),
        "log_loss": float(-(labels_hard * np.log(clipped) + (1 - labels_hard) * np.log(1 - clipped)).mean()),
        "accuracy": float((hard == labels_hard).mean()) if len(labels_hard) else 0.0,
    }


def train_model(
    train: list[Example],
    val: list[Example],
    test: list[Example],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[RugRiskNet, dict[str, Any], dict[str, int]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    deployers = sorted({item.deployer for item in train + val + test})
    deployer_to_id = {deployer: index + 1 for index, deployer in enumerate(deployers)}
    deployer_to_id["__unknown__"] = 0

    train_noisy = inject_label_noise(train, 0.02, seed)
    train_features = np.array([item.features for item in train_noisy], dtype=np.float32)
    feature_mean = train_features.mean(axis=0)
    feature_std = np.clip(train_features.std(axis=0), 1e-4, None)

    model = RugRiskNet(
        num_deployers=len(deployer_to_id),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [param for name, param in model.named_parameters() if not name.startswith("deployer_embedding")], "lr": lr},
            {"params": model.deployer_embedding.parameters(), "lr": lr * 0.1},
        ],
        weight_decay=0.01,
    )

    train_tensors = build_tensors(train_noisy, deployer_to_id)
    val_tensors = build_tensors(val, deployer_to_id)
    loader = DataLoader(TensorDataset(*train_tensors), batch_size=batch_size, shuffle=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for x, deployer, seq, y_rug, y_time, y_dd, y_pump, weights in loader:
            optimizer.zero_grad()
            outputs = model(x, deployer, seq)
            loss = multitask_loss(outputs, (y_rug, y_time, y_dd, y_pump), weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_loss(model, val_tensors) if val else sum(train_losses) / max(len(train_losses), 1)
        train_loss = sum(train_losses) / max(len(train_losses), 1)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break

    if best_state:
        model.load_state_dict(best_state)

    metrics = {
        "history": history,
        "train": binary_metrics(model, build_tensors(train, deployer_to_id)),
        "validation": binary_metrics(model, val_tensors),
        "test": binary_metrics(model, build_tensors(test, deployer_to_id)),
        "best_val_loss": best_val_loss,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
    }
    return model, metrics, deployer_to_id


def permutation_importance(model: RugRiskNet, examples: list[Example], deployer_to_id: dict[str, int], seed: int) -> dict[str, Any]:
    if not examples:
        return {"features": {}, "leakage_flags": []}
    rng = np.random.default_rng(seed)
    base_tensors = build_tensors(examples, deployer_to_id)
    base_loss = evaluate_loss(model, base_tensors)
    raw_scores: dict[str, float] = {}
    for feature_index, name in enumerate(FEATURE_NAMES):
        permuted = [
            Example(
                timestamp=item.timestamp,
                deployer=item.deployer,
                features=list(item.features),
                sequence=item.sequence,
                rug_label=item.rug_label,
                time_to_rug_hours=item.time_to_rug_hours,
                max_drawdown_pct=item.max_drawdown_pct,
                pump_2x_label=item.pump_2x_label,
                weight=item.weight,
            )
            for item in examples
        ]
        column = [item.features[feature_index] for item in permuted]
        rng.shuffle(column)
        for item, value in zip(permuted, column):
            item.features[feature_index] = float(value)
        raw_scores[name] = max(0.0, evaluate_loss(model, build_tensors(permuted, deployer_to_id)) - base_loss)
    total = sum(raw_scores.values()) or 1.0
    normalized = {name: score / total for name, score in raw_scores.items()}
    flags = [name for name, score in normalized.items() if score > 0.35]
    return {"features": normalized, "base_loss": base_loss, "leakage_flags": flags}


def time_series_cv(examples: list[Example], epochs: int, batch_size: int, lr: float, seed: int) -> list[dict[str, float]]:
    ordered = sorted(examples, key=lambda item: item.timestamp)
    if len(ordered) < 100:
        return []
    fold_size = max(1, len(ordered) // 6)
    folds: list[dict[str, float]] = []
    for fold in range(5):
        train_end = fold_size * (fold + 1)
        val_end = min(len(ordered), train_end + fold_size)
        if val_end <= train_end:
            break
        model, metrics, _ = train_model(
            ordered[:train_end],
            ordered[train_end:val_end],
            ordered[val_end:],
            epochs=max(4, min(epochs, 16)),
            batch_size=batch_size,
            lr=lr,
            seed=seed + fold + 100,
        )
        del model
        folds.append({
            "fold": float(fold + 1),
            "train_count": float(train_end),
            "validation_count": float(val_end - train_end),
            "validation_auc": float(metrics["validation"]["auc"]),
            "validation_log_loss": float(metrics["validation"]["log_loss"]),
        })
    return folds


def cap_split(examples: list[Example], max_samples: int, seed: int) -> list[Example]:
    if max_samples <= 0 or len(examples) <= max_samples:
        return examples
    rng = random.Random(seed)
    sampled = rng.sample(examples, max_samples)
    return sorted(sampled, key=lambda item: item.timestamp)


def export_onnx(model: RugRiskNet, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    set_force_dropout(model, True)
    tabular = torch.zeros(1, len(FEATURE_NAMES), dtype=torch.float32)
    deployer = torch.zeros(1, dtype=torch.long)
    sequence = torch.zeros(1, SEQUENCE_LENGTH, 6, dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            (tabular, deployer, sequence),
            output_path,
            input_names=["tabular", "deployer_id", "sequence"],
            output_names=["rug_prob", "time_to_rug_hours", "max_drawdown_pct", "pump_2x_prob"],
            dynamic_axes={
                "tabular": {0: "batch"},
                "deployer_id": {0: "batch"},
                "sequence": {0: "batch"},
                "rug_prob": {0: "batch"},
                "time_to_rug_hours": {0: "batch"},
                "max_drawdown_pct": {0: "batch"},
                "pump_2x_prob": {0: "batch"},
            },
            opset_version=15,
            do_constant_folding=False,
            dynamo=False,
        )
    finally:
        set_force_dropout(model, False)


def set_force_dropout(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, MCDropout):
            module.force_dropout = enabled


def save_outputs(
    model: RugRiskNet,
    metrics: dict[str, Any],
    deployer_to_id: dict[str, int],
    output_path: str,
    checkpoint_path: str,
    meta_path: str,
    importance: dict[str, Any],
    cv: list[dict[str, float]],
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "deployer_to_id": deployer_to_id,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    torch.save(
        {
            "state_dict": {"embedding.weight": model.deployer_embedding.weight.detach().cpu()},
            "deployer_to_id": deployer_to_id,
            "embedding_dim": EMBEDDING_DIM,
            "source": "main_model_low_lr_finetuned_embedding",
        },
        os.path.join(os.path.dirname(output_path), "deployer_embeddings.pt"),
    )
    export_onnx(model, output_path)
    metadata = {
        "version": 1,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "val_auc": metrics["validation"]["auc"],
        "feature_schema_hash": hashlib.sha256(json.dumps(FEATURE_NAMES, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "tabular_residual_mlp_sequence_gru_deployer_embedding_multitask",
        "onnx_opset": 15,
        "feature_names": FEATURE_NAMES,
        "removed_leakage_features": ["rugcheckScore", "futureReturnPct"],
        "temporal_split": {
            "train": "< 2024-10-01",
            "validation": "2024-10-01 <= t < 2025-01-01",
            "test": ">= 2025-01-01",
        },
        "regularization": {
            "dropout": [0.3, 0.2],
            "weight_decay": 0.01,
            "label_noise": 0.02,
        },
        "metrics": metrics,
        "permutation_importance": importance,
        "time_series_cv": cv,
        "leakage_flagged": bool(importance.get("leakage_flags")),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with open(os.path.join(os.path.dirname(output_path), "deployer_lookup.json"), "w", encoding="utf-8") as handle:
        json.dump(deployer_to_id, handle, indent=2)


def parquet_row_to_example(row: pd.Series) -> Example:
    timestamp = parse_timestamp(row.get("decision_timestamp") or row.get("timestamp"))
    tabular = row.get("tabular")
    if isinstance(tabular, str):
        tabular = json.loads(tabular)
    if tabular is None:
        tabular = [safe_float(row.get(name)) for name in FEATURE_NAMES]
    sequence = row.get("sequence")
    if isinstance(sequence, str):
        sequence = json.loads(sequence)
    if sequence is None:
        features = dict(zip(FEATURE_NAMES, tabular))
        sequence = default_sequence(features)
    return Example(
        timestamp=timestamp,
        deployer=str(row.get("deployer") or "unknown"),
        features=[safe_float(value) for value in tabular[: len(FEATURE_NAMES)]],
        sequence=sequence,
        rug_label=float(row.get("rug_label") or 0.0),
        time_to_rug_hours=float(row.get("time_to_rug_hours") or 24.0),
        max_drawdown_pct=float(row.get("max_drawdown_pct") or 0.0),
        pump_2x_label=float(row.get("pump_2x_label") or row.get("pump_2x") or 0.0),
        weight=float(row.get("weight") or 1.0),
    )


def load_parquet_examples(path: str) -> list[Example]:
    frame = pd.read_parquet(path)
    return [parquet_row_to_example(row) for _, row in frame.iterrows()]


def split_temporal_parquet(examples: list[Example]) -> tuple[list[Example], list[Example], list[Example]]:
    train = [item for item in examples if PARQUET_TRAIN_START <= item.timestamp <= PARQUET_TRAIN_END]
    val = [item for item in examples if PARQUET_VAL_START <= item.timestamp <= PARQUET_VAL_END]
    test = [item for item in examples if item.timestamp >= PARQUET_TEST_START]
    if train and val and test:
        return train, val, test
    return split_temporal(examples)


XGB_FEATURE_NAMES = [f for f in FEATURE_NAMES if f != "rugPullRisk"]  # 13 features, no leakage


def train_xgb_classifier(
    train: list[Example],
    val: list[Example],
) -> XGBModel:
    xgb_model = XGBModel(
        task="classification",
        params={
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.4,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "eval_metric": "auc",
            "tree_method": "hist",
        },
    )
    x_train = np.asarray([item.features[1:] for item in train], dtype=np.float32)
    y_train = np.asarray([item.rug_label for item in train], dtype=np.float32)
    x_val = np.asarray([item.features[1:] for item in val], dtype=np.float32)
    y_val = np.asarray([item.rug_label for item in val], dtype=np.float32)
    xgb_model.train_model(x_train, y_train, x_val, y_val)
    return xgb_model


def evaluate_hybrid_ensemble(
    xgb_model: XGBModel,
    torch_model: RugRiskNet,
    test: list[Example],
    deployer_to_id: dict[str, int],
) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    x_test = np.asarray([item.features[1:] for item in test], dtype=np.float32)
    y_test = np.asarray([item.rug_label for item in test], dtype=np.float32)
    xgb_probs = xgb_model.predict_batch(x_test)
    torch_probs, _ = predict_rug(torch_model, build_tensors(test, deployer_to_id))
    ensemble = (xgb_probs * XGB_ENSEMBLE_WEIGHT) + (torch_probs * TORCH_ENSEMBLE_WEIGHT)
    xgb_auc = float(roc_auc_score(y_test, xgb_probs)) if len(np.unique(y_test)) > 1 else 0.5
    torch_auc = float(roc_auc_score(y_test, torch_probs)) if len(np.unique(y_test)) > 1 else 0.5
    ensemble_auc = float(roc_auc_score(y_test, ensemble)) if len(np.unique(y_test)) > 1 else 0.5
    correlation = float(np.corrcoef(xgb_probs, torch_probs)[0, 1]) if len(xgb_probs) > 1 else 0.0
    return {
        "xgb_test_auc": xgb_auc,
        "pytorch_test_auc": torch_auc,
        "ensemble_test_auc": ensemble_auc,
        "xgb_vs_pytorch_correlation": correlation,
    }


def save_hybrid_outputs(
    xgb_model: XGBModel,
    torch_model: RugRiskNet,
    deployer_to_id: dict[str, int],
    metrics: dict[str, Any],
    ensemble_metrics: dict[str, float],
    xgb_out: str,
    rug_out: str,
    meta_out: str,
    deployer_out: str,
    feature_importance_out: str,
    n_train: int,
) -> dict[str, Any]:
    os.makedirs(os.path.dirname(xgb_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(rug_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(meta_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(deployer_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(feature_importance_out) or ".", exist_ok=True)

    xgb_model.export_onnx(xgb_out, n_features=len(XGB_FEATURE_NAMES))
    xgb_model.validate_onnx(xgb_out, n_features=len(XGB_FEATURE_NAMES))
    export_onnx(torch_model, rug_out)

    importances = xgb_model.feature_importance(FEATURE_NAMES)
    trained_at = datetime.now(timezone.utc).isoformat()
    feature_payload = {
        "trained_at": trained_at,
        "top_features": [
            {"feature": name, "importance": round(float(score), 4)}
            for name, score in importances.items()
        ],
        "xgb_vs_pytorch_correlation": round(ensemble_metrics["xgb_vs_pytorch_correlation"], 4),
    }
    with open(feature_importance_out, "w", encoding="utf-8") as handle:
        json.dump(feature_payload, handle, indent=2)

    metadata = {
        "version": 2,
        "trained_at": trained_at,
        "train_period": str(PARQUET_TRAIN_END.date()),
        "val_period": str(PARQUET_VAL_END.date()),
        "test_period": "2026-01-01 to present",
        "xgb_test_auc": round(ensemble_metrics["xgb_test_auc"], 4),
        "pytorch_test_auc": round(ensemble_metrics["pytorch_test_auc"], 4),
        "ensemble_test_auc": round(ensemble_metrics["ensemble_test_auc"], 4),
        "quality_gate_passed": True,
        "sequence_length": SEQUENCE_LENGTH,
        "tabular_features": len(FEATURE_NAMES),
        "xgb_n_estimators": 300,
        "xgb_max_depth": 5,
        "ensemble_weights": {"xgb": XGB_ENSEMBLE_WEIGHT, "pytorch": TORCH_ENSEMBLE_WEIGHT},
        "architecture": "hybrid_xgb_pytorch_rug_risknet",
        "onnx_opset": 15,
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
        "n_train": n_train,
    }
    with open(meta_out, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with open(deployer_out, "w", encoding="utf-8") as handle:
        json.dump(deployer_to_id, handle, indent=2)
    return metadata


def push_to_github(token: str, files: list[str], message: str) -> None:
    if not token:
        print("GITHUB_TOKEN not set — skipping push")
        return
    repo_dest = {
        "models/xgb_model.onnx": "models/xgb_model.onnx",
        "models/rug_model.onnx": "models/rug_model.onnx",
        "models/deployer_lookup.json": "models/deployer_lookup.json",
        "models/rug_model_meta.json": "models/rug_model_meta.json",
        "ml/data/raw_data.parquet": "ml/data/raw_data.parquet",
        "ml/data/feature_importance.json": "ml/data/feature_importance.json",
    }
    local_to_repo = {}
    for path in files:
        normalized = path.replace("\\", "/")
        basename = os.path.basename(normalized)
        for repo_path in repo_dest:
            if normalized.endswith(repo_path) or basename == os.path.basename(repo_path):
                local_to_repo[path] = repo_path
                break
        else:
            if basename in {os.path.basename(item) for item in repo_dest}:
                for repo_path in repo_dest:
                    if os.path.basename(repo_path) == basename:
                        local_to_repo[path] = repo_path
                        break
    if not local_to_repo:
        print("No recognized files to push — skipping")
        return
    try:
        with tempfile.TemporaryDirectory(prefix="retrain_repo_") as temp_dir:
            clone_dir = os.path.join(temp_dir, "repo")
            repo_url = GITHUB_REPO.replace("https://", f"https://{token}@")
            subprocess.run(["git", "clone", repo_url, clone_dir], check=True, capture_output=True)
            for src, dst in local_to_repo.items():
                if not os.path.exists(src):
                    continue
                target = os.path.join(clone_dir, dst)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(src, target)
            subprocess.run(["git", "add", *local_to_repo.values()], cwd=clone_dir, check=True, capture_output=True)
            status = subprocess.run(["git", "status", "--porcelain"], cwd=clone_dir, check=True, capture_output=True, text=True)
            if not status.stdout.strip():
                print("No git changes to push")
                return
            subprocess.run(["git", "config", "user.email", "retrain@bot"], cwd=clone_dir, check=True)
            subprocess.run(["git", "config", "user.name", "Retrain Bot"], cwd=clone_dir, check=True)
            subprocess.run(["git", "commit", "-m", message], cwd=clone_dir, check=True, capture_output=True)
            push = subprocess.run(["git", "push", "origin", "master"], cwd=clone_dir, capture_output=True, text=True)
            if push.returncode != 0:
                push = subprocess.run(["git", "push", "origin", "main"], cwd=clone_dir, capture_output=True, text=True)
            if push.returncode == 0:
                print(f"Pushed to GitHub: {message}")
            else:
                print(f"GitHub push failed: {push.stderr or push.stdout}")
    except Exception as exc:
        print(f"GitHub push warning: {exc}")


def run_hybrid_training(args: argparse.Namespace) -> None:
    examples = load_parquet_examples(args.data)
    if len(examples) < 100:
        raise SystemExit(f"not_enough_training_examples:{len(examples)} in {args.data}")

    train, val, test = split_temporal_parquet(examples)
    if not train or not val or not test:
        raise SystemExit("temporal_split_empty: need train, validation, and test samples")

    print(f"Training XGBoost on {len(train)} train / {len(val)} val samples...")
    xgb_model = train_xgb_classifier(train, val)
    importances = xgb_model.feature_importance(XGB_FEATURE_NAMES)
    print("Top 5 XGB features:")
    for name, score in list(importances.items())[:5]:
        print(f"  {name}: {score:.4f}")

    print(f"Training PyTorch RugRiskNet on {len(train)} train samples...")
    torch_model, metrics, deployer_to_id = train_model(
        train,
        val,
        test,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
    )

    ensemble_metrics = evaluate_hybrid_ensemble(xgb_model, torch_model, test, deployer_to_id)
    ensemble_auc = ensemble_metrics["ensemble_test_auc"]
    print(json.dumps(ensemble_metrics, indent=2))

    if ensemble_auc < args.min_auc:
        print(f"QUALITY GATE FAILED: {ensemble_auc:.4f}")
        print("Existing models NOT overwritten.")
        sys.exit(1)

    metadata = save_hybrid_outputs(
        xgb_model,
        torch_model,
        deployer_to_id,
        metrics,
        ensemble_metrics,
        args.xgb_out,
        args.rug_out,
        args.meta_out,
        args.deployer_out,
        args.feature_importance_out,
        len(train),
    )

    commit_msg = (
        f"retrain: hybrid XGB+PyTorch {datetime.now(timezone.utc).date()} | "
        f"ensemble_auc={ensemble_auc:.4f} | "
        f"n_train={len(train)} | xgb_auc={ensemble_metrics['xgb_test_auc']:.4f} | "
        f"torch_auc={ensemble_metrics['pytorch_test_auc']:.4f}"
    )
    push_files = [
        args.xgb_out,
        args.rug_out,
        args.deployer_out,
        args.meta_out,
        args.data,
        args.feature_importance_out,
    ]
    push_to_github(os.environ.get("GITHUB_TOKEN", ""), push_files, commit_msg)
    print(json.dumps({"metadata": metadata, "ensemble_metrics": ensemble_metrics}, indent=2))


def run_legacy_training(args: argparse.Namespace) -> None:
    examples = load_examples(args.data_dir)
    if len(examples) < 1000:
        raise SystemExit(f"not_enough_training_examples:{len(examples)} in {args.data_dir}")

    train, val, test = split_temporal(examples)
    if not train or not val or not test:
        raise SystemExit("temporal_split_empty: need train, validation, and test samples")
    if args.max_samples_per_split > 0:
        train = cap_split(train, args.max_samples_per_split, args.seed)
        val = cap_split(val, args.max_samples_per_split, args.seed + 1)
        test = cap_split(test, args.max_samples_per_split, args.seed + 2)

    model, metrics, deployer_to_id = train_model(train, val, test, args.epochs, args.batch_size, args.lr, args.seed)
    importance = permutation_importance(model, val or test, deployer_to_id, args.seed + 7)
    cv = [] if args.skip_cv else time_series_cv(train + val, args.epochs, args.batch_size, args.lr, args.seed)
    save_outputs(model, metrics, deployer_to_id, args.output, args.checkpoint, args.meta, importance, cv)
    print(json.dumps({
        "output": args.output,
        "checkpoint": args.checkpoint,
        "meta": args.meta,
        "samples": len(examples),
        "train": len(train),
        "validation": len(val),
        "test": len(test),
        "test_auc": metrics["test"]["auc"],
        "leakage_flags": importance.get("leakage_flags", []),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", help="Parquet training dataset for hybrid pipeline")
    parser.add_argument("--xgb-out", default="models/xgb_model.onnx")
    parser.add_argument("--rug-out", default="models/rug_model.onnx")
    parser.add_argument("--meta-out", default="models/rug_model_meta.json")
    parser.add_argument("--deployer-out", default="models/deployer_lookup.json")
    parser.add_argument("--feature-importance-out", default="ml/data/feature_importance.json")
    parser.add_argument("--min-auc", type=float, default=0.65)
    parser.add_argument("--data-dir", default="data/training")
    parser.add_argument("--output", default="models/rug_model.onnx")
    parser.add_argument("--checkpoint", default="models/rug_model_best.pt")
    parser.add_argument("--meta", default="models/rug_model_meta.json")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    args = parser.parse_args()

    if args.data:
        run_hybrid_training(args)
        return
    run_legacy_training(args)


if __name__ == "__main__":
    main()
