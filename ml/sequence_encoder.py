"""GRU encoder for hourly token state sequences.

The main rug model consumes a 24 step history of knowable market state. Each
step has:

holder_count, liquidity_sol, volume_sol, buy_sell_ratio, price_velocity, tx_count
"""

from __future__ import annotations

import torch
from torch import nn


SEQUENCE_LENGTH = 16
SEQUENCE_FEATURES = [
    "holder_count",
    "liquidity_sol",
    "volume_sol",
    "buy_sell_ratio",
    "price_velocity",
    "tx_count",
]


class SequenceEncoder(nn.Module):
    """Encode {SEQUENCE_LENGTH} hourly snapshots into a compact temporal embedding."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 16, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError(f"sequence tensor must be [batch, {SEQUENCE_LENGTH}, 6]")
        _, hidden = self.gru(sequence)
        return hidden[-1]


def export_onnx(output_path: str = "models/sequence_encoder.onnx") -> None:
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = SequenceEncoder()
    model.eval()
    sample = torch.zeros(1, SEQUENCE_LENGTH, len(SEQUENCE_FEATURES), dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["sequence"],
        output_names=["temporal_embedding"],
        dynamic_axes={"sequence": {0: "batch"}, "temporal_embedding": {0: "batch"}},
        opset_version=15,
    )


if __name__ == "__main__":
    export_onnx()
