"""Pydantic config system for the reasoning layer."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SSMConfig(BaseModel):
    d_model: int = 128
    d_state: int = 16
    n_layers: int = 2
    d_conv: int = 4
    expand: int = 2  # inner-dimension expansion factor


class EventEncoderConfig(BaseModel):
    n_event_types: int = 5   # whale_buy, whale_sell, lp_add, lp_remove, rug_pull
    d_event_emb: int = 16    # embedding dim for event type
    n_mlp_layers: int = 2    # must be >= 1


class WalletGNNConfig(BaseModel):
    node_feature_dim: int = 8
    d_hidden: int = 64
    n_gnn_layers: int = Field(default=2, ge=1)
    max_nodes: int = 200
    executor_workers: int = 1


class MoEConfig(BaseModel):
    n_experts: int = 6
    top_k: int = 2
    balance_loss_coeff: float = 0.01   # Switch-style load balancing
    z_loss_coeff: float = 0.001        # Router z-loss
    noise_std: float = 1.0             # Noisy top-k gating noise


class FiLMConfig(BaseModel):
    """Per-chain FiLM modulation config."""
    # chain metadata: [block_time_sec, mev_intensity, gas_mechanic_enc, finality_blocks]
    metadata_dim: int = 4
    hidden_dim: int = 64


class UncertaintyConfig(BaseModel):
    n_mc_passes: int = 5
    dropout_rate: float = 0.1
    n_regimes: int = 6
    mahalanobis_feature_dim: int = 128


class RAGConfig(BaseModel):
    embedding_dim: int = 256
    n_neighbors: int = 16
    refresh_every_n_blocks: int = 3
    index_type: Literal["hnsw", "ivf"] = "hnsw"


class TrainingConfig(BaseModel):
    train_month_range: tuple[int, int] = (1, 18)
    val_month_range: tuple[int, int] = (19, 21)
    test_month_range: tuple[int, int] = (22, 24)
    # PGD adversarial input perturbation
    pgd_epsilon: float = 0.01
    pgd_alpha: float = 0.002
    pgd_steps: int = 5
    pgd_loss_coeff: float = 0.1      # β
    chain_adv_loss_coeff: float = 0.01  # γ
    rug_oversample_ratio: float = 5.0


class KillSwitchConfig(BaseModel):
    ood_threshold: float = 3.0
    epistemic_threshold: float = 0.5
    max_drawdown_frac: float = 0.15
    rug_rate_threshold: float = 0.5
    lp_depth_drop_threshold: float = 0.30  # fraction drop in 1 block


class ChainConfig(BaseModel):
    chain_id: str
    block_time_sec: float
    mev_intensity: float = Field(ge=0.0, le=1.0)
    gas_mechanic: Literal["eip1559", "priority_fee", "none"]
    finality_blocks: int

    def as_metadata_vector(self) -> list[float]:
        """Encode chain metadata as a float vector for FiLM conditioning."""
        gas_enc = {"eip1559": 0.0, "priority_fee": 1.0, "none": 2.0}[self.gas_mechanic]
        return [
            self.block_time_sec,
            self.mev_intensity,
            gas_enc,
            float(self.finality_blocks),
        ]


_DEFAULT_CHAINS: list[ChainConfig] = [
    ChainConfig(chain_id="solana",   block_time_sec=0.4,  mev_intensity=0.3,
                gas_mechanic="priority_fee", finality_blocks=32),
    ChainConfig(chain_id="base",     block_time_sec=2.0,  mev_intensity=0.5,
                gas_mechanic="eip1559",      finality_blocks=1),
    ChainConfig(chain_id="bsc",      block_time_sec=3.0,  mev_intensity=0.8,
                gas_mechanic="priority_fee", finality_blocks=15),
    ChainConfig(chain_id="ethereum", block_time_sec=12.0, mev_intensity=0.9,
                gas_mechanic="eip1559",      finality_blocks=64),
]


class ReasoningConfig(BaseModel):
    ssm: SSMConfig = Field(default_factory=SSMConfig)
    event: EventEncoderConfig = Field(default_factory=EventEncoderConfig)
    wallet: WalletGNNConfig = Field(default_factory=WalletGNNConfig)
    moe: MoEConfig = Field(default_factory=MoEConfig)
    film: FiLMConfig = Field(default_factory=FiLMConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    chains: list[ChainConfig] = Field(default_factory=lambda: list(_DEFAULT_CHAINS))
    max_inference_ms: float = 100.0
    device: str = "cuda"  # "cuda" | "cpu"

    def chain_by_id(self, chain_id: str) -> ChainConfig:
        for c in self.chains:
            if c.chain_id == chain_id:
                return c
        raise KeyError(f"Unknown chain_id: {chain_id!r}")
