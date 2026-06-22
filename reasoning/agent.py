"""Phase 8: End-to-end ReasoningAgent — wires all seven phases into one nn.Module.

Forward-pass data flow (training / batch):

    tick_features (B, T, 4)    → ContinuousEncoder.encode_batch  → h_cont  (B, T, d)
                                                               last step  → h_cont  (B, d)
    event_features (B, 2)      → EventEncoder.encode_event (loop)         → h_event (B, d)
    wallet_embs (B, d)         — pre-fetched by caller via WalletGNN
    chain_meta (B, 4)          ─┐
    h_cont, h_event, wallet_embs → MultiChainFusion.fuse               → h_fused (B, d)
    h_fused                    → MoE.forward                           → h_moe   (B, d)
                                                          + {balance_loss, z_loss}
    h_moe                      → OutputHeads.forward                   → outputs dict

Streaming-step data flow (inference):

    tick (1, 4), dt → ContinuousEncoder.encode_step → h_cont (1, d)
    event + dt      → EventEncoder.encode_event     → h_event (1, d)
    wallet_emb (1, d) — caller reads from WalletGNN cache
    then same fusion → MoE → heads path

The WalletGNN and AsyncRAG are held as non-trainable attributes; callers drive
their async lifecycles independently. The KillSwitch is sync and checked
inside step_block().
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import ReasoningConfig
from .event_encoder import EventEncoder, build_event_encoder
from .film_fusion import MultiChainFusion, build_fusion
from .kill_switch import KillSwitch, KillSignal, build_kill_switch
from .moe import MoE, build_moe
from .output_heads import OutputHeads, build_output_heads
from .rag import AsyncRAG, build_rag
from .schema import BlockData, DiscreteEvent
from .ssm import ContinuousEncoder, build_encoder
from .wallet_gnn import WalletGNN


# ── ReasoningAgent ─────────────────────────────────────────────────────────────

class ReasoningAgent(nn.Module):
    """Full reasoning agent: SSM → event/wallet encoders → FiLM fusion → MoE → heads.

    Trainable sub-modules (registered as nn.Module children):
        encoder, event_encoder, fusion, moe, heads

    Non-trainable components (not nn.Module children):
        wallet_gnn (async GNN), rag (async RAG), kill_switch (sync supervisor)
    """

    D_IN_TICKS: int = 4  # ContinuousTick.as_feature_vector() length

    def __init__(self, cfg: ReasoningConfig, device: str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        self._device = device
        d = cfg.ssm.d_model

        # ── Trainable modules ──────────────────────────────────────────────
        self.encoder: ContinuousEncoder = build_encoder(
            cfg.ssm, d_in_features=self.D_IN_TICKS, device=device
        )
        self.event_encoder: EventEncoder = build_event_encoder(
            cfg.event, d_model=d, device=device
        )
        self.fusion: MultiChainFusion = build_fusion(cfg.film, d_model=d, device=device)
        self.moe: MoE = build_moe(cfg.moe, d_model=d, device=device)
        self.heads: OutputHeads = build_output_heads(
            cfg.uncertainty, d_model=d, device=device
        )

        # ── Non-trainable components ───────────────────────────────────────
        self.wallet_gnn = WalletGNN(cfg.wallet)
        self.rag = build_rag(cfg.rag)
        self.kill_switch = build_kill_switch(cfg.kill_switch)

    # ── Training / batch forward ───────────────────────────────────────────────

    def forward(
        self,
        tick_features: torch.Tensor,    # (B, T, 4)
        tick_dts: torch.Tensor,         # (T,)
        event_features: torch.Tensor,   # (B, 2)  — [type_idx_float, log1p_amount]
        event_dts: torch.Tensor,        # (B,)    — dt since last event per sample
        wallet_embs: torch.Tensor,      # (B, d_model) — pre-fetched from WalletGNN
        chain_meta: torch.Tensor,       # (B, 4)  — ChainConfig.as_metadata_vector()
    ) -> dict[str, torch.Tensor]:
        """Batch forward pass. Returns all output keys plus MoE aux losses."""
        B = tick_features.shape[0]

        # 1. Continuous encoding → last-step summary (B, d)
        h_cont_seq = self.encoder.encode_batch(tick_features, tick_dts)
        h_cont = h_cont_seq[:, -1, :]                       # (B, d)

        # 2. Event encoding — loop is correct: encode_event operates on (1, 2) inputs
        h_event_list = []
        for b in range(B):
            h_b = self.event_encoder.encode_event(
                event_features[b : b + 1], float(event_dts[b].item())
            )                                               # (1, d)
            h_event_list.append(h_b)
        h_event = torch.cat(h_event_list, dim=0)            # (B, d)

        # 3. FiLM-modulated fusion
        h_fused = self.fusion.fuse(h_cont, h_event, wallet_embs, chain_meta)  # (B, d)

        # 4. MoE routing
        h_moe, aux = self.moe(h_fused)                      # (B, d), dict

        # 5. Output heads
        out = self.heads(h_moe)
        out.update(aux)
        return out

    # ── Streaming / inference step ─────────────────────────────────────────────

    def step_block(
        self,
        tick: torch.Tensor,             # (1, 4)  — latest tick features
        dt: float,                      # seconds since last tick
        event_features: torch.Tensor,   # (1, 2)
        event_dt: float,
        wallet_emb: torch.Tensor,       # (1, d_model)
        chain_meta: torch.Tensor,       # (1, 4)
    ) -> dict:
        """Single-block streaming inference. Updates kill switch state.

        Returns: output dict (same keys as forward) + 'kill_signals' list.
        Does NOT update RAG — caller manages add() / query_async().
        """
        with torch.no_grad():
            h_cont = self.encoder.encode_step(tick, dt)     # (1, d)
            h_event = self.event_encoder.encode_event(event_features, event_dt)  # (1, d)
            h_fused = self.fusion.fuse(h_cont, h_event, wallet_emb, chain_meta)
            h_moe, aux = self.moe(h_fused)
            out = self.heads(h_moe)
            out.update(aux)

        # Kill-switch check using OOD + epistemic from this block
        ood_val = float(out["ood_score"].mean())
        epi_val = float(out["epistemic_var"].mean())
        signals = self.kill_switch.check_block(
            ood_score=ood_val, epistemic_var=epi_val
        )
        out["kill_signals"] = signals
        return out

    # ── Parameter groups (useful for optimizers) ──────────────────────────────

    def trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters from the five trainable sub-modules."""
        return list(self.parameters())


# ── Factory ───────────────────────────────────────────────────────────────────

def build_agent(cfg: ReasoningConfig, device: str = "cpu") -> ReasoningAgent:
    return ReasoningAgent(cfg, device=device)
