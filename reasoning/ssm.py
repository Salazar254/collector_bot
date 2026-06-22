"""Phase 1: Selective SSM (Mamba-style) continuous encoder.

Architecture per block:
    x (batch, d_model)
    → in_proj  → x_branch (batch, d_inner), z_gate (batch, d_inner)
    → causal depthwise conv1d on x_branch
    → SSM: h_t = Ā·h_{t-1} + B̄·x_t,  y_t = C·h_t + D·x_t
    → gate: out = y · silu(z_gate)
    → out_proj → (batch, d_model)
    + residual from before the block

Δt encoding:
    dt = softplus(dt_proj(x_branch) + dt_bias + log1p(actual_dt))
    A_bar = exp(dt[:,:,None] * A[None,:,:])   (ZOH; A = -exp(A_log), negative definite)
    B_bar = dt[:,:,None] * B[:,None,:]        (simplified ZOH for rank-1 B)

Streaming state per MambaBlock:
    conv_state: (batch, d_inner, d_conv)     — circular buffer
    ssm_state:  (batch, d_inner, d_state)    — recurrent hidden state

Streaming vs batch consistency guarantee:
    encode_batch uses the same block.step() recurrence as encode_step;
    outputs are bit-identical given identical initial states.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SSMConfig


# ── Primitives ────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ── Core Mamba block ─────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Single Mamba-style SSM block with Δt-aware ZOH discretization."""

    def __init__(
        self, d_model: int, d_state: int, d_conv: int, expand: int
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # Input projection splits into (x_branch, z_gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise causal conv — no padding here; handled explicitly
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=0, bias=True,
        )

        # Project x_branch → [dt_proj (d_inner), B (d_state), C (d_state)]
        self.x_proj = nn.Linear(
            self.d_inner, self.d_inner + d_state * 2, bias=False
        )

        # dt bias: per output-channel scalar added before softplus
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))

        # A: log(-A), HiPPO-inspired init: A_n = -n  →  A_log_n = log(n)
        A_log = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
        ).unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(A_log)

        # Skip connection weight
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ── helpers ────────────────────────────────────────────────────────────

    def _A(self) -> torch.Tensor:
        """Continuous A matrix, always negative: (d_inner, d_state)."""
        return -torch.exp(self.A_log)

    def _discretize(
        self,
        dt: torch.Tensor,   # (batch, d_inner) — already includes Δt factor
        B: torch.Tensor,    # (batch, d_state)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ZOH discretization.

        Returns:
            A_bar: (batch, d_inner, d_state)
            B_bar: (batch, d_inner, d_state)
        """
        A = self._A()                       # (d_inner, d_state)
        dt3 = dt.unsqueeze(-1)             # (batch, d_inner, 1)
        A3 = A.unsqueeze(0)                # (1, d_inner, d_state)
        A_bar = torch.exp(dt3 * A3)        # (batch, d_inner, d_state)
        # Simplified rank-1 ZOH for B: B_bar = dt ⊗ B
        B_bar = dt3 * B.unsqueeze(1)       # (batch, d_inner, d_state)
        return A_bar, B_bar

    # ── streaming step ──────────────────────────────────────────────────────

    def step(
        self,
        x: torch.Tensor,             # (batch, d_model)
        conv_state: torch.Tensor,    # (batch, d_inner, d_conv)
        ssm_state: torch.Tensor,     # (batch, d_inner, d_state)
        actual_dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single streaming step.

        Returns:
            output:     (batch, d_model)
            conv_state: (batch, d_inner, d_conv)   new buffer (cat, autograd-safe)
            ssm_state:  (batch, d_inner, d_state)  new state
        """
        xz = self.in_proj(x)                       # (batch, 2*d_inner)
        x_branch, z_gate = xz.chunk(2, dim=-1)    # each (batch, d_inner)

        # Update conv buffer: shift left, append new sample (autograd-safe cat)
        conv_state = torch.cat(
            [conv_state[:, :, 1:], x_branch.unsqueeze(-1)], dim=-1
        )

        # Depthwise conv as dot-product over the buffer window
        # conv1d.weight: (d_inner, 1, d_conv)
        x_conv = (conv_state * self.conv1d.weight[:, 0, :]).sum(-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = F.silu(x_conv)                    # (batch, d_inner)

        # SSM projections
        proj = self.x_proj(x_conv)                 # (batch, d_inner + 2*d_state)
        dt_raw = proj[:, : self.d_inner]
        B = proj[:, self.d_inner: self.d_inner + self.d_state]
        C = proj[:, self.d_inner + self.d_state:]

        # Incorporate actual elapsed time via log1p so all chain block-times
        # are mapped to a sensible, numerically stable range.
        log_dt = math.log1p(actual_dt)
        dt = F.softplus(dt_raw + self.dt_bias + log_dt).clamp(min=1e-3, max=10.0)

        A_bar, B_bar = self._discretize(dt, B)

        # Recurrence: h_new = A_bar * h + B_bar * x_conv
        new_ssm = (
            A_bar * ssm_state
            + B_bar * x_conv.unsqueeze(-1)
        )                                           # (batch, d_inner, d_state)

        # Readout + skip
        y = (new_ssm * C.unsqueeze(1)).sum(-1) + self.D * x_conv  # (batch, d_inner)

        # Gating
        out = y * F.silu(z_gate)                   # (batch, d_inner)
        out = self.out_proj(out)                   # (batch, d_model)

        return out, conv_state, new_ssm

    # ── sequence forward (uses step() for streaming consistency) ────────────

    def forward_sequence(
        self,
        x: torch.Tensor,    # (batch, seq_len, d_model)
        dts: torch.Tensor,  # (seq_len,) — one Δt per time step, shared across batch
    ) -> torch.Tensor:
        """Process a sequence by looping over step().

        Starting state is always zero so this agrees with a reset encode_step loop.
        """
        batch, seq_len, _ = x.shape
        dev, dtype = x.device, x.dtype

        conv_state = torch.zeros(batch, self.d_inner, self.d_conv, device=dev, dtype=dtype)
        ssm_state = torch.zeros(batch, self.d_inner, self.d_state, device=dev, dtype=dtype)

        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            out_t, conv_state, ssm_state = self.step(
                x[:, t, :], conv_state, ssm_state, float(dts[t].item())
            )
            outputs.append(out_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)


# ── Multi-layer encoder ───────────────────────────────────────────────────────

class ContinuousEncoder(nn.Module):
    """2-layer Mamba encoder.  Critical-path interface: encode_step / reset_state.

    Streaming state lives in self._states (list of (conv_state, ssm_state) per layer).
    Call reset_state() between episodes.
    """

    def __init__(self, cfg: SSMConfig, d_in_features: int = 4) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.d_in = d_in_features

        self.input_proj = nn.Linear(d_in_features, cfg.d_model)

        self.norms = nn.ModuleList(
            [RMSNorm(cfg.d_model) for _ in range(cfg.n_layers)]
        )
        self.blocks = nn.ModuleList(
            [
                MambaBlock(cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand)
                for _ in range(cfg.n_layers)
            ]
        )

        # Streaming state: None until first step, then list[(conv, ssm)]
        self._states: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None

    # ── state management ───────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Clear streaming hidden state. Must be called between independent episodes."""
        self._states = None

    def _init_states(self, batch: int, device: torch.device, dtype: torch.dtype) -> None:
        self._states = [
            (
                torch.zeros(batch, blk.d_inner, blk.d_conv, device=device, dtype=dtype),
                torch.zeros(batch, blk.d_inner, blk.d_state, device=device, dtype=dtype),
            )
            for blk in self.blocks
        ]

    # ── critical path ──────────────────────────────────────────────────────

    def encode_step(self, features: torch.Tensor, dt: float) -> torch.Tensor:
        """One streaming update.

        Args:
            features: (batch, d_in_features) tick feature vector
            dt:       elapsed seconds since the previous step
        Returns:
            (batch, d_model) hidden-state output
        """
        if self._states is None:
            self._init_states(features.shape[0], features.device, features.dtype)
        assert self._states is not None

        x = self.input_proj(features)          # (batch, d_model)

        for i, (norm, blk) in enumerate(zip(self.norms, self.blocks)):
            residual = x
            out, new_conv, new_ssm = blk.step(norm(x), self._states[i][0], self._states[i][1], dt)
            x = residual + out
            self._states[i] = (new_conv, new_ssm)

        return x

    # ── batch path (training / streaming consistency check) ────────────────

    def encode_batch(
        self, features: torch.Tensor, dts: torch.Tensor
    ) -> torch.Tensor:
        """Encode a full sequence starting from zero hidden state.

        Does NOT modify self._states — purely functional.

        Args:
            features: (batch, seq_len, d_in_features)
            dts:      (seq_len,) actual Δt values
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.input_proj(features)          # (batch, seq_len, d_model)

        for norm, blk in zip(self.norms, self.blocks):
            residual = x
            out = blk.forward_sequence(norm(x), dts)   # (batch, seq_len, d_model)
            x = residual + out

        return x


# ── Factory ───────────────────────────────────────────────────────────────────

def build_encoder(
    cfg: SSMConfig,
    d_in_features: int = 4,
    device: str = "cpu",
    compile_model: bool = False,
) -> ContinuousEncoder:
    """Build, move to device, optionally compile with torch.compile.

    On CUDA: caller is responsible for converting to fp16 before compile.
    torch.compile(mode='reduce-overhead') is enabled only when explicitly
    requested to avoid compilation overhead during development.
    """
    enc = ContinuousEncoder(cfg, d_in_features).to(device)
    if compile_model and torch.cuda.is_available():
        enc = torch.compile(enc, mode="reduce-overhead")  # type: ignore[assignment]
    return enc
