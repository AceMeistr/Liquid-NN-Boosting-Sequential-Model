from __future__ import annotations

import importlib
import math
import warnings
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ncps.torch import LTC

    HAS_LTC = True
except Exception:
    LTC = None
    HAS_LTC = False

try:
    import numpy as np
except ImportError:
    pass

try:
    _mamba_mod = importlib.import_module("mamba_ssm")
    NativeMamba = getattr(_mamba_mod, "Mamba")
    HAS_MAMBA = True
except Exception:
    NativeMamba = None
    HAS_MAMBA = False


@torch.no_grad()
def predict_lnn_batched(
    model: nn.Module,
    x_seq: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
    return_latent: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Batched LNN inference to handle large datasets without OOM.
    If return_latent=True, also returns the high-dimensional hidden embeddings.
    """
    model.eval()
    preds = np.empty(len(x_seq), dtype=np.float64)
    # The latent vector size is the hidden size before the final dual-layer head
    # The final linear layer expects `hidden_size // 4`. Depending on architecture,
    # the pre-prediction state is dim of output of block / backbone_norm.
    # We will extract `last` directly from forward(retun_latent=True)
    latents_list = [] if return_latent else None

    for start in range(0, len(x_seq), batch_size):
        end = min(start + batch_size, len(x_seq))
        chunk = torch.tensor(x_seq[start:end], dtype=torch.float32, device=device)
        
        if return_latent:
            out_pred, out_latent = model(chunk, return_latent=True)
            preds[start:end] = out_pred.detach().cpu().numpy().reshape(-1)
            latents_list.append(out_latent.detach().cpu().numpy())
        else:
            preds[start:end] = model(chunk).detach().cpu().numpy().reshape(-1)
            
        del chunk

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if return_latent:
        return preds, np.concatenate(latents_list, axis=0)
    return preds, None


# ---------------------------------------------------------------------------
# Squeezeâ€‘andâ€‘Excitation feature gating
# ---------------------------------------------------------------------------

class SqueezeExcite(nn.Module):
    """Channel-wise attention: learns to re-weight feature importance."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        scale = x.mean(dim=1)  # (B, C)
        scale = self.fc(scale).unsqueeze(1)  # (B, 1, C)
        return x * scale


# ---------------------------------------------------------------------------
# Multiâ€‘Head Selfâ€‘Attention block with causal mask
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Multi-head self-attention with causal masking for timeâ€‘series."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        z = self.norm(x)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(z, z, z, attn_mask=mask, need_weights=False)
        return x + self.dropout(attn_out)


# ---------------------------------------------------------------------------
# Feedâ€‘Forward block with residual
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class InceptionTemporalStem(nn.Module):
    """Inception-style multi-scale temporal feature extractor."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        kernels = (1, 3, 7, 15)
        branch_channels = max(channels // 4, 4)
        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(channels, branch_channels, kernel_size=k, padding=k // 2)
                for k in kernels
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(channels, branch_channels, kernel_size=1),
        )
        merged_channels = branch_channels * (len(kernels) + 1)
        self.proj = nn.Conv1d(merged_channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        z = x.transpose(1, 2)
        parts = [F.silu(branch(z)) for branch in self.conv_branches]
        parts.append(F.silu(self.pool_branch(z)))
        merged = torch.cat(parts, dim=1)
        fused = self.proj(merged).transpose(1, 2)
        return self.norm(x + self.dropout(fused))


class FallbackMambaCore(nn.Module):
    """Pure PyTorch selective state-space fallback for environments without mamba-ssm."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        _ = d_state  # interface compatibility
        inner = max(d_model * expand, d_model)
        self.in_proj = nn.Linear(d_model, inner * 2)
        self.dt_proj = nn.Linear(d_model, inner)
        self.dw_conv = nn.Conv1d(
            inner,
            inner,
            kernel_size=max(d_conv, 2),
            padding=max(d_conv, 2) - 1,
            groups=inner,
        )
        self.out_proj = nn.Linear(inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        u, gate = self.in_proj(x).chunk(2, dim=-1)

        # Causal depth-wise convolution over time dimension.
        u = self.dw_conv(u.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        u = F.silu(u)
        gate = torch.sigmoid(gate)
        delta = torch.sigmoid(self.dt_proj(x))

        state = torch.zeros(x.size(0), u.size(-1), dtype=x.dtype, device=x.device)
        out = torch.empty_like(u)
        for t in range(seq_len):
            dt = delta[:, t, :]
            state = (1.0 - dt) * state + dt * u[:, t, :]
            out[:, t, :] = state * gate[:, t, :]
        return self.out_proj(out)


class MambaBlock(nn.Module):
    """Residual Mamba block with LayerNorm and dropout."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        prefer_native: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.is_native = False
        if prefer_native and HAS_MAMBA:
            try:
                native_cls = cast(Any, NativeMamba)
                self.core = native_cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                self.is_native = True
            except Exception:
                self.core = FallbackMambaCore(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.core = FallbackMambaCore(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        y = self.core(z)
        return x + self.dropout(y)


# ---------------------------------------------------------------------------
# Positional encoding (sinusoidal)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.pe: torch.Tensor
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = cast(torch.Tensor, self.pe)
        return x + pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# Advanced MarketLNN with attention, residual, and LTC/GRU backbone
# ---------------------------------------------------------------------------

class MarketLNN(nn.Module):
    """Production-grade Liquid Neural Network for financial time-series.

    Architecture:
      Input projection -> Positional Encoding -> SE gating ->
      [TemporalAttention + FeedForward] x num_layers ->
      LTC/GRU backbone -> Dropout -> Multi-layer head -> Output

    Supports Monte Carlo dropout for uncertainty estimation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_layers: int = 2,
        use_attention: bool = True,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        backbone_type: str = "mamba",
        cnn_frontend: str = "inception",
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ) -> None:
        super().__init__()
        if backbone_type not in {"mamba", "attention_gru"}:
            raise ValueError("backbone_type must be one of {'mamba', 'attention_gru'}")
        if cnn_frontend not in {"none", "inception"}:
            raise ValueError("cnn_frontend must be one of {'none', 'inception'}")

        self.use_attention = use_attention
        self.use_residual = use_residual
        self.hidden_size = hidden_size
        self.backbone_type = backbone_type
        self.cnn_frontend = cnn_frontend

        # --- Input projection ---
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        self.pos_enc = PositionalEncoding(hidden_size)

        # --- Feature gating ---
        self.se_gate = SqueezeExcite(hidden_size)
        self.temporal_stem = InceptionTemporalStem(hidden_size, dropout=dropout) if cnn_frontend == "inception" else nn.Identity()

        if backbone_type == "mamba":
            self.sequence_blocks = nn.ModuleList(
                [
                    MambaBlock(
                        d_model=hidden_size,
                        dropout=dropout,
                        d_state=mamba_d_state,
                        d_conv=mamba_d_conv,
                        expand=mamba_expand,
                        prefer_native=True,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.attn_layers = nn.ModuleList()
            self.backbone = None
        else:
            self.sequence_blocks = nn.ModuleList()
            if use_attention:
                self.attn_layers = nn.ModuleList([
                    nn.ModuleList([
                        TemporalAttention(hidden_size, num_heads=num_heads, dropout=dropout),
                        FeedForward(hidden_size, expansion=4, dropout=dropout),
                    ])
                    for _ in range(num_layers)
                ])
            else:
                self.attn_layers = nn.ModuleList()

            # Legacy temporal backbone path (for backward compatibility).
            if HAS_LTC and LTC is not None:
                self.backbone = LTC(hidden_size, hidden_size)
            else:
                warnings.warn("ncps unavailable; using GRU fallback.", RuntimeWarning)
                self.backbone = nn.GRU(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=2,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )

        self.backbone_norm = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()

        # --- Prediction head ---
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size // 4, output_size),
        )

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, input_size)
        h = self.input_norm(self.input_proj(x))
        h = self.pos_enc(h)
        h = self.se_gate(h)
        h = self.temporal_stem(h)

        if self.backbone_type == "mamba":
            for block in self.sequence_blocks:
                h = block(h)
            h = self.backbone_norm(h)
        else:
            if self.use_attention:
                for layer in self.attn_layers:
                    pair = cast(nn.ModuleList, layer)
                    attn = cast(nn.Module, pair[0])
                    ff = cast(nn.Module, pair[1])
                    h = attn(h)
                    h = ff(h)

            if self.backbone is None:
                raise RuntimeError("Legacy backbone is not initialized")

            backbone_out, _ = self.backbone(h)
            if self.use_residual:
                backbone_out = backbone_out + h
            h = self.backbone_norm(backbone_out)

        last = self.dropout(h[:, -1, :])
        pred = self.head(last)
        
        if return_latent:
            return pred, last
        return pred

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo Dropout: run forward pass n_samples times with dropout
        enabled to estimate prediction uncertainty.

        Returns (mean_prediction, std_prediction).
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")

        was_training = self.training
        self.train()  # enable dropout
        preds: list[torch.Tensor] = []
        with torch.inference_mode():
            for _ in range(n_samples):
                preds.append(self.forward(x))

        stacked = torch.stack(preds, dim=0)
        if was_training:
            self.train()
        else:
            self.eval()
        return stacked.mean(dim=0), stacked.std(dim=0)
