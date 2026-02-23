from __future__ import annotations

import warnings

import torch
import torch.nn as nn

try:
    from ncps.torch import LTC

    HAS_LTC = True
except Exception:
    LTC = None
    HAS_LTC = False


class MarketLNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        if HAS_LTC:
            self.backbone = LTC(input_size, hidden_size)
        else:
            warnings.warn("ncps unavailable; using GRU fallback.", RuntimeWarning)
            self.backbone = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.backbone(x)
        out = self.dropout(out[:, -1, :])
        return self.head(out)
