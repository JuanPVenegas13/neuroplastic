"""Auto-atención multi-cabeza con proyecciones rastreables por K-FAC.

Todos los parámetros entrenables viven en las 4 proyecciones lineales
(q, k, v, out). El softmax y QK^T no tienen pesos, así que K-FAC se aplica solo a
esas Linear. El KFACManager se engancha a ellas desde el modelo (ver model.py)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ModelConfig


class MHA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.h = cfg.n_heads
        self.dk = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def kfac_linears(self, prefix: str) -> dict[str, nn.Linear]:
        return {f"{prefix}.q": self.q, f"{prefix}.k": self.k,
                f"{prefix}.v": self.v, f"{prefix}.out": self.out}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        def split(t):
            return t.view(B, T, self.h, self.dk).transpose(1, 2)  # [B,h,T,dk]
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        # máscara causal
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(o)
