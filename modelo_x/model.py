"""Ensamblaje de Modelo X: bloque (atención + SDE líquido + FG-SGELU) y modelo.

Cada bloque sustituye el FFN convencional por una celda líquida de salto-difusión
seguida de la compuerta FG-SGELU termodinámica. Las señales térmicas (σ_gate,
σ_epist, λ, estado) se escriben en el bloque antes del forward por el train loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MHA
from .config import ModelConfig
from .liquid_sde import LiquidSDECell
from .thermodynamics import FGSGELU, FROZEN


class ModeloXBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, idx: int):
        super().__init__()
        self.idx = idx
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MHA(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)   # f(x,W) del SDE
        self.sde = LiquidSDECell(cfg)
        self.act = FGSGELU(sigma_floor=1e-3)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # señales térmicas (escritas por el controlador cada paso)
        self.sigma_gate = 1.0
        self.sigma_epist = 0.1
        self.lam = 0.02
        self.state = "PLASTIC"
        self.last_jump_activity = 0.0
        self.deterministic = False   # modo evaluación: apaga difusión y saltos

    def kfac_linears(self) -> dict[str, nn.Linear]:
        d = self.attn.kfac_linears(f"b{self.idx}.attn")
        d[f"b{self.idx}.in"] = self.in_proj
        d[f"b{self.idx}.out"] = self.out_proj
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x + self.attn(self.ln1(x))
        drive = self.in_proj(self.ln2(a))
        if self.deterministic:
            h, ja = self.sde(drive, sigma_epist=0.0, lam=0.0)
            g = self.act(h, sigma=self.sigma_gate, sample=False)
        else:
            h, ja = self.sde(drive, self.sigma_epist, self.lam)
            g = self.act(h, sigma=self.sigma_gate, sample=(self.state == "MELTED"))
        self.last_jump_activity = ja
        return a + self.out_proj(g)


class ModeloX(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([ModeloXBlock(cfg, i) for i in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def kfac_linears(self) -> dict[str, nn.Linear]:
        out: dict[str, nn.Linear] = {}
        for b in self.blocks:
            out.update(b.kfac_linears())
        return out

    def block_names(self) -> list[str]:
        """Nombre representativo por bloque (la proyección de salida) para el
        controlador térmico: una decisión térmica por bloque."""
        return [f"b{b.idx}.out" for b in self.blocks]

    def jump_logp_sum(self):
        """Suma de log-probs de las decisiones de salto sobre los bloques (REINFORCE)."""
        total = None
        for b in self.blocks:
            lp = b.sde.last_logp
            if lp is not None:
                total = lp if total is None else total + lp
        return total

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        return self.tok(idx) + self.pos(pos)

    def forward(self, idx: torch.Tensor):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))
