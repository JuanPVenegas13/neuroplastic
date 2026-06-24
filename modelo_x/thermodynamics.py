"""Termodinámica de Modelo X (Incógnita 3 de la Fase 1).

Contiene:
  * FGSGELU: GELU estocástica cuya anchura de compuerta la fija la incertidumbre
    epistémica (σ² = η·Tr(F^{-1})). σ→0 ⇒ ReLU dura (FROZEN); σ→∞ ⇒ ~lineal (MELTED).
  * ThermalController: máquina de estados por capa con HISTÉRESIS (banda muerta
    entre θ_freeze y θ_melt) + EMA de la traza, para evitar el parpadeo de estado.
    Esta es la parte "termostato" que el documento implica pero no formaliza.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .config import ThermoConfig

FROZEN, PLASTIC, MELTED = "FROZEN", "PLASTIC", "MELTED"

class FGSGELU(nn.Module):
    """Fisher-Gated Stochastic GELU.  a(x) = x · Φ((x-μ)/σ).

    σ entra como escalar por-capa (derivado de la traza de Fisher inverso).
    `sample=True` muestrea la compuerta (ruido exploratorio); por defecto usa la
    forma esperada (suave y reparametrizable, estable para backprop)."""

    def __init__(self, sigma_floor: float = 1e-3):
        super().__init__()
        self.sigma_floor = sigma_floor
        self.mu = 0.0  # fijo en 0 (GELU estándar); podría hacerse aprendible

    def forward(self, x: torch.Tensor, sigma: float, sample: bool = False) -> torch.Tensor:
        s = max(float(sigma), self.sigma_floor)
        z = (x - self.mu) / s
        if not sample:
            # Φ gaussiana = 0.5(1+erf(z/√2)); con s→0 satura al escalón (ReLU).
            gate = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            return x * gate
        # versión muestreada: compuerta dura con umbral ruidoso ε~N(μ,σ²)
        eps = self.mu + s * torch.randn_like(x)
        return x * (x > eps).to(x.dtype)


@dataclass
class LayerThermo:
    state: str = PLASTIC
    trace_bar: float = 0.35      # EMA de la traza normalizada (arranca en banda PLÁSTICA)
    baseline: float = 0.0        # línea base lenta de la traza CRUDA (relativa, §3.4)
    sigma_gate: float = 1.0      # σ para FG-SGELU
    sigma_epist: float = 0.1     # σ de la difusión del SDE
    lam: float = 0.02            # intensidad base de salto


class ThermalController:
    """Mantiene el estado térmico de cada capa rastreada y deriva sus señales."""

    def __init__(self, layer_names: list[str], cfg: ThermoConfig):
        self.cfg = cfg
        self.names = layer_names
        self.thermo: dict[str, LayerThermo] = {n: LayerThermo() for n in layer_names}
        self.melt_timer = 0      # ventana refractaria global tras un Cisne Negro

    def _eta(self, step: int) -> float:
        """Recocido coseno de η: explora temprano, consolida tarde."""
        c = self.cfg
        if step >= c.anneal_steps:
            return c.eta_min
        cos = 0.5 * (1 + math.cos(math.pi * step / c.anneal_steps))
        return c.eta_min + (c.eta - c.eta_min) * cos

    def update(self, raw_traces: dict[str, float], jump_detected: bool, step: int):
        """Actualiza estados con histéresis. `raw_traces` son trazas CRUDAS
        Tr(A^{-1})Tr(S^{-1}); aquí se normalizan contra una línea base lenta y se
        mapean a [0,1] vía tanh(log-ratio), dándoles rango dinámico real."""
        c = self.cfg
        eta = self._eta(step)
        if jump_detected:
            self.melt_timer = c.melt_hold          # (re)arma la ventana refractaria
        in_refractory = self.melt_timer > 0
        if self.melt_timer > 0:
            self.melt_timer -= 1
        for n in self.names:
            t = self.thermo[n]
            raw = raw_traces.get(n, None)
            if raw is not None and raw > 0:
                if t.baseline <= 0:
                    t.baseline = raw                      # inicializa la línea base
                ratio = math.log(raw / max(t.baseline, 1e-12))
                # traza grande vs base -> hacia MELT; pequeña -> hacia FREEZE
                t_hat = 0.35 + 0.4 * math.tanh(c.trace_gain * ratio)
                t.baseline = c.baseline_ema * t.baseline + (1 - c.baseline_ema) * raw
                t.trace_bar = c.trace_ema * t.trace_bar + (1 - c.trace_ema) * t_hat

            # --- máquina de estados con banda muerta (histéresis) ---
            if in_refractory or t.trace_bar > c.theta_melt:
                t.state = MELTED
            elif t.trace_bar < c.theta_freeze and step >= c.freeze_warmup:
                t.state = FROZEN     # solo se congela tras el warmup (deja aprender primero)
            else:
                pass  # zona PLÁSTICA: conserva el estado anterior

            # --- señales derivadas del estado ---
            if t.state == FROZEN:
                t.sigma_gate = c.gate_sigma_floor      # compuerta dura, determinista
                t.sigma_epist = 0.0                    # sin difusión
                t.lam = 0.0                            # sin saltos
            elif t.state == MELTED:
                t.sigma_gate = eta * (t.trace_bar + 0.5)   # compuerta blanda
                t.sigma_epist = eta * t.trace_bar
                t.lam = 0.05 + 0.10 * (1.0 if in_refractory else 0.0)
            else:  # PLASTIC
                t.sigma_gate = 0.5 * eta * (t.trace_bar + 0.2)
                t.sigma_epist = 0.5 * eta * t.trace_bar
                t.lam = 0.02

    def signal(self, name: str) -> LayerThermo:
        return self.thermo[name]

    def states(self) -> dict[str, str]:
        return {n: t.state for n, t in self.thermo.items()}
