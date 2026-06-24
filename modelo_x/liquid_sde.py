"""Bloque neuronal de tiempo líquido (Incógnita 2 de la Fase 1).

Integra dh = [-h/τ + f] dt + σ dW + dJ vía operator splitting:
  * Drift rígido  -> integrador EXPONENCIAL (forma cerrada, incond. estable).
  * Difusión      -> Euler-Maruyama (orden 1 bajo ruido aditivo).
  * Salto         -> Poisson-thinning con compuerta de Bernoulli, gradiente
                     vía Straight-Through Estimator (Gemini #1).

Sobre el STE: la indicadora 1[u<λΔt] no es diferenciable. En forward usamos el
umbral duro; en backward dejamos pasar el gradiente de un surrogate sigmoide
(idéntico patrón que los "surrogate gradients" de redes neuronales de pulsos).
Es matemáticamente impuro pero barato y robusto para validar el lazo de la
Fase 2; el adjoint estocástico riguroso queda para la Fase 3.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ModelConfig


class _JumpGateSTE(torch.autograd.Function):
    """Compuerta de salto: forward duro, backward surrogate (Straight-Through)."""

    @staticmethod
    def forward(ctx, score: torch.Tensor, temp: float):
        ctx.save_for_backward(score)
        ctx.temp = temp
        return (score > 0).to(score.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (score,) = ctx.saved_tensors
        temp = ctx.temp
        s = torch.sigmoid(score / temp)
        return grad_out * s * (1 - s) / temp, None


def jump_gate(score: torch.Tensor, temp: float) -> torch.Tensor:
    return _JumpGateSTE.apply(score, temp)


class LiquidSDECell(nn.Module):
    """Una celda líquida con dinámica de salto-difusión.

    `drive` es f(x,W): el término de entrada que empuja el estado hacia τ·f.
    σ_epist y λ se inyectan desde el controlador térmico en cada paso."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # τ aprendible y positivo (vía softplus) por dimensión del modelo.
        self._tau_raw = nn.Parameter(torch.full((cfg.d_model,), math.log(math.e * cfg.tau_init - 1.0)))
        # Fase 4.1: logit aprendible de la probabilidad de salto (compartido en la celda)
        if cfg.learn_jump:
            self.jump_logit = nn.Parameter(torch.tensor(float(cfg.jump_logit_init)))
        self.last_logp = None    # log-prob acumulado de las decisiones de salto (REINFORCE)

    @property
    def tau(self) -> torch.Tensor:
        return nn.functional.softplus(self._tau_raw) + 1e-3

    def _learned_jump(self, h):
        """Salto con probabilidad aprendible p=σ(jump_logit). Devuelve (delta_h, logp, activity).
        STE: gradiente pathwise vía surrogate. REINFORCE: muestra sin grad + log-prob.
        jump_shared: una sola decisión por posición (compartida entre canales) -> reduce
        drásticamente la varianza del score-function (menos decisiones independientes)."""
        cfg = self.cfg
        eps = 1e-6
        p = torch.sigmoid(self.jump_logit).clamp(eps, 1 - eps)
        zeta = cfg.jump_mu + cfg.jump_sigma * torch.randn_like(h)   # magnitud reparametrizada
        # forma de la decisión de salto
        if cfg.jump_shared:
            shape = h.shape[:-1] + (1,)        # [B,T,1] compartido entre canales
        else:
            shape = h.shape                    # [B,T,d] por elemento
        if cfg.jump_grad == "reinforce":
            occur = torch.bernoulli(p.expand(shape)).detach()
            logp = (occur * torch.log(p) + (1 - occur) * torch.log(1 - p)).sum()
            return occur * zeta, logp, float(occur.mean().detach())
        # STE: gate diferenciable en p
        score = p - torch.rand(shape, device=h.device, dtype=h.dtype)
        gate = jump_gate(score, cfg.ste_temp)
        return gate * zeta, None, float(gate.mean().detach())

    def step(self, h, drive, sigma_epist: float, lam: float):
        """Un subpaso de integración. Devuelve (h_nuevo, actividad_de_salto, logp_o_None)."""
        cfg = self.cfg
        dt = cfg.dt
        tau = self.tau
        # 1) Drift exponencial (cerrado, estable): h* = τ·f
        h_eq = tau * drive
        decay = torch.exp(-dt / tau)
        h = h_eq + (h - h_eq) * decay
        # 2) Difusión Euler-Maruyama (orden 1, ruido aditivo)
        if sigma_epist > 0:
            h = h + sigma_epist * math.sqrt(dt) * torch.randn_like(h)
        # 3) Salto
        jump_activity = 0.0
        logp = None
        if cfg.learn_jump:
            # Fase 4.1: intensidad aprendible (la termodinámica habilita/inhibe vía lam>0)
            if lam > 0:
                dh, logp, jump_activity = self._learned_jump(h)
                h = h + dh
        elif lam > 0:
            # Fases 2-3: λ exógena (termostato) + STE, con subdivisión adaptativa
            import math as _m
            m_sub = max(1, _m.ceil(lam * dt / cfg.max_lambda_dt))
            sub_dt = dt / m_sub
            acc = 0.0
            for _ in range(m_sub):
                score = lam * sub_dt - torch.rand_like(h)    # >0 ⇒ salto
                gate = jump_gate(score, cfg.ste_temp)
                zeta = cfg.jump_mu + cfg.jump_sigma * torch.randn_like(h)
                h = h + gate * zeta
                acc += float(gate.mean().detach())
            jump_activity = acc
        return h, jump_activity, logp

    def forward(self, drive, sigma_epist: float, lam: float):
        """Integra `sde_steps` subpasos partiendo de h0=0. drive: [B,T,d]."""
        h = torch.zeros_like(drive)
        activity = 0.0
        logp_total = None
        for _ in range(self.cfg.sde_steps):
            h, ja, logp = self.step(h, drive, sigma_epist, lam)
            activity += ja
            if logp is not None:
                logp_total = logp if logp_total is None else logp_total + logp
        self.last_logp = logp_total
        return h, activity / max(1, self.cfg.sde_steps)
