"""Fase 3.1 — Estimadores del gradiente del salto discreto.

El problema (Fase 1, Incógnita 2): la indicadora 1[u<p] del Poisson-thinning no es
diferenciable. La Fase 2 usó un Straight-Through Estimator (STE): barato y de baja
varianza, pero SESGADO. Aquí construimos el estimador riguroso por score-function
(REINFORCE) —insesgado— y MEDIMOS el sesgo/varianza de cada uno contra una verdad
analítica.

Montaje aislado (un paso, un salto escalar) donde el gradiente exacto es cerrado:
    p = σ(θ)                      # prob. de salto, parametrizada por el logit θ
    occur ~ Bernoulli(p)
    ζ ~ N(μ_J, σ_J²)              # magnitud (reparametrizable, no es el problema)
    h = h0 + occur·ζ
    L = ½ (h - target)²
    E[L](p) = (1-p)·L_no + p·E_ζ[L_sí]
    dE[L]/dθ = (E_ζ[L_sí] - L_no) · p(1-p)        <-- VERDAD ANALÍTICA
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class JumpSetup:
    h0: float = 0.0
    target: float = 1.0
    mu_j: float = 0.0
    sigma_j: float = 1.0

    def L_no(self) -> float:
        return 0.5 * (self.h0 - self.target) ** 2

    def L_yes_expected(self) -> float:
        # E_ζ[½(h0+ζ-target)²] = ½[(h0+μ-target)² + σ²]
        m = self.h0 + self.mu_j - self.target
        return 0.5 * (m * m + self.sigma_j ** 2)

    def true_grad_theta(self, theta: float) -> float:
        p = 1.0 / (1.0 + math.exp(-theta))
        return (self.L_yes_expected() - self.L_no()) * p * (1 - p)


def _loss(h, target):
    return 0.5 * (h - target) ** 2


def ste_grad_samples(setup: JumpSetup, theta: float, n: int, temp: float = 0.1,
                     seed: int = 0) -> torch.Tensor:
    """Muestras del estimador STE de dL/dθ. Forward duro, backward surrogate sigmoide."""
    g = torch.Generator().manual_seed(seed)
    p = torch.sigmoid(torch.tensor(float(theta)))
    u = torch.rand(n, generator=g)
    zeta = setup.mu_j + setup.sigma_j * torch.randn(n, generator=g)
    occur = (u < p).float()
    h = setup.h0 + occur * zeta
    dL_dh = (h - setup.target)          # dL/dh
    dh_dgate = zeta                     # dh/d occur
    # surrogate: d gate/d score con score = p - u  (gate = 1[score>0])
    score = p - u
    sig = torch.sigmoid(score / temp)
    dgate_dscore = sig * (1 - sig) / temp
    dscore_dtheta = p * (1 - p)         # dp/dθ
    return dL_dh * dh_dgate * dgate_dscore * dscore_dtheta


def reinforce_grad_samples(setup: JumpSetup, theta: float, n: int,
                           baseline: float | None = None, seed: int = 0) -> torch.Tensor:
    """Muestras del estimador REINFORCE de dL/dθ.
    score-function: d logP(occur)/dθ = occur - p. Estimador = (L - b)(occur - p)."""
    g = torch.Generator().manual_seed(seed)
    p = torch.sigmoid(torch.tensor(float(theta)))
    u = torch.rand(n, generator=g)
    zeta = setup.mu_j + setup.sigma_j * torch.randn(n, generator=g)
    occur = (u < p).float()
    h = setup.h0 + occur * zeta
    L = _loss(h, setup.target)
    b = L.mean().item() if baseline is None else baseline   # línea base óptima ~ E[L]
    return (L - b) * (occur - p)
