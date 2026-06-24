"""Fase 7 — Tareas para robustez ante derivas múltiples, graduales y recurrentes.

Hasta la Fase 6 todo fue UN solo Cisne Negro abrupto. Aquí se prueba el riesgo real
de estos sistemas: ¿la consolidación se ACUMULA bien o INTERFIERE tras varios cambios?
  * MultiDriftTask: cadena de regímenes (p.ej. lag 1 -> 2 -> 3 -> 1) en pasos dados.
  * GradualDriftTask: transición GRADUAL (rampa) entre dos regímenes mezclando lag_pre
    y lag_post con probabilidad creciente -> pone a prueba el detector de z-score, que
    está calibrado para SALTOS, no para rampas lentas.
"""
from __future__ import annotations

import torch

from ..config import DriftConfig, ModelConfig
from ..drift_data import DriftTask


def _make(x, lag, shift, V):
    xlag = torch.roll(x, shifts=lag, dims=1)
    xlag[:, :lag] = 0
    return (x + xlag + shift) % V


class MultiDriftTask(DriftTask):
    """Cadena de regímenes: schedule = [(hasta_paso, lag), ...] ordenado por paso."""

    def __init__(self, mcfg, dcfg: DriftConfig, device, schedule, seed: int = 0):
        super().__init__(mcfg, dcfg, device, seed)
        self.schedule = schedule           # p.ej. [(120,1),(240,2),(360,3),(10**9,1)]

    def lag(self, step: int) -> int:
        for until, lag in self.schedule:
            if step < until:
                return lag
        return self.schedule[-1][1]

    def batch(self, step: int, bs: int):
        V, T = self.m.vocab_size, self.m.seq_len
        x = torch.randint(0, V, (bs, T), generator=self.g)
        y = _make(x, self.lag(step), self.d.shift, V)
        return x.to(self.device), y.to(self.device)


class GradualDriftTask(DriftTask):
    """Transición GRADUAL de lag_pre a lag_post en [ramp_start, ramp_end]: cada secuencia
    usa lag_post con probabilidad p(step) que sube linealmente de 0 a 1 en la rampa."""

    def __init__(self, mcfg, dcfg: DriftConfig, device, ramp_start, ramp_end, seed: int = 0):
        super().__init__(mcfg, dcfg, device, seed)
        self.r0 = ramp_start; self.r1 = ramp_end

    def p_post(self, step: int) -> float:
        if step <= self.r0:
            return 0.0
        if step >= self.r1:
            return 1.0
        return (step - self.r0) / (self.r1 - self.r0)

    def lag(self, step: int) -> int:
        return self.d.lag_post if self.p_post(step) >= 0.5 else self.d.lag_pre

    def batch(self, step: int, bs: int):
        V, T = self.m.vocab_size, self.m.seq_len
        x = torch.randint(0, V, (bs, T), generator=self.g)
        p = self.p_post(step)
        use_post = torch.rand(bs, generator=self.g) < p     # mezcla por-secuencia
        lp = _make(x, self.d.lag_pre, self.d.shift, V)
        ls = _make(x, self.d.lag_post, self.d.shift, V)
        y = torch.where(use_post.unsqueeze(1), ls, lp)
        return x.to(self.device), y.to(self.device)
