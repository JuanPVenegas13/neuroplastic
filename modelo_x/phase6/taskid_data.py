"""Fase 6.3 — Tarea CONTINUA con identificador de tarea (no-olvido bien planteado).

5.1 mostró que con una sola cabeza y tareas que se contradicen, la retención simultánea
es imposible por construcción. Aquí lo arreglamos: el token de la posición 0 codifica
la TAREA (0=A, 1=B) y la regla de salida depende de él. Así A y B son COMPATIBLES (el
modelo puede aprender ambas leyendo el token de tarea), y la retención simultánea pasa
a ser un test legítimo de continual learning, comparable con la literatura (EWC).

  x[:,0] in {0,1}  -> identificador de tarea
  x[:,1:]          -> datos en {0..V-1}
  y[:,t] = (x[t] + x[t-1] + shift(tarea)) % V   para t>=2   (t<2 enmascarado)
  shift(A)=3, shift(B)=7
"""
from __future__ import annotations

import torch

from ..config import DriftConfig, ModelConfig
from ..drift_data import DriftTask

SHIFT = {0: 3, 1: 7}
IGNORE = -100


class TaskIDData:
    """Generador crudo (tarea explícita). Para baselines naive/EWC."""

    def __init__(self, mcfg: ModelConfig, device, seed: int = 0):
        self.m = mcfg; self.device = device
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def batch(self, task: int, bs: int):
        V, T = self.m.vocab_size, self.m.seq_len
        x = torch.randint(0, V, (bs, T), generator=self.g)
        x[:, 0] = task                      # token de tarea en la posición 0
        xlag = torch.roll(x, shifts=1, dims=1); xlag[:, 0] = 0
        y = (x + xlag + SHIFT[task]) % V
        y[:, :2] = IGNORE                   # enmascara pos 0 (token de tarea) y 1 (sin lag válido)
        return x.to(self.device), y.to(self.device)


class TaskIDDriftTask(DriftTask):
    """Versión compatible con el Trainer: tarea A en [0,switch), B en [switch,).
    evaluate(step) mide la tarea vigente en ese step (A si step<switch, B si >=)."""

    def __init__(self, mcfg, dcfg: DriftConfig, device, switch: int, seed: int = 0):
        super().__init__(mcfg, dcfg, device, seed)
        self.switch = switch

    def lag(self, step: int) -> int:
        return 1                            # lag fijo; lo que cambia es el shift por tarea

    def batch(self, step: int, bs: int):
        task = 0 if step < self.switch else 1
        V, T = self.m.vocab_size, self.m.seq_len
        x = torch.randint(0, V, (bs, T), generator=self.g)
        x[:, 0] = task
        xlag = torch.roll(x, shifts=1, dims=1); xlag[:, 0] = 0
        y = (x + xlag + SHIFT[task]) % V
        y[:, :2] = IGNORE
        return x.to(self.device), y.to(self.device)
