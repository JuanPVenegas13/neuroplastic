"""Fase 5 — Infraestructura de validación de la AFIRMACIÓN CENTRAL.

Hasta ahora todo se midió sobre UNA deriva y sin re-evaluar nunca la tarea vieja.
Aquí se prueba lo que el framework promete: neuroplasticidad SIN olvido catastrófico
mediante congelamiento selectivo. Para ello:
  * AblatedTrainer: desactiva subsistemas (termodinámica, detección, salto, K-FAC)
    para cuantificar qué carga cada pieza.
  * ReversibleTask: deriva A -> B -> A para medir retención y "ahorro" (re-aprender
    A más rápido la segunda vez si su estructura quedó consolidada).
"""
from __future__ import annotations

import torch

from ..config import Config
from ..drift_data import DriftTask
from ..train import Trainer, linear_to_block
from ..thermodynamics import FROZEN


class ReversibleTask(DriftTask):
    """Deriva A->B->A: lag_pre en [0,d1), lag_post en [d1,d2), lag_pre de nuevo en [d2,)."""

    def __init__(self, mcfg, dcfg, device, d1: int, d2: int, seed: int = 0):
        super().__init__(mcfg, dcfg, device, seed)
        self.d1 = d1; self.d2 = d2

    def lag(self, step: int) -> int:
        if step < self.d1:
            return self.d.lag_pre        # A
        if step < self.d2:
            return self.d.lag_post       # B
        return self.d.lag_pre            # A de nuevo


class AblatedTrainer(Trainer):
    """Trainer con un subsistema desactivado. ablation in:
       'full' | 'no_thermo' | 'no_detect' | 'no_jump' | 'no_kfac'."""

    def __init__(self, cfg: Config, device, ablation: str = "full", task=None):
        # no_detect: el detector nunca dispara (z imposible); no_jump: magnitud de salto 0
        if ablation == "no_detect":
            cfg.drift.jump_z = 1e9
        if ablation == "no_jump":
            cfg.model.jump_sigma = 0.0
        super().__init__(cfg, device)
        self.ablation = ablation
        if task is not None:
            self.task = task
        if ablation == "no_kfac":
            # mete TODOS los parámetros en Adam (sin gradiente natural)
            self.base_opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr_base)

    def _write_thermal_signals(self):
        if self.ablation == "no_thermo":
            # siempre plástico, sin congelar ni inyectar ruido de fusión
            for b in self.model.blocks:
                b.sigma_gate = self.cfg.thermo.eta
                b.sigma_epist = 0.0
                b.lam = 0.0
                b.state = "PLASTIC"
            return
        super()._write_thermal_signals()

    def _apply_natural_gradient_step(self):
        if self.ablation == "no_kfac":
            return                       # sin paso de gradiente natural (solo Adam)
        if self.ablation == "no_thermo":
            # aplica gradiente natural a TODO (nunca congela)
            lr = self.cfg.train.lr_kfac
            for name, lin in self.kfac_linears.items():
                if lin.weight.grad is None:
                    continue
                nat = self.kfac.natural_gradient(name, lin.weight.grad)
                gn = nat.norm()
                if gn > 10.0:
                    nat = nat * (10.0 / (gn + 1e-9))
                lin.weight.data.add_(nat, alpha=-lr)
                lin.weight.grad = None
            return
        super()._apply_natural_gradient_step()


def steps_to_acc(log, start_step: int, target: float, end_step: int | None = None):
    """Nº de pasos desde start_step hasta que eval_acc >= target (None si no se alcanza).
    El eval_acc registrado se mide sobre la tarea vigente en cada paso, así que dentro de
    una ventana A mide A y dentro de una ventana B mide B."""
    for r in log:
        if r["step"] < start_step:
            continue
        if end_step is not None and r["step"] >= end_step:
            break
        if r["eval_acc"] is not None and r["eval_acc"] >= target:
            return r["step"] - start_step
    return None
