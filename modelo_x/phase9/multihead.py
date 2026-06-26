"""Fase 9 — Infraestructura: arquitectura SEPARABLE por tarea (multi-cabeza).

La Fase 8 mostró que la CABEZA ÚNICA es lo que hace imposible la retención simultánea
para TODOS (EWC y Modelo X por igual): A y B exigen salidas distintas para las mismas
features y se contradicen punto a punto. Aquí damos capacidad separable: tronco
COMPARTIDO + una cabeza por tarea (task-incremental, el escenario estándar de CL donde
EWC está pensado para funcionar). Así:

  * El olvido se LOCALIZA en el tronco compartido (la cabeza inactiva no recibe
    gradiente en el forward -> se preserva sola, sin necesidad de protección).
  * El tronco es el recurso DISPUTADO donde compiten el congelamiento por BLOQUE de
    Modelo X (K-FAC opera sobre las linears del tronco) y el soft-freeze por-PARÁMETRO
    de EWC. Ése es el head-to-head limpio que la Fase 8 no pudo montar.

Tarea ESTRUCTURAL (A=lag1, B=lag2): el tronco debe atender a posiciones distintas, no
absorbible por la cabeza -> terreno propio de Modelo X (gradiente natural).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config, ModelConfig, DriftConfig
from ..drift_data import DriftTask, ManifoldDriftDetector
from ..kfac import KFACManager
from ..model import ModeloX
from ..thermodynamics import ThermalController
from ..phase5.ablated_trainer import AblatedTrainer

SHIFT = 3


class MultiHeadModeloX(ModeloX):
    """ModeloX con tronco compartido y una cabeza por tarea. `current_task` selecciona
    la cabeza en el forward; las demás cabezas no entran al grafo -> no reciben gradiente
    -> se preservan solas. K-FAC sigue operando sólo sobre las linears del TRONCO."""

    def __init__(self, cfg: ModelConfig, n_tasks: int = 2):
        super().__init__(cfg)
        self.heads = nn.ModuleList([nn.Linear(cfg.d_model, cfg.vocab_size)
                                    for _ in range(n_tasks)])
        del self.head                      # la cabeza única del padre no se usa
        self.current_task = 0

    def forward(self, idx: torch.Tensor):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return self.heads[self.current_task](self.ln_f(x))


class MultiHeadDriftTask(DriftTask):
    """Tarea reversible/secuencial por LAG con id de tarea explícito (vía current_task,
    no en la entrada). batch(step) usa lag_A en [0,switch) y lag_B en [switch,)."""

    def __init__(self, mcfg, dcfg: DriftConfig, device, switch: int,
                 lag_a: int = 1, lag_b: int = 2, seed: int = 0):
        super().__init__(mcfg, dcfg, device, seed)
        self.switch = switch
        self.lag_a, self.lag_b = lag_a, lag_b

    def task_of(self, step: int) -> int:
        return 0 if step < self.switch else 1

    def lag(self, step: int) -> int:
        return self.lag_a if step < self.switch else self.lag_b

    def batch_task(self, task: int, bs: int):
        """Batch explícito de una tarea (para eval y entreno por-fase)."""
        V, T = self.m.vocab_size, self.m.seq_len
        lag = self.lag_a if task == 0 else self.lag_b
        x = torch.randint(0, V, (bs, T), generator=self.g)
        xlag = torch.roll(x, shifts=lag, dims=1)
        xlag[:, :lag] = 0
        y = (x + xlag + SHIFT) % V
        return x.to(self.device), y.to(self.device)

    def batch(self, step: int, bs: int):
        return self.batch_task(self.task_of(step), bs)


class MultiHeadTrainer(AblatedTrainer):
    """Trainer de Modelo X sobre la red MULTI-CABEZA. Reusa K-FAC + termostato +
    detector (y las ablaciones de AblatedTrainer), pero construye MultiHeadModeloX y
    deja que `current_task` lo fije el loop externo (set antes de cada step)."""

    def __init__(self, cfg: Config, device, ablation: str = "no_jump", n_tasks: int = 2):
        if ablation == "no_detect":
            cfg.drift.jump_z = 1e9
        if ablation in ("no_jump",):
            cfg.model.jump_sigma = 0.0
        # --- réplica de Trainer.__init__ con MultiHeadModeloX (no se puede usar super
        #     porque el padre hardcodea ModeloX) ---
        self.cfg = cfg; self.device = device
        torch.manual_seed(cfg.train.seed)
        self.model = MultiHeadModeloX(cfg.model, n_tasks).to(device)
        self.kfac_linears = self.model.kfac_linears()
        self.kfac = KFACManager(self.kfac_linears, cfg.kfac, device)
        self.thermo = ThermalController(self.model.block_names(), cfg.thermo)
        self.detector = ManifoldDriftDetector(cfg.model.d_model, cfg.drift, device)
        kfac_params = {id(l.weight) for l in self.kfac_linears.values()}
        base_params = [p for p in self.model.parameters() if id(p) not in kfac_params]
        self.base_opt = torch.optim.Adam(base_params, lr=cfg.train.lr_base)
        self.traces = {}; self.prev_loss = None; self._reinf_baseline = None; self.log = []
        self.ablation = ablation
        self.task = None
        if ablation == "no_kfac":
            self.base_opt = torch.optim.Adam(self.model.parameters(), lr=cfg.train.lr_base)
