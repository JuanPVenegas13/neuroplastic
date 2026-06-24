"""Fase 8 — Eje 2: AHORRO al revisitar (deriva reversible A->B->A, lag ESTRUCTURAL).

Éste es el eje donde Modelo X SÍ está validado (5.2: re-aprende A ~3.5x más rápido).
La pregunta del head-to-head: ¿EWC (sobre Adam) también ahorra aquí?

Dos controles de honestidad imprescindibles (los descubrimos midiendo, ver README):

  1. GATE iso-acc_B sobre el ahorro: el ahorro sólo cuenta si el método REALMENTE
     aprendió B (accB_max >= TARGET en la ventana B). EWC con λ alto "ahorra" porque
     nunca soltó A (no aprendió B): ahorro FALSO, descalificado.

  2. La deriva de lag es ESTRUCTURAL (cambia a qué posición atender). Adam PLANO
     (naive y EWC, que viven sobre Adam) NO la navega: no llega a aprender B (5.3:
     "Adam plano aprende A pero NO re-adapta a B"). Modelo X (gradiente natural K-FAC)
     sí. Es una diferencia ARQUITECTÓNICA, no un confound: en su terreno propio Modelo X
     compite contra un EWC que ni siquiera puede entrar.

Reportamos A1, A2 y accB_max por separado para que el confound de "pesos calientes"
(re-aprender una tarea ya vista es barato incluso para naive) quede a la vista.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config, get_device
from ..model import ModeloX
from ..phase5.ablated_trainer import AblatedTrainer, ReversibleTask, steps_to_acc
from .decoupled_ewc import _decoupled_pull, snapshot

D1, D2, STEPS, TARGET = 150, 320, 470, 0.9
SEEDS = (0, 1, 2)


# ----------------------------------------------------- camino plano (naive / EWC)
def _fisher_lag(model, task, n=40, bs=64):
    fis = {nm: torch.zeros_like(p) for nm, p in model.named_parameters()}
    for _ in range(n):
        x, y = task.batch(0, bs)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).reshape(-1, model.cfg.vocab_size), y.reshape(-1)).backward()
        for nm, p in model.named_parameters():
            if p.grad is not None:
                fis[nm] += p.grad.detach() ** 2 / n
    return fis


@torch.no_grad()
def _ev(model, task, step, n=4, bs=64):
    for b in model.blocks:
        b.deterministic = True
    c = t = 0
    for _ in range(n):
        x, y = task.batch(step, bs)
        lo = model(x); c += (lo.argmax(-1) == y).sum().item(); t += y.numel()
    for b in model.blocks:
        b.deterministic = False
    return c / t


def _run_plain(seed, lam=None):
    """naive (lam=None) o EWC desacoplado (lam) sobre Adam plano, tarea reversible lag."""
    dev = get_device()
    c = Config(); torch.manual_seed(seed)
    m = ModeloX(c.model).to(dev)
    for b in m.blocks:
        b.deterministic = True
    task = ReversibleTask(c.model, c.drift, dev, d1=D1, d2=D2, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    star = fis = None
    log = []
    for s in range(STEPS):
        if lam is not None and s == D1:           # consolida A justo antes de B
            star = snapshot(m); fis = _fisher_lag(m, task)
        x, y = task.batch(s, 64)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).reshape(-1, c.model.vocab_size), y.reshape(-1)).backward()
        opt.step()
        if lam is not None and star is not None:
            _decoupled_pull(m, fis, star, lam, 3e-3)
        if s % 5 == 0 or s == STEPS - 1:
            log.append({"step": s, "eval_acc": _ev(m, task, s)})
    return log


# ----------------------------------------------------- camino Modelo X (K-FAC)
def _run_modelox(seed, ablation):
    dev = get_device()
    c = Config()
    c.train.steps = STEPS; c.train.batch_size = 64; c.train.seed = seed
    c.train.log_every = 5
    c.drift.drift_step = D1
    c.thermo.freeze_warmup = 100; c.thermo.anneal_steps = STEPS
    tr = AblatedTrainer(c, dev, ablation=ablation)
    tr.task = ReversibleTask(c.model, c.drift, dev, d1=D1, d2=D2, seed=seed)
    for s in range(STEPS):
        tr.step(s)
    tr.kfac.remove_hooks()
    return tr.log


def _metrics(log):
    a1 = steps_to_acc(log, 0, TARGET, end_step=D1)
    a2 = steps_to_acc(log, D2, TARGET)
    accB = max((r["eval_acc"] for r in log
                if r["eval_acc"] is not None and D1 <= r["step"] < D2), default=0.0)
    sv = (a1 - a2) if (a1 is not None and a2 is not None) else None
    return a1, a2, sv, accB


def run():
    print("\n=== Fase 8.2 — Ahorro al revisitar (reversible lag estructural) ===")
    print(f"calendario: A[0,{D1}) -> B[{D1},{D2}) -> A[{D2},{STEPS})  objetivo acc>={TARGET}\n")
    methods = [
        ("ModeloX full", lambda s: _run_modelox(s, "full")),
        ("ModeloX no_thermo", lambda s: _run_modelox(s, "no_thermo")),
        ("naive (Adam)", lambda s: _run_plain(s, None)),
        ("EWC desac. 5e8", lambda s: _run_plain(s, 5e8)),
    ]
    print(f"  {'método':<20} {'A1':>5} {'A2':>5} {'ahorro':>7} {'accB_max':>9}  válido?")
    res = {}
    for name, fn in methods:
        rows = [_metrics(fn(s)) for s in SEEDS]
        a1 = np.mean([r[0] for r in rows if r[0] is not None]) if any(r[0] is not None for r in rows) else None
        a2 = np.mean([r[1] for r in rows if r[1] is not None]) if any(r[1] is not None for r in rows) else None
        accB = np.mean([r[3] for r in rows])
        sv = (a1 - a2) if (a1 is not None and a2 is not None) else None
        valido = accB >= TARGET
        res[name] = (a1, a2, sv, accB, valido)
        f = lambda v: f"{v:.0f}" if v is not None else "  -"
        print(f"  {name:<20} {f(a1):>5} {f(a2):>5} {(f'{sv:.0f}' if sv is not None else '  -'):>7} "
              f"{accB:9.2f}  {'SÍ' if valido else 'NO (ahorro falso/no aprende B)'}")

    print("\n--- Checks ---")
    mf = res["ModeloX full"]; nv = res["naive (Adam)"]; ew = res["EWC desac. 5e8"]
    print(f"[check] Modelo X aprende B en deriva ESTRUCTURAL (accB>={TARGET}): {mf[4]}")
    print(f"[check] naive/EWC (Adam plano) NO aprenden B estructural -> ahorro inválido: "
          f"{(not nv[4]) and (not ew[4])}")
    print(f"[check] en su terreno (estructural) Modelo X compite contra un EWC que no entra: "
          f"{mf[4] and not ew[4]}")
    print("\nLectura: el ahorro de Modelo X (5.2) vive en deriva ESTRUCTURAL, donde el")
    print("gradiente natural K-FAC es prerequisito para re-adaptar. EWC, atado a Adam, no")
    print("llega a aprender B ahí, así que su 'ahorro' es falso (nunca soltó A). En drift")
    print("ABSORBIBLE (shift, eje 1) EWC sí juega, pero ahí nadie retiene simultáneo. EWC y")
    print("Modelo X resuelven problemas casi DISJUNTOS.")
    return res


if __name__ == "__main__":
    run()
