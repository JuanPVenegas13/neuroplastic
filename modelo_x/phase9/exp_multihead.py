"""Fase 9 — ¿El congelamiento por BLOQUE de Modelo X compite con el soft-freeze
por-PARÁMETRO de EWC, donde la retención simultánea SÍ es alcanzable?

La Fase 8 cerró que en una cabeza ÚNICA nadie retiene (límite estructural). La Fase 9
da capacidad separable (tronco compartido + una cabeza por tarea, task-incremental) y
mide. Hallazgos (todos medidos, varios refutan la hipótesis de partida):

  9.1  Multi-cabeza es NECESARIO pero NO SUFICIENTE: con tronco JUSTO (d=64) NADIE
       retiene (EWC either/or, Modelo X olvida). El cuello de botella es el tronco
       COMPARTIDO, no la cabeza.
  9.2  La retención simultánea exige CAPACIDAD SOBRANTE + ANCLAJE FUERTE: sólo con
       tronco ancho (d=256) y EWC por-parámetro a λ alto se logra (min(ret_A,acc_B)~0.7).
  9.3  GRANULARIDAD (refuta la predicción falsable): engrosar el Fisher de EWC a nivel
       de BLOQUE (mímica de Modelo X) DESTRUYE la retención (min 0.7 -> ~0.2). El
       congelamiento por bloque es estructuralmente incapaz de explotar la holgura
       SUB-BLOQUE. Per-param y bloque DIVERGEN, no convergen.
  9.4  Modelo X (congelamiento por bloque real) NO retiene (ret_A~0.1) ni siquiera con
       capacidad, confirmando 9.3 sobre el mecanismo real.

Conclusión: se localiza una limitación CONCRETA del termostato de Modelo X (granularidad
de bloque), y EWC por-parámetro es la mejor herramienta de retención incluso en el
régimen construido para favorecer el congelamiento. No se fabrica un ganador para Modelo X.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import replace

from ..config import Config, get_device
from .multihead import MultiHeadModeloX, MultiHeadDriftTask, MultiHeadTrainer

SA = SB = 250
SWITCH, TOTAL = 250, 500
SEEDS = (0, 1, 2)
LAM_GRID = (1e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11)


# --------------------------------------------------------------------------- helpers
def _cfg(d_model):
    c = Config(); c.model = replace(c.model, d_model=d_model); c.train.batch_size = 64
    return c


def _fresh(seed, d_model):
    c = _cfg(d_model); torch.manual_seed(seed)
    m = MultiHeadModeloX(c.model).to(get_device())
    for b in m.blocks:
        b.deterministic = True
    task = MultiHeadDriftTask(c.model, c.drift, get_device(), switch=SWITCH, seed=seed)
    return c, m, task


@torch.no_grad()
def acc(m, task, t, n=8, bs=64):
    m.current_task = t
    for b in m.blocks:
        b.deterministic = True
    c = tot = 0
    for _ in range(n):
        x, y = task.batch_task(t, bs)
        lo = m(x); c += (lo.argmax(-1) == y).sum().item(); tot += y.numel()
    return c / tot


def fisher(m, task, t, n=40, bs=64):
    """Fisher diagonal VERDADERO (y~p_modelo) sobre la tarea t (mejor versión de EWC)."""
    m.current_task = t
    fis = {nm: torch.zeros_like(p) for nm, p in m.named_parameters()}
    V = m.cfg.vocab_size
    for _ in range(n):
        x, _ = task.batch_task(t, bs)
        m.zero_grad(set_to_none=True)
        logits = m(x).reshape(-1, V)
        with torch.no_grad():
            tgt = torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)
        F.cross_entropy(logits, tgt).backward()
        for nm, p in m.named_parameters():
            if p.grad is not None:
                fis[nm] += p.grad.detach() ** 2 / n
    return fis


def _block_group(name):
    if name.startswith("blocks.0"):
        return "b0"
    if name.startswith("blocks.1"):
        return "b1"
    return "shared"


def coarsen_to_block(fis):
    """Engrosa el Fisher a granularidad de BLOQUE: cada parámetro recibe la MEDIA de
    Fisher de su bloque. Mímica del congelamiento por bloque (Modelo X) sobre EWC."""
    tot, cnt = defaultdict(float), defaultdict(float)
    for n, f in fis.items():
        tot[_block_group(n)] += f.sum().item(); cnt[_block_group(n)] += f.numel()
    means = {g: tot[g] / cnt[g] for g in tot}
    return {n: torch.full_like(f, means[_block_group(n)]) for n, f in fis.items()}


def snapshot(m):
    return {nm: p.detach().clone() for nm, p in m.named_parameters()}


@torch.no_grad()
def _pull(m, fis, star, lam, lr):
    for nm, p in m.named_parameters():
        if nm in fis:
            c = (lr * lam * fis[nm]).clamp_(0.0, 1.0)
            p.add_(c * (star[nm] - p))


def train_plain(m, task, t, steps, lr=3e-3, decoupled=None, bs=64):
    m.current_task = t
    for b in m.blocks:
        b.deterministic = True
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        x, y = task.batch_task(t, bs)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).reshape(-1, m.cfg.vocab_size), y.reshape(-1)).backward()
        opt.step()
        if decoupled is not None:
            _pull(m, *decoupled, lr=lr)


def _seq(seed, d_model, lam=None, coarse=False):
    """A -> consolida -> B (EWC desac. opcional, per-param o engrosado). (ret_A, acc_B)."""
    c, m, task = _fresh(seed, d_model)
    train_plain(m, task, 0, SA)
    star = snapshot(m)
    if lam is not None:
        fis = fisher(m, task, 0)
        if coarse:
            fis = coarsen_to_block(fis)
        train_plain(m, task, 1, SB, decoupled=(fis, star, lam))
    else:
        train_plain(m, task, 1, SB)
    return acc(m, task, 0), acc(m, task, 1)


def _modelox(seed, d_model, ablation="no_jump"):
    dev = get_device()
    c = _cfg(d_model)
    c.train.steps = TOTAL; c.train.seed = seed
    c.thermo.freeze_warmup = 120; c.thermo.anneal_steps = TOTAL
    c.drift.drift_step = SWITCH
    tr = MultiHeadTrainer(c, dev, ablation=ablation)
    tr.task = MultiHeadDriftTask(c.model, c.drift, dev, switch=SWITCH, seed=seed)
    for s in range(TOTAL):
        tr.model.current_task = tr.task.task_of(s)
        tr.step(s)
    tr.kfac.remove_hooks()
    return acc(tr.model, tr.task, 0), acc(tr.model, tr.task, 1)


def _agg(rows):
    a = np.array(rows); return a[:, 0].mean(), a[:, 1].mean()


def _frontier(d_model, coarse=False):
    return {lam: _agg([_seq(s, d_model, lam, coarse) for s in SEEDS]) for lam in LAM_GRID}


# --------------------------------------------------------------------------- run
def run():
    print("\n=== Fase 9 — Bloque (Modelo X) vs por-parámetro (EWC) en multi-cabeza ===\n")

    # 0) benchmark válido (d=256): la solución conjunta existe (intercalado)
    c, m, task = _fresh(0, 256)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for i in range(1000):
        t = i % 2; m.current_task = t
        x, y = task.batch_task(t, 64)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).reshape(-1, c.model.vocab_size), y.reshape(-1)).backward(); opt.step()
    joint = (acc(m, task, 0), acc(m, task, 1))
    print(f"(0) benchmark válido (d=256): intercalado -> accA={joint[0]:.2f} accB={joint[1]:.2f}\n")

    for d in (64, 256):
        print(f"================= TRONCO d_model={d} =================")
        naive = _agg([_seq(s, d) for s in SEEDS])
        mx = _agg([_modelox(s, d, "no_jump") for s in SEEDS])
        pp = _frontier(d, coarse=False)
        print(f"  naive:              ret_A={naive[0]:.2f} acc_B={naive[1]:.2f}")
        print(f"  Modelo X (no_jump): ret_A={mx[0]:.2f} acc_B={mx[1]:.2f}")
        print(f"  --- EWC desacoplado POR-PARÁMETRO (frontera) ---")
        print(f"  {'λ':>8} {'ret_A':>6} {'acc_B':>6} {'min':>6}")
        for lam in LAM_GRID:
            rA, aB = pp[lam]
            print(f"  {lam:8.0e} {rA:6.2f} {aB:6.2f} {min(rA, aB):6.2f}")
        best_pp = max(LAM_GRID, key=lambda l: min(pp[l]))
        print(f"  -> mejor min per-param = {min(pp[best_pp]):.2f} "
              f"(ret_A={pp[best_pp][0]:.2f} acc_B={pp[best_pp][1]:.2f} @ λ={best_pp:.0e})")

        if d == 256:   # la prueba de granularidad sólo es informativa donde per-param SÍ retiene
            co = _frontier(d, coarse=True)
            best_co = max(LAM_GRID, key=lambda l: min(co[l]))
            print(f"  --- EWC ENGROSADO A BLOQUE (mímica Modelo X) ---")
            print(f"  -> mejor min bloque    = {min(co[best_co]):.2f} "
                  f"(ret_A={co[best_co][0]:.2f} acc_B={co[best_co][1]:.2f} @ λ={best_co:.0e})")
            _checks_and_plot(joint, naive, mx, pp, co, best_pp, best_co)


def _checks_and_plot(joint, naive, mx, pp, co, best_pp, best_co):
    print("\n--- Checks (d=256, el régimen donde la retención ES alcanzable) ---")
    print(f"[check] benchmark válido (intercalado>0.9 ambos): {min(joint) > 0.9}")
    print(f"[check] naive olvida A: {naive[0] < 0.3}")
    print(f"[check] EWC POR-PARÁMETRO logra retención simultánea (min>0.5): {min(pp[best_pp]) > 0.5}")
    print(f"[check] engrosar a BLOQUE la destruye (min cae >0.3 abs): "
          f"{min(pp[best_pp]) - min(co[best_co]) > 0.3}")
    print(f"[check] Modelo X (bloque real) NO retiene (ret_A<0.3): {mx[0] < 0.3}")
    print("\nLectura: per-param y bloque DIVERGEN. El congelamiento por bloque de Modelo X")
    print("es estructuralmente demasiado crudo para explotar la holgura SUB-BLOQUE que la")
    print("retención simultánea exige. Refuta la predicción 'EWC alto -> hard-freeze de Modelo X'.")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        p = np.array([pp[l] for l in LAM_GRID]); c = np.array([co[l] for l in LAM_GRID])
        plt.figure(figsize=(6, 5))
        plt.plot(p[:, 1], p[:, 0], "o-", label="EWC por-parámetro")
        plt.plot(c[:, 1], c[:, 0], "s--", color="gray", label="EWC engrosado a bloque (≈Modelo X)")
        plt.scatter([naive[1]], [naive[0]], c="k", marker="x", s=90, label="naive", zorder=5)
        plt.scatter([mx[1]], [mx[0]], c="red", marker="*", s=240, label="Modelo X (bloque real)", zorder=6)
        plt.xlabel("acc_B"); plt.ylabel("retención_A")
        plt.title("Fase 9 (d=256) — granularidad: por-parámetro vs bloque\nretención simultánea alcanzable sólo per-param")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("phase9_granularity.png", dpi=120)
        print("\n[fig] phase9_granularity.png")
    except Exception:
        pass


if __name__ == "__main__":
    run()
