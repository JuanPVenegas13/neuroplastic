"""Fase 10.4 — Dinámica del olvido en Modelo X y verificación de la señal PER-LINEAR.

Los experimentos 9/10.1-10.3 miden el ESTADO FINAL (ret_A, acc_B). Eso deja dos
preguntas sin responder que condicionan el curso de acción de la Fase 11:

  (1) ¿CUÁNDO se pierde A? Si A colapsa de golpe en el melt del switch, el problema es
      el melt global (fundir TODO al detectar deriva); si se erosiona gradualmente
      durante B, el problema es la ausencia de anclaje durante la re-adaptación. Ambos
      piden fixes distintos.
  (2) ¿EXISTE la señal per-Linear? La Fase 11 propone decidir el freeze por-Linear con
      la propia curvatura K-FAC. Modelo X YA calcula Tr(F⁻¹) por Linear (kfac.py) pero
      sólo consulta el agregado del bloque (block_names). Si las trazas de las 6 Linears
      de un bloque son casi idénticas, la premisa de F11 es débil; si se diferencian
      (spread grande y estable), la señal existe y el rediseño es viable sin coste extra.

Salidas: `phase10_dynamics.png` (3 paneles: accs por tarea, estados térmicos, trazas
per-Linear) y `phase10_dynamics.json` (resumen cuantitativo).
"""
from __future__ import annotations

import json

import numpy as np
import torch

from ..config import Config, get_device
from ..phase9.exp_multihead import SWITCH, TOTAL, _cfg
from ..phase9.multihead import MultiHeadDriftTask, MultiHeadTrainer

D_MODEL = 256
EVAL_EVERY = 10
STATE_CODE = {"PLASTIC": 0, "FROZEN": 1, "MELTED": 2}


@torch.no_grad()
def _eval(m, task, t, n=2, bs=64):
    """Eval que RESTAURA deterministic=False (a mitad de entrenamiento no debe
    contaminar la estocasticidad del forward de los pasos siguientes)."""
    m.current_task = t
    for b in m.blocks:
        b.deterministic = True
    c = tot = 0
    for _ in range(n):
        x, y = task.batch_task(t, bs)
        lo = m(x); c += (lo.argmax(-1) == y).sum().item(); tot += y.numel()
    for b in m.blocks:
        b.deterministic = False
    return c / tot


def run(seed: int = 0):
    dev = get_device()
    c = _cfg(D_MODEL)
    c.train.steps = TOTAL; c.train.seed = seed
    c.train.log_every = 10 ** 9        # la eval fina la hace este script, no el Trainer
    c.thermo.freeze_warmup = 120; c.thermo.anneal_steps = TOTAL
    c.drift.drift_step = SWITCH
    tr = MultiHeadTrainer(c, dev, ablation="no_jump")
    tr.task = MultiHeadDriftTask(c.model, c.drift, dev, switch=SWITCH, seed=seed)

    print(f"\n=== Fase 10.4 — Dinámica del olvido (d={D_MODEL}, seed={seed}) ===\n")
    evals, states_hist, traces_hist = [], [], []
    for s in range(TOTAL):
        tr.model.current_task = tr.task.task_of(s)
        rec = tr.step(s)
        states_hist.append((s, dict(rec["states"])))
        if s % c.kfac.invert_every == 0 and s > 0:
            traces_hist.append((s, {name: tr.kfac.inverse_fisher_trace(name)
                                    for name in tr.kfac_linears}))
        if s % EVAL_EVERY == 0 or s == TOTAL - 1:
            aA = _eval(tr.model, tr.task, 0)
            aB = _eval(tr.model, tr.task, 1)
            tr.model.current_task = tr.task.task_of(s)   # restaurar cabeza vigente
            evals.append((s, aA, aB))
    tr.kfac.remove_hooks()

    # --- (1) cuándo se pierde A ---
    pre = [e for e in evals if e[0] < SWITCH]
    post = [e for e in evals if e[0] >= SWITCH]
    aA_pre = pre[-1][1]
    collapse = next((e for e in evals if e[0] >= SWITCH and e[1] < 0.5), None)
    half_life = (collapse[0] - SWITCH) if collapse else None
    melt_step = next((s for s, st in states_hist
                      if s >= SWITCH and any(v == "MELTED" for v in st.values())), None)
    print(f"  acc_A justo antes del switch: {aA_pre:.2f}")
    print(f"  primer MELTED tras el switch: paso {melt_step} (switch={SWITCH})")
    print(f"  acc_A cae bajo 0.5 en el paso {collapse[0] if collapse else '—'} "
          f"({half_life} pasos tras el switch)")
    grad_loss = None
    if collapse:
        between = [e for e in post if e[0] <= collapse[0]]
        grad_loss = len(between) > 2   # ¿hubo evals intermedios con degradación gradual?

    # --- (2) señal per-Linear: spread de trazas dentro de cada bloque ---
    spreads = {}
    for blk in ("b0", "b1"):
        ratios = []
        for s, tra in traces_hist:
            if s < 60:                  # ignora el arranque (factores aún fríos)
                continue
            vals = [v for n, v in tra.items() if n.startswith(blk) and v > 0]
            if len(vals) >= 2:
                ratios.append(max(vals) / min(vals))
        spreads[blk] = (float(np.median(ratios)), float(np.max(ratios))) if ratios else (1.0, 1.0)
        print(f"  spread per-Linear en {blk}: mediana max/min = {spreads[blk][0]:.1f}x "
              f"(pico {spreads[blk][1]:.1f}x)")

    print("\n--- Checks ---")
    print(f"[check] A estaba aprendida antes del switch (acc>0.9): {aA_pre > 0.9}")
    print(f"[check] el melt dispara en el switch (≤15 pasos): "
          f"{melt_step is not None and melt_step - SWITCH <= 15}")
    print(f"[check] señal per-Linear EXISTE (spread mediano >2x en algún bloque): "
          f"{any(sp[0] > 2.0 for sp in spreads.values())}")
    if collapse:
        print(f"[check] colapso de A: {'GRADUAL durante B' if grad_loss else 'INMEDIATO en el melt'} "
              f"(bajo 0.5 a {half_life} pasos del switch)")

    _plot(evals, states_hist, traces_hist)
    _save(evals, spreads, melt_step, half_life, aA_pre)
    return dict(evals=evals, spreads=spreads, melt_step=melt_step, half_life=half_life)


def _plot(evals, states_hist, traces_hist):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    e = np.array(evals)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 3]})
    # panel 1: accs
    ax1.plot(e[:, 0], e[:, 1], "o-", ms=3, label="acc tarea A (lag1, cabeza A)")
    ax1.plot(e[:, 0], e[:, 2], "s-", ms=3, label="acc tarea B (lag2, cabeza B)")
    ax1.axvline(SWITCH, color="red", ls="--", lw=1, label=f"switch A→B ({SWITCH})")
    ax1.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax1.set_ylabel("accuracy"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_title("Fase 10.4 — Dinámica del olvido en Modelo X (multi-cabeza, d=256)")
    # panel 2: estados térmicos por bloque
    for i, blk in enumerate(("b0.out", "b1.out")):
        xs = [s for s, st in states_hist]
        ys = [STATE_CODE.get(st.get(blk, "PLASTIC"), 0) for s, st in states_hist]
        ax2.step(xs, [y + i * 0.1 for y in ys], where="post", lw=1.2, label=blk.split(".")[0])
    ax2.set_yticks([0, 1, 2]); ax2.set_yticklabels(["PLASTIC", "FROZEN", "MELTED"])
    ax2.axvline(SWITCH, color="red", ls="--", lw=1)
    ax2.legend(fontsize=7, loc="upper left"); ax2.grid(alpha=0.3)
    # panel 3: trazas per-Linear (la señal de la Fase 11)
    if traces_hist:
        names = sorted(traces_hist[0][1].keys())
        xs = [s for s, _ in traces_hist]
        for n in names:
            ys = [tra[n] for _, tra in traces_hist]
            blk = n.split(".")[0]
            ax3.plot(xs, ys, lw=1, alpha=0.85, ls="-" if blk == "b0" else "--",
                     label=n)
        ax3.set_yscale("log"); ax3.axvline(SWITCH, color="red", ls="--", lw=1)
        ax3.set_ylabel("Tr(F⁻¹) per-Linear (log)"); ax3.set_xlabel("paso")
        ax3.legend(fontsize=5.5, ncol=4); ax3.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("phase10_dynamics.png", dpi=120)
    print("\n[fig] phase10_dynamics.png")


def _save(evals, spreads, melt_step, half_life, aA_pre):
    out = {
        "acc_A_pre_switch": aA_pre,
        "melt_step": melt_step,
        "steps_to_A_below_0.5": half_life,
        "perlinear_trace_spread": {k: {"median_maxmin": v[0], "peak": v[1]}
                                   for k, v in spreads.items()},
        "evals": [{"step": int(s), "acc_A": a, "acc_B": b} for s, a, b in evals],
    }
    with open("phase10_dynamics.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[json] phase10_dynamics.json")


if __name__ == "__main__":
    run()
