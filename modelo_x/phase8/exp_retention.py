"""Fase 8 — Eje 1: RETENCIÓN SIMULTÁNEA (benchmark task-id, drift de shift absorbible).

Re-abre el head-to-head contra EWC que 6.3 dejó "confundido por el optimizador". El
hallazgo central CORRIGE a 6.3 con honestidad:

  * 6.3 concluyó "Adam NEUTRALIZA a EWC" tras barrer λ ∈ {1e4, 1e6} (ver exp_baselines).
    Pero el Fisher empírico de esta arquitectura es DIMINUTO (~1e-7), así que una
    penalización EWC con efecto necesita λ ~ 1e8. 6.3 se quedó ~100× corto.
  * Empujando λ a la escala del Fisher, INCLUSO EL EWC ACOPLADO (penalización vía Adam)
    MUERDE: a λ=1e8 retiene A casi perfecto (a costa de B). Adam NO neutraliza a EWC;
    el "confound" de 6.3 era un artefacto del rango de λ, no del optimizador.
  * El EWC DESACOPLADO (AdamW-style, post-Adam, coef clamp a [0,1] = soft-freeze por
    curvatura) es la palanca MÁS LIMPIA/ESTABLE, pero llega a la MISMA frontera.

Criterio rector (honestidad): comparar contra Modelo X A IGUAL acc_B. "EWC retiene 0.9"
con acc_B caído NO es victoria.

Resultado: en este benchmark de UNA cabeza la solución conjunta exige condicionar en el
token de tarea; θ*_A lo ignora, así que el anclaje de pesos (EWC) sólo traza un trade-off
DURO (ningún λ con ambos altos). Modelo X (melt->adapt) queda clavado en el punto de
olvido de naive. En ESTE eje, EWC es la herramienta y Modelo X no tiene reclamo (5.1).
"""
from __future__ import annotations

import numpy as np

from ..config import Config, get_device
from ..phase6.taskid_data import TaskIDDriftTask, TaskIDData
from ..phase5.ablated_trainer import AblatedTrainer
from .decoupled_ewc import fresh_model, masked_acc, fisher_diag, train, snapshot

SA = SB = 200
SEEDS = (0, 1, 2)
LAM_GRID = (1e4, 1e6, 1e7, 3e7, 1e8, 2e8, 3e8, 5e8, 1e9, 1e10)
SWITCH, TOTAL = 200, 400


def _consolidate_A(seed):
    dev = get_device()
    m, data = fresh_model(dev, seed)
    train(m, data, 0, SA)
    star = snapshot(m)
    fis_true = fisher_diag(m, data, 0, mode="true")
    fis_emp = fisher_diag(m, data, 0, mode="empirical")
    return m, data, star, fis_true, fis_emp


def _from_A(base):
    mm = type(base)(Config().model).to(get_device())
    mm.load_state_dict(base.state_dict())
    return mm


def _modelox_point(seed):
    """Modelo X (no_jump) sobre task-id: (ret_A, acc_B) con eval enmascarada propia."""
    dev = get_device()
    c = Config()
    c.train.steps = TOTAL; c.train.batch_size = 64; c.train.seed = seed
    c.thermo.freeze_warmup = 100; c.thermo.anneal_steps = TOTAL
    c.drift.drift_step = SWITCH
    tr = AblatedTrainer(c, dev, ablation="no_jump")
    tr.task = TaskIDDriftTask(c.model, c.drift, dev, switch=SWITCH, seed=seed)
    for s in range(c.train.steps):        # sin el print por-paso de fit()
        tr.step(s)
    tr.kfac.remove_hooks()
    ev = TaskIDData(c.model, dev, seed=seed)
    return masked_acc(tr.model, ev, 0), masked_acc(tr.model, ev, 1)


def run():
    print("\n=== Fase 8.1 — Retención simultánea: EWC (acoplado y desacoplado) vs Modelo X ===\n")
    naive, mx = [], []
    coup, dec = {l: [] for l in LAM_GRID}, {l: [] for l in LAM_GRID}
    for seed in SEEDS:
        base, data, star, fis_t, fis_e = _consolidate_A(seed)
        mm = _from_A(base); train(mm, data, 1, SB)
        naive.append((masked_acc(mm, data, 0), masked_acc(mm, data, 1)))
        for lam in LAM_GRID:
            mm = _from_A(base); train(mm, data, 1, SB, coupled=(fis_e, star, lam))   # como 6.3
            coup[lam].append((masked_acc(mm, data, 0), masked_acc(mm, data, 1)))
            mm = _from_A(base); train(mm, data, 1, SB, decoupled=(fis_t, star, lam))  # el fix
            dec[lam].append((masked_acc(mm, data, 0), masked_acc(mm, data, 1)))
        mx.append(_modelox_point(seed))
        print(f"  seed {seed} listo")

    def agg(rows):
        a = np.array(rows); return a[:, 0].mean(), a[:, 1].mean()

    naive_m, mx_m = agg(naive), agg(mx)
    coup_m = {l: agg(coup[l]) for l in LAM_GRID}
    dec_m = {l: agg(dec[l]) for l in LAM_GRID}

    print("\n--- Corrección de 6.3: el EWC ACOPLADO (vía Adam) SÍ muerde a λ alto ---")
    print(f"  {'λ':>8} {'ret_A':>6} {'acc_B':>6}   (6.3 barrió sólo hasta 1e6)")
    for lam in LAM_GRID:
        rA, aB = coup_m[lam]
        flag = "  <- rango de 6.3" if lam <= 1e6 else ("  <- MUERDE" if rA > 0.8 else "")
        print(f"  {lam:8.0e} {rA:6.2f} {aB:6.2f}{flag}")

    print("\n--- Frontera retención_A vs acc_B (EWC DESACOPLADO, Fisher verdadero) ---")
    print(f"  {'λ':>8} {'ret_A':>6} {'acc_B':>6} {'min':>6}")
    for lam in LAM_GRID:
        rA, aB = dec_m[lam]
        print(f"  {lam:8.0e} {rA:6.2f} {aB:6.2f} {min(rA, aB):6.2f}")

    print(f"\n  naive (Adam):       ret_A={naive_m[0]:.2f}  acc_B={naive_m[1]:.2f}")
    print(f"  Modelo X (no_jump): ret_A={mx_m[0]:.2f}  acc_B={mx_m[1]:.2f}")

    iso_lam = min(LAM_GRID, key=lambda l: abs(dec_m[l][1] - mx_m[1]))
    best = max(LAM_GRID, key=lambda l: min(dec_m[l]))
    print(f"\n--- Comparación ISO-acc_B (al acc_B de Modelo X = {mx_m[1]:.2f}) ---")
    print(f"  EWC desacoplado @ acc_B≈{dec_m[iso_lam][1]:.2f} (λ={iso_lam:.0e}): ret_A={dec_m[iso_lam][0]:.2f}")
    print(f"  Modelo X        @ acc_B={mx_m[1]:.2f}: ret_A={mx_m[0]:.2f}")
    print(f"  mejor punto BALANCEADO de EWC: ret_A={dec_m[best][0]:.2f} acc_B={dec_m[best][1]:.2f} (λ={best:.0e})")

    print("\n--- Checks ---")
    print(f"[check] 6.3 era artefacto de rango: acoplado neutralizado a λ=1e6 "
          f"(ret_A={coup_m[1e6][0]:.2f}) pero muerde a λ=1e8 (ret_A={coup_m[1e8][0]:.2f}): "
          f"{coup_m[1e6][0] < 0.2 and coup_m[1e8][0] > 0.8}")
    print(f"[check] EWC bajo Adam SÍ protege A (desacoplado máx ret_A>0.9): "
          f"{max(dec_m[l][0] for l in LAM_GRID) > 0.9}")
    print(f"[check] NO hay retención simultánea (ningún λ con min(ret_A,acc_B)>0.6): "
          f"{max(min(dec_m[l]) for l in LAM_GRID) < 0.6}")
    print(f"[check] Modelo X clavado en olvido en este eje (ret_A<0.1): {mx_m[0] < 0.1}")

    _plot(coup_m, dec_m, naive_m, mx_m)
    return dict(naive=naive_m, modelox=mx_m, coupled=coup_m, decoupled=dec_m)


def _plot(coup_m, dec_m, naive_m, mx_m):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    dc = np.array([dec_m[l] for l in LAM_GRID])
    cp = np.array([coup_m[l] for l in LAM_GRID])
    plt.figure(figsize=(6, 5))
    plt.plot(dc[:, 1], dc[:, 0], "o-", label="EWC desacoplado (AdamW-style)")
    plt.plot(cp[:, 1], cp[:, 0], "s--", color="gray", label="EWC acoplado (vía Adam)")
    plt.scatter([naive_m[1]], [naive_m[0]], c="k", marker="x", s=90, label="naive (Adam)", zorder=5)
    plt.scatter([mx_m[1]], [mx_m[0]], c="red", marker="*", s=240, label="Modelo X (no_jump)", zorder=6)
    plt.xlabel("acc_B (tarea nueva)"); plt.ylabel("retención_A (tarea vieja)")
    plt.title("Fase 8.1 — Frontera retención vs aprender B\n(comparar a IGUAL acc_B)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("phase8_retention_frontier.png", dpi=120)
    print("\n[fig] phase8_retention_frontier.png")


if __name__ == "__main__":
    run()
