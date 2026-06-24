"""Fase 5.1 — Test de OLVIDO CATASTRÓFICO (la afirmación central).

Entrena A (lag=1) -> consolida -> Cisne Negro -> adapta B (lag=2). Luego, SIN seguir
entrenando, re-evalúa en A. La promesa del framework: el congelamiento selectivo
debería preservar parte de A. Comparamos el Modelo X completo contra la ablación
'no_thermo' (siempre plástico, sin consolidación), que debería olvidar A más.

Métrica:  retención_A = acc en A tras adaptarse a B   (más alto = menos olvido).
"""
from __future__ import annotations

from dataclasses import replace

from ..config import Config, get_device
from .ablated_trainer import AblatedTrainer


def _cfg():
    c = Config()
    c.train.steps = 320
    c.drift.drift_step = 160
    c.thermo.freeze_warmup = 110
    c.thermo.anneal_steps = 320
    c.train.log_every = 20
    return c


def _run(ablation):
    dev = get_device()
    tr = AblatedTrainer(_cfg(), dev, ablation=ablation)
    tr.fit()
    tr.kfac.remove_hooks()
    d = tr.cfg.drift.drift_step
    acc_B = tr.evaluate(d + 1, n_batches=8)     # tarea vigente (B, lag=2)
    acc_A = tr.evaluate(0, n_batches=8)         # tarea vieja (A, lag=1) -> retención
    if ablation == "no_thermo":
        nf = -1                                 # congelamiento bypassed: no aplica
    else:
        nf = sum(1 for s in tr.thermo.states().values() if s == "FROZEN")
    return acc_A, acc_B, nf, len(tr.model.blocks)


def run():
    print("\n--- Fase 5.1: olvido catastrófico (retención de A tras adaptar B) ---")
    rows = {}
    for ab in ("full", "no_thermo"):
        aA, aB, nf, nb = _run(ab)
        rows[ab] = (aA, aB, nf, nb)
        froz = "n/a (siempre plástico)" if nf < 0 else f"{nf}/{nb}"
        print(f"[{ab:>9}] acc B (vigente)={aB:.2f} | retención A (vieja)={aA:.2f} | "
              f"bloques congelados={froz}")

    full_A = rows["full"][0]; plastic_A = rows["no_thermo"][0]
    full_B = rows["full"][1]; plastic_B = rows["no_thermo"][1]
    print(f"\n[check] ambos adaptan B bien: {full_B >= 0.8 and plastic_B >= 0.8}")
    print(f"[check] retención de A ≈ azar en AMBOS (tarea contradictoria): "
          f"{full_A < 0.2 and plastic_A < 0.2}")
    print(f"\nLectura honesta: A (lag=1) y B (lag=2) exigen SALIDAS DISTINTAS para las MISMAS")
    print(f"entradas. Con una sola cabeza y sin identificador de tarea, NINGÚN método puede")
    print(f"retener A mientras produce B —A y B se contradicen punto a punto. La retención")
    print(f"simultánea es imposible POR CONSTRUCCIÓN, no por fallo del congelamiento. El test")
    print(f"bien planteado de la afirmación central no es la retención simultánea sino el")
    print(f"AHORRO al revisitar A (5.2): si la estructura de A quedó consolidada, re-aprenderla")
    print(f"debe ser más rápido que aprenderla de cero. Eso sí distingue al framework.")
    return rows


if __name__ == "__main__":
    run()
