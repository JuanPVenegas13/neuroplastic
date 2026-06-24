"""Fase 5.2 — Deriva reversible A->B->A y AHORRO (el test bien planteado).

Si el congelamiento consolida la estructura de A, al volver a A (tras pasar por B)
re-aprenderla debe costar MENOS pasos que aprenderla de cero. Comparamos el "ahorro"
del Modelo X completo contra la ablación 'no_thermo' (siempre plástico): si el
congelamiento aporta, el completo debe ahorrar MÁS.

  ahorro = pasos_para_A_la_1a_vez  -  pasos_para_A_la_2a_vez   (positivo = más rápido)
"""
from __future__ import annotations

from ..config import Config, get_device
from ..config import ModelConfig
from dataclasses import replace
from .ablated_trainer import AblatedTrainer, ReversibleTask, steps_to_acc


D1, D2, STEPS = 150, 320, 470
TARGET = 0.9


def _cfg():
    c = Config()
    c.train.steps = STEPS
    c.drift.drift_step = D1            # el detector usa drift_step solo para covariate; el melt
                                       #  se dispara por el salto de pérdida (sirve en D1 y D2)
    c.thermo.freeze_warmup = 100
    c.thermo.anneal_steps = STEPS
    c.train.log_every = 10
    return c


def _run(ablation):
    dev = get_device()
    cfg = _cfg()
    tr = AblatedTrainer(cfg, dev, ablation=ablation)
    tr.task = ReversibleTask(cfg.model, cfg.drift, dev, d1=D1, d2=D2, seed=cfg.train.seed)
    tr.fit()
    tr.kfac.remove_hooks()
    s_A1 = steps_to_acc(tr.log, 0, TARGET, end_step=D1)
    s_A2 = steps_to_acc(tr.log, D2, TARGET)
    # acc de A al final (recuperada)
    accA_final = next((r["eval_acc"] for r in reversed(tr.log) if r["eval_acc"] is not None), None)
    return s_A1, s_A2, accA_final


def run():
    print("\n--- Fase 5.2: deriva reversible A->B->A, ahorro al revisitar A ---")
    print(f"calendario: A en [0,{D1}) -> B en [{D1},{D2}) -> A en [{D2},{STEPS})  | objetivo acc>={TARGET}")
    rows = {}
    for ab in ("full", "no_thermo"):
        s1, s2, af = _run(ab)
        rows[ab] = (s1, s2, af)
        sv = (s1 - s2) if (s1 is not None and s2 is not None) else None
        print(f"[{ab:>9}] A 1ª vez={s1} pasos | A 2ª vez={s2} pasos | "
              f"ahorro={sv} | acc A final={af:.2f}")

    f1, f2, _ = rows["full"]; p1, p2, _ = rows["no_thermo"]
    full_sav = (f1 - f2) if (f1 and f2) else None
    plastic_sav = (p1 - p2) if (p1 and p2) else None
    print(f"\n[check] el completo recupera A más rápido la 2ª vez (ahorro>0): "
          f"{full_sav is not None and full_sav > 0}")
    if full_sav is not None and plastic_sav is not None:
        print(f"[check] el completo ahorra MÁS que el siempre-plástico: "
              f"{full_sav > plastic_sav} (full={full_sav} vs plástico={plastic_sav})")
    print(f"\nLectura: el ahorro positivo indica que la consolidación dejó la estructura de A")
    print(f"recuperable; si el completo ahorra más que el plástico, el congelamiento selectivo")
    print(f"es lo que la preserva (y no solo que los pesos no estén aleatorios).")
    return rows


if __name__ == "__main__":
    run()
