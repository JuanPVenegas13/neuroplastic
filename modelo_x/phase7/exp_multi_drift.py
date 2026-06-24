"""Fase 7.1 — Derivas MÚLTIPLES en cadena: ¿se acumula o interfiere la consolidación?

Cadena lag 1 -> 2 -> 3 -> 1. Se mide si el arco sobrevive a cada Cisne Negro (detecta,
funde, recupera) y, crucialmente, si al VOLVER a lag=1 hay AHORRO (la estructura
original quedó recuperable pese a las consolidaciones intermedias) o si las
consolidaciones sucesivas INTERFIRIERON (recuperar lag=1 cuesta como de cero).
"""
from __future__ import annotations

from ..config import Config, get_device
from ..phase5.ablated_trainer import AblatedTrainer, steps_to_acc
from .multi_drift import MultiDriftTask


SEG = 140
SCHED = [(SEG, 1), (2 * SEG, 2), (3 * SEG, 3), (10 ** 9, 1)]   # 1->2->3->1
STEPS = 4 * SEG


def run():
    dev = get_device()
    cfg = Config()
    cfg.train.steps = STEPS
    cfg.drift.drift_step = SEG
    cfg.thermo.freeze_warmup = 90
    cfg.thermo.anneal_steps = STEPS
    cfg.train.log_every = 10
    tr = AblatedTrainer(cfg, dev, ablation="no_jump")     # framework simplificado (Fase 6.2)
    tr.task = MultiDriftTask(cfg.model, cfg.drift, dev, SCHED, seed=cfg.train.seed)
    tr.fit(); tr.kfac.remove_hooks()
    log = tr.log

    print("\n--- Fase 7.1: cadena de derivas lag 1->2->3->1 ---")
    bounds = [SEG, 2 * SEG, 3 * SEG]
    # recuperación tras cada Cisne Negro
    recov = {}
    for i, b in enumerate(bounds):
        det = any(r["jump"] for r in log if b <= r["step"] < b + 30)
        rec = steps_to_acc(log, b, 0.9, end_step=b + SEG)
        seg_final = max((r["eval_acc"] for r in log if b <= r["step"] < b + SEG
                         and r["eval_acc"] is not None), default=0)
        recov[b] = (det, rec, seg_final)
        print(f"  deriva {i+1} (paso {b}, lag->{SCHED[i+1][1]}): detecta={det} | "
              f"recupera en {rec} pasos | acc máx en el tramo={seg_final:.2f}")

    # AHORRO al volver a lag=1: 1ª vez (tramo inicial) vs última (tramo final, lag=1 de nuevo)
    s_first = steps_to_acc(log, 0, 0.9, end_step=SEG)
    s_last = steps_to_acc(log, 3 * SEG, 0.9)
    savings = (s_first - s_last) if (s_first is not None and s_last is not None) else None
    final_acc = next((r["eval_acc"] for r in reversed(log) if r["eval_acc"] is not None), 0)
    print(f"\n  volver a lag=1: 1ª vez={s_first} pasos | última vez={s_last} pasos | "
          f"ahorro={savings} | acc final={final_acc:.2f}")

    survived = all(v[2] >= 0.9 for v in recov.values())
    print(f"\n[check] el arco sobrevive a las 3 derivas (recupera en cada tramo): {survived}")
    print(f"[check] al volver a lag=1 hay ahorro (no interfirió la cadena): "
          f"{savings is not None and savings > 0}")
    if savings is not None and savings > 0:
        print("Lectura: la consolidación se ACUMULA sin interferir: tras pasar por lag 2 y 3,")
        print("revisitar lag 1 sigue siendo más rápido que aprenderlo de cero.")
    else:
        print("Lectura: revisitar lag=1 NO mostró ahorro -> las consolidaciones intermedias")
        print("interfirieron con la estructura original (límite honesto ante cadenas largas).")
    return recov, savings


if __name__ == "__main__":
    run()
