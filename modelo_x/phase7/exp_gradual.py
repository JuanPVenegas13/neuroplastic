"""Fase 7.2 — Deriva GRADUAL (rampa): el punto ciego del detector de z-score.

El detector dispara la fusión por un SALTO de la pérdida (z-score). Una deriva lenta
(rampa) puede no producir ese salto -> el modelo, ya consolidado/congelado, podría NO
re-plastificarse y quedarse atascado. Se mide: ¿dispara el detector en la rampa? ¿el
modelo termina adaptado al régimen nuevo? Si falla, es un límite honesto y se propone
la alternativa (señal de deriva LENTA además del salto).
"""
from __future__ import annotations

from ..config import Config, get_device
from ..phase5.ablated_trainer import AblatedTrainer
from .multi_drift import GradualDriftTask


def _run(ramp_len, tag):
    dev = get_device()
    cfg = Config()
    R0 = 150; R1 = R0 + ramp_len
    cfg.train.steps = R1 + 130
    cfg.drift.drift_step = R0
    cfg.thermo.freeze_warmup = 95
    cfg.thermo.anneal_steps = cfg.train.steps
    cfg.train.log_every = 10
    tr = AblatedTrainer(cfg, dev, ablation="no_jump")
    tr.task = GradualDriftTask(cfg.model, cfg.drift, dev, ramp_start=R0, ramp_end=R1,
                               seed=cfg.train.seed)
    tr.fit(); tr.kfac.remove_hooks()
    log = tr.log
    fired = sum(1 for r in log if r["jump"] and R0 <= r["step"] <= R1 + 20)
    final = next((r["eval_acc"] for r in reversed(log) if r["eval_acc"] is not None), 0)
    pre = max((r["eval_acc"] for r in log if r["eval_acc"] is not None and r["step"] < R0), default=0)
    print(f"  [{tag}] rampa={ramp_len} pasos: aprende pre={pre:.2f} | "
          f"disparos del detector={fired} | acc final (post)={final:.2f}")
    return fired, final


def run():
    print("\n--- Fase 7.2: deriva gradual (rampa) vs el detector de saltos ---")
    abrupt = _run(1, "abrupta")            # rampa de 1 paso ≈ Cisne Negro
    slow = _run(120, "gradual")            # rampa lenta de 120 pasos

    print(f"\n[check] la deriva abrupta dispara el detector y recupera: "
          f"{abrupt[0] > 0 and abrupt[1] >= 0.9}")
    adapted_slow = slow[1] >= 0.9
    print(f"[check] la deriva gradual termina adaptada: {adapted_slow} "
          f"(disparos={slow[0]})")
    if adapted_slow and slow[0] == 0:
        print("Lectura: la rampa NO dispara el detector (no hay salto de pérdida), pero el")
        print("modelo se adapta igual porque la consolidación aún no había congelado, o la")
        print("plasticidad residual basta. El detector es para Cisnes Negros, no para rampas.")
    elif not adapted_slow:
        print("Lectura (límite honesto): la rampa no dispara la fusión y el modelo consolidado")
        print("queda atascado -> el detector tiene un punto ciego ante deriva lenta. Alternativa:")
        print("añadir una señal de deriva LENTA (tendencia sostenida de la KL de covariables o")
        print("EMA larga de la pérdida) que dispare re-plastificación sin necesidad de un salto.")
    else:
        print(f"Lectura: la rampa disparó el detector {slow[0]} veces (la pérdida sube lo")
        print("suficiente durante la transición) y el modelo se adaptó.")
    return abrupt, slow


if __name__ == "__main__":
    run()
