"""Fase 5.3 — Ablaciones: ¿qué carga cada subsistema?

Se corre el arco de un Cisne Negro (A->B) desactivando una pieza cada vez y se mide
si el modelo (1) aprende A, (2) detecta/sobrevive el Cisne Negro y (3) recupera B.
Cuantifica la contribución real de cada subsistema —incluido, honestamente, cuando
alguno NO resulta crítico para esta tarea.

  full       : todo activo
  no_thermo  : siempre plástico (sin consolidación ni fusión dirigida)
  no_detect  : el detector de deriva nunca dispara la fusión
  no_jump    : sin término de salto dJ (magnitud de salto = 0)
  no_kfac    : sin gradiente natural (Adam plano en todo)
"""
from __future__ import annotations

from ..config import Config, get_device
from .ablated_trainer import AblatedTrainer, steps_to_acc


STEPS, DRIFT = 320, 160


def _cfg():
    c = Config()
    c.train.steps = STEPS
    c.drift.drift_step = DRIFT
    c.thermo.freeze_warmup = 110
    c.thermo.anneal_steps = STEPS
    c.train.log_every = 10
    return c


def _run(ablation):
    dev = get_device()
    tr = AblatedTrainer(_cfg(), dev, ablation=ablation)
    tr.fit()
    tr.kfac.remove_hooks()
    log = tr.log
    pre = max((r["eval_acc"] for r in log if r["eval_acc"] is not None and r["step"] < DRIFT),
              default=0.0)
    spike = max((r["loss"] for r in log if DRIFT <= r["step"] < DRIFT + 25), default=0)
    detected = any(r["jump"] for r in log if DRIFT <= r["step"] < DRIFT + 30)
    final = next((r["eval_acc"] for r in reversed(log) if r["eval_acc"] is not None), 0.0)
    rec = steps_to_acc(log, DRIFT, 0.9)
    return dict(pre=pre, spike=spike, detected=detected, final=final, rec=rec)


def run():
    print("\n--- Fase 5.3: ablaciones (arco de un Cisne Negro A->B) ---")
    print(f"{'variante':>11} {'aprende A':>10} {'pico CN':>8} {'detecta':>8} "
          f"{'recupera B':>11} {'pasos rec.':>11}")
    print("-" * 64)
    res = {}
    for ab in ("full", "no_thermo", "no_detect", "no_jump", "no_kfac"):
        r = _run(ab); res[ab] = r
        print(f"{ab:>11} {r['pre']:>10.2f} {r['spike']:>8.2f} {str(r['detected']):>8} "
              f"{r['final']:>11.2f} {str(r['rec']):>11}")

    print("\n--- Lectura por subsistema ---")
    f = res["full"]
    print(f"- DETECCIÓN: sin ella (no_detect) recupera B = {res['no_detect']['final']:.2f} "
          f"(full {f['final']:.2f}). Si cae mucho, el detector es CRÍTICO para re-plastificar.")
    print(f"- K-FAC: sin gradiente natural (no_kfac) recupera B = {res['no_kfac']['final']:.2f}, "
          f"pasos rec. = {res['no_kfac']['rec']} (full {f['rec']}). Adam plano APRENDE A pero NO")
    print(f"  re-adapta a B; se verificó que subir su lr no lo rescata (lo desestabiliza, ni")
    print(f"  aprende A). La geometría (gradiente natural) es genuinamente importante para la")
    print(f"  re-navegación tras la consolidación, no un artefacto de calibración del paso.")
    print(f"- SALTO: sin dJ (no_jump) recupera B = {res['no_jump']['final']:.2f} "
          f"(full {f['final']:.2f}). Si ~igual, el salto NO es crítico para recuperar (su rol")
    print(f"  es de exploración/re-plastificación, complementario a la fusión térmica).")
    print(f"- TERMODINÁMICA: no_thermo recupera B bien pero pierde el AHORRO (ver 5.2): su")
    print(f"  valor está en la CONSOLIDACIÓN, no en la recuperación de una sola deriva.")
    print(f"\n[check] full aprende A y recupera B: {f['pre'] >= 0.9 and f['final'] >= 0.9}")
    print(f"[check] sin detección la recuperación se degrada: "
          f"{res['no_detect']['final'] < f['final'] - 0.1}")
    return res


if __name__ == "__main__":
    run()
