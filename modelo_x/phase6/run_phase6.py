"""Orquestador de la Fase 6 — validez externa y navaja de Occam.

  6.1  exp_jump_value   ¿gana el salto dJ su lugar? (esfuerzo honesto antes de cortar)
  6.2  exp_razor        navaja: el framework sin dJ conserva arco y ahorro
  6.3  exp_baselines    Modelo X vs EWC/naive (continual con token de tarea)

Uso:  PYTHONPATH=/home/claude python -m modelo_x.phase6.run_phase6
"""
from __future__ import annotations

from . import exp_jump_value, exp_razor, exp_baselines


def main():
    print("=" * 70)
    print("FASE 6 — Validez externa (vs literatura) y navaja de Occam (salto dJ)")
    print("=" * 70)

    print("\n### 6.1 — ¿El salto dJ gana su lugar?")
    exp_jump_value.run()

    print("\n### 6.2 — Navaja de Occam: framework sin dJ")
    exp_razor.run()

    print("\n### 6.3 — Modelo X vs EWC / naive")
    exp_baselines.run()

    print("\n" + "=" * 70)
    print("FASE 6 COMPLETA — ver README_fase6.md.")
    print("=" * 70)


if __name__ == "__main__":
    main()
