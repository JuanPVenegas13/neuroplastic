"""Orquestador de la Fase 5 — validación de la afirmación central.

  5.1  exp_forgetting   olvido catastrófico (por qué la retención simultánea está mal planteada)
  5.2  exp_reversible   deriva reversible A->B->A y AHORRO (el test bien planteado)
  5.3  exp_ablations    contribución de cada subsistema (qué carga cada pieza)

Uso:  PYTHONPATH=/home/claude python -m modelo_x.phase5.run_phase5
"""
from __future__ import annotations

from . import exp_forgetting, exp_reversible, exp_ablations


def main():
    print("=" * 70)
    print("FASE 5 — Validación de la afirmación central (neuroplasticidad sin")
    print("         olvido catastrófico) con tareas reversibles y ablaciones")
    print("=" * 70)

    print("\n### 5.1 — Olvido catastrófico (encuadre del test)")
    exp_forgetting.run()

    print("\n### 5.2 — Deriva reversible A->B->A y ahorro")
    exp_reversible.run()

    print("\n### 5.3 — Ablaciones por subsistema")
    exp_ablations.run()

    print("\n" + "=" * 70)
    print("FASE 5 COMPLETA — ver README_fase5.md para la síntesis honesta.")
    print("=" * 70)


if __name__ == "__main__":
    main()
