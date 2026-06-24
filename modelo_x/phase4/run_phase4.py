"""Orquestador de la Fase 4 — corre las cuatro verificaciones en orden.

  4.1  verify_reinforce      estimador de salto insesgado in-model + varianza
  4.1  exp_learnable_loop    lazo completo con saltos aprendibles (STE/REINFORCE)
  4.2  exp_coupled_precond   preacondicionador acoplado Q-K (selectivo por umbral)
  4.3  exp_cross_curvature   curvatura cruzada entre sublocaciones (medición exacta)
  4.4  exp_scale_real        escalado real con timing y supervivencia del arco

Uso:  PYTHONPATH=/home/claude python -m modelo_x.phase4.run_phase4
"""
from __future__ import annotations

from . import verify_reinforce, exp_learnable_loop, exp_coupled_precond
from . import exp_cross_curvature, exp_scale_real


def main():
    print("=" * 70)
    print("FASE 4 — Integración y escalado (verificación empírica de cada paso)")
    print("=" * 70)

    print("\n### 4.1a — Insesgamiento del estimador de salto en el modelo completo")
    verify_reinforce.run()

    print("\n### 4.1b — Saltos aprendibles en el lazo completo")
    exp_learnable_loop.run()

    print("\n### 4.2 — Preacondicionador acoplado Q-K")
    exp_coupled_precond.run(out_path="phase4_coupled_qk.png")

    print("\n### 4.3 — Curvatura cruzada entre sublocaciones")
    exp_cross_curvature.run()

    print("\n### 4.4 — Escalado real")
    exp_scale_real.run()

    print("\n" + "=" * 70)
    print("FASE 4 COMPLETA — ver README_fase4.md para la síntesis honesta.")
    print("=" * 70)


if __name__ == "__main__":
    main()
