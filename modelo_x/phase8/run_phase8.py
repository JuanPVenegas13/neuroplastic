"""Fase 8 — Head-to-head LIMPIO contra EWC, resuelto el confound del optimizador de 6.3.

  8.1 (exp_retention)  eje RETENCIÓN simultánea (drift de shift, absorbible):
                       corrige a 6.3 (su "Adam neutraliza EWC" era artefacto del rango
                       de λ) y traza la frontera retención_A vs acc_B; comparación
                       iso-acc_B contra Modelo X.
  8.2 (exp_savings)    eje AHORRO al revisitar (drift de lag, estructural): el terreno
                       propio de Modelo X, donde EWC (atado a Adam) no puede entrar.

Correr: python -m modelo_x.phase8.run_phase8
"""
from __future__ import annotations

from . import exp_retention, exp_savings


def main():
    exp_retention.run()
    exp_savings.run()
    print("\n=== Fase 8 completa. Ver README_fase8.md para la síntesis honesta. ===")


if __name__ == "__main__":
    main()
