"""Fase 10 — ¿Cierra la granularidad sub-bloque la brecha que la Fase 9 dejó abierta?

  10.1 (exp_granularity)    Continuo de granularidad del anclaje (param/row/tensor/block):
                            ¿cuánta granularidad hace falta para retener? Resultado: el
                            acantilado es BLOQUE-específico; per-tensor (per-Linear) ya basta.
  10.2 (exp_channel_freeze) Hard-freeze por CANAL vs por BLOQUE en el mecanismo REAL de
                            Modelo X: la granularidad fina cruza el umbral; el freeze
                            graduado (soft) lo remata.

Correr: python -m modelo_x.phase10.run_phase10   (desde el padre de modelo_x/)
"""
from __future__ import annotations

from . import exp_granularity, exp_channel_freeze


def main():
    exp_granularity.run()
    exp_channel_freeze.run()
    print("\n=== Fase 10 completa. Ver README_fase10.md para la síntesis honesta. ===")


if __name__ == "__main__":
    main()
