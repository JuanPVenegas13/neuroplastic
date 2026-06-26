"""Fase 9 — Bloque (Modelo X) vs por-parámetro (EWC) donde la retención SÍ es alcanzable.

Arquitectura SEPARABLE (tronco compartido + una cabeza por tarea). Mide si el
congelamiento por bloque de Modelo X compite con el soft-freeze por-parámetro de EWC, y
pone a prueba (refutándola) la predicción 'EWC alto -> hard-freeze de Modelo X'. Ver
README_fase9.md para la síntesis honesta.

Correr: python -m modelo_x.phase9.run_phase9
"""
from __future__ import annotations

from . import exp_multihead


def main():
    exp_multihead.run()
    print("\n=== Fase 9 completa. Ver README_fase9.md para la síntesis honesta. ===")


if __name__ == "__main__":
    main()
