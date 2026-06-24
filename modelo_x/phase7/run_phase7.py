"""Orquestador de la Fase 7 — robustez ante derivas y cabos sueltos.

  7.1  exp_multi_drift       cadena 1->2->3->1 (¿se acumula o interfiere la consolidación?)
  7.2  exp_gradual           deriva gradual (rampa) vs el detector de saltos
  7.3  exp_serial_curvature  curvatura cruzada en serie (4.3): corrección vs límite
  7.4  exp_device            portabilidad / preparación para Apple Silicon (MPS)

Uso:  PYTHONPATH=/home/claude python -m modelo_x.phase7.run_phase7
"""
from __future__ import annotations

from . import exp_multi_drift, exp_gradual, exp_serial_curvature, exp_device


def main():
    print("=" * 70)
    print("FASE 7 — Robustez ante derivas múltiples/graduales + cabos teóricos/plataforma")
    print("=" * 70)

    print("\n### 7.1 — Derivas múltiples en cadena")
    exp_multi_drift.run()

    print("\n### 7.2 — Deriva gradual")
    exp_gradual.run()

    print("\n### 7.3 — Curvatura cruzada en serie")
    exp_serial_curvature.run()

    print("\n### 7.4 — Portabilidad de dispositivo (MPS)")
    exp_device.run()

    print("\n" + "=" * 70)
    print("FASE 7 COMPLETA — ver README_fase7.md.")
    print("=" * 70)


if __name__ == "__main__":
    main()
