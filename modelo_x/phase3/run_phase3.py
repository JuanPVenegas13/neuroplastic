"""Fase 3 — Orquestador: corre los cuatro experimentos y guarda figuras + tablas.

Uso:  python -m modelo_x.phase3.run_phase3
"""
from __future__ import annotations

import os

from . import exp_jump_bias, exp_jump_learning, exp_coupling, exp_adaptive_dt, exp_scale


def main():
    out = os.environ.get("MODELO_X_OUT", ".")
    p = lambda name: os.path.join(out, name)

    print("=" * 70)
    print("FASE 3.1a — Sesgo y varianza del gradiente del salto (STE vs REINFORCE)")
    print("=" * 70)
    exp_jump_bias.run(out_path=p("phase3_jump_bias.png"))

    print("\n" + "=" * 70)
    print("FASE 3.1b — Consecuencia: ¿el sesgo lleva a una política equivocada?")
    print("=" * 70)
    exp_jump_learning.run(out_path=p("phase3_jump_learning.png"))

    print("\n" + "=" * 70)
    print("FASE 3.2 — Curvatura fuera de la diagonal W_Q–W_K")
    print("=" * 70)
    exp_coupling.run()

    print("\n" + "=" * 70)
    print("FASE 3.3 — Paso Δt adaptativo y fidelidad multi-salto")
    print("=" * 70)
    exp_adaptive_dt.run(out_path=p("phase3_adaptive_dt.png"))

    print("\n" + "=" * 70)
    print("FASE 3.4 — Submuestreo de tokens en K-FAC (4 capas, T=64)")
    print("=" * 70)
    exp_scale.run(out_path=p("phase3_subsample.png"))

    print("\nFase 3 completa.")


if __name__ == "__main__":
    main()
