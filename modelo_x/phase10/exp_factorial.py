"""Fase 10.3 — Factorial GRANULARIDAD × DUREZA: el cuadro de decisión para la Fase 11.

10.1 y 10.2 sugirieron que la limitación de Modelo X es doble (grano de bloque Y freeze
binario), pero cada experimento barrió UN eje con el otro fijo, y las celdas hard@param
y hard@tensor nunca se midieron. Aquí se mide el factorial completo 2×4 con presupuestos
comparables, para que el patrón (¿qué factor domina?, ¿interactúan?) sea visible de una
vez y sirva de base para proponer el curso de acción:

    mecanismo   : hard (máscara binaria de gradiente, top-τ params por importancia)
                  soft (anclaje graduado EWC-desacoplado, fuerza lr·λ·F clampada)
    granularidad: param | row (canal) | tensor (Linear) | block

Cada celda = MEJOR min(ret_A, acc_B) sobre su barrido (λ para soft, τ para hard),
media de 3 semillas, en el benchmark multi-cabeza d=256 (Fase 9). Salidas:
`phase10_factorial.png` (gráfica) y `phase10_factorial.json` (todas las celdas y grids,
para inspección posterior).
"""
from __future__ import annotations

import json

import numpy as np

from ..config import get_device
from ..phase9.exp_multihead import SEEDS, _modelox
from . import exp_granularity as gran
from . import exp_channel_freeze as chan

GRAINS = ("param", "row", "tensor", "block")
SOFT_LAMS = (1e9, 3e9, 1e10)          # alrededor del óptimo conocido (10.1)
HARD_TAUS = (0.5, 0.65, 0.8, 0.9)     # alrededor del óptimo conocido (10.2)
D_MODEL = 256


def _agg(rows):
    a = np.array(rows); return float(a[:, 0].mean()), float(a[:, 1].mean())


def _cell(mech, grain, verbose=True):
    """Barrido de la celda; devuelve (grid {k: (retA,accB)}, mejor k por min)."""
    if mech == "soft":
        grid = {lam: _agg([gran._seq(s, lam, grain) for s in SEEDS]) for lam in SOFT_LAMS}
    else:
        lvl = "channel" if grain == "row" else grain
        grid = {tau: _agg([chan._seq(s, tau, lvl) for s in SEEDS]) for tau in HARD_TAUS}
    best = max(grid, key=lambda k: min(grid[k]))
    if verbose:
        rA, aB = grid[best]
        print(f"    [{mech:>4} × {grain:<6}] mejor min={min(rA, aB):.2f} "
              f"(ret_A={rA:.2f} acc_B={aB:.2f} @ {best:.2g})")
    return grid, best


def run():
    print("\n=== Fase 10.3 — Factorial granularidad × dureza (d=256, 3 semillas) ===\n")
    res = {}
    for mech in ("hard", "soft"):
        for grain in GRAINS:
            res[(mech, grain)] = _cell(mech, grain)
    mx = _agg([_modelox(s, D_MODEL, "no_jump") for s in SEEDS])
    print(f"    [Modelo X real     ] min={min(mx):.2f} (ret_A={mx[0]:.2f} acc_B={mx[1]:.2f})")

    mins = {k: min(res[k][0][res[k][1]]) for k in res}
    print("\n--- Matriz min(ret_A, acc_B) [mejor sobre el barrido] ---")
    print(f"  {'':>6} " + " ".join(f"{g:>7}" for g in GRAINS))
    for mech in ("hard", "soft"):
        print(f"  {mech:>6} " + " ".join(f"{mins[(mech, g)]:7.2f}" for g in GRAINS))

    # efectos principales e interacción (sobre los mins de la matriz)
    sub = [g for g in GRAINS if g != "block"]
    soft_gain_sub = np.mean([mins[("soft", g)] - mins[("hard", g)] for g in sub])
    grain_gain_hard = np.mean([mins[("hard", g)] for g in sub]) - mins[("hard", "block")]
    grain_gain_soft = np.mean([mins[("soft", g)] for g in sub]) - mins[("soft", "block")]
    print(f"\n  efecto DUREZA (soft−hard, media sub-bloque): {soft_gain_sub:+.2f}")
    print(f"  efecto GRANO (sub-bloque−block) en hard: {grain_gain_hard:+.2f} | en soft: {grain_gain_soft:+.2f}")

    print("\n--- Checks ---")
    print(f"[check] block es el peor grano en AMBOS mecanismos: "
          f"{all(mins[(m, 'block')] <= min(mins[(m, g)] for g in sub) + 0.02 for m in ('hard', 'soft'))}")
    print(f"[check] soft ≥ hard en todos los granos sub-bloque: "
          f"{all(mins[('soft', g)] >= mins[('hard', g)] - 0.02 for g in sub)}")
    print(f"[check] la mejor celda combina sub-bloque + soft: "
          f"{max(mins, key=mins.get)[0] == 'soft' and max(mins, key=mins.get)[1] != 'block'}")
    print(f"[check] Modelo X real queda por debajo de TODA celda sub-bloque: "
          f"{min(mx) < min(mins[(m, g)] for m in ('hard', 'soft') for g in sub)}")

    _save(res, mins, mx)
    _plot(mins, mx)
    return dict(res=res, mins=mins, modelox=mx)


def _save(res, mins, mx):
    out = {"modelox_real": {"ret_A": mx[0], "acc_B": mx[1], "min": min(mx)},
           "cells": {}}
    for (mech, grain), (grid, best) in res.items():
        out["cells"][f"{mech}.{grain}"] = {
            "grid": {f"{k:.3g}": {"ret_A": v[0], "acc_B": v[1], "min": min(v)}
                     for k, v in grid.items()},
            "best_knob": float(best),
            "best_min": mins[(mech, grain)],
        }
    with open("phase10_factorial.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n[json] phase10_factorial.json")


def _plot(mins, mx):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    x = np.arange(len(GRAINS)); w = 0.36
    hard = [mins[("hard", g)] for g in GRAINS]
    soft = [mins[("soft", g)] for g in GRAINS]
    plt.figure(figsize=(7, 4.6))
    bh = plt.bar(x - w / 2, hard, w, label="hard (máscara binaria)", color="#7a9cc6")
    bs = plt.bar(x + w / 2, soft, w, label="soft (anclaje graduado)", color="#d98e63")
    for bars in (bh, bs):
        for b in bars:
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                     f"{b.get_height():.2f}", ha="center", fontsize=8)
    plt.axhline(min(mx), color="red", ls="--", lw=1.2,
                label=f"Modelo X real (bloque+hard) = {min(mx):.2f}")
    plt.axhline(0.5, color="gray", ls=":", alpha=0.6, label="umbral retención simultánea")
    plt.xticks(x, ["param", "row/canal", "tensor\n(per-Linear)", "block"])
    plt.ylabel("min(ret_A, acc_B) — mejor sobre λ/τ")
    plt.title("Fase 10.3 — Factorial granularidad × dureza (d=256, 3 semillas)")
    plt.ylim(0, 0.85); plt.legend(fontsize=8, loc="upper right")
    plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig("phase10_factorial.png", dpi=120)
    print("[fig] phase10_factorial.png")


if __name__ == "__main__":
    run()
