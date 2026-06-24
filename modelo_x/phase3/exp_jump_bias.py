"""Fase 3.1 — Experimento: sesgo y varianza de STE vs REINFORCE.

Barre la probabilidad de salto p y, para cada una, estima dE[L]/dθ con ambos
estimadores sobre N muestras, comparándolos contra la verdad analítica.
Resultado esperado y a verificar empíricamente:
  * STE: sesgo NO nulo (sistemático), varianza baja.
  * REINFORCE: sesgo ~0 (insesgado), varianza mayor; la línea base la reduce.
"""
from __future__ import annotations

import math

import torch

from .jump_estimators import JumpSetup, ste_grad_samples, reinforce_grad_samples


def run(n_samples: int = 20000, out_path: str | None = None):
    setup = JumpSetup(h0=0.0, target=1.0, mu_j=0.0, sigma_j=1.0)
    thetas = [-2.2, -1.4, -0.85, -0.4, 0.0, 0.4, 0.85, 1.4, 2.2]
    rows = []
    for th in thetas:
        p = 1 / (1 + math.exp(-th))
        truth = setup.true_grad_theta(th)
        ste = ste_grad_samples(setup, th, n_samples)
        rfb = reinforce_grad_samples(setup, th, n_samples, baseline=None)   # con baseline E[L]
        rf0 = reinforce_grad_samples(setup, th, n_samples, baseline=0.0)    # sin baseline
        rows.append({
            "p": p, "truth": truth,
            "ste_mean": ste.mean().item(), "ste_std": ste.std().item(),
            "rfb_mean": rfb.mean().item(), "rfb_std": rfb.std().item(),
            "rf0_std": rf0.std().item(),
        })

    print(f"\n{'p':>6} {'verdad':>9} | {'STE media':>10} {'STE sesgo':>10} {'STE σ':>8} "
          f"| {'RF media':>9} {'RF sesgo':>9} {'RF σ(b)':>8} {'RF σ(0)':>8}")
    print("-" * 95)
    for r in rows:
        ste_bias = r["ste_mean"] - r["truth"]
        rf_bias = r["rfb_mean"] - r["truth"]
        print(f"{r['p']:6.2f} {r['truth']:9.4f} | {r['ste_mean']:10.4f} {ste_bias:10.4f} "
              f"{r['ste_std']:8.3f} | {r['rfb_mean']:9.4f} {rf_bias:9.4f} "
              f"{r['rfb_std']:8.3f} {r['rf0_std']:8.3f}")

    if out_path:
        _plot(rows, out_path)
        print(f"\nFigura guardada en {out_path}")
    return rows


def _plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = [r["p"] for r in rows]
    truth = [r["truth"] for r in rows]
    ste_m = [r["ste_mean"] for r in rows]
    rf_m = [r["rfb_mean"] for r in rows]
    ste_bias = [abs(r["ste_mean"] - r["truth"]) for r in rows]
    rf_bias = [abs(r["rfb_mean"] - r["truth"]) for r in rows]
    ste_sd = [r["ste_std"] for r in rows]
    rfb_sd = [r["rfb_std"] for r in rows]
    rf0_sd = [r["rf0_std"] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(ps, truth, "k-", lw=2, label="verdad")
    ax[0].plot(ps, ste_m, "o--", color="#d62728", label="STE (media)")
    ax[0].plot(ps, rf_m, "s--", color="#1f77b4", label="REINFORCE (media)")
    ax[0].set_xlabel("p (prob. de salto)"); ax[0].set_ylabel("dE[L]/dθ")
    ax[0].set_title("Estimación vs verdad"); ax[0].legend()

    ax[1].plot(ps, ste_bias, "o-", color="#d62728", label="|sesgo| STE")
    ax[1].plot(ps, rf_bias, "s-", color="#1f77b4", label="|sesgo| REINFORCE")
    ax[1].set_xlabel("p"); ax[1].set_ylabel("|sesgo|"); ax[1].set_title("Sesgo")
    ax[1].legend()

    ax[2].plot(ps, ste_sd, "o-", color="#d62728", label="σ STE")
    ax[2].plot(ps, rfb_sd, "s-", color="#1f77b4", label="σ REINFORCE (baseline)")
    ax[2].plot(ps, rf0_sd, "^--", color="#9467bd", label="σ REINFORCE (sin baseline)")
    ax[2].set_xlabel("p"); ax[2].set_ylabel("desv. estándar"); ax[2].set_yscale("log")
    ax[2].set_title("Varianza"); ax[2].legend()
    fig.suptitle("Fase 3.1 — Sesgo y varianza: STE vs REINFORCE", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    run(out_path="phase3_jump_bias.png")
