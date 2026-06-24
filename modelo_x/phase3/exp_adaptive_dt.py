"""Fase 3.3 — Paso Δt adaptativo y fidelidad multi-salto.

En un intervalo [0,T] con intensidad λ, el número real de saltos es Poisson(λT)
(media λT). El thinning de UN solo paso con p=min(λT,1) produce a lo sumo 1 salto:
satura y subcuenta cuando λT>1 (justo un Cisne Negro). El submuestreo ADAPTATIVO
parte Δt en N subpasos tales que λ·δt ≤ umbral, recuperando Binomial(N, λT/N) ->
media λT, que tiende a Poisson(λT). En PyTorch el N variable no recompila el grafo,
así que esto es viable aquí (a diferencia de JAX/jit).
"""
from __future__ import annotations

import math

import torch


def count_jumps(lam: float, T: float, mode: str, max_ldt: float = 0.1,
                samples: int = 20000, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if mode == "fixed":
        p = min(lam * T, 1.0)
        return (torch.rand(samples, generator=g) < p).float()
    # adaptive / reference: subpasos
    if mode == "adaptive":
        N = max(1, math.ceil(lam * T / max_ldt))
    else:  # reference (muy fino)
        N = 5000
    dt = T / N
    p = lam * dt
    occ = (torch.rand(samples, N, generator=g) < p).float()
    return occ.sum(dim=1)


def run(out_path: str | None = None):
    T = 1.0
    lams = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    print(f"\n{'λT':>5} {'verdad':>8} {'fijo (1 paso)':>14} {'adaptativo':>12} {'N_sub':>7}")
    print("-" * 52)
    rows = []
    for lam in lams:
        true_mean = lam * T
        fixed = count_jumps(lam, T, "fixed")
        adapt = count_jumps(lam, T, "adaptive")
        N = max(1, math.ceil(lam * T / 0.1))
        rows.append((lam * T, true_mean, fixed.mean().item(), adapt.mean().item(), N))
        print(f"{lam*T:5.1f} {true_mean:8.2f} {fixed.mean().item():14.2f} "
              f"{adapt.mean().item():12.2f} {N:7d}")
    print("\nLectura: el paso fijo satura en ~1 salto (subcuenta severa para λT>1);")
    print("el adaptativo recupera la media de Poisson λT manteniendo λ·δt acotado.")
    if out_path:
        _plot(rows, out_path)
        print(f"\nFigura guardada en {out_path}")
    return rows


def _plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lt = [r[0] for r in rows]
    truth = [r[1] for r in rows]
    fixed = [r[2] for r in rows]
    adapt = [r[3] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lt, truth, "k-", lw=2, label="verdad: Poisson(λT)")
    ax.plot(lt, fixed, "o--", color="#d62728", label="paso fijo (1 subpaso)")
    ax.plot(lt, adapt, "s--", color="#1f77b4", label="adaptativo (λ·δt ≤ 0.1)")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.6, label="techo del paso fijo")
    ax.set_xlabel("λT (saltos esperados)"); ax.set_ylabel("conteo medio de saltos")
    ax.set_title("Fase 3.3 — Fidelidad multi-salto: fijo vs adaptativo")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=110)


if __name__ == "__main__":
    run(out_path="phase3_adaptive_dt.png")
