"""Fase 3.1 (consecuencia) — ¿El sesgo del STE lleva a un óptimo equivocado?

Tarea con óptimo INTERIOR: en K subpasos se decide saltar con prob p; el conteo
J ~ Binomial(K, p). La pérdida iguala J a un objetivo J*:
    E[L](p) = Var[J] + (E[J]-J*)² = K p(1-p) + (K p - J*)²
que es cuadrática en p con mínimo interior p* = (2J*-1) / (2(K-1)).

Optimizamos el logit θ (p=σ(θ)) por descenso de gradiente usando el gradiente
estimado por STE o por REINFORCE, y comparamos el p alcanzado contra p*.
Predicción: STE -> p sesgado; REINFORCE -> p*.
"""
from __future__ import annotations

import math

import torch


def p_star(K: int, Jstar: float) -> float:
    return (2 * Jstar - 1) / (2 * (K - 1))


def expected_loss(p: float, K: int, Jstar: float) -> float:
    return K * p * (1 - p) + (K * p - Jstar) ** 2


def _sample(p, K, gen):
    u = torch.rand(K, generator=gen)
    return (u < p).float()           # occur_i


def optimize(estimator: str, K=10, Jstar=3.0, steps=4000, lr=0.02,
             temp=0.1, seed=0, batch=64):
    gen = torch.Generator().manual_seed(seed)
    theta = torch.tensor(0.0)        # arranca en p=0.5
    traj = []
    ema_b = None                     # línea base para REINFORCE
    for t in range(steps):
        p = torch.sigmoid(theta)
        grads = []
        for _ in range(batch):
            occur = _sample(p, K, gen)
            J = occur.sum()
            L = (J - Jstar) ** 2
            if estimator == "ste":
                # surrogate: dJ/dθ = Σ dgate_i/dscore · dp/dθ ;  dL/dJ = 2(J-J*)
                score = p - torch.rand(K, generator=gen)  # ruido surrogate independiente
                sig = torch.sigmoid(score / temp)
                dJ_dtheta = (sig * (1 - sig) / temp * p * (1 - p)).sum()
                grads.append((2 * (J - Jstar) * dJ_dtheta).item())
            else:  # reinforce
                b = L.item() if ema_b is None else ema_b
                score_fn = (occur - p).sum()        # d logP(J)/dθ
                grads.append(((L - b) * score_fn).item())
                ema_b = 0.99 * (ema_b if ema_b is not None else L.item()) + 0.01 * L.item()
        g = sum(grads) / len(grads)
        theta = theta - lr * g
        if t % 50 == 0 or t == steps - 1:
            traj.append((t, torch.sigmoid(theta).item()))
    return torch.sigmoid(theta).item(), traj


def run(out_path: str | None = None, seeds=(0, 1, 2, 3, 4)):
    K, Jstar = 10, 3.0
    ps = p_star(K, Jstar)
    print(f"\nÓptimo analítico p* = {ps:.4f}  (K={K}, J*={Jstar})\n")
    res = {"ste": [], "reinforce": []}
    trajs = {"ste": None, "reinforce": None}
    for est in ("ste", "reinforce"):
        for s in seeds:
            pf, tj = optimize(est, K=K, Jstar=Jstar, seed=s)
            res[est].append(pf)
            if s == 0:
                trajs[est] = tj
    for est in ("ste", "reinforce"):
        arr = torch.tensor(res[est])
        bias = arr.mean().item() - ps
        print(f"{est:>10}: p_final = {arr.mean():.4f} ± {arr.std():.4f}  "
              f"| sesgo vs p* = {bias:+.4f}")
    if out_path:
        _plot(trajs, ps, out_path)
        print(f"\nFigura guardada en {out_path}")
    return res, ps


def _plot(trajs, ps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(ps, color="k", ls="--", lw=2, label=f"p* = {ps:.3f} (óptimo)")
    for est, color in (("ste", "#d62728"), ("reinforce", "#1f77b4")):
        xs = [t for t, _ in trajs[est]]
        ys = [p for _, p in trajs[est]]
        ax.plot(xs, ys, color=color, label=est.upper())
    ax.set_xlabel("paso de optimización"); ax.set_ylabel("p aprendido")
    ax.set_title("Fase 3.1 — Convergencia de la política de salto")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=110)


if __name__ == "__main__":
    run(out_path="phase3_jump_learning.png")
