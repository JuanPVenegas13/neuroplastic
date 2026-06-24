"""Fase 4.1 — Verificación: ¿el estimador integrado en el núcleo es insesgado?

Comprueba EN EL MODELO COMPLETO (1 capa, para tener un único jump_logit) que:
  * REINFORCE produce un gradiente de dE[L]/d(jump_logit) que coincide con la
    verdad por diferencias finitas (insesgado).
  * El STE produce un gradiente sesgado respecto de esa verdad.
La verdad por DF se estima promediando la pérdida estocástica (solo el salto es
aleatorio; la difusión se apaga) sobre muchas pasadas, con el mismo lote.
"""
from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from ..config import Config, ModelConfig, get_device
from ..model import ModeloX
from ..drift_data import DriftTask


def _build(jump_grad: str, d_model=16, seq_len=6, jump_shared=False):
    mcfg = ModelConfig(n_layers=1, d_model=d_model, n_heads=2, seq_len=seq_len, vocab_size=16,
                       learn_jump=True, jump_grad=jump_grad, jump_sigma=1.0,
                       jump_logit_init=-1.0, jump_shared=jump_shared)
    cfg = replace(Config(), model=mcfg)
    dev = get_device(); torch.manual_seed(0)
    m = ModeloX(mcfg).to(dev)
    # entrena un poco (determinista) para no estar en init
    for b in m.blocks:
        b.deterministic = True
    task = DriftTask(mcfg, cfg.drift, dev, seed=0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(80):
        x, y = task.batch(0, 32)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x).reshape(-1, mcfg.vocab_size), y.reshape(-1))
        loss.backward(); opt.step()
    # activa los saltos: no determinista, sin difusión (aísla el salto), lam>0
    for b in m.blocks:
        b.deterministic = False
        b.sigma_epist = 0.0
        b.sigma_gate = 0.05
        b.lam = 1.0
        b.state = "PLASTIC"
    return m, task, mcfg


def _loss_on(m, x, y, V):
    return F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))


def finite_diff_truth(m, x, y, V, eps=0.5, n=1200):
    logit0 = m.blocks[0].sde.jump_logit.data.clone()
    def meanL(val):
        m.blocks[0].sde.jump_logit.data = val
        with torch.no_grad():
            return torch.stack([_loss_on(m, x, y, V) for _ in range(n)]).mean().item()
    lp = meanL(logit0 + eps); lm = meanL(logit0 - eps)
    m.blocks[0].sde.jump_logit.data = logit0
    return (lp - lm) / (2 * eps)


def estimator_grad(m, x, y, V, mode, n=1500):
    """Promedio del gradiente estimado de dE[L]/d(jump_logit) sobre n muestras."""
    grads = []
    ema_b = None
    jl = m.blocks[0].sde.jump_logit
    for _ in range(n):
        m.zero_grad(set_to_none=True)
        loss = _loss_on(m, x, y, V)
        if mode == "reinforce":
            logp = m.jump_logp_sum()
            b = loss.item() if ema_b is None else ema_b
            surrogate = (loss.detach() - b) * logp
            surrogate.backward()
            ema_b = 0.99 * (ema_b if ema_b is not None else loss.item()) + 0.01 * loss.item()
        else:  # ste
            loss.backward()
        grads.append(jl.grad.item())
    g = torch.tensor(grads)
    return g.mean().item(), g.std().item() / (n ** 0.5), g.std().item()  # media, EE, std


def run():
    rows = []
    for mode in ("ste", "reinforce"):
        m, task, mcfg = _build(mode)
        x, y = task.batch(0, 32); V = mcfg.vocab_size
        truth = finite_diff_truth(m, x, y, V)
        mean, se, std = estimator_grad(m, x, y, V, mode)
        rows.append((mode + " (elem)", truth, mean, se, std))
    # REINFORCE con decisión COMPARTIDA por posición (reducción de varianza)
    m, task, mcfg = _build("reinforce", jump_shared=True)
    x, y = task.batch(0, 32); V = mcfg.vocab_size
    truth = finite_diff_truth(m, x, y, V)
    mean, se, std = estimator_grad(m, x, y, V, "reinforce")
    rows.append(("reinforce (compart.)", truth, mean, se, std))

    print("\n--- Verificación 4.1: gradiente de dE[L]/d(jump_logit) ---")
    print(f"{'modo':>21} {'verdad':>9} {'estim.':>9} {'sesgo':>9} {'±EE':>8} {'std/muestra':>12}")
    print("-" * 74)
    for name, truth, mean, se, std in rows:
        print(f"{name:>21} {truth:9.4f} {mean:9.4f} {mean-truth:+9.4f} {se:8.4f} {std:12.4f}")

    ste, rfe, rfs = rows[0], rows[1], rows[2]
    ste_biased = abs(ste[2] - ste[1]) > 3 * ste[3]
    rf_unbiased = abs(rfe[2] - rfe[1]) < 3 * rfe[3] + 1e-4
    var_reduced = rfs[4] < rfe[4]
    print(f"\n[check] STE tiene sesgo estadísticamente significativo: {ste_biased}")
    print(f"[check] REINFORCE (elem) es consistente con la verdad (insesgado): {rf_unbiased}")
    print(f"[check] REINFORCE compartido reduce varianza: {var_reduced} "
          f"({rfe[4]:.3f} -> {rfs[4]:.3f}, ×{rfe[4]/max(rfs[4],1e-9):.1f})")
    print(f"[check] STE mucho menos ruidoso que REINFORCE: std {ste[4]:.4f} vs {rfe[4]:.3f}")
    return rows


if __name__ == "__main__":
    run()
