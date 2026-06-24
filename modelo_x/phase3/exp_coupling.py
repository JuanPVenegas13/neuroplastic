"""Fase 3.2 — Experimento: ¿importa el acoplamiento W_Q–W_K?

Entrena brevemente un Modelo X, mide el ratio de acoplamiento por bloque y cuánto
se desvía la dirección del gradiente natural acoplado respecto del diagonal.
Decisión: si el coseno ≈ 1 y rel_diff ≈ 0, la aproximación diagonal está justificada;
si no, el acoplamiento debe modelarse.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import Config, get_device
from ..model import ModeloX
from ..drift_data import DriftTask
from .coupling import QKCouplingProbe


def run(train_steps: int = 150, probe_batches: int = 20):
    cfg = Config(); dev = get_device(); torch.manual_seed(0)
    m = ModeloX(cfg.model).to(dev)
    for b in m.blocks:
        b.deterministic = True               # aislar la curvatura del ruido del SDE
    task = DriftTask(cfg.model, cfg.drift, dev, seed=0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    # entrenamiento breve para tener un modelo con estructura
    for s in range(train_steps):
        x, y = task.batch(0, 64)
        opt.zero_grad(set_to_none=True)
        logits = m(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.model.vocab_size), y.reshape(-1))
        loss.backward(); opt.step()
    print(f"modelo entrenado: loss={loss.item():.3f}")

    # sondeo de acoplamiento
    probe = QKCouplingProbe(m)
    for _ in range(probe_batches):
        x, y = task.batch(0, 64)
        m.zero_grad(set_to_none=True)
        logits = m(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.model.vocab_size), y.reshape(-1))
        loss.backward()
    ratios = probe.coupling_ratios()

    print("\n--- Acoplamiento de curvatura W_Q–W_K por bloque ---")
    print(f"{'bloque':>7} {'ratio acopl.':>13} {'coseno nat':>11} {'rel_diff':>10}")
    for i in sorted(ratios):
        comp = probe.natural_grad_comparison(i)
        cos = comp["cosine"] if comp else float("nan")
        rel = comp["rel_diff"] if comp else float("nan")
        print(f"{i:>7} {ratios[i]:>13.4f} {cos:>11.4f} {rel:>10.4f}")
    probe.remove()

    print("\nLectura: ratio ~0 y coseno ~1 => la aproximación diagonal por bloques está")
    print("justificada; valores altos => el factor cruzado S_QK debe incorporarse.")
    return ratios


if __name__ == "__main__":
    run()
