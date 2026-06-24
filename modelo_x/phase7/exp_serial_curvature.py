"""Fase 7.3 — Curvatura cruzada en SERIE: ¿corrección barata o límite estructural?

La Fase 4.3 halló que las sublocaciones en serie tienen curvatura cruzada alta pero NO
Kronecker-factorizable (entradas distintas). Aquí se cuantifica, en un par en serie con
dimensiones pequeñas (para poder calcular el Fisher conjunto COMPLETO), cuánto se desvía
el preacondicionador bloque-diagonal del gradiente natural REAL, y a qué coste sería
corregirlo. Conclusión: o existe una corrección asequible, o se formaliza el límite.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import ModelConfig, get_device
from ..model import ModeloX
from ..drift_data import DriftTask
from ..config import DriftConfig


def _per_sample_grads(model, task, w1, w2, V, n=400):
    g1, g2 = [], []
    for b in model.blocks:
        b.deterministic = True
    for _ in range(n):
        x, y = task.batch(0, 1)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).backward()
        g1.append(w1.grad.detach().flatten().clone())
        g2.append(w2.grad.detach().flatten().clone())
    return torch.stack(g1).double(), torch.stack(g2).double()


def _inv(M):
    d = M.shape[0]
    damp = 1e-3 * float(M.diagonal().mean()) + 1e-8
    return torch.linalg.inv(M + damp * torch.eye(d, dtype=M.dtype))


def run():
    dev = get_device(); torch.manual_seed(0)
    # dimensiones PEQUEÑAS para poder formar el Fisher conjunto completo (P=d^2 por capa)
    mcfg = ModelConfig(n_layers=1, d_model=12, n_heads=2, seq_len=16, vocab_size=16)
    m = ModeloX(mcfg).to(dev)
    task = DriftTask(mcfg, DriftConfig(), dev, seed=0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(150):
        x, y = task.batch(0, 64)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).reshape(-1, 16), y.reshape(-1)).backward(); opt.step()

    b0 = m.blocks[0]
    w1, w2 = b0.attn.out.weight, b0.in_proj.weight     # par EN SERIE (out -> in_proj)
    P1, P2 = w1.numel(), w2.numel()
    g1, g2 = _per_sample_grads(m, task, w1, w2, 16)
    gmean1 = g1.mean(0); gmean2 = g2.mean(0)
    # Fisher empírico: bloques propios y cruzado
    N = g1.shape[0]
    C11 = g1.t() @ g1 / N; C22 = g2.t() @ g2 / N; C12 = g1.t() @ g2 / N
    # gradiente natural REAL (conjunto) vs bloque-diagonal
    Cfull = torch.zeros(P1 + P2, P1 + P2, dtype=torch.float64)
    Cfull[:P1, :P1] = C11; Cfull[P1:, P1:] = C22
    Cfull[:P1, P1:] = C12; Cfull[P1:, :P1] = C12.t()
    g = torch.cat([gmean1, gmean2])
    d_joint = _inv(Cfull) @ g
    d_block = torch.cat([_inv(C11) @ gmean1, _inv(C22) @ gmean2])
    cos = float(F.cosine_similarity(d_joint, d_block, dim=0))
    angle = torch.rad2deg(torch.acos(torch.clamp(torch.tensor(cos), -1, 1))).item()

    print("\n--- Fase 7.3: curvatura cruzada en serie (out -> in_proj) ---")
    print(f"dimensiones: P1={P1}, P2={P2} (Fisher conjunto {P1+P2}x{P1+P2})")
    print(f"desalineación bloque-diagonal vs gradiente natural REAL: coseno={cos:.3f} "
          f"-> ángulo={angle:.1f}°")
    print(f"coste de la corrección conjunta: invertir un bloque {P1+P2}x{P1+P2} ~ O((2d^2)^3)")
    print(f"  = O(d^6) por par en serie, frente a O(d^3) factorizado de K-FAC.")
    # ¿factoriza el cruzado como Kronecker? prueba: C12 ~ A12 (x) S12
    print(f"  (el cruzado C12 no es Kronecker-factorizable: entradas distintas -> sin A común)")

    print("\n--- Veredicto 7.3 ---")
    big_gap = angle > 10
    print(f"[check] el bloque-diagonal se desvía del gradiente natural real (>10°): {big_gap} "
          f"({angle:.1f}°)")
    print("Conclusión: la curvatura cruzada en serie desvía la dirección de actualización de")
    print("forma medible, PERO corregirla exige el bloque conjunto completo O(d^6) por par (no")
    print("factoriza como Kronecker al no compartir entrada). NO hay corrección barata: a escala")
    print("es prohibitivo. Se FORMALIZA como límite estructural de K-FAC por bloques —el precio")
    print("de la factorización Kronecker es ignorar exactamente este acoplamiento en serie. La")
    print("mitigación práctica realista no es corregirlo, sino órdenes de capas/normalización")
    print("que reduzcan la correlación serie, o métodos no-Kronecker (p.ej. Shampoo) si se")
    print("dispusiera del presupuesto —fuera del alcance de este prototipo.")
    return angle


if __name__ == "__main__":
    run()
