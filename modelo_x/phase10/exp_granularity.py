"""Fase 10.1 — El CONTINUO de granularidad: ¿dónde se cierra la brecha?

La Fase 9 mostró los dos extremos en el benchmark multi-cabeza (d=256): el anclaje
por-PARÁMETRO logra retención simultánea (min≈0.65) y el engrosado a BLOQUE la pierde
(min≈0.25, como Modelo X). Pero "bloque" y "parámetro" son los extremos; el cabo abierto
era: ¿basta una granularidad SUB-BLOQUE (por fila/canal) para recuperar la retención, o
hace falta el grano fino de parámetro?

Aquí barremos el continuo, aislando la granularidad como ÚNICA variable: se ancla con el
MISMO mecanismo (EWC desacoplado, Fisher verdadero) pero engrosando el Fisher a:

  param  -> por elemento (grano fino, = Fase 9)
  row    -> por FILA de salida (cada neurona/canal comparte un Fisher)  <- sub-bloque
  tensor -> por matriz de pesos (cada Linear comparte un Fisher)
  block  -> por bloque del Transformer (= mímica de Modelo X, Fase 9)

Monotonía de coarseness en las matrices del tronco (2D): param ⊂ row ⊂ tensor ⊂ block.
Resultado interpretable: la curva min(ret_A,acc_B) vs granularidad dice exactamente
cuánta granularidad hace falta — y por tanto si rediseñar el termostato a fila/canal
salvaría a Modelo X.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import get_device
from ..phase9.exp_multihead import (
    LAM_GRID, SEEDS, SA, SB, _fresh, acc, fisher, snapshot, train_plain,
    coarsen_to_block, _modelox,
)

LEVELS = ("param", "row", "tensor", "block")
D_MODEL = 256


def coarsen(fis, level):
    """Engrosa el Fisher a la granularidad pedida (ver módulo)."""
    if level == "param":
        return fis
    if level == "block":
        return coarsen_to_block(fis)
    out = {}
    for n, f in fis.items():
        if level == "tensor":
            out[n] = torch.full_like(f, f.mean().item())
        elif level == "row":
            if f.ndim >= 2:                       # [out, in...] -> media por fila de salida
                m = f.mean(dim=tuple(range(1, f.ndim)), keepdim=True)
                out[n] = m.expand_as(f).contiguous()
            else:                                 # 1D (bias/LN): grano fino (pocos params)
                out[n] = f.clone()
    return out


def _seq(seed, lam, level):
    c, m, task = _fresh(seed, D_MODEL)
    train_plain(m, task, 0, SA)
    star = snapshot(m)
    fis = coarsen(fisher(m, task, 0), level)
    train_plain(m, task, 1, SB, decoupled=(fis, star, lam))
    return acc(m, task, 0), acc(m, task, 1)


def _agg(rows):
    a = np.array(rows); return a[:, 0].mean(), a[:, 1].mean()


def run():
    print("\n=== Fase 10.1 — Continuo de granularidad del anclaje (d=256) ===\n")
    front = {lv: {lam: _agg([_seq(s, lam, lv) for s in SEEDS]) for lam in LAM_GRID}
             for lv in LEVELS}
    mx = _agg([_modelox(s, D_MODEL, "no_jump") for s in SEEDS])

    print(f"  {'granularidad':>10} | mejor punto (ret_A, acc_B) y min")
    best = {}
    for lv in LEVELS:
        blam = max(LAM_GRID, key=lambda l: min(front[lv][l]))
        rA, aB = front[lv][blam]; best[lv] = (rA, aB, blam)
        print(f"  {lv:>10} | ret_A={rA:.2f} acc_B={aB:.2f}  min={min(rA, aB):.2f}  (λ={blam:.0e})")
    print(f"  {'Modelo X':>10} | ret_A={mx[0]:.2f} acc_B={mx[1]:.2f}  min={min(mx):.2f}  (bloque real)")

    mins = {lv: min(best[lv][0], best[lv][1]) for lv in LEVELS}
    subblock = ("param", "row", "tensor")
    sub_min = min(mins[lv] for lv in subblock)
    print("\n--- Lectura del continuo (min por granularidad) ---")
    print("  " + "  ".join(f"{lv}={mins[lv]:.2f}" for lv in LEVELS))
    print(f"\n[check] por-BLOQUE no retiene (min<0.35): {mins['block'] < 0.35}")
    print(f"[check] TODA granularidad SUB-BLOQUE retiene (param/row/tensor min>0.5): {sub_min > 0.5}")
    print(f"[check] el acantilado es BLOQUE-específico (sub-bloque − bloque > 0.25): "
          f"{sub_min - mins['block'] > 0.25}")
    print("\nLectura (corregida con multi-semilla; el 'pico por canal' de 1 semilla era ruido):")
    print("la curva NO es 'más fino siempre mejor'. param/row/tensor quedan TODOS en ~0.6")
    print("(dentro del ruido entre sí); el único que cae es BLOQUE (~0.29). El acantilado")
    print("está entre per-TENSOR y per-BLOQUE: distinguir importancia POR MATRIZ DE PESOS")
    print("(cada Linear q/k/v/out/in/out_proj por separado) BASTA; lumpar las 6 Linears de")
    print("un bloque en una sola decisión es lo que mata la retención. Diseño: como Modelo X")
    print("YA rastrea K-FAC por-Linear, el fix MÍNIMO es la decisión térmica por-Linear")
    print("(per-tensor) en vez de por-bloque. El hard-freeze por canal se prueba en 10.2.")

    _plot(mins, min(mx))
    return dict(front=front, best=best, mins=mins, modelox=mx)


def _plot(mins, mx_min):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = list(range(len(LEVELS)))
    ys = [mins[lv] for lv in LEVELS]
    plt.figure(figsize=(6, 4.5))
    plt.plot(xs, ys, "o-", label="EWC anclado a esta granularidad")
    plt.axhline(mx_min, color="red", ls="--", label=f"Modelo X (bloque real) min={mx_min:.2f}")
    plt.axhline(0.5, color="gray", ls=":", alpha=0.6, label="umbral de retención simultánea")
    plt.xticks(xs, [f"{lv}\n(fino→grueso)" if lv == "param" else lv for lv in LEVELS])
    plt.ylabel("min(ret_A, acc_B)  [mejor sobre λ]")
    plt.title("Fase 10.1 — ¿Cuánta granularidad hace falta para retener?")
    plt.ylim(0, 1); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("phase10_granularity_continuum.png", dpi=120)
    print("\n[fig] phase10_granularity_continuum.png")


if __name__ == "__main__":
    run()
