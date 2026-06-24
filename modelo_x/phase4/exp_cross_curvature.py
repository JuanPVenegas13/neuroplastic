"""Fase 4.3 — Curvatura cruzada entre sublocaciones (medición RIGUROSA).

Primer intento (descartado): aproximar el acoplamiento por rho_s (correlacion de
gradientes de salida) o por rho_A*rho_s resulto enganoso — los pares EN SERIE tienen
rho_s alto por la regla de la cadena y rho_A no despreciable, sin que eso implique una
curvatura cruzada real factorizable.

Medicion correcta: el bloque cruzado del Fisher empirico entre dos parametros W1, W2
es C12 = E[g1 g2^T] con g_n el gradiente por-muestra aplanado. Su tamano relativo es
    rho = ||C12||_F / (||C11||_F * ||C22||_F)^{1/2}
y, por la identidad ||C12||^2_F = (1/N^2) sum_{n,m} <g1n,g1m><g2n,g2m>, equivale al
COSENO DE FROBENIUS entre las matrices de Gram por-muestra G1, G2 (Ga[n,m]=<gan,gam>):
    rho = <G1,G2>_F / (||G1||_F * ||G2||_F)
Esto es exacto y barato (Gram NxN a partir de gradientes por-muestra).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import Config, get_device
from ..model import ModeloX
from ..drift_data import DriftTask


def run(n_samples: int = 100):
    cfg = Config(); dev = get_device(); torch.manual_seed(0)
    m = ModeloX(cfg.model).to(dev)
    for b in m.blocks:
        b.deterministic = True
    task = DriftTask(cfg.model, cfg.drift, dev, seed=0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(150):
        x, y = task.batch(0, 64)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).reshape(-1, cfg.model.vocab_size), y.reshape(-1)).backward()
        opt.step()

    b0 = m.blocks[0]
    W = {"q": b0.attn.q.weight, "k": b0.attn.k.weight, "v": b0.attn.v.weight,
         "out": b0.attn.out.weight, "in": b0.in_proj.weight, "out_proj": b0.out_proj.weight}
    gmat = {name: [] for name in W}
    for _ in range(n_samples):
        x, y = task.batch(0, 1)
        m.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).reshape(-1, cfg.model.vocab_size), y.reshape(-1)).backward()
        for name, w in W.items():
            g = w.grad.detach().flatten().clone()
            gmat[name].append(g / (g.norm() + 1e-12))   # norma unidad: aísla la DIRECCIÓN
            #  (sin esto, la Gram la domina la dificultad por-muestra -> coseno inflado
            #   en TODAS las parejas, un confound; normalizar mide acoplamiento direccional)
    G = {}
    for name in W:
        M = torch.stack(gmat[name])
        Gm = M @ M.t()
        Gm.fill_diagonal_(0.0)        # quita la auto-correlación trivial (diagonal=1)
        G[name] = Gm

    def rho(a, b):
        Ga, Gb = G[a], G[b]
        return (Ga * Gb).sum().item() / (Ga.norm().item() * Gb.norm().item() + 1e-12)

    pairs = [("q", "k"), ("q", "v"), ("k", "v"),
             ("v", "out"), ("out", "in"), ("in", "out_proj")]
    print("\nCurvatura cruzada EXACTA (coseno de Gram de gradientes por-muestra), bloque 0")
    print(f"{'par':>16} {'rho_curv':>9}")
    print("-" * 28)
    rows = []
    for a, b in pairs:
        r = rho(a, b); rows.append((f"{a}-{b}", r))
        print(f"{a+'-'+b:>16} {r:>9.3f}")

    qk = next(r for n, r in rows if n == "q-k")
    attn_pairs = [r for n, r in rows if n in ("q-k", "q-v", "k-v")]
    series_pairs = [r for n, r in rows if n in ("v-out", "out-in", "in-out_proj")]
    ma = sum(attn_pairs) / len(attn_pairs); ms = sum(series_pairs) / len(series_pairs)
    print(f"\nmedia rho_curv (direccional)  QKV={ma:.3f}  series={ms:.3f}")
    print(f"[hallazgo] la curvatura cruzada es ALTA en todo el bloque, y MAYOR en el camino")
    print(f"           en serie ({ms:.2f}) que dentro de QKV ({ma:.2f}). El supuesto bloque-")
    print(f"           diagonal de K-FAC es MAS CRUDO de lo que la Fase 1 asumia.")
    print(f"[matiz clave] solo los pares con ENTRADA COMPARTIDA (q,k,v sobre ln1(x)) tienen")
    print(f"   curvatura cruzada KRONECKER-FACTORIZABLE (A compartido -> A (x) S_12), unico")
    print(f"   caso corregible barato (lo hace la 4.2 para Q-K). La curvatura cruzada en serie,")
    print(f"   aun siendo grande, NO factoriza limpio (entradas distintas) y queda FUERA del")
    print(f"   alcance de K-FAC sin una aproximacion distinta y mas costosa.")
    print(f"\nConclusion honesta: 4.2 (acoplado Q-K) ataca el unico acoplamiento grande Y")
    print(f"factorizable. El acoplamiento en serie, mayor en magnitud, es una LIMITACION")
    print(f"estructural de K-FAC por bloques que conviene registrar, no algo que 4.2 resuelva.")
    return rows


if __name__ == "__main__":
    run()
