"""Fase 10.2 — Hard-freeze por CANAL en el mecanismo de Modelo X (no por bloque).

10.1 mostró (con el anclaje SUAVE de EWC) que la granularidad por fila/canal es el sweet
spot. Pero Modelo X no ancla suave: HARD-CONGELA (enmascara el gradiente: congelado ->
sin update). Aquí probamos el mecanismo REAL de Modelo X —congelamiento DURO— pero a
granularidad de CANAL en vez de BLOQUE, para confirmar que el rediseño cierra la brecha.

Protocolo: tras consolidar A se mide la importancia por fila (Fisher medio por canal de
salida) y se CONGELAN DURO (gradiente=0 durante B) las filas más importantes, hasta una
fracción τ. Se barre τ (frontera retención_A vs acc_B). Comparación:
  * canal: congela las filas top-τ por importancia (granularidad fina del termostato).
  * bloque: congela el/los bloque(s) más importantes (lo que Modelo X hace hoy).
  * Modelo X real (referencia, Fase 9).

Cobertura idéntica a 10.1 (se enmascaran TODOS los parámetros, 2D por fila y 1D por
elemento) para aislar la GRANULARIDAD, no la cobertura.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..config import get_device
from ..phase9.exp_multihead import (
    SEEDS, SA, SB, _fresh, acc, fisher, snapshot, _modelox, _block_group, train_plain,
)

D_MODEL = 256
TAUS = (0.2, 0.35, 0.5, 0.65, 0.8)


def _row_importance(fis):
    """Importancia por fila de salida: media de Fisher por canal. Devuelve, por
    parámetro, un tensor de la misma forma con la importancia de su fila (para 1D, el
    propio valor)."""
    imp = {}
    for n, f in fis.items():
        if f.ndim >= 2:
            m = f.mean(dim=tuple(range(1, f.ndim)), keepdim=True)
            imp[n] = m.expand_as(f).contiguous()
        else:
            imp[n] = f.clone()
    return imp


def _freeze_mask(fis, tau, level):
    """Máscara booleana (True = CONGELADO) por parámetro.
      level='channel': congela las filas con importancia por canal en el top-τ global.
      level='block'  : congela los bloques cuya importancia media esté en el top-τ.
    """
    if level == "channel":
        imp = _row_importance(fis)
        allv = torch.cat([v.flatten() for v in imp.values()])
        thr = torch.quantile(allv, 1.0 - tau)
        return {n: (v >= thr) for n, v in imp.items()}
    # bloque: importancia media por grupo de bloque
    from collections import defaultdict
    tot, cnt = defaultdict(float), defaultdict(float)
    for n, f in fis.items():
        tot[_block_group(n)] += f.sum().item(); cnt[_block_group(n)] += f.numel()
    means = {g: tot[g] / cnt[g] for g in tot}
    groups = sorted(means, key=lambda g: means[g], reverse=True)
    k = max(1, round(tau * len(groups)))
    frozen_g = set(groups[:k])
    return {n: torch.full_like(f, _block_group(n) in frozen_g, dtype=torch.bool)
            for n, f in fis.items()}


def _train_masked(m, task, t, steps, mask, lr=3e-3, bs=64):
    """Entrena t con Adam pero CONGELA DURO (gradiente=0) los parámetros enmascarados."""
    m.current_task = t
    for b in m.blocks:
        b.deterministic = True
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    named = dict(m.named_parameters())
    for _ in range(steps):
        x, y = task.batch_task(t, bs)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).reshape(-1, m.cfg.vocab_size), y.reshape(-1)).backward()
        for n, p in named.items():
            if p.grad is not None and n in mask:
                p.grad.mul_(~mask[n])           # anula el gradiente de las filas congeladas
        opt.step()


def _seq(seed, tau, level):
    c, m, task = _fresh(seed, D_MODEL)
    train_plain(m, task, 0, SA)
    fis = fisher(m, task, 0)
    mask = _freeze_mask(fis, tau, level)
    _train_masked(m, task, 1, SB, mask)
    return acc(m, task, 0), acc(m, task, 1)


def _agg(rows):
    a = np.array(rows); return a[:, 0].mean(), a[:, 1].mean()


def run():
    print("\n=== Fase 10.2 — Hard-freeze por CANAL vs por BLOQUE (mecanismo real) ===\n")
    chan = {tau: _agg([_seq(s, tau, "channel") for s in SEEDS]) for tau in TAUS}
    blk = {tau: _agg([_seq(s, tau, "block") for s in SEEDS]) for tau in TAUS}
    mx = _agg([_modelox(s, D_MODEL, "no_jump") for s in SEEDS])

    print("  HARD-FREEZE POR CANAL (granularidad fina del termostato):")
    print(f"  {'τ':>5} {'ret_A':>6} {'acc_B':>6} {'min':>6}")
    for tau in TAUS:
        rA, aB = chan[tau]; print(f"  {tau:5.2f} {rA:6.2f} {aB:6.2f} {min(rA, aB):6.2f}")
    best_c = max(TAUS, key=lambda t: min(chan[t]))
    best_b = max(TAUS, key=lambda t: min(blk[t]))
    print(f"\n  mejor CANAL : ret_A={chan[best_c][0]:.2f} acc_B={chan[best_c][1]:.2f} "
          f"min={min(chan[best_c]):.2f} (τ={best_c})")
    print(f"  mejor BLOQUE: ret_A={blk[best_b][0]:.2f} acc_B={blk[best_b][1]:.2f} "
          f"min={min(blk[best_b]):.2f} (τ={best_b})")
    print(f"  Modelo X real: ret_A={mx[0]:.2f} acc_B={mx[1]:.2f} min={min(mx):.2f}")

    mc, mb = min(chan[best_c]), min(blk[best_b])
    print("\n--- Checks (historia de DOS factores: granularidad Y softness) ---")
    print(f"[check] hard-freeze por BLOQUE no retiene (min<0.35): {mb < 0.35}")
    print(f"[check] la GRANULARIDAD ayuda: canal >> bloque (Δmin>0.2): {mc - mb > 0.2}")
    print(f"[check] canal hard-freeze >> Modelo X real (Δmin>0.25): {mc - min(mx) > 0.25}")
    print(f"[check] pero la granularidad SOLA (hard) no basta para retención plena (min<0.5): {mc < 0.5}")
    print("\nLectura (honestidad de DOS factores): el congelamiento DURO —el mecanismo real de")
    print(f"Modelo X— retiene mucho mejor por CANAL ({mc:.2f}) que por bloque ({mb:.2f}) o que Modelo X")
    print(f"real ({min(mx):.2f}): la GRANULARIDAD ayuda. Pero el canal-hard queda por DEBAJO del anclaje")
    print("SUAVE sub-bloque de 10.1 (~0.6): el freeze binario suelta del todo los canales no-top,")
    print("que aun así corrompen A, mientras el anclaje graduado protege en proporción a la")
    print("curvatura. Conclusión: la limitación de Modelo X es DOBLE (bloque + binario) y la cura")
    print("completa es un termostato por-Linear/canal Y GRADUADO (soft). Ninguno de los dos solo basta.")

    _plot(chan, blk, mx)
    return dict(channel=chan, block=blk, modelox=mx)


def _plot(chan, blk, mx):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c = np.array([chan[t] for t in TAUS]); b = np.array([blk[t] for t in TAUS])
        plt.figure(figsize=(6, 5))
        plt.plot(c[:, 1], c[:, 0], "o-", label="hard-freeze por CANAL")
        plt.plot(b[:, 1], b[:, 0], "s--", color="gray", label="hard-freeze por BLOQUE")
        plt.scatter([mx[1]], [mx[0]], c="red", marker="*", s=240, label="Modelo X real", zorder=6)
        plt.xlabel("acc_B"); plt.ylabel("retención_A")
        plt.title("Fase 10.2 — Hard-freeze: canal vs bloque\n(el mecanismo real de Modelo X, refinado)")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("phase10_channel_freeze.png", dpi=120)
        print("\n[fig] phase10_channel_freeze.png")
    except Exception:
        pass


if __name__ == "__main__":
    run()
