"""Fase 6.1 — ¿Gana el salto dJ su lugar en ALGÚN régimen razonable?

Antes de aplicar la navaja de Occam hay que intentar HONESTAMENTE que el salto sea
útil. Se le da su mejor oportunidad: una deriva DURA y potencialmente multimodal
(lag 1 -> 4) que obliga a re-localizar la atención saltando sobre atractores
intermedios (lag 2, 3). Si la exploración por saltos sirve para algo, es aquí.

Se compara, sobre varias semillas, el modelo completo (salto activo) contra 'no_jump'
(magnitud de salto = 0), midiendo la recuperación de B. Veredicto basado en si el
salto gana de forma ROBUSTA (no por una semilla afortunada).
"""
from __future__ import annotations

from ..config import Config, get_device
from ..phase5.ablated_trainer import AblatedTrainer


def _cfg(seed, lag_post, eta=None):
    c = Config()
    c.train.steps = 300
    c.train.seed = seed
    c.drift.drift_step = 150
    c.drift.lag_post = lag_post           # deriva dura: 1 -> lag_post
    c.thermo.freeze_warmup = 105
    c.thermo.anneal_steps = 300
    c.train.log_every = 15
    if eta is not None:
        c.thermo.eta = eta                # baja difusión -> el salto sería la única exploración
    return c


def _recover(ablation, seed, lag_post, eta=None):
    dev = get_device()
    tr = AblatedTrainer(_cfg(seed, lag_post, eta), dev, ablation=ablation)
    tr.fit(); tr.kfac.remove_hooks()
    return next((r["eval_acc"] for r in reversed(tr.log) if r["eval_acc"] is not None), 0.0)


def _sweep(lag_post, eta, seeds, tag):
    print(f"\n[{tag}] deriva lag 1->{lag_post}"
          + (f", eta={eta} (difusión baja)" if eta is not None else "") + f"  | semillas {seeds}")
    full, noj = [], []
    for s in seeds:
        f = _recover("full", s, lag_post, eta)
        n = _recover("no_jump", s, lag_post, eta)
        full.append(f); noj.append(n)
        print(f"   semilla {s}: full(salto)={f:.2f}  no_jump={n:.2f}  Δ={f-n:+.2f}")
    import statistics as st
    mf, mn = st.mean(full), st.mean(noj)
    wins = sum(1 for f, n in zip(full, noj) if f > n + 0.05)
    print(f"   media: full={mf:.2f}  no_jump={mn:.2f}  Δ={mf-mn:+.2f}  | "
          f"el salto gana en {wins}/{len(seeds)} semillas")
    return mf, mn, wins, len(seeds)


def run():
    seeds = [0, 1, 2, 3]
    print("\n=== Fase 6.1: ¿el salto dJ gana su lugar? (esfuerzo honesto antes de cortar) ===")
    r1 = _sweep(lag_post=4, eta=None, seeds=seeds, tag="deriva dura")
    r2 = _sweep(lag_post=4, eta=0.05, seeds=seeds, tag="deriva dura + difusión baja")

    total_wins = r1[2] + r2[2]; total = r1[3] + r2[3]
    robust = (r1[0] > r1[1] + 0.05) or (r2[0] > r2[1] + 0.05)
    print("\n--- Veredicto 6.1 ---")
    print(f"el salto gana de forma robusta en algún régimen: {robust} "
          f"(victorias totales {total_wins}/{total})")
    if not robust:
        print("=> El salto NO gana su lugar ni en el régimen que más lo favorece. Procede la")
        print("   navaja de Occam: eliminarlo y verificar que el framework simplificado conserva")
        print("   todos los resultados (6.2).")
    else:
        print("=> El salto SÍ aporta en este régimen; se conserva con la configuración que lo")
        print("   hace útil, en vez de eliminarlo.")
    return robust


if __name__ == "__main__":
    run()
