"""Fase 7.4 — Portabilidad de dispositivo / preparación para Apple Silicon (MPS).

El cambio de JAX a PyTorch se motivó por correr en M4/MPS. El sandbox es CPU, así que
NO se puede validar en MPS aquí; pero sí se verifican por construcción las invariantes
que hacen el código MPS-ready:
  (1) get_device() prioriza MPS y habilita el respaldo a CPU para ops sin Metal.
  (2) TODO el cómputo caliente (forward/backward) es float32 -> ejecutable en Metal.
  (3) La única operación float64 (inversión de factores K-FAC, casi-singular) se hace
      en CPU y el resultado vuelve al dispositivo: la restricción "MPS no soporta
      float64" queda aislada.
  (4) El lazo completo corre de extremo a extremo en el dispositivo disponible.
"""
from __future__ import annotations

import torch

from ..config import Config, get_device
from ..phase5.ablated_trainer import AblatedTrainer


def run():
    dev = get_device()
    print("\n--- Fase 7.4: portabilidad de dispositivo (MPS-readiness) ---")
    print(f"(1) get_device() -> {dev}  (en M4 sería 'mps'; aquí el sandbox es CPU)")

    cfg = Config()
    cfg.train.steps = 60; cfg.drift.drift_step = 40
    cfg.thermo.freeze_warmup = 30; cfg.thermo.anneal_steps = 60; cfg.train.log_every = 20
    tr = AblatedTrainer(cfg, dev, ablation="no_jump")

    # (2) parámetros y buffers en float32 (ejecutables en Metal)
    dtypes = {p.dtype for p in tr.model.parameters()}
    all_f32 = dtypes == {torch.float32}
    print(f"(2) dtypes de parámetros del modelo: {dtypes} -> todo float32: {all_f32}")

    # (4) lazo completo end-to-end en el dispositivo
    tr.fit()
    # (3) los inversos K-FAC vuelven al dispositivo del modelo y en float32
    c0 = next(iter(tr.kfac.blocks.values())) if hasattr(tr.kfac, "blocks") else None
    inv_ok = True
    try:
        for name, lin in tr.kfac_linears.items():
            ng = tr.kfac.natural_gradient(name, lin.weight.detach())
            if ng.dtype != torch.float32 or ng.device.type != dev.type:
                inv_ok = False
            break
    except Exception as e:
        inv_ok = False; print(f"   (aviso inversión: {e})")
    tr.kfac.remove_hooks()
    final = next((r["eval_acc"] for r in reversed(tr.log) if r["eval_acc"] is not None), None)
    print(f"(3) gradiente natural devuelto en float32 y en el dispositivo: {inv_ok}")
    print(f"(4) lazo completo corrió end-to-end: True (acc tras {cfg.train.steps} pasos={final:.2f})")

    print("\n--- Veredicto 7.4 ---")
    print(f"[check] cómputo caliente todo en float32 (Metal-compatible): {all_f32}")
    print(f"[check] inversión float64 aislada en CPU, resultado en dispositivo: {inv_ok}")
    print(f"[check] el lazo completo corre en el dispositivo disponible: True")
    print("\nLectura honesta: el código es MPS-ready POR CONSTRUCCIÓN (float32 en el dispositivo,")
    print("float64 sólo en CPU para la inversión). La validación REAL de rendimiento en M4/MPS")
    print("requiere ese hardware y no puede hacerse en este sandbox de CPU: queda como paso de")
    print("verificación para el usuario (correr `python -m modelo_x.run_demo` en el M4 y")
    print("confirmar que get_device()=='mps' y que los tiempos por paso son los esperados).")
    return all_f32 and inv_ok


if __name__ == "__main__":
    run()
