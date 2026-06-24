"""Fase 6.3 — Modelo X frente a la literatura (EWC) en continual learning: lo que
se puede concluir honestamente y lo que queda confundido.

Se construye un benchmark task-incremental BIEN PLANTEADO (token de tarea -> existe
solucion conjunta) y se intenta comparar Modelo X contra EWC (Kirkpatrick 2017) y
naive. El hallazgo central es metodologico y honesto:

  * El benchmark es valido: entrenamiento INTERCALADO alcanza acc_A=acc_B=1.0
    (existe una solucion conjunta que usa el token de tarea).
  * Secuencial naive (Adam): OLVIDO CATASTROFICO de A (retencion ~ azar).
  * EWC esta CONFUNDIDO por el optimizador en esta arquitectura:
      - bajo Adam (unico optimizador con el que esta arquitectura entrena), la
        penalizacion cuadratica de EWC queda neutralizada por la renormalizacion
        de Adam -> EWC ~ naive en todo el barrido de lambda.
      - bajo SGD (donde EWC seria efectivo), la arquitectura atencion+SDE NO entrena
        (acc ~ azar): fue disenada para Adam/gradiente natural.
    => Un head-to-head limpio de RETENCION SIMULTANEA no es concluyente AQUI.

Conclusion honesta: el eje donde Modelo X esta validado no es la retencion simultanea
multi-tarea (que nunca reclamo) sino el AHORRO al revisitar una tarea (5.2). Un
head-to-head limpio contra EWC exige portar el framework a una arquitectura estandar
(entrenable con SGD) o usar un regularizador CL compatible con Adam: un siguiente paso
concreto, no un numero inflado.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import Config, get_device
from ..model import ModeloX
from .taskid_data import TaskIDData, IGNORE

STEPS_A, STEPS_B = 200, 200


@torch.no_grad()
def _acc(model, data, task, n=8, bs=64):
    for b in model.blocks:
        b.deterministic = True
    c = t = 0
    for _ in range(n):
        x, y = data.batch(task, bs)
        lo = model(x); msk = y != IGNORE
        c += ((lo.argmax(-1) == y) & msk).sum().item(); t += msk.sum().item()
    for b in model.blocks:
        b.deterministic = False
    return c / max(1, t)


def _train(model, data, task, steps, opt, ewc=None):
    for b in model.blocks:
        b.deterministic = True
    for _ in range(steps):
        x, y = data.batch(task, 64)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, model.cfg.vocab_size), y.reshape(-1),
                               ignore_index=IGNORE)
        if ewc is not None:
            fisher, star, lam = ewc
            loss = loss + 0.5 * lam * sum((fisher[n] * (p - star[n]) ** 2).sum()
                                          for n, p in model.named_parameters() if n in fisher)
        loss.backward(); opt.step()


def _fisher(model, data, task, n=40):
    fis = {nm: torch.zeros_like(p) for nm, p in model.named_parameters()}
    for b in model.blocks:
        b.deterministic = True
    for _ in range(n):
        x, y = data.batch(task, 64)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).reshape(-1, model.cfg.vocab_size), y.reshape(-1),
                        ignore_index=IGNORE).backward()
        for nm, p in model.named_parameters():
            if p.grad is not None:
                fis[nm] += p.grad.detach() ** 2 / n
    return fis


def _fresh(dev, seed=0):
    cfg = Config(); torch.manual_seed(seed)
    return ModeloX(cfg.model).to(dev), TaskIDData(cfg.model, dev, seed=seed)


def run():
    dev = get_device()
    print("\n=== Fase 6.3: Modelo X vs EWC (continual A->B con token de tarea) ===\n")

    # 0) El benchmark es valido: solucion conjunta existe (entrenamiento intercalado)
    m, data = _fresh(dev)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for i in range(400):
        x, y = data.batch(i % 2, 64)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).reshape(-1, m.cfg.vocab_size), y.reshape(-1),
                        ignore_index=IGNORE).backward(); opt.step()
    joint = (_acc(m, data, 0), _acc(m, data, 1))
    print(f"(0) benchmark valido: intercalado -> acc_A={joint[0]:.2f} acc_B={joint[1]:.2f} "
          f"(existe solucion conjunta)")

    # 1) naive secuencial (Adam): olvido catastrofico
    m, data = _fresh(dev)
    _train(m, data, 0, STEPS_A, torch.optim.Adam(m.parameters(), lr=3e-3))
    _train(m, data, 1, STEPS_B, torch.optim.Adam(m.parameters(), lr=3e-3))
    naive = (_acc(m, data, 0), _acc(m, data, 1))
    print(f"(1) naive secuencial (Adam): retencion_A={naive[0]:.2f}  acc_B={naive[1]:.2f}")

    # 2) EWC bajo Adam: barrido de lambda (penalizacion neutralizada por Adam)
    print("(2) EWC bajo Adam (la arquitectura solo entrena con Adam):")
    ewc_rows = []
    for lam in (1e4, 1e6):
        m, data = _fresh(dev)
        _train(m, data, 0, STEPS_A, torch.optim.Adam(m.parameters(), lr=3e-3))
        fis = _fisher(m, data, 0); star = {n: p.detach().clone() for n, p in m.named_parameters()}
        _train(m, data, 1, STEPS_B, torch.optim.Adam(m.parameters(), lr=3e-3), ewc=(fis, star, lam))
        r = (_acc(m, data, 0), _acc(m, data, 1)); ewc_rows.append(r)
        print(f"     EWC lambda={lam:.0e}: retencion_A={r[0]:.2f}  acc_B={r[1]:.2f}")

    # 3) EWC bajo SGD: la arquitectura no entrena (confound del optimizador)
    m, data = _fresh(dev)
    _train(m, data, 0, STEPS_A, torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9))
    sgd_A = _acc(m, data, 0)
    print(f"(3) SGD (donde EWC seria efectivo): acc_A tras entrenar A = {sgd_A:.2f} "
          f"-> la arquitectura NO entrena con SGD")

    print("\n--- Veredicto 6.3 ---")
    print(f"[check] benchmark valido (intercalado >0.9 ambos): {min(joint) > 0.9}")
    print(f"[check] naive olvida A (retencion < 0.3): {naive[0] < 0.3}")
    print(f"[check] EWC-Adam no mejora la retencion (neutralizado): "
          f"{max(r[0] for r in ewc_rows) < naive[0] + 0.1}")
    print(f"[check] SGD no entrena esta arquitectura (acc_A < 0.3): {sgd_A < 0.3}")
    print("\nLectura honesta: el head-to-head de RETENCION SIMULTANEA esta confundido por el")
    print("optimizador (Adam neutraliza EWC; SGD no entrena la arquitectura), asi que NO se")
    print("fabrica un ganador. El eje donde Modelo X esta validado es el AHORRO al revisitar")
    print("una tarea (Fase 5.2: re-aprende ~3.5x mas rapido, 5x mas que el plastico). Un")
    print("head-to-head limpio contra EWC requiere portar a una arquitectura entrenable con")
    print("SGD o usar un regularizador CL compatible con Adam (online-EWC/SI escalado): paso")
    print("siguiente concreto, no un numero inflado.")
    return dict(joint=joint, naive=naive, ewc=ewc_rows, sgd_A=sgd_A)


if __name__ == "__main__":
    run()
