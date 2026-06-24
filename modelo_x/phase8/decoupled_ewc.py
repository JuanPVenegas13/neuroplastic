"""Fase 8 — EWC DESACOPLADO (estilo AdamW): resolver el confound del optimizador de 6.3.

Diagnóstico de 6.3 (reproducido en M4): bajo Adam, el gradiente de la penalización
cuadrática de EWC `λ·F·(θ-θ*)` pasa por la renormalización por-parámetro de Adam (se
divide por la RMS corriente del gradiente total), así que la "rigidez" que EWC quiere
imponer a los parámetros de alto Fisher se APLANA sin importar su magnitud. Por eso
λ=1e4 y λ=1e6 dan la misma retención (~naive): subir λ no ayuda porque Adam normaliza
la magnitud. Online-EWC / SI son penalizaciones cuadráticas igual de neutralizables.

Fix (raíz del confound): DESACOPLAR el tirón de EWC del preacondicionador de Adam,
exactamente como AdamW desacopla el weight decay. El gradiente de la TAREA pasa por
Adam; el tirón de EWC se aplica como update directo POST-Adam:

    θ ← θ - lr·λ·F·(θ - θ*)          (clamp del coef a [0,1] por estabilidad)

Con el coef `c = lr·λ·F` recortado a [0,1] el update es `θ ← (1-c)θ + c·θ*`: un
SOFT-FREEZE por-parámetro hacia θ*, con fuerza = curvatura de Fisher. Es el análogo
justo del hard-freeze por curvatura de Modelo X, sobre la MISMA red y el MISMO
optimizador (Adam) — head-to-head sin confound.

ADVERTENCIA de honestidad (criterio rector del experimento): el desacoplado puede
MORDER DEMASIADO y degradar acc_B (proteger tanto A que ya no aprende B). Por eso la
comparación contra Modelo X es A IGUAL acc_B, sobre la frontera retención_A vs acc_B
barriendo λ (ver exp_retention_frontier.py). "EWC retiene 0.9" con acc_B caído NO es
victoria; es otro punto del trade-off.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import Config, get_device
from ..model import ModeloX
from ..phase6.taskid_data import TaskIDData, IGNORE


# ----------------------------------------------------------------------------- utils
def fresh_model(dev, seed: int = 0):
    """ModeloX + generador de datos task-id, ambos con la misma semilla."""
    cfg = Config()
    torch.manual_seed(seed)
    model = ModeloX(cfg.model).to(dev)
    data = TaskIDData(cfg.model, dev, seed=seed)
    return model, data


@torch.no_grad()
def masked_acc(model, data, task: int, n: int = 8, bs: int = 64) -> float:
    """Accuracy enmascarando IGNORE (pos 0=token de tarea, 1=sin lag válido).

    NB: imprescindible — la eval del Trainer NO enmascara y deflactaría el tope a
    ~22/24=0.917, sesgando cualquier comparación de retención.
    """
    for b in model.blocks:
        b.deterministic = True
    c = t = 0
    for _ in range(n):
        x, y = data.batch(task, bs)
        lo = model(x)
        msk = y != IGNORE
        c += ((lo.argmax(-1) == y) & msk).sum().item()
        t += msk.sum().item()
    for b in model.blocks:
        b.deterministic = False
    return c / max(1, t)


def fisher_diag(model, data, task: int, n: int = 40, bs: int = 64, mode: str = "true"):
    """Fisher diagonal sobre la tarea `task` (posiciones IGNORE enmascaradas).

    mode="empirical": E[grad(-log p(y_true|x))^2] con las etiquetas REALES. Barato pero
      a convergencia (acc≈1) los grads ≈0 y SUBESTIMA la curvatura.
    mode="true" (def.): Fisher de verdad, y ~ p_modelo(·|x) muestreado del modelo. Es
      la importancia correcta y le da a EWC su mejor versión (no penalizar a EWC por
      usar un estimador débil). Ver Fase 8.
    """
    fis = {nm: torch.zeros_like(p) for nm, p in model.named_parameters()}
    for b in model.blocks:
        b.deterministic = True
    V = model.cfg.vocab_size
    for _ in range(n):
        x, y = data.batch(task, bs)
        model.zero_grad(set_to_none=True)
        logits = model(x).reshape(-1, V)
        if mode == "true":
            with torch.no_grad():
                y_s = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
            y_s = y_s.masked_fill(y.reshape(-1) == IGNORE, IGNORE)   # respeta el enmascarado
            target = y_s
        else:
            target = y.reshape(-1)
        F.cross_entropy(logits, target, ignore_index=IGNORE).backward()
        for nm, p in model.named_parameters():
            if p.grad is not None:
                fis[nm] += p.grad.detach() ** 2 / n
    for b in model.blocks:
        b.deterministic = False
    return fis


# --------------------------------------------------------------------------- entreno
def train(model, data, task: int, steps: int, lr: float = 3e-3, *,
          coupled=None, decoupled=None, bs: int = 64):
    """Entrena `task` con Adam plano (red determinista, como los baselines de 6.3).

      coupled   = (fisher, star, lam)  -> penalización EWC clásica DENTRO de la pérdida
                                          (pasa por Adam; se neutraliza — control de 6.3)
      decoupled = (fisher, star, lam)  -> tirón EWC desacoplado POST-Adam (el fix)

    Devuelve el optimizador (por si se quiere continuidad; aquí se crea fresco por fase,
    igual que en 6.3).
    """
    for b in model.blocks:
        b.deterministic = True
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = data.batch(task, bs)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, model.cfg.vocab_size), y.reshape(-1),
                               ignore_index=IGNORE)
        if coupled is not None:
            fis, star, lam = coupled
            loss = loss + 0.5 * lam * sum((fis[n] * (p - star[n]) ** 2).sum()
                                          for n, p in model.named_parameters() if n in fis)
        loss.backward()
        opt.step()
        if decoupled is not None:
            _decoupled_pull(model, *decoupled, lr=lr)
    for b in model.blocks:
        b.deterministic = False
    return opt


@torch.no_grad()
def _decoupled_pull(model, fisher, star, lam, lr):
    """θ ← θ - clamp(lr·λ·F, 0, 1)·(θ - θ*).  Soft-freeze hacia θ* por curvatura."""
    for n, p in model.named_parameters():
        if n in fisher:
            c = (lr * lam * fisher[n]).clamp_(0.0, 1.0)
            p.add_(c * (star[n] - p))


def snapshot(model):
    """θ* = copia de los parámetros tras consolidar una tarea."""
    return {n: p.detach().clone() for n, p in model.named_parameters()}
