"""Orquestador de entrenamiento de Modelo X.

Sigue el pseudocódigo de la especificación, conectando los cuatro subsistemas y
registrando el LAZO DE RETROALIMENTACIÓN (traza → estado térmico → salto → traza)
que la Fase 1 identificó como el mayor riesgo del modelo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .drift_data import DriftTask, ManifoldDriftDetector
from .kfac import KFACManager
from .model import ModeloX
from .thermodynamics import ThermalController, FROZEN


def linear_to_block(name: str) -> str:
    """Mapea el nombre de una Linear (p.ej. 'b1.attn.q') al nombre térmico del bloque ('b1.out')."""
    return name.split(".")[0] + ".out"


class Trainer:
    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        self.device = device
        torch.manual_seed(cfg.train.seed)

        self.model = ModeloX(cfg.model).to(device)
        self.kfac_linears = self.model.kfac_linears()
        self.kfac = KFACManager(self.kfac_linears, cfg.kfac, device)
        self.thermo = ThermalController(self.model.block_names(), cfg.thermo)
        self.task = DriftTask(cfg.model, cfg.drift, device, seed=cfg.train.seed)
        self.detector = ManifoldDriftDetector(cfg.model.d_model, cfg.drift, device)

        kfac_params = {id(l.weight) for l in self.kfac_linears.values()}
        base_params = [p for p in self.model.parameters() if id(p) not in kfac_params]
        self.base_opt = torch.optim.Adam(base_params, lr=cfg.train.lr_base)

        self.traces: dict[str, float] = {}
        self.prev_loss: float | None = None
        self._reinf_baseline: float | None = None   # línea base EMA para REINFORCE (Fase 4.1)
        self.log: list[dict] = []

    def _write_thermal_signals(self):
        for b in self.model.blocks:
            sig = self.thermo.signal(f"b{b.idx}.out")
            b.sigma_gate = sig.sigma_gate
            b.sigma_epist = sig.sigma_epist
            b.lam = sig.lam
            b.state = sig.state

    def _apply_natural_gradient_step(self):
        lr = self.cfg.train.lr_kfac
        states = self.thermo.states()
        for name, lin in self.kfac_linears.items():
            if lin.weight.grad is None:
                continue
            block = linear_to_block(name)
            if states.get(block) == FROZEN:           # bloque congelado: no se actualiza
                lin.weight.grad = None
                continue
            nat = self.kfac.natural_gradient(name, lin.weight.grad)
            # recorte de norma para estabilidad
            gnorm = nat.norm()
            if gnorm > 10.0:
                nat = nat * (10.0 / (gnorm + 1e-9))
            lin.weight.data.add_(nat, alpha=-lr)
            lin.weight.grad = None

    @torch.no_grad()
    def evaluate(self, step: int, n_batches: int = 4) -> float:
        """Accuracy LIMPIA: apaga difusión y saltos del SDE para medir la calidad
        real del modelo sin la penalización del ruido estocástico del forward."""
        for b in self.model.blocks:
            b.deterministic = True
        correct = total = 0
        for _ in range(n_batches):
            x, y = self.task.batch(step, self.cfg.train.batch_size)
            logits = self.model(x)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
        for b in self.model.blocks:
            b.deterministic = False
        return correct / max(1, total)

    def step(self, step: int):
        cfg = self.cfg
        x, y = self.task.batch(step, cfg.train.batch_size)

        # 1) Monitoreo topológico: deriva y detección de Cisne Negro
        with torch.no_grad():
            emb = self.model.embed(x)
        drift_score, jump_detected, cov_kl = self.detector.update(emb, self.prev_loss)

        # 2) Estado térmico (histéresis) usando trazas de la última inversión
        self.thermo.update(self.traces, jump_detected, step)
        self._write_thermal_signals()

        # 3) Forward + pérdida + backward (hooks K-FAC capturan estadística)
        self.base_opt.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.model.vocab_size), y.reshape(-1))
        # Fase 4.1: si la intensidad de salto es aprendible con REINFORCE, añade el
        # surrogate score-function (loss.detach()-baseline)·Σlogp para gradiente insesgado.
        if cfg.model.learn_jump and cfg.model.jump_grad == "reinforce":
            logp = self.model.jump_logp_sum()
            if logp is not None:
                b = loss.item() if self._reinf_baseline is None else self._reinf_baseline
                total = loss + (loss.detach() - b) * logp
                self._reinf_baseline = (0.99 * (self._reinf_baseline if self._reinf_baseline
                                        is not None else loss.item()) + 0.01 * loss.item())
                total.backward()
            else:
                loss.backward()
        else:
            loss.backward()

        # 4) Inversión amortizada -> trazas termodinámicas + monitor de salud
        cond_warnings = []
        if step % cfg.kfac.invert_every == 0:
            cond_warnings = self.kfac.invert_all()
            self.traces = {b: self.kfac.inverse_fisher_trace(b)
                           for b in self.model.block_names()}

        # 5) Paso de gradiente natural (K-FAC) + paso base (Adam)
        self._apply_natural_gradient_step()
        self.base_opt.step()

        # 6) Instrumentación del lazo de retroalimentación
        acc = (logits.argmax(-1) == y).float().mean().item()
        self.prev_loss = loss.item()
        do_eval = (step % self.cfg.train.log_every == 0) or (step == self.cfg.train.steps - 1)
        eval_acc = self.evaluate(step) if do_eval else None
        rec = {
            "step": step, "loss": loss.item(), "acc": acc, "eval_acc": eval_acc,
            "drift_score": drift_score, "jump": int(jump_detected),
            "covariate_kl": cov_kl,
            "states": dict(self.thermo.states()),
            "traces": {k: round(v, 2) for k, v in self.traces.items()},
            "jump_activity": {f"b{b.idx}": round(b.last_jump_activity, 4)
                              for b in self.model.blocks},
            "cond_warnings": cond_warnings,
        }
        self.log.append(rec)
        return rec

    def fit(self):
        for s in range(self.cfg.train.steps):
            rec = self.step(s)
            if s % self.cfg.train.log_every == 0 or s == self.cfg.train.steps - 1:
                self._print(rec)
        return self.log

    def _print(self, rec):
        st = " ".join(f"{k.split('.')[0]}:{v[0]}" for k, v in rec["states"].items())
        warn = "  ⚠cond" if rec["cond_warnings"] else ""
        ev = f" | eval_acc {rec['eval_acc']:.2f}" if rec["eval_acc"] is not None else ""
        print(f"step {rec['step']:4d} | loss {rec['loss']:.3f} | acc {rec['acc']:.2f}{ev} "
              f"| drift {rec['drift_score']:.2f} | jump {rec['jump']} | {st}{warn}")
