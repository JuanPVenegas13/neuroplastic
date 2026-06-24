"""K-FAC para Modelo X (Incógnita 1 de la Fase 1).

Implementa la curvatura Kronecker-factorizada F^(l) ≈ A^(l) ⊗ S^(l) sobre las
capas lineales (las 4 proyecciones de atención y las proyecciones del bloque
líquido). Decisiones de diseño:

  * Estrategia EXPAND: cada token es una muestra independiente del colector
    (coherente con la hipótesis de neuroplasticidad local).
  * Submuestreo de tokens (Gemini #2): los factores se estiman sobre una
    fracción de las posiciones, no sobre las B·T. Como A y S son esperanzas
    SEPARADAS bajo la aproximación de Kronecker, NO requieren los mismos
    índices: se submuestrean de forma independiente sin sesgo adicional.
  * Inversión en CPU/float64 (Gemini #3 + restricción Metal): MPS no soporta
    float64 y la inversión de factores casi-singulares lo necesita. Como la
    inversión está amortizada (cada K pasos) y los factores son pequeños, el
    viaje a CPU es barato.
  * El backward pass NO se modifica: los hooks corren en paralelo al grafo y la
    transformación G -> S^{-1} G A^{-1} es post-hoc (ver model/train).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import KFACConfig


@dataclass
class LayerCurvature:
    """Estado de curvatura de una capa lineal rastreada."""
    name: str
    A: torch.Tensor          # E[a a^T]  (in x in)
    S: torch.Tensor          # E[g g^T]  (out x out)
    A_inv: torch.Tensor | None = None
    S_inv: torch.Tensor | None = None
    tr_A_inv: float = 0.0
    tr_S_inv: float = 0.0
    cond: float = 1.0        # número de condición estimado (monitor de salud)
    seen: bool = False       # ¿ya recibió estadística?


class KFACManager:
    """Rastrea un conjunto de nn.Linear y administra su curvatura K-FAC."""

    def __init__(self, named_linears: dict[str, nn.Linear], cfg: KFACConfig, device):
        self.cfg = cfg
        self.device = device
        self.layers: dict[str, nn.Linear] = named_linears
        self.curv: dict[str, LayerCurvature] = {}
        self._a_cache: dict[str, torch.Tensor] = {}
        self._handles = []

        for name, lin in named_linears.items():
            n_in, n_out = lin.in_features, lin.out_features
            self.curv[name] = LayerCurvature(
                name=name,
                A=torch.zeros(n_in, n_in, device=device),
                S=torch.zeros(n_out, n_out, device=device),
            )
            self._handles.append(lin.register_forward_hook(self._fwd_hook(name)))
            self._handles.append(lin.register_full_backward_hook(self._bwd_hook(name)))

    # ----- captura de estadística vía hooks (paralelos al grafo) -----
    def _subsample(self, mat: torch.Tensor) -> torch.Tensor:
        """mat: [N, dim] -> submuestrea filas según token_subsample."""
        n = mat.shape[0]
        frac = self.cfg.token_subsample
        if frac >= 1.0 or n <= 1:
            return mat
        k = max(1, int(n * frac))
        idx = torch.randint(0, n, (k,), device=mat.device)
        return mat.index_select(0, idx)

    def _fwd_hook(self, name):
        def hook(_module, inputs, _output):
            a = inputs[0].detach()
            a = a.reshape(-1, a.shape[-1])          # [B*T, in]  (EXPAND)
            self._a_cache[name] = self._subsample(a)
        return hook

    def _bwd_hook(self, name):
        def hook(_module, _grad_input, grad_output):
            g = grad_output[0].detach()
            g = g.reshape(-1, g.shape[-1])          # [B*T, out]
            g = self._subsample(g)
            a = self._a_cache.get(name)
            if a is None:
                return
            rho = self.cfg.ema_decay
            c = self.curv[name]
            A_new = (a.t() @ a) / max(1, a.shape[0])
            S_new = (g.t() @ g) / max(1, g.shape[0])
            if not c.seen:
                c.A.copy_(A_new); c.S.copy_(S_new); c.seen = True
            else:
                c.A.mul_(rho).add_(A_new, alpha=1 - rho)
                c.S.mul_(rho).add_(S_new, alpha=1 - rho)
        return hook

    # ----- inversión amortizada (CPU/float64) -----
    @staticmethod
    def _invert_factor(M: torch.Tensor, rel: float, floor: float):
        """Invierte (M + λ_adapt·I) en CPU/float64 vía eigendescomposición simétrica,
        con AMORTIGUAMIENTO ADAPTATIVO: λ_adapt = rel·mean(eval) + floor. Esto evita
        que la traza sature cuando el espectro del factor es diminuto frente a un
        damping absoluto fijo (la causa del estado MELTED permanente).
        Devuelve (M_inv float32, traza de M_inv, número de condición regularizado)."""
        # MPS rechaza el cast directo a float64; hay que salir a CPU ANTES de castear
        # (en CPU es un no-op de dispositivo). Ver Fase 8: bug aflorado al correr en M4.
        M64 = M.detach().cpu().to(torch.float64)
        M64 = 0.5 * (M64 + M64.t())
        evals, evecs = torch.linalg.eigh(M64)
        evals = evals.clamp_min(0.0)
        mean_ev = float(evals.mean()) + 1e-12
        damp = rel * mean_ev + floor
        ev_d = evals + damp
        inv = (evecs * (1.0 / ev_d)) @ evecs.t()
        tr_inv = float((1.0 / ev_d).sum())
        cond = float(ev_d.max() / ev_d.min())
        return inv.to(torch.float32), tr_inv, cond

    def invert_all(self):
        """Recalcula A^{-1}, S^{-1}, sus trazas y el número de condición."""
        rel = self.cfg.damping_rel
        floor = self.cfg.damping_floor
        warnings = []
        for name, c in self.curv.items():
            if not c.seen:
                continue
            c.A_inv, c.tr_A_inv, condA = self._invert_factor(c.A, rel, floor)
            c.S_inv, c.tr_S_inv, condS = self._invert_factor(c.S, rel, floor)
            c.A_inv = c.A_inv.to(self.device)
            c.S_inv = c.S_inv.to(self.device)
            c.cond = max(condA, condS)
            if c.cond > self.cfg.cond_warn:
                warnings.append((name, c.cond))
        return warnings

    # ----- consumo: traza termodinámica y gradiente natural -----
    def inverse_fisher_trace(self, name: str) -> float:
        """Tr(F^{-1}) ≈ Tr(A^{-1})·Tr(S^{-1}) CRUDA. La normalización relativa
        (línea base + mapeo a [0,1]) la hace el controlador térmico."""
        c = self.curv[name]
        return c.tr_A_inv * c.tr_S_inv

    def natural_gradient(self, name: str, grad_W: torch.Tensor) -> torch.Tensor:
        """G -> S^{-1} G A^{-1}. grad_W tiene forma [out, in] (convención de nn.Linear)."""
        c = self.curv[name]
        if c.A_inv is None or c.S_inv is None:
            return grad_W
        return c.S_inv @ grad_W @ c.A_inv

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
