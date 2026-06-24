"""Fase 3.2 — Curvatura fuera de la diagonal entre W_Q y W_K.

K-FAC por bloques asume que la curvatura entre proyecciones distintas es nula. Pero
en la atención, Q y K interactúan multiplicativamente en QKᵀ y comparten la MISMA
entrada x, así que su gradiente es g = a ⊗ s (a = activación de entrada, s = grad de
salida). El acoplamiento cruzado se factoriza como:

    E[g_Q g_Kᵀ] = E[(a⊗s_Q)(a⊗s_K)ᵀ] ≈ (A) ⊗ (S_QK),   S_QK = E[s_Q s_Kᵀ]

con A compartido (misma entrada). Por tanto TODO el acoplamiento vive en el factor
cruzado de salida S_QK. Medimos su tamaño relativo y, si no es despreciable,
construimos el preacondicionador ACOPLADO sobre [W_Q; W_K] y comparamos su dirección
contra la diagonal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QKCouplingProbe:
    """Engancha las Linear q,k de cada bloque y acumula A, S_QQ, S_KK, S_QK."""

    def __init__(self, model):
        self.model = model
        self.stats = {}     # idx -> dict de factores
        self._cache = {}    # (idx,'q'/'k') -> (a, s)
        self._handles = []
        for b in model.blocks:
            i = b.idx
            self.stats[i] = {"A": 0.0, "Sqq": 0.0, "Skk": 0.0, "Sqk": 0.0, "n": 0}
            self._handles.append(b.attn.q.register_forward_hook(self._fwd(i, "q")))
            self._handles.append(b.attn.k.register_forward_hook(self._fwd(i, "k")))
            self._handles.append(b.attn.q.register_full_backward_hook(self._bwd(i, "q")))
            self._handles.append(b.attn.k.register_full_backward_hook(self._bwd(i, "k")))

    def _fwd(self, i, which):
        def hook(_m, inp, _out):
            self._cache[(i, which, "a")] = inp[0].detach().reshape(-1, inp[0].shape[-1])
        return hook

    def _bwd(self, i, which):
        def hook(_m, _gi, go):
            self._cache[(i, which, "s")] = go[0].detach().reshape(-1, go[0].shape[-1])
            # cuando tenemos s_q y s_k del mismo paso, acumulamos
            if (i, "q", "s") in self._cache and (i, "k", "s") in self._cache:
                a = self._cache[(i, "q", "a")]      # a compartido (q y k ven la misma x)
                sq = self._cache[(i, "q", "s")]
                sk = self._cache[(i, "k", "s")]
                st = self.stats[i]
                n = sq.shape[0]
                st["A"] = st["A"] + (a.t() @ a)
                st["Sqq"] = st["Sqq"] + (sq.t() @ sq)
                st["Skk"] = st["Skk"] + (sk.t() @ sk)
                st["Sqk"] = st["Sqk"] + (sq.t() @ sk)
                st["n"] += n
                # limpia para el siguiente paso
                for w in ("q", "k"):
                    self._cache.pop((i, w, "s"), None)
        return hook

    def coupling_ratios(self) -> dict[int, float]:
        """||S_QK||_F / sqrt(||S_QQ||_F · ||S_KK||_F) por bloque (0=desacoplado)."""
        out = {}
        for i, st in self.stats.items():
            if st["n"] == 0:
                continue
            sqk = st["Sqk"].norm().item()
            sqq = st["Sqq"].norm().item()
            skk = st["Skk"].norm().item()
            out[i] = sqk / (max(sqq * skk, 1e-12) ** 0.5)
        return out

    @staticmethod
    def _inv_adaptive(M: torch.Tensor, rel: float = 0.05, floor: float = 1e-12):
        """Inversa con amortiguamiento RELATIVO al espectro (lección de la Fase 2:
        un damping absoluto fijo aplasta factores diminutos y borra su estructura)."""
        d = M.shape[0]
        damp = rel * float(M.diagonal().sum()) / d + floor
        return torch.linalg.inv(M + damp * torch.eye(d, dtype=M.dtype))

    def natural_grad_comparison(self, i: int):
        """Compara la dirección del gradiente natural ACOPLADO vs DIAGONAL para [W_Q;W_K]
        del bloque i, usando los grads reales de las Linear. Devuelve coseno y norma rel."""
        b = next(bb for bb in self.model.blocks if bb.idx == i)
        Gq = b.attn.q.weight.grad
        Gk = b.attn.k.weight.grad
        if Gq is None or Gk is None:
            return None
        st = self.stats[i]
        n = st["n"]
        A = (st["A"] / n).cpu().double()
        Sqq = (st["Sqq"] / n).cpu().double()
        Skk = (st["Skk"] / n).cpu().double()
        Sqk = (st["Sqk"] / n).cpu().double()
        d = A.shape[0]
        A_inv = self._inv_adaptive(A)

        # diagonal: nat = S_·· ^{-1} G A^{-1}
        Sqq_inv = self._inv_adaptive(Sqq)
        Skk_inv = self._inv_adaptive(Skk)
        nq_diag = Sqq_inv @ Gq.detach().cpu().double() @ A_inv
        nk_diag = Skk_inv @ Gk.detach().cpu().double() @ A_inv

        # acoplado: S_joint (2d x 2d) sobre el espacio de salida apilado [s_q; s_k]
        Sj = torch.zeros(2 * d, 2 * d, dtype=torch.float64)
        Sj[:d, :d] = Sqq; Sj[d:, d:] = Skk
        Sj[:d, d:] = Sqk; Sj[d:, :d] = Sqk.t()
        Sj_inv = self._inv_adaptive(Sj)
        Gstack = torch.cat([Gq.detach().cpu().double(), Gk.detach().cpu().double()], dim=0)  # 2d x d
        nj = Sj_inv @ Gstack @ A_inv
        nq_coup, nk_coup = nj[:d], nj[d:]

        v_diag = torch.cat([nq_diag.flatten(), nk_diag.flatten()])
        v_coup = torch.cat([nq_coup.flatten(), nk_coup.flatten()])
        cos = F.cosine_similarity(v_diag, v_coup, dim=0).item()
        rel = (v_coup - v_diag).norm().item() / (v_coup.norm().item() + 1e-12)
        return {"cosine": cos, "rel_diff": rel}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
