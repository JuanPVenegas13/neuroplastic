"""Fase 4.2 — Preacondicionador K-FAC ACOPLADO para el par W_Q, W_K.

La Fase 3.2 midió que el acoplamiento Q–K no es despreciable (ratio ~0.41) y que
ignorarlo desalinea el gradiente natural ~7°. Aquí lo incorporamos: mantenemos los
factores A (compartido), S_QQ, S_KK y el CRUZADO S_QK por capa de atención y, donde
el ratio de acoplamiento supere un umbral, aplicamos el gradiente natural CONJUNTO
sobre [W_Q; W_K]; en el resto, el diagonal. Amortiguamiento adaptativo (lección
transversal de la Fase 2) para que el factor cruzado participe sin explotar.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _inv_adaptive(M, rel=0.05, floor=1e-9):
    d = M.shape[0]
    damp = rel * float(M.diagonal().sum()) / d + floor
    inv = torch.linalg.inv(M + damp * torch.eye(d, dtype=M.dtype, device=M.device))
    cond = float(torch.linalg.cond(M + damp * torch.eye(d, dtype=M.dtype, device=M.device)))
    return inv, cond


class CoupledQKManager:
    """Factores por capa de atención (A compartido, S_QQ, S_KK y el CRUZADO S_QK);
    gradiente natural diagonal o acoplado según un umbral de acoplamiento.

    Acumulación por SUMAS corrientes (con reset periódico), NO por EMA: se verificó
    que el EMA del factor cruzado S_QK es frágil (las contribuciones por lote apuntan
    en direcciones inconsistentes y el EMA las cancela, dando ratios espurios ~0.03),
    mientras que la suma las acumula coherentemente y coincide con la verdad por
    captura manual (ratio 0.407). La suma es además invariante de escala para el ratio
    y se normaliza por el conteo en el gradiente natural."""

    def __init__(self, model, couple_threshold=0.15):
        self.model = model
        self.thr = couple_threshold
        self.st = {}
        self._cache = {}
        self._h = []
        for b in model.blocks:
            i = b.idx
            self.reset_block(i)
            self._h.append(b.attn.q.register_forward_hook(self._fwd(i, "q")))
            self._h.append(b.attn.k.register_forward_hook(self._fwd(i, "k")))
            self._h.append(b.attn.q.register_full_backward_hook(self._bwd(i, "q")))
            self._h.append(b.attn.k.register_full_backward_hook(self._bwd(i, "k")))

    def reset_block(self, i):
        self.st[i] = {"A": 0.0, "Sqq": 0.0, "Skk": 0.0, "Sqk": 0.0, "n": 0}

    def reset(self):
        for i in list(self.st):
            self.reset_block(i)

    def _fwd(self, i, w):
        def hook(_m, inp, _o):
            self._cache[(i, w, "a")] = inp[0].detach().reshape(-1, inp[0].shape[-1])
        return hook

    def _bwd(self, i, w):
        def hook(_m, _gi, go):
            self._cache[(i, w, "s")] = go[0].detach().reshape(-1, go[0].shape[-1])
            if (i, "q", "s") in self._cache and (i, "k", "s") in self._cache:
                a = self._cache[(i, "q", "a")]
                sq = self._cache[(i, "q", "s")]; sk = self._cache[(i, "k", "s")]
                s = self.st[i]
                s["A"] = s["A"] + a.t() @ a            # SUMAS corrientes (coherentes)
                s["Sqq"] = s["Sqq"] + sq.t() @ sq
                s["Skk"] = s["Skk"] + sk.t() @ sk
                s["Sqk"] = s["Sqk"] + sq.t() @ sk
                s["n"] += sq.shape[0]
                self._cache.pop((i, "q", "s")); self._cache.pop((i, "k", "s"))
        return hook

    def coupling_ratio(self, i):
        s = self.st[i]
        if s["n"] == 0:
            return 0.0
        return s["Sqk"].norm().item() / (max(s["Sqq"].norm().item() * s["Skk"].norm().item(), 1e-12) ** 0.5)

    def natural_grad(self, i, coupled: bool):
        """Devuelve (nat_Q, nat_K, cond). Si coupled y ratio>umbral usa el conjunto."""
        s = self.st[i]
        b = next(bb for bb in self.model.blocks if bb.idx == i)
        Gq = b.attn.q.weight.grad; Gk = b.attn.k.weight.grad
        if Gq is None or Gk is None or s["n"] == 0:
            return None, None, 1.0
        nrm = s["n"]
        A = (s["A"] / nrm).double(); Sqq = (s["Sqq"] / nrm).double()
        Skk = (s["Skk"] / nrm).double(); Sqk = (s["Sqk"] / nrm).double()
        d = A.shape[0]
        A_inv, _ = _inv_adaptive(A)
        use_coupled = coupled and self.coupling_ratio(i) > self.thr
        if not use_coupled:
            Sqq_i, c1 = _inv_adaptive(Sqq); Skk_i, c2 = _inv_adaptive(Skk)
            nq = Sqq_i @ Gq.double() @ A_inv
            nk = Skk_i @ Gk.double() @ A_inv
            return nq.float(), nk.float(), max(c1, c2)
        Sj = torch.zeros(2 * d, 2 * d, dtype=torch.float64)
        Sj[:d, :d] = Sqq; Sj[d:, d:] = Skk; Sj[:d, d:] = Sqk; Sj[d:, :d] = Sqk.t()
        Sj_i, cj = _inv_adaptive(Sj)
        G = torch.cat([Gq.double(), Gk.double()], 0)
        nj = Sj_i @ G @ A_inv
        return nj[:d].float(), nj[d:].float(), cj

    def remove(self):
        for h in self._h:
            h.remove()
        self._h.clear()
