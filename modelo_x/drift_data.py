"""Tarea sintética con concept drift + detector de deriva (colector estadístico).

Tarea: y_t = (x_t + x_{t-1} + shift) mod V. La regla `shift` cambia en
`drift_step` (el "Cisne Negro"). Necesita atención (mira al token previo) y la
deriva provoca un salto en la distribución de la pérdida/gradiente.

Detector: el "colector" es una gaussiana diagonal (μ, var) sobre los embeddings,
actualizada por EMA. drift_score = D_KL(batch || colector) (proxy fiel al Pilar B).
El salto se dispara cuando drift_score supera media+z·desv (z-score adaptativo,
sin umbral mágico)."""
from __future__ import annotations

import torch

from .config import DriftConfig, ModelConfig


class DriftTask:
    def __init__(self, mcfg: ModelConfig, dcfg: DriftConfig, device, seed: int = 0):
        self.m = mcfg
        self.d = dcfg
        self.device = device
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def lag(self, step: int) -> int:
        return self.d.lag_pre if step < self.d.drift_step else self.d.lag_post

    def batch(self, step: int, bs: int):
        V, T = self.m.vocab_size, self.m.seq_len
        x = torch.randint(0, V, (bs, T), generator=self.g)
        lag = self.lag(step)
        xlag = torch.roll(x, shifts=lag, dims=1)
        xlag[:, :lag] = 0
        y = (x + xlag + self.d.shift) % V
        return x.to(self.device), y.to(self.device)


class ManifoldDriftDetector:
    """Colector estadístico gaussiano diagonal + disparador de salto por z-score."""

    def __init__(self, d_model: int, dcfg: DriftConfig, device):
        self.d = dcfg
        self.mu = torch.zeros(d_model, device=device)
        self.var = torch.ones(d_model, device=device)
        self.inited = False
        # estadística del propio drift_score para el z-score adaptativo
        self.score_mu = 0.0
        self.score_var = 1.0
        self.score_inited = False

    @staticmethod
    def _kl_diag(mu0, var0, mu1, var1):
        """D_KL(N0 || N1) para gaussianas diagonales, promediado por dimensión."""
        eps = 1e-6
        var0 = var0 + eps; var1 = var1 + eps
        term = (var0 / var1) + (mu1 - mu0) ** 2 / var1 - 1.0 + torch.log(var1 / var0)
        return float(0.5 * term.mean())

    def update(self, emb: torch.Tensor, prev_loss: float | None):
        """emb: [B,T,d]; prev_loss: pérdida del paso anterior (señal de concept drift).
        Devuelve (drift_score, jump_detected, covariate_kl).

        El concept drift (cambio del mapeo x->y) NO aparece en la distribución de
        entradas, así que se detecta por un SALTO en la pérdida (z-score adaptativo).
        La KL de embeddings se conserva como señal de covariate drift (registro)."""
        flat = emb.detach().reshape(-1, emb.shape[-1])
        bmu = flat.mean(0)
        bvar = flat.var(0, unbiased=False)
        covariate_kl = 0.0
        if not self.inited:
            self.mu.copy_(bmu); self.var.copy_(bvar); self.inited = True
        else:
            covariate_kl = self._kl_diag(bmu, bvar, self.mu, self.var)
            a = self.d.manifold_ema
            self.mu.mul_(a).add_(bmu, alpha=1 - a)
            self.var.mul_(a).add_(bvar, alpha=1 - a)

        if prev_loss is None:
            return 0.0, False, covariate_kl
        if not self.score_inited:
            self.score_mu = prev_loss
            self.score_var = max(prev_loss * 0.1, 1e-2) ** 2
            self.score_inited = True
            return 0.0, False, covariate_kl
        std = max(self.score_var ** 0.5, 1e-6)
        z = (prev_loss - self.score_mu) / std
        jump = z > self.d.jump_z
        b = 0.95
        self.score_mu = b * self.score_mu + (1 - b) * prev_loss
        self.score_var = b * self.score_var + (1 - b) * (prev_loss - self.score_mu) ** 2
        return float(z), bool(jump), covariate_kl
