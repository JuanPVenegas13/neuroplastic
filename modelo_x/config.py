"""Configuración central de Modelo X.

Todos los hiperparámetros viven aquí (un solo lugar para tocar). La selección de
dispositivo está pensada para Apple Silicon (M4): prioriza MPS, con respaldo a
CUDA y CPU. Nota crítica de Metal: MPS NO soporta float64. Por eso las
inversiones numéricamente delicadas de K-FAC se hacen en CPU/float64 (ver kfac.py),
mientras todo el cómputo "caliente" se queda en float32 sobre el dispositivo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch


def get_device() -> torch.device:
    """Elige el mejor dispositivo disponible. En M4 será 'mps'."""
    if torch.backends.mps.is_available():
        # Habilita respaldo a CPU para ops aún no implementadas en Metal.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class ModelConfig:
    vocab_size: int = 16
    seq_len: int = 24
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    # --- Bloque líquido (SDE de salto-difusión) ---
    sde_steps: int = 3          # subpasos de integración por bloque
    dt: float = 0.25            # Δt FIJO (ver nota JIT/torch.compile en README)
    tau_init: float = 1.0       # constante de tiempo líquida inicial
    jump_mu: float = 0.0        # media de la magnitud del salto ζ
    jump_sigma: float = 0.3     # desviación de la magnitud del salto ζ
    ste_temp: float = 0.1       # temperatura del gradiente surrogate del salto
    max_lambda_dt: float = 1.0  # Fase 3.3: si λ·δt supera esto, subdivide el subpaso
                                #  para captar multi-saltos (1.0 ≈ inerte en el demo)
    # --- Fase 4.1: intensidad de salto APRENDIBLE + estimador del gradiente ---
    learn_jump: bool = False    # activa una compuerta de salto con prob. aprendible
    jump_grad: str = "ste"      # "ste" (sesgado, barato) | "reinforce" (insesgado)
    jump_logit_init: float = -1.0  # σ(-1)≈0.27 prob. de salto inicial
    jump_shared: bool = False   # Fase 4.1: una decisión de salto por posición (reduce varianza)


@dataclass
class KFACConfig:
    ema_decay: float = 0.95     # ρ de la media móvil de los factores A, S
    damping_rel: float = 0.1    # λ relativo al valor propio medio (amortiguamiento adaptativo)
    damping_floor: float = 1e-6  # piso absoluto de Tikhonov (estabilidad numérica)
    invert_every: int = 20      # amortización: invertir cada K pasos
    token_subsample: float = 0.2  # fracción de tokens usados por los factores (Gemini #2)
    cond_warn: float = 1e6      # umbral de número de condición para alerta (Gemini #3)


@dataclass
class ThermoConfig:
    theta_freeze: float = 0.10  # banda inferior (normalizada, adimensional)
    theta_melt: float = 0.55    # banda superior -> histéresis entre ambos
    trace_ema: float = 0.6      # suavizado EMA rápido de la traza por capa
    baseline_ema: float = 0.99  # EMA lento: línea base relativa de la traza (§3.4 opción 3)
    trace_gain: float = 1.0     # ganancia del mapeo tanh(log-ratio) -> [0,1]
    freeze_warmup: int = 200    # no se permite CONGELAR antes de este paso (deja aprender primero)
    melt_hold: int = 60         # ventana refractaria: tras un Cisne Negro, fuerza MELTED
                                #  estos pasos para que el núcleo se re-plastifique de verdad
    eta: float = 0.3            # escala de ruido base de la compuerta/difusión
    eta_min: float = 0.02       # piso del recocido de η
    anneal_steps: int = 600     # horizonte del recocido coseno de η
    min_plastic_frac: float = 0.0  # piso anti-congelamiento (0 en este demo de 2 capas)
    gate_sigma_floor: float = 1e-3  # evita división por cero al endurecer la compuerta


@dataclass
class DriftConfig:
    drift_step: int = 300       # paso donde ocurre el "Cisne Negro" sintético
    lag_pre: int = 1            # régimen previo: y_t depende de x_{t-1}
    lag_post: int = 2           # régimen posterior: y_t depende de x_{t-2}
                                #  (drift ESTRUCTURAL: cambia a qué posición atender,
                                #   no absorbible por la cabeza -> exige adaptar el núcleo)
    shift: int = 3              # constante aditiva fija (común a ambos regímenes)
    manifold_ema: float = 0.97  # EMA del colector estadístico (μ, var por dim)
    jump_z: float = 4.0         # z-score de drift_score que dispara el salto


@dataclass
class TrainConfig:
    steps: int = 600
    batch_size: int = 32
    lr_kfac: float = 0.1        # lr del gradiente natural (calibrado: la norma del paso
                                #  natural es ~10^5x la del crudo; con recorte a 10, 0.1 es estable)
    lr_base: float = 5e-3       # lr de params no-KFAC (Adam: embeddings, norms, head)
    seed: int = 0
    log_every: int = 25


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    kfac: KFACConfig = field(default_factory=KFACConfig)
    thermo: ThermoConfig = field(default_factory=ThermoConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
