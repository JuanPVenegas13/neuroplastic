# Modelo X — Neuroplasticidad Geométrica para LLMs

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CPU-ee4c2c)
![status](https://img.shields.io/badge/status-research%20prototype-orange)

Prototipo de investigación (PyTorch, corre en CPU y Apple Silicon/MPS) de un
Transformer pequeño con un **lazo de auto-regulación cerrado**:

```
traza de Fisher (K-FAC) → estado termodinámico (congelar / plástico / fundir)
→ SDE de tiempo líquido → adaptación por gradiente natural → detección de deriva
```

La pregunta: ¿puede una red **detectar** que su tarea cambió, **fundir** justo el
sub-módulo responsable y **re-adaptarlo** por geometría (gradiente natural), en vez
de reentrenar a ciegas? Se prueba sobre una tarea sintética de *concept drift*
(predecir `y_t` a partir de `x_t` y `x_{t-lag}`).

> **Estilo del proyecto: medir, no afirmar.** Cada paso se verifica con un experimento
> que imprime `[check]`s; los confounds se nombran, las hipótesis refutadas se reportan
> tal cual, y **no se fabrica ningún ganador**. La honestidad estadística es el punto.

## Problema

El olvido catastrófico se suele atacar con penalizaciones globales (p. ej. EWC). Aquí
se explora una alternativa **geométrica y local**: usar la curvatura (Fisher vía K-FAC)
como termostato que congela lo que ya aprendió y funde selectivamente lo que el drift
invalidó, adaptándolo por gradiente natural en una neurona de **tiempo líquido**
(SDE de salto-difusión).

## Datos

Tarea sintética controlada de *concept drift* (`drift_data.py`): el mapeo `x → y`
depende de un lag estructural que cambia en un instante ("Cisne Negro"), con un
detector de deriva por **z-score de la pérdida** (la KL de covariables se registra
aparte). Sin datos externos: todo es reproducible y con verdad-de-terreno conocida,
ideal para aislar mecanismos.

## Enfoque / Modelo

| Módulo | Subsistema |
|---|---|
| `kfac.py` | K-FAC `F ≈ A⊗S`, submuestreo de tokens, inversión amortizada CPU/float64, gradiente natural `S⁻¹GA⁻¹` |
| `liquid_sde.py` | Neurona de tiempo líquido `dh=[-h/τ+f]dt+σdW+dJ` (operator splitting; salto STE/REINFORCE) |
| `thermodynamics.py` | Controlador térmico FG-SGELU (FROZEN/PLASTIC/MELTED) con histéresis y ventana refractaria |
| `attention.py` · `model.py` | Atención rastreable por K-FAC + bloque Modelo X (SDE líquido en lugar del FFN) |
| `drift_data.py` · `train.py` | Tarea con drift + detector; orquestador del lazo completo |

## Resultados (hallazgos honestos)

Resumen por fase (detalle en `phaseN/README_faseN.md`):

- **El lazo cierra (Fase 2–3).** Arco completo de un Cisne Negro: aprende → consolida
  (FROZEN) → drift estructural → detecta y funde → re-adapta por gradiente natural →
  recupera 100 %. Cada comportamiento exigió corregir un bug real, documentado
  (calibración del paso natural ~10⁵×, damping relativo vs absoluto, covariate vs
  concept drift, confound de la cabeza).
- **La afirmación central, medida (Fase 5).** El no-olvido *simultáneo* es imposible
  con una sola cabeza (A y B se contradicen): se descarta. El test válido es el
  **AHORRO**: en drift reversible A→B→A el modelo re-aprende A **~3.5× más rápido**.
  Ablaciones: **detección y K-FAC son críticos**; la termodinámica da el ahorro; el
  **salto `dJ` NO es crítico**.
- **Navaja de Occam (Fase 6).** Tras un intento genuino de salvarlo, `dJ` no se gana su
  lugar → modo recomendado `jump_sigma=0`. **No se fabricó un ganador** vs EWC.
- **Head-to-head limpio (Fase 8).** Un confound previo ("Adam neutraliza a EWC") resultó
  ser un **artefacto del rango de λ** (Fisher ~1e-7 ⇒ hace falta λ~1e8). Con λ a escala,
  EWC sí protege bajo Adam. Conclusión honesta: **EWC y Modelo X resuelven problemas
  casi disjuntos**, no uno domina al otro.
- **Predicción falsable, REFUTADA (Fase 9).** Engrosar el Fisher de EWC a granularidad
  de bloque (mímica de Modelo X) **destruye** la retención en vez de imitarla:
  per-parámetro y por-bloque **divergen**. Límite estructural localizado: el termostato
  por bloque es demasiado crudo para la holgura sub-bloque.

> Este repo vale como muestra de **rigor experimental**: hipótesis falsables, ablaciones,
> confounds nombrados y resultados negativos reportados — no como un modelo "que gana".

## Stack

Python 3.12 · PyTorch (MPS en Apple Silicon, CPU en cualquier lado) · NumPy · Matplotlib.
Inversiones K-FAC en CPU/float64 (MPS no soporta float64); cómputo caliente en float32.

## Cómo correr

Las importaciones son **relativas dentro del paquete**, así que se ejecuta como módulo
**desde la raíz del repo** (el directorio que contiene `modelo_x/`):

```bash
pip install -r requirements.txt

python -m modelo_x.run_demo            # demo base: el arco de un Cisne Negro
python -m modelo_x.phase5.run_phase5   # orquestador de una fase (N = 4..9)
python -m modelo_x.phase5.exp_reversible   # experimento suelto: el test de AHORRO
```

En CPU los experimentos tardan de segundos a minutos; en M4/MPS van más rápido. El
racional de diseño de la Fase 2 (por qué PyTorch+MPS y no JAX) está en
`modelo_x/README.md`, y el `porqué` honesto de cada decisión en los `README_faseN.md`.
