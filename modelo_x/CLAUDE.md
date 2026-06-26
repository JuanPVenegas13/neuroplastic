# Modelo X — Neuroplasticidad Geométrica para LLMs

Prototipo de investigación en PyTorch (corre en CPU y en Apple Silicon / MPS). Un
Transformer pequeño con un lazo de auto-regulación: traza de Fisher (K-FAC) → estado
termodinámico (congelar / plástico / fundir) → SDE de tiempo líquido → adaptación por
gradiente natural → detección de deriva. Probado sobre una tarea sintética de
*concept drift* (predecir `y_t` a partir de `x_t` y `x_{t-lag}`).

> **Antes de hacer nada, lee los READMEs de fase en orden:** `phase3/README_fase3.md`,
> `phase4/README_fase4.md`, `phase5/README_fase5.md`, `phase6/README_fase6.md`,
> `phase7/README_fase7.md`. Contienen el *porqué* honesto de cada decisión (incluidos
> los bugs encontrados y las hipótesis refutadas), que NO se deduce solo del código.

---

## Estilo de trabajo (IMPORTANTE — respetarlo)

**Medir, no afirmar.** Cada paso se verifica empíricamente con un experimento que
imprime `[check]`s. Si una afirmación falla, se **ajusta y se implementa una
alternativa**, documentando con honestidad lo que NO funcionó. No inflar resultados,
no esconder confounds, no amañar tareas para que un mecanismo gane. Cuando un resultado
es inconcluso o incómodo, se dice tal cual y se nombra el siguiente paso. Las
conclusiones se ganan con datos, no se asumen.

---

## Estado del proyecto (resumen honesto por fase)

- **Fases 2–3 (núcleo + aproximaciones).** Lazo completo implementado y verificado:
  K-FAC con submuestreo de tokens e inversión amortizada en CPU/float64; termodinámica
  FG-SGELU (congelar/plástico/fundir con histéresis); SDE de salto-difusión de tiempo
  líquido; detección de deriva por z-score. Las tres aproximaciones temidas resultaron
  manejables/medibles; el riesgo real es la salud numérica del condicionamiento
  (amortiguamiento adaptativo es prerequisito transversal).
- **Fase 4 (integración + escalado).** Estimador de salto insesgado (REINFORCE) vs STE
  verificado in-model; preacondicionador acoplado Q–K (bug de EMA frágil encontrado y
  corregido con sumas corrientes; el acoplamiento crece con la convergencia);
  curvatura cruzada medida rigurosamente (el bloque-diagonal es más crudo de lo
  asumido, pero solo Q–K es Kronecker-corregible); escalado con coste sublineal y arco
  superviviente (el calendario de consolidación debe escalar con la profundidad).
- **Fase 5 (afirmación central).** El no-olvido SIMULTÁNEO es imposible por
  construcción (una cabeza, sin task-id; A y B se contradicen). El test válido es el
  **AHORRO**: en deriva reversible A→B→A el modelo re-aprende A ~3.5× más rápido (50
  pasos de ahorro vs 10 del siempre-plástico). Ablaciones: **detección y K-FAC son
  críticos**; la **termodinámica** da el ahorro; el **salto dJ NO es crítico**.
- **Fase 6 (navaja de Occam + literatura).** Tras un esfuerzo genuino por salvarlo, el
  **salto dJ no se gana su lugar** → modo recomendado `jump_sigma=0` (elimina STE,
  REINFORCE, cabeza aprendible, subdivisión). La comparación con **EWC** quedó
  **confundida por el optimizador**: bajo Adam (único con el que esta arquitectura
  entrena) la penalización de EWC se neutraliza; bajo SGD la arquitectura no entrena.
  No se fabricó un ganador.
- **Fase 7 (robustez + cabos).** Cadena 1→2→3→1: la consolidación **acumula sin
  interferir** (sigue habiendo ahorro al volver a lag=1). Deriva gradual: el detector
  aguanta. Curvatura cruzada en serie: **41° de desvío, O(d⁶), no factoriza** →
  formalizada como límite estructural. Código **MPS-ready por construcción** (float32
  en el dispositivo; float64 solo en CPU para invertir).
- **Fase 8 (head-to-head limpio contra EWC + MPS real).** Al correr por primera vez en
  M4 afloró un **bug MPS en `kfac.py`** (cast directo float64 sobre tensor MPS) →
  corregido (CPU antes de castear); ahora el lazo corre de verdad en `mps`. El
  head-to-head **corrige a 6.3**: su "Adam neutraliza a EWC" era un **artefacto del
  rango de λ** (Fisher ~1e-7 ⇒ hace falta λ~1e8; 6.3 barrió hasta 1e6). Con λ a escala,
  **EWC sí protege A bajo Adam** (acoplado o desacoplado AdamW-style). Conclusión honesta:
  **EWC y Modelo X resuelven problemas casi disjuntos** — EWC retiene en drift
  *absorbible* (frontera dura, sin retención simultánea para nadie en una sola cabeza;
  Modelo X clavado en el olvido de naive ahí, nunca lo reclamó); Modelo X navega drift
  *estructural* por gradiente natural y ahorra al revisitar, **donde EWC/Adam ni siquiera
  aprende la tarea nueva**. No se fabricó un ganador.
- **Fase 9 (bloque vs por-parámetro, retención alcanzable).** Arquitectura SEPARABLE
  (tronco compartido + una cabeza por tarea) para que la retención simultánea SÍ sea
  posible y comparar el congelamiento por **bloque** (Modelo X) contra el soft-freeze
  por-**parámetro** (EWC). Tres hipótesis cayeron, medidas: (a) multi-cabeza es
  **necesario pero no suficiente** — con tronco justo (d=64) nadie retiene (cuello de
  botella = tronco compartido); (b) la retención simultánea exige **capacidad sobrante
  + anclaje fuerte** (d=256, EWC λ~1e10 → min(ret_A,acc_B)≈0.65); (c) **predicción
  falsable REFUTADA**: engrosar el Fisher de EWC a granularidad de bloque (mímica de
  Modelo X) **destruye** la retención (min 0.65→0.25), al nivel del fracaso de Modelo X
  (ret_A≈0.10). Per-param y bloque **divergen, no convergen**. Limitación concreta
  localizada: el termostato por **bloque** es demasiado crudo para la holgura sub-bloque
  que la retención simultánea exige. Sin ganador fabricado.

---

## Cómo correr

Las importaciones son **relativas dentro del paquete** (`from ..config import ...`), así
que hay que ejecutar como módulo **desde el directorio que CONTIENE la carpeta
`modelo_x/`** (no desde dentro de ella).

```bash
# 1) dependencias (en M4, `pip install torch` ya trae MPS)
pip install -r modelo_x/requirements.txt        # torch, matplotlib

# 2) desde el directorio PADRE de modelo_x/:
python -m modelo_x.run_demo                      # demo base (arco de un Cisne Negro)
python -m modelo_x.phase4.run_phase4             # orquestador de cada fase (N = 4..7)
python -m modelo_x.phase5.run_phase5
python -m modelo_x.phase6.run_phase6
python -m modelo_x.phase7.run_phase7

# experimentos sueltos:
python -m modelo_x.phase5.exp_reversible         # p.ej. el test de ahorro
```

Notas: en CPU los experimentos son lentos (segundos a minutos); en M4/MPS deberían ir
más rápido. Varios scripts generan figuras `.png` en el directorio actual.

---

## Mapa de archivos

```
modelo_x/
  config.py          # todas las dataclasses de config + get_device() (prioriza MPS)
  model.py           # ModeloX y ModeloXBlock (attn + SDE líquido + compuerta térmica)
  attention.py       # MHA con q/k/v/out rastreables por K-FAC
  liquid_sde.py      # celda SDE de tiempo líquido (drift + difusión + salto); salto
                     #   aprendible STE/REINFORCE (Fase 4; recomendado OFF, Fase 6)
  thermodynamics.py  # FG-SGELU + controlador térmico (congelar/plástico/fundir)
  kfac.py            # gestor K-FAC: hooks, submuestreo, inversión CPU/float64 amortizada
  drift_data.py      # DriftTask + detector de deriva (KL de covariables + z-score)
  train.py           # Trainer (lazo completo); surrogate REINFORCE opcional
  run_demo.py        # demo de la Fase 2 (genera feedback_loop.png/csv)
  README.md          # racional Fase 2 (por qué PyTorch+MPS y no JAX)
  phase3/  ...  phase7/   # cada una con run_phaseN.py, README_faseN.md y experimentos
```

Subclases/infra reutilizadas: `phase5/ablated_trainer.py` (AblatedTrainer + ReversibleTask
+ steps_to_acc), `phase6/taskid_data.py` (tarea con identificador), `phase7/multi_drift.py`
(MultiDriftTask + GradualDriftTask), `phase8/decoupled_ewc.py` (EWC desacoplado AdamW-style
+ Fisher verdadero/empírico + eval enmascarada). El modo recomendado de operación es
`ablation="no_jump"` (equivalente a `jump_sigma=0`), por la navaja de la Fase 6.

---

## Próximos pasos pendientes (NO deducibles del código)

1. **~~Head-to-head limpio contra EWC.~~ HECHO (Fase 8).** Se resolvió midiendo: el
   confound de 6.3 era un **artefacto del rango de λ**, no el optimizador (Fisher ~1e-7 ⇒
   hace falta λ~1e8). EWC protege A bajo Adam (acoplado o desacoplado AdamW-style); ver
   `phase8/README_fase8.md`. Resultado: EWC y Modelo X son **casi disjuntos** (EWC retiene
   en drift absorbible, frontera dura; Modelo X navega drift estructural + ahorro).
2. **Validar rendimiento REAL en M4/MPS — PARCIAL (Fase 8.0).** Ya corre en `mps`
   (`get_device()=='mps'`); se encontró y corrigió un bug float64→MPS en `kfac.py`.
   **Falta** medir ms/paso CPU vs MPS de forma sistemática (`run_demo` con timing).
3. **~~Benchmark con cabezas por tarea (retención alcanzable).~~ HECHO (Fase 9).** Multi-
   cabeza separable; resultado: la retención simultánea exige capacidad + anclaje
   por-parámetro, y el congelamiento por **bloque** de Modelo X es demasiado crudo
   (refuta la predicción "EWC→hard-freeze"). Ver `phase9/README_fase9.md`.
4. **(Cabo de F9) Termostato con granularidad SUB-BLOQUE.** Congelar por fila/canal según
   curvatura local (no por bloque entero) y re-correr el test 9.3: ¿cierra la brecha con
   EWC por-parámetro? Es la mejora de diseño que el resultado de F9 señala directamente.
5. **(Opcional) Salir del juguete.** Tarea más realista que el lag-shift sintético
   (p.ej. texto a nivel carácter con deriva real) para reforzar la validez externa.
6. **(Opcional) Medir ms/paso CPU vs MPS** sistemático (cierre cuantitativo del paso #2).
7. **(Opcional) Calendario de consolidación principiado** que escale con la profundidad
   automáticamente (en la Fase 4.4 se ajustó a mano).
