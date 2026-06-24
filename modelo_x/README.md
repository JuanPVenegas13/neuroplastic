# Modelo X — Framework de Neuroplasticidad Geométrica para LLMs
### Prototipo de la Fase 2 (PyTorch + Apple Silicon / MPS)

Implementación modular y ejecutable del Modelo X sobre un Transformer pequeño
(2 capas) entrenado en una tarea sintética con *concept drift*. El objetivo de la
Fase 2 no es rendimiento, sino **validar empíricamente el lazo de retroalimentación**
que la Fase 1 señaló como el mayor riesgo: traza de Fisher → estado térmico →
salto → adaptación → traza.

---

## 1. Por qué PyTorch+MPS y no JAX

La Fase 1 (y la revisión de Gemini) favorecían JAX por `scan`/`vmap`/`diffrax`.
Pero el requisito de correr en **M4** invalida esa elección de raíz:

- El proyecto oficial `jax-metal` quedó sin mantenimiento (diciembre 2025).
- Las alternativas comunitarias (`jax-mps`, `applejax`) están en alfa y, sobre
  Metal, **no soportan float64** y **`eigh`/`SVD`/`QR` dentro de `scan`/`while_loop`
  crashean**. Eso es justo el patrón del Modelo X (descomposición de factores K-FAC
  dentro del desenrollado del SDE).

Por eso el prototipo usa **PyTorch + MPS**, maduro en Apple Silicon. Esto *disuelve*
el cuarto cuello de botella de Gemini (el choque del compilador JAX con tasas
variables) en lugar de resolverlo. La sabiduría de fondo —Δt fijo— se conserva de
todos modos (las formas dinámicas también penalizan bajo `torch.compile`).

> Restricción de Metal heredada: MPS tampoco soporta float64. Por eso las
> inversiones K-FAC (numéricamente delicadas) se hacen en **CPU/float64** y están
> **amortizadas** (cada `invert_every` pasos), mientras todo el cómputo caliente
> vive en float32 sobre el dispositivo.

## 2. Mapa de módulos

| Módulo | Subsistema de la especificación |
|---|---|
| `kfac.py` | K-FAC: F ≈ A⊗S, estrategia *expand*, submuestreo de tokens, inversión CPU/float64, traza Tr(A⁻¹)Tr(S⁻¹), gradiente natural S⁻¹GA⁻¹ |
| `liquid_sde.py` | Neurona de tiempo líquido: dh=[-h/τ+f]dt+σdW+dJ vía *operator splitting*; salto Poisson con gradiente **STE** |
| `thermodynamics.py` | FG-SGELU + controlador térmico (FROZEN/PLASTIC/MELTED) con **histéresis** y ventana refractaria |
| `attention.py` | Auto-atención con proyecciones rastreables por K-FAC |
| `model.py` | Bloque Modelo X (atención + SDE líquido en lugar del FFN + compuerta FG-SGELU) y el Transformer |
| `drift_data.py` | Tarea con *concept drift* estructural + detector de deriva (colector gaussiano + z-score de pérdida) |
| `train.py` | Orquestador del lazo completo y evaluación determinista |
| `run_demo.py` | Corre el experimento y guarda `feedback_loop.csv` + `feedback_loop.png` |

## 3. Cómo correr

```bash
pip install -r requirements.txt
python -m modelo_x.run_demo          # detecta MPS en M4 automáticamente
```

Salidas: traza por consola del lazo, `feedback_loop.csv` y `feedback_loop.png`.

## 4. Cómo se incorporaron las sugerencias de Gemini

1. **Gradiente del salto → STE.** `liquid_sde._JumpGateSTE`: forward duro,
   backward con surrogate sigmoide (mismo patrón que los *surrogate gradients* de
   redes de pulsos). El *adjoint* riguroso queda para la Fase 3. **Matiz:** el sesgo
   del STE es real y es justo lo que hay que medir; por eso es aislable y debe ser
   lo primero a validar (ver §6).
2. **Submuestreo de tokens en K-FAC.** `kfac._subsample` toma una fracción de las
   B·T posiciones. **Matiz/corrección:** como A y S son esperanzas *separadas* bajo
   la aproximación de Kronecker, **no** requieren los mismos índices; se submuestrean
   de forma independiente sin sesgo adicional.
3. **Monitoreo de salud de la traza.** En vez del estimador de Hutchinson (que para
   la *inversa* requiere resolver sistemas y es algo circular), se monitorea
   directamente el **número de condición** de los factores regularizados —barato,
   ya que sale de la `eigh` que de todos modos calculamos— y emite alerta `⚠cond`.
   Atrapa el mismo modo de fallo (MELTED permanente por colapso numérico).
4. **Trampa de JIT en JAX.** Disuelta por el cambio a PyTorch (ver §1).

## 5. Qué demuestra el experimento (hallazgos honestos)

El arco completo es visible en `feedback_loop.png`:

1. **Aprende** (PLÁSTICO): 100% de accuracy hacia el paso ~125.
2. **Consolida** (FROZEN tras el *warmup*): la pérdida cae a ~2e-3 y se estabiliza.
3. **Cisne Negro** (paso 300, drift *estructural* lag 1→2): la pérdida estalla a
   ~12 y la accuracy cae a azar.
4. **Detecta y funde** el núcleo (MELTED, ventana refractaria).
5. **Re-adapta** el núcleo vía gradiente natural y recupera 100%.
6. **Reconsolida** de forma heterogénea (un bloque vuelve a congelarse, el otro
   permanece plástico).

Cada uno de estos comportamientos requirió corregir un defecto real, todos
documentados en el código:

- **Calibración del gradiente natural.** La norma del paso natural resultó ~10⁵×
  la del gradiente crudo (los valores propios diminutos de los factores amplifican
  la inversión). Con `lr_kfac=1.0` el modelo no aprendía nada; con `0.1` (más
  recorte de norma a 10) aprende perfecto. Calibrar este escalar es no-negociable.
- **Amortiguamiento adaptativo.** Un damping absoluto fijo (1e-2) saturaba la traza
  (0.9997 constante) → MELTED permanente. Se cambió a damping **relativo** al
  espectro (10% del valor propio medio), devolviéndole rango dinámico a la traza.
- **Covariate vs concept drift.** El detector original vigilaba la distribución de
  *embeddings*, pero el drift está en el *mapeo* x→y, invisible ahí. Se cambió el
  disparador a un **z-score de la pérdida** (la KL de embeddings se conserva solo
  como señal de covariate drift, registrada).
- **Confound de la cabeza.** Un drift de *shift aditivo* (mod V) era una mera
  reetiquetación absorbible por la cabeza de salida sin tocar el núcleo congelado;
  la neuroplasticidad no se ejercía. Se cambió a un drift **estructural** (cambia a
  qué posición debe atender) + **ventana refractaria de fusión**, de modo que la
  recuperación la carga el mecanismo de fusión, no un atajo.
- **Congelar antes de aprender mata.** Sin *warmup*, los bloques se congelaban en el
  paso ~40 en estado de azar y nunca aprendían. Se añadió `freeze_warmup`.
- **Ruido en la medición.** El SDE inyecta ruido en cada forward; medir accuracy con
  ese ruido penaliza artificialmente. Se añadió un modo **determinista** de
  evaluación (apaga difusión y saltos).

## 6. Para la Fase 3

- Sustituir el STE por el *adjoint* estocástico riguroso (o REINFORCE con línea
  base) y **medir el sesgo** contra el STE en la tarea aislada del salto.
- Curvatura fuera de la diagonal por bloques (acoplamiento W_Q–W_K).
- Paso Δt adaptativo (multi-salto) ahora que el lazo está validado.
- Escalar a 4+ capas y secuencias largas para ejercer el submuestreo de tokens.
