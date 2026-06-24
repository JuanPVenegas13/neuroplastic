# Fase 4 — Integración y escalado (con verificación empírica de cada paso)

La consigna de esta fase fue estricta: **verificar que cada paso hace lo que dice, y
si no, ajustar e implementar una alternativa.** Lo que sigue documenta no solo lo que
funcionó, sino los bugs que aparecieron, las hipótesis que se refutaron y las
alternativas que hubo que implementar. Esa es la parte honesta — y la valiosa.

Correr todo: `PYTHONPATH=/home/claude python -m modelo_x.phase4.run_phase4`

---

## 4.1 — Estimador de salto insesgado, integrado en el núcleo

**Qué se hizo.** Se añadió al núcleo (`liquid_sde.py`) una intensidad de salto
*aprendible* `p = σ(jump_logit)` con dos estimadores de gradiente: STE (sesgado,
barato) y REINFORCE (insesgado, score-function). El surrogate de REINFORCE
`(L−b)·Σlogp` se conectó al `Trainer` (`train.py`).

**Verificación (`verify_reinforce.py`).** En el modelo completo (no en un juguete),
se comparó el gradiente de `dE[L]/d(jump_logit)` de cada estimador contra la verdad
por **diferencias finitas**:

| modo | verdad (DF) | estimador | sesgo | std/muestra |
|---|---|---|---|---|
| STE (per-elemento) | 0.0083 | 0.0069 | −0.0014 (signif.) | 0.0034 |
| REINFORCE (per-elemento) | 0.0083 | 0.0109 | +0.0027 (ruido) | 0.296 |
| REINFORCE (compartido) | 0.0085 | 0.0089 | +0.0004 | 0.073 |

- **STE tiene un sesgo pequeño pero estadísticamente significativo** (EE 0.0001).
- **REINFORCE es insesgado** (consistente con la verdad) pero **~87× más ruidoso**.

**Problema encontrado y alternativa implementada.** La varianza de REINFORCE escala
con el número de decisiones Bernoulli independientes (≈B·T·d ≈ 147k por pasada),
volviéndolo poco práctico. La alternativa implementada —**decisión de salto
compartida por posición** (`jump_shared`, una Bernoulli por posición en vez de por
canal)— mantiene el insesgamiento y **reduce la varianza 4×**.

**Verificación en el lazo (`exp_learnable_loop.py`).** Ambos estimadores completan el
arco completo y recuperan (acc 1.00). **Ajuste honesto de una predicción fallida:**
yo esperaba que la tarea *suprimiera* el salto (p→0); no ocurrió — `p` se movió poco
respecto del init. La conclusión correcta, guiada por los datos, es que la tarea
supervisada ejerce **señal débil** sobre la intensidad de salto (solo recibe gradiente
en fases plástica/fundida y el salto no ayuda a la pérdida inmediata), lo que respalda
su rol **exógeno** (re-plastificación ante deriva), no como palanca de la tarea.

---

## 4.2 — Preacondicionador K-FAC acoplado Q–K (selectivo por umbral)

**Qué se hizo.** `coupled_kfac.py` mantiene los factores `A`, `S_QQ`, `S_KK` y el
**cruzado `S_QK`** por capa de atención, y aplica el gradiente natural **conjunto**
sobre `[W_Q; W_K]` cuando el ratio de acoplamiento supera un umbral; diagonal en el
resto. Amortiguamiento adaptativo para que el factor cruzado participe sin explotar.

**Bug encontrado y corregido.** La primera versión (EMA de los factores) reportaba un
ratio de acoplamiento ~0.013, en contradicción con el **0.41 verificado en la Fase
3.2**. Una captura manual limpia confirmó que **la verdad es 0.407** (el probe de 3.2
era correcto). Diagnóstico: el **EMA del factor cruzado `S_QK` es frágil** — las
contribuciones por-lote apuntan en direcciones inconsistentes y el EMA las cancela,
mientras que la **suma acumulada** las acumula coherentemente y coincide con la verdad.
**Alternativa implementada:** se reemplazó el EMA por **sumas corrientes** (con
`reset()` para reflejar la curvatura actual).

**Verificación (`exp_coupled_precond.py`), todos los checks pasan.** Tras el arreglo,
el cuadro **reconcilia con la Fase 3.2**: el acoplamiento Q–K **crece con la
convergencia** (≈0.065 con pesos aleatorios → ≈0.63 al converger; el 0.41 de 3.2 era
el punto intermedio). El gate por umbral se activa al converger; el acoplado está bien
condicionado (cond≈448) y da una mejora pequeña pero real (Δpérdida ≈ +0.003).
Figura: `phase4_coupled_qk.png`.

---

## 4.3 — Curvatura cruzada más allá de Q–K

**Qué se hizo y dos intentos descartados honestamente.** Se quiso medir el
acoplamiento de curvatura entre *otras* sublocaciones (atención↔SDE en serie).
- Intento 1 (ρ_s, correlación de gradientes de salida): **engañoso** — los pares en
  serie dan ρ_s altísimo por la regla de la cadena (el grad de salida de una capa es
  el de la siguiente retropropagado), sin que eso implique curvatura cruzada real.
- Intento 2 (ρ_A·ρ_s heurístico): también dudoso (ρ_A no despreciable en serie porque
  las activaciones están correlacionadas a través de la red).

**Medición correcta (`exp_cross_curvature.py`).** El bloque cruzado del Fisher
empírico `C₁₂ = E[g₁g₂ᵀ]` (gradientes por-muestra) tiene tamaño relativo igual al
**coseno de Frobenius entre las matrices de Gram por-muestra** (identidad exacta),
con normalización a norma unidad para quitar el confound de dificultad por-muestra.

**Hallazgo (refuta la hipótesis inicial, honestamente).** La curvatura cruzada
direccional es **alta en todo el bloque**, y **mayor en el camino en serie**
(media ≈0.73) **que dentro del trío QKV** (media ≈0.39). Es decir, el supuesto
**bloque-diagonal de K-FAC es más crudo de lo que asumía la Fase 1.**

**Matiz clave que rescata 4.2.** Solo los pares con **entrada compartida** (q,k,v
sobre `ln1(x)`) tienen curvatura cruzada **Kronecker-factorizable** (`A⊗S₁₂`), el único
caso corregible barato — y es justo lo que hace 4.2 para Q–K. El acoplamiento en serie,
aun siendo mayor en magnitud, **no factoriza limpio** (entradas distintas) y queda
**fuera del alcance de K-FAC** sin una aproximación distinta y más costosa: una
limitación estructural que conviene registrar, no algo que 4.2 resuelva.

---

## 4.4 — Escalado real

**Qué se hizo (`exp_scale_real.py`).** Se corrió el lazo completo a mayor escala
(6 capas, d=96, T=48; 342k params) con submuestreo de tokens al 20% (validado en 3.4),
midiendo tiempo real en CPU y la supervivencia del arco.

**Problema encontrado y alternativa.** A escala, el modelo de 6 capas **sí aprendía y
consolidaba** la primera tarea, pero **no re-adaptaba** tras el Cisne Negro con el
calendario heredado del modelo de 2 capas. Diagnóstico: el modelo profundo converge y
**re-adapta ~2–3× más lento**, y se re-congelaba antes de recuperar. **Alternativa:**
escalar el **calendario** (pasos, ventana post-deriva, `freeze_warmup`) con la
profundidad.

**Verificación (todos los checks pasan).** Con el calendario escalado:
- **Coste sublineal:** 6.3× params → ~4.9× ms/paso (el submuestreo cumple).
- **El arco sobrevive a 6 capas:** detecta el Cisne Negro (pico ≈10.8), funde
  **selectivamente** (re-plastificación heterogénea: unas capas funden, otras quedan
  congeladas) y **re-adapta hasta recuperar** (acc 0.07 → 0.99).
- **Honestidad sobre el sandbox:** es CPU; los tiempos absolutos no representan
  GPU/MPS, pero el **escalado relativo** y la **supervivencia funcional** del lazo sí.

---

## Síntesis de la Fase 4

Las cuatro piezas quedaron integradas y verificadas empíricamente. El patrón se
repitió: cada afirmación se midió, varias fallaron en su forma original, y el valor
estuvo en **ajustar honestamente**:

1. **El estimador insesgado funciona pero es ruidoso** → alternativa de decisión
   compartida (4× menos varianza); y el salto resultó ser **exógeno**, no una palanca
   de tarea (predicción corregida con datos).
2. **El acoplado Q–K funciona tras corregir un bug real** (EMA frágil → sumas), y
   reconcilia con 3.2 (el acoplamiento **crece con la convergencia**).
3. **El bloque-diagonal es más crudo de lo asumido** (curvatura cruzada alta, mayor en
   serie) — pero solo Q–K es **factorizable y corregible barato**; el resto es una
   limitación estructural registrada.
4. **El lazo escala (coste sublineal) y el arco sobrevive en profundidad**, con la
   condición de que el **calendario escale con la profundidad**.

Ninguna de estas conclusiones estaba garantizada de antemano; todas salieron de medir,
fallar y corregir.
