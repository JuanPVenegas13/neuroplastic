# Fase 5 — Validación de la afirmación central

Las fases 1–4 construyeron y verificaron los *mecanismos* (K-FAC, SDE de salto, FG-SGELU,
detección) y sus *aproximaciones*. Pero la promesa de cabecera del framework —
**neuroplasticidad en tiempo real SIN olvido catastrófico** mediante congelamiento
selectivo— nunca se había puesto a prueba directamente: todo se midió sobre una sola
deriva y sin re-evaluar jamás la tarea vieja. La Fase 5 cierra ese hueco.

Correr todo: `PYTHONPATH=/home/claude python -m modelo_x.phase5.run_phase5`

---

## 5.1 — Olvido catastrófico: por qué el test obvio está mal planteado

**Qué se hizo (`exp_forgetting.py`).** Entrenar A (lag=1) → consolidar → Cisne Negro →
adaptar B (lag=2); luego re-evaluar A. Comparar el Modelo X completo contra la ablación
siempre-plástica.

**Resultado honesto.** Retención de A ≈ azar (0.10) en **ambos**. El congelamiento no
mostró ventaja. Pero la causa no es un fallo del framework: **A (lag=1) y B (lag=2)
exigen salidas DISTINTAS para las MISMAS entradas**, y el modelo tiene una sola cabeza
sin identificador de tarea. Por construcción, ningún método puede producir B y a la vez
retener A —se contradicen punto a punto. **La retención simultánea está mal planteada
para esta familia de tareas.** El test correcto de la afirmación central es el *ahorro*
al revisitar A (5.2), no la retención simultánea.

---

## 5.2 — Deriva reversible A→B→A: el AHORRO (test bien planteado)

**Qué se hizo (`exp_reversible.py`).** Calendario A→B→A. Medir los pasos para alcanzar
acc≥0.9 en A la **primera** vez (aprendizaje de cero) y la **segunda** (tras pasar por
B). Si la consolidación preservó la estructura de A, re-aprenderla debe costar menos
(*ahorro*). Comparar contra la ablación siempre-plástica.

**Resultado (valida la afirmación central).**

| variante | A 1ª vez | A 2ª vez | ahorro |
|---|---|---|---|
| **full** | 70 pasos | **20 pasos** | **50** |
| no_thermo (siempre plástico) | 60 | 50 | 10 |

El Modelo X completo re-aprende A **~3.5× más rápido** la segunda vez, y ese ahorro es
**5× mayor** que el de la ablación plástica. La ventaja proviene específicamente del
**congelamiento selectivo** (la consolidación deja la estructura de A recuperable), no
de que los pesos simplemente no estén aleatorios —si fuera eso, la ablación plástica
ahorraría lo mismo.

---

## 5.3 — Ablaciones: qué carga cada subsistema

**Qué se hizo (`exp_ablations.py`).** Arco de un Cisne Negro (A→B) desactivando una
pieza cada vez.

| variante | aprende A | detecta CN | recupera B | pasos rec. |
|---|---|---|---|---|
| full | 1.00 | sí | 1.00 | 40 |
| no_thermo | 1.00 | sí | 0.79 | (no llega a 0.9) |
| **no_detect** | 1.00 | **no** | **0.18** | — |
| no_jump | 1.00 | sí | 1.00 | 50 |
| **no_kfac** | 0.97 | sí | **0.22** | — |

**Lectura por subsistema (incluida la parte humilde):**

- **Detección — CRÍTICA.** Sin el disparador de fusión, el modelo consolidado (congelado
  en A) no puede re-plastificarse y queda atascado (B=0.18). Es el componente más
  claramente imprescindible para adaptarse a un Cisne Negro.
- **K-FAC (gradiente natural) — CRÍTICO.** Sin él, Adam plano aprende A pero **no**
  re-adapta a B (0.22). Se verificó que **subir el lr de Adam no lo rescata** (lo
  desestabiliza, ni aprende A): el daño es genuino, no de calibración. La geometría
  importa de verdad para re-navegar tras la consolidación.
- **Salto dJ — NO crítico (hallazgo honesto).** `no_jump` recupera B igual de bien
  (1.00, solo ~10 pasos más lento). Uno de los cuatro subsistemas estrella resulta
  **no ser necesario** para la recuperación en esta tarea; su rol es exploratorio/
  complementario a la fusión térmica, no esencial. Conviene registrarlo así en vez de
  inflar su importancia.
- **Termodinámica — su valor es la CONSOLIDACIÓN, no la recuperación.** `no_thermo`
  recupera una deriva única razonablemente, pero pierde el ahorro de 5.2: el
  congelamiento no se nota en una sola deriva, se nota cuando hay que *preservar* y
  *revisitar*.

---

## Síntesis de la Fase 5

1. **La retención simultánea de tareas contradictorias es imposible por construcción**
   (una cabeza, sin task-ID); el test honesto es el ahorro.
2. **La afirmación central queda validada en su forma bien planteada:** la consolidación
   por congelamiento selectivo hace que revisitar una tarea sea ~3.5× más rápido, y esa
   ventaja es atribuible al mecanismo térmico (5× sobre el plástico).
3. **Las ablaciones ordenan la importancia real:** detección y gradiente natural son
   imprescindibles para sobrevivir un Cisne Negro; la termodinámica es lo que da el
   no-olvido (ahorro); y el salto de Poisson **no es crítico** para la recuperación —el
   resultado más humilde y más útil de la fase, porque dice dónde NO está el valor.

Como en todas las fases anteriores, ninguna conclusión se asumió: 5.1 obligó a
reformular un test mal planteado, 5.2 lo midió correctamente, y 5.3 nombró —sin
adornos— qué pieza carga el peso y cuál no.
