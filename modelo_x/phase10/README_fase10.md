# Fase 10 — ¿Cierra la granularidad sub-bloque la brecha que dejó la Fase 9?

La Fase 9 localizó una limitación concreta de Modelo X: su termostato decide el
congelamiento **por bloque**, y eso le impide la retención simultánea que el anclaje
**por-parámetro** de EWC sí logra (min(ret_A, acc_B): per-param ≈0.65 vs bloque ≈0.25 vs
Modelo X real ≈0.10, en el benchmark multi-cabeza d=256). El cabo: ¿bastaría una
granularidad **sub-bloque** (por fila/canal/matriz) para cerrar la brecha? Se midió, y la
respuesta tiene **dos factores** — uno de los cuales refutó la lectura optimista del
primer smoke (medir, no afirmar).

Correr: `python -m modelo_x.phase10.run_phase10`  (desde el padre de `modelo_x/`). 3 semillas.

---

## 10.1 — El CONTINUO de granularidad (anclaje SUAVE, EWC desacoplado)

Mismo mecanismo (EWC desacoplado, Fisher verdadero), variando **sólo** la granularidad a
la que se engrosa el Fisher: `param` (por elemento) → `row` (por fila/canal de salida) →
`tensor` (por matriz de pesos = cada Linear) → `block` (por bloque del Transformer = mímica
de Modelo X). Mejor min(ret_A, acc_B) sobre el barrido de λ:

| granularidad | ret_A | acc_B | **min** |
|---|---|---|---|
| param  | 0.73 | 0.60 | **0.60** |
| row    | 0.56 | 0.95 | **0.56** |
| tensor | 0.67 | 0.73 | **0.67** |
| **block** | 0.75 | 0.29 | **0.29** |
| Modelo X (bloque real) | 0.10 | 0.99 | 0.10 |

**Hallazgo (corrige el smoke de 1 semilla, que veía un "pico por canal" que NO sobrevivió
a multi-semilla):** la curva **no** es "más fino siempre mejor". `param`, `row` y `tensor`
quedan **todos en ~0.6** (dentro del ruido entre sí); el único que se desploma es `block`
(0.29). **El acantilado es BLOQUE-específico**, entre per-tensor y per-bloque.

La lectura accionable: lo que importa es **distinguir la importancia POR MATRIZ DE PESOS**
(cada Linear q/k/v/out/in/out_proj por separado). *Lumpar* las 6 Linears de un bloque en
una sola decisión térmica es exactamente lo que mata la retención. Y como **Modelo X ya
rastrea K-FAC por-Linear**, el fix **mínimo** es decidir el freeze **por-Linear
(per-tensor)** en vez de por-bloque — un cambio pequeño, no un rediseño grande.
Figura: `phase10_granularity_continuum.png`.

## 10.2 — Hard-freeze por CANAL vs por BLOQUE (el mecanismo REAL de Modelo X)

10.1 usa el anclaje **suave** de EWC. Pero Modelo X no ancla suave: **HARD-congela**
(enmascara el gradiente). Aquí se prueba el mecanismo real —freeze duro— a granularidad de
canal vs de bloque (congelar las filas top-τ por importancia, barriendo τ). Mejor min:

| método (hard-freeze) | ret_A | acc_B | **min** |
|---|---|---|---|
| por **CANAL** (τ=0.8) | 0.40 | 0.98 | **0.40** |
| por **BLOQUE** (τ=0.5) | 0.17 | 0.99 | **0.17** |
| Modelo X real | 0.10 | 1.00 | **0.10** |

**Historia de DOS factores (honesta):**
- **La GRANULARIDAD ayuda mucho:** canal-hard (0.40) ≫ bloque-hard (0.17) ≫ Modelo X (0.10).
- **Pero la granularidad SOLA no basta:** el canal-**hard** (0.40) queda por debajo del
  anclaje **suave** sub-bloque de 10.1 (~0.6). El freeze **binario** suelta del todo los
  canales no-top, que aun así corrompen A; el anclaje **graduado** protege en proporción a
  la curvatura. → La limitación de Modelo X es **DOBLE**: granularidad de **bloque** *y*
  freeze **binario** (las dos peores elecciones para retener). Figura: `phase10_channel_freeze.png`.

---

## Síntesis de la Fase 10 (sin fabricar un ganador)

1. **La brecha de la Fase 9 SÍ es cerrable, y se sabe con qué:** la retención simultánea
   no exige el grano de parámetro; cualquier granularidad **sub-bloque** (per-Linear basta)
   la alcanza con el anclaje adecuado. El acantilado de la Fase 9 era específicamente el
   **agregado por bloque**.
2. **Dos factores, ambos necesarios:** (a) granularidad **sub-bloque** (per-Linear/canal),
   y (b) freeze **graduado** (soft), no binario. Cada uno aporta ~0.2 de min; juntos
   alcanzan ~0.6, separados se quedan cortos (canal-hard 0.40, per-param-soft pero
   bloque 0.29).
3. **Rediseño concreto que esto señala** (no implementado en el núcleo; es trabajo de una
   Fase 11): un termostato que decida **por-Linear** (Modelo X ya tiene la curvatura K-FAC
   por-Linear) y aplique una **compuerta graduada** por canal según curvatura local, en vez
   de congelar/fundir bloques enteros de forma binaria. Eso convertiría el congelamiento de
   Modelo X en un competidor real de EWC en el eje de **retención** — sin perder su ventaja
   en **adaptación** estructural (Fases 5, 8).
4. **Honestidad sobre el alcance:** todo se midió en el juguete multi-cabeza (lag1/lag2,
   d=256, 3 semillas). El número de retención plena alcanzable (~0.6–0.7) tiene un trade-off
   residual; no es 1.0/1.0. Lo medido es lo reportado.

### Cabo para una Fase 11
- **Implementar el termostato per-Linear + compuerta graduada** en `thermodynamics.py` /
  el paso de gradiente natural, y re-correr 9.1/10: ¿alcanza Modelo X el ~0.6 de 10.1 con
  su PROPIA señal de curvatura (sin el Fisher de EWC)? Es la prueba de fuego del rediseño.
