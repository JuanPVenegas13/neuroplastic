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
2. **Dos factores, ambos necesarios y ~aditivos** (factorial completo en **10.3**):
   dureza +0.17, grano +0.20/+0.25; sólo-granularidad (hard sub-bloque) = 0.33–0.53;
   sólo-softness (soft bloque) = 0.38; ninguno (≈ Modelo X) = 0.10–0.17; ambos ≈ 0.51–0.68.
   *Nota de ruido:* entre corridas del mismo protocolo hay ±0.1 (MPS + muestreo del
   Fisher, 3 semillas); los efectos principales y el acantilado de block son robustos,
   el orden fino DENTRO de los granos sub-bloque no.
3. **Rediseño concreto que esto señala** (no implementado en el núcleo; es trabajo de una
   Fase 11): un termostato que decida **por-Linear** (Modelo X ya tiene la curvatura K-FAC
   por-Linear) y aplique una **compuerta graduada** por canal según curvatura local, en vez
   de congelar/fundir bloques enteros de forma binaria. Eso convertiría el congelamiento de
   Modelo X en un competidor real de EWC en el eje de **retención** — sin perder su ventaja
   en **adaptación** estructural (Fases 5, 8).
4. **Honestidad sobre el alcance:** todo se midió en el juguete multi-cabeza (lag1/lag2,
   d=256, 3 semillas). El número de retención plena alcanzable (~0.6–0.7) tiene un trade-off
   residual; no es 1.0/1.0. Lo medido es lo reportado.

## 10.3 — Factorial GRANULARIDAD × DUREZA (el cuadro de decisión)

10.1/10.2 barrieron cada eje con el otro fijo, y las celdas hard@param / hard@tensor no
existían. `exp_factorial.py` mide el 2×4 completo con presupuestos comparables (en los
granos finos, τ = fracción de PARÁMETROS congelados; λ para el soft), 3 semillas.
Matriz de min(ret_A, acc_B), mejor sobre el barrido (λ/τ):

| | param | row/canal | tensor (per-Linear) | block |
|---|---|---|---|---|
| **hard** (máscara binaria) | 0.53 | 0.39 | 0.33 | 0.17 |
| **soft** (anclaje graduado) | **0.68** | 0.51 | 0.57 | 0.38 |

(Modelo X real = 0.10.) **Efectos principales, ambos reales y ~aditivos:** dureza
(soft−hard) **+0.17**; grano (sub-bloque−block) **+0.20/+0.25**. Los 4 checks pasan:
block es el peor grano en ambos mecanismos, soft ≥ hard en todo grano sub-bloque, la
mejor celda es soft×sub-bloque, y Modelo X real queda por debajo de TODA celda sub-bloque.

**Matiz honesto que el factorial añade sobre 10.1:** dentro de los granos sub-bloque el
orden fluctúa entre corridas (±0.1 de ruido run-a-run); lo ROBUSTO es el acantilado de
block y los dos efectos principales. Y "per-tensor basta" vale para el mecanismo
graduado (soft×tensor=0.57) pero NO para el binario (hard×tensor=0.33): la receta mínima
verificada es **grano sub-bloque Y graduado, juntos**. El rediseño mínimo de la Fase 11
(decisión per-Linear + compuerta graduada) apunta a la celda soft×tensor ≈ **0.57**;
con grano por fila (diag del factor S de K-FAC, casi gratis) el techo es ≈0.5–0.7.
Figura: `phase10_factorial.png`; celdas completas en `phase10_factorial.json`.

## 10.4 — Dinámica del olvido + verificación de la señal per-Linear

Los experimentos de estado-final no dicen *cuándo* se pierde A ni si la señal que la
Fase 11 necesita existe. `exp_dynamics.py` instrumenta el run de Modelo X (multi-cabeza,
d=256): accs de AMBAS tareas cada 10 pasos, estados térmicos por paso, y las trazas
K-FAC **per-Linear** (que el núcleo ya calcula pero solo consulta agregadas por bloque).

**Hallazgos (`phase10_dynamics.png/json`):**
1. **El colapso de A es INMEDIATO en el melt, no gradual:** acc_A pasa de 1.00 a <0.5 en
   ~10 pasos tras el switch (melt dispara en el paso 251 de 250). El olvido de Modelo X
   no es erosión durante B: es el **melt GLOBAL** (melt_hold fuerza MELTED en todos los
   bloques) el que destruye A casi al instante. → Un freeze per-Linear/graduado durante
   B **no bastaría por sí solo**: la Fase 11 también necesita **melt selectivo** (fundir
   solo lo que la deriva exige, no todo).
2. **La consolidación funciona pero no vuelve:** ambos bloques FROZEN durante A (el
   termostato consolida bien); tras el switch, **b0 queda MELTED hasta el final** (b1
   re-congela en ~310). La re-consolidación es heterogénea e incompleta.
3. **La señal per-Linear EXISTE:** el spread de Tr(F⁻¹) entre las 6 Linears de un bloque
   es ~3–4× mediano (pico 5×) y estable en el tiempo. La premisa de la Fase 11 (decidir
   per-Linear con la curvatura que K-FAC ya calcula, coste extra cero) es viable.

### Cabo para una Fase 11 (refinado por 10.3/10.4)
- **Implementar el termostato per-Linear + compuerta graduada** en `thermodynamics.py` /
  el paso de gradiente natural, usando la traza per-Linear (verificada en 10.4), **y
  además melt SELECTIVO** (10.4 muestra que el melt global destruye A en ~10 pasos; sin
  eso, el freeze fino no puede salvar nada). Prueba de fuego: ¿alcanza Modelo X el ~0.6
  del factorial con su PROPIA señal de curvatura (sin el Fisher de EWC)?
