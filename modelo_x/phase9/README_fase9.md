# Fase 9 — Bloque (Modelo X) vs por-parámetro (EWC) donde la retención SÍ es alcanzable

La Fase 8 cerró que en una cabeza ÚNICA **nadie** retiene de forma simultánea (límite
estructural: A y B se contradicen punto a punto). La pregunta que quedó: en un terreno
donde la retención simultánea **sí** sea alcanzable, ¿el congelamiento por **bloque** de
Modelo X compite con el soft-freeze por-**parámetro** de EWC? La predicción falsable que
se puso sobre la mesa: *EWC desacoplado a fuerza alta debería converger hacia el
hard-freeze por bloque de Modelo X.*

**Se midió, y la predicción quedó REFUTADA** (per-param y bloque divergen, no convergen).
Como en todas las fases, varias hipótesis de partida cayeron; el valor está en eso.

Correr: `python -m modelo_x.phase9.run_phase9`  (desde el padre de `modelo_x/`)

Arquitectura (`phase9/multihead.py`): tronco COMPARTIDO + **una cabeza por tarea**
(task-incremental). La cabeza inactiva no recibe gradiente en el forward → se preserva
sola; el **tronco compartido** es el recurso disputado donde compiten bloque vs
por-parámetro. Tarea ESTRUCTURAL A=lag1 / B=lag2. Benchmark válido: intercalado alcanza
accA=accB=1.00 (la solución conjunta existe). 3 semillas.

---

## 9.1 — Multi-cabeza es NECESARIO pero NO SUFICIENTE (refuta la premisa simple)

Con tronco **justo** (d_model=64), pese a las cabezas separadas, **nadie retiene**:

| d=64 | ret_A | acc_B | min |
|---|---|---|---|
| naive | 0.09 | 1.00 | — |
| Modelo X (no_jump) | 0.10 | 0.99 | — |
| EWC por-parámetro (mejor) | 0.24 | 0.59 | **0.24** |

El cuello de botella **no es la cabeza, es el TRONCO compartido**: la solución conjunta
exige que el tronco haga lag1 **y** lag2 (estructura nueva que θ\*_A no tiene), y anclar
pesos (EWC) resiste *todo* movimiento, incluido el productivo. Es la limitación conocida
de la CL por regularización: no aprende tareas que exigen alejarse de la solución vieja.
Mi hipótesis de partida ("cabezas separadas ⇒ retención alcanzable") era **falsa**.

## 9.2 — La retención simultánea exige CAPACIDAD SOBRANTE + ANCLAJE FUERTE

Con tronco **ancho** (d_model=256) hay parámetros de bajo Fisher (holgura) que B puede
usar sin pisar los de A. Sólo entonces, y con λ alto, EWC por-parámetro lo logra:

| d=256, EWC por-parámetro | ret_A | acc_B | min |
|---|---|---|---|
| λ=1e9 | 0.33 | 0.99 | 0.33 |
| λ=3e9 | 0.57 | 0.81 | 0.57 |
| **λ=1e10** | **0.65** | **0.78** | **0.65** |
| λ=3e10 | 1.00 | 0.15 | 0.15 (sobre-congela) |

`min=0.65`: retención simultánea **real** (ambas muy por encima del azar 0.10), con un
trade-off suave. Requiere las DOS cosas: capacidad (d=64 no lo logra con ningún λ) **y**
anclaje fuerte (λ~1e10).

## 9.3 — GRANULARIDAD: la predicción falsable, refutada

¿Es la granularidad **por-parámetro** lo que importa, o bastaría el congelamiento por
**bloque**? Test directo: se **engrosa** el Fisher de EWC a nivel de bloque (cada
parámetro recibe la media de Fisher de su bloque) — una mímica del congelamiento por
bloque de Modelo X — y se compara a igual barrido de λ:

| d=256, mejor punto | ret_A | acc_B | **min** |
|---|---|---|---|
| EWC **por-parámetro** | 0.65 | 0.78 | **0.65** |
| EWC **engrosado a bloque** | 0.25 | 0.70 | **0.25** |
| Modelo X (bloque real, no_jump) | 0.10 | 0.99 | **0.10** |

**Engrosar a bloque DESTRUYE la retención** (min 0.65 → 0.25), cayendo al nivel del
fracaso de Modelo X. La retención simultánea exige sostener los parámetros críticos de A
**mientras se liberan los sobrantes DENTRO del mismo bloque** (holgura sub-bloque). El
congelamiento todo-o-nada por bloque es estructuralmente incapaz de eso. **Per-param y
bloque DIVERGEN, no convergen** — lo contrario de la predicción. Figura:
`phase9_granularity.png`.

## 9.4 — Modelo X real lo confirma

El Modelo X completo (no_jump, no_thermo y no_detect probados) **no retiene** A
(ret_A≈0.10) ni siquiera con capacidad (d=256). Dos causas, ambas honestas: (a) su
mecanismo de adaptación es **fundir** ante deriva (antitético a retener), y (b) su
congelamiento es **por bloque** y además **no cubre** embeddings/LayerNorm/cabezas (EWC
ancla TODO parámetro). Coincide con 9.3: el termostato por bloque es demasiado crudo.

---

## Síntesis de la Fase 9 (sin fabricar un ganador)

1. **Premisa de partida refutada:** la capacidad separable por cabeza NO basta; el
   tronco compartido es el cuello de botella. La retención simultánea sólo aparece con
   **capacidad sobrante + anclaje por-parámetro fuerte**.
2. **Predicción falsable refutada:** EWC por-parámetro y el congelamiento por bloque
   **divergen**. Engrosar EWC a granularidad de bloque cae al nivel de Modelo X
   (min 0.25 vs 0.65). La granularidad **sub-bloque** es esencial.
3. **Limitación CONCRETA de Modelo X localizada:** el termostato decide **por bloque**, y
   eso le impide explotar la holgura sub-bloque que la retención simultánea exige. EWC
   por-parámetro es la mejor herramienta de retención **incluso en el régimen construido
   para favorecer el congelamiento**. No se fabrica una victoria de Modelo X.
4. **Consistencia con todo el arco:** el valor de Modelo X sigue siendo la **adaptación**
   a deriva estructural por gradiente natural + ahorro al revisitar (Fases 5, 8), **no**
   la retención simultánea, que es terreno de EWC y exige granularidad por-parámetro.

### Cabos para una Fase 10 (opcional)
- Dar al termostato granularidad **sub-bloque** (congelar por-fila/canal según curvatura
  local) y re-correr 9.3: ¿cierra la brecha con EWC por-parámetro? Sería la mejora de
  diseño que este resultado señala directamente.
- Métodos de **parameter-isolation** (máscaras por tarea) o **replay**, que sí alcanzan
  la solución conjunta de 9.2 sin el trade-off, como techo de referencia.
