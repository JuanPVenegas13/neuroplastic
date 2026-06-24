# Fase 6 — Validez externa y navaja de Occam

Dos preguntas que las fases internas no podían responder: ¿alguna pieza sobra
(navaja de Occam sobre el salto dJ)? y ¿el framework aporta algo frente a un método
reconocido (EWC) fuera del juguete? Como siempre, se midió de verdad —incluido
cuando el resultado fue incómodo o inconcluso.

Correr todo: `PYTHONPATH=/home/claude python -m modelo_x.phase6.run_phase6`

---

## 6.1 — ¿El salto dJ gana su lugar? (esfuerzo honesto antes de cortar)

La Fase 5.3 sugirió que el salto de Poisson (dJ) no es crítico. Antes de eliminarlo se
le dio su **mejor oportunidad**: una deriva dura y potencialmente multimodal (lag 1→4,
con lags intermedios como atractores engañosos) y un régimen de **difusión baja**
(donde el salto sería la única exploración), sobre 4 semillas.

**Resultado:** el salto **no gana de forma robusta** en ningún régimen
(full≈1.00 vs no_jump≈0.96–0.99; victorias 1/8, dentro del ruido). No se gana su lugar.

## 6.2 — Navaja de Occam: el framework SIN dJ conserva todo

Se verificó que eliminar el salto (`jump_sigma=0`, sin cabeza aprendible, sin
STE/REINFORCE) no degrada lo de cabecera:

| | con dJ (ref.) | sin dJ |
|---|---|---|
| arco (recupera B) | 1.00 | **1.00** |
| ahorro (5.2) | 50 | **40** |

**Conclusión:** la detección de deriva + la fusión térmica + el gradiente natural hacen
el trabajo que el dJ pretendía. Se **recomienda operar con `jump_sigma=0`**, eliminando
todo el aparato de salto (compuerta STE, REINFORCE, cabeza aprendible, subdivisión
adaptativa) — una reducción grande de complejidad sin pérdida medible. (El código del
salto se conserva como registro de las Fases 3–4; el modo recomendado es sin dJ.)
Nota: esto revisa uno de los cuatro pilares teóricos originales (el proceso de saltos
para "Cisnes Negros"): empíricamente, el manejo del Cisne Negro vive en el **detector**
y el **termostato**, no en el dJ.

---

## 6.3 — Modelo X vs EWC / naive (continual con token de tarea)

Se construyó un benchmark task-incremental **bien planteado**: el token de la posición 0
codifica la tarea, así que A y B son compatibles (existe solución conjunta). Esto corrige
el problema de 5.1 (tareas contradictorias sin task-id).

**Lo que se estableció:**
- **Benchmark válido:** entrenamiento intercalado alcanza acc_A=acc_B=1.00 (la solución
  conjunta existe y es aprendible usando el token de tarea).
- **Naive secuencial (Adam):** olvido catastrófico de A (retención ≈ azar).
- **EWC está CONFUNDIDO por el optimizador en esta arquitectura:**
  - bajo **Adam** (único optimizador con el que esta arquitectura entrena), la
    penalización cuadrática de EWC queda **neutralizada** por la renormalización de Adam
    → EWC ≈ naive en todo el barrido de λ (hasta 10⁶).
  - bajo **SGD** (donde EWC sería efectivo), la arquitectura atención+SDE **no entrena**
    (acc ≈ azar): fue diseñada para Adam/gradiente natural.

**Conclusión honesta (sin fabricar un ganador):** el head-to-head de *retención
simultánea* no es concluyente en esta arquitectura por el confound del optimizador. Pero
el eje donde Modelo X **sí** está validado no es la retención multi-tarea simultánea
(que nunca reclamó) sino el **AHORRO al revisitar una tarea** (Fase 5.2: re-aprende
~3.5× más rápido, 5× más que el plástico). Un head-to-head limpio contra EWC exige un
**siguiente paso concreto**: portar el framework a una arquitectura entrenable con SGD,
o usar un regularizador CL compatible con Adam (online-EWC / Synaptic Intelligence con
escalado apropiado).

---

## Síntesis de la Fase 6

1. **Navaja aplicada con evidencia:** tras un intento genuino de salvarlo, el salto dJ
   no aporta y se recomienda eliminarlo —simplificando sustancialmente el framework y
   borrando justo las partes que más coste tuvieron (la varianza de REINFORCE, el sesgo
   del STE).
2. **Validez externa, con honestidad metodológica:** se montó un benchmark CL válido y
   se documentó por qué una comparación limpia con EWC está bloqueada aquí (optimizador),
   en vez de reportar un número engañoso. El valor real del framework se reubica en el
   eje del *ahorro*, ya validado, y se nombra el experimento que faltaría para el
   head-to-head limpio.

El patrón se mantiene: medir, y cuando el resultado es inconcluso o incómodo, decirlo
tal cual y señalar el siguiente paso, en lugar de forzar una conclusión favorable.
