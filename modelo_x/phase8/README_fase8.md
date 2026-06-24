# Fase 8 — Head-to-head LIMPIO contra EWC (resuelto el confound del optimizador de 6.3)

La Fase 6.3 dejó el head-to-head contra EWC "confundido por el optimizador": bajo Adam
la penalización de EWC parecía neutralizada, bajo SGD la arquitectura no entrena, y se
nombró como paso pendiente portar a SGD o usar un regularizador CL compatible con Adam.
Esta fase ataca ese paso **midiendo**, y el resultado obliga a **corregir el propio
diagnóstico de 6.3** — exactamente el estilo de la casa (medir, no afirmar; documentar
lo que estaba mal).

Correr todo: `python -m modelo_x.phase8.run_phase8`  (desde el padre de `modelo_x/`)

---

## 8.0 — Bug real aflorado al correr por primera vez en M4/MPS

El sandbox previo era CPU, así que el camino caliente nunca se ejecutó en Metal. En el
M4, `kfac.py` crasheaba: `M.detach().to("cpu", torch.float64)` sobre un tensor **MPS**
intenta el cast a float64 **antes** de salir de MPS, y Metal no soporta float64. Fix:
salir a CPU **antes** de castear (`M.detach().cpu().to(torch.float64)`). Con esto el
lazo completo (K-FAC incluido) corre end-to-end en MPS.

**Verificación del arco en MPS (no solo CPU).** `run_demo` en M4 con `get_device()=='mps'`
sostiene el arco completo en float32: aprende A (acc→1.00), consolida (b0:F b1:F), el
Cisne Negro del paso 300 dispara el detector (pérdida 0.002→**11.67**, acc→0.10), funde
(b0:M b1:M), **recupera a acc 1.00** (pérdida final 0.002 = pre-drift) y re-congela
selectivo (b0:M b1:F). El cambio de dispositivo NO movió el resultado por orden de
operaciones. Esto **cierra el paso #2** en lo cualitativo (falta solo el timing ms/paso
CPU-vs-MPS sistemático).

## 8.1 — Eje RETENCIÓN simultánea (drift de *shift*, absorbible) — corrige a 6.3

Benchmark task-id de 6.3 (token de tarea ⇒ existe solución conjunta). `exp_retention.py`.

**Corrección honesta de 6.3.** 6.3 concluyó *"Adam neutraliza a EWC"* tras barrer
λ ∈ {1e4, 1e6}. Pero el **Fisher empírico de esta arquitectura es DIMINUTO** (~1e-7), así
que una penalización EWC con efecto necesita **λ ~ 1e8**: 6.3 se quedó ~100× corto.
Empujando λ a la escala del Fisher, **incluso el EWC ACOPLADO (penalización vía Adam)
MUERDE**:

| λ (acoplado, Fisher empírico) | ret_A | acc_B | |
|---|---|---|---|
| 1e4 | 0.06 | 1.00 | ← rango de 6.3 → "neutralizado" |
| 1e6 | 0.09 | 0.96 | ← rango de 6.3 |
| 1e7 | 0.42 | 0.56 | |
| 3e7 | 0.83 | 0.18 | ← **muerde** |
| 1e8 | 0.98 | 0.06 | ← muerde |

**Adam NO neutraliza a EWC.** El "confound del optimizador" de 6.3 era un **artefacto del
rango de λ** frente a la magnitud del Fisher, no una propiedad de Adam. (3 semillas.)

**EWC desacoplado (AdamW-style).** Aun así se implementó el fix principiado: desacoplar
el tirón de EWC del preacondicionador de Adam y aplicarlo post-paso,
`θ ← θ − clamp(lr·λ·F, 0, 1)·(θ − θ*)`. Con el coef recortado a [0,1] es un **soft-freeze
por-parámetro hacia θ\*, con fuerza = curvatura de Fisher** — el análogo justo del
hard-freeze por curvatura de Modelo X. Funciona y es más estable/controlable, pero llega
a la **misma frontera** que el acoplado (no era estrictamente necesario para des-confundir;
es la versión limpia y el puente conceptual).

**Frontera retención_A vs acc_B (criterio rector: comparar a IGUAL acc_B).**

| | acc_B≈1.0 | mejor balance | ret_A máx |
|---|---|---|---|
| EWC desacoplado | ret_A 0.04 | ret_A 0.74 / acc_B 0.25 | 1.00 (acc_B 0.03) |
| EWC acoplado | ret_A 0.06 | ret_A 0.42 / acc_B 0.56 | 1.00 |
| **Modelo X (no_jump)** | **ret_A 0.00 / acc_B 1.00** | — (sin palanca) | — |
| naive (Adam) | ret_A 0.05 / acc_B 1.00 | — | — |

**Es un trade-off DURO: ningún λ logra ambos altos** (max min(ret_A, acc_B) ≈ 0.25–0.42 < 0.6).
Razón mecánica: la solución conjunta exige **condicionar en el token de tarea**; θ\*_A lo
ignora (siempre produce shift-3), y EWC ancla **pesos, no función** — no puede re-derivar
"shift-3 cuando token=0" viendo solo el gradiente de B. **Modelo X queda clavado en el
punto de olvido de naive** (ret_A 0.00): su melt→adapt sobrescribe la única cabeza. En
ESTE eje **Modelo X no tiene reclamo** (consistente con 5.1); EWC al menos tiene la
palanca de la frontera. A iso-acc_B=1.0, nadie retiene. Figura: `phase8_retention_frontier.png`.

## 8.2 — Eje AHORRO al revisitar (drift de *lag*, estructural) — terreno propio de Modelo X

Deriva reversible A→B→A con cambio de **lag** (a qué posición atender), la tarea de 5.2.
`exp_savings.py`. Dos controles de honestidad, **descubiertos midiendo**:

1. **Gate iso-acc_B sobre el ahorro:** el ahorro sólo cuenta si el método **realmente
   aprendió B** (accB_max ≥ 0.9 en la ventana B). EWC con λ alto "ahorra" porque nunca
   soltó A → ahorro FALSO.
2. **"Pesos calientes":** re-aprender una tarea ya vista es barato hasta para naive;
   reportamos A1 y A2 por separado para que el confound quede a la vista.

| método | A1 | A2 | ahorro | accB_max | ¿válido? |
|---|---|---|---|---|---|
| **ModeloX full** | 52 | 18 | **33** | **1.00** | SÍ |
| ModeloX no_thermo | 50 | 38 | 12 | 0.99 | SÍ |
| naive (Adam) | 138 | 43 | 95 | **0.77** | NO (no aprende B) |
| EWC desac. λ=5e8 | 138 | 0 | 138 | **0.11** | NO (ahorro falso) |

**El drift de lag es ESTRUCTURAL: Adam plano (naive y EWC, que viven sobre Adam) NO lo
navega** — no llega a aprender B (5.3: "Adam plano aprende A pero NO re-adapta a B"). El
gradiente natural K-FAC de Modelo X **sí**. Es una diferencia **arquitectónica, no un
confound**: en su terreno propio, Modelo X compite contra un EWC que **ni siquiera puede
entrar**. (3 semillas.)

**Qué responde esta tabla y qué NO (el confound del ahorro queda ACOTADO, no resuelto).**
- **SÍ responde:** el ahorro de Modelo X sobre no_thermo es **real** (33 vs 12), y con
  A1 casi idéntico (52 vs 50) → la diferencia es atribuible al **congelamiento selectivo**,
  no a una velocidad de arranque distinta. Esto **reconcilia con 5.2**.
- **NO responde (y es importante decirlo):** cuánto del ahorro **absoluto** es solo
  **"pesos calientes"** (re-aprender una tarea ya vista es barato incluso sin protección;
  lo destapamos midiendo: naive "ahorra" 95). El control natural para aislar eso —un
  aprendiz **sin gradiente natural** (naive-Adam)— **se rompe en esta tarea**: el drift
  de lag es estructural y Adam plano ni completa el arco (accB 0.77 < 0.9), y un control
  que no aprende B no puede medir ahorro. Por tanto el confound de pesos calientes queda
  **acotado pero NO aislable sobre el lag-shift**, por esta razón concreta. Aislarlo
  exigiría una tarea con drift **absorbible** donde un control sin gradiente natural sí
  aprenda B — pero ahí el congelamiento estructural de Modelo X deja de ser lo relevante.
  No se resuelve; se nombra el porqué.

---

## Síntesis de la Fase 8 (sin fabricar un ganador)

1. **6.3 estaba mal diagnosticada.** No era el optimizador: era la escala de λ frente a
   un Fisher de ~1e-7. EWC **sí protege A bajo Adam** (acoplado o desacoplado) una vez λ
   se escala. Se documenta la corrección en vez de heredar la conclusión.
2. **EWC y Modelo X resuelven problemas casi DISJUNTOS:**
   - **EWC** = preservación de pesos para **retención en drift absorbible** (tiene
     frontera ahí; Modelo X no compite, queda en el punto de olvido).
   - **Modelo X** = navegación por **gradiente natural de drift estructural** + ahorro al
     revisitar (donde EWC, atado a Adam, no llega a aprender la tarea nueva).
   - El solapamiento es mínimo. **No hay ganador global**; hay dos herramientas para dos
     regímenes. Esto sitúa el valor de Modelo X **con precisión**: su aporte distintivo
     es la deriva ESTRUCTURAL, **no** la retención multi-tarea simultánea (que nunca
     reclamó, 5.1).
3. **Honestidad sobre los límites:** (a) en ningún benchmark de una cabeza hubo retención
   simultánea para nadie; (b) el ahorro de Modelo X sobre no_thermo es real (33 vs 12, A1
   igualado), pero el confound de **"pesos calientes" queda ACOTADO, no aislado**, sobre
   el lag-shift — el único control sin gradiente natural (naive-Adam) no aprende el drift
   estructural y por eso no puede medir el ahorro de fondo. Se nombra el porqué en vez de
   declararlo resuelto. Lo medido es lo reportado.

### Cabos para una Fase 9 (opcional)
- Benchmark con **cabezas/módulos por tarea** (no una sola cabeza) donde la retención
  simultánea SÍ sea alcanzable: ahí EWC y un Modelo X con congelamiento podrían por fin
  compararse en el MISMO eje sin el límite estructural de la cabeza única.
- Un método CL que ancle **función** (p.ej. destilación/replay), no solo pesos, para ver
  si cierra la frontera dura del eje 1.
