# Modelo X — Fase 3: Rigor de los estimadores y validación de las aproximaciones

La Fase 2 validó el lazo de retroalimentación completo. La Fase 3 ataca las cuatro
deudas teóricas que quedaron abiertas, **midiendo** en vez de afirmar. Cada
experimento es ejecutable de forma independiente o todo junto:

```bash
python -m modelo_x.phase3.run_phase3        # corre los 4 frentes
# o individualmente:
python -m modelo_x.phase3.exp_jump_bias
python -m modelo_x.phase3.exp_jump_learning
python -m modelo_x.phase3.exp_coupling
python -m modelo_x.phase3.exp_adaptive_dt
python -m modelo_x.phase3.exp_scale
```

---

## 3.1 — El gradiente del salto: STE vs *adjoint* riguroso (REINFORCE)

**Montaje:** un salto Bernoulli aislado donde el gradiente exacto `dE[L]/dθ` es
cerrado (verdad analítica). Se compara el STE (Fase 2) contra el estimador
score-function (REINFORCE), con y sin línea base.

**Hallazgo (sesgo/varianza, `exp_jump_bias`):**
- El STE es **sesgado**, y su sesgo es **máximo cuando los saltos son raros**
  (p≈0.1: sesgo ≈ −0.023), que es precisamente el régimen del Cisne Negro. El
  estimador barato es el menos fiable justo donde el término de salto importa más.
- REINFORCE es **insesgado** (sesgo ~5e-4) a costa de ~2–3× más varianza; la línea
  base recupera buena parte de esa varianza.

**Consecuencia operativa (`exp_jump_learning`):** al optimizar una política de salto
con óptimo interior conocido `p*=0.278`, el STE converge a un punto fijo
**sistemáticamente equivocado** (p≈0.302 ± 0.005), mientras REINFORCE aterriza en
el óptimo (p≈0.280 ± 0.009). El sesgo no es académico: produce una política errónea.

**Recomendación:** usar REINFORCE+baseline para la intensidad de salto cuando la
fidelidad en eventos raros sea crítica; el STE es admisible para velocidad si se
acepta su sesgo conocido (sobre-estima la frecuencia de salto).

## 3.2 — Curvatura fuera de la diagonal W_Q–W_K

Como Q y K comparten la entrada `x`, su gradiente es `g=a⊗s` y todo el acoplamiento
se concentra en el factor cruzado de salida `S_QK = E[s_Q s_Kᵀ]` (`coupling.py`).

**Hallazgo:** el acoplamiento **no es despreciable** (ratio ‖S_QK‖/√(‖S_QQ‖‖S_KK‖)
≈ 0.41). Ignorarlo (K-FAC diagonal por bloques) **desalinea** la dirección del
gradiente natural ~7° (coseno ≈ 0.93) y la cambia ~37% en norma.

**Bonus metodológico:** con el damping absoluto de un primer intento, el efecto
salía exactamente cero — porque los factores `S` son diminutos (‖S‖~1e-8) y el
damping fijo los aplastaba. Hizo falta el **amortiguamiento adaptativo** (la lección
de la Fase 2) para que el acoplamiento siquiera fuese visible. Confirma que esa
corrección es transversal, no un parche local.

## 3.3 — Paso Δt adaptativo (multi-salto)

El thinning de un solo paso satura en **≤1 salto** cuando `λT>1`: en λT=5 cuenta 1
de 5 (pierde el 80% de un Cisne Negro). El submuestreo **adaptativo** (parte Δt en N
subpasos con `λ·δt ≤ 0.1`) recupera la media de Poisson `λT` (4.98 en λT=5).
Conectado al núcleo vía `ModelConfig.max_lambda_dt`. Viable aquí porque en PyTorch el
N variable **no recompila** el grafo —el problema exacto que evitamos al dejar JAX—.

## 3.4 — Submuestreo de tokens en K-FAC (validación de Gemini #2)

Sobre un modelo de **4 capas, T=64**: estimar los factores con el **10–20%** de los
tokens conserva el gradiente natural (coseno ≈ 0.99 contra el 100%) y la traza
termodinámica (dentro del ~1.5%). El ahorro de ~1 orden de magnitud en cómputo de
K-FAC es esencialmente gratis, como predijo Gemini.

---

## Síntesis para la Fase 4

1. **Integrar REINFORCE+baseline** como modo de gradiente del salto en el núcleo
   (ya hay un cabezal de intensidad aprendible esbozado); medir su efecto en el lazo
   completo, no solo aislado.
2. **Preacondicionador acoplado Q–K** (factor `S_QK`) en las capas donde el ratio de
   acoplamiento supere un umbral; el resto, diagonal.
3. **Curvatura cruzada entre sublocaciones** más allá de Q–K (p.ej. atención↔SDE).
4. **Escalado real** (decenas de capas, secuencias de miles) explotando el
   submuestreo validado y la inversión amortizada en CPU/float64.
