# Fase 7 — Robustez ante derivas múltiples/graduales y cabos sueltos

Cierra las direcciones 3 (robustez ante más de un Cisne Negro, y ante deriva lenta) y
4 (los dos cabos: curvatura cruzada en serie de 4.3, y la plataforma Apple Silicon que
motivó todo el cambio a PyTorch).

Correr todo: `PYTHONPATH=/home/claude python -m modelo_x.phase7.run_phase7`

---

## 7.1 — Derivas múltiples en cadena (¿acumula o interfiere?)

Cadena lag **1 → 2 → 3 → 1**. El riesgo real de estos sistemas: tras varias
consolidaciones, ¿la estructura original sobrevive o queda corrompida?

**Resultado:** el arco sobrevive a las **tres** derivas (detecta, funde, recupera a
1.00 en cada tramo) y, al **volver a lag=1** tras pasar por 2 y 3, sigue habiendo
**ahorro** (40 pasos vs 70 la primera vez). La consolidación se **acumula sin
interferir**: la estructura original queda recuperable pese a los cambios intermedios.

## 7.2 — Deriva gradual (rampa) vs el detector de saltos

El detector dispara por un SALTO de la pérdida (z-score); una rampa lenta podría no
dispararlo y dejar al modelo consolidado atascado.

**Resultado:** una rampa de 120 pasos **sí** dispara el detector (una vez) y el modelo
se adapta a 1.00. Incluso una transición gradual eleva la pérdida lo bastante para
cruzar el umbral en algún punto. El detector resultó más robusto a deriva lenta de lo
temido (con la advertencia de que una rampa extremadamente lenta seguiría siendo un
punto ciego teórico, mitigable con una señal de deriva lenta adicional).

## 7.3 — Curvatura cruzada en serie: ¿corrección barata o límite?

La Fase 4.3 dejó abierto si el acoplamiento en serie (alto pero no Kronecker-
factorizable) admite una corrección barata. Midiendo en un par en serie pequeño
(out → in_proj) el gradiente natural REAL (Fisher conjunto completo) vs el bloque-
diagonal:

- **Desalineación: 41°** (coseno 0.75) — el bloque-diagonal se desvía de forma medible.
- **Coste de corregirlo:** el bloque conjunto completo es O(d⁶) por par (no factoriza
  como Kronecker al no compartir entrada), frente a O(d³) factorizado.

**Veredicto:** **no hay corrección barata**; a escala es prohibitiva. Se **formaliza
como límite estructural** de K-FAC por bloques: el precio de la factorización Kronecker
es ignorar exactamente este acoplamiento en serie. Mitigación realista: ordenamientos/
normalización que reduzcan la correlación serie, o métodos no-Kronecker (Shampoo) si se
dispusiera del presupuesto —fuera del alcance del prototipo.

## 7.4 — Portabilidad / preparación para Apple Silicon (MPS)

El sandbox es CPU, así que no se valida MPS aquí, pero se confirman por construcción las
invariantes MPS-ready: (1) `get_device()` prioriza MPS con respaldo a CPU; (2) **todo el
cómputo caliente es float32** (cero float64 en model/attention/liquid_sde/thermodynamics/
drift_data) → ejecutable en Metal; (3) la única operación float64 (inversión de factores
K-FAC) se aísla en CPU y el resultado vuelve al dispositivo; (4) el lazo completo corre
end-to-end en el dispositivo disponible. **Validación real en M4 queda como paso del
usuario** (correr `run_demo` en el M4 y confirmar `get_device()=='mps'` y los tiempos).

---

## Síntesis de la Fase 7

- **Robustez (dir. 3): positiva.** La consolidación acumula sin interferir a lo largo de
  una cadena de derivas, y el detector aguanta deriva gradual razonable.
- **Cabos (dir. 4): cerrados con honestidad.** El acoplamiento en serie se formaliza como
  límite estructural sin corrección asequible (41° de desvío, coste O(d⁶)); y el código se
  verifica MPS-ready por construcción, dejando la validación de rendimiento real como paso
  del usuario en su M4.
