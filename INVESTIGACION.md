# Investigación: segmentación de GTVp y GTVn en PET/CT (HECKTOR 2025) con MambaLite3D y fusión bimodal

Estado al 15 de octubre de 2026, 15:00 (hora de Lima). Reúne lo que se sabe hasta ahora, de dónde sale cada cifra y qué falta. Las cifras se tomaron de los archivos de `runs/`; la sección 12 dice cuáles.

---

## 1. Objetivo

- **Meta de la tesis** (`modelos/mamba/TAREAS.md`): DSC agregado medio (métrica HECKTOR 2022, GTVp + GTVn) ≥ 0,75 **sin post-procesado**, con muchos menos recursos que el estado del arte (ganador de 2022: 0,788), en una laptop con RTX 3050 de 4 GB.
- **Pregunta actual** (`modelos/bifusion`): ¿una fusión PET/CT con codificador siamés mejora a MambaLite3D, sobre todo en ganglios (GTVn), sin aumentar mucho el costo? ¿Y de dónde viene la mejora: del siamés, de Mamba como fusión o del cómputo extra?

## 2. Datos

HECKTOR 2025, Tarea 1: 680 estudios PET/CT de 7 centros. Partición estratificada por centro 70/20/10, semilla 12345, la misma para todos los modelos.

| Centro | Train | Val | Test | Total |
|---|---|---|---|---|
| MDA | 277 | 79 | 40 | 396 (58 %) |
| CHUP | 51 | 14 | 7 | 72 |
| CHUS | 51 | 14 | 7 | 72 |
| CHUM | 39 | 11 | 6 | 56 |
| HGJ | 38 | 11 | 6 | 55 |
| HMR | 12 | 4 | 2 | 18 |
| USZ | 8 | 2 | 1 | 11 |
| **Total** | **476** | **135** | **69** | **680** |

- MDA domina: 58 % del total y 79 de los 135 casos de validación. Las métricas globales reflejan sobre todo a MDA.
- El test interno (69 casos de los mismos centros) no es el test oficial del challenge (centros no vistos).
- **10 pacientes de validación no tienen GTVp** en la referencia; todos son de MDA.

## 3. Pipeline (común a todos los modelos)

1. Lectura de los NIfTI y reorientación a LPS.
2. ROI de cabeza y cuello sin usar la etiqueta: cuerpo en una malla de 4 mm (CT > −500 HU o SUV > 0,5, componente principal), ancla en el corte más craneal con SUV > 2 (cerebro) y 270 mm hacia abajo, con margen.
3. Un solo remuestreo a 1 × 1 × 3 mm (anisotrópico): CT y PET trilineal, etiqueta por vecino más cercano.
4. CT en ventana [−200, 400] HU → [0, 1]; PET con z-score sobre toda la caja de la ROI.
5. Caché rodante en RAM (60 casos, 10 nuevos por época), parches de 32 × 128 × 128, la mitad centrados en tumor, aumento de datos en GPU.
6. SGD Nesterov (0,99), lr 0,01 con decaimiento polinomial, 250 × 250 iteraciones, lote 2, AMP, Dice + CE con supervisión profunda.
7. Inferencia por ventana deslizante (50 % de solapamiento, peso gaussiano) y vuelta a la malla original por vecino más cercano.
8. Métricas HECKTOR 2025 (DSC GTVp por paciente, DSCagg y F1agg de GTVn) y 2022 (media de DSCagg).

La arquitectura y el pipeline de `siam_full` en detalle están en `siam_full_arquitectura_pipeline.md`.

## 4. Modelos entrenados

| Modelo | Qué es | Parámetros | GMACs (32×128×128) | s/it (smoke) |
|---|---|---|---|---|
| `base` (`full_base_fold0`) | MambaLite3D, fusión temprana (CT y PET como 2 canales) | 4,05 M | 12,5 | 0,72 |
| `wide` (`exp_wide`) | `base` con más canales | 15,8 M | 39,7 | 0,79 |
| `unet` (`full_unet`) | U-Net residual tipo nnU-Net | 44,6 M | 135,0 | 0,42 |
| `siam_light` | Siamés en los niveles 0–1 + compuerta | 4,05 M | 17,9 | 0,90 |
| `siam_full` | Siamés en todos los niveles: compuerta en 0–3, Cross/Concat Mamba en 4–5 | 4,26 M | 19,7 | 1,13 |

Los s/it de `base`, `siam_light` y `siam_full` vienen de `bifusion/README.md`; los de `wide` y `unet`, de `mamba/README.md`. Son mediciones distintas: sirven para comparar dentro de cada grupo.

## 5. Resultados

### 5.1 Validación (135 casos, sin post-procesado, checkpoint elegido con la métrica 2025)

| Modelo | Checkpoint | GTVp DSC | GTVn DSCagg | GTVn F1agg | Media DSCagg 2022 |
|---|---|---|---|---|---|
| base | best | 0,657 | 0,671 | 0,296 | 0,701 |
| wide | final | 0,646 | 0,703 | 0,333 | 0,721 |
| unet | final | 0,667 | 0,727 | 0,399 | 0,739 |
| siam_light | best | 0,655 | 0,680 | 0,325 | 0,706 |
| **siam_full** | **final** | **0,655** | **0,735** | **0,438** | **0,738** |

Con el checkpoint `best` en todos, siam_full da 0,652 / 0,730 / 0,360 y unet 0,659 / 0,719 / 0,327 (`estadisticas/comparativa_bifusion_mamba.md`).

### 5.2 Test (69 casos, sin post-procesado)

| Modelo | Checkpoint | GTVp DSC | GTVn DSCagg | GTVn F1agg | Media DSCagg 2022 |
|---|---|---|---|---|---|
| base | best | 0,701 | 0,671 | 0,349 | 0,719 |
| wide | final | 0,695 | 0,707 | 0,367 | 0,731 |
| unet | final | 0,711 | 0,744 | 0,432 | 0,757 |
| siam_full | final | pendiente | pendiente | pendiente | pendiente |

Con post-procesado (reglas ajustadas en validación), el base sube en test a GTVp 0,715, GTVn DSCagg 0,714 y F1 0,70 (`mamba/README.md`).

### 5.3 Comparaciones pareadas en validación (135 pacientes, checkpoint best en ambos)

Diferencia B − A con IC del 95 % por bootstrap (`runs/comparaciones`, `estadisticas/comparativa_bifusion_mamba.md`).

| A → B | GTVp DSC | GTVn DSCagg | GTVn F1agg |
|---|---|---|---|
| base → siam_light | −0,002 [−0,019; 0,015] | +0,009 [−0,009; 0,026] | +0,028 [−0,002; 0,059] |
| base → siam_full | −0,004 [−0,019; 0,010] | **+0,059 [0,042; 0,079]** | **+0,063 [0,036; 0,092]** |
| siam_light → siam_full | −0,002 [−0,015; 0,010] | **+0,051 [0,033; 0,069]** | **+0,035 [0,003; 0,067]** |
| unet → siam_full | −0,007 [−0,020; 0,007] | +0,011 [−0,002; 0,025] | +0,033 [−0,001; 0,068] |
| wide → siam_full | **+0,016 [0,000; 0,032]** | **+0,034 [0,013; 0,057]** | **−0,065 [−0,106; −0,024]** |

En negrita, los intervalos que no incluyen el 0. Con los checkpoints `final` en ambos, siam_full supera a base también en GTVp (+0,041 [0,024; 0,057]), porque el GTVp del base cae de 0,657 (best) a 0,615 (final).

## 6. Hallazgos

1. **siam_full mejora los ganglios frente a base y siam_light** (+0,05–0,06 de DSCagg, IC lejos del 0) con +5 % de parámetros y +58 % de GMACs. **Frente a la U-Net de 44,6 M no hay diferencia significativa**: siam_full la iguala con 10 veces menos parámetros.
2. **El GTVp no cambia con la fusión.** Diferencia frente a base −0,004 [−0,019; 0,010]. Entre los 125 pacientes con primario, siam_full (final) saca 0,708 y base (best) 0,709.
3. **La mayor pérdida medida en GTVp está en los 10 pacientes sin primario.** Todos los modelos, incluida la U-Net, predicen ahí un GTVp falso de 2 a 12 ml. Cada uno cuenta como DSC 0, y juntos restan 0,074 al promedio de todos los modelos por igual.
4. **El F1 de ganglios sin post-procesado es inestable entre checkpoints.** siam_full pasa de 0,360 (best) a 0,438 (final); base, de 0,296 a 0,220. El post-procesado lo cambia mucho: en el base, de 0,34 a 0,70 en test.
5. **La caché rodante dejó casos de entrenamiento sin ver** cuando el entrenamiento se cortó y se reanudó, porque cada reanudación vuelve a empezar la caché (`runs/verificacion/cache_entrenamientos.json`):

   | Run | Sesiones | Casos de train nunca vistos (de 476) | Épocas en caché por caso (p50 / p95 / máx.) |
   |---|---|---|---|
   | base | 1 | 1 | 30 / 60 / 99 |
   | unet | 5 | 17 | 26 / 80 / 141 |
   | siam_full | 5 | 31 | 26 / 76,5 / 134 |
   | siam_light | 3 | 39 | 27 / 83 / 132 |
   | wide | 5 | 85 | 21 / 94 / 194 |

   Los modelos no se entrenaron con los mismos datos efectivos, y eso afecta a todas las comparaciones de la sección 5. La caché por ventanas que propone la herramienta (`propuesta_ventana`) pasa por los 476 casos, con 30–36 épocas por caso.
6. **El preprocesamiento es correcto en los 680 casos** (`runs/verificacion`, veredicto "ok", 0 fallos), con 17 avisos:
   - **ROI:** con 270 mm quedan fuera partes de GTVn en 3 casos de MDA (MDA-264 val, 2,3 ml; MDA-059 val, 0,14 ml; MDA-150 test, 0,46 ml), siempre por abajo. Con 300 mm ninguno, a cambio de un 17 % más de vóxeles.
   - **Escala del PET:** CHUM-010 está unas 12 veces por debajo de SUV, y por eso no se encontró el ancla. MDA-012 (test), MDA-167 y MDA-168 (train) están unas 5 veces por encima. Referencia: SUV mediano del tejido blando 0,82 (`runs/experimento_preproc`).
   - **Etiquetas:** el vecino más cercano funde o parte ganglios en 10 casos (7 fusiones, 3 particiones). El remuestreo ganglio por ganglio lo corrige.
   - Otros avisos: volumen de etiqueta en 2 casos (MDA-143, MDA-265) e ida y vuelta en CHUM-054.
7. **La normalización del PET usa toda la caja de la ROI**, con aire y cerebro (`mambalite3d/data/io.py`, `_normalize`). La alternativa con SUV fijo (`pet_norm: clip`, `suv_clip: 20`) está implementada; `exp_suv` se detuvo en la época 0.
8. **AnisoCrossNeXt-3D es viable en la GPU** (`runs/viabilidad_aniso`): 3,40 M parámetros. Con parche 48 × 128 × 128 y lote 1 va a 0,49 s/it, con 2,43 GB de pico y ~8,5 h por entrenamiento (con checkpointing). Con el protocolo actual (32 × 128 × 128, lote 2) necesita checkpointing: 0,66 s/it y 3,0 GB.

## 7. Limitaciones y amenazas a la validez

- **Una sola semilla por configuración.** Los IC de la sección 5.3 miden la variabilidad entre pacientes, no la del entrenamiento.
- **Cobertura de la caché desigual entre runs** (hallazgo 5).
- **Checkpoint distinto según el modelo** (best o final), elegido en validación con 135 casos. Las tablas de `estadisticas/` usan best en todos; las de `runs/resultados*.md`, el elegido.
- **La ganancia en GTVn no está atribuida todavía:** falta `siam_gate` (Mamba frente a compuerta) y un control de cómputo para siam_full. `wide` ya mejora el GTVn del base con más canales.
- **La validación intermedia usa 21 casos** (unos 40 ganglios). Sus decisiones son ruidosas.
- **MDA concentra el 58 % de los casos.**
- **El test interno** ya se usó para base, wide y unet. Para siam_full y las variantes nuevas debe usarse una sola vez, con la configuración final.

## 8. Decisiones tomadas

Según `modelos/bifusion/PENDIENTE.md`, actualización del 14 de octubre a las 21:50:

- ROI de 300 mm.
- Etiquetas ganglio por ganglio (`nodos`) y suaves (`suave`), para comparar.
- Ancla relativa al tejido blando del paciente, sin reescalar la imagen del PET.
- Código escrito: `bifusion3d/data/preproc.py`, `bifusion3d/soft_train.py` y las configs `b0_siam_full_v2` y `b1_siam_full_suave`.

**Falta implementar** (también según `PENDIENTE.md`): la caché por ventanas, las semillas por época para que `--resume` repita lo mismo, los pesos EMA y el bucle `trainer.py`. `b0` y `b1` todavía usan la caché rodante (`refresh_per_epoch: 10`).

## 9. Estado de la cola (15 de octubre, ~15:00)

| Estado | Trabajo | Qué hace |
|---|---|---|
| En curso | `29_evaluar_variantes` | siam_full, siam_light y base en validación con ROI de 270 o 300 mm y vuelta cercana o lineal, sin reentrenar. Termina hacia las 15:30 |
| En cola | `31_experimento_ancla` | Pruebas del preprocesamiento v2 y ancla absoluta frente a relativa en los 680 casos, con ROI de 300 mm. CPU, ~30 min |
| Pausado (plan nuevo) | `50_b0_siam_full_v2`, `51_b1_siam_full_suave` | siam_full con el preprocesamiento v2, con etiquetas por ganglio o suaves (~15 h cada uno) |
| Pausado (control) | `13_siam_gate` | siam_full solo con compuertas (~12 h) |
| Pausado (fuera del plan) | `11_ctrl_ancho`, `14_siam_cga`, `15_base_bifusion`, `16_siam_full_bifusion`, `30_ctx2mm_base` | Plan del 9 de octubre |

Terminados desde el 13 de octubre: `26_verificar_preproc`, `27_viabilidad_aniso` y `28_experimento_preproc`.

## 10. Próximos pasos, por prioridad

1. **Implementar la caché por ventanas y las semillas por época antes de lanzar `50_b0`.** Si no, b0 puede volver a dejar casos sin ver y la comparación con siam_full no vale.
2. **Leer los resultados del 29 y del 31** y fijar la ROI (270 o 300), la vuelta (cercana o lineal) y el ancla.
3. **Post-procesado de siam_full en validación**, con la regla por caso para pacientes sin primario (~1 h, sin entrenar):
   `python -m bifusion3d.tool analyze --run runs/siam_full --split val --checkpoint final`
   `python -m bifusion3d.tool tune_postprocess --run runs/siam_full --checkpoint final`
   El documento del plan lo deja para la fase D; hacerlo ahora mide cuánto de los 0,074 de GTVp se recupera.
4. **`50_b0` y `51_b1`** con la caché nueva.
5. **`13_siam_gate` con el mismo preprocesamiento y la misma caché que b0**, para saber si Cross/Concat Mamba aporta algo frente a una compuerta.
6. **Fases A2 y C del documento del plan:** split fijo de validación cruzada (5 folds sobre 611 casos, test de 69 fijo), CV de la configuración final y ensamble.
7. **Test, una sola vez**, con la configuración elegida.
8. **Opcionales:** normalización del PET con SUV fijo; hipótesis del análisis crítico (pseudo-siamés en N0–N2, compuerta 2·G, espejo solo en x).

## 11. Estado de la documentación (revisado el 15 de octubre)

| Archivo | Fecha | Estado | Qué falta o está mal |
|---|---|---|---|
| `modelos/bifusion/PENDIENTE.md` | 15 oct, 02:34 | Casi al día | La sección 1 todavía dice que el 28 está en marcha y el 29 pendiente (solo la nota del final está al día). No avisa de que `b0` y `b1` usan la caché rodante |
| `modelos/bifusion/README.md` | 14 oct, 23:49 | Parcialmente desactualizado | Dice que los arreglos están "en prueba" y apagados, pero ya se adoptaron. Describe `pet_scale` reescalando CHUM-010, pero se decidió el ancla relativa sin reescalar. No menciona `preproc.py`, `soft_train.py`, `experimento_ancla`, las configs b0 y b1, los trabajos 27–31, 50 y 51, los resultados de la verificación ni la cobertura de la caché. Cita un "documento del plan" que no está en la carpeta |
| `modelos/bifusion/runs/resultados_siam_full.md` | 12 oct | Al día | Incluye a siam_light y base; solo falta el test de siam_full |
| `modelos/bifusion/runs/resultados_siam_light.md` | 10 oct | Superado | Su contenido está en `resultados_siam_full.md` |
| `estadisticas/comparativa_bifusion_mamba.md` | 12 oct | Inconsistente | Usa best en todos los modelos, a diferencia de `resultados*.md`. Dice "+14,2 pp de F1", pero su propia tabla da +6,3 (14,2 sale de comparar siam_full final con base best). Afirma que la fusión "reduce drásticamente los falsos positivos" sin medirlo. Su VRAM de siam_full (1,46 GB) es solo de la red, frente a 1,92 GB del paso completo, y no lo aclara. No tiene fecha |
| `modelos/mamba/TAREAS.md` | 6 oct | Desactualizado | La U-Net figura "en entrenamiento" (ya terminó). La cola 25–34 de mamba ya no corre: la cola única está en bifusion. `exp_suv` se detuvo en la época 0 |
| `modelos/mamba/README.md` | 9 oct | Al día como referencia | Dice que la etiqueta queda al 100 % dentro de la ROI (prueba de 21 casos); en los 680 casos no se cumple en 3 |
| `modelos/mamba/runs/resultados.md`, `comparacion_arquitecturas.md` | 7 y 9 oct | Al día | Solo base, wide y unet |
| `modelos/mamba/DEVIATIONS.md` | 30 sep | Sin cambios detectados | Describe decisiones de diseño; no se revisó línea por línea |

## 12. Fuentes de cada cifra

| Dato | Archivo |
|---|---|
| Resultados de validación y test | `modelos/bifusion/runs/resultados_siam_full.md`, `modelos/mamba/runs/resultados.md` |
| Resultados por paciente | `runs/<modelo>/per_case_val_*.csv` |
| Comparaciones pareadas | `modelos/bifusion/runs/comparaciones/*.json`, `estadisticas/comparativa_bifusion_mamba.md` |
| Auditoría de datos y cobertura de la ROI | `modelos/bifusion/runs/auditoria/resumen.json` |
| Verificación del preprocesamiento y cobertura de la caché | `modelos/bifusion/runs/verificacion/resumen.json`, `cache_entrenamientos.json` |
| Escala del PET, ROI de 270/300/330 mm y remuestreo de etiquetas | `modelos/bifusion/runs/experimento_preproc/resumen.json` |
| Viabilidad de AnisoCrossNeXt-3D | `modelos/bifusion/runs/viabilidad_aniso/resultado.json` |
| Parámetros, GMACs y tiempos | `modelos/bifusion/README.md`, `modelos/mamba/README.md`, `modelos/bifusion/runs/eficiencia_*.json` |
| Partición | `modelos/bifusion/runs/siam_full/split.json` |
