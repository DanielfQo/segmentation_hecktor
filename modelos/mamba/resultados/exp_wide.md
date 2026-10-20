# Resultados exp_wide (MambaLite3D wide, 250 épocas)

Corrida: `runs/exp_wide`, config `configs/exp_wide.yaml`, split estratificado 70/20/10 (476 train, 135 val, 69 test), semilla 12345.
Comparación: `runs/full_base_fold0` (base, checkpoint final, época 250).

## 1. Test (69 casos): con y sin postprocesado

Las reglas de postprocesado se ajustaron solo en validación (`tools.tune_postprocess`, regla `rules_best_dice`) y después se aplicaron al test. Mismo procedimiento para base y wide.

| Modelo | Métrica | Sin postprocesado | Con postprocesado |
|---|---|---|---|
| Base | GTVp DSC | 0.676 | **0.714** |
| Base | GTVn DSCagg | 0.699 | **0.703** |
| Base | GTVn F1agg | 0.345 | **0.664** |
| Base | Media HECKTOR 2022 | 0.719 | **0.741** |
| Wide | GTVp DSC | 0.679 | **0.711** |
| Wide | GTVn DSCagg | 0.704 | **0.740** |
| Wide | GTVn F1agg | 0.266 | **0.703** |
| Wide | Media HECKTOR 2022 | 0.722 | **0.760** |

Reglas elegidas en validación:

- Base: GTVp `min_maxp` 0.99, GTVn `min_ml` 0.2 y `min_maxp` 0.998.
- Wide: GTVp `min_maxp` 0.995, GTVn `min_ml` 1.0 y `min_maxp` 0.995.

## 2. Validación final (135 casos), sin postprocesado

| Modelo | GTVp DSC | GTVn DSCagg | GTVn F1agg | Media HECKTOR 2022 |
|---|---|---|---|---|
| Base | 0.619 | 0.686 | 0.308 | 0.690 |
| Wide | 0.630 | 0.688 | 0.223 | 0.696 |

## 3. Evolución de la validación durante el entrenamiento de wide (135 casos, sin postprocesado)

| Época | GTVp DSC | GTVn DSCagg | GTVn F1agg |
|---|---|---|---|
| 25 | 0.408 | 0.487 | 0.094 |
| 50 | 0.567 | 0.509 | 0.092 |
| 75 | 0.622 | 0.553 | 0.141 |
| 100 | 0.614 | 0.642 | 0.157 |
| 125 | 0.634 | 0.670 | 0.324 |
| 150 | 0.600 | 0.635 | 0.268 |
| 175 | 0.642 | 0.679 | 0.286 |
| 200 | 0.583 | 0.536 | 0.116 |
| 225 | 0.637 | 0.656 | 0.155 |
| 250 (final) | 0.630 | 0.688 | 0.223 |

La curva oscila bastante entre validaciones. El checkpoint final es el que se usa para la comparación; no se seleccionó el mejor por validación.

## 4. Conclusiones

- Sin postprocesado, wide y base están prácticamente empatados en test (media HECKTOR 2022: 0.722 frente a 0.719). La diferencia principal está en GTVn F1agg, donde base es mejor (0.345 frente a 0.266).
- Con postprocesado, wide queda por delante en la media HECKTOR 2022 (0.760 frente a 0.741), sobre todo en GTVn DSCagg (0.740 frente a 0.703). En GTVp DSC base y wide están empatados (0.714 frente a 0.711).
- El postprocesado sube mucho el F1 de GTVn en los dos modelos (de 0.27–0.35 a 0.66–0.70).
- Con 69 casos de test, una diferencia de ~0.02 puede ser ruido. Antes de afirmar que wide es mejor, hay que correr la prueba pareada (`tools.compare_runs`).

## 5. Notas

- La corrida `runs/exp_wide_10ep` (10 épocas) fue una prueba y no forma parte de estos resultados. Su evaluación final se interrumpió; sus archivos siguen en disco.
- El postprocesado de wide se hizo con `tools.analyze` (val y test, checkpoint final), `tools.tune_postprocess` y `tools.evaluate --postprocess`. Log: `runs/exp_wide_postproc.log`.
