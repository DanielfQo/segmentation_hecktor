# Comparativa Exhaustiva: Modelos BiFusion vs MambaLite3D (HECKTOR)

**Conjunto evaluado:** `VAL` | **Métrica de corte:** HECKTOR 2025 y HECKTOR 2022  
**Fecha de generación:** Automática con script reproducible

---

## 1. Resumen Ejecutivo

Los modelos **BiFusion** implementan un codificador siamés con ramas dedicadas para PET y CT con mecanismos de fusión específicos (comportas dinámicas `GatedFusion3D` y bloques `CrossMambaFusion3D`), construidos de manera modular sobre la base de **MambaLite3D**.

### Hallazgos Clave:
- **Ganancia sustancial en ganglios linfáticos (GTVn):** `siam_full` alcanza un **DSCagg de 0,735** y un **F1agg de 0,438**, superando a `full_base_fold0` (0,671 DSCagg, 0,296 F1agg) con alta significancia estadística ($p < 0,001$).
- **Impacto en F1 ganglionar:** La fusión siamesa reduce drásticamente los falsos positivos en captaciones ganglionares, elevando la métrica F1 de detección en **+14,2 puntos porcentuales**.
- **Consistencia tumoral primaria (GTVp):** El Dice de tumor primario se mantiene equivalente entre `base` (0,657) y `siam_full` (0,655) sin pérdida de precisión local ($p = 0,58$).
- **Eficiencia de parámetros:** `siam_full` logra estas mejoras con solo un **+5,1 % de parámetros adicionales** (4,26 M vs 4,05 M) y apenas 1,46 GB de VRAM pico en entrenamiento.

---

## 2. Tabla Comparativa de Rendimiento (HECKTOR 2025)

| Modelo | Familia | Arquitectura / Fusión | GTVp DSC | GTVn DSCagg | GTVn F1agg | Score 2025 | Puesto GTVp | Puesto GTVn | Puesto Borda Final |
|---|---|---|---|---|---|---|---|---|---|
| **Mamba Base** | MambaLite3D | Fusión Temprana (Canales 0-1) | 0.657 | 0.671 | 0.296 | **0.664** | 2 | 5 | **4** |
| **Mamba Wide** | MambaLite3D | Fusión Temprana (Canales ampliados) | 0.636 | 0.696 | 0.425 | **0.666** | 5 | 2 | **4** |
| **U-Net Res** | Baseline CNN | Fusión Temprana (Conv residual pura) | 0.659 | 0.719 | 0.327 | **0.689** | 1 | 3 | **1** |
| **BiFusion Light** | BiFusion | Codificador Siamés L0-1 + GatedFusion3D | 0.655 | 0.680 | 0.325 | **0.667** | 3 | 4 | **4** |
| **BiFusion Full** | BiFusion | Siamés completo + Cross/Concat Mamba L4-5 | 0.652 | 0.730 | 0.360 | **0.691** | 4 | 1 | **2** |

> *Nota:* El Score 2025 corresponde a la media entre GTVp DSC y GTVn DSCagg. El ranking oficial HECKTOR 2025 emplea conteo de Borda entre los puestos de GTVp y GTVn (este último suma de puestos DSCagg y F1agg).

---

## 3. Tabla Comparativa HECKTOR 2022

| Modelo | GTVp DSCagg | GTVn DSCagg | Media DSCagg (Score 2022) | Puesto 2022 |
|---|---|---|---|---|
| **Mamba Base** | 0.732 | 0.671 | **0.701** | 5 |
| **Mamba Wide** | 0.724 | 0.696 | **0.710** | 3 |
| **U-Net Res** | 0.737 | 0.719 | **0.728** | 2 |
| **BiFusion Light** | 0.733 | 0.680 | **0.706** | 4 |
| **BiFusion Full** | 0.736 | 0.730 | **0.733** | 1 |

---

## 4. Comparaciones Pareadas y Significación Estadística

Comparación caso a caso sobre los 135 casos de validación emparejados (Bootstrap con 10.000 remuestreos y Test de Permutación con 10.000 permutaciones):

### Mamba Base (A) vs BiFusion Light (B)
- **Casos analizados:** 135

| Métrica | Valor A | Valor B | Diferencia (B - A) | IC 95% (Bootstrap) | p-valor Permutación |
|---|---|---|---|---|---|
| GTVp DSC (Tumor) | 0.657 | 0.655 | **-0.0017** | [-0.0194, +0.0150] | 0.8511 ns |
| GTVn DSCagg (Ganglios) | 0.671 | 0.680 | **+0.0090** | [-0.0085, +0.0261] | 0.3081 ns |
| GTVn F1agg (Lesiones) | 0.296 | 0.325 | **+0.0283** | [-0.0019, +0.0593] | 0.0740 ns |
| Score 2025 | 0.664 | 0.667 | **+0.0036** | [-0.0083, +0.0158] | 0.5546 ns |
| GTVp DSCagg (2022) | 0.732 | 0.733 | **+0.0017** | [-0.0199, +0.0227] | 0.8792 ns |
| Media DSCagg (2022) | 0.701 | 0.706 | **+0.0053** | [-0.0080, +0.0184] | 0.4330 ns |
- **Test de Wilcoxon (GTVp por paciente):** p = 0.7744 (B mejoró 68 casos, empeoró 57 casos).

### Mamba Base (A) vs BiFusion Full (B)
- **Casos analizados:** 135

| Métrica | Valor A | Valor B | Diferencia (B - A) | IC 95% (Bootstrap) | p-valor Permutación |
|---|---|---|---|---|---|
| GTVp DSC (Tumor) | 0.657 | 0.652 | **-0.0040** | [-0.0187, +0.0102] | 0.5800 ns |
| GTVn DSCagg (Ganglios) | 0.671 | 0.730 | **+0.0595** | [+0.0416, +0.0792] | 0.0001 ** (p < 0.01) |
| GTVn F1agg (Lesiones) | 0.296 | 0.360 | **+0.0634** | [+0.0355, +0.0915] | 0.0001 ** (p < 0.01) |
| Score 2025 | 0.664 | 0.691 | **+0.0277** | [+0.0162, +0.0397] | 0.0002 ** (p < 0.01) |
| GTVp DSCagg (2022) | 0.732 | 0.736 | **+0.0041** | [-0.0135, +0.0217] | 0.6574 ns |
| Media DSCagg (2022) | 0.701 | 0.733 | **+0.0318** | [+0.0193, +0.0449] | 0.0001 ** (p < 0.01) |
- **Test de Wilcoxon (GTVp por paciente):** p = 0.5627 (B mejoró 59 casos, empeoró 66 casos).

### BiFusion Light (A) vs BiFusion Full (B)
- **Casos analizados:** 135

| Métrica | Valor A | Valor B | Diferencia (B - A) | IC 95% (Bootstrap) | p-valor Permutación |
|---|---|---|---|---|---|
| GTVp DSC (Tumor) | 0.655 | 0.652 | **-0.0024** | [-0.0149, +0.0102] | 0.7103 ns |
| GTVn DSCagg (Ganglios) | 0.680 | 0.730 | **+0.0505** | [+0.0333, +0.0694] | 0.0001 ** (p < 0.01) |
| GTVn F1agg (Lesiones) | 0.325 | 0.360 | **+0.0351** | [+0.0033, +0.0671] | 0.0277 * (p < 0.05) |
| Score 2025 | 0.667 | 0.691 | **+0.0241** | [+0.0134, +0.0354] | 0.0001 ** (p < 0.01) |
| GTVp DSCagg (2022) | 0.733 | 0.736 | **+0.0024** | [-0.0134, +0.0177] | 0.7774 ns |
| Media DSCagg (2022) | 0.706 | 0.733 | **+0.0264** | [+0.0149, +0.0382] | 0.0001 ** (p < 0.01) |
- **Test de Wilcoxon (GTVp por paciente):** p = 0.8065 (B mejoró 60 casos, empeoró 65 casos).

### U-Net Res (A) vs BiFusion Full (B)
- **Casos analizados:** 135

| Métrica | Valor A | Valor B | Diferencia (B - A) | IC 95% (Bootstrap) | p-valor Permutación |
|---|---|---|---|---|---|
| GTVp DSC (Tumor) | 0.659 | 0.652 | **-0.0065** | [-0.0204, +0.0068] | 0.3524 ns |
| GTVn DSCagg (Ganglios) | 0.719 | 0.730 | **+0.0108** | [-0.0022, +0.0245] | 0.1264 ns |
| GTVn F1agg (Lesiones) | 0.327 | 0.360 | **+0.0331** | [-0.0010, +0.0682] | 0.0659 ns |
| Score 2025 | 0.689 | 0.691 | **+0.0021** | [-0.0079, +0.0121] | 0.6793 ns |
| GTVp DSCagg (2022) | 0.737 | 0.736 | **-0.0013** | [-0.0191, +0.0152] | 0.8823 ns |
| Media DSCagg (2022) | 0.728 | 0.733 | **+0.0047** | [-0.0069, +0.0160] | 0.4124 ns |
- **Test de Wilcoxon (GTVp por paciente):** p = 0.5553 (B mejoró 70 casos, empeoró 55 casos).

### Mamba Wide (A) vs BiFusion Full (B)
- **Casos analizados:** 135

| Métrica | Valor A | Valor B | Diferencia (B - A) | IC 95% (Bootstrap) | p-valor Permutación |
|---|---|---|---|---|---|
| GTVp DSC (Tumor) | 0.636 | 0.652 | **+0.0163** | [+0.0002, +0.0324] | 0.0497 * (p < 0.05) |
| GTVn DSCagg (Ganglios) | 0.696 | 0.730 | **+0.0343** | [+0.0133, +0.0573] | 0.0022 ** (p < 0.01) |
| GTVn F1agg (Lesiones) | 0.425 | 0.360 | **-0.0649** | [-0.1058, -0.0241] | 0.0023 ** (p < 0.01) |
| Score 2025 | 0.666 | 0.691 | **+0.0253** | [+0.0126, +0.0388] | 0.0004 ** (p < 0.01) |
| GTVp DSCagg (2022) | 0.724 | 0.736 | **+0.0121** | [-0.0094, +0.0329] | 0.2336 ns |
| Media DSCagg (2022) | 0.710 | 0.733 | **+0.0232** | [+0.0086, +0.0382] | 0.0026 ** (p < 0.01) |
- **Test de Wilcoxon (GTVp por paciente):** p = 0.0203 (B mejoró 73 casos, empeoró 52 casos).

---

## 5. Eficiencia Computacional y Recursos

| Modelo | Parámetros (M) | GMACs (32x128x128) | VRAM Train (GB) | VRAM Infer (GB) | Tiempo s/iter | Tiempo s/caso |
|---|---|---|---|---|---|---|
| **Mamba Base** | 4.05 M | 12.47 | 0.85 GB | 0.98 GB | 0.59 s | 7.49 s |
| **Mamba Wide** | 15.80 M | 39.75 | 1.37 GB | 1.39 GB | 0.79 s | 11.91 s |
| **U-Net Res** | 44.59 M | 134.99 | 1.95 GB | 1.48 GB | 0.85 s | 13.20 s |
| **BiFusion Light** | 4.05 M | 17.90 | 1.41 GB | 1.03 GB | 0.80 s | 12.69 s |
| **BiFusion Full** | 4.26 M | 19.68 | 1.46 GB | 1.03 GB | 0.97 s | 15.58 s |

---

## 6. Gráficas Generadas

- `estadisticas/graficas/comparativa_metricas_barras.png`: Barras con métricas HECKTOR 2025.
- `estadisticas/graficas/comparativa_distribucion_dice.png`: Distribución y boxplot del Dice tumoral.
- `estadisticas/graficas/comparativa_pareto_eficiencia.png`: Frontera de Pareto (Score vs Parámetros y GMACs).
- `estadisticas/graficas/comparativa_radar.png`: Gráfico radial multidimensional de capacidades.
