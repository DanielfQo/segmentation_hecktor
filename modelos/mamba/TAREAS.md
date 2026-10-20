# Tareas: MambaLite3D en HECKTOR

Actualizado: 6 de octubre de 2026, 09:30

**Meta:** DSC agregado medio (GTVp + GTVn, métrica de HECKTOR 2022) ≥ 0,75 **sin post-procesado**, con menos recursos que el estado del arte (ganador de 2022: 0,788).

Los comandos se ejecutan en Anaconda Prompt, dentro de `modelos\mamba`, después de `conda activate mambalite3d`. Deja la laptop enchufada (con batería se suspende a las 3 h), con la tapa abierta y con `runner.bat` abierto.

## Cómo funciona la cola

- `runner.bat` ejecuta uno por uno, en orden, los `.bat` de `jobs\pending`.
- Avance de cada trabajo: `jobs\logs\<trabajo>.log`. Avance de cada entrenamiento: `runs\<experimento>\log.txt`.
- Para pausar un trabajo que aún no empezó, mueve su `.bat` de `jobs\pending` a `jobs\paused`.
- Si la laptop se apaga a mitad de un entrenamiento, vuelve a poner el mismo `.bat` en `jobs\pending`: todos usan `--resume` y siguen desde la última época guardada.
- No presiones Ctrl+C en la ventana del ejecutor: detiene el entrenamiento. Para copiar texto, selecciónalo con el mouse y presiona Enter, o abre los logs.

---

## Resultados verificados (5 de octubre)

Recalculados desde los CSV por paciente; coinciden con los JSON. Los tres modelos usan el mismo split (476 / 135 / 69).

### Dice (métrica 2025, sin post-procesado, espacio original)

Formato: score (GTVp / GTVn / F1 de ganglios). ✓ = checkpoint elegido por validación.

| Modelo | Checkpoint | Validación (135) | Test (69) |
|---|---|---|---|
| base | final (época 250) | 0,647 (0,615 / 0,680 / 0,22) | 0,670 (0,646 / 0,694 / 0,28) |
| base | best (época 225) ✓ | 0,664 (0,657 / 0,671 / 0,30) | **0,686** (0,701 / 0,671 / 0,35) |
| wide | final (época 250) ✓ | 0,674 (0,646 / 0,703 / 0,33) | **0,701** (0,695 / 0,707 / 0,37) |
| wide | best (época 225) | 0,666 (0,636 / 0,696 / 0,42) | 0,685 (0,681 / 0,689 / 0,49) |
| U-Net | en entrenamiento | época 50, 21 casos: 0,542 (0,617 / 0,467 / 0,07) | pendiente |

Métrica 2022 (DSC agregado medio): base best 0,701 en validación y 0,719 en test. El resto sale con el trabajo 24.

### Eficiencia (medida el 5 de octubre con `tools/efficiency.py`)

Un patch de 32 × 128 × 128. Entrenamiento: batch 2 con AMP, solo la red. Inferencia: una ROI típica de 104 × 288 × 448 sin TTA, solo la red.

| Modelo | Parámetros | GMACs | GFLOPs | VRAM entrenamiento (red) | s/iteración | VRAM inferencia | s/caso | s/caso con TTA |
|---|---|---|---|---|---|---|---|---|
| base | 4,05 M | 12,5 | 24,9 | 0,85 GB | 0,59 | 0,98 GB | 7,5 | 26,9 |
| wide | 15,8 M | 39,7 | 79,5 | 1,37 GB | 0,79 | 1,39 GB | 11,9 | 46,8 |
| U-Net | 44,6 M | 135,0 | 270,0 | 1,84 GB | 0,43 | 1,81 GB | 7,6 | 29,6 |

- GMACs es lo que fvcore y thop (y la mayoría de papers) llaman "GFLOPs"; aquí GFLOPs = 2 × GMACs. Al citar, indica la convención.
- VRAM pico del entrenamiento completo, con la aumentación en GPU: base 1,33 GB, wide 1,93 GB, U-Net 2,12 GB.
- La U-Net es tan rápida como el base aunque tiene 11 veces más operaciones. En Windows, Mamba corre en PyTorch puro, sin los kernels de CUDA.

## PRIORIDAD: fase de arquitectura (antes de las ablaciones)

Implementado el 19 de octubre en `mambalite3d/models/network.py`. Dos preguntas de diseño:

- **¿Dónde combinar CT y PET?** `base` los junta en la entrada. `dual` (`DualEncoder`) usa un codificador completo por modalidad y los combina en cada nivel (concatenación + convolución 1×1×1). `dualmid` separa solo los niveles convolucionales 0 a 3 y comparte los niveles con Mamba (`dual_fusion: mid`). `dualmid` frente a `dual` solo difiere en si los niveles con Mamba van separados o compartidos, con el mismo cómputo.
- **¿En qué resolución aporta Mamba?** `m3lite` y `m4lite` ponen Mamba también en el nivel 3 (2048 posiciones) o en los niveles 2 y 3 (16 384 posiciones). La atención global (LGFM3D) queda solo en los 2 niveles más profundos, como en `base` (`n_lgfm_levels: 2`), así que cada paso añade solo Mamba. Con atención en el nivel 2 el entrenamiento sería 2,2 veces más lento.

Medido con `mamba_ssm`, patch 32×128×128 (los GFLOPs con `mamba_ssm` no incluyen el barrido de Mamba, que es pequeño; para el paper, medirlos con `MAMBALITE_FORCE_TORCH=1`):

| Modelo | Parámetros | GFLOPs | VRAM entrenamiento | s/iteración | s/caso |
|---|---|---|---|---|---|
| base | 4,05 M | 24,8 | 0,78 GB | 0,29 | 3,8 |
| dual | 5,26 M | 37,8 | 1,22 GB | 0,41 | 5,3 |
| dualmid | 4,31 M | 37,6 | 1,21 GB | 0,38 | 5,2 |
| m3lite | 4,05 M | 24,8 | 0,80 GB | 0,30 | 4,0 |
| m4lite | 4,06 M | 25,0 | 0,94 GB | 0,32 | 4,4 |
| wide | 15,80 M | 79,5 | 1,32 GB | 0,34 | 5,6 |
| dualwide | 20,56 M | 120,4 | 2,06 GB | 0,57 | 8,6 |

Lanzar toda la cola en Linux (reanudable; `--lista` muestra el estado):
```
./cola_experimentos.sh --lista
nohup ./cola_experimentos.sh > runs/cola.log 2>&1 &
```
Orden: fase de arquitectura (1 `exp_dual`, 2 `arq_dualmid`, 3 `exp_dualwide`, 4 `arq_m3lite`, 5 `arq_m4lite`); después 6 `full_unet`, 7 `full_base_heldout_chup`, 8 `exp_suv`, 9 `full_ctx2mm`, 10 `exp_patch160`, 11 `full_base_e500`. Cada uno entrena, evalúa val y test con el checkpoint final y aplica el post-procesado. Comparar cada uno con su referencia con `tools/compare_runs` (prueba pareada por paciente).

- Decisión tras `exp_dual` y `arq_dualmid`: qué fusión usar en la versión ancha. Si gana `dualmid`, conviene cambiar `exp_dualwide` por la fusión intermedia antes de que empiece.
- Las ablaciones (sin Mamba, sin CGA3D, mismo barrido) quedan fuera de la cola hasta elegir la arquitectura final: se harán sobre ese modelo, no sobre base.

---

## Cola de trabajos de Windows (obsoleta: reemplazada por `cola_experimentos.sh`)

Fechas aproximadas, si la laptop no se apaga.

| # | Trabajo | Qué hace | Duración | Estado |
|---|---|---|---|---|
| 21 | `21_pruebas_metricas` | 97 pruebas, evaluación rápida con las métricas 2022 y 2025, y el reporte | ~10 min | ✓ hecho (5 oct) |
| 22 | `22_reevaluar_metricas` | Se cortó el 5 oct a las 17:32 sin terminar ninguna evaluación; pasó al trabajo 25 | — | cancelado |
| 23 | `23_unet` | Reanudó la U-Net el 6 oct a las 09:23 desde la época 60; luego la evalúa y actualiza el reporte | ~8 h | en curso (termina ~17:00 del 6 oct) |
| 25 | `25_tta_val` | Reevalúa base y wide (best y final, validación y test) con las dos métricas, actualiza el reporte y hace TTA en validación (base best, wide final) | ~7,5 h | pendiente |
| 26 | `26_ensamble_val` | Ensamble base + wide en validación, sin y con TTA (`runs\ens_base_wide`) | ~4,5 h | pendiente |
| 27 | `27_exp_suv` | PET en SUV absoluto (base) | ~11 h | pendiente |
| 28 | `28_exp_patch160` | Más contexto: patch 48 × 160 × 160 (base) | ~21 h | pendiente |
| 29 | `29_exp_ctx2mm` | Más contexto: 2 mm en el plano (base) | ~18 h | pendiente |
| 30 | `30_base_e500` | 500 épocas (base) | ~20 h | pendiente |
| 31 | `31_heldout_chup` | Generalización: CHUP fuera del entrenamiento (base) | ~11 h | pendiente |
| 32 | `32_abl_nomamba` | Ablación: Mamba reemplazado por MLP | ~10 h | pendiente |
| 33 | `33_abl_samescan` | Ablación: mismo orden de barrido en todos los grupos | ~10 h | pendiente |
| 34 | `34_abl_nocga` | Ablación: sin CGA en los skips | ~10 h | pendiente |

Todo junto son unos 6 días de GPU.

---

## Lista de tareas

### 1. U-Net (baseline)
- [x] Verificar el estado: se había cortado el 4 de octubre a las 00:22, en la época 51.
- [ ] Terminar el entrenamiento y evaluarla (trabajo 23; reanudado el 6 oct desde la época 60).
- [ ] Compararla con base y wide con la prueba pareada:
  ```
  python -m mambalite3d.tools.compare_runs --a runs/full_base_fold0:best --b runs/full_unet:best --split test
  ```

### 2. Subir el Dice sin reentrenar
- [x] Métrica 2022 en `evaluate` (`gtvp_dice_agg`, `dsc_agg_mean`, `score_2022` y columnas nuevas en `per_case_*.csv`).
- [x] Métricas oficiales de las dos ediciones con sus rankings (`eval/hecktor.py`) y tabla de resultados (`tools/report.py` → `runs\resultados.md`):
  ```
  python -m mambalite3d.tools.report --models base=runs/full_base_fold0 wide=runs/exp_wide unet=runs/full_unet
  ```
- [x] Comparación pareada: `tools/compare_runs.py`.
- [x] Ensamble en `evaluate` (`--ensemble RUN:CHECKPOINT ...`). Rechaza modelos con otro split o con otro preprocesamiento.
- [ ] Reevaluar base y wide con la métrica 2022 (trabajo 25, después de la U-Net).
- [ ] TTA en validación (trabajo 25) y ensamble en validación (trabajo 26).
- [ ] **Decidir con validación** la mejor configuración de inferencia (modelo solo, TTA o ensamble) y recién entonces correrla en test.

### 3. Reentrenar para reducir ganglios falsos
- [x] Config `configs/exp_suv.yaml` (PET = min(SUV, 20) / 20).
- [ ] PET en SUV absoluto (trabajo 27).
- [ ] Más contexto (trabajos 28 y 29). Ahora validan con 21 casos, igual que el base.
- [ ] 500 épocas (trabajo 30).
- [ ] Aplicar a wide lo que mejore al base (nuevos trabajos cuando haya resultados).

### 4. Generalización
- [ ] Entrenar con CHUP fuera (trabajo 31).

### 5. Eficiencia
- [x] Contador de GFLOPs (`tools/efficiency.py`): convoluciones, lineales y matmul con FlopCounterMode; barrido de Mamba y atención de forma analítica.
- [x] Tabla de base, wide y U-Net (arriba).

### 6. Ablaciones
- [ ] `nomamba`, `samescan` y `nocga` (trabajos 32 a 34).
- [ ] Opcional: variante con el módulo NRM de Diff-UMamba (no implementada).

### 7. Redacción de la tesis
- [ ] Actualizar el objetivo general: alcanzar un DSC agregado medio ≥ 0,75 en HECKTOR, cercano al estado del arte, con menos recursos.
- [ ] Declarar la métrica (HECKTOR 2022) y reportar también la de 2025. Para los GFLOPs, indicar la convención.
- [ ] Explicar que el test interno (69 casos de los mismos centros) no es el test oficial (centros no vistos), y respaldarlo con el experimento sin CHUP.

## Decisiones tomadas

- Carpeta limpia (5 de octubre): en `runs` solo quedan base (`full_base_fold0`), `exp_wide`, `full_unet` y `eficiencia.json`. Se borraron las corridas de prueba (smoke, crash, CV de prueba, overfit), los logs de trabajos viejos, las cachés y los `checkpoint_latest.pth` de base y wide, que repetían el final.
- Sin post-procesado (4 de octubre). Como referencia, con post-procesado el base llegaba a 0,750 en test.
- El checkpoint se elige con validación y el test se usa una sola vez por configuración final.
- Todos los entrenamientos usan el mismo protocolo que el base: mismo split, 250 épocas y validación intermedia con 21 casos. Las excepciones son la variable que se prueba y wide, que ya estaba entrenado con los 135 casos.
- Pendiente: elegir entre `final` y `best` con la métrica 2022, que es la de la meta. Hasta ahora se eligió con la de 2025.
