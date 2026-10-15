# MambaLite3D para HECKTOR 2025 (PET/CT, Task 1)

U-Net 3D bimodal (CT + PET) con Mamba solo en los niveles profundos, AMF3D multidireccional, LGFM3D y
CGA3D en los skips, adaptada de MambaLiteUNet y SegMamba. Todo el preprocesamiento (ROI de cabeza y
cuello, remuestreo, normalización) y la augmentation se hacen **al vuelo**: no se escribe ninguna copia
del dataset en disco.

## Estructura

```
mamba/
├── setup_env.bat            crea el entorno conda "mambalite3d" (PyTorch + CUDA + dependencias)
├── requirements.txt         dependencias (PyTorch se instala aparte con su CUDA)
├── requirements-lock.txt    versiones exactas instaladas en este portátil
├── configs/
│   ├── smoke_subset.yaml    prueba corta: 5 casos por centro, split 70/20/10 estratificado
│   ├── smoke_heldout.yaml   prueba corta con CHUP como test aislado
│   ├── full.yaml            entrenamiento completo, split 70/20/10 estratificado por centro
│   ├── full_heldout_chup.yaml  entrenamiento completo, test = institución CHUP aislada
│   ├── full_cv7.yaml        validación cruzada de 7 folds estratificada por centro
│   ├── full_loco.yaml       validación cruzada leave-one-center-out (7 hospitales = 7 folds)
│   └── smoke_cv7.yaml, smoke_resume.yaml  pruebas cortas de CV y de reanudación
├── mambalite3d/
│   ├── data/                lectura NIfTI, ROI, caché en RAM, muestreo de patches, augmentation GPU
│   ├── models/              MambaLite3D, Mamba en PyTorch puro, AMF3D/LGFM3D/CGA3D, plan de strides
│   ├── training/            entrenamiento, pérdida Dice+CE con deep supervision, inferencia sliding window
│   ├── eval/                métricas HECKTOR (DSC GTVp, DSC agregado GTVn, F1 de ganglios)
│   └── tools/               cross_validate, evaluate, analyze, tune_postprocess, check_run, check_env, smoke_vram
├── tests/                   79 tests (pytest)
├── runs/                    resultados de cada experimento
├── jobs/ + runner.bat       cola de trabajos usada para lanzar comandos en este portátil (opcional)
└── DEVIATIONS.md            diferencias con los papers/plan y por qué
```

El dataset se lee de `../../HECKTOR2025_raw` (una carpeta por caso con `<caso>__CT.nii.gz`,
`<caso>__PT.nii.gz` y `<caso>.nii.gz`). Se cambia en `data.root` del YAML.

## Instalación

El entorno ya está creado en este portátil (`mambalite3d`, Python 3.11, PyTorch 2.14 + CUDA 13.0).
En otra máquina Windows con Miniconda en `%USERPROFILE%\miniconda3`:

```bat
cd C:\ruta\a\petct\modelos\mamba
setup_env.bat
```

En Linux se puede instalar además `mamba_ssm` (kernels CUDA); el código lo detecta y lo usa solo.

## Uso

Desde "Anaconda Prompt" (o cmd tras activar conda):

```bat
conda activate mambalite3d
cd C:\Users\danie\OneDrive\Documentos\Universidad\petct\modelos\mamba

:: comprobar el entorno y el código
python -m pytest -q tests
python -m mambalite3d.tools.check_env

:: prueba corta (~10 min)
python -m mambalite3d.training.train --config configs/smoke_subset.yaml

:: entrenamiento completo, opcion A: split 70/20/10 estratificado por centro
python -m mambalite3d.training.train --config configs/full.yaml

:: entrenamiento completo, opcion B: test = institucion aislada (CHUP)
python -m mambalite3d.training.train --config configs/full_heldout_chup.yaml

:: validacion cruzada de 7 folds (7 entrenamientos seguidos; ver el reparto sin entrenar con --dry-run)
python -m mambalite3d.tools.cross_validate --config configs/full_cv7.yaml --dry-run
python -m mambalite3d.tools.cross_validate --config configs/full_cv7.yaml

:: continuar un entrenamiento interrumpido (apagon, cierre de la ventana, error...)
python -m mambalite3d.training.train --config configs/full.yaml --resume
:: en validacion cruzada basta con repetir el mismo comando de cross_validate

:: evaluar el mejor checkpoint (segun validacion) en el test
python -m mambalite3d.tools.evaluate --run runs/full_base_fold0 --split test --checkpoint best

:: post-procesado: 1) analizar las predicciones (GPU, una vez), 2) ajustar reglas en validacion (sin GPU),
:: 3) evaluar el test con esas reglas
python -m mambalite3d.tools.analyze --run runs/full_base_fold0 --split val --checkpoint final
python -m mambalite3d.tools.analyze --run runs/full_base_fold0 --split test --checkpoint final
python -m mambalite3d.tools.tune_postprocess --run runs/full_base_fold0 --checkpoint final
python -m mambalite3d.tools.evaluate --run runs/full_base_fold0 --split test --checkpoint final --postprocess runs/full_base_fold0/postprocess_final.json

:: test-time augmentation con flips (mas lento)
python -m mambalite3d.tools.evaluate --run runs/full_base_fold0 --split test --checkpoint final --mirror

:: ensamble: promedia las probabilidades de varios modelos (mismo split y mismo preprocesamiento);
:: resultados en runs/ens_<nombre>/ con la etiqueta <split>_ens[_tta]
python -m mambalite3d.tools.evaluate --ensemble runs/full_base_fold0:best runs/exp_wide:final --name base_wide --split val

:: comparacion pareada de dos evaluaciones (bootstrap, permutacion y Wilcoxon por paciente)
python -m mambalite3d.tools.compare_runs --a runs/full_base_fold0:best --b runs/exp_wide:final --split test

:: tabla de eficiencia: parametros, GMACs/GFLOPs, VRAM y tiempo de entrenamiento e inferencia
python -m mambalite3d.tools.efficiency --models runs/full_base_fold0:best runs/exp_wide:final runs/full_unet:best --tta
```

Cada evaluación guarda en `metrics_*.json` las métricas oficiales de las dos ediciones (`mambalite3d/eval/hecktor.py`)
y escribe una línea `resumen [...]` en el log con ambas:

- `hecktor2025`: `GTVp_DSC` (Dice por paciente, promediado; 1 si predicción y referencia están vacías, 0 si solo
  una lo está), `GTVn_DSCagg` (Dice agregado) y `GTVn_F1agg` (F1 de detección, acierto con IoU > 0,3).
  El ranking oficial es el conteo de Borda de un puesto en GTVp y uno en GTVn (Borda de DSCagg y F1agg).
- `hecktor2022`: `GTVp_DSCagg`, `GTVn_DSCagg` y `DSCagg_media` (su promedio, con el que se ordenaba el challenge).

`score` (promedio de GTVp DSC y GTVn DSCagg) es solo el número con el que se elige el mejor checkpoint.
Los `per_case_*.csv` antiguos no tienen los volúmenes del GTVp: para la métrica 2022 hay que volver a evaluar.

```bat
:: tabla de resultados de varios modelos con las dos métricas y sus rankings (runs/resultados.md y .json)
python -m mambalite3d.tools.report --models base=runs/full_base_fold0 wide=runs/exp_wide unet=runs/full_unet
:: --select 2022 elige el checkpoint (final o best) con la métrica 2022 en validación

:: comparativa integral de arquitecturas (parámetros, GMACs, VRAM, métricas HECKTOR y gráfica 4 paneles)
python comparar_arquitecturas.py
:: o como herramienta del módulo:
python -m mambalite3d.tools.compare_architectures --variants base wide unet
```

Opciones útiles de `train`: `--name` (carpeta en `runs/`), `--variant`, `--epochs`, `--iters`,
`--batch`, `--patch Z Y X`, `--overfit N` (sobreajustar N casos sin augmentation, para depurar),
`--no-aug`, `--resume`.

### Splits train / validación / test (`data.split` en el YAML)

| modo | cómo se reparte | train / val / test (680 casos) | qué mide el test |
|---|---|---|---|
| `stratified` (`full.yaml`) | 70/20/10 dentro de cada centro | 476 / 135 / 69 | rendimiento en los mismos 7 centros (misma distribución) |
| `center` + `test_centers: [CHUP]` (`full_heldout_chup.yaml`) | todo CHUP es test; el resto 70:20 por centro | 474 / 134 / 72 | generalización a una institución nunca vista |

El test nunca se usa durante el entrenamiento. La validación se usa para elegir el mejor checkpoint
(`checkpoint_best.pth`, score = media de DSC GTVp y DSC agregado GTVn sobre `val_cases_during_training`
casos repartidos por centro, cada `val_every` épocas). Al terminar se evalúan todos los casos de validación y
de test con los pesos finales, con métricas globales y por centro. Otro centro aislado: cambia `test_centers`.

### Validación cruzada

| config | folds | train / val / test por fold | nota |
|---|---|---|---|
| `full_cv7.yaml` (`n_folds: 7`, estratificado) | 7 | ~452 / ~130 / ~97 (66 / 19 / 14 %) | cada centro repartido en los 7 folds |
| `full_loco.yaml` (`mode: loco`) | 7 (uno por hospital) | variable: 222-521 / 62-148 / 11-396 | el fold MDA deja el 58 % en test |

Con 7 folds el test de cada fold es 1/7 ≈ 14 % (no 10 %); train:val se mantiene 70:20. Cada caso es test
exactamente una vez. Cada fold se guarda en `runs/<experimento>_fold<k>/` y al final
`runs/<experimento>_cv_summary.json` tiene las métricas por fold, media ± desviación y las métricas
agregadas sobre todos los casos de test. Un fold suelto: `train --config configs/full_cv7.yaml --fold 3`.

### Reanudar tras un corte

- `checkpoint_latest.pth` se guarda en cada época de forma atómica (archivo temporal + renombrado), igual
  que `history.json`, `best.json` y `split.json`: un corte nunca deja archivos a medias.
- `--resume` continúa desde la última época guardada (se pierde como mucho una época). Si el corte fue
  durante la evaluación final, solo repite la evaluación. Si no hay checkpoint, empieza de cero.
- Si el split ya no coincide con `split.json` (otro config, semilla o fold), se niega a reanudar para no
  mezclar casos de test con entrenamiento.
- `cross_validate` salta los folds terminados (`done.json`) y reanuda el fold interrumpido.

### Qué se guarda en `runs/<nombre>/`

- `log.txt`: todo el registro (reparto por centro, ROI, cobertura de la etiqueta, pérdida, s/iteración, VRAM, ETA).
- `history.json`: una entrada por época. `split.json`: casos de train/val/test. `config_used.json`.
- `checkpoint_latest.pth` (para `--resume`), `checkpoint_best.pth` (mejor en validación), `checkpoint_final.pth` y `done.json` (run terminado).
- `metrics_val_final.json` y `metrics_test_final.json` (espacio del ROI, malla original y por centro),
  `per_case_*.csv`, `metrics_epN.json` (validaciones intermedias) y `qc_*.png` (CT, PET, referencia, predicción).

### Variantes para las ablaciones de la tesis (`--variant`)

| variante   | qué cambia                                                        |
|------------|-------------------------------------------------------------------|
| `base`     | MambaLite3D completo (4 grupos AMF con órdenes fwd/rev/slice/ap)  |
| `samescan` | todos los grupos AMF en el mismo orden (aísla la aportación 3D)   |
| `nomamba`  | Mamba sustituido por un MLP de tokens con los mismos parámetros   |
| `noamf`    | sin AMF3D (bloque residual convolucional)                         |
| `nolgfm`   | sin LGFM3D                                                        |
| `nocga`    | sin CGA3D en los skips                                            |
| `g1`, `g2` | 1 o 2 grupos en AMF                                               |
| `m3`       | Mamba en 3 niveles en vez de 2                                    |
| `unet`     | U-Net residual tipo nnU-Net como referencia (44,6 M parámetros)   |
| `wide`     | MambaLite3D más ancho: canales 24-32-64-96-128-192, bottleneck 256 |
| `wide2`    | aún más ancho: canales 32-48-96-128-192-256, bottleneck 320       |

Experimento de contexto: `configs/exp_ctx2mm.yaml` (spacing 2 × 2 × 3 mm, patch 48 × 96 × 96 →
144 × 192 × 192 mm de contexto con un coste parecido al actual).

Ejemplo: `python -m mambalite3d.training.train --config configs/full.yaml --variant samescan --name full_samescan_fold0`

## Datos al vuelo y uso de disco/RAM

- Cada caso se lee del NIfTI original, se reorienta a LPS, se calcula el ROI (cuerpo = CT > -500 HU o
  SUV > 0,5, componente principal para quitar la camilla; tope en el SUV > 2 más craneal; 270 mm hacia
  abajo) y se remuestrea directamente a 1x1x3 mm dentro del ROI. CT en ventana [-200, 400] → [0, 1];
  PET normalizado (z-score) dentro del ROI.
- Los casos quedan en RAM en float16 (~60 MB por caso). `full.yaml` usa una caché rodante de 60 casos
  (~3,6 GB) que renueva 10 casos por época en segundo plano; la validación final se hace en streaming.
- La augmentation (rotación, escala, espejo, brillo, contraste, gamma, ruido, blur) se aplica en la GPU a
  cada batch, así que no hay datos aumentados en disco.
- En disco solo se escriben los checkpoints (~33 MB el de reanudación, ~16 MB el final) y los logs.

## Rendimiento medido en este portátil (RTX 3050 Laptop 4 GB)

Batch 2, una iteración de entrenamiento con AMP (medido el 01-10-2026 con `smoke_vram`):

| configuración | parámetros | VRAM pico | s/iteración |
|---|---|---|---|
| `base`, patch 32x128x128 (1x1x3 mm) | 4,0 M | 1,33 GB | 0,5-0,6 |
| `base`, patch 48x160x160 | 4,0 M | 2,97 GB | 0,99 |
| `base`, patch 64x192x192 | 4,1 M | 3,44 GB (al límite de 4 GB) | 2,75 |
| `wide`, patch 32x128x128 | 15,8 M | 1,80 GB | 0,79 |
| `wide`, patch 48x160x160 | 15,8 M | 2,99 GB (se desborda) | 4,08 |
| `wide2`, patch 32x128x128 | 28,7 M | 2,26 GB | 0,99 |
| `unet` (nnU-Net-like), 32x128x128 | 44,6 M | 2,11 GB | 0,42 |
| `base`, 2x2x3 mm, patch 48x96x96 (144x192x192 mm) | 4,0 M | 1,24 GB | 0,97 |
| `base`, 2x2x3 mm, patch 64x112x112 | 4,1 M | 2,19 GB | 1,60 |
| `wide`, 2x2x3 mm, patch 48x96x96 | 15,7 M | 2,13 GB | 1,75 |

`full.yaml` (250 épocas x 250 iteraciones, patch 32x128x128) tardó **~9,5 h** en este portátil
(~0,5 s/iteración) más ~1 h de evaluación final. Con la A1000 de 8 GB conviene `--patch 48 160 160` o mayor.

Consejos para entrenamientos largos en Windows: portátil enchufado y sin suspensión, cerrar navegadores
(usan VRAM de la GPU) y marcar la carpeta del dataset en OneDrive como "Conservar siempre en este dispositivo".

## Post-procesado (medido en fold0)

`runs/full_base_fold0/postprocess_rules.json`: GTVp = solo la región más grande; GTVn = se borran regiones
< 0,5 ml o con probabilidad máxima < 0,995. Ajustado solo en validación; en el test (69 casos, checkpoint best):
DSC GTVp 0,701 → 0,715, DSC agregado GTVn 0,673 → 0,714, F1 de ganglios 0,34 → 0,70.

```bat
python -m mambalite3d.tools.evaluate --run runs/full_base_fold0 --split test --checkpoint best --postprocess runs/full_base_fold0/postprocess_rules.json
```

## Verificación realizada (30-09-2026, en este portátil)

| prueba | resultado |
|---|---|
| `pytest tests` | 71 tests pasan (scan paralelo = secuencial en float64, Mamba agrupado = capas separadas, ROI sin camilla, métricas, sliding window, caché rodante, E/S) |
| `check_env` | ok (GPU, versiones, forward/backward con AMP) |
| `runs/overfit2`: 2 casos, 300 iteraciones, sin augmentation | DSC GTVp 0,82, DSC GTVn 0,83: la red aprende |
| `runs/smoke_subset_final`: 21 casos (3 por centro), 14 train / 7 val, 400 iteraciones con augmentation y caché rodante | todo el ciclo funciona; 0,73 s/it, 1,06 GB de VRAM, 0,6 GB de RAM; evaluación 10-22 s por caso; la etiqueta queda al 100 % dentro del ROI |
| `runs/resume_check` | `--resume` continúa en la época correcta y conserva el historial |

Las métricas de la prueba corta (DSC GTVp ~0,4-0,5 en validación, muchos falsos positivos de ganglios)
son las esperables tras ~7 minutos de entrenamiento; solo sirven para comprobar que el código funciona.
