# BiFusion: fusión PET/CT de dos ramas sobre MambaLite3D

Variantes de fusión bimodal (CT + PET) para HECKTOR 2025, construidas sobre `modelos/mamba` **sin modificarlo**.

- **Qué se reutiliza.** Este paquete importa `mambalite3d` en modo de solo lectura. Lectura y ROI al vuelo, caché rodante, muestreo, augmentation, pérdida, entrenamiento, inferencia, métricas, ensambles y herramientas son exactamente los mismos. Solo cambia la red, así que cualquier diferencia de resultados se debe a la fusión.
- **Cómo se engancha.** Al ejecutarse, `bifusion3d.patch.install()` sustituye en memoria la función `build_model` que usan las herramientas de mambalite3d. La nueva construye las variantes de fusión y delega las demás (`base`, `wide`, `unet`, ablaciones) en MambaLite3D, igual que antes.
- **Protección de `modelos/mamba`.** No se escriben `.pyc` y los comandos se niegan a correr desde `modelos/mamba` o a escribir en un run de esa carpeta (`--run ../mamba/...`). Leer sus checkpoints sí está permitido, por ejemplo para ensambles, `efficiency`, `compare_runs` o `report`.
- **Dónde se guarda.** Los resultados van a `runs/` de esta carpeta.

## Variantes

Medido en la RTX 3050 de 4 GB con patch 32×128×128. GMACs según la convención de `tools/efficiency`. La VRAM y los s/it son de `smoke_vram`: un paso completo con batch 2 y AMP.

| Variante | Paso | Qué cambia | Basado en | Parámetros | GMACs | VRAM pico | s/it |
|---|---|---|---|---|---|---|---|
| `base` (referencia) | – | Fusión temprana: CT y PET como 2 canales | MambaLite3D | 4,05 M | 12,5 | 1,29 GB | 0,72 |
| `siam_light` | 1 | Codificador siamés en los niveles 0–1 y compuerta de fusión en sus skips | Kumar et al. 2020; vMambaX 2025 | 4,05 M | 17,9 | 1,85 GB | 0,90 |
| `ctrl_ancho` (config) | control 1 | `base` con 20 canales en los niveles 0–1: mismos GMACs que `siam_light` | – | 4,08 M | 17,5 | 1,46 GB | 0,77 |
| `siam_full` | 2 | Siamés en todos los niveles: compuerta en 0–3 y Cross/Concat Mamba en 4–5 | Sigma (WACV 2025), CIPA (CVPR 2025) | 4,26 M | 19,7 | 1,92 GB | 1,13 |
| `siam_gate` | control 2 | `siam_full` con compuertas en todos los niveles (sin Mamba en la fusión) | – | 4,15 M | 19,6 | 1,89 GB | 0,95 |
| `siam_cga` | 3 | Siamés en todos los niveles: compuerta en 0–2 y CGA3D PET↔CT en 3–5 | H2ASeg (MICCAI 2024), CGA de MambaLiteUNet | 4,61 M | 19,9 | 1,94 GB | 1,52 |
| `base_bifusion` | 4 | `base` con una rama CNN en paralelo dentro de cada MambaStage (BiFusion) | TransFuse (MICCAI 2021) | 6,81 M | 12,7 | 1,31 GB | 0,82 |
| `siam_full_bifusion` | 2 + 4 | `siam_full` con BiFusion | – | 7,02 M | 20,1 | 1,94 GB | 1,47 |

Las tablas completas, con VRAM y tiempo de inferencia, están en `runs/eficiencia_*.json`.

### Paso 1: codificador siamés ligero (`siam_light`)

- **Siamés.** La entrada (B, 2, D, H, W) se pasa como (2B, 1, D, H, W): CT y PET recorren el mismo stem (niveles 0 y 1) con pesos compartidos. Cada InstanceNorm normaliza cada modalidad por su cuenta.
- **Coste.** Se duplican los mapas de alta resolución, que es la capacidad que les falta a los ganglios, y los parámetros solo suben un 0,11 % (las compuertas).
- **`GatedFusion3D`.** Hace la fusión con z = [CT, PET]:
  - compuerta de canal sigmoid(MLP(GAP(z))) y compuerta espacial sigmoid(conv(z)), una por modalidad;
  - modulación residual f·(1 + g), que nunca baja de la entrada;
  - proyección 1×1 + InstanceNorm + LeakyReLU.
- **Después del nivel 1.** La salida fusionada de cada nivel es su skip, y desde el nivel 2 la red es la del base.

### Paso 2: siamés completo con Cross/Concat Mamba (`siam_full`)

- **Siamés en todos los niveles.** Cada nivel produce su skip fusionado y el cuello de botella recibe el del nivel más profundo.
- **Niveles 4–5: `CrossMambaFusion3D`.** Ahí el codificador ya tiene Mamba. Los canales se reparten en 4 grupos con los órdenes de AMF3D (fwd, rev, slice, ap), iguales para CT y PET, de modo que el token t es el mismo vóxel en ambas.
  - **Cross Mamba Block.** Cada modalidad escanea con sus propias A, B y Δ, pero lee su estado con la C de la otra (`CrossGroupedMamba`). Es la extensión de `GroupedMambaTorch` con `c_from`, verificada contra el scan secuencial en float64, también en los gradientes.
  - **Concat Mamba Block.** Escanea [CT; PET] (2L tokens, 128–512) hacia delante y la secuencia invertida, de modo que cada modalidad ve a la otra. Después un residual con escala aprendida y una proyección 2C → C.
- **Papel de Mamba.** En esta variante Mamba es el mecanismo de fusión, con los mismos ~4 M parámetros (+5 %) porque el codificador comparte pesos.
- **Control `siam_gate`.** Mismo codificador siamés con compuertas en todos los niveles. Si `siam_full` no lo supera, la mejora (si la hay) viene del siamés y no de Mamba.

### Paso 3: CGA3D como compuerta PET↔CT (`siam_cga`)

- **Reutiliza la CGA3D de MambaLite3D** (`CGAFusion3D`, mismos parámetros) en los niveles 3–5, pero entre modalidades: x = CT, g = PET.
- **Fusión.** CGA3D ya calcula una compuerta en las dos direcciones, h_CT·σ(h_PET) + h_PET·σ(h_CT), y de ella un mapa ψ de un canal. En vez de devolver x·ψ, aquí devuelve **ψ·CT + (1−ψ)·PET**: en cada vóxel decide cuánto viene de cada modalidad. Un test comprueba que ψ es exactamente el de CGA3D.
- **Comparación con H2ASeg.** Es una interacción bidireccional parecida a la de H2ASeg, pero mucho más barata.

### Paso 4: BiFusion CNN ↔ Mamba (`base_bifusion`)

Cada MambaStage (codificador, cuello de botella y decoder) pasa a ser un `BiFusionStage` con dos ramas en paralelo:

- **Rama global** t = LGFM3D(AMF3D(GSC3D(x))), la misma de MambaStage.
- **Rama local** g = BasicResBlock(x).
- **Fusión como en TransFuse.**
  - SE sobre t y atención espacial (CBAM) sobre g.
  - b = Conv(t·W₁ ⊙ g·W₂), donde ⊙ es el producto elemento a elemento.
  - Bloque residual sobre [b, t', g'].

Para no inflar la red, la convolución de b y la del bloque residual son separables en profundidad. Aun así, la rama CNN añade unos 2,8 M parámetros, en su mayoría los BasicResBlock de los niveles profundos. El cómputo casi no sube, porque esos mapas son de 4×4×4 a 4×8×8. Las ganancias publicadas para BiFusion son modestas (+0,6 a +1,5 de Dice frente a concatenar).

### Paso 5 (no implementado a propósito)

La atención guiada solo por PET (MSAM, Cho et al. 2024) realza las captaciones fisiológicas y puede aumentar los falsos positivos de ganglios, que ya son el error principal del base. Si se quiere como línea base, es una variante corta (una subred de PET que multiplica los skips).

### Paso 6: votación como cota superior

No hace falta código nuevo: `bifusion3d.evaluate --ensemble` promedia las probabilidades de varios modelos (también runs de `modelos/mamba`, que solo se leen) y guarda el resultado en `runs/ens_<nombre>`. Multiplica el coste de inferencia, así que sirve como cota superior, no como modelo "lite".

## Cómo se usa

Desde Anaconda Prompt, con el mismo entorno que `modelos/mamba`:

```bat
conda activate mambalite3d
cd C:\Users\danie\OneDrive\Documentos\Universidad\petct\modelos\bifusion

:: tests (CPU + GPU si hay CUDA)
python -m pytest -q tests

:: entrenamiento (mismas opciones que mambalite3d.training.train; --no-test no evalua el test al final)
python -m bifusion3d.train --config configs/siam_light.yaml --resume --no-test

:: validacion del checkpoint best (final ya se evalua al terminar el entrenamiento)
python -m bifusion3d.evaluate --run runs/siam_light --split val --checkpoint best

:: paso 6: votacion (ensamble) en validacion, sin y con TTA
python -m bifusion3d.evaluate --ensemble ../mamba/runs/full_base_fold0:best runs/siam_light:best --name base_siam_light --split val
python -m bifusion3d.evaluate --ensemble ../mamba/runs/full_base_fold0:best runs/siam_light:best --name base_siam_light --split val --mirror

:: coste: parametros, GMACs, VRAM y tiempos; VRAM de un paso de entrenamiento
python -m bifusion3d.tool efficiency --variants base siam_light siam_full --config configs/siam_full.yaml
python -m bifusion3d.tool smoke_vram --variant siam_full

:: tablas y comparacion pareada (leen los runs de mamba sin escribir en ellos)
python -m bifusion3d.tool report --models base=../mamba/runs/full_base_fold0 siam_light=runs/siam_light --out runs/resultados_fusion
python -m bifusion3d.tool compare_runs --a ../mamba/runs/full_base_fold0:best --b runs/siam_light:best --split val

:: test, una sola vez y solo para la configuracion final elegida en validacion
python -m bifusion3d.evaluate --run runs/siam_light --split test --checkpoint best

:: validacion cruzada (cada fold se entrena con bifusion3d.train)
python -m bifusion3d.cross_validate --config configs/<config con n_folds>.yaml --dry-run
```

`python -m bifusion3d.tool` (sin argumentos) lista todas las herramientas disponibles.

## Cola de trabajos

Hay **una sola cola para todos los experimentos nuevos**: `bifusion3d.train` también construye las variantes de `modelos/mamba` (`base`, `wide`, `unet`, ablaciones), exactamente igual. Por eso los experimentos que estaban pendientes en la cola de mamba (por ejemplo, más contexto) se lanzan desde aquí, con sus resultados en `runs/` de esta carpeta. Así `modelos/mamba` queda congelada como referencia, y su ejecutor no hace falta. Nunca abras los dos ejecutores a la vez, porque comparten la GPU.

**`runner.bat`**
- Ejecuta en orden alfabético los `.bat` de `jobs\pending`. Los logs se acumulan en `jobs\logs`.
- Solo admite una instancia: si ya hay uno abierto, el segundo se niega a arrancar.
- Si se cortó (apagón, reinicio de Windows, ventana cerrada), al abrirlo de nuevo devuelve el trabajo a medias a la cola, y este continúa desde la última época con `--resume`.

**Uso**
- Para lanzar un trabajo, mueve su `.bat` de `jobs\paused` a `jobs\pending`.
- Para pausar uno que aún no empezó, devuélvelo a `jobs\paused`.
- Deja la laptop enchufada y con la tapa abierta: con batería se suspende a las 3 h.

**Linux:** `./runner.sh` hace lo mismo con los `.sh` de `jobs/pending` (cada trabajo activa conda con `jobs/entorno.sh`). En `jobs/paused` están `00_pruebas.sh`, `50_b0_siam_full_v2.sh` y `51_b1_siam_full_suave.sh`; para lanzarlos se mueven a `jobs/pending`. El entorno se instala con `requirements.txt` (instrucciones dentro). Cada ejecutor solo toma sus propios archivos (`.bat` o `.sh`).

### Plan por fases (anterior)

Es el plan del 9 de octubre. El vigente, con las fases A a D del 13 de octubre, está en el documento del plan; de esta tabla solo sigue en uso `13_siam_gate`, como control.

| Fase | Trabajos | Pregunta que responde | Duración |
|---|---|---|---|
| A | `10_siam_light`, `11_ctrl_ancho`, `30_ctx2mm_base` | ¿Ayuda el siamés más allá del cómputo extra? ¿Ayuda más contexto al base? | ~40 h |
| B | `12_siam_full`, `13_siam_gate` | ¿Ayuda Mamba como fusión más allá de una compuerta? Solo si en A el siamés ganó a su control | ~26 h |
| B' | la mejor variante de A/B con el contexto de `ctx2mm` | ¿Se suman fusión y contexto? Solo si `ctx2mm_base` ganó al base | ~20 h |
| C | `14_siam_cga`, `15_base_bifusion` | Alternativas de fusión (pasos 3 y 4) | ~30 h |
| D | 2–3 semillas o folds de la configuración final; test una sola vez; CHUP aislado | Que la diferencia no sea ruido; generalización | según el resultado |

Después de cada fase, el trabajo de reporte deja la tabla en `runs\resultados_*.md` y las comparaciones pareadas en `runs\comparaciones`. Con eso se decide la fase siguiente; no conviene encolar todo de una vez.

| Trabajo | Qué hace | Duración aproximada |
|---|---|---|
| `10_siam_light` | Paso 1 | 11–12 h |
| `11_ctrl_ancho` | Control del paso 1 (mismos GMACs, fusión temprana) | 10 h |
| `12_siam_full` | Paso 2 | 13–14 h |
| `13_siam_gate` | Control del paso 2 | 12 h |
| `14_siam_cga` | Paso 3 (la más lenta: CGA3D sobre 2048 tokens en el nivel 3) | 19–20 h |
| `15_base_bifusion` | Paso 4 | 10–11 h |
| `16_siam_full_bifusion` | Pasos 2 + 4 (solo si ambos ayudan por separado) | 17–18 h |
| `30_ctx2mm_base` | Base con más contexto (2×2×3 mm, 144×192×192 mm); el trabajo 29 de mamba, desde esta cola | ~18 h |

## Protocolo para que la comparación sea defendible

- **Mismo protocolo que `full_base_fold0`**: split 70/20/10 por centro, semilla 12345, 250 épocas, patch 32×128×128 y validación intermedia con 21 casos. Las configs lo replican y un test lo comprueba.
- **Comparar en validación (135 casos).** El test se evalúa una sola vez, al final, para la configuración elegida. Por eso los trabajos entrenan con `--no-test`.
- **Controles:**
  - `ctrl_ancho` separa "mejor fusión" de "más cómputo".
  - `siam_gate` separa "Mamba como fusión" de "siamés".
- **Métricas a mirar además de los DSC:** F1 y falsos positivos por caso de ganglios, y el Dice de los casos sin GTVp. Están en `per_case_*.csv`.
- **Más de una semilla.** Antes de afirmar una diferencia de 0,01–0,02, repite las 2–3 configuraciones finales con otra semilla o con folds de CV. `compare_runs` solo mide la variabilidad entre pacientes, no la del entrenamiento.

## Verificación del preprocesamiento y uso en una laptop

- **`bifusion3d.tools.verify_preproc`** corre `load_case`, el mismo preprocesamiento del entrenamiento, la validación y la inferencia, en los 680 casos. En cada caso comprueba que el ROI sigue la fórmula (recalculada aparte), que contiene el ancla y el cuerpo, la imagen (forma, CT en [0, 1], PET con media 0 y desviación 1), la etiqueta (valores, volumen y componentes), las coordenadas de muestreo, la vuelta a la rejilla original y que dos cargas den lo mismo. Solo CPU; si se corta, se relanza y sigue.
  `python -m bifusion3d.tools.verify_preproc --config configs/siam_full.yaml --out runs/verificacion`
- **`bifusion3d.tools.cache_coverage`** reproduce la caché rodante de un entrenamiento terminado: qué casos de train no entraron nunca y cuántas épocas estuvo cada uno. Comprueba la simulación contra `cache_new_cases` del log.
  `python -m bifusion3d.tools.cache_coverage --runs runs/siam_full ../mamba/runs/full_base_fold0`
- **`bifusion3d/eval_groups.py`** cuenta la detección de ganglios con la regla de grupos del artículo del dataset HECKTOR 2025 (Fig. 5), al lado del emparejamiento húngaro de mambalite3d.
- **`--thermal 83:72`** en `train`, `evaluate` y `tool` pausa si la GPU llega a 83 °C y sigue al bajar a 72 °C; `83:72:120:10` añade 10 min de descanso cada 120 min (`bifusion3d/thermal.py`). Si la laptop se apaga, al abrir el runner de nuevo devuelve a la cola el trabajo cortado, que sigue con `--resume`.

## Arreglos de los avisos de la verificación

`bifusion3d/data` reúne los arreglos de los 17 casos con aviso, sin sacar casos del entrenamiento. Se activan desde el config; sin esas claves todo es igual que en mamba.

- **Ancla relativa** (`data.roi.ancla: relativa`, `preproc.py`): los umbrales del cuerpo (0,5) y del ancla (2) se multiplican por tejido blando / 0,82 de cada paciente. Arregla CHUM-010 (PET fuera de SUV) sin tocar la imagen. Trabajo 31: ningún caso sin ancla ni con etiqueta fuera; el 90 % se mueve ≤ 8 mm.
- **ROI de 300 mm** (`data.roi.down_mm: 300`): con 270 mm, 3 casos de val/test pierden ganglio; con 300 mm, ninguno (+17 % de vóxeles). En inferencia sola, con modelos entrenados a 270 mm, no mejora (trabajo 29): se adopta entrenando con él.
- **Etiquetas por ganglio** (`data.etiquetas: nodos`, `labels.py`): sin fusiones ni particiones de ganglios (10 casos); F1 de ida y vuelta 0,9967 → 0,9992.
- **Etiquetas suaves** (`data.etiquetas: suave`, `soft_train.py`): fracción de tumor por vóxel como objetivo; techo de GTVp en cortes ≤ 2 mm 0,944 → 0,976.
- **Vuelta lineal** (`restore.py`): interpola probabilidades a la rejilla original. Trabajo 29, en val (135): mejora a los tres modelos; siam_full GTVp +0,006, DSCagg +0,005, F1 +0,03 (0,438 → 0,468). Adoptada.

Experimentos: `experimento_preproc` (28), `evaluar_variantes` (29) y `experimento_ancla` (31), resultados en `runs/`.
Entrenamientos preparados en `jobs\paused`: `50_b0_siam_full_v2` (ancla relativa, 300 mm, por ganglio) y `51_b1_siam_full_suave`. Falta el bucle v2 (caché por ventanas, semillas por época, EMA) antes de lanzarlos.

## Archivos

```
bifusion/
├── bifusion3d/
│   ├── __init__.py        rutas a modelos/mamba, sin .pyc
│   ├── fusion.py          GatedFusion3D, CrossGroupedMamba, CrossMambaFusion3D, CGAFusion3D, BiFusionStage
│   ├── network.py         BiFusionNet, FUSION_VARIANTS y build_model
│   ├── patch.py           engancha build_model en las herramientas de mambalite3d
│   ├── guard.py           impide escribir dentro de modelos/mamba
│   ├── train.py           entrenamiento (+ --no-test, --thermal)
│   ├── evaluate.py        evaluacion y ensambles (+ --thermal)
│   ├── cross_validate.py  validacion cruzada con bifusion3d.train
│   ├── tool.py            efficiency, smoke_vram, report, compare_runs, analyze...
│   ├── thermal.py         pausas por temperatura de la GPU
│   ├── eval_groups.py     deteccion de ganglios por grupos (articulo del dataset)
│   ├── data/              preprocesado v2 (ancla relativa, etiquetas por ganglio y suaves) y vuelta lineal
│   ├── soft_train.py      entrenamiento con etiquetas suaves
│   └── tools/             audit_data, verify_preproc, cache_coverage y experimento_preproc (CPU);
│                          evaluar_variantes y viabilidad_aniso (GPU)
├── configs/               una por variante (+ smoke_siam_full, prueba corta con datos reales)
├── tests/                 pasos 1-4 y enganche con las herramientas
├── jobs/                  pending, paused, running, done, logs
└── runner.bat
```
