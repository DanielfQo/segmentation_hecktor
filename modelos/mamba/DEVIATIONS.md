# Desviaciones respecto a los papers y al plan

Lista de decisiones de implementación que difieren de MambaLiteUNet (Rahman et al., arXiv 2604.20286),
SegMamba o del plan de tesis, y por qué. Todas están cubiertas por tests (`tests/`).

## Mamba sin `mamba_ssm` (Windows)

- `mamba_ssm` (kernels CUDA de Mamba) solo compila en Linux. En Windows se usa `MambaTorch`
  (`mambalite3d/models/ssm.py`): mismos nombres de parámetros e inicialización que
  `mamba_ssm.modules.mamba_simple.Mamba`, y el mismo cálculo (conv causal, proyecciones, selective scan).
- El selective scan se calcula en forma cerrada por bloques (`selective_scan_parallel`): dentro de cada
  bloque de `chunk_size` pasos se usan sumas acumuladas de `log(a)` (exponentes siempre <= 0, estable),
  y entre bloques la misma recurrencia de forma recursiva. Es matemáticamente idéntico al scan secuencial
  (test en float64 con error < 1e-9, también en gradientes). Se calcula en float32 aunque el resto use AMP fp16.
- Los mixers de un mismo bloque (4 grupos de AMF, 8 flujos de CGA) se evalúan juntos con
  `GroupedMambaTorch` (resultado idéntico a capas separadas, test incluido). Si el problema es grande, los
  grupos se procesan por tramos (`max_scan_elems`) con checkpointing para no subir la VRAM.
- En Linux con `mamba_ssm` instalado, `MambaMixer` usa los kernels CUDA automáticamente.
  Variables de entorno: `MAMBALITE_FORCE_TORCH=1` fuerza la versión PyTorch; `MAMBALITE_SCAN=chunked`
  usa el scan con bucle (más lento, menos memoria).

## Arquitectura

- Red 3D con strides anisotrópicos derivados del patch y del spacing (heurística de nnU-Net,
  `models/plan.py`) en lugar de los strides 2D fijos del paper. Stem depthwise 7 en el primer downsampling.
- Mamba solo en los 2 niveles más profundos + bottleneck (`n_mamba_levels`), como en el plan: ahí la
  secuencia es corta (L <= 512 tokens con el patch 32x128x128) y el coste es bajo.
- AMF3D: los canales se dividen en 4 grupos y cada grupo recorre el volumen en un orden distinto
  (`fwd`, `rev`, `slice`, `ap`). Es la aportación 3D; la variante `samescan` (todos `fwd`) sirve de ablación.
- LGFM3D: atención global multi-cabeza con `scaled_dot_product_attention` sobre todos los tokens del nivel
  + conv depthwise local, como unidad residual.
- CGA3D en las 3 conexiones skip más profundas (`n_cga_levels`), FUE en las demás. El mapa de atención
  final `psi` tiene un canal (eq. 6-7).
- Deep supervision con pesos 1/2^i normalizados (la salida más gruesa con peso 0, como nnU-Net).

## Datos y ROI

- Nada se escribe en disco: cada caso se lee del NIfTI original, se recorta y remuestrea al vuelo y se
  guarda en RAM (float16). Con muchos casos, la caché en RAM es rodante (`max_cached_cases`,
  `refresh_per_epoch`). La validación final se evalúa en streaming (máximo ~3 casos en RAM).
- ROI (método agnóstico al tumor del plan): se añadió un filtro de **componente principal del cuerpo**
  (`roi.main_body`, apertura morfológica de 2 vóxeles de 4 mm + mayor componente conexa). Sin él, la
  camilla del PET/CT y el reposacabezas entran en la máscara de cuerpo y el ROI ocupaba casi todo el FOV
  (p. ej. 101x425x489 vóxeles frente a ~100x300x450 con el filtro), lo que triplicaba RAM y tiempo de
  inferencia. El ancla superior (SUV > 2) se busca solo dentro del cuerpo. Se sigue comprobando que la
  etiqueta queda al 100 % dentro del ROI (`cobertura` en el log).
- Augmentation en GPU por batch (rotación, escala, espejo, brillo, contraste, gamma, ruido, blur) sobre un
  patch con contexto extra, recortado al patch final después de la transformación espacial.

## Hardware

- El portátil tiene una **RTX 3050 Laptop de 4 GB**, no la A1000 de 8 GB prevista en el plan. Con 4 GB (de los
  que Windows usa ~0,8 GB) se usa patch 32x128x128 y batch 2 (pico medido 1,29 GB, 0,7-0,85 s/iteración).
  Para la A1000 8 GB se recomienda patch 48x160x160, batch 2 (pico medido 2,97 GB, ~1,6 s/it en la 3050).

## Métricas

- `eval/metrics.py` implementa las métricas de HECKTOR 2025 Task 1 tal como se describen (DSC GTVp medio,
  DSC agregado de GTVn en mm3, F1 de detección de ganglios con componentes 26-conexas, emparejamiento
  húngaro e IoU > 0,3). Se calculan tanto en el espacio del ROI como en la malla original de la etiqueta.
  Para resultados publicables conviene contrastarlas con el código oficial del challenge.
