# siam_full: arquitectura y pipeline completo

Segmentación de GTVp y GTVn en PET/CT (HECKTOR 2025, Tarea 1) con MambaLite3D y un codificador siamés con fusión CT/PET en cada nivel.

- **Configuración:** `modelos/bifusion/configs/siam_full.yaml` (`model.variant: siam_full`)
- **Código:** `bifusion3d/network.py` (BiFusionNet), `bifusion3d/fusion.py` (operadores de fusión). Los datos, el entrenamiento, la inferencia y las métricas vienen de `modelos/mamba/mambalite3d/` y no se modifican.
- **Parámetros:** 4.258 M (la base tiene 4.0 M).
- **Costo por parche 32×128×128:** 19.7 GMACs (la base, 12.5).

---

## 1. Idea general

CT y PET no se concatenan como dos canales desde la entrada. Cada modalidad recorre el codificador por separado con los **mismos pesos**. En cada nivel, un módulo de fusión combina sus mapas de características en un único skip. El decoder, el cuello de botella y el resto del entrenamiento son los de MambaLite3D.

```
   CT (1×32×128×128)                 PET (1×32×128×128)
          │                                  │
          └──────────── mismos pesos ────────┘
                          │  reshape: (B,2,…) → (2B,1,…)
                          ▼
  ┌───────────────────────────────────────────────────────┐
  │  Codificador siamés, niveles N0 … N5                  │
  │  los dos flujos (CT y PET) siguen separados           │
  └───────────────────────────────────────────────────────┘
        │  en cada nivel l = 0…5:
        │    skip_l = Φ_l(F_CT,l , F_PET,l)     ← fusión (un solo mapa)
        │  solo en el último nivel, el mapa fusionado sigue hacia el cuello
        ▼
  Cuello de botella: MambaStage 96→128 (entrada: skip_5)
        │
        ▼
  Decoder ÚNICO: sube nivel a nivel y en cada nivel suma h + s
        │   skips 0–2 → FUE (sin parámetros)
        │   skips 3–5 → CGA3D (compuerta guiada por el decoder)
        ▼
  Salida: 3 clases (fondo, GTVp, GTVn)
          + cabezas auxiliares en N0–N4 (solo durante el entrenamiento)
```

Hay **un solo decoder**. No existe una red por modalidad.

---

## 2. Datos

- **Conjunto:** HECKTOR 2025, Tarea 1, 680 estudios PET/CT de cabeza y cuello. La evaluación por centro reconoce siete centros: CHUM, CHUP, CHUS, HGJ, HMR, MDA y USZ.
- **Partición:** estratificada, 70/20/10, semilla 12345 (`split.json`).

| Subconjunto | Casos | Uso |
|---|---|---|
| Entrenamiento | 476 | aprendizaje |
| Validación | 135 | elegir checkpoint, comparar variantes, ajustar post-procesado |
| Prueba | 69 | evaluación final, una sola vez, con la configuración ya fijada |

---

## 3. Preprocesamiento (al vuelo, sin copias intermedias en disco)

Cada caso se lee de los NIfTI originales, se procesa y se guarda en RAM en float16.

### 3.1 Reorientación

Cada volumen se reorienta a la convención **LPS** (x hacia la izquierda del paciente, y hacia la espalda, z hacia la cabeza). Así todos los casos comparten el sentido de los ejes.

### 3.2 ROI de cabeza y cuello (sin usar la etiqueta)

La ROI se calcula solo con CT y PET. Así el mismo procedimiento sirve en inferencia.

1. **Malla gruesa de 4 mm (solo para la ROI).** CT (relleno −1024 HU) y PET (relleno 0) se remuestrean trilinealmente a 4 × 4 × 4 mm.
2. **Máscara del cuerpo:** `CT > −500 HU` o `SUV > 0.5`.
3. **Componente principal:** apertura morfológica de 2 iteraciones (2 × 4 mm) y se conserva la componente conexa más grande. Así se quitan la camilla y otros objetos que no son el paciente.
4. **Límite superior:** el corte más craneal del cuerpo con `SUV > 2` (captación cerebral).
5. **Límite inferior:** 270 mm por debajo del superior.
6. **Ajuste en el plano:** x e y se ajustan al contorno del cuerpo dentro de ese rango de cortes.
7. **Margen:** 8 vóxeles del paso final por eje (8 mm en x e y, 24 mm en z), más media celda gruesa (2 mm).
8. **Tamaño mínimo:** 96 × 96 × 32 vóxeles del paso final (96 × 96 × 96 mm). La ROI se recorta al campo del CT.

Las copias de 4 mm se descartan después y no entran al modelo. La verificación de los 680 casos (`runs/verificacion`) no encontró fallos. Con 270 mm, 3 casos de MDA pierden parte del GTVn por abajo (MDA-264, MDA-059 y MDA-150); con 300 mm, ninguno.

### 3.3 Remuestreo a la malla final: 1 × 1 × 3 mm (anisotrópica)

La malla final se obtiene en **un solo remuestreo desde el original**. La malla de 4 mm solo sirvió para calcular la ROI.

- **Origen:** esquina de la ROI.
- **Espaciado:** (x, y, z) = (1, 1, 3) mm. El eje z es 3 veces más grueso que el plano.
- **Dirección:** identidad (LPS).
- **Tamaño:** `floor((hi − lo) / espaciado) + 1` vóxeles.

| Volumen | Interpolación | Relleno fuera del campo | Motivo |
|---|---|---|---|
| CT | trilineal | −1024 HU (aire) | valores continuos |
| PET | trilineal | 0 SUV | valores continuos |
| Máscara | vecino más cercano | 0 (fondo) | no se mezclan etiquetas en el borde |

Ejemplo, CHUM-021: CT de 0.98 × 0.98 × 3.27 mm y PET de 3.52 × 3.52 × 3.27 mm. En el plano, el PET se interpola unas 3.5 veces: suaviza, pero no añade detalle.

### 3.4 Normalización

- **CT:** se recorta a [−200, 400] HU y se escala a [0, 1].
- **PET:** z-score dentro de la ROI del paciente, `(SUV − μ_ROI) / σ_ROI`.

---

## 4. Muestreo de parches y aumento de datos

### 4.1 Muestreo

- **Parche:** 32 × 128 × 128 vóxeles (z × y × x), equivalente a 96 × 128 × 128 mm.
- **Lote:** 2 parches.
- **Muestreo con foreground:** `fg_fraction = 0.33`. Con lote 2, `round(2 × 0.33) = 1`: **uno de cada dos parches** se centra en un vóxel tumoral. La clase se elige al azar entre GTVp y GTVn, así que los ganglios se muestrean tanto como el tumor primario. El otro parche cae en una posición aleatoria dentro de la ROI.
- **Caché:** 60 casos en RAM (`max_cached_cases`). Cada época se renuevan 10 en segundo plano (`refresh_per_epoch`). Pero cada reanudación tras un corte vuelve a empezar la caché: en siam_full (5 sesiones) 31 de los 476 casos de entrenamiento nunca entraron (`runs/verificacion/cache_entrenamientos.json`).

### 4.2 Aumento de datos (en GPU, por lote)

Se extrae un parche de contexto 25 % mayor por eje, se aplican las transformaciones geométricas con un solo remuestreo y luego se recorta al tamaño final.

| Transformación | Probabilidad | Parámetros |
|---|---|---|
| Rotación | 0.2 | ±30° en el plano axial. Fuera del plano no, por la anisotropía (máx/mín del espaciado > 2) |
| Escalado | 0.2 | factor 0.7–1.4 |
| Espejo | 0.5 por eje | ejes z, y, x |
| Brillo | 0.15 | ×0.75–1.25 |
| Contraste | 0.15 | ×0.75–1.25 |
| Gamma | 0.3 | γ ∈ [0.7, 1.5] |
| Ruido gaussiano | 0.1 | σ ∈ [0, 0.1] |
| Desenfoque gaussiano | 0.2 | σ ∈ [0.5, 1] |

Las transformaciones de intensidad se aplican **canal por canal**, es decir, CT y PET por separado.

---

## 5. Arquitectura

### 5.1 Codificador siamés

La entrada `(B, 2, 32, 128, 128)` se reorganiza como `(2B, 1, 32, 128, 128)`. Cada paciente aporta dos muestras, CT y PET, que pasan por los mismos filtros. La primera capa tiene 1 canal de entrada, no 2.

Las capas de normalización son InstanceNorm3d, que normaliza cada muestra por separado. Por eso CT y PET no comparten estadísticas dentro del lote.

En cada nivel `l`, los mapas `F_CT,l` y `F_PET,l` se juntan en un par `(B, 2, C, …)` y el operador `Φ_l` produce un skip `S_l`. Los dos flujos **no** se fusionan dentro del codificador: siguen separados hasta el último nivel, y solo entonces el mapa fusionado pasa al cuello de botella.

### 5.2 Niveles del codificador

| Nivel | Tamaño (z × y × x) | Espaciado (z / y / x, mm) | Canales | Submuestreo hasta el nivel | Bloque |
|---|---|---|---|---|---|
| N0 | 32 × 128 × 128 | 3 / 1 / 1 | 16 | stem: ConvIN 1→16, kernel 1×3×3 | ConvIN + ResBlock 16 |
| N1 | 32 × 64 × 64 | 3 / 2 / 2 | 16 | depthwise 3×7×7, paso (1,2,2), luego 1×1×1 | ResBlock 16 (kernel 3³) |
| N2 | 16 × 32 × 32 | 6 / 4 / 4 | 32 | MaxPool (2,2,2) | ConvIN 16→32 + ResBlock 32 |
| N3 | 8 × 16 × 16 | 12 / 8 / 8 | 48 | MaxPool (2,2,2) | ConvIN 32→48 + ResBlock 48 |
| N4 | 4 × 8 × 8 | 24 / 16 / 16 | 64 | MaxPool (2,2,2) | **MambaStage** 48→64 |
| N5 | 4 × 4 × 4 | 24 / 32 / 32 | 96 | MaxPool (1,2,2) | **MambaStage** 64→96 |
| Cuello | 4 × 4 × 4 | 24 / 32 / 32 | 128 | — | **MambaStage** 96→128 |

- La configuración usa Mamba solo en los dos niveles más profundos (`n_mamba_levels = 2`). En N4 hay 256 vóxeles (tokens) y en N5 y el cuello hay 64. Tras la fusión CT/PET, las secuencias de Cross/Concat Mamba tienen 512 y 128 tokens.
- La relación de resolución z:plano es **3** en la entrada, **1.5** de N1 a N4 y **0.75** en N5 y el cuello (24 mm en z frente a 32 mm en el plano). La anisotropía no se mantiene hasta el final: se invierte en N5.

### 5.3 Fusión, filtro del skip y decoder por nivel

| Nivel | Operador de fusión `Φ_l` | Filtro del skip en el decoder | Bloque del decoder | Cabeza auxiliar |
|---|---|---|---|---|
| N0 | Compuerta, kernel 1×3×3 | FUE | ResBlock | sí |
| N1 | Compuerta, kernel 3×3×3 | FUE | ResBlock | sí |
| N2 | Compuerta, kernel 3×3×3 | FUE | ResBlock | sí |
| N3 | Compuerta, kernel 3×3×3 | CGA3D | ResBlock | sí |
| N4 | Cross/Concat Mamba | CGA3D | MambaStage | sí |
| N5 | Cross/Concat Mamba | CGA3D | MambaStage | no |

En N3–N5 el skip pasa por **dos** operadores: primero la fusión CT/PET del codificador, y después el CGA3D del decoder, que lo filtra con el mapa del decoder.

### 5.4 Fusión por compuerta (`GatedFusion3D`, N0–N3)

Sea `Z = [F_CT ; F_PET]` con 2C canales.

- **Compuerta de canal:** `g_c = σ(W₂ · LeakyReLU(W₁ · GAP(Z)))`, con `W₁: 2C → max(2C/4, 4)` y `W₂` de vuelta a 2C.
- **Compuerta espacial:** `g_s = σ(Conv_k(Z))`, con 2 mapas (uno por modalidad) y el kernel del nivel.
- **Compuerta por modalidad:** `G_m = g_c^m ⊙ g_s^m ∈ (0, 1)`.
- **Modulación residual:** `S = φ(W_p [F_CT ⊙ (1 + G_CT) ; F_PET ⊙ (1 + G_PET)])`
  - `W_p` es una convolución 1×1×1 de 2C a C canales.
  - `φ` = InstanceNorm + LeakyReLU(0.01).

Como `1 + G ∈ (1, 2)`, la compuerta solo puede **amplificar** cada canal entre 1 y 2 veces antes de la proyección. No puede apagar una modalidad por completo.

### 5.5 Fusión con Mamba cruzado (`CrossMambaFusion3D`, N4–N5)

Adaptación 3D del Cross Mamba Block y el Concat Mamba Block de Sigma (Wan et al.).

1. **Orden de escaneo.** Los canales se dividen en 4 grupos. Cada grupo se aplana con uno de los órdenes de AMF3D (`fwd`, `rev`, `slice`, `ap`). CT y PET usan el mismo orden, así que el token `t` de ambas secuencias corresponde al mismo vóxel.
2. **Cross Mamba Block (CroMB).** Cada modalidad escanea su secuencia con sus propias `A`, `B` y `Δ`, pero lee el estado con la matriz `C` de la otra modalidad (intercambio de `C`). La salida se suma a la entrada (residual).
3. **Concat Mamba Block (ConMB).** En cada grupo se concatena `[CT ; PET]` (2L tokens) y se escanea en dos sentidos con **parámetros distintos**: hacia delante (el PET ve el estado del CT) y hacia atrás (el CT ve el del PET). Ambos sentidos se suman, se separan por modalidad y se añaden con un factor escalar aprendido por modalidad, inicializado en 1 (residual).
4. **Salida:** `S = W_p [H_CT + α_CT · O_CT ; H_PET + α_PET · O_PET]`, con `W_p` una 1×1×1 de 2C a C canales.

### 5.6 Bloques internos de MambaLite3D

**MambaStage** (N4, N5 y cuello): proyección 1×1×1 + norm → GSC3D → AMF3D → LGFM3D → ChannelLayerNorm → proyección 1×1×1.

- **GSC3D** (SegMamba): `GSC(z) = z + C₃(C₃(z) ⊙ C₁(z))`.
- **AMF3D** (4 grupos de canales):
  1. `x` se normaliza y se reparte en 4 grupos.
  2. Un `MultiDirectionalMixer` aplica un mezclador Mamba por grupo, cada uno con su orden de escaneo. Luego `z = concat(z_i + α · x_i)`.
  3. Compuerta espacial `S = σ(PW(DW(z)))`, `zs = S ⊙ z`.
  4. `T = PW(DW(zs)) + zs`, y la salida es `T + x`.
- **LGFM3D**: convolución depthwise local + atención multi-cabeza global (8 cabezas) sobre todos los vóxeles del nivel, en forma residual.
- **Mamba** (selective scan): `h_t = Ā_t h_{t−1} + B̄_t x_t`, `y_t = C_t h_t + D x_t`, con `Ā = exp(ΔA)` y `B̄ = ΔB`. `Δ`, `B` y `C` dependen de la entrada. Parámetros: estado 16, convolución 4, expansión 2, bloques de 8. Sin `mamba_ssm` (que no compila en Windows), se usa una implementación en PyTorch puro con escaneo por bloques.

**CGA3D** (skips 3–5 del decoder): entradas `x` (skip) y `g` (mapa del decoder).

1. `x` y `g` se normalizan y se reparten en 4 grupos cada uno.
2. Un `MultiDirectionalMixer` de 8 flujos (4 de x y 4 de g, cada uno con su orden) produce `h_x` y `h_g`.
3. Por grupo: `cross = h_x · σ(DW(h_g)) + h_g · σ(DW(h_x))`, seguido de una rama conv (depthwise 3×3×3 + 1×1).
4. Se concatenan los grupos y una conv 3×3×3 + 1×1 produce `ψ = σ(·) ∈ (0, 1)` de un canal.
5. Salida: `x · ψ`.

**FUE** (skips 0–2 del decoder, sin parámetros): `z̄ = σ(mean_c(z))`, `u = −z̄ · log(z̄ + 10⁻⁶)`, salida `z + z · (1 − u)`.

**Decoder** (un solo decoder): para cada nivel `l` de arriba abajo:
- `h` = convolución transpuesta desde el nivel inferior (en el nivel más profundo, una 1×1×1 desde el cuello).
- `s` = filtro del skip (FUE o CGA3D).
- `d = block(h + s)`, con `block` = ResBlock en N0–N3 y MambaStage en N4–N5.
- Cabeza 1×1×1 a 3 clases en N0–N4.

**Inicialización:** Kaiming para convoluciones (con pendiente 10⁻²); normal truncada con desviación 0.02 para lineales. Los bloques Mamba conservan su inicialización propia.

---

## 6. Entrenamiento

| Parámetro | Valor |
|---|---|
| Optimizador | SGD con momento de Nesterov 0.99 |
| Tasa inicial | 1 × 10⁻² |
| Decaimiento | polinomial por época: `η_e = η₀ · (1 − e/E)^0.9` |
| Weight decay | 3 × 10⁻⁵ |
| Épocas | 250 × 250 iteraciones = 62 500 pasos |
| Lote | 2 parches |
| Precisión | AMP (fp16); el escaneo selectivo en fp32 |
| Recorte de gradiente | norma 12 |
| Semilla | 12345 |

### 6.1 Función de pérdida

Cada salida del decoder usa Dice + entropía cruzada:

- **CE:** entropía cruzada sobre las 3 clases.
- **Dice:** calculado sobre todo el lote, solo para GTVp y GTVn (el fondo no entra). Con `ε = 10⁻⁵`.

Hay cinco salidas (N0–N4), con pesos `∝ 2^(−l)` normalizados, y el de la última (N4) fijado en 0:

| Salida | N0 | N1 | N2 | N3 | N4 |
|---|---|---|---|---|---|
| Peso | 0.533 | 0.267 | 0.133 | 0.067 | 0 |

En la práctica, la pérdida de supervisión profunda la reciben N0–N3, y N0 domina.

### 6.2 Validación y checkpoints

- Cada 25 épocas se evalúan 21 casos de validación.
- **Score de selección:** `(DSC GTVp + DSCagg GTVn) / 2`. Se guarda `checkpoint_best.pth` cuando mejora.
- `checkpoint_latest.pth` se guarda cada época, y `--resume` retoma desde ahí.
- Al terminar se evalúan los 135 casos de validación con los pesos finales (`checkpoint_final`).

---

## 7. Inferencia

1. **Ventana deslizante** sobre la ROI, con parches 32 × 128 × 128 y solapamiento del 50 % (`step_fraction = 0.5`).
2. **Peso gaussiano** por ventana: σ = tamaño del parche / 8 en cada eje, normalizado a 1 en el centro, con mínimo 10⁻⁴. Los bordes pesan menos.
3. **Promedio** de probabilidades (softmax) sobre todas las ventanas y `argmax` por vóxel. En inferencia el decoder devuelve solo la salida N0.
4. **TTA de espejo** (opción `--mirror`), desactivada por defecto. Mejora la precisión a cambio de más tiempo.
5. **Regreso a la malla original:** la predicción de 1 × 1 × 3 mm se remuestrea con vecino más cercano a la malla de la etiqueta original.
6. **Métricas** calculadas en dos espacios: el de la ROI y el original. Las cifras de este documento son del **espacio original**. La diferencia entre ambos es pequeña (por ejemplo, F1 de GTVn 0.3598 frente a 0.3560).

---

## 8. Post-procesado

Se aplica sobre componentes conexas de 26 vecinos, con reglas que se eligen en validación por búsqueda en malla.

**Para siam_full todavía no hay reglas.** Las cifras de validación y de prueba de este documento están sin post-procesado.

Como referencia, las reglas del run base (`full_base_fold0`), ajustadas solo en validación, son:

- **GTVp:** conservar solo la región más grande.
- **GTVn:** borrar regiones de menos de 0.5 ml o con probabilidad máxima menor que 0.995.

Según el README de MambaLite3D, en el test (69 casos, checkpoint best) estas reglas pasaron el DSC de GTVp de 0.701 a 0.715, el DSC agregado de GTVn de 0.673 a 0.714 y el F1 de ganglios de 0.34 a 0.70. Son cifras del README, no las he recalculado.

---

## 9. Evaluación (métricas HECKTOR 2025)

- **DSC GTVp:** Dice por paciente, promediado. Vale 1 si predicción y referencia están vacías, y 0 si solo una lo está.
- **DSCagg GTVn:** `2 Σ|P∩G| / Σ(|P| + |G|)` sumando todos los pacientes, en mm³.
- **F1agg GTVn:** `2TP / (2TP + FP + FN)`. Las componentes predichas y de referencia se emparejan con el algoritmo húngaro; cuenta como detectada si IoU > 0.3. TP, FP y FN se suman sobre todos los pacientes antes de calcular la F1.
- **Ranking oficial:** conteo de Borda de un puesto en GTVp y uno en GTVn, con Borda de DSCagg y F1agg para GTVn.
- **HECKTOR 2022** (referencia): media de las DSCagg de GTVp y GTVn.
- **Comparaciones entre modelos:** pruebas pareadas por paciente (bootstrap, permutación y Wilcoxon).

---

## 10. Estado y resultados

**Estado:** entrenamiento terminado (250 épocas, `done.json`). Validación final hecha. **Prueba de siam_full no evaluada** (no existe `metrics_test`).

### 10.1 Validación (135 casos, sin post-procesado, espacio original)

| Modelo | Checkpoint | GTVp DSC | GTVn DSCagg | GTVn F1agg | Puesto Borda |
|---|---|---|---|---|---|
| **siam_full** | **final** | **0.655** | **0.735** | **0.438** | **1** |
| siam_full | best (época 225) | 0.653 | 0.730 | 0.360 | — |
| base (`full_base_fold0`) | best | 0.657 | 0.671 | 0.296 | 2 |
| siam_light | best | 0.655 | 0.680 | 0.325 | 3 |

El checkpoint **final** es el elegido: es el mejor en validación según la métrica 2025. Los valores de base y siam_light vienen de `runs/resultados_siam_full.md`.

Según el mismo archivo, en HECKTOR 2022 (validación) siam_full obtiene una media de DSCagg de 0.738, frente a 0.701 de la base.

### 10.2 Referencia de prueba (69 casos)

| Modelo | GTVp DSC | GTVn DSCagg | GTVn F1agg |
|---|---|---|---|
| base (`full_base_fold0`, best) | 0.701 | 0.671 | 0.349 |
| siam_full (final) | pendiente | pendiente | pendiente |

Notas:

- Durante el entrenamiento, el GTVn DSCagg en la validación intermedia (21 casos) fue de 0.620 en la época 200 y de 0.742 en la 225. La elección del checkpoint influye mucho.
- Hay **una sola semilla**. Diferencias de 0.01–0.02 en validación no son concluyentes por sí solas.
- Para el test conviene aplicar la misma configuración a todos los modelos comparados: sin post-procesado en siam_full y, si se compara con base, también sin post-procesado o con sus reglas en ambos.

---

## 11. Costo computacional

| Medida | Valor | Fuente |
|---|---|---|
| Parámetros | 4.258 M (base: 4.0 M) | README de bifusion |
| GMACs por parche (32×128×128, un paso hacia delante) | 19.7 (base: 12.5) | README |
| VRAM reservada pico, lote 2, AMP | 1.92 GB | README; el log marca 2.22 GB en la época 0 y 1.90 GB después |
| Tiempo por iteración | 0.77–0.85 s en las primeras épocas | `log.txt` |
| Tiempo por época | ≈ 200 s (≈ 3.4 min) | `log.txt` |
| Duración del entrenamiento | ≈ 14 h para 250 épocas | `log.txt` |

GPU: NVIDIA RTX 3050 Laptop, 4 GB.

---

## 12. Reproducción

Desde Anaconda Prompt, con el entorno de MambaLite3D (según el README de bifusion):

```bat
conda activate mambalite3d
cd C:\Users\danie\OneDrive\Documentos\Universidad\petct\modelos\bifusion

:: entrenamiento (con --resume retoma desde el último checkpoint; --no-test evita evaluar el test)
python -m bifusion3d.train --config configs/siam_full.yaml --resume --no-test

:: validación
python -m bifusion3d.evaluate --run runs/siam_full --split val --checkpoint final

:: prueba: UNA sola vez, con la configuración ya fijada
python -m bifusion3d.evaluate --run runs/siam_full --split test --checkpoint final

:: lista de herramientas disponibles (incluye el ajuste del post-procesado)
python -m bifusion3d.tool
```

Archivos de la ejecución en `runs/siam_full/`: `log.txt`, `history.json`, `split.json`, `config_used.json`, `done.json`, `best.json`, `checkpoint_best.pth`, `checkpoint_final.pth`, `checkpoint_latest.pth`, `metrics_ep*.json` (cada 25 épocas), `metrics_val_best.json`, `metrics_val_final.json`, `per_case_val_best.csv`, `per_case_val_final.csv`, `qc_val_*_CHUM-011.png` y `qc_val_*_CHUM-021.png`.

---

## 13. Limitaciones

- **Espaciado z de 3 mm** frente a 1 mm en el plano. Los GTVn pequeños ocupan pocos cortes, y la rotación fuera del plano está desactivada.
- **PET con interpolación:** el PET original tiene unos 3.5 mm en el plano y se lleva a 1 mm sin añadir detalle. Un ganglio de 1 cm ocupa unos 3 vóxeles de PET en el plano.
- **Una sola semilla** y un checkpoint elegido en validación.
- **Caché desigual:** 31 casos de entrenamiento nunca se vieron, y el tiempo en caché varía mucho entre casos (p50 26 épocas, máximo 134).
- **Mamba sin `mamba_ssm`:** se usa la implementación en PyTorch puro. Es la que permite correr en Windows, pero no es el kernel CUDA de referencia, así que su velocidad no es comparable con la de ese kernel.
- **Post-procesado no ajustado** para siam_full.
