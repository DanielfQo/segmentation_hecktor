# Lo que falta correr

Actualizado el 15 de octubre de 2026, 16:00 (Lima).

## Cómo retomar
- Abre `runner.bat`. Si se cortó, el trabajo vuelve a la cola y sigue donde quedó.
- No abras a la vez el runner de `modelos\mamba`.

## Hecho
- 28 experimento_preproc, 29 evaluar_variantes, 31 experimento_ancla (ver README, "Arreglos de los avisos").
- Adoptado: ancla relativa, ROI de 300 mm (entrenando), etiquetas por ganglio, vuelta lineal. Etiquetas suaves: en B1.

## Por escribir (Claude)
1. Bucle v2: caché por ventanas, semillas por época, EMA. Prueba corta (`42_humo_pre`).
2. Herramientas de la Fase A: TTA (45), componentes de FP (46), VRAM de parches mayores (48).
3. Split fijo de validación cruzada (`cv_fijo`) para la Fase C.

## Entrenamientos (GPU, uno a la vez, ~16 h cada uno)
| Run | Cambio |
|---|---|
| B0 `50_b0_siam_full_v2` | Bucle v2 + ancla relativa + ROI 300 mm + etiquetas por ganglio |
| B1 `51_b1_siam_full_suave` | B0 + etiquetas suaves |
| B2 | Muestreo por componente |
| B3 | PET en log(1+SUV) (CHUM-010 reescalado) |
| B4 | ROI cervical ±120 mm |
| Control | siam_gate con la receta final |

Opcionales: aumentos del PET, pérdida por componente, muestreo por centro, parche mayor, sin CHUP.

## Después
- C: 5 folds sobre 611 casos (~90 h) y ensamble en el test de 69.
- D: postprocesado ajustado en la validación cruzada; test sin y con él.
- Estimado: experimentos completos ~26–28 de octubre con la laptop prendida.

## En pausa del plan anterior
`11_ctrl_ancho`, `13_siam_gate` (se rehará con la receta final), `14_siam_cga`, `15_base_bifusion`, `16_siam_full_bifusion`, `30_ctx2mm_base`.
