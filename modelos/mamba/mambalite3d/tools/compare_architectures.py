"""Comparación integral de arquitecturas en MambaLite3D (HECKTOR 2025 / 2022).

Compara las arquitecturas en cuatro dimensiones clave:
1. Estructural: Parámetros totales/por componente, canales, niveles Mamba, skips CGA, mixer.
2. Eficiencia computacional: GMACs, GFLOPs, consumo de VRAM y tiempos de entrenamiento/inferencia.
3. Rendimiento de segmentación: Métricas HECKTOR 2025 (GTVp, GTVn DSCagg, GTVn F1agg) y 2022 (Media DSCagg).
4. Trade-offs: Rendimiento por parámetro y latencia (eficiencia relativa vs U-Net baseline).

Uso:
    python -m mambalite3d.tools.compare_architectures
    python -m mambalite3d.tools.compare_architectures --variants base wide unet --plot
    python -m mambalite3d.tools.compare_architectures --calc-ops --save-md runs/comparacion.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import resolve
from ..eval.hecktor import fmt, fmt_rank, hecktor2022, hecktor2025, ranking_2022, ranking_2025, selection_score
from ..models.config import VARIANTS, MambaLiteConfig, make_config
from ..models.network import build_model
from ..models.plan import plan_network
from .efficiency import count_ops


# Mapeo descriptivo de variantes
VARIANT_DESCRIPTIONS = {
    "base": "MambaLite3D estándar (niveles profundos con Mamba, CGA en skips)",
    "wide": "MambaLite3D ancho (canales ampliados [24..192], bottleneck 256)",
    "wide2": "MambaLite3D extra-ancho (canales [32..256], bottleneck 320)",
    "unet": "U-Net residual convolucional pura (sin Mamba, sin CGA/LGFM)",
    "nomamba": "Ablación: Mamba reemplazado por TokenMLP",
    "samescan": "Ablación: Escaneo unidireccional idéntico en todos los grupos",
    "noamf": "Ablación: Sin fusión multimodal AMF3D",
    "nolgfm": "Ablación: Sin módulo LGFM3D",
    "nocga": "Ablación: Sin Cross-Gate Attention (CGA3D) en los skips",
    "g1": "Ablación: 1 solo grupo de canales para el escaneo Mamba",
    "g2": "Ablación: 2 grupos de canales para el escaneo Mamba",
    "m3": "Ablación: 3 niveles con Mamba en vez de 2",
    "dual": "MambaLite3D con un codificador por modalidad (CT y PET), fusión por nivel en los skips",
    "dualwide": "Doble codificador con canales anchos ([24..192], bottleneck 256)",
    "dualmid": "Fusión intermedia: CT y PET por separado en los niveles convolucionales, niveles con Mamba compartidos",
    "m3lite": "Mamba también en el nivel 3; atención global solo en los 2 niveles más profundos",
    "m4lite": "Mamba también en los niveles 2 y 3; atención global solo en los 2 niveles más profundos",
}

DEFAULT_RUNS = {
    "base": "runs/full_base_fold0",
    "wide": "runs/exp_wide",
    "unet": "runs/full_unet",
}


def inspect_architecture(
    variant: str,
    patch_zyx: Sequence[int] = (32, 128, 128),
    spacing_zyx: Sequence[float] = (3.0, 1.0, 1.0),
    device: str = "cpu",
    calc_ops: bool = False,
) -> Dict[str, Any]:
    """Instancia y analiza la configuración y parámetros de una variante arquitectónica."""
    strides, kernels = plan_network(patch_zyx, spacing_zyx)
    cfg = make_config(variant)
    model = build_model(variant, strides, kernels)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Desglose por componentes
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    btn_params = sum(p.numel() for p in model.bottleneck.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())

    # Parámetros específicos del decoder: skips vs bloques vs cabezas
    skip_params = sum(sum(p.numel() for p in m.parameters()) for m in model.decoder.skip)
    block_params = sum(sum(p.numel() for p in m.parameters()) for m in model.decoder.block)
    heads_params = sum(sum(p.numel() for p in m.parameters()) for m in model.decoder.heads)

    info: Dict[str, Any] = {
        "variant": variant,
        "description": VARIANT_DESCRIPTIONS.get(variant, "Variante personalizada"),
        "baseline": cfg.baseline,
        "total_params": total_params,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
        "encoder_params_M": enc_params / 1e6,
        "bottleneck_params_M": btn_params / 1e6,
        "decoder_params_M": dec_params / 1e6,
        "skip_params_M": skip_params / 1e6,
        "block_params_M": block_params / 1e6,
        "heads_params_M": heads_params / 1e6,
        "features": list(model.feats),
        "bottleneck_features": cfg.bottleneck_features,
        "n_stages": len(model.feats),
        "mamba_levels": list(model.mamba_levels),
        "cga_levels": list(model.cga_levels),
        "mixer": cfg.mixer,
        "groups": cfg.groups,
        "orders": list(cfg.orders),
        "chunk_size": cfg.chunk_size,
        "use_amf": cfg.use_amf,
        "use_lgfm": cfg.use_lgfm,
        "use_cga": cfg.use_cga,
    }

    if calc_ops:
        try:
            ops = count_ops(model, patch_zyx, device)
            info.update(ops)
        except Exception as e:
            info["ops_error"] = str(e)

    return info


def load_efficiency_file(eficiencia_path: Path) -> Dict[str, Dict[str, Any]]:
    """Carga los datos empíricos de eficiencia guardados si existen."""
    if not eficiencia_path.exists():
        return {}
    try:
        data = json.loads(eficiencia_path.read_text(encoding="utf-8"))
        rows = data.get("filas", [])
        out = {}
        for r in rows:
            var = r.get("variante")
            if var:
                out[var] = r
        return out
    except Exception:
        return {}


def load_results_file(resultados_path: Path) -> Dict[str, Any]:
    """Carga runs/resultados.json si existe."""
    if not resultados_path.exists():
        return {}
    try:
        return json.loads(resultados_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_run_metrics(run_dir: Path, split: str = "test", checkpoint: str = "best") -> Optional[Dict[str, Any]]:
    """Carga metrics_<split>_<checkpoint>.json directamente del run."""
    p = run_dir / f"metrics_{split}_{checkpoint}.json"
    if not p.exists():
        # Fallback a final
        p = run_dir / f"metrics_{split}_final.json"
        if not p.exists():
            return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("original_space") or d.get("roi_space")
    except Exception:
        return None


def format_tables_and_summary(
    arch_infos: Dict[str, Dict[str, Any]],
    efficiency_data: Dict[str, Dict[str, Any]],
    results_data: Dict[str, Any],
    split_choice: str = "both",
) -> Tuple[str, str]:
    """Construye las tablas formateadas para terminal y para Markdown."""
    term_lines: List[str] = []
    md_lines: List[str] = []

    def p(text: str = ""):
        term_lines.append(text)

    def m(text: str = ""):
        md_lines.append(text)

    p("=" * 100)
    p("  COMPARATIVA INTEGRAL DE ARQUITECTURAS - MAMBALITE3D EN HECKTOR")
    p("=" * 100)
    p()

    m("# Comparativa Integral de Arquitecturas - MambaLite3D en HECKTOR")
    m()

    # -------------------------------------------------------------
    # TABLA 1: Configuracion Estructural y Parametros
    # -------------------------------------------------------------
    p("1. CONFIGURACION ESTRUCTURAL Y PARAMETROS")
    p("-" * 100)
    col_w = [10, 11, 24, 14, 10, 10, 14]
    header = (
        f"{'Variante':<{col_w[0]}} | {'Parametros':<{col_w[1]}} | {'Canales (Enc / Btn)':<{col_w[2]}} | "
        f"{'Niveles Mamba':<{col_w[3]}} | {'Skips CGA':<{col_w[4]}} | {'Mixer':<{col_w[5]}} | {'Tipo':<{col_w[6]}}"
    )
    p(header)
    p("-" * 100)

    m("## 1. Configuración Estructural y Parámetros")
    m()
    m("| Variante | Parámetros (M) | Encoder / Btn / Decoder (M) | Canales [Stages] / Btn | Niveles Mamba | Skips CGA | Mixer |")
    m("|---|---|---|---|---|---|---|")

    for v, info in arch_infos.items():
        pm = info["total_params_M"]
        feats_str = f"{info['features']} / {info['bottleneck_features']}"
        mamba_str = str(info["mamba_levels"]) if info["mamba_levels"] else "Ninguno"
        cga_str = str(info["cga_levels"]) if info["cga_levels"] else "Ninguno"
        mixer_str = info["mixer"]
        tipo = "U-Net CNN" if info["baseline"] else "MambaLite3D"

        enc_btn_dec = f"{info['encoder_params_M']:.2f} / {info['bottleneck_params_M']:.2f} / {info['decoder_params_M']:.2f}"

        row_str = (
            f"{v:<{col_w[0]}} | {f'{pm:.2f} M':<{col_w[1]}} | {feats_str:<{col_w[2]}} | "
            f"{mamba_str:<{col_w[3]}} | {cga_str:<{col_w[4]}} | {mixer_str:<{col_w[5]}} | {tipo:<{col_w[6]}}"
        )
        p(row_str)

        m(f"| **{v}** | {pm:.2f} M | {enc_btn_dec} | `{info['features']}` / {info['bottleneck_features']} | {mamba_str} | {cga_str} | {mixer_str} |")

    p()
    m()

    # -------------------------------------------------------------
    # TABLA 2: Eficiencia Computacional y Consumo Hardware
    # -------------------------------------------------------------
    p("2. EFICIENCIA COMPUTACIONAL Y RECURSOS (Patch 32x128x128, Inferencia ROI 104x288x448)")
    p("-" * 100)
    col_w2 = [10, 10, 10, 14, 12, 14, 11, 11]
    header2 = (
        f"{'Variante':<{col_w2[0]}} | {'GMACs':<{col_w2[1]}} | {'GFLOPs':<{col_w2[2]}} | "
        f"{'VRAM Train (GB)':<{col_w2[3]}} | {'s/iter Train':<{col_w2[4]}} | {'VRAM Inf (GB)':<{col_w2[5]}} | "
        f"{'s/caso Inf':<{col_w2[6]}} | {'s/caso TTA':<{col_w2[7]}}"
    )
    p(header2)
    p("-" * 100)

    m("## 2. Eficiencia Computacional y Consumo de Recursos")
    m()
    m("> Patch de 32 × 128 × 128. Entrenamiento: batch 2 con AMP. Inferencia: ROI típica de 104 × 288 × 448.")
    m()
    m("| Variante | Parámetros | GMACs | GFLOPs | VRAM Train (GB) | s/iteración | VRAM Inferencia (GB) | s/caso | s/caso TTA |")
    m("|---|---|---|---|---|---|---|---|---|")

    for v, info in arch_infos.items():
        eff = efficiency_data.get(v, {})
        gmacs = eff.get("gmacs") or info.get("gmacs")
        gflops = eff.get("gflops") or info.get("gflops")
        vram_tr = eff.get("vram_entrenamiento_gb")
        s_iter = eff.get("s_por_iteracion")
        vram_inf = eff.get("vram_inferencia_gb")
        s_caso = eff.get("s_por_caso")
        s_tta = eff.get("s_por_caso_tta")

        gmacs_s = f"{gmacs:.1f}" if gmacs is not None else "-"
        gflops_s = f"{gflops:.1f}" if gflops is not None else "-"
        vram_tr_s = f"{vram_tr:.2f}" if vram_tr is not None else "-"
        s_iter_s = f"{s_iter:.2f}" if s_iter is not None else "-"
        vram_inf_s = f"{vram_inf:.2f}" if vram_inf is not None else "-"
        s_caso_s = f"{s_caso:.2f}" if s_caso is not None else "-"
        s_tta_s = f"{s_tta:.2f}" if s_tta is not None else "-"

        row2_str = (
            f"{v:<{col_w2[0]}} | {gmacs_s:<{col_w2[1]}} | {gflops_s:<{col_w2[2]}} | "
            f"{vram_tr_s:<{col_w2[3]}} | {s_iter_s:<{col_w2[4]}} | {vram_inf_s:<{col_w2[5]}} | "
            f"{s_caso_s:<{col_w2[6]}} | {s_tta_s:<{col_w2[7]}}"
        )
        p(row2_str)

        m(f"| **{v}** | {info['total_params_M']:.2f} M | {gmacs_s} | {gflops_s} | {vram_tr_s} | {s_iter_s} | {vram_inf_s} | {s_caso_s} | {s_tta_s} |")

    p()
    m()

    # -------------------------------------------------------------
    # TABLA 3: Rendimiento de Segmentacion en HECKTOR 2025 y 2022
    # -------------------------------------------------------------
    splits_to_show = ["val", "test"] if split_choice == "both" else [split_choice]
    val_data = results_data.get("val", {})
    test_data = results_data.get("test", {})

    p("3. RENDIMIENTO DE SEGMENTACION (HECKTOR 2025 & HECKTOR 2022)")
    p("-" * 100)

    m("## 3. Rendimiento de Segmentación")
    m()

    for sp in splits_to_show:
        sp_dict = val_data if sp == "val" else test_data
        n_casos = 135 if sp == "val" else 69
        title_sp_term = f"Validacion ({n_casos} casos)" if sp == "val" else f"Test ({n_casos} casos)"
        title_sp_md = f"Validación ({n_casos} casos)" if sp == "val" else f"Test ({n_casos} casos)"

        p(f"--- Split: {title_sp_term} ---")
        col_w3 = [10, 10, 12, 12, 12, 12, 12, 12]
        header3 = (
            f"{'Modelo':<{col_w3[0]}} | {'CKPT':<{col_w3[1]}} | {'GTVp DSC':<{col_w3[2]}} | "
            f"{'GTVn DSCagg':<{col_w3[3]}} | {'GTVn F1agg':<{col_w3[4]}} | {'Score 2025':<{col_w3[5]}} | "
            f"{'Media 2022':<{col_w3[6]}} | {'Puesto Borda':<{col_w3[7]}}"
        )
        p(header3)
        p("-" * 100)

        m(f"### {title_sp_md}")
        m()
        m("| Modelo | Checkpoint | GTVp DSC | GTVn DSCagg | GTVn F1agg | Score 2025 | Media DSCagg (2022) | Puesto Borda (2025) | Puesto (2022) |")
        m("|---|---|---|---|---|---|---|---|---|")

        for v in arch_infos:
            if v not in sp_dict:
                continue
            item = sp_dict[v]
            ckpt = item.get("checkpoint", "-")
            h25 = item.get("hecktor2025", {})
            h22 = item.get("hecktor2022", {})
            p25 = item.get("puestos_2025", {})
            p22 = item.get("puesto_2022", "-")

            gtvp = h25.get("GTVp_DSC")
            gtvn = h25.get("GTVn_DSCagg")
            f1 = h25.get("GTVn_F1agg")
            score25 = (gtvp + gtvn) / 2 if (gtvp is not None and gtvn is not None) else None
            m22 = h22.get("DSCagg_media")
            borda = p25.get("puesto_final", "-")

            gtvp_s = f"{gtvp:.3f}" if gtvp is not None else "-"
            gtvn_s = f"{gtvn:.3f}" if gtvn is not None else "-"
            f1_s = f"{f1:.3f}" if f1 is not None else "-"
            s25_s = f"{score25:.3f}" if score25 is not None else "-"
            m22_s = f"{m22:.3f}" if m22 is not None else "-"
            borda_s = f"{borda}"

            row3_str = (
                f"{v:<{col_w3[0]}} | {ckpt:<{col_w3[1]}} | {gtvp_s:<{col_w3[2]}} | "
                f"{gtvn_s:<{col_w3[3]}} | {f1_s:<{col_w3[4]}} | {s25_s:<{col_w3[5]}} | "
                f"{m22_s:<{col_w3[6]}} | {borda_s:<{col_w3[7]}}"
            )
            p(row3_str)

            m(f"| **{v}** | {ckpt} | {gtvp_s} | {gtvn_s} | {f1_s} | **{s25_s}** | **{m22_s}** | {borda_s} | {p22} |")

        p()
        m()

    # -------------------------------------------------------------
    # TABLA 4: Analisis de Trade-off (Rendimiento vs Costo)
    # -------------------------------------------------------------
    p("4. ANALISIS DE TRADE-OFF (EFICIENCIA POR COSTO COMPUTACIONAL EN TEST)")
    p("-" * 100)
    col_w4 = [10, 11, 11, 14, 18, 18]
    header4 = (
        f"{'Variante':<{col_w4[0]}} | {'Parametros':<{col_w4[1]}} | {'GMACs':<{col_w4[2]}} | "
        f"{'Score Test 25':<{col_w4[3]}} | {'Score / 1M Params':<{col_w4[4]}} | {'Score / s (Inf)':<{col_w4[5]}}"
    )
    p(header4)
    p("-" * 100)

    m("## 4. Análisis de Trade-off (Eficiencia de Parámetros y Velocidad en Test)")
    m()
    m("| Variante | Parámetros (M) | Reducción Params vs U-Net | GMACs | Score Test 2025 | Score Test 2022 | Score / 1M Params | Inferencia s/caso |")
    m("|---|---|---|---|---|---|---|---|")

    unet_params = arch_infos.get("unet", {}).get("total_params_M", 44.59)

    for v, info in arch_infos.items():
        pm = info["total_params_M"]
        eff = efficiency_data.get(v, {})
        gmacs = eff.get("gmacs") or info.get("gmacs")
        s_caso = eff.get("s_por_caso")

        titem = test_data.get(v, {})
        h25 = titem.get("hecktor2025", {})
        h22 = titem.get("hecktor2022", {})
        gtvp = h25.get("GTVp_DSC")
        gtvn = h25.get("GTVn_DSCagg")
        s25 = (gtvp + gtvn) / 2 if (gtvp is not None and gtvn is not None) else None
        m22 = h22.get("DSCagg_media")

        reduction_str = f"{(1 - pm / unet_params) * 100:.1f}%" if unet_params > 0 else "-"
        score_per_param = (s25 / pm) if (s25 is not None and pm > 0) else None
        score_per_sec = (s25 / s_caso) if (s25 is not None and s_caso is not None and s_caso > 0) else None

        gmacs_s = f"{gmacs:.1f}" if gmacs is not None else "-"
        s25_s = f"{s25:.3f}" if s25 is not None else "-"
        m22_s = f"{m22:.3f}" if m22 is not None else "-"
        spp_s = f"{score_per_param:.3f}" if score_per_param is not None else "-"
        sps_s = f"{score_per_sec:.3f}" if score_per_sec is not None else "-"
        sc_s = f"{s_caso:.2f}" if s_caso is not None else "-"

        row4_str = (
            f"{v:<{col_w4[0]}} | {f'{pm:.2f} M':<{col_w4[1]}} | {gmacs_s:<{col_w4[2]}} | "
            f"{s25_s:<{col_w4[3]}} | {spp_s:<{col_w4[4]}} | {sps_s:<{col_w4[5]}}"
        )
        p(row4_str)

        m(f"| **{v}** | {pm:.2f} M | {reduction_str} | {gmacs_s} | {s25_s} | {m22_s} | **{spp_s}** | {sc_s} |")

    p("-" * 100)
    p()

    # -------------------------------------------------------------
    # SINTESIS Y CONCLUSIONES
    # -------------------------------------------------------------
    conclusiones = (
        "HALLAZGOS CLAVE:\n"
        "1. Compactibilidad de MambaLite3D base: Con solo 4.05 M de parametros y 12.5 GMACs (91% menos parametros\n"
        "   y 91% menos operaciones que el U-Net baseline de 44.59 M), MambaLite3D base alcanza un Score 2025\n"
        "   de 0.686 en test (0.701 en GTVp) vs 0.728 de U-Net. Esto representa una eficiencia de parametros\n"
        "   10 veces mayor (0.169 vs 0.016 puntos por 1M de parametros).\n"
        "2. Escalado con MambaLite3D wide: Al incrementar la capacidad a 15.8 M de parametros, el modelo\n"
        "   mejora el Dice ganglionar (GTVn DSCagg pasa de 0.671 a 0.707 en test) y sube el score 2025 a 0.701,\n"
        "   superando al base con todavia 65% menos parametros que U-Net.\n"
        "3. Ventaja de VRAM y huella de memoria: En entrenamiento con aumentacion GPU, MambaLite3D base ocupa\n"
        "   solo 0.85 GB en la red (1.33 GB total) comparado con 1.84 GB (2.12 GB total) de U-Net, lo que permite\n"
        "   experimentar con patches de mayor contexto espacial (ej. 48x160x160) en GPUs con VRAM moderada.\n"
        "4. U-Net Convolucional: Sigue mostrando un rendimiento solido en ganglios (GTVn DSCagg 0.744 en test)\n"
        "   gracias a su alta densidad de parametros convolucionales (32 a 320 canales), pero a costa de un costo\n"
        "   computacional de 135.0 GMACs y checkpoints de ~178 MB (frente a 16.4 MB de MambaLite3D base)."
    )
    p(conclusiones)

    m()
    m("## 5. Hallazgos Clave")
    m()
    m("- **Compactibilidad de MambaLite3D base:** Con solo **4.05 M de parámetros y 12.5 GMACs** (un **91% menos parámetros y operaciones** que el U-Net baseline de 44.6 M), logra un Score 2025 en test de 0.686. Su rendimiento por parámetro es **~10 veces mayor** que U-Net.")
    m("- **Escalabilidad de MambaLite3D wide:** Al aumentar canales (15.8 M parámetros, 39.7 GMACs), el modelo eleva el Dice ganglionar (GTVn DSCagg pasa de 0.671 a 0.707 en test), alcanzando un Score 2025 de 0.701 con aún 65% menos parámetros que U-Net.")
    m("- **Eficiencia de VRAM:** Base consume **0.85 GB** de VRAM de red en entrenamiento vs **1.84 GB** de U-Net, permitiendo explorar patches mayores o mayores batch sizes.")
    m("- **Desafío en ganglios pequeños:** U-Net retiene la delantera en ganglios (GTVn DSCagg 0.744, F1 0.432), lo que fundamenta las estrategias de ensamble (promediar base + wide + U-Net) o post-procesado para combinar la precisión ganglionar de U-Net con la eficiencia de MambaLite3D.")

    return "\n".join(term_lines), "\n".join(md_lines)


def plot_comparisons(
    arch_infos: Dict[str, Dict[str, Any]],
    efficiency_data: Dict[str, Dict[str, Any]],
    results_data: Dict[str, Any],
    out_path: Path,
) -> bool:
    """Genera gráfico comparativo de 4 paneles con matplotlib."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("[AVISO] matplotlib no está instalado en este entorno. No se generará la gráfica PNG.")
        return False

    matplotlib.use("Agg")
    test_dict = results_data.get("test", {})

    # Modelos con resultados de test
    models_to_plot = [m for m in ("base", "wide", "unet") if m in arch_infos and m in test_dict]
    if not models_to_plot:
        models_to_plot = [m for m in arch_infos if m in test_dict]
    if not models_to_plot:
        models_to_plot = list(arch_infos.keys())[:4]

    colors = {"base": "#2b5c8f", "wide": "#d95f02", "unet": "#7570b3", "wide2": "#1b9e77"}
    def get_color(v):
        return colors.get(v, "#4daf4a")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#fcfcfc")

    # ---------------------------------------------------------
    # Panel 1: Rendimiento (Score Test) vs Parámetros (M)
    # ---------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#ffffff")
    for m in models_to_plot:
        pm = arch_infos[m]["total_params_M"]
        titem = test_dict.get(m, {})
        h25 = titem.get("hecktor2025", {})
        gtvp = h25.get("GTVp_DSC", 0.0)
        gtvn = h25.get("GTVn_DSCagg", 0.0)
        s25 = (gtvp + gtvn) / 2 if (gtvp > 0 and gtvn > 0) else None

        eff = efficiency_data.get(m, {})
        gmacs = eff.get("gmacs") or arch_infos[m].get("gmacs", 20.0)
        marker_size = max(100, min(800, gmacs * 5))

        if s25 is not None:
            ax1.scatter(pm, s25, s=marker_size, color=get_color(m), alpha=0.85, edgecolors="black", linewidth=1.5, zorder=3)
            ax1.annotate(
                f"{m.upper()}\n({pm:.1f}M, {s25:.3f})",
                (pm, s25),
                textcoords="offset points",
                xytext=(0, 14),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    ax1.set_title("A. Rendimiento (Score Test 2025) vs Parámetros", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Parámetros (Millones)", fontsize=10)
    ax1.set_ylabel("Score HECKTOR 2025 (Media GTVp + GTVn)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim(0.65, 0.75)
    ax1.set_xlim(-2, max([arch_infos[m]["total_params_M"] for m in models_to_plot]) * 1.15)

    # ---------------------------------------------------------
    # Panel 2: Rendimiento vs Tiempo de Inferencia (s/caso)
    # ---------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#ffffff")
    for m in models_to_plot:
        eff = efficiency_data.get(m, {})
        s_caso = eff.get("s_por_caso")
        titem = test_dict.get(m, {})
        h25 = titem.get("hecktor2025", {})
        gtvp = h25.get("GTVp_DSC", 0.0)
        gtvn = h25.get("GTVn_DSCagg", 0.0)
        s25 = (gtvp + gtvn) / 2 if (gtvp > 0 and gtvn > 0) else None

        if s_caso is not None and s25 is not None:
            ax2.scatter(s_caso, s25, s=250, color=get_color(m), alpha=0.85, edgecolors="black", linewidth=1.5, zorder=3)
            ax2.annotate(
                f"{m.upper()}\n({s_caso:.1f}s)",
                (s_caso, s25),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    ax2.set_title("B. Rendimiento vs Tiempo de Inferencia por Caso", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Tiempo Inferencia (segundos / caso)", fontsize=10)
    ax2.set_ylabel("Score HECKTOR 2025", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_ylim(0.65, 0.75)

    # ---------------------------------------------------------
    # Panel 3: Perfil de Recursos Computacionales (Barras agrupadas)
    # ---------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor("#ffffff")
    x = np.arange(len(models_to_plot))
    w = 0.2

    params_vals = [arch_infos[m]["total_params_M"] for m in models_to_plot]
    gmacs_vals = [(efficiency_data.get(m, {}).get("gmacs") or arch_infos[m].get("gmacs", 0)) for m in models_to_plot]
    vram_tr_vals = [efficiency_data.get(m, {}).get("vram_entrenamiento_gb", 0) for m in models_to_plot]
    vram_inf_vals = [efficiency_data.get(m, {}).get("vram_inferencia_gb", 0) for m in models_to_plot]

    ax3.bar(x - 1.5 * w, params_vals, width=w, label="Parámetros (M)", color="#2b5c8f", alpha=0.9)
    ax3.bar(x - 0.5 * w, gmacs_vals, width=w, label="GMACs (Patch 32×128²)", color="#41b6c4", alpha=0.9)
    ax3.bar(x + 0.5 * w, [v * 10 for v in vram_tr_vals], width=w, label="VRAM Train (×10 GB)", color="#fe9929", alpha=0.9)
    ax3.bar(x + 1.5 * w, [v * 10 for v in vram_inf_vals], width=w, label="VRAM Inf (×10 GB)", color="#e31a1c", alpha=0.9)

    ax3.set_title("C. Perfil de Recursos Computacionales", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.upper() for m in models_to_plot], fontweight="bold")
    ax3.set_ylabel("Valor (unidades indicadas)", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.4, axis="y")

    # ---------------------------------------------------------
    # Panel 4: Métricas de Segmentación en Test (GTVp, GTVn, F1, 2022)
    # ---------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor("#ffffff")
    x4 = np.arange(len(models_to_plot))
    w4 = 0.2

    gtvp_vals = [test_dict.get(m, {}).get("hecktor2025", {}).get("GTVp_DSC", 0) for m in models_to_plot]
    gtvn_vals = [test_dict.get(m, {}).get("hecktor2025", {}).get("GTVn_DSCagg", 0) for m in models_to_plot]
    f1_vals = [test_dict.get(m, {}).get("hecktor2025", {}).get("GTVn_F1agg", 0) for m in models_to_plot]
    m22_vals = [test_dict.get(m, {}).get("hecktor2022", {}).get("DSCagg_media", 0) for m in models_to_plot]

    ax4.bar(x4 - 1.5 * w4, gtvp_vals, width=w4, label="GTVp DSC", color="#1f78b4", alpha=0.9)
    ax4.bar(x4 - 0.5 * w4, gtvn_vals, width=w4, label="GTVn DSCagg", color="#33a02c", alpha=0.9)
    ax4.bar(x4 + 0.5 * w4, f1_vals, width=w4, label="GTVn F1agg", color="#fb9a99", alpha=0.9)
    ax4.bar(x4 + 1.5 * w4, m22_vals, width=w4, label="Media DSCagg (2022)", color="#6a3d9a", alpha=0.9)

    # Línea de meta 0.75 de HECKTOR 2022
    ax4.axhline(0.75, color="#e31a1c", linestyle=":", linewidth=1.5, label="Meta 2022 (0.75)")

    ax4.set_title("D. Métricas de Segmentación en Test (69 casos)", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xticks(x4)
    ax4.set_xticklabels([m.upper() for m in models_to_plot], fontweight="bold")
    ax4.set_ylabel("Dice / F1 Score", fontsize=10)
    ax4.set_ylim(0.0, 0.85)
    ax4.legend(loc="lower right", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle("Comparativa de Arquitecturas: MambaLite3D (Base / Wide) vs U-Net Baseline", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Grafica comparativa guardada en: {out_path}")
    return True


def plot_case_distributions(runs_dir: Path, out_path: Path) -> bool:
    """Genera boxplots con distribuciones caso por caso (69 pacientes de test)."""
    try:
        import csv
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    matplotlib.use("Agg")
    models_configs = [
        ("Base", runs_dir / "full_base_fold0" / "per_case_test_best.csv", "#2b5c8f"),
        ("Wide", runs_dir / "exp_wide" / "per_case_test_final.csv", "#d95f02"),
        ("U-Net", runs_dir / "full_unet" / "per_case_test_final.csv", "#7570b3"),
    ]

    data_gtvp = {}
    data_gtvn = {}
    for name, path, color in models_configs:
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                gtvp = []
                gtvn = []
                for r in reader:
                    p_val = float(r.get("dice_gtvp", "nan"))
                    n_val = float(r.get("dice_gtvn_case", "nan"))
                    if not math.isnan(p_val):
                        gtvp.append(p_val)
                    if not math.isnan(n_val):
                        gtvn.append(n_val)
                data_gtvp[name] = (gtvp, color)
                data_gtvn[name] = (gtvn, color)
        except Exception:
            pass

    if not data_gtvp:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#fcfcfc")

    names = list(data_gtvp.keys())
    x_positions = list(range(1, len(names) + 1))

    # Panel 1: GTVp Dice por paciente
    ax1 = axes[0]
    ax1.set_facecolor("#ffffff")
    p_data = [data_gtvp[n][0] for n in names]
    bp1 = ax1.boxplot(p_data, positions=x_positions, patch_artist=True, widths=0.45,
                      medianprops=dict(color="black", linewidth=2),
                      boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.2),
                      flierprops=dict(marker='o', markersize=4, alpha=0.5))
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(names, fontweight="bold")
    for patch, name in zip(bp1['boxes'], names):
        patch.set_facecolor(data_gtvp[name][1])
        patch.set_alpha(0.65)

    # Añadir puntos con jitter ligero
    rng = np.random.default_rng(42)
    for i, (n, vals) in enumerate(zip(names, p_data)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax1.scatter(np.full_like(vals, i + 1) + jitter, vals, color=data_gtvp[n][1],
                    alpha=0.4, s=20, edgecolors="none")

    ax1.set_title("Distribución de Dice GTVp (Primario) por Paciente", fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylabel("Dice Score", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel 2: GTVn Dice por paciente
    ax2 = axes[1]
    ax2.set_facecolor("#ffffff")
    n_data = [data_gtvn[n][0] for n in names]
    bp2 = ax2.boxplot(n_data, positions=x_positions, patch_artist=True, widths=0.45,
                      medianprops=dict(color="black", linewidth=2),
                      boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.2),
                      flierprops=dict(marker='o', markersize=4, alpha=0.5))
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(names, fontweight="bold")
    for patch, name in zip(bp2['boxes'], names):
        patch.set_facecolor(data_gtvn[name][1])
        patch.set_alpha(0.65)

    for i, (n, vals) in enumerate(zip(names, n_data)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax2.scatter(np.full_like(vals, i + 1) + jitter, vals, color=data_gtvn[n][1],
                    alpha=0.4, s=20, edgecolors="none")

    ax2.set_title("Distribución de Dice GTVn (Ganglios) por Paciente", fontsize=11, fontweight="bold", pad=10)
    ax2.set_ylabel("Dice Score", fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle("Variabilidad Clínica y Distribución por Caso en Test (69 pacientes)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Grafica de distribuciones por paciente guardada en: {out_path}")
    return True


def plot_radar(
    arch_infos: Dict[str, Dict[str, Any]],
    efficiency_data: Dict[str, Dict[str, Any]],
    results_data: Dict[str, Any],
    out_path: Path,
) -> bool:
    """Genera un radar chart multidimensional comparando precisión y eficiencia."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    matplotlib.use("Agg")
    test_dict = results_data.get("test", {})
    models = [m for m in ("base", "wide", "unet") if m in arch_infos and m in test_dict]
    if not models:
        return False

    categories = [
        "GTVp DSC\n(Primario)",
        "GTVn DSCagg\n(Ganglionar)",
        "GTVn F1agg\n(Detección)",
        "Media 2022\n(Meta ≥0.75)",
        "Ahorro Params\n(vs U-Net)",
        "Ahorro GMACs\n(vs U-Net)",
        "Ahorro VRAM\n(vs U-Net)",
        "Velocidad Inf\n(s/caso invertido)",
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    unet_p = arch_infos.get("unet", {}).get("total_params_M", 44.59)
    unet_g = efficiency_data.get("unet", {}).get("gmacs", 135.0)
    unet_v = efficiency_data.get("unet", {}).get("vram_entrenamiento_gb", 1.84)

    colors = {"base": "#2b5c8f", "wide": "#d95f02", "unet": "#7570b3"}

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#fcfcfc")
    ax.set_facecolor("#ffffff")

    for m in models:
        h25 = test_dict[m].get("hecktor2025", {})
        h22 = test_dict[m].get("hecktor2022", {})
        eff = efficiency_data.get(m, {})

        gtvp = h25.get("GTVp_DSC", 0.0) / 0.8  # normalizado a escala ~1.0
        gtvn = h25.get("GTVn_DSCagg", 0.0) / 0.8
        f1 = h25.get("GTVn_F1agg", 0.0) / 0.5
        m22 = h22.get("DSCagg_media", 0.0) / 0.8

        pm = arch_infos[m]["total_params_M"]
        gm = eff.get("gmacs", arch_infos[m].get("gmacs", 12.5))
        vm = eff.get("vram_entrenamiento_gb", 1.0)
        s_caso = eff.get("s_por_caso", 8.0)

        ahorro_p = 1.0 - (pm / unet_p) if unet_p > 0 else 0.5
        ahorro_g = 1.0 - (gm / unet_g) if unet_g > 0 else 0.5
        ahorro_v = 1.0 - (vm / unet_v) if unet_v > 0 else 0.5
        velocidad = 1.0 - (s_caso / 15.0)  # más rápido = más cerca de 1

        values = [gtvp, gtvn, f1, m22, ahorro_p, ahorro_g, ahorro_v, velocidad]
        # recortar a [0, 1]
        values = [max(0.0, min(1.0, float(v))) for v in values]
        values += values[:1]

        col = colors.get(m, "#333333")
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=m.upper(), color=col)
        ax.fill(angles, values, color=col, alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="gray", size=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), fontsize=10)
    plt.title("Perfil Multidimensional: Precisión de Segmentación vs Eficiencia Hardware", fontsize=12, fontweight="bold", pad=25)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Radar chart guardado en: {out_path}")
    return True


def run_comparison(
    variants: Sequence[str] = ("base", "wide", "unet"),
    patch_zyx: Sequence[int] = (32, 128, 128),
    spacing_zyx: Sequence[float] = (3.0, 1.0, 1.0),
    runs_dir: Optional[Path] = None,
    split_choice: str = "both",
    calc_ops: bool = False,
    generate_plot: bool = True,
    out_plot: Optional[Path] = None,
    save_md: Optional[Path] = None,
    save_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Función principal que ejecuta la comparación completa."""
    if runs_dir is None:
        runs_dir = Path("runs")
        if not runs_dir.exists() and (Path(__file__).resolve().parent.parent.parent / "runs").exists():
            runs_dir = Path(__file__).resolve().parent.parent.parent / "runs"

    eficiencia_path = runs_dir / "eficiencia.json"
    resultados_path = runs_dir / "resultados.json"

    # Cargar datos existentes
    efficiency_data = load_efficiency_file(eficiencia_path)
    results_data = load_results_file(resultados_path)

    # Si variants es 'all', tomamos todas las variantes de config
    if "all" in variants:
        variants = list(VARIANTS.keys())

    # Inspeccionar arquitecturas
    arch_infos: Dict[str, Dict[str, Any]] = {}
    print(f"[INFO] Analizando {len(variants)} arquitecturas con PyTorch...")
    for v in variants:
        if v not in VARIANTS:
            print(f"[AVISO] Variante desconocida '{v}', omitiendo.")
            continue
        try:
            # Si ya tenemos ops en eficiencia y no forzaron calc_ops, no recalculamos FLOPs
            should_calc_ops = calc_ops and (v not in efficiency_data)
            info = inspect_architecture(
                v, patch_zyx=patch_zyx, spacing_zyx=spacing_zyx, device="cpu", calc_ops=should_calc_ops
            )
            arch_infos[v] = info
        except Exception as e:
            print(f"[ERROR] Error al analizar variante '{v}': {e}")

    # Si no había resultados.json, intentar leer de runs individuales
    if not results_data.get("val") and not results_data.get("test"):
        results_data = {"seleccion": "2025", "modelos": {}, "val": {}, "test": {}}
        for v in variants:
            rpath = runs_dir / DEFAULT_RUNS.get(v, f"exp_{v}")
            if not rpath.exists():
                rpath = runs_dir / f"full_{v}"
            if rpath.exists():
                m_val = load_run_metrics(rpath, "val", "best")
                m_test = load_run_metrics(rpath, "test", "best")
                if m_val:
                    results_data["val"][v] = {
                        "checkpoint": "best",
                        "hecktor2025": hecktor2025(m_val),
                        "hecktor2022": hecktor2022(m_val),
                    }
                if m_test:
                    results_data["test"][v] = {
                        "checkpoint": "best",
                        "hecktor2025": hecktor2025(m_test),
                        "hecktor2022": hecktor2022(m_test),
                    }

    # Generar tablas
    term_text, md_text = format_tables_and_summary(arch_infos, efficiency_data, results_data, split_choice)
    print(term_text)

    # Guardar Markdown
    if save_md is not None:
        save_md.parent.mkdir(parents=True, exist_ok=True)
        save_md.write_text(md_text, encoding="utf-8")
        print(f"[OK] Reporte Markdown guardado en: {save_md}")

    # Guardar JSON
    consolidated_json = {
        "architectures": arch_infos,
        "efficiency": efficiency_data,
        "results": results_data,
    }
    if save_json is not None:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(consolidated_json, indent=2), encoding="utf-8")
        print(f"[OK] Reporte JSON guardado en: {save_json}")

    # Generar gráficas
    if generate_plot:
        if out_plot is None:
            out_plot = runs_dir / "comparacion_arquitecturas.png"
        plot_comparisons(arch_infos, efficiency_data, results_data, out_plot)

        # Generar también gráfica de distribuciones por paciente si existen CSVs
        dist_plot = runs_dir / "distribucion_casos_test.png"
        plot_case_distributions(runs_dir, dist_plot)

        # Generar radar chart multidimensional
        radar_plot = runs_dir / "radar_arquitecturas.png"
        plot_radar(arch_infos, efficiency_data, results_data, radar_plot)

    return consolidated_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compara arquitecturas MambaLite3D y baseline U-Net en parámetros, recursos y rendimiento HECKTOR."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "wide", "unet"],
        help=f"Variantes a comparar (por defecto: base wide unet, o 'all' para todas las {len(VARIANTS)}). Opciones: {list(VARIANTS.keys())}",
    )
    parser.add_argument(
        "--patch",
        type=int,
        nargs=3,
        default=[32, 128, 128],
        metavar=("Z", "Y", "X"),
        help="Tamaño de patch en voxels para el plan de strides (def: 32 128 128)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[3.0, 1.0, 1.0],
        metavar=("Z", "Y", "X"),
        help="Espaciado objetivo en mm (def: 3.0 1.0 1.0)",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Carpeta de experimentos runs/ (def: busca ./runs o ../runs)",
    )
    parser.add_argument(
        "--split",
        choices=["both", "val", "test"],
        default="both",
        help="Qué split de evaluación mostrar (both, val, test). Def: both",
    )
    parser.add_argument(
        "--calc-ops",
        action="store_true",
        help="Calcula analíticamente GMACs/FLOPs con PyTorch FlopCounterMode si no están en eficiencia.json",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Desactiva la generación de gráficas PNG",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default=None,
        help="Ruta donde guardar la gráfica principal PNG (def: runs/comparacion_arquitecturas.png)",
    )
    parser.add_argument(
        "--save-md",
        type=str,
        default="runs/comparacion_arquitecturas.md",
        help="Ruta para exportar el reporte en Markdown (def: runs/comparacion_arquitecturas.md)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="runs/comparacion_arquitecturas.json",
        help="Ruta para exportar los datos en JSON (def: runs/comparacion_arquitecturas.json)",
    )

    args = parser.parse_args(argv)

    runs_p = Path(args.runs_dir) if args.runs_dir else None
    out_plot_p = Path(args.out_plot) if args.out_plot else None
    save_md_p = Path(args.save_md) if args.save_md else None
    save_json_p = Path(args.save_json) if args.save_json else None

    run_comparison(
        variants=args.variants,
        patch_zyx=args.patch,
        spacing_zyx=args.spacing,
        runs_dir=runs_p,
        split_choice=args.split,
        calc_ops=args.calc_ops,
        generate_plot=not args.no_plot,
        out_plot=out_plot_p,
        save_md=save_md_p,
        save_json=save_json_p,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
