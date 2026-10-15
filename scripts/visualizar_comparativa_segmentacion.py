#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador y Comparador Cualitativo de Segmentación 3D: MambaLite3D vs BiFusion.

Carga un caso de HECKTOR 2025 (CT + PET + Ground Truth), ejecuta la inferencia
con MambaLite3D Base y BiFusion Siam Full, y genera una comparativa visual
lado a lado de los cortes axiales más informativos (Tumor primario GTVp y Ganglios GTVn).

Características:
- Superposición médica con mapas de color dedicados (Tumor GTVp en rojo, Ganglios GTVn en verde).
- Vistas sobre fondo de TC (escala de grises) y fondo de PET (mapa térmico / hot).
- Cálculo in-situ del Dice local para el caso visualizado.
- Selección automática del caso más representativo (o por argumento --case ID).

Genera:
- Gráfica médica de alta resolución: estadisticas/graficas/comparativa_segmentacion_casos.png

Uso:
    python scripts/visualizar_comparativa_segmentacion.py
    python scripts/visualizar_comparativa_segmentacion.py --case CHUM-001
    python scripts/visualizar_comparativa_segmentacion.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Auto-relanzar con el entorno conda 'mambalite3d' si es necesario
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
except ImportError as err:
    user_prof = os.environ.get("USERPROFILE", "")
    candidates = [
        Path(user_prof) / "miniconda3" / "envs" / "mambalite3d" / "python.exe",
        Path(user_prof) / "anaconda3" / "envs" / "mambalite3d" / "python.exe",
    ]
    conda_py = next((c for c in candidates if c.exists()), None)
    if conda_py and Path(sys.executable).resolve() != conda_py.resolve():
        cmd = [str(conda_py), str(Path(__file__).resolve())] + sys.argv[1:]
        sys.exit(subprocess.call(cmd))
    else:
        print(f"[ERROR] Dependencias faltantes: {err}")
        sys.exit(1)

SCRIPT_PATH = Path(__file__).resolve()
ROOT_DIR = SCRIPT_PATH.parents[1]
MAMBA_DIR = ROOT_DIR / "modelos" / "mamba"
BIFUSION_DIR = ROOT_DIR / "modelos" / "bifusion"
PLOTS_DIR = ROOT_DIR / "estadisticas" / "graficas"

# Asegurar directorios en sys.path
for p in (str(MAMBA_DIR), str(BIFUSION_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from mambalite3d.config import load_config
from mambalite3d.data.io import CaseFiles, center_of, find_cases, load_case
from mambalite3d.models.network import build_model as build_mamba_model
from mambalite3d.models.plan import plan_network
from mambalite3d.training.inference import sliding_window_predict

# Parche bifusion para poder construir la red de bifusion
import bifusion3d.patch
bifusion3d.patch.install()
from bifusion3d.network import build_model as build_bifusion_model


def compute_binary_dice(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    p_sum = pred_binary.sum()
    g_sum = gt_binary.sum()
    if p_sum == 0 and g_sum == 0:
        return 1.0
    if p_sum == 0 or g_sum == 0:
        return 0.0
    inter = np.logical_and(pred_binary, gt_binary).sum()
    return float(2.0 * inter / (p_sum + g_sum))


def load_model_weights(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Comparador visual de segmentación 3D: Mamba vs BiFusion.")
    parser.add_argument("--case", type=str, default=None, help="ID específico del paciente (ej. CHUM-001)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Dispositivo (cuda o cpu)")
    parser.add_argument("--step-fraction", type=float, default=0.5, help="Paso de la ventana deslizante")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Dispositivo de ejecución: {device}")

    # Cargar configuración base
    cfg_path = MAMBA_DIR / "configs" / "full.yaml"
    cfg = load_config(cfg_path)
    data_cfg = cfg["data"]
    data_cfg["root"] = str(ROOT_DIR / "HECKTOR2025_raw")
    patch_zyx = tuple(cfg["model"]["patch_size_zyx"])
    spacing_xyz = tuple(data_cfg["target_spacing_xyz"])

    # Buscar casos
    all_cases = find_cases(data_cfg["root"])
    labeled_cases = [c for c in all_cases if c.label is not None and c.label.exists()]
    if not labeled_cases:
        print("[ERROR] No se encontraron casos etiquetados en HECKTOR2025_raw.")
        sys.exit(1)

    # Seleccionar caso
    selected_case = None
    if args.case:
        for c in labeled_cases:
            if args.case.lower() in c.case_id.lower():
                selected_case = c
                break
        if selected_case is None:
            print(f"[AVISO] Caso {args.case} no encontrado. Se seleccionará uno automáticamente.")

    if selected_case is None:
        # Priorizar casos de validación conocidos con buen volumen tumoral (ej. CHUM-001 o CHUS)
        for c in labeled_cases:
            if "CHUM-001" in c.case_id or "CHUS-003" in c.case_id:
                selected_case = c
                break
        if selected_case is None:
            selected_case = labeled_cases[0]

    print(f"[INFO] Cargando y preprocesando caso: {selected_case.case_id}...")
    cdata = load_case(selected_case, data_cfg, with_label=True)
    image = cdata.image  # (2, Z, Y, X)
    gt_label = cdata.label  # (Z, Y, X)

    # Construir modelos
    strides, kernels = plan_network(patch_zyx, (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]))

    print("[INFO] Instanciando y cargando pesos de MambaLite3D Base...")
    mamba_model = build_mamba_model("base", strides, kernels).to(device)
    mamba_ckpt = MAMBA_DIR / "runs" / "full_base_fold0" / "checkpoint_best.pth"
    if not mamba_ckpt.exists():
        mamba_ckpt = MAMBA_DIR / "runs" / "full_base_fold0" / "checkpoint_final.pth"
    load_model_weights(mamba_model, mamba_ckpt, device)

    print("[INFO] Instanciando y cargando pesos de BiFusion Siam Full...")
    bifusion_model = build_bifusion_model("siam_full", strides, kernels).to(device)
    bifusion_ckpt = BIFUSION_DIR / "runs" / "siam_full" / "checkpoint_best.pth"
    if not bifusion_ckpt.exists():
        bifusion_ckpt = BIFUSION_DIR / "runs" / "siam_full" / "checkpoint_final.pth"
    load_model_weights(bifusion_model, bifusion_ckpt, device)

    # Inferencia con ventana deslizante
    print("[INFO] Ejecutando inferencia con MambaLite3D Base...")
    prob_mamba = sliding_window_predict(
        mamba_model, image, patch_zyx, num_classes=3, device=device,
        step_fraction=args.step_fraction, batch_size=2
    )
    pred_mamba = np.argmax(prob_mamba, axis=0)

    print("[INFO] Ejecutando inferencia con BiFusion Siam Full...")
    prob_bifusion = sliding_window_predict(
        bifusion_model, image, patch_zyx, num_classes=3, device=device,
        step_fraction=args.step_fraction, batch_size=2
    )
    pred_bifusion = np.argmax(prob_bifusion, axis=0)

    # Métricas de este caso
    dice_p_mamba = compute_binary_dice(pred_mamba == 1, gt_label == 1)
    dice_n_mamba = compute_binary_dice(pred_mamba == 2, gt_label == 2)

    dice_p_bi = compute_binary_dice(pred_bifusion == 1, gt_label == 1)
    dice_n_bi = compute_binary_dice(pred_bifusion == 2, gt_label == 2)

    print(f"\n[RESULTADOS DEL CASO {selected_case.case_id}]")
    print(f"  * Mamba Base    -> GTVp Dice: {dice_p_mamba:.3f} | GTVn Dice: {dice_n_mamba:.3f}")
    print(f"  * BiFusion Full -> GTVp Dice: {dice_p_bi:.3f} | GTVn Dice: {dice_n_bi:.3f}")

    # Encontrar el mejor corte axial: aquel con mayor área tumoral o ganglionar
    area_gtvp = np.sum(gt_label == 1, axis=(1, 2))
    area_gtvn = np.sum(gt_label == 2, axis=(1, 2))

    slice_tumor = int(np.argmax(area_gtvp)) if area_gtvp.max() > 0 else gt_label.shape[0] // 2
    slice_lymph = int(np.argmax(area_gtvn)) if area_gtvn.max() > 0 else slice_tumor

    slices_to_plot = [("Corte Axial: Tumor Primario (GTVp)", slice_tumor)]
    if slice_lymph != slice_tumor and area_gtvn.max() > 0:
        slices_to_plot.append(("Corte Axial: Ganglios Linfáticos (GTVn)", slice_lymph))

    # Generar visualización
    n_slices = len(slices_to_plot)
    fig, axes = plt.subplots(n_slices * 2, 3, figsize=(15, 5.5 * n_slices * 2), dpi=300)
    fig.patch.set_facecolor("#111111")

    # Mapeo de colores para etiquetas: 1 = GTVp (Rojo), 2 = GTVn (Verde brillante)
    color_map = {
        1: (1.0, 0.2, 0.2, 0.7),  # Rojo translúcido
        2: (0.1, 0.9, 0.3, 0.7),  # Verde translúcido
    }

    def overlay_mask(ax, bg_img, mask, cmap_bg="gray"):
        ax.imshow(bg_img, cmap=cmap_bg)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        for val, col in color_map.items():
            overlay[mask == val] = col
        ax.imshow(overlay, interpolation="nearest")
        ax.axis("off")

    row = 0
    for title_slice, z in slices_to_plot:
        ct_slice = image[0, z]
        pet_slice = image[1, z]
        gt_s = gt_label[z]
        mamba_s = pred_mamba[z]
        bi_s = pred_bifusion[z]

        # Fila A: Fondo CT
        overlay_mask(axes[row, 0], ct_slice, gt_s, cmap_bg="gray")
        axes[row, 0].set_title(f"{title_slice}\nGround Truth (sobre TC)", color="white", fontsize=10, fontweight="bold")

        overlay_mask(axes[row, 1], ct_slice, mamba_s, cmap_bg="gray")
        axes[row, 1].set_title(f"Mamba Base (sobre TC)\nDice GTVp: {dice_p_mamba:.3f} | GTVn: {dice_n_mamba:.3f}",
                               color="#5dade2", fontsize=10, fontweight="bold")

        overlay_mask(axes[row, 2], ct_slice, bi_s, cmap_bg="gray")
        axes[row, 2].set_title(f"BiFusion Full (sobre TC)\nDice GTVp: {dice_p_bi:.3f} | GTVn: {dice_n_bi:.3f}",
                               color="#e74c3c", fontsize=10, fontweight="bold")

        # Fila B: Fondo PET
        overlay_mask(axes[row + 1, 0], pet_slice, gt_s, cmap_bg="hot")
        axes[row + 1, 0].set_title("Ground Truth (sobre PET)", color="white", fontsize=10, fontweight="bold")

        overlay_mask(axes[row + 1, 1], pet_slice, mamba_s, cmap_bg="hot")
        axes[row + 1, 1].set_title("Mamba Base (sobre PET)", color="#5dade2", fontsize=10, fontweight="bold")

        overlay_mask(axes[row + 1, 2], pet_slice, bi_s, cmap_bg="hot")
        axes[row + 1, 2].set_title("BiFusion Full (sobre PET)", color="#e74c3c", fontsize=10, fontweight="bold")

        row += 2

    # Leyenda global
    fig.text(0.5, 0.015,
             "Leyenda de Colores: ■ Rojo = Tumor Primario (GTVp)  |  ■ Verde = Ganglios Linfáticos (GTVn)",
             ha="center", fontsize=12, color="white", fontweight="bold")

    plt.suptitle(f"Comparativa de Segmentación Multimodal: Paciente {selected_case.case_id}",
                 color="white", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    out_plot = PLOTS_DIR / "comparativa_segmentacion_casos.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"\n[EXITO] Imagen de segmentación comparativa guardada en:\n  {out_plot}")


if __name__ == "__main__":
    main()
