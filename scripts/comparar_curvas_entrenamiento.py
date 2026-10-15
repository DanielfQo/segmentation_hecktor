#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparación de Curvas de Entrenamiento y Dinámica de Aprendizaje: MambaLite3D vs BiFusion.

Carga history.json y metrics_ep*.json de los runs de Mamba y BiFusion:
- Pérdida de entrenamiento (Loss) a lo largo de las épocas (con suavizado exponencial opcional).
- Evolución del Score de validación (HECKTOR 2025).
- Progresión de GTVp DSC y GTVn DSC en validación periódica.
- Tiempo por época y consumo de VRAM asignada.

Genera:
- Gráfica combinada de 4 paneles: estadisticas/graficas/comparativa_curvas_entrenamiento.png
- Resumen en consola de velocidad y convergencia.

Uso:
    python scripts/comparar_curvas_entrenamiento.py
    python scripts/comparar_curvas_entrenamiento.py --smooth 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Re-lanzar con el intérprete de conda 'mambalite3d' si es necesario
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
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

RUN_CONFIGS = [
    {
        "id": "full_base_fold0",
        "path": MAMBA_DIR / "runs" / "full_base_fold0",
        "label": "Mamba Base (Early Fusion)",
        "color": "#1f77b4",
        "family": "MambaLite3D",
        "linestyle": "-",
    },
    {
        "id": "exp_wide",
        "path": MAMBA_DIR / "runs" / "exp_wide",
        "label": "Mamba Wide (Canales x1.5)",
        "color": "#17becf",
        "family": "MambaLite3D",
        "linestyle": "--",
    },
    {
        "id": "full_unet",
        "path": MAMBA_DIR / "runs" / "full_unet",
        "label": "U-Net Res (Baseline Conv)",
        "color": "#7f7f7f",
        "family": "CNN Baseline",
        "linestyle": ":",
    },
    {
        "id": "siam_light",
        "path": BIFUSION_DIR / "runs" / "siam_light",
        "label": "BiFusion Light (Siamés L0-1)",
        "color": "#ff7f0e",
        "family": "BiFusion",
        "linestyle": "-.",
    },
    {
        "id": "siam_full",
        "path": BIFUSION_DIR / "runs" / "siam_full",
        "label": "BiFusion Full (Cross/Concat Mamba)",
        "color": "#d62728",
        "family": "BiFusion",
        "linestyle": "-",
    },
]


def smooth_curve(scalars: List[float], weight: float = 0.8) -> List[float]:
    """Suavizado exponencial tipo TensorBoard."""
    if not scalars or weight <= 0:
        return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        if np.isnan(point):
            smoothed.append(last)
            continue
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def load_run_history(run_dir: Path) -> Optional[Dict]:
    history_file = run_dir / "history.json"
    if not history_file.exists():
        return None

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception as e:
        print(f"[AVISO] No se pudo leer {history_file}: {e}")
        return None

    if not isinstance(history, list) or len(history) == 0:
        return None

    epochs = [h.get("epoch", i) for i, h in enumerate(history)]
    losses = [h.get("loss", float("nan")) for h in history]
    lrs = [h.get("lr", float("nan")) for h in history]
    seconds = [h.get("seconds", float("nan")) for h in history]
    vram = [h.get("peak_alloc_gb", float("nan")) for h in history]

    # Cargar métricas de checkpoints periódicos
    val_epochs = []
    val_scores = []
    val_gtvp = []
    val_gtvn = []

    for metric_file in run_dir.glob("metrics_ep*.json"):
        match = re.search(r"metrics_ep(\d+)\.json", metric_file.name)
        if match:
            ep = int(match.group(1))
            try:
                with open(metric_file, "r", encoding="utf-8") as f:
                    mdata = json.load(f)
                val_epochs.append(ep)
                score = mdata.get("score")
                roi = mdata.get("roi_space") or mdata.get("original_space") or {}
                gtvp = roi.get("gtvp_dice_mean", float("nan"))
                gtvn = roi.get("gtvn_dice_agg", float("nan"))

                if score is None or np.isnan(score):
                    score = 0.5 * (gtvp + gtvn) if not np.isnan(gtvp) and not np.isnan(gtvn) else float("nan")

                val_scores.append(score)
                val_gtvp.append(gtvp)
                val_gtvn.append(gtvn)
            except Exception:
                pass

    if val_epochs:
        sort_idx = np.argsort(val_epochs)
        val_epochs = [val_epochs[i] for i in sort_idx]
        val_scores = [val_scores[i] for i in sort_idx]
        val_gtvp = [val_gtvp[i] for i in sort_idx]
        val_gtvn = [val_gtvn[i] for i in sort_idx]

    return {
        "epochs": epochs,
        "losses": losses,
        "lrs": lrs,
        "seconds": seconds,
        "vram": vram,
        "val_epochs": val_epochs,
        "val_scores": val_scores,
        "val_gtvp": val_gtvp,
        "val_gtvn": val_gtvn,
    }


def main():
    parser = argparse.ArgumentParser(description="Compara las curvas de entrenamiento de MambaLite3D y BiFusion.")
    parser.add_argument("--smooth", type=float, default=0.75, help="Factor de suavizado exponencial (0.0 a 0.99)")
    args = parser.parse_args()

    print("[INFO] Cargando historias de entrenamiento de MambaLite3D y BiFusion...")
    loaded_data = []
    for cfg in RUN_CONFIGS:
        hdata = load_run_history(cfg["path"])
        if hdata is not None:
            cfg_copy = dict(cfg)
            cfg_copy.update(hdata)
            loaded_data.append(cfg_copy)
            print(f"  * {cfg['label']}: {len(hdata['epochs'])} épocas, {len(hdata['val_epochs'])} checkpoints val")

    if not loaded_data:
        print("[ERROR] No se encontraron archivos history.json válidos.")
        sys.exit(1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor("#fcfcfc")

    for row in axes:
        for ax in row:
            ax.set_facecolor("#ffffff")
            ax.grid(True, linestyle="--", alpha=0.45)

    ax_loss = axes[0, 0]
    ax_score = axes[0, 1]
    ax_gtvn = axes[1, 0]
    ax_speed = axes[1, 1]

    # Panel 1: Pérdida (Loss)
    for d in loaded_data:
        losses = d["losses"]
        epochs = d["epochs"]
        smoothed = smooth_curve(losses, weight=args.smooth)
        ax_loss.plot(epochs, losses, color=d["color"], alpha=0.18, linewidth=0.8)
        ax_loss.plot(epochs, smoothed, color=d["color"], label=d["label"],
                    linestyle=d["linestyle"], linewidth=2.0)

    ax_loss.set_title("Curva de Pérdida de Entrenamiento (Dice + CE)", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Época", fontsize=10)
    ax_loss.set_ylabel("Pérdida (Loss)", fontsize=10)
    ax_loss.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Panel 2: Validación Score 2025
    for d in loaded_data:
        if d["val_epochs"]:
            ax_score.plot(d["val_epochs"], d["val_scores"], marker="o", markersize=4,
                         color=d["color"], label=d["label"], linestyle=d["linestyle"], linewidth=1.8)

    ax_score.set_title("Evolución del Score de Validación HECKTOR 2025", fontsize=11, fontweight="bold")
    ax_score.set_xlabel("Época de Validación", fontsize=10)
    ax_score.set_ylabel("Score (0.5 GTVp + 0.5 GTVn)", fontsize=10)
    ax_score.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # Panel 3: Evolución de Ganglios Linfáticos (GTVn DSC)
    for d in loaded_data:
        if d["val_epochs"]:
            ax_gtvn.plot(d["val_epochs"], d["val_gtvn"], marker="s", markersize=4,
                        color=d["color"], label=d["label"], linestyle=d["linestyle"], linewidth=1.8)

    ax_gtvn.set_title("Convergencia en Ganglios Linfáticos (GTVn DSC)", fontsize=11, fontweight="bold")
    ax_gtvn.set_xlabel("Época de Validación", fontsize=10)
    ax_gtvn.set_ylabel("GTVn DSC", fontsize=10)
    ax_gtvn.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # Panel 4: Tiempo por Época y Consumo VRAM
    for d in loaded_data:
        epochs = d["epochs"]
        secs = smooth_curve(d["seconds"], weight=0.6)
        ax_speed.plot(epochs, secs, color=d["color"], label=f"{d['label']} ({np.nanmean(d['seconds']):.1f} s/ep)",
                     linestyle=d["linestyle"], linewidth=1.8)

    ax_speed.set_title("Tiempo de Cómputo por Época (Segundos)", fontsize=11, fontweight="bold")
    ax_speed.set_xlabel("Época", fontsize=10)
    ax_speed.set_ylabel("Segundos por Época", fontsize=10)
    ax_speed.legend(fontsize=8, loc="upper right", framealpha=0.9)

    plt.suptitle("Dinámica de Entrenamiento y Convergencia: MambaLite3D vs Modelos BiFusion",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_file = PLOTS_DIR / "comparativa_curvas_entrenamiento.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[EXITO] Gráfica de entrenamiento guardada en: {out_file}")


if __name__ == "__main__":
    main()
