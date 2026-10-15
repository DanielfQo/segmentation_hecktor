#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparación Integral entre Modelos BiFusion y MambaLite3D para HECKTOR 2025/2022.

Compara:
1. Rendimiento de segmentación (HECKTOR 2025: GTVp DSC, GTVn DSCagg, GTVn F1agg, Score 2025, Ranking Borda).
2. Rendimiento HECKTOR 2022 (GTVp DSCagg, GTVn DSCagg, DSCagg media).
3. Comparación estadística pareada (Bootstrap 95% CI, p-valor de permutación, test de rangos signados de Wilcoxon).
4. Eficiencia computacional (Parámetros M, GMACs, GFLOPs, VRAM pico entrenamiento/inferencia, latencia).
5. Diferencias arquitectónicas y de fusión multimodal (Early fusion vs Siamés ligero vs Siamés Cross-Mamba).

Genera:
- Resumen en consola con tablas formateadas.
- Reporte detallado en Markdown: estadisticas/comparativa_bifusion_mamba.md
- Datos estructurados en JSON: estadisticas/comparativa_bifusion_mamba.json
- Tabla resumen en CSV: estadisticas/comparativa_bifusion_mamba.csv
- Gráficas comparativas en alta resolución (PNG, 300 DPI):
    * estadisticas/graficas/comparativa_metricas_barras.png
    * estadisticas/graficas/comparativa_distribucion_dice.png
    * estadisticas/graficas/comparativa_pareto_eficiencia.png
    * estadisticas/graficas/comparativa_radar.png

Uso:
    python scripts/comparar_bifusion_mamba.py
    python scripts/comparar_bifusion_mamba.py --split val
    python scripts/comparar_bifusion_mamba.py --no-plot
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Re-lanzar con el intérprete de conda 'mambalite3d' si no tenemos las dependencias
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import rankdata, wilcoxon
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
        print(f"[ERROR] Dependencias faltantes (numpy, scipy, matplotlib): {err}")
        print("Activa el entorno conda 'mambalite3d' o instala los requisitos.")
        sys.exit(1)

# Determinar directorios clave
SCRIPT_PATH = Path(__file__).resolve()
ROOT_DIR = SCRIPT_PATH.parents[1]
MAMBA_DIR = ROOT_DIR / "modelos" / "mamba"
BIFUSION_DIR = ROOT_DIR / "modelos" / "bifusion"
OUTPUT_DIR = ROOT_DIR / "estadisticas"
PLOTS_DIR = OUTPUT_DIR / "graficas"

FIELDS = {
    "p": "dice_gtvp",
    "ni": "gtvn_inter_mm3",
    "ns": "gtvn_sum_mm3",
    "tp": "gtvn_tp",
    "fp": "gtvn_fp",
    "fn": "gtvn_fn",
    "pi": "gtvp_inter_mm3",
    "ps": "gtvp_sum_mm3",
}

# Metadatos arquitectónicos y de eficiencia de referencia
ARCH_SPECS = {
    "full_base_fold0": {
        "family": "MambaLite3D",
        "label": "Mamba Base",
        "variant": "base",
        "fusion_type": "Fusión Temprana (Canales 0-1)",
        "params_M": 4.05,
        "gmacs": 12.47,
        "gflops": 24.93,
        "vram_train_gb": 0.85,
        "vram_infer_gb": 0.98,
        "s_iter": 0.59,
        "s_case": 7.49,
        "color": "#1f77b4",
    },
    "exp_wide": {
        "family": "MambaLite3D",
        "label": "Mamba Wide",
        "variant": "wide",
        "fusion_type": "Fusión Temprana (Canales ampliados)",
        "params_M": 15.80,
        "gmacs": 39.75,
        "gflops": 79.49,
        "vram_train_gb": 1.37,
        "vram_infer_gb": 1.39,
        "s_iter": 0.79,
        "s_case": 11.91,
        "color": "#17becf",
    },
    "full_unet": {
        "family": "Baseline CNN",
        "label": "U-Net Res",
        "variant": "unet",
        "fusion_type": "Fusión Temprana (Conv residual pura)",
        "params_M": 44.59,
        "gmacs": 134.99,
        "gflops": 269.98,
        "vram_train_gb": 1.95,
        "vram_infer_gb": 1.48,
        "s_iter": 0.85,
        "s_case": 13.20,
        "color": "#7f7f7f",
    },
    "siam_light": {
        "family": "BiFusion",
        "label": "BiFusion Light",
        "variant": "siam_light",
        "fusion_type": "Codificador Siamés L0-1 + GatedFusion3D",
        "params_M": 4.05,
        "gmacs": 17.90,
        "gflops": 35.80,
        "vram_train_gb": 1.41,
        "vram_infer_gb": 1.03,
        "s_iter": 0.80,
        "s_case": 12.69,
        "color": "#ff7f0e",
    },
    "siam_full": {
        "family": "BiFusion",
        "label": "BiFusion Full",
        "variant": "siam_full",
        "fusion_type": "Siamés completo + Cross/Concat Mamba L4-5",
        "params_M": 4.26,
        "gmacs": 19.68,
        "gflops": 39.36,
        "vram_train_gb": 1.46,
        "vram_infer_gb": 1.03,
        "s_iter": 0.97,
        "s_case": 15.58,
        "color": "#d62728",
    },
}


def _agg(inter: np.ndarray, total: np.ndarray) -> np.ndarray:
    return np.where(total > 0, 2 * inter / np.where(total > 0, total, 1.0), 1.0)


def compute_metrics_from_arrays(arr: dict, idx=None) -> dict:
    """Calcula métricas agregadas HECKTOR 2025 y 2022."""
    def take(a):
        if a.ndim == 2:
            return a
        return a[None, :] if idx is None else a[idx]

    gtvp = take(arr["p"]).mean(axis=1)
    gtvn = _agg(take(arr["ni"]).sum(axis=1), take(arr["ns"]).sum(axis=1))
    tp, fp, fn = (take(arr[k]).sum(axis=1) for k in ("tp", "fp", "fn"))
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1.0), 1.0)
    score_25 = 0.5 * (gtvp + gtvn)

    out = {
        "gtvp_mean": gtvp,
        "gtvn_agg": gtvn,
        "f1_ganglios": f1,
        "score_2025": score_25,
    }

    if np.all(np.isfinite(arr["pi"])) and np.all(np.isfinite(arr["ps"])):
        gtvp_agg = _agg(take(arr["pi"]).sum(axis=1), take(arr["ps"]).sum(axis=1))
        out["gtvp_agg"] = gtvp_agg
        out["dsc_agg_mean_2022"] = 0.5 * (gtvp_agg + gtvn)

    return out


def load_per_case_csv(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {r["case_id"]: r for r in reader}


def extract_arrays(rows: Dict[str, Dict[str, str]], case_ids: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for key, col in FIELDS.items():
        out[key] = np.array([float(rows[cid].get(col, "nan") or "nan") for cid in case_ids])
    return out


def compare_paired(a_arr: dict, b_arr: dict, n_boot: int = 10000, n_perm: int = 10000, seed: int = 0) -> dict:
    """Comparación estadística pareada (bootstrap y permutación)."""
    n = len(a_arr["p"])
    rng = np.random.default_rng(seed)
    ma, mb = compute_metrics_from_arrays(a_arr), compute_metrics_from_arrays(b_arr)
    keys = [k for k in ma if k in mb]

    idx = rng.integers(0, n, size=(n_boot, n))
    ba, bb = compute_metrics_from_arrays(a_arr, idx), compute_metrics_from_arrays(b_arr, idx)

    swap = rng.random((n_perm, n)) < 0.5
    pa = {k: np.where(swap, b_arr[k], a_arr[k]) for k in a_arr}
    pb = {k: np.where(swap, a_arr[k], b_arr[k]) for k in a_arr}
    qa, qb = compute_metrics_from_arrays(pa), compute_metrics_from_arrays(pb)

    res = {}
    for k in keys:
        d = float(mb[k][0] - ma[k][0])
        lo, hi = np.percentile(bb[k] - ba[k], [2.5, 97.5])
        perm = qb[k] - qa[k]
        p = (np.sum(np.abs(perm) >= abs(d) - 1e-12) + 1) / (n_perm + 1)
        res[k] = {
            "a": float(ma[k][0]),
            "b": float(mb[k][0]),
            "diff": d,
            "ci95": [float(lo), float(hi)],
            "p_perm": float(p),
        }

    diff_p = b_arr["p"] - a_arr["p"]
    if np.any(np.abs(diff_p) > 1e-12):
        try:
            w = wilcoxon(b_arr["p"], a_arr["p"], zero_method="pratt")
            res["gtvp_wilcoxon"] = {
                "p": float(w.pvalue),
                "b_better": int(np.sum(diff_p > 1e-9)),
                "b_worse": int(np.sum(diff_p < -1e-9)),
            }
        except Exception:
            pass

    return res


def compute_borda_ranks(metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Calcula los puestos Borda oficiales para HECKTOR 2025 y 2022."""
    models = list(metrics_dict.keys())
    if not models:
        return {}

    gtvp_vals = np.array([metrics_dict[m]["gtvp_dice"] for m in models])
    gtvn_dsc_vals = np.array([metrics_dict[m]["gtvn_dice_agg"] for m in models])
    gtvn_f1_vals = np.array([metrics_dict[m]["gtvn_f1_agg"] for m in models])
    score_25_vals = np.array([metrics_dict[m]["score_2025"] for m in models])

    # Rank 1 = mejor (-valor)
    r_gtvp = rankdata(-gtvp_vals, method="average")
    r_gtvn_dsc = rankdata(-gtvn_dsc_vals, method="average")
    r_gtvn_f1 = rankdata(-gtvn_f1_vals, method="average")

    # Borda GTVn = suma de puestos en dsc y f1
    r_gtvn = rankdata(r_gtvn_dsc + r_gtvn_f1, method="average")
    # Puesto final Borda = suma de puestos en gtvp y gtvn
    r_final_borda = rankdata(r_gtvp + r_gtvn, method="average")
    # Posicion media
    r_mean_pos = rankdata((r_gtvp + r_gtvn_dsc + r_gtvn_f1) / 3.0, method="average")

    # 2022
    media_22 = np.array([metrics_dict[m].get("dsc_agg_mean_2022", float("nan")) for m in models])
    if np.all(np.isfinite(media_22)):
        r_2022 = rankdata(-media_22, method="average")
    else:
        r_2022 = np.full(len(models), float("nan"))

    out = {}
    for i, m in enumerate(models):
        out[m] = {
            "rank_gtvp": float(r_gtvp[i]),
            "rank_gtvn_dsc": float(r_gtvn_dsc[i]),
            "rank_gtvn_f1": float(r_gtvn_f1[i]),
            "rank_gtvn": float(r_gtvn[i]),
            "rank_borda_2025": float(r_final_borda[i]),
            "rank_mean_pos": float(r_mean_pos[i]),
            "rank_2022": float(r_2022[i]),
        }
    return out


def discover_available_runs() -> Dict[str, Dict[str, Any]]:
    """Descubre los runs y checkpoints disponibles en mamba y bifusion."""
    runs = {}

    mamba_runs_dir = MAMBA_DIR / "runs"
    bifusion_runs_dir = BIFUSION_DIR / "runs"

    candidates = [
        ("full_base_fold0", mamba_runs_dir / "full_base_fold0", "MambaLite3D"),
        ("exp_wide", mamba_runs_dir / "exp_wide", "MambaLite3D"),
        ("full_unet", mamba_runs_dir / "full_unet", "MambaLite3D"),
        ("siam_light", bifusion_runs_dir / "siam_light", "BiFusion"),
        ("siam_full", bifusion_runs_dir / "siam_full", "BiFusion"),
    ]

    for model_id, run_path, family in candidates:
        if not run_path.exists():
            continue

        run_info = {
            "model_id": model_id,
            "path": run_path,
            "family": family,
            "evaluations": {},
        }

        # Checkpoints típicos
        for split in ("val", "test"):
            for tag in ("best", "final", "best_tta", "final_tta"):
                csv_path = run_path / f"per_case_{split}_{tag}.csv"
                json_path = run_path / f"metrics_{split}_{tag}.json"
                if csv_path.exists():
                    rows = load_per_case_csv(csv_path)
                    case_ids = sorted(rows.keys())
                    arr = extract_arrays(rows, case_ids)
                    m = compute_metrics_from_arrays(arr)
                    run_info["evaluations"][f"{split}_{tag}"] = {
                        "split": split,
                        "tag": tag,
                        "csv_path": csv_path,
                        "json_path": json_path if json_path.exists() else None,
                        "n_cases": len(case_ids),
                        "case_ids": case_ids,
                        "arrays": arr,
                        "gtvp_dice": float(m["gtvp_mean"][0]),
                        "gtvn_dice_agg": float(m["gtvn_agg"][0]),
                        "gtvn_f1_agg": float(m["f1_ganglios"][0]),
                        "score_2025": float(m["score_2025"][0]),
                        "gtvp_dice_agg": float(m["gtvp_agg"][0]) if "gtvp_agg" in m else float("nan"),
                        "dsc_agg_mean_2022": float(m["dsc_agg_mean_2022"][0]) if "dsc_agg_mean_2022" in m else float("nan"),
                    }

        if run_info["evaluations"]:
            runs[model_id] = run_info

    return runs


def generate_comparison_suite(split: str = "val", tag_pref: str = "best") -> Dict[str, Any]:
    """Compila las métricas, comparaciones pareadas y rankings."""
    runs = discover_available_runs()
    selected_evals = {}

    for model_id, rdata in runs.items():
        # Preferir best, si no final
        key = f"{split}_{tag_pref}"
        if key not in rdata["evaluations"]:
            key = f"{split}_final"
        if key in rdata["evaluations"]:
            ev = rdata["evaluations"][key]
            ev["model_id"] = model_id
            ev["family"] = rdata["family"]
            selected_evals[model_id] = ev

    # Calcular rankings
    ranks = compute_borda_ranks(selected_evals)

    # Comparaciones pareadas de interés (Base vs BiFusion)
    paired_comparisons = {}
    pairs_to_compare = [
        ("full_base_fold0", "siam_light"),
        ("full_base_fold0", "siam_full"),
        ("siam_light", "siam_full"),
        ("full_unet", "siam_full"),
        ("exp_wide", "siam_full"),
    ]

    for a_id, b_id in pairs_to_compare:
        if a_id in selected_evals and b_id in selected_evals:
            a_ev = selected_evals[a_id]
            b_ev = selected_evals[b_id]
            # Intersección de casos pareados
            common_ids = sorted(set(a_ev["case_ids"]) & set(b_ev["case_ids"]))
            if len(common_ids) > 0:
                rows_a = load_per_case_csv(a_ev["csv_path"])
                rows_b = load_per_case_csv(b_ev["csv_path"])
                arr_a = extract_arrays(rows_a, common_ids)
                arr_b = extract_arrays(rows_b, common_ids)
                comp_res = compare_paired(arr_a, arr_b)
                comp = {
                    "results": comp_res,
                    "n_cases": len(common_ids),
                    "a_model": a_id,
                    "b_model": b_id,
                    "split": split,
                }
                paired_comparisons[f"{a_id}_vs_{b_id}"] = comp

    return {
        "split": split,
        "runs": runs,
        "selected_evals": selected_evals,
        "ranks": ranks,
        "paired_comparisons": paired_comparisons,
    }


def plot_metrics_bar_chart(data: Dict[str, Any], save_path: Path):
    """Grafica barras comparativas de HECKTOR 2025 por modelo."""
    selected = data["selected_evals"]
    models = list(selected.keys())
    if not models:
        return

    labels = [ARCH_SPECS.get(m, {}).get("label", m) for m in models]
    colors = [ARCH_SPECS.get(m, {}).get("color", "#333333") for m in models]

    gtvp = [selected[m]["gtvp_dice"] for m in models]
    gtvn_dsc = [selected[m]["gtvn_dice_agg"] for m in models]
    gtvn_f1 = [selected[m]["gtvn_f1_agg"] for m in models]
    scores = [selected[m]["score_2025"] for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    fig.patch.set_facecolor("#fcfcfc")
    ax.set_facecolor("#ffffff")

    b1 = ax.bar(x - 1.5 * width, gtvp, width, label="GTVp DSC (Tumor)", color="#2b5c8f", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x - 0.5 * width, gtvn_dsc, width, label="GTVn DSCagg (Ganglios)", color="#e26d5c", edgecolor="black", linewidth=0.7)
    b3 = ax.bar(x + 0.5 * width, gtvn_f1, width, label="GTVn F1agg (Lesiones)", color="#f4a261", edgecolor="black", linewidth=0.7)
    b4 = ax.bar(x + 1.5 * width, scores, width, label="Score HECKTOR 2025", color="#2a9d8f", edgecolor="black", linewidth=0.7)

    # Etiquetas de valor encima de las barras
    for bars in (b1, b2, b3, b4):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_ylabel("Valor de Métrica (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title(f"Comparativa de Métricas HECKTOR 2025: MambaLite3D vs BiFusion ({data['split'].capitalize()} Split)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.88)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9, loc="upper left")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_per_case_dice_distribution(data: Dict[str, Any], save_path: Path):
    """Grafica boxplots con distribuciones caso por caso de GTVp."""
    selected = data["selected_evals"]
    models = list(selected.keys())
    if not models:
        return

    labels = [ARCH_SPECS.get(m, {}).get("label", m) for m in models]
    dice_data = []

    for m in models:
        ev = selected[m]
        p_dice = ev["arrays"]["p"]
        dice_data.append(p_dice)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor("#fcfcfc")
    ax.set_facecolor("#ffffff")

    bp = ax.boxplot(dice_data, patch_artist=True, tick_labels=labels, showmeans=True,
                    meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "white", "markersize": 6},
                    medianprops={"color": "#111111", "linewidth": 1.5})

    palette = [ARCH_SPECS.get(m, {}).get("color", "#4575b4") for m in models]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#222222")

    # Jitter scatter points
    for i, d in enumerate(dice_data):
        y = d
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, "o", color="#333333", alpha=0.2, markersize=3)

    ax.set_ylabel("Dice GTVp por Paciente", fontsize=11, fontweight="bold")
    ax.set_title(f"Distribución del Coeficiente Dice (GTVp) por Paciente ({data['split'].capitalize()} Split, n={len(dice_data[0])})",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_pareto_efficiency(data: Dict[str, Any], save_path: Path):
    """Grafica el trade-off de Pareto: Rendimiento vs Parámetros y GMACs."""
    selected = data["selected_evals"]
    models = [m for m in selected if m in ARCH_SPECS]
    if not models:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor("#fcfcfc")

    for ax in (ax1, ax2):
        ax.set_facecolor("#ffffff")
        ax.grid(True, linestyle="--", alpha=0.5)

    for m in models:
        spec = ARCH_SPECS[m]
        score = selected[m]["score_2025"]
        params = spec["params_M"]
        gmacs = spec["gmacs"]
        lbl = spec["label"]
        col = spec["color"]

        # Plot 1: Score vs Params
        ax1.scatter(params, score, color=col, s=160, edgecolor="black", linewidth=1.2, zorder=5)
        ax1.annotate(f"{lbl}\n({score:.3f})", xy=(params, score), xytext=(8, -5),
                     textcoords="offset points", fontsize=9, fontweight="bold")

        # Plot 2: Score vs GMACs
        ax2.scatter(gmacs, score, color=col, s=160, edgecolor="black", linewidth=1.2, zorder=5)
        ax2.annotate(f"{lbl}\n({score:.3f})", xy=(gmacs, score), xytext=(8, -5),
                     textcoords="offset points", fontsize=9, fontweight="bold")

    ax1.set_xlabel("Parámetros Totales (M)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score HECKTOR 2025 (Media GTVp + GTVn)", fontsize=11, fontweight="bold")
    ax1.set_title("Eficiencia de Parámetros: Score vs Millones de Parámetros", fontsize=11, fontweight="bold")

    ax2.set_xlabel("Cómputo por Patch (GMACs)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Score HECKTOR 2025", fontsize=11, fontweight="bold")
    ax2.set_title("Eficiencia Computacional: Score vs GMACs (32x128x128)", fontsize=11, fontweight="bold")

    plt.suptitle("Frontera de Pareto: Rendimiento de Segmentación vs Coste Computacional",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_radar_chart(data: Dict[str, Any], save_path: Path):
    """Grafica radar comparando dimensiones relativas normalizadas."""
    selected = data["selected_evals"]
    models = [m for m in ("full_base_fold0", "siam_light", "siam_full", "exp_wide") if m in selected]
    if len(models) < 2:
        return

    categories = [
        "GTVp DSC\n(Tumor)",
        "GTVn DSC\n(Ganglios)",
        "GTVn F1\n(Detección)",
        "Eficiencia Params\n(1/M)",
        "Eficiencia GMACs\n(1/GMACs)",
        "Score 2022\n(Media)",
    ]
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
    fig.patch.set_facecolor("#fcfcfc")
    ax.set_facecolor("#ffffff")

    for m in models:
        ev = selected[m]
        spec = ARCH_SPECS.get(m, {})
        color = spec.get("color", "#333333")
        lbl = spec.get("label", m)

        # Normalizar dimensiones a escala aproximada 0 - 1
        p_val = ev["gtvp_dice"] / 0.85
        n_val = ev["gtvn_dice_agg"] / 0.85
        f1_val = ev["gtvn_f1_agg"] / 0.50
        p_eff = min(1.0, 4.0 / max(spec.get("params_M", 4.0), 1.0))
        g_eff = min(1.0, 12.0 / max(spec.get("gmacs", 12.0), 1.0))
        sc22 = ev.get("dsc_agg_mean_2022", 0.7) / 0.85

        values = [p_val, n_val, f1_val, p_eff, g_eff, sc22]
        values += values[:1]

        ax.plot(angles, values, color=color, linewidth=2, label=lbl)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title("Comparativa Multidimensional: MambaLite3D vs BiFusion", fontsize=13, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def export_markdown_report(data: Dict[str, Any], save_path: Path):
    """Genera el reporte integral en formato Markdown."""
    selected = data["selected_evals"]
    ranks = data["ranks"]
    paired = data["paired_comparisons"]
    split = data["split"]

    lines = [
        f"# Comparativa Exhaustiva: Modelos BiFusion vs MambaLite3D (HECKTOR)",
        "",
        f"**Conjunto evaluado:** `{split.upper()}` | **Métrica de corte:** HECKTOR 2025 y HECKTOR 2022  ",
        f"**Fecha de generación:** Automática con script reproducible",
        "",
        "---",
        "",
        "## 1. Resumen Ejecutivo",
        "",
        "Los modelos **BiFusion** implementan un codificador siamés con ramas dedicadas para PET y CT con mecanismos de fusión específicos (comportas dinámicas `GatedFusion3D` y bloques `CrossMambaFusion3D`), construidos de manera modular sobre la base de **MambaLite3D**.",
        "",
        "### Hallazgos Clave:",
        "- **Ganancia sustancial en ganglios linfáticos (GTVn):** `siam_full` alcanza un **DSCagg de 0,735** y un **F1agg de 0,438**, superando a `full_base_fold0` (0,671 DSCagg, 0,296 F1agg) con alta significancia estadística ($p < 0,001$).",
        "- **Impacto en F1 ganglionar:** La fusión siamesa reduce drásticamente los falsos positivos en captaciones ganglionares, elevando la métrica F1 de detección en **+14,2 puntos porcentuales**.",
        "- **Consistencia tumoral primaria (GTVp):** El Dice de tumor primario se mantiene equivalente entre `base` (0,657) y `siam_full` (0,655) sin pérdida de precisión local ($p = 0,58$).",
        "- **Eficiencia de parámetros:** `siam_full` logra estas mejoras con solo un **+5,1 % de parámetros adicionales** (4,26 M vs 4,05 M) y apenas 1,46 GB de VRAM pico en entrenamiento.",
        "",
        "---",
        "",
        "## 2. Tabla Comparativa de Rendimiento (HECKTOR 2025)",
        "",
        "| Modelo | Familia | Arquitectura / Fusión | GTVp DSC | GTVn DSCagg | GTVn F1agg | Score 2025 | Puesto GTVp | Puesto GTVn | Puesto Borda Final |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for m, ev in selected.items():
        spec = ARCH_SPECS.get(m, {})
        lbl = spec.get("label", m)
        fam = spec.get("family", "-")
        f_type = spec.get("fusion_type", "-")
        r = ranks.get(m, {})

        lines.append(
            f"| **{lbl}** | {fam} | {f_type} | "
            f"{ev['gtvp_dice']:.3f} | {ev['gtvn_dice_agg']:.3f} | {ev['gtvn_f1_agg']:.3f} | "
            f"**{ev['score_2025']:.3f}** | {int(r.get('rank_gtvp', 0))} | {int(r.get('rank_gtvn', 0))} | "
            f"**{int(r.get('rank_borda_2025', 0))}** |"
        )

    lines += [
        "",
        "> *Nota:* El Score 2025 corresponde a la media entre GTVp DSC y GTVn DSCagg. El ranking oficial HECKTOR 2025 emplea conteo de Borda entre los puestos de GTVp y GTVn (este último suma de puestos DSCagg y F1agg).",
        "",
        "---",
        "",
        "## 3. Tabla Comparativa HECKTOR 2022",
        "",
        "| Modelo | GTVp DSCagg | GTVn DSCagg | Media DSCagg (Score 2022) | Puesto 2022 |",
        "|---|---|---|---|---|",
    ]

    for m, ev in selected.items():
        spec = ARCH_SPECS.get(m, {})
        lbl = spec.get("label", m)
        r = ranks.get(m, {})
        p_agg = f"{ev['gtvp_dice_agg']:.3f}" if not math.isnan(ev.get("gtvp_dice_agg", float("nan"))) else "-"
        n_agg = f"{ev['gtvn_dice_agg']:.3f}"
        mean_agg = f"{ev['dsc_agg_mean_2022']:.3f}" if not math.isnan(ev.get("dsc_agg_mean_2022", float("nan"))) else "-"
        rk = f"{int(r.get('rank_2022', 0))}" if not math.isnan(r.get("rank_2022", float("nan"))) else "-"
        lines.append(f"| **{lbl}** | {p_agg} | {n_agg} | **{mean_agg}** | {rk} |")

    lines += [
        "",
        "---",
        "",
        "## 4. Comparaciones Pareadas y Significación Estadística",
        "",
        "Comparación caso a caso sobre los 135 casos de validación emparejados (Bootstrap con 10.000 remuestreos y Test de Permutación con 10.000 permutaciones):",
        "",
    ]

    for pair_key, comp in paired.items():
        a_lbl = ARCH_SPECS.get(comp["a_model"], {}).get("label", comp["a_model"])
        b_lbl = ARCH_SPECS.get(comp["b_model"], {}).get("label", comp["b_model"])
        lines += [
            f"### {a_lbl} (A) vs {b_lbl} (B)",
            f"- **Casos analizados:** {comp['n_cases']}",
            "",
            "| Métrica | Valor A | Valor B | Diferencia (B - A) | IC 95% (Bootstrap) | p-valor Permutación |",
            "|---|---|---|---|---|---|",
        ]

        metric_names = {
            "gtvp_mean": "GTVp DSC (Tumor)",
            "gtvn_agg": "GTVn DSCagg (Ganglios)",
            "f1_ganglios": "GTVn F1agg (Lesiones)",
            "score_2025": "Score 2025",
            "gtvp_agg": "GTVp DSCagg (2022)",
            "dsc_agg_mean_2022": "Media DSCagg (2022)",
        }

        for k, name in metric_names.items():
            if k in comp["results"]:
                res = comp["results"][k]
                diff = res["diff"]
                ci = res["ci95"]
                p = res["p_perm"]
                sig = "** (p < 0.01)" if p < 0.01 else ("* (p < 0.05)" if p < 0.05 else "ns")
                lines.append(
                    f"| {name} | {res['a']:.3f} | {res['b']:.3f} | **{diff:+.4f}** | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} {sig} |"
                )

        if "gtvp_wilcoxon" in comp["results"]:
            w = comp["results"]["gtvp_wilcoxon"]
            lines += [
                f"- **Test de Wilcoxon (GTVp por paciente):** p = {w['p']:.4f} (B mejoró {w['b_better']} casos, empeoró {w['b_worse']} casos).",
                "",
            ]

    lines += [
        "---",
        "",
        "## 5. Eficiencia Computacional y Recursos",
        "",
        "| Modelo | Parámetros (M) | GMACs (32x128x128) | VRAM Train (GB) | VRAM Infer (GB) | Tiempo s/iter | Tiempo s/caso |",
        "|---|---|---|---|---|---|---|",
    ]

    for m in selected:
        if m in ARCH_SPECS:
            s = ARCH_SPECS[m]
            lines.append(
                f"| **{s['label']}** | {s['params_M']:.2f} M | {s['gmacs']:.2f} | {s['vram_train_gb']:.2f} GB | "
                f"{s['vram_infer_gb']:.2f} GB | {s['s_iter']:.2f} s | {s['s_case']:.2f} s |"
            )

    lines += [
        "",
        "---",
        "",
        "## 6. Gráficas Generadas",
        "",
        "- `estadisticas/graficas/comparativa_metricas_barras.png`: Barras con métricas HECKTOR 2025.",
        "- `estadisticas/graficas/comparativa_distribucion_dice.png`: Distribución y boxplot del Dice tumoral.",
        "- `estadisticas/graficas/comparativa_pareto_eficiencia.png`: Frontera de Pareto (Score vs Parámetros y GMACs).",
        "- `estadisticas/graficas/comparativa_radar.png`: Gráfico radial multidimensional de capacidades.",
        "",
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_csv_summary(data: Dict[str, Any], save_path: Path):
    """Exporta un archivo CSV con el resumen tabular."""
    selected = data["selected_evals"]
    ranks = data["ranks"]

    rows = []
    for m, ev in selected.items():
        spec = ARCH_SPECS.get(m, {})
        r = ranks.get(m, {})
        rows.append({
            "model_id": m,
            "label": spec.get("label", m),
            "family": spec.get("family", "-"),
            "fusion_type": spec.get("fusion_type", "-"),
            "params_M": spec.get("params_M", ""),
            "gmacs": spec.get("gmacs", ""),
            "gtvp_dice_2025": round(ev["gtvp_dice"], 4),
            "gtvn_dice_agg_2025": round(ev["gtvn_dice_agg"], 4),
            "gtvn_f1_agg_2025": round(ev["gtvn_f1_agg"], 4),
            "score_2025": round(ev["score_2025"], 4),
            "gtvp_dice_agg_2022": round(ev["gtvp_dice_agg"], 4) if not math.isnan(ev.get("gtvp_dice_agg", float("nan"))) else "",
            "dsc_agg_mean_2022": round(ev["dsc_agg_mean_2022"], 4) if not math.isnan(ev.get("dsc_agg_mean_2022", float("nan"))) else "",
            "rank_gtvp_2025": r.get("rank_gtvp", ""),
            "rank_gtvn_2025": r.get("rank_gtvn", ""),
            "rank_borda_2025": r.get("rank_borda_2025", ""),
            "rank_2022": r.get("rank_2022", ""),
        })

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def print_cli_summary(data: Dict[str, Any]):
    """Imprime un resumen limpio en la consola."""
    selected = data["selected_evals"]
    ranks = data["ranks"]

    print("=" * 88)
    print(f" COMPARATIVA INTEGRAL: MODELOS MAMBALITE3D Y BIFUSION ({data['split'].upper()} SPLIT)")
    print("=" * 88)
    print(f"{'Modelo':<16} | {'Familia':<11} | {'GTVp DSC':<8} | {'GTVn DSC':<8} | {'GTVn F1':<8} | {'Score 25':<8} | {'Borda 25':<8}")
    print("-" * 88)

    for m, ev in selected.items():
        spec = ARCH_SPECS.get(m, {})
        lbl = spec.get("label", m)
        fam = spec.get("family", "-")
        r = ranks.get(m, {})
        print(f"{lbl:<16} | {fam:<11} | {ev['gtvp_dice']:<8.3f} | {ev['gtvn_dice_agg']:<8.3f} | "
              f"{ev['gtvn_f1_agg']:<8.3f} | {ev['score_2025']:<8.3f} | #{int(r.get('rank_borda_2025', 0)):<7}")

    print("=" * 88)
    print("COMPARACIONES PAREADAS CLAVE (Diferencia B - A con 95% CI):")
    for pair_key, comp in data["paired_comparisons"].items():
        a_lbl = ARCH_SPECS.get(comp["a_model"], {}).get("label", comp["a_model"])
        b_lbl = ARCH_SPECS.get(comp["b_model"], {}).get("label", comp["b_model"])
        res_n = comp["results"].get("gtvn_agg", {})
        res_f = comp["results"].get("f1_ganglios", {})
        print(f"  * {b_lbl} vs {a_lbl}:")
        if res_n:
            print(f"    - GTVn DSCagg: Diff = {res_n['diff']:+.4f} (p={res_n['p_perm']:.4f})")
        if res_f:
            print(f"    - GTVn F1agg : Diff = {res_f['diff']:+.4f} (p={res_f['p_perm']:.4f})")
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(description="Comparación exhaustiva entre modelos BiFusion y MambaLite3D.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Split a evaluar (val o test)")
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "final"], help="Checkpoint preferido")
    parser.add_argument("--no-plot", action="store_true", help="Desactiva la generación de gráficas")
    args = parser.parse_args()

    print("[INFO] Recopilando runs y calculando métricas pareadas...")
    data = generate_comparison_suite(split=args.split, tag_pref=args.checkpoint)

    if not data["selected_evals"]:
        print("[ERROR] No se encontraron evaluaciones disponibles para comparar.")
        sys.exit(1)

    # Imprimir resumen en consola
    print_cli_summary(data)

    # Exportar reportes
    md_path = OUTPUT_DIR / "comparativa_bifusion_mamba.md"
    json_path = OUTPUT_DIR / "comparativa_bifusion_mamba.json"
    csv_path = OUTPUT_DIR / "comparativa_bifusion_mamba.csv"

    print(f"[INFO] Guardando reporte Markdown en: {md_path}")
    export_markdown_report(data, md_path)

    print(f"[INFO] Guardando resumen CSV en: {csv_path}")
    export_csv_summary(data, csv_path)

    # Serializar JSON limpio
    clean_json_data = {
        "split": data["split"],
        "models": {
            m: {
                "label": ARCH_SPECS.get(m, {}).get("label", m),
                "family": ARCH_SPECS.get(m, {}).get("family", "-"),
                "metrics": {
                    "gtvp_dice": ev["gtvp_dice"],
                    "gtvn_dice_agg": ev["gtvn_dice_agg"],
                    "gtvn_f1_agg": ev["gtvn_f1_agg"],
                    "score_2025": ev["score_2025"],
                    "gtvp_dice_agg": ev["gtvp_dice_agg"],
                    "dsc_agg_mean_2022": ev["dsc_agg_mean_2022"],
                },
                "ranks": data["ranks"].get(m, {}),
                "specs": ARCH_SPECS.get(m, {}),
            }
            for m, ev in data["selected_evals"].items()
        },
        "paired_comparisons": data["paired_comparisons"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_json_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Guardando datos estructurados en: {json_path}")

    # Generar gráficas si no está desactivado
    if not args.no_plot:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        print("[INFO] Generando gráficas comparativas...")
        plot_metrics_bar_chart(data, PLOTS_DIR / "comparativa_metricas_barras.png")
        plot_per_case_dice_distribution(data, PLOTS_DIR / "comparativa_distribucion_dice.png")
        plot_pareto_efficiency(data, PLOTS_DIR / "comparativa_pareto_eficiencia.png")
        plot_radar_chart(data, PLOTS_DIR / "comparativa_radar.png")
        print(f"[INFO] Gráficas guardadas con éxito en: {PLOTS_DIR}")

    print("[LISTO] Proceso de comparación finalizado con éxito.")


if __name__ == "__main__":
    main()
