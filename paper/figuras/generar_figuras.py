"""Genera las figuras del paper a partir de los resultados en modelos/mamba/runs.

Uso (desde cualquier carpeta):
    python paper/figuras/generar_figuras.py
Salida: paper/figuras/entrenamiento.pdf y paper/figuras/dsc_por_centro.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RUNS = AQUI.parents[1] / "modelos" / "mamba" / "runs"
MODELOS = {"Base (4.0 M)": "full_base_fold0", "MambaLite3D (15.8 M)": "exp_wide"}
COLOR = {"Base (4.0 M)": "#2a78d6", "MambaLite3D (15.8 M)": "#eb6834"}
MARCA = {"Base (4.0 M)": "o", "MambaLite3D (15.8 M)": "s"}
TINTA, TINTA_2, REJILLA = "#1f1f1e", "#5f5e58", "#e4e3de"
ANCHO_COL = 3.5  # pulgadas, una columna IEEE

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": TINTA_2,
    "axes.labelcolor": TINTA,
    "xtick.color": TINTA_2,
    "ytick.color": TINTA_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
})


def media_movil(x: np.ndarray, w: int = 10) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


def figura_entrenamiento() -> None:
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(ANCHO_COL, 3.6), constrained_layout=True)

    # (a) pérdida de entrenamiento por época: mismos 250 x 250 iteraciones en ambos modelos
    for nombre, run in MODELOS.items():
        hist = json.load(open(RUNS / run / "history.json"))
        ep = np.array([h["epoch"] for h in hist]) + 1
        loss = np.array([h["loss"] for h in hist])
        ax_a.plot(ep, loss, color=COLOR[nombre], lw=0.5, alpha=0.3)
        ax_a.plot(ep, media_movil(loss), color=COLOR[nombre], lw=1.5, label=nombre)
    ax_a.set_xlabel("Época")
    ax_a.set_ylabel("Pérdida DiceCE")
    ax_a.set_xlim(0, 250)
    ax_a.legend(frameon=False, loc="upper right")
    ax_a.set_title("(a) Pérdida de entrenamiento", loc="left")

    # (b) validación de MambaLite3D cada 25 épocas, 135 casos, sin postprocesado
    run = RUNS / MODELOS["MambaLite3D (15.8 M)"]
    epocas = list(range(25, 250, 25))
    filas = [json.load(open(run / f"metrics_ep{e}.json"))["hecktor2025"] for e in epocas]
    final = json.load(open(run / "metrics_val_final.json"))["hecktor2025"]
    epocas.append(250)
    filas.append(final)
    color = COLOR["MambaLite3D (15.8 M)"]
    series = [
        ("GTVp_DSC", "GTVp DSC", "-", "o"),
        ("GTVn_DSCagg", "GTVn DSCagg", "--", "s"),
        ("GTVn_F1agg", "GTVn F1agg", ":", "^"),
    ]
    for clave, etiqueta, estilo, marca in series:
        y = [f[clave] for f in filas]
        ax_b.plot(epocas, y, color=color, ls=estilo, lw=1.5, marker=marca, ms=3.5,
                  mfc="white", mec=color, mew=1.0, label=etiqueta)
        ax_b.annotate(etiqueta, (epocas[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                      va="center", fontsize=7, color=TINTA)
    ax_b.set_xlabel("Época")
    ax_b.set_ylabel("Métrica (validación)")
    ax_b.set_xlim(0, 290)
    ax_b.set_xticks(range(0, 251, 50))
    ax_b.set_ylim(0, 0.8)
    ax_b.set_title("(b) Validación de MambaLite3D", loc="left")

    fig.savefig(AQUI / "entrenamiento.pdf")
    fig.savefig(AQUI / "entrenamiento.png", dpi=200)
    plt.close(fig)


def figura_dsc_por_centro() -> None:
    datos = []
    for nombre, run in MODELOS.items():
        df = pd.read_csv(RUNS / run / "per_case_test_final_pp.csv")
        df["modelo"] = nombre
        df["centro"] = df["case_id"].str.split("-").str[0]
        datos.append(df)
        oficial = json.load(open(RUNS / run / "metrics_test_final_pp.json"))["hecktor2025"]["GTVp_DSC"]
        print(f"{nombre}: media GTVp DSC por caso {df['dice_gtvp'].mean():.4f} (metrics json {oficial:.4f})")
    datos = pd.concat(datos)

    centros = sorted(datos["centro"].unique())
    n_por_centro = datos[datos["modelo"] == next(iter(MODELOS))].groupby("centro").size()
    grupos = centros + ["Todos"]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(ANCHO_COL, 2.3), constrained_layout=True)
    desplaz = {"Base (4.0 M)": -0.18, "MambaLite3D (15.8 M)": 0.18}
    for i, grupo in enumerate(grupos):
        for nombre in MODELOS:
            sel = datos[datos["modelo"] == nombre]
            if grupo != "Todos":
                sel = sel[sel["centro"] == grupo]
            y = sel["dice_gtvp"].to_numpy()
            x = i + desplaz[nombre] + rng.uniform(-0.07, 0.07, size=len(y))
            ax.scatter(x, y, s=9, color=COLOR[nombre], marker=MARCA[nombre], alpha=0.75,
                       edgecolors="white", linewidths=0.4,
                       label=nombre if i == 0 else None, zorder=3)
            ax.hlines(np.median(y), i + desplaz[nombre] - 0.14, i + desplaz[nombre] + 0.14,
                      color=TINTA, lw=1.2, zorder=4)
    etiquetas = [f"{c}\n(n={n_por_centro[c]})" for c in centros] + [f"Todos\n(n={int(n_por_centro.sum())})"]
    ax.set_xticks(range(len(grupos)))
    ax.set_xticklabels(etiquetas)
    ax.axvline(len(centros) - 0.5, color=TINTA_2, lw=0.6)
    ax.set_ylabel("GTVp DSC por caso")
    ax.set_ylim(-0.04, 1.02)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0), handletextpad=0.2)

    fig.savefig(AQUI / "dsc_por_centro.pdf")
    fig.savefig(AQUI / "dsc_por_centro.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    figura_entrenamiento()
    figura_dsc_por_centro()
    print("figuras guardadas en", AQUI)
