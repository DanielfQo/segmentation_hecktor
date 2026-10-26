#!/usr/bin/env python3
"""Histograma de intensidades de CT y PET por centro.

Parte de salidas/histogramas.csv (lo vuelca caracterizacion_dataset.py) y produce:

  figuras/fig1_histogramas_por_centro.png   distribucion de niveles por centro
  salidas/distancia_histogramas.csv         distancia de Wasserstein al histograma medio

Los bordes de los bins son fijos para todo el dataset (CT_BINS y PET_BINS en
descriptores.py), que es lo que permite promediar histogramas entre casos y
comparar centros.

Uso:
    analisis/.venv/bin/python analisis/figura_histogramas.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from descriptores import CT_BINS, PET_BINS  # noqa: E402
from estilo import REJILLA, SERIE_1, TINTA, aplicar_estilo, limpiar  # noqa: E402

BASE = Path(__file__).resolve().parent


def fig_histogramas(hist: pd.DataFrame, fig_dir: Path, salidas: Path) -> pd.DataFrame:
    """Crestas: histograma medio de cada centro, normalizado. CT y PET."""
    centros = sorted(hist["centro"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.62 * len(centros) + 2.4))
    distancias = []

    for ax, (pref, bins, titulo, unidad, xlim) in zip(axes, [
            ("ct_", CT_BINS, "CT: distribucion de niveles",
             "HU (el ultimo bin acumula todo >= 1000)", None),
            ("pet_", PET_BINS, "PET: distribucion de niveles",
             "SUV (vista 0-8; por encima queda <1% del area)", (0, 8))]):
        cols = [c for c in hist.columns if c.startswith(pref)]
        M = hist[cols].to_numpy(dtype=float)
        M = M / np.maximum(M.sum(axis=1, keepdims=True), 1)   # a densidad
        centros_x = (bins[:-1] + bins[1:]) / 2
        medio_global = M.mean(axis=0)
        alto = 1.0
        for i, c in enumerate(centros):
            sel = hist["centro"].to_numpy() == c
            medio = M[sel].mean(axis=0)
            y = i * alto
            esc = 0.9 * alto / max(medio.max(), 1e-9)
            ax.fill_between(centros_x, y, y + medio * esc, color=SERIE_1,
                            alpha=0.75, linewidth=0)
            ax.plot(centros_x, y + medio * esc, color=TINTA, linewidth=0.8)
            ax.axhline(y, color=REJILLA, linewidth=0.6, zorder=0)
            for j in np.flatnonzero(sel):
                distancias.append({
                    "caso": hist["caso"].iloc[j], "centro": c,
                    "modalidad": pref.rstrip("_"),
                    "wasserstein_al_medio": stats.wasserstein_distance(
                        centros_x, centros_x, M[j], medio_global)})
        # La identidad del centro la lleva la etiqueta del eje, no el color.
        ax.set_yticks([i * alto for i in range(len(centros))],
                      [f"{c}  n={int((hist['centro'] == c).sum())}"
                       for c in centros], fontsize=8)
        ax.set_ylim(-0.15, len(centros) * alto + 0.2)
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_xlabel(unidad)
        ax.set_title(titulo)
        limpiar(ax, rejilla="x")

    fig.suptitle("Distribucion de niveles de intensidad por centro "
                 "(histograma medio, area normalizada)", y=1.0,
                 fontsize=11, color=TINTA, fontweight="bold")
    fig.savefig(fig_dir / "fig1_histogramas_por_centro.png")
    plt.close(fig)

    dh = pd.DataFrame(distancias)
    dh.to_csv(salidas / "distancia_histogramas.csv", index=False)
    return dh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salidas", type=Path, default=BASE / "salidas")
    ap.add_argument("--figuras", type=Path, default=BASE / "figuras")
    args = ap.parse_args()

    aplicar_estilo()
    args.figuras.mkdir(parents=True, exist_ok=True)
    ruta = args.salidas / "histogramas.csv"
    if not ruta.exists():
        print(f"ERROR: falta {ruta}. Ejecuta primero caracterizacion_dataset.py",
              file=sys.stderr)
        return 1

    hist = pd.read_csv(ruta).sort_values("caso").reset_index(drop=True)

    # Si esta la caracterizacion al lado, nos quedamos con sus mismos casos:
    # asi la figura no incluye los que fallaron al procesarse.
    ruta_c = args.salidas / "caracterizacion.csv"
    if ruta_c.exists():
        df = pd.read_csv(ruta_c)
        con_error = df["error"].notna() & (df["error"].astype(str).str.len() > 0)
        validos = set(df.loc[~con_error, "caso"])
        antes = len(hist)
        hist = hist[hist["caso"].isin(validos)].reset_index(drop=True)
        if len(hist) != antes:
            print(f"Aviso: {antes - len(hist)} casos excluidos por la caracterizacion")

    print(f"{len(hist)} casos, {hist['centro'].nunique()} centros")
    dh = fig_histogramas(hist, args.figuras, args.salidas)
    print("fig1 lista")

    # Cuanto mayor la distancia, mas atipico es el centro. Las dos modalidades
    # estan en sus unidades nativas (HU y SUV), asi que no son comparables
    # entre columnas, solo entre centros dentro de una columna.
    resumen = (dh.groupby(["modalidad", "centro"])["wasserstein_al_medio"]
               .mean().unstack(0))
    print("\nDistancia de Wasserstein media al histograma medio del dataset:")
    print(resumen.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
