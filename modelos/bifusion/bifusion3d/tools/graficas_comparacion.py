"""Graficas de la comparacion de modelos en validacion (per_case_val_<ckpt>.csv de cada run). Solo CPU.

    python -m bifusion3d.tools.graficas_comparacion --out runs/graficas

Genera:
  metricas.png      GTVp DSC, GTVn DSCagg y F1 de cada modelo con IC del 95 % (bootstrap por paciente)
  diferencias.png   siam_full menos cada modelo, con IC del 95 % (bootstrap pareado)
  fp_por_caso.png   falsos positivos de ganglios por caso
  tamano_vs_f1.png  parametros frente a F1 de ganglios
  resumen.csv       las cifras de las graficas
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

MODELOS = [  # nombre, carpeta del run (relativa a modelos/bifusion), parametros en millones
    ("base", "../mamba/runs/full_base_fold0", 4.05),
    ("wide", "../mamba/runs/exp_wide", 15.8),
    ("U-Net", "../mamba/runs/full_unet", 44.6),
    ("siam_light", "runs/siam_light", 4.05),
    ("siam_full", "runs/siam_full", 4.26),
]
METRICAS = ("GTVp DSC", "GTVn DSCagg", "GTVn F1")
COLORES = {"base": "#8c8c8c", "wide": "#b0a27a", "U-Net": "#5b84b1", "siam_light": "#9fc5a8", "siam_full": "#2e7d4f"}


def leer(run: Path, ckpt: str) -> dict:
    with open(run / f"per_case_val_{ckpt}.csv", encoding="utf-8") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def metricas(datos: dict, ids, idx: np.ndarray) -> np.ndarray:
    """(3, B): GTVp DSC medio, DSCagg y F1 agregada para cada remuestreo de pacientes."""
    a = lambda k: np.array([float(datos[i][k]) for i in ids])[idx]  # noqa: E731
    inter, suma = a("gtvn_inter_mm3").sum(1), a("gtvn_sum_mm3").sum(1)
    tp, fp, fn = a("gtvn_tp").sum(1), a("gtvn_fp").sum(1), a("gtvn_fn").sum(1)
    return np.stack([a("dice_gtvp").mean(1), 2 * inter / np.maximum(suma, 1e-9), 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/graficas")
    ap.add_argument("--checkpoint", default="final", choices=["final", "best"])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--raiz", default=".", help="carpeta modelos/bifusion")
    args = ap.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raiz, out = Path(args.raiz), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datos = {n: leer(raiz / p, args.checkpoint) for n, p, _ in MODELOS}
    ids = sorted(set.intersection(*[set(d) for d in datos.values()]))
    rng = np.random.default_rng(0)
    idx = np.concatenate([np.arange(len(ids))[None], rng.integers(0, len(ids), (args.boot, len(ids)))])
    M = {n: metricas(datos[n], ids, idx) for n, _, _ in MODELOS}
    nombres = [n for n, _, _ in MODELOS]
    ic = lambda v: (np.percentile(v[1:], 2.5), np.percentile(v[1:], 97.5))  # noqa: E731
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1. metricas con IC
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for j, (ax, m) in enumerate(zip(axes, METRICAS)):
        vals = [M[n][j, 0] for n in nombres]
        err = np.array([[M[n][j, 0] - ic(M[n][j])[0], ic(M[n][j])[1] - M[n][j, 0]] for n in nombres]).T
        ax.bar(nombres, vals, color=[COLORES[n] for n in nombres], yerr=err, capsize=3)
        for x, v in enumerate(vals):
            ax.text(x, v + err[1][x] + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_title(m)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Validacion interna ({len(ids)} casos), checkpoint {args.checkpoint}; IC 95 % por bootstrap")
    fig.tight_layout()
    fig.savefig(out / "metricas.png", dpi=200)
    plt.close(fig)

    # 2. diferencias pareadas
    otros = [n for n in nombres if n != "siam_full"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    filas = []
    for j, (ax, m) in enumerate(zip(axes, METRICAS)):
        for y, n in enumerate(otros):
            d = M["siam_full"][j] - M[n][j]
            lo, hi = ic(d)
            sig = lo > 0 or hi < 0
            ax.errorbar(d[0], y, xerr=[[d[0] - lo], [hi - d[0]]], fmt="o", capsize=3,
                        color="#2e7d4f" if sig else "#8c8c8c")
            filas.append([m, n, round(d[0], 4), round(lo, 4), round(hi, 4)])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(m)
        ax.set_xlabel("siam_full menos el modelo")
    axes[0].set_yticks(range(len(otros)), otros)
    fig.suptitle("Diferencia pareada por paciente con IC del 95 % (verde: significativa)")
    fig.tight_layout()
    fig.savefig(out / "diferencias.png", dpi=200)
    plt.close(fig)

    # 3. FP por caso
    fig, ax = plt.subplots(figsize=(6, 3.8))
    fp = [[float(datos[n][i]["gtvn_fp"]) for i in ids] for n in nombres]
    bp = ax.boxplot(fp, showfliers=False, patch_artist=True, showmeans=True)
    ax.set_xticks(range(1, len(nombres) + 1), nombres, rotation=30)
    for patch, n in zip(bp["boxes"], nombres):
        patch.set_facecolor(COLORES[n])
    ax.set_ylabel("Falsos positivos de ganglios por caso")
    ax.set_title("Falsos positivos de GTVn (triangulo: media)")
    fig.tight_layout()
    fig.savefig(out / "fp_por_caso.png", dpi=200)
    plt.close(fig)

    # 4. tamano frente a F1
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for n, _, p in MODELOS:
        ax.scatter(p, M[n][2, 0], s=60, color=COLORES[n], zorder=3)
        ax.annotate(n, (p, M[n][2, 0]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Parametros (millones, escala log)")
    ax.set_ylabel("F1 de ganglios")
    ax.set_title("Tamano del modelo frente a deteccion de ganglios")
    fig.tight_layout()
    fig.savefig(out / "tamano_vs_f1.png", dpi=200)
    plt.close(fig)

    with open(out / "resumen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "parametros_M", *METRICAS, "fp_por_caso"])
        for (n, _, p), f_ in zip(MODELOS, fp):
            w.writerow([n, p, *[round(M[n][j, 0], 4) for j in range(3)], round(float(np.mean(f_)), 2)])
        w.writerow([])
        w.writerow(["metrica", "frente_a", "diferencia_siam_full", "ic95_lo", "ic95_hi"])
        w.writerows(filas)
    print(f"graficas en {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
