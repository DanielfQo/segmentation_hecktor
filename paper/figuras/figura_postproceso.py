"""Figura cualitativa: predicción de MambaLite3D sin y con postprocesado en casos de test.

Corre la inferencia (checkpoint final de runs/exp_wide) sobre unos pocos casos, aplica las reglas de
postprocess_final.json (las mismas que tools/evaluate) y dibuja un corte axial por caso.

Uso (entorno mambalite3d_cuda o mambalite3d):
    python paper/figuras/figura_postproceso.py
Salida: paper/figuras/postproceso.pdf y .png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

AQUI = Path(__file__).resolve().parent
MAMBA = AQUI.parents[1] / "modelos" / "mamba"
sys.path.insert(0, str(MAMBA))

from mambalite3d.config import resolve  # noqa: E402
from mambalite3d.data.io import find_cases, load_case  # noqa: E402
from mambalite3d.eval.components import postprocess_prediction  # noqa: E402
from mambalite3d.tools.evaluate import load_member  # noqa: E402
from mambalite3d.training.inference import sliding_window_predict  # noqa: E402

RUN = MAMBA / "runs" / "exp_wide"
CASOS = ["MDA-161", "MDA-357", "HGJ-065"]
COLOR_GTVP, COLOR_GTVN = "#eb6834", "#1baf7a"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "pdf.fonttype": 42})


def elegir_corte(gt: np.ndarray, raw: np.ndarray, pp: np.ndarray) -> int:
    """Corte axial con más vóxeles eliminados por el postprocesado, entre los que contienen referencia."""
    quitado = (raw != pp).sum(axis=(1, 2)).astype(float)
    con_ref = (gt > 0).sum(axis=(1, 2)) > 0
    puntaje = quitado + 0.2 * (gt > 0).sum(axis=(1, 2))
    if con_ref.any():
        puntaje[~con_ref] = -1
    return int(np.argmax(puntaje))


def recorte(*mascaras: np.ndarray, margen: int = 25, forma=None):
    union = np.zeros_like(mascaras[0], dtype=bool)
    for m in mascaras:
        union |= m > 0
    ys, xs = np.nonzero(union)
    h, w = forma
    if len(ys) == 0:
        return slice(0, h), slice(0, w)
    y0, y1 = max(ys.min() - margen, 0), min(ys.max() + margen + 1, h)
    x0, x1 = max(xs.min() - margen, 0), min(xs.max() + margen + 1, w)
    return slice(y0, y1), slice(x0, x1)


def superponer(ax, ct: np.ndarray, etiqueta: np.ndarray) -> None:
    ax.imshow(ct, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for clase, color in ((1, COLOR_GTVP), (2, COLOR_GTVN)):
        m = (etiqueta == clase).astype(float)
        if m.any():
            ax.imshow(np.ma.masked_where(m == 0, m), cmap=ListedColormap([color]), alpha=0.45,
                      interpolation="nearest")
            ax.contour(m, levels=[0.5], colors=[color], linewidths=0.8)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_member(RUN, "final", device)
    reglas = json.loads((RUN / "postprocess_final.json").read_text(encoding="utf-8"))["rules_best_dice"]
    patch = [int(p) for p in cfg["model"]["patch_size_zyx"]]
    k = cfg["train"]["num_classes"]
    archivos = {c.case_id: c for c in find_cases(resolve(cfg["data"]["root"]))}
    raw_csv = pd.read_csv(RUN / "per_case_test_final.csv").set_index("case_id")
    pp_csv = pd.read_csv(RUN / "per_case_test_final_pp.csv").set_index("case_id")

    fig, axes = plt.subplots(len(CASOS), 4, figsize=(7.16, 1.95 * len(CASOS)), constrained_layout=True)
    for fila, caso_id in enumerate(CASOS):
        caso = load_case(archivos[caso_id], cfg["data"])
        probs = sliding_window_predict(model, caso.image.astype(np.float32), patch, k, device,
                                       step_fraction=cfg["infer"]["step_fraction"], amp=cfg["train"]["amp"],
                                       mirror=False, pad_values=caso.pad_values,
                                       batch_size=cfg["infer"].get("batch_size", 4))
        raw = probs.argmax(0).astype(np.uint8)
        pp = postprocess_prediction(raw, probs, reglas, float(np.prod(caso.spacing_xyz)))
        z = elegir_corte(caso.label, raw, pp)
        ct, pet = caso.image[0, z].astype(float), caso.image[1, z].astype(float)
        sy, sx = recorte(caso.label[z], raw[z], forma=ct.shape)

        ax = axes[fila, 0]
        ax.imshow(ct[sy, sx], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        pet_c = pet[sy, sx]
        ax.imshow(pet_c, cmap="hot", alpha=np.clip((pet_c - pet_c.min()) / (np.ptp(pet_c) + 1e-6), 0, 1) * 0.8,
                  interpolation="bilinear")
        ax.set_title(f"{caso_id}: CT + PET (corte {z})")

        r, p = raw_csv.loc[caso_id], pp_csv.loc[caso_id]
        paneles = [
            (caso.label, "Referencia"),
            (raw, f"Sin PP: GTVp {r.dice_gtvp:.2f}, GTVn {r.dice_gtvn_case:.2f}\nFP de ganglios: {int(r.gtvn_fp)}"),
            (pp, f"Con PP: GTVp {p.dice_gtvp:.2f}, GTVn {p.dice_gtvn_case:.2f}\nFP de ganglios: {int(p.gtvn_fp)}"),
        ]
        for col, (vol, titulo) in enumerate(paneles, start=1):
            superponer(axes[fila, col], ct[sy, sx], vol[z][sy, sx])
            axes[fila, col].set_title(titulo)
        for ax in axes[fila]:
            ax.set_xticks([])
            ax.set_yticks([])
        print(f"{caso_id}: corte {z}, vóxeles eliminados en el corte {(raw[z] != pp[z]).sum()}, "
              f"en el volumen {(raw != pp).sum()}")

    fig.legend(handles=[Patch(color=COLOR_GTVP, alpha=0.7, label="GTVp"),
                        Patch(color=COLOR_GTVN, alpha=0.7, label="GTVn")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(AQUI / "postproceso.pdf", bbox_inches="tight")
    fig.savefig(AQUI / "postproceso.png", dpi=200, bbox_inches="tight")
    print("figura guardada en", AQUI / "postproceso.pdf")


if __name__ == "__main__":
    main()
