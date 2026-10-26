#!/usr/bin/env python3
"""Etapa 1 - Caracterizacion del dataset: una fila de descriptores por caso.

Recorre HECKTOR2025_raw, calcula los descriptores de `descriptores.py` y vuelca:

  salidas/caracterizacion.csv   un caso por fila, ~70 columnas
  salidas/histogramas.csv       histograma de 64 bins de CT y PET por caso

Escribe de forma incremental, asi que se puede interrumpir y reanudar con
--continuar sin perder lo ya calculado.

Uso tipico:
    analisis/.venv/bin/python analisis/caracterizacion_dataset.py -p 6
    analisis/.venv/bin/python analisis/caracterizacion_dataset.py --limite 10   # prueba
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from descriptores import CT_BINS, PET_BINS, caracterizar_caso, centro_de  # noqa: E402

RAIZ_DEFECTO = Path(__file__).resolve().parent.parent / "HECKTOR2025_raw"
SALIDA_DEFECTO = Path(__file__).resolve().parent / "salidas"


def descubrir_casos(raiz: Path) -> list[tuple[str, Path]]:
    casos = []
    for d in sorted(raiz.iterdir()):
        if not d.is_dir():
            continue
        caso = d.name
        if not (d / f"{caso}__CT.nii.gz").exists():
            continue
        if not (d / f"{caso}__PT.nii.gz").exists() or not (d / f"{caso}.nii.gz").exists():
            continue
        casos.append((caso, d))
    return casos


def _trabajo(args):
    caso, directorio, margen_mm, anillo_mm = args
    t0 = time.time()
    try:
        res = caracterizar_caso(caso, directorio, margen_mm, anillo_mm)
    except Exception:
        res = {"fila": {"caso": caso, "centro": centro_de(caso),
                        "error": traceback.format_exc(limit=1).replace("\n", " ")},
               "hist_ct": np.zeros(len(CT_BINS) - 1),
               "hist_pet": np.zeros(len(PET_BINS) - 1)}
    res["fila"]["segundos"] = round(time.time() - t0, 2)
    return res


def _casos_ya_hechos(ruta: Path) -> set[str]:
    if not ruta.exists():
        return set()
    with ruta.open(newline="") as f:
        return {r["caso"] for r in csv.DictReader(f) if r.get("caso")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=RAIZ_DEFECTO,
                    help="carpeta con un subdirectorio por caso")
    ap.add_argument("--salida", type=Path, default=SALIDA_DEFECTO)
    ap.add_argument("-p", "--procesos", type=int, default=6,
                    help="procesos en paralelo (cada uno carga un CT completo)")
    ap.add_argument("--limite", type=int, default=0, help="procesar solo los N primeros")
    ap.add_argument("--casos", nargs="*", default=None, help="IDs concretos a procesar")
    ap.add_argument("--margen-mm", type=float, default=30.0,
                    help="margen alrededor del tumor para el recorte de trabajo")
    ap.add_argument("--anillo-mm", type=float, default=10.0,
                    help="grosor del anillo peritumoral usado como fondo")
    ap.add_argument("--continuar", action="store_true",
                    help="saltar los casos que ya estan en el CSV de salida")
    args = ap.parse_args()

    if not args.raiz.is_dir():
        print(f"ERROR: no existe {args.raiz}", file=sys.stderr)
        return 1

    casos = descubrir_casos(args.raiz)
    if args.casos:
        pedidos = set(args.casos)
        casos = [c for c in casos if c[0] in pedidos]
    if args.limite:
        casos = casos[: args.limite]

    args.salida.mkdir(parents=True, exist_ok=True)
    ruta_csv = args.salida / "caracterizacion.csv"
    ruta_hist = args.salida / "histogramas.csv"

    hechos = _casos_ya_hechos(ruta_csv) if args.continuar else set()
    if hechos:
        antes = len(casos)
        casos = [c for c in casos if c[0] not in hechos]
        print(f"--continuar: {antes - len(casos)} casos ya estaban hechos")
    if not casos:
        print("Nada que hacer.")
        return 0

    modo = "a" if (args.continuar and ruta_csv.exists()) else "w"
    print(f"{len(casos)} casos | {args.procesos} procesos | salida {ruta_csv}")

    tareas = [(c, str(d), args.margen_mm, args.anillo_mm) for c, d in casos]
    t0 = time.time()
    escritor = None
    f_csv = ruta_csv.open(modo, newline="")
    f_hist = ruta_hist.open(modo, newline="")
    w_hist = csv.writer(f_hist)
    if modo == "w":
        w_hist.writerow(["caso", "centro"]
                        + [f"ct_{i:02d}" for i in range(len(CT_BINS) - 1)]
                        + [f"pet_{i:02d}" for i in range(len(PET_BINS) - 1)])
    n_err = 0
    try:
        with Pool(processes=max(1, args.procesos)) as pool:
            for i, res in enumerate(pool.imap_unordered(_trabajo, tareas), start=1):
                fila = res["fila"]
                if escritor is None:
                    escritor = csv.DictWriter(f_csv, fieldnames=list(fila),
                                              extrasaction="ignore")
                    if modo == "w":
                        escritor.writeheader()
                escritor.writerow(fila)
                w_hist.writerow([fila["caso"], fila["centro"]]
                                + [int(v) for v in res["hist_ct"]]
                                + [int(v) for v in res["hist_pet"]])
                f_csv.flush()
                f_hist.flush()
                if fila.get("error"):
                    n_err += 1
                    print(f"  [{i}/{len(tareas)}] {fila['caso']} ERROR: "
                          f"{fila['error'][:120]}", flush=True)
                elif i % 20 == 0 or i == len(tareas):
                    tr = (time.time() - t0) / i * (len(tareas) - i)
                    print(f"  [{i}/{len(tareas)}] {fila['caso']} "
                          f"({fila['segundos']}s/caso) restan ~{tr/60:.1f} min",
                          flush=True)
    finally:
        f_csv.close()
        f_hist.close()

    print(f"\nListo en {(time.time()-t0)/60:.1f} min. Errores: {n_err}")
    print(f"  {ruta_csv}\n  {ruta_hist}")
    if modo == "a":
        print("Nota: con --continuar las filas se anexan; el orden no es alfabetico.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
