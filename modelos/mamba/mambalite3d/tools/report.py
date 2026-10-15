"""Results table with the HECKTOR 2025 and HECKTOR 2022 metrics of several models, with each edition's ranking.

    python -m mambalite3d.tools.report --models base=runs/full_base_fold0 wide=runs/exp_wide unet=runs/full_unet
    python -m mambalite3d.tools.report --models base=runs/full_base_fold0:best wide=runs/exp_wide:final --select 2022

Each model is NAME=RUN (the checkpoint, final or best, is chosen on validation) or NAME=RUN:TAG (fixed tag, e.g.
best, final, best_tta, ens). The choice uses the selection score of --select: 2025 = mean of GTVp DSC and GTVn
DSCagg (the official 2025 ranking has no single score), 2022 = mean aggregated Dice. The validation and test
results of the chosen checkpoint are shown with the metrics of both editions and ranked among the listed models:
2025 = Borda count of a GTVp rank and a GTVn rank (itself the Borda count of the DSCagg and F1agg ranks);
2022 = mean aggregated Dice. Writes runs/resultados.md and runs/resultados.json (or --out).
Evaluations made before the 2022 metric existed have no GTVp volumes: their 2022 values show as "-" until the
run is evaluated again with tools/evaluate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from ..config import resolve
from ..eval.hecktor import fmt, fmt_rank, hecktor2022, hecktor2025, ranking_2022, ranking_2025, selection_score


def parse_model(spec: str):
    name, eq, rest = spec.partition("=")
    if not eq or not name or not rest:
        raise SystemExit(f"modelo invalido {spec!r}: usa NOMBRE=RUN o NOMBRE=RUN:TAG")
    run, colon, tag = rest.rpartition(":")
    if not colon or "/" in tag or "\\" in tag or not run or run.endswith(("/", "\\")) or len(run) == 1:
        run, tag = rest, None  # no tag (or a Windows drive letter such as C:)
    path = resolve(run) if not Path(run).is_absolute() else Path(run)
    return name, path, tag


def load_agg(run: Path, split: str, tag: str):
    path = run / f"metrics_{split}_{tag}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get("original_space") or d.get("roi_space")


def choose(run: Path, tag, edition: str):
    """Returns (tag, note): the fixed tag, or final/best with the higher validation selection score."""
    if tag:
        return tag, "fijo"
    scores = {}
    for t in ("final", "best"):
        agg = load_agg(run, "val", t)
        if agg is not None:
            scores[t] = selection_score(agg, edition)
    valid = {t: s for t, s in scores.items() if not math.isnan(s)}
    if valid:
        best = max(valid, key=valid.get)
        return best, f"validacion {edition}"
    if scores:  # e.g. 2022 requested on old evaluations without GTVp volumes
        alt = {t: selection_score(load_agg(run, "val", t), "2025") for t in scores}
        return max(alt, key=alt.get), "validacion 2025 (falta la 2022)"
    return None, "sin evaluar todavia"


def build(models, edition: str) -> dict:
    out = {"seleccion": edition, "modelos": {}, "val": {}, "test": {}}
    for name, run, tag in models:
        chosen, note = choose(run, tag, edition)
        out["modelos"][name] = {"run": str(run), "checkpoint": chosen, "eleccion": note}
        if chosen is None:
            continue
        for split in ("val", "test"):
            agg = load_agg(run, split, chosen)
            if agg is not None:
                out[split][name] = {"checkpoint": chosen, "n_casos": agg.get("n_cases"),
                                    "hecktor2025": hecktor2025(agg), "hecktor2022": hecktor2022(agg)}
    for split in ("val", "test"):
        rows = out[split]
        if rows:
            r25 = ranking_2025({n: v["hecktor2025"] for n, v in rows.items()})
            r22 = ranking_2022({n: v["hecktor2022"] for n, v in rows.items()})
            for n in rows:
                rows[n]["puestos_2025"] = r25[n]
                rows[n]["puesto_2022"] = r22[n]
    return out


def to_markdown(res: dict) -> str:
    lines = [f"# Resultados (sin post-procesado; checkpoint elegido con la métrica {res['seleccion']} en validación)", ""]
    for name, m in res["modelos"].items():
        ck = m["checkpoint"] or "-"
        lines.append(f"- {name}: `{Path(m['run']).name}`, checkpoint {ck} ({m['eleccion']})")
    for split, title in (("val", "Validación"), ("test", "Test")):
        rows = res[split]
        if not rows:
            continue
        n_cases = sorted({v["n_casos"] for v in rows.values()})
        lines += ["", f"## {title} ({', '.join(str(n) for n in n_cases)} casos)", "",
                  "### HECKTOR 2025", "",
                  "| Modelo | Checkpoint | GTVp DSC | GTVn DSCagg | GTVn F1agg | Puesto GTVp | Puesto GTVn "
                  "| Puesto final (Borda) | Puesto (posición media) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for n, v in rows.items():
            h, p = v["hecktor2025"], v["puestos_2025"]
            lines.append(f"| {n} | {v['checkpoint']} | {fmt(h['GTVp_DSC'])} | {fmt(h['GTVn_DSCagg'])} | {fmt(h['GTVn_F1agg'])} "
                         f"| {fmt_rank(p['puesto_GTVp'])} | {fmt_rank(p['puesto_GTVn'])} | **{fmt_rank(p['puesto_final'])}** "
                         f"| {fmt_rank(p['puesto_posicion_media'])} |")
        lines += ["", "### HECKTOR 2022", "",
                  "| Modelo | Checkpoint | GTVp DSCagg | GTVn DSCagg | Media DSCagg | Puesto |", "|---|---|---|---|---|---|"]
        for n, v in rows.items():
            h = v["hecktor2022"]
            lines.append(f"| {n} | {v['checkpoint']} | {fmt(h['GTVp_DSCagg'])} | {fmt(h['GTVn_DSCagg'])} "
                         f"| **{fmt(h['DSCagg_media'])}** | {fmt_rank(v['puesto_2022'])} |")
    lines += ["", "Puestos: 1 = mejor, solo entre estos modelos. 2025: conteo de Borda del puesto en GTVp y del puesto en GTVn "
              "(este último, Borda de los puestos en DSCagg y F1agg), como lo describe la página del challenge; la "
              "posición media de las tres columnas es la que muestra su leaderboard. 2022: media de los dos DSC agregados. "
              "\"-\": evaluación anterior sin los volúmenes del GTVp; hay que volver a evaluar con `tools/evaluate`."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, metavar="NOMBRE=RUN[:TAG]")
    ap.add_argument("--select", choices=["2025", "2022"], default="2025",
                    help="metric used on validation to choose between the final and best checkpoints")
    ap.add_argument("--out", default="runs/resultados", help="output prefix (.md and .json are added)")
    args = ap.parse_args(argv)
    models = [parse_model(s) for s in args.models]
    res = build(models, args.select)
    md = to_markdown(res)
    out = resolve(args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(md)
    print(f"guardado {out.with_suffix('.md')} y {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
