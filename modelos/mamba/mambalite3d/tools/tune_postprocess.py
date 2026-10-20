"""Tunes connected-component post-processing on validation and checks it on test (no GPU needed).

    python -m mambalite3d.tools.tune_postprocess --run runs/full_base_fold0 --checkpoint final

Reads components_val_<ckpt>.json (and components_test_<ckpt>.json if present) written by
mambalite3d.tools.analyze, searches simple rules on validation only, and writes
postprocess_<ckpt>.json (the chosen rules + metrics before/after on val and test).

Rules (volumes in ml):
- GTVp: drop components smaller than min_ml or with max probability < min_maxp; optionally keep only
  the largest remaining one; then drop the whole GTVp if what is left is smaller than case_min_ml
  (cases without a primary tumour).
- GTVn: drop components smaller than min_ml or with max probability < min_maxp.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from ..eval.components import score_cases

GTVP_GRID = {
    "largest": [False, True],
    "min_ml": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
    "min_maxp": [0.0, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999],
    "case_min_ml": [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
}
GTVN_GRID = {
    "min_ml": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
    "min_maxp": [0.0, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999],
}


def _ml(comp, voxel_mm3):
    return comp["vox"] * voxel_mm3 / 1000.0


def apply_rules(cases, rules):
    """Returns a copy of the component lists with only the kept components (same format)."""
    rp, rn = rules.get("gtvp", {}), rules.get("gtvn", {})
    out = []
    for case in cases:
        v = case["voxel_mm3"]
        comps = [c for c in case["1"]["pred"]
                 if _ml(c, v) >= rp.get("min_ml", 0) and c["maxp"] >= rp.get("min_maxp", 0)]
        if rp.get("largest") and comps:
            comps = [max(comps, key=lambda c: c["vox"])]
        if sum(_ml(c, v) for c in comps) < rp.get("case_min_ml", 0):
            comps = []
        comps_n = [c for c in case["2"]["pred"]
                   if _ml(c, v) >= rn.get("min_ml", 0) and c["maxp"] >= rn.get("min_maxp", 0)]
        new = dict(case)
        new["1"] = {"pred": comps, "gt": case["1"]["gt"]}
        new["2"] = {"pred": comps_n, "gt": case["2"]["gt"]}
        out.append(new)
    return out


def search(cases):
    base = score_cases(cases)
    best_p, best_p_rule = base["gtvp_dice_mean"], {}
    for vals in itertools.product(*GTVP_GRID.values()):
        rule = dict(zip(GTVP_GRID.keys(), vals))
        m = score_cases(apply_rules(cases, {"gtvp": rule}))
        if m["gtvp_dice_mean"] > best_p + 1e-9:
            best_p, best_p_rule = m["gtvp_dice_mean"], rule
    best_n_dsc, best_n_dsc_rule = base["gtvn_dice_agg"], {}
    best_n_avg, best_n_avg_rule = 0.5 * (base["gtvn_dice_agg"] + base["gtvn_f1_agg"]), {}
    for vals in itertools.product(*GTVN_GRID.values()):
        rule = dict(zip(GTVN_GRID.keys(), vals))
        m = score_cases(apply_rules(cases, {"gtvn": rule}))
        if m["gtvn_dice_agg"] > best_n_dsc + 1e-9:
            best_n_dsc, best_n_dsc_rule = m["gtvn_dice_agg"], rule
        avg = 0.5 * (m["gtvn_dice_agg"] + m["gtvn_f1_agg"])
        if avg > best_n_avg + 1e-9:
            best_n_avg, best_n_avg_rule = avg, rule
    return best_p_rule, best_n_dsc_rule, best_n_avg_rule


def describe(cases):
    """Diagnostics: no-primary cases, Dice vs size, false-positive components, class confusion."""
    d = {}
    v = cases[0]["voxel_mm3"]
    nop = [c for c in cases if sum(c["1"]["gt"]) == 0]
    d["cases_without_gtvp"] = [
        {"case": c["case_id"], "pred_ml": round(sum(_ml(p, v) for p in c["1"]["pred"]), 2),
         "n_comps": len(c["1"]["pred"]), "maxp": max([p["maxp"] for p in c["1"]["pred"]], default=0.0),
         "pred_on_gtvn_ml": round(sum(p["other"] for p in c["1"]["pred"]) * v / 1000, 2)}
        for c in nop
    ]
    rows = []
    for c in cases:
        g = sum(c["1"]["gt"])
        if g:
            p = sum(x["vox"] for x in c["1"]["pred"])
            i = sum(x["inter"] for x in c["1"]["pred"])
            rows.append((g * v / 1000, 2 * i / (p + g)))
    rows.sort()
    q = [a for a in np.array_split(np.array(rows), 4) if len(a)]  # fewer than 4 cases leave empty quartiles
    d["gtvp_dice_by_size_quartile"] = [
        {"ml_range": [round(float(a[:, 0].min()), 1), round(float(a[:, 0].max()), 1)], "mean_dice": round(float(a[:, 1].mean()), 3)}
        for a in q
    ]
    for cls, name in ((1, "gtvp"), (2, "gtvn")):
        tp_like = [x for c in cases for x in c[str(cls)]["pred"] if x["inter"] > 0]
        fp_like = [x for c in cases for x in c[str(cls)]["pred"] if x["inter"] == 0]
        fp_ml = np.array([_ml(x, v) for x in fp_like]) if fp_like else np.zeros(1)
        d[f"{name}_components"] = {
            "overlapping_reference": len(tp_like),
            "without_overlap": len(fp_like),
            "without_overlap_on_other_class": sum(1 for x in fp_like if x["other"] > 0),
            "fp_ml_quantiles_10_50_90": [round(float(x), 3) for x in np.quantile(fp_ml, [0.1, 0.5, 0.9])],
            "fp_total_ml": round(float(fp_ml.sum()), 1),
            "tp_total_pred_ml": round(float(sum(_ml(x, v) for x in tp_like)), 1),
            "fp_maxp_median": round(float(np.median([x["maxp"] for x in fp_like])), 3) if fp_like else None,
            "tp_maxp_median": round(float(np.median([x["maxp"] for x in tp_like])), 3) if tp_like else None,
        }
    conf = np.sum([np.array(c["confusion"]) for c in cases], axis=0)
    d["confusion_ml_rows_ref_cols_pred"] = (conf * v / 1000).round(1).tolist()
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="final")
    ap.add_argument("--suffix", default="", help="e.g. _tta")
    args = ap.parse_args(argv)
    run = Path(args.run)
    tag = f"{args.checkpoint}{args.suffix}"
    val = json.loads((run / f"components_val_{tag}.json").read_text(encoding="utf-8"))
    test_path = run / f"components_test_{tag}.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else None

    rule_p, rule_n_dsc, rule_n_avg = search(val)
    chosen = {"gtvp": rule_p, "gtvn": rule_n_dsc}
    alt = {"gtvp": rule_p, "gtvn": rule_n_avg}
    report = {"rules_best_dice": chosen, "rules_best_dice_plus_f1": alt, "val": {}, "test": {}}
    for name, cases in (("val", val), ("test", test)):
        if cases is None:
            continue
        report[name]["sin_postproceso"] = score_cases(cases)
        report[name]["best_dice"] = score_cases(apply_rules(cases, chosen))
        report[name]["best_dice_plus_f1"] = score_cases(apply_rules(cases, alt))
        report[name]["diagnostico"] = describe(cases)
    out = run / f"postprocess_{tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rules_best_dice", "rules_best_dice_plus_f1")}, indent=2))
    for name in ("val", "test"):
        for k in ("sin_postproceso", "best_dice", "best_dice_plus_f1"):
            if k in report[name]:
                m = report[name][k]
                print(f"{name:4s} {k:18s} GTVp {m['gtvp_dice_mean']:.4f} (ref {m['gtvp_dice_mean_ref_only']:.4f})  "
                      f"GTVn DSCagg {m['gtvn_dice_agg']:.4f}  F1 {m['gtvn_f1_agg']:.4f}  TP/FP/FN {m['gtvn_tp']}/{m['gtvn_fp']}/{m['gtvn_fn']}")
    print(f"guardado {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
