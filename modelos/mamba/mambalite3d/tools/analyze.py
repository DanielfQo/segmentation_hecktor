"""Runs the model once over a split and stores per-component statistics for post-processing studies.

    python -m mambalite3d.tools.analyze --run runs/full_base_fold0 --split val --checkpoint final

Writes runs/<run>/components_<split>_<checkpoint>[_tta].json (a few hundred KB, no images), from which
post-processing rules are tuned offline with mambalite3d.eval.components.score_cases.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..config import resolve
from ..data.io import find_cases
from ..data.sampler import iter_cases
from ..eval.components import component_stats, score_cases
from ..models.network import build_model, model_overrides
from ..training.inference import sliding_window_predict
from ..training.train import Logger, save_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", default="final", choices=["final", "best", "latest", "auto"],
                    help="auto = the one (final/best) with the higher score in metrics_val_final/metrics_val_best.json")
    ap.add_argument("--mirror", action="store_true", help="flip test-time augmentation")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    run = resolve(args.run) if not Path(args.run).is_absolute() else Path(args.run)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.checkpoint == "auto":
        scores = {}
        for name in ("final", "best"):
            path = run / f"metrics_val_{name}.json"
            if path.exists():
                scores[name] = json.loads(path.read_text(encoding="utf-8")).get("score", -1)
        args.checkpoint = max(scores, key=scores.get) if scores else "final"
        print(f"checkpoint elegido por validacion: {args.checkpoint} {scores}", flush=True)
    ck = torch.load(run / f"checkpoint_{args.checkpoint}.pth", map_location=device, weights_only=False)
    cfg = ck["config"]
    overrides = model_overrides(cfg["model"])
    model = build_model(cfg["model"]["variant"], ck["strides"], ck["kernels"], in_channels=2,
                        num_classes=cfg["train"]["num_classes"], **overrides).to(device)
    model.load_state_dict(ck["model"])
    with open(run / "split.json", encoding="utf-8") as f:
        ids = set(json.load(f)[args.split])
    files = [c for c in find_cases(resolve(cfg["data"]["root"])) if c.case_id in ids]
    if args.max_cases:
        files = files[: args.max_cases]
    tag = f"{args.split}_{args.checkpoint}" + ("_tta" if args.mirror else "")
    log = Logger(run / "log.txt")
    log(f"analisis de componentes {run.name} [{tag}] ({len(files)} casos)")
    patch = [int(p) for p in cfg["model"]["patch_size_zyx"]]
    cases, t0 = [], time.time()
    for i, case in enumerate(iter_cases(files, cfg["data"], threads=cfg["data"]["load_threads"])):
        probs = sliding_window_predict(model, case.image.astype(np.float32), patch, 3, device,
                                       step_fraction=cfg["infer"]["step_fraction"], amp=cfg["train"]["amp"],
                                       mirror=args.mirror, pad_values=case.pad_values,
                                       batch_size=cfg["infer"].get("batch_size", 4))
        pred = probs.argmax(0).astype(np.uint8)
        st = component_stats(pred, case.label, probs, case.image[1].astype(np.float32), float(np.prod(case.spacing_xyz)))
        st.update(case_id=case.case_id, center=case.center, roi_shape=list(case.image.shape[1:]))
        cases.append(st)
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(files)} casos ({time.time() - t0:.0f} s)")
    out = run / f"components_{tag}.json"
    save_json(cases, out)
    log(f"metricas sin post-procesado [{tag}] (espacio ROI): {json.dumps(score_cases(cases))}")
    log(f"guardado {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
