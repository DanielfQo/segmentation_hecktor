"""Checks that a run finished cleanly (used by the crash/resume tests).

    python -m mambalite3d.tools.check_run --run runs/X --epochs 2 [--expect-resume]
"""
from __future__ import annotations

import argparse
import json
import sys

from ..config import resolve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--expect-resume", action="store_true")
    ap.add_argument("--expect-crash", action="store_true", help="the log must show the simulated crash")
    args = ap.parse_args(argv)
    run = resolve(args.run)
    problems = []
    hist = json.loads((run / "history.json").read_text(encoding="utf-8"))
    epochs = [h["epoch"] for h in hist]
    if epochs != list(range(args.epochs)):
        problems.append(f"historial de epocas {epochs}, se esperaba {list(range(args.epochs))}")
    for f in ("done.json", "checkpoint_final.pth", "checkpoint_latest.pth", "split.json", "metrics_val_final.json"):
        if not (run / f).exists():
            problems.append(f"falta {f}")
    split = json.loads((run / "split.json").read_text(encoding="utf-8"))
    if split.get("test") and not (run / "metrics_test_final.json").exists():
        problems.append("falta metrics_test_final.json")
    sets = [set(split[k]) for k in ("train", "val", "test")]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        problems.append("hay casos repetidos entre train/val/test")
    log = (run / "log.txt").read_text(encoding="utf-8")
    if args.expect_resume and "reanudando desde la epoca" not in log:
        problems.append("el log no muestra la reanudacion")
    if args.expect_crash and "terminacion abrupta simulada" not in log:
        problems.append("el log no muestra la terminacion simulada")
    leftovers = [p.name for p in run.glob("*.tmp")]
    print(f"{run.name}: epocas {epochs}; temporales sobrantes {leftovers}")
    if problems:
        print("FALLO: " + "; ".join(problems))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
