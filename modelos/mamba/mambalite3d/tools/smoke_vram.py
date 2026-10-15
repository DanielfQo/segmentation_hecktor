"""Peak GPU memory and speed of one training step, for a patch size and batch size.

    python -m mambalite3d.tools.smoke_vram --patch 32 128 128 --batch 2 --variant base
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from mambalite3d.config import load_config
from mambalite3d.models.network import build_model
from mambalite3d.models.plan import check_patch, plan_network
from mambalite3d.training.loss import DeepSupervisionLoss, DiceCELoss


def run(patch, batch, variant, spacing_zyx, steps=6, chunk_size=16, n_cga_levels=None, n_mamba_levels=None, max_stages=6,
        features=None, bottleneck=None):
    device = torch.device("cuda")
    strides, kernels = plan_network(patch, spacing_zyx, max_stages)
    check_patch(patch, strides)
    overrides = {"chunk_size": chunk_size}
    if n_cga_levels is not None:
        overrides["n_cga_levels"] = n_cga_levels
    if n_mamba_levels is not None:
        overrides["n_mamba_levels"] = n_mamba_levels
    if features:
        overrides["features"] = list(features)
    if bottleneck:
        overrides["bottleneck_features"] = bottleneck
    model = build_model(variant, strides, kernels, **overrides).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = DeepSupervisionLoss(DiceCELoss())
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps):
        x = torch.randn(batch, 2, *patch, device=device)
        y = torch.randint(0, 3, (batch, 1, *patch), device=device)
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    free, total = torch.cuda.mem_get_info()
    res = {
        "variant": variant,
        "patch_zyx": list(patch),
        "batch": batch,
        "strides": strides,
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "s_per_iter": round(sum(times[2:]) / max(1, len(times) - 2), 3),
        "gpu_total_gb": round(total / 1024**3, 2),
        "gpu_free_after_gb": round(free / 1024**3, 2),
    }
    del model, opt
    torch.cuda.empty_cache()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--patch", type=int, nargs=3, action="append", default=None, metavar=("Z", "Y", "X"))
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--n-cga-levels", type=int, default=None)
    ap.add_argument("--features", type=int, nargs="+", default=None, help="channels per stage, e.g. 24 32 64 96 128 192")
    ap.add_argument("--bottleneck", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    spacing_zyx = list(reversed(cfg["data"]["target_spacing_xyz"]))
    patches = args.patch or [cfg["model"]["patch_size_zyx"]]
    results = []
    for p in patches:
        try:
            r = run(p, args.batch, args.variant, spacing_zyx, args.steps, args.chunk, args.n_cga_levels,
                    features=args.features, bottleneck=args.bottleneck)
        except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
            r = {"patch_zyx": list(p), "batch": args.batch, "variant": args.variant, "error": "OOM", "detail": str(e)[:200]}
            torch.cuda.empty_cache()
        print(json.dumps(r), flush=True)
        results.append(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
