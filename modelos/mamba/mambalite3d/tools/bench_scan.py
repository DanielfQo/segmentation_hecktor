"""Timing of the selective scan implementations and of full training steps per variant.

    python -m mambalite3d.tools.bench_scan
"""
from __future__ import annotations

import json
import sys
import time

import torch

from mambalite3d.models import ssm


def bench(fn, shape, chunk, reps=5):
    b, d, l, n = shape
    dev = "cuda"
    u = torch.randn(b, d, l, device=dev, requires_grad=True)
    delta = torch.randn(b, d, l, device=dev, requires_grad=True)
    a = -torch.rand(d, n, device=dev) - 0.1
    bm = torch.randn(b, n, l, device=dev, requires_grad=True)
    cm = torch.randn(b, n, l, device=dev, requires_grad=True)
    dd = torch.randn(d, device=dev)
    z = torch.randn(b, d, l, device=dev, requires_grad=True)
    bias = torch.zeros(d, device=dev)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        y = fn(u, delta, a, bm, cm, dd, z, bias, True, chunk_size=chunk)
        y.sum().backward()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return round(1000 * sum(times[1:]) / (reps - 1), 2), round(torch.cuda.max_memory_allocated() / 1024**2, 1)


def main() -> int:
    shapes = [(2, 24, 2048, 16), (2, 32, 256, 16), (2, 64, 64, 16), (2, 24, 9216, 16)]
    for shape in shapes:
        row = {"shape_bdln": shape}
        row["chunked64_ms_MB"] = bench(ssm.selective_scan_chunked, shape, 64) if shape[2] <= 2048 else None
        for t in (8, 16, 32):
            row[f"parallel{t}_ms_MB"] = bench(ssm.selective_scan_parallel, shape, t)
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
