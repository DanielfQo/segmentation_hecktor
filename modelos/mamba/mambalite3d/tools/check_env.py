"""Environment check: versions, GPU, and a forward/backward pass of MambaLite3D on the GPU.

    python -m mambalite3d.tools.check_env
"""
from __future__ import annotations

import json
import platform
import sys
import time

import torch


def main() -> int:
    ok = True
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__}
    info["cuda_available"] = torch.cuda.is_available()
    try:
        import numpy
        import scipy
        import SimpleITK as sitk
        import yaml  # noqa: F401

        info.update(numpy=numpy.__version__, scipy=scipy.__version__, simpleitk=sitk.Version_VersionString())
    except Exception as e:  # noqa: BLE001
        info["import_error"] = repr(e)
        ok = False
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, vram_total_gb=round(total / 1024**3, 2),
                    vram_free_gb=round(free / 1024**3, 2), bf16=torch.cuda.is_bf16_supported())
    else:
        ok = False

    from mambalite3d.models.network import build_model
    from mambalite3d.models.plan import plan_network
    from mambalite3d.models.ssm import HAS_MAMBA_SSM, SCAN_IMPL, selective_scan, selective_scan_ref

    info["mamba_ssm"] = HAS_MAMBA_SSM
    info["scan_impl"] = SCAN_IMPL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # scan equivalence on the device
    g = torch.Generator(device="cpu").manual_seed(0)
    b, d, n, l = 2, 8, 16, 300
    u = torch.randn(b, d, l, generator=g)
    delta = torch.randn(b, d, l, generator=g) * 0.5
    a = -torch.rand(d, n, generator=g) * 2
    bm = torch.randn(b, n, l, generator=g)
    cm = torch.randn(b, n, l, generator=g)
    dd = torch.randn(d, generator=g)
    z = torch.randn(b, d, l, generator=g)
    ref = selective_scan_ref(u, delta, a, bm, cm, dd, z, None, True)
    out = selective_scan(*(t.to(device) for t in (u, delta, a, bm, cm, dd, z)), None, True).cpu()
    err = (ref - out).abs().max().item()
    info["scan_max_abs_err"] = err
    ok &= err < 1e-3

    # forward/backward of the full network with AMP
    patch = [32, 64, 64]
    strides, kernels = plan_network(patch, [3.0, 1.0, 1.0])
    model = build_model("base", strides, kernels).to(device)
    x = torch.randn(2, 2, *patch, device=device)
    y = torch.randint(0, 3, (2, 1, *patch), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        outs = model(x)
        loss = sum(torch.nn.functional.cross_entropy(o.float(), torch.nn.functional.interpolate(y.float(), size=o.shape[2:]).long()[:, 0]) for o in outs)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
        info["test_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    info["test_fwd_bwd_s"] = round(time.time() - t0, 2)
    finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    info["grads_finite"] = finite
    ok &= finite and torch.isfinite(loss).item()
    info["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
    info["ok"] = bool(ok)
    print(json.dumps(info, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
