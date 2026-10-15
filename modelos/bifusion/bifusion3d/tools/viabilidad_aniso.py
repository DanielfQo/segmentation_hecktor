"""Viabilidad de AnisoCrossNeXt-3D en la RTX 3050 de 4 GB: parametros, VRAM y segundos por iteracion.

El modelo es el de la seccion 4 del adjunto, copiado sin cambios. Se mide con el mismo paso de
entrenamiento que mambalite3d (AMP fp16 con GradScaler, cudnn.benchmark, SGD y Dice + CE con supervision
profunda) sobre datos sinteticos, asi que no lee el dataset ni escribe en los runs.

Referencia de siam_full en esta GPU (su log): 0,772 s/iteracion y 1,94 GB de pico con batch 2 de 32 x 128 x 128.

    python -m bifusion3d.tools.viabilidad_aniso --out runs/viabilidad_aniso
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class SqueezeExcite3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, mid, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class RobustPETCTStem(nn.Module):
    """
    Stem anisotrópico (kernel 1x3x3) que recibe 3 canales: [CT_soft, CT_wide, PET].
    Tolera pequeños desregistros PET-CT y atenúa artefactos dentales en CT.
    """
    def __init__(self, base_c: int = 24):
        super().__init__()
        half_c = base_c // 2

        # Rama CT (2 canales: ventana tejido blando + ventana ósea/artefactos)
        self.ct_conv = nn.Sequential(
            nn.Conv3d(2, half_c, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(4, half_c),
            nn.SiLU(inplace=True)
        )
        # Compuerta para suprimir artefactos metálicos dentales en CT
        self.ct_artifact_gate = nn.Sequential(
            nn.Conv3d(half_c, half_c, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Rama PET (1 canal SUV normalizado)
        self.pet_conv = nn.Sequential(
            nn.Conv3d(1, half_c, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(4, half_c),
            nn.SiLU(inplace=True)
        )
        # MaxPool 1x3x3 da tolerancia espacial a desregistros de 2-3 mm entre PET y CT
        self.pet_tolerance_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(half_c, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(base_c, base_c, kernel_size=1, bias=False),
            nn.GroupNorm(8, base_c),
            nn.SiLU(inplace=True),
            SqueezeExcite3D(base_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ct = x[:, 0:2, ...]   # [B, 2, Z, Y, X]
        pet = x[:, 2:3, ...]  # [B, 1, Z, Y, X]

        f_ct = self.ct_conv(ct)
        f_ct = f_ct * self.ct_artifact_gate(f_ct)

        f_pet = self.pet_conv(pet)
        att_pet = self.spatial_gate(self.pet_tolerance_pool(f_pet))

        # Modulación suave tolerante a desregistro
        f_ct_guided = f_ct * (1.0 + att_pet)
        return self.fuse(torch.cat([f_ct_guided, f_pet], dim=1))


class AnisoNeXtBlock3D(nn.Module):
    """
    Bloque ConvNeXt 3D con soporte para kernels anisotrópicos (kz, ky, kx)
    adaptados al espaciado 3x1x1 mm.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: tuple = (3, 5, 5),
        expansion: int = 2,
        dilation: tuple = (1, 1, 1),
        use_se: bool = False
    ):
        super().__init__()
        pad = tuple((k // 2) * d for k, d in zip(kernel_size, dilation))
        hidden = channels * expansion

        self.dwconv = nn.Conv3d(
            channels, channels, kernel_size=kernel_size,
            padding=pad, dilation=dilation, groups=channels, bias=False
        )
        self.norm = nn.GroupNorm(8, channels)
        self.pw1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(hidden, channels, kernel_size=1)
        self.se = SqueezeExcite3D(channels) if use_se else nn.Identity()

    def _impl(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.se(x)
        return res + x

    def forward(self, x: torch.Tensor, use_ckpt: bool = False) -> torch.Tensor:
        if use_ckpt and x.requires_grad:
            return checkpoint(self._impl, x, use_reentrant=False)
        return self._impl(x)


class BilateralSymmetryGate3D(nn.Module):
    """
    Compara el hemisferio izquierdo y derecho en el eje X para descartar
    captación fisiológica simétrica (amígdalas, glándulas salivales) y resaltar
    asimetrías tumorales (GTVp y GTVn).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.asym_proj = nn.Sequential(
            nn.Conv3d(channels, channels // 4, kernel_size=1, bias=False),
            nn.GroupNorm(4, channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels // 4, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flipped = torch.flip(x, dims=[-1])  # Flip en eje X (Izquierda-Derecha)
        diff = torch.abs(x - x_flipped)
        asym_map = self.asym_proj(diff)
        return x * (1.0 + 0.5 * asym_map)


class AnisoDownStage(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        stride: tuple,
        kernel_size: tuple,
        num_blocks: int,
        use_se: bool = False,
        use_symmetry: bool = False
    ):
        super().__init__()
        self.down = nn.Sequential(
            nn.GroupNorm(8, in_c),
            nn.Conv3d(in_c, out_c, kernel_size=stride, stride=stride, bias=False)
        )
        self.blocks = nn.ModuleList([
            AnisoNeXtBlock3D(out_c, kernel_size=kernel_size, expansion=2, use_se=use_se)
            for _ in range(num_blocks)
        ])
        self.sym = BilateralSymmetryGate3D(out_c) if use_symmetry else nn.Identity()

    def forward(self, x: torch.Tensor, use_ckpt: bool = False) -> torch.Tensor:
        x = self.down(x)
        for blk in self.blocks:
            x = blk(x, use_ckpt=use_ckpt)
        return self.sym(x)


class AnisoUpStage(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int, stride: tuple, kernel_size: tuple):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=stride, stride=stride, bias=False)
        self.skip_proj = nn.Conv3d(skip_c, out_c, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, out_c)
        self.block = AnisoNeXtBlock3D(out_c, kernel_size=kernel_size, expansion=2, use_se=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, use_ckpt: bool = False) -> torch.Tensor:
        x = self.up(x)
        x = self.norm(x + self.skip_proj(skip))
        return self.block(x, use_ckpt=use_ckpt)


class AnisoCrossNeXt3D(nn.Module):
    """
    Diseñada para parches (Z, Y, X) = (48, 128, 128) con espaciado (3.0, 1.0, 1.0) mm.
    Entrena en ~2.45 GB de VRAM con FP16 + Checkpointing.
    """
    def __init__(
        self,
        num_classes: int = 3,
        dims: tuple = (24, 48, 96, 192, 320),
        use_checkpoint: bool = True,
        deep_supervision: bool = True
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.deep_supervision = deep_supervision

        # Etapa 1: (48, 128, 128) | Vóxel: 3x1x1 mm -> Kernel axial (1, 5, 5) + acoplamiento Z ligero
        self.stem = RobustPETCTStem(base_c=dims[0])
        self.enc1 = nn.ModuleList([
            AnisoNeXtBlock3D(dims[0], kernel_size=(1, 5, 5), expansion=2),
            AnisoNeXtBlock3D(dims[0], kernel_size=(3, 3, 3), expansion=2)
        ])

        # Etapa 2: Stride (1, 2, 2) -> (48, 64, 64) | Vóxel: 3x2x2 mm -> Kernel (3, 5, 5)
        self.enc2 = AnisoDownStage(
            dims[0], dims[1], stride=(1, 2, 2), kernel_size=(3, 5, 5),
            num_blocks=2, use_se=False, use_symmetry=False
        )

        # Etapa 3: Stride (2, 2, 2) -> (24, 32, 32) | Vóxel: 6x4x4 mm -> Kernel (5, 5, 5) + Asimetría
        self.enc3 = AnisoDownStage(
            dims[1], dims[2], stride=(2, 2, 2), kernel_size=(5, 5, 5),
            num_blocks=3, use_se=True, use_symmetry=True
        )

        # Etapa 4: Stride (2, 2, 2) -> (12, 16, 16) | Vóxel: 12x8x8 mm -> Kernel (5, 5, 5) + Asimetría
        self.enc4 = AnisoDownStage(
            dims[2], dims[3], stride=(2, 2, 2), kernel_size=(5, 5, 5),
            num_blocks=3, use_se=True, use_symmetry=True
        )

        # Bottleneck: Stride (2, 2, 2) -> (6, 8, 8) | Vóxel: 24x16x16 mm
        self.down_bottle = nn.Sequential(
            nn.GroupNorm(8, dims[3]),
            nn.Conv3d(dims[3], dims[4], kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        )
        self.b1 = AnisoNeXtBlock3D(dims[4], kernel_size=(3, 3, 3), dilation=(1, 1, 1), use_se=True)
        self.b2 = AnisoNeXtBlock3D(dims[4], kernel_size=(3, 3, 3), dilation=(1, 2, 2), use_se=True)

        # Decoder Asimétrico con Strides Inversos Exactos
        self.dec4 = AnisoUpStage(dims[4], dims[3], dims[3], stride=(2, 2, 2), kernel_size=(5, 5, 5))
        self.dec3 = AnisoUpStage(dims[3], dims[2], dims[2], stride=(2, 2, 2), kernel_size=(5, 5, 5))
        self.dec2 = AnisoUpStage(dims[2], dims[1], dims[1], stride=(2, 2, 2), kernel_size=(3, 5, 5))
        # El último paso sube solo Y y X con stride (1, 2, 2)
        self.dec1 = AnisoUpStage(dims[1], dims[0], dims[0], stride=(1, 2, 2), kernel_size=(1, 5, 5))

        self.head_main = nn.Conv3d(dims[0], num_classes, kernel_size=1)
        if self.deep_supervision:
            self.head_ds2 = nn.Conv3d(dims[1], num_classes, kernel_size=1)
            self.head_ds3 = nn.Conv3d(dims[2], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        ckpt = self.use_checkpoint and self.training

        s1 = self.stem(x)
        for blk in self.enc1:
            s1 = blk(s1, use_ckpt=ckpt)

        s2 = self.enc2(s1, use_ckpt=ckpt)
        s3 = self.enc3(s2, use_ckpt=ckpt)
        s4 = self.enc4(s3, use_ckpt=ckpt)

        b = self.down_bottle(s4)
        b = self.b1(b, use_ckpt=ckpt)
        b = self.b2(b, use_ckpt=ckpt)

        d4 = self.dec4(b, s4, use_ckpt=ckpt)
        d3 = self.dec3(d4, s3, use_ckpt=ckpt)
        d2 = self.dec2(d3, s2, use_ckpt=ckpt)
        d1 = self.dec1(d2, s1, use_ckpt=ckpt)

        out_main = self.head_main(d1)

        if self.training and self.deep_supervision:
            out_ds2 = F.interpolate(self.head_ds2(d2), size=x.shape[2:], mode="trilinear", align_corners=False)
            out_ds3 = F.interpolate(self.head_ds3(d3), size=x.shape[2:], mode="trilinear", align_corners=False)
            return [out_main, out_ds2, out_ds3]

        return out_main


# ---------------------------------------------------------------------------------------------------------
# Medicion

SIAM_FULL = {"s_por_iter": 0.772, "pico_gb": 1.942, "batch": 2, "parche": [32, 128, 128]}
ITERS_RUN = 250 * 250  # 250 epocas x 250 iteraciones, el protocolo de los runs actuales

CASOS = [
    ("propuesta", (48, 128, 128), 1, True),
    ("propuesta sin checkpointing", (48, 128, 128), 1, False),
    ("protocolo actual", (32, 128, 128), 2, True),
    ("protocolo actual sin checkpointing", (32, 128, 128), 2, False),
]


def _etiquetas(batch, patch, device):
    """Etiqueta sintetica con un GTVp (1) y dos ganglios (2) por parche, para que Dice y CE trabajen."""
    z, y, x = patch
    t = torch.zeros(batch, 1, z, y, x, dtype=torch.long, device=device)
    t[:, :, z // 4: z // 2, y // 3: y // 2, x // 3: x // 2] = 1
    t[:, :, z // 2: z // 2 + 4, 2 * y // 3: 2 * y // 3 + 8, x // 6: x // 6 + 8] = 2
    t[:, :, z // 3: z // 3 + 4, y // 6: y // 6 + 6, 3 * x // 4: 3 * x // 4 + 6] = 2
    return t


def medir(nombre, patch, batch, ckpt, iters, warmup, loss_fn):
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    res = {"caso": nombre, "parche": list(patch), "batch": batch, "checkpointing": ckpt}
    model = opt = x = t = None
    try:
        model = AnisoCrossNeXt3D(use_checkpoint=ckpt).to(device).train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
        scaler = torch.amp.GradScaler("cuda")
        x = torch.randn(batch, 3, *patch, device=device)
        t = _etiquetas(batch, patch, device)
        for i in range(warmup + iters):
            if i == warmup:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
                loss = loss_fn(out, t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        s_iter = (time.perf_counter() - t0) / iters
        res.update({
            "cabe": True,
            "s_por_iter": round(s_iter, 3),
            "voxeles_por_s": round(batch * patch[0] * patch[1] * patch[2] / s_iter),
            "pico_asignado_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "pico_reservado_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            "horas_250x250": round(s_iter * ITERS_RUN / 3600, 1),
            "loss_final": round(float(loss.detach()), 4),
        })
        # inferencia de una ventana, como en la ventana deslizante (sin gradientes, AMP)
        model.eval()
        xi = x[:1]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(3):
                model(xi)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(10):
                model(xi)
            torch.cuda.synchronize()
        res["s_por_ventana_inferencia"] = round((time.perf_counter() - t1) / 10, 3)
    except torch.cuda.OutOfMemoryError as e:
        res.update({"cabe": False, "error": "sin memoria: " + str(e).splitlines()[0][:200]})
    finally:
        del model, opt, x, t
        torch.cuda.empty_cache()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/viabilidad_aniso")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: no hay GPU CUDA")
        return 1
    import bifusion3d  # noqa: F401  (anade modelos/mamba a sys.path)
    from bifusion3d import thermal
    from mambalite3d.training.loss import DeepSupervisionLoss, DiceCELoss

    torch.backends.cudnn.benchmark = True  # como en el entrenamiento de mambalite3d
    guard = thermal.ThermalGuard({"every": 1}, log=print)
    libre, total = torch.cuda.mem_get_info()
    params = sum(p.numel() for p in AnisoCrossNeXt3D().parameters())
    print(f"GPU {torch.cuda.get_device_name(0)}: libre {libre / 2**30:.2f} de {total / 2**30:.2f} GB; torch {torch.__version__}")
    print(f"AnisoCrossNeXt-3D: {params / 1e6:.2f} M parametros (siam_full: 4,26 M)")

    loss_fn = DeepSupervisionLoss(DiceCELoss())
    resultados = []
    for nombre, patch, batch, ckpt in CASOS:
        guard.tick(force=True)  # pausa si la GPU esta caliente antes de cada medicion
        r = medir(nombre, patch, batch, ckpt, args.iters, args.warmup, loss_fn)
        resultados.append(r)
        if r["cabe"]:
            print(f"{nombre:36s} batch {batch} parche {patch}: {r['s_por_iter']:.3f} s/iter, "
                  f"pico {r['pico_asignado_gb']:.2f} GB (reservado {r['pico_reservado_gb']:.2f}), "
                  f"~{r['horas_250x250']} h por run de 250 x 250, inferencia {r['s_por_ventana_inferencia']:.3f} s/ventana")
        else:
            print(f"{nombre:36s} batch {batch} parche {patch}: NO CABE ({r['error']})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resumen = {
        "modelo": "AnisoCrossNeXt-3D (adjunto, seccion 4, sin cambios)",
        "parametros_m": round(params / 1e6, 3),
        "gpu": torch.cuda.get_device_name(0),
        "vram_libre_gb": round(libre / 2**30, 3),
        "torch": torch.__version__,
        "referencia_siam_full": SIAM_FULL,
        "temperatura_max_c": guard.max_seen,
        "pausas_termicas": guard.pauses,
        "mediciones": resultados,
    }
    (out / "resultado.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"resultado en {out / 'resultado.json'}; GPU maxima {guard.max_seen} C")
    return 0 if any(r["cabe"] for r in resultados) else 2


if __name__ == "__main__":
    raise SystemExit(main())
