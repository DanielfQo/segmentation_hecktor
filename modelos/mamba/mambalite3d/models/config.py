"""Architecture configuration and ablation variants."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

from .scans import ORDERS


@dataclass
class MambaLiteConfig:
    groups: int = 4  # channel groups (branches) in AMF3D and CGA3D
    orders: Sequence[str] = ORDERS  # scan order per group
    mixer: str = "mamba"  # "mamba" or "mlp" (Mamba-free control)
    n_mamba_levels: int = 2  # deepest encoder/decoder levels with MambaStage (plus the bottleneck)
    n_cga_levels: int = 3  # deepest skip connections filtered by CGA3D (others use FUE)
    use_gsc: bool = True
    use_amf: bool = True
    use_lgfm: bool = True
    use_cga: bool = True
    heads: int = 8
    alpha_init: float = 1.0
    features: Optional[List[int]] = None  # None -> [16, 16, 32, 48, 64, 96] cut to the number of stages
    bottleneck_features: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    chunk_size: int = 8  # chunk length of the parallel selective scan
    baseline: bool = False  # plain residual U-Net (no Mamba, no LGFM/CGA) for comparison

    def group_orders(self) -> List[str]:
        orders = list(self.orders)
        return [orders[i % len(orders)] for i in range(self.groups)]


VARIANTS = {
    "base": {},
    "samescan": {"orders": ("fwd", "fwd", "fwd", "fwd")},
    "nomamba": {"mixer": "mlp"},
    "noamf": {"use_amf": False},
    "nolgfm": {"use_lgfm": False},
    "nocga": {"use_cga": False},
    "g1": {"groups": 1},
    "g2": {"groups": 2},
    "m3": {"n_mamba_levels": 3},
    "unet": {"baseline": True, "features": [32, 64, 128, 256, 320, 320], "bottleneck_features": 320},
    # wider MambaLite3D (more capacity; most extra VRAM is in the shallow levels)
    "wide": {"features": [24, 32, 64, 96, 128, 192], "bottleneck_features": 256},
    "wide2": {"features": [32, 48, 96, 128, 192, 256], "bottleneck_features": 320},
}


def make_config(variant: str = "base", **overrides) -> MambaLiteConfig:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; choose one of {sorted(VARIANTS)}")
    cfg = replace(MambaLiteConfig(), **VARIANTS[variant])
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **clean) if clean else cfg
