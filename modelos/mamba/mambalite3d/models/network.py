"""MambaLite3D: U-Net with Mamba only at the deepest levels, built from nnU-Net-style strides."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .blocks import FUE, BasicResBlock, ConvIN, norm3d
from .config import MambaLiteConfig, make_config
from .modules import CGA3D, MambaStage
from .ssm import GroupedMambaTorch, MambaMixer, TokenMLP

DEFAULT_FEATURES = [16, 16, 32, 48, 64, 96]


def stage_features(n_stages: int, base: Optional[Sequence[int]] = None, keep: str = "last") -> List[int]:
    """Channels per stage. keep='last' drops shallow entries (MambaLite3D keeps its deep widths);
    keep='first' drops deep entries (nnU-Net style baseline)."""
    base = list(base) if base is not None else list(DEFAULT_FEATURES)
    if n_stages <= len(base):
        return base[len(base) - n_stages :] if keep == "last" else base[:n_stages]
    if keep == "last":
        return [base[0]] * (n_stages - len(base)) + base
    return base + [base[-1]] * (n_stages - len(base))


class Encoder(nn.Module):
    def __init__(self, in_channels: int, feats: List[int], strides, kernels, cfg: MambaLiteConfig, mamba_levels):
        super().__init__()
        n = len(feats)
        self.stages = nn.ModuleList()
        self.stages.append(nn.Sequential(ConvIN(in_channels, feats[0], kernels[0]), BasicResBlock(feats[0], kernels[0])))
        for lvl in range(1, n):
            if lvl == 1:
                k_stem = [7 if s > 1 else 3 for s in strides[1]]
                down = nn.Sequential(
                    nn.Conv3d(feats[0], feats[0], k_stem, stride=strides[1], padding=[k // 2 for k in k_stem], groups=feats[0]),
                    nn.Conv3d(feats[0], feats[1], 1),
                    norm3d(feats[1]),
                    nn.LeakyReLU(0.01, inplace=True),
                )
                cin = feats[1]
            else:
                down = nn.MaxPool3d(kernel_size=strides[lvl], stride=strides[lvl])
                cin = feats[lvl - 1]
            if lvl in mamba_levels:
                block = MambaStage(cin, feats[lvl], cfg)
            elif cin != feats[lvl]:
                block = nn.Sequential(ConvIN(cin, feats[lvl], kernels[lvl]), BasicResBlock(feats[lvl], kernels[lvl]))
            else:
                block = BasicResBlock(feats[lvl], kernels[lvl])
            self.stages.append(nn.Sequential(down, block))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class Decoder(nn.Module):
    def __init__(self, feats, bottleneck_ch, num_classes, strides, kernels, cfg, mamba_levels, cga_levels, deep_supervision):
        super().__init__()
        n = len(feats)
        self.n = n
        self.deep_supervision = deep_supervision
        self.cga_levels = set(cga_levels)
        self.up = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.block = nn.ModuleList()
        self.heads = nn.ModuleList()
        for lvl in range(n):
            if lvl == n - 1:
                self.up.append(nn.Sequential(nn.Conv3d(bottleneck_ch, feats[lvl], 1), norm3d(feats[lvl])))
            else:
                self.up.append(
                    nn.ConvTranspose3d(feats[lvl + 1], feats[lvl], kernel_size=strides[lvl + 1], stride=strides[lvl + 1])
                )
            if cfg.baseline:
                self.skip.append(nn.Identity())
            elif lvl in self.cga_levels:
                self.skip.append(CGA3D(feats[lvl], cfg))
            else:
                self.skip.append(FUE())
            if lvl in mamba_levels:
                self.block.append(MambaStage(feats[lvl], feats[lvl], cfg))
            else:
                self.block.append(BasicResBlock(feats[lvl], kernels[lvl]))
            self.heads.append(nn.Conv3d(feats[lvl], num_classes, 1) if lvl <= n - 2 else nn.Identity())

    def forward(self, skips: List[torch.Tensor], bottleneck: torch.Tensor):
        n = self.n
        outs = {}
        d = None
        for lvl in reversed(range(n)):
            h = self.up[lvl](bottleneck if lvl == n - 1 else d)
            if lvl in self.cga_levels:
                s = self.skip[lvl](skips[lvl], h)
            else:
                s = self.skip[lvl](skips[lvl])
            d = self.block[lvl](h + s)
            if lvl <= n - 2 and (self.deep_supervision or lvl == 0):
                outs[lvl] = self.heads[lvl](d)
        if self.deep_supervision:
            return [outs[lvl] for lvl in range(n - 1)]
        return outs[0]


class MambaLite3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        strides: Sequence[Sequence[int]],
        kernels: Sequence[Sequence[int]],
        cfg: Optional[MambaLiteConfig] = None,
        deep_supervision: bool = True,
    ):
        super().__init__()
        cfg = cfg or MambaLiteConfig()
        n = len(strides)
        if n < 2:
            raise ValueError("need at least 2 stages")
        feats = stage_features(n, cfg.features, keep="first" if cfg.baseline else "last")
        if cfg.baseline:
            mamba_levels, cga_levels = set(), set()
        else:
            mamba_levels = set(range(max(n - cfg.n_mamba_levels, 1), n))
            cga_levels = set(range(max(n - cfg.n_cga_levels, 0), n)) if cfg.use_cga else set()
        self.cfg = cfg
        self.feats = feats
        self.mamba_levels = sorted(mamba_levels)
        self.cga_levels = sorted(cga_levels)
        self.strides = [list(s) for s in strides]
        self.encoder = Encoder(in_channels, feats, strides, kernels, cfg, mamba_levels)
        if cfg.baseline:
            self.bottleneck = nn.Sequential(
                ConvIN(feats[-1], cfg.bottleneck_features, kernels[-1]), BasicResBlock(cfg.bottleneck_features, kernels[-1])
            )
        else:
            self.bottleneck = MambaStage(feats[-1], cfg.bottleneck_features, cfg)
        self.decoder = Decoder(
            feats, cfg.bottleneck_features, num_classes, strides, kernels, cfg, mamba_levels, cga_levels, deep_supervision
        )
        init_weights(self)

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        return self.decoder(skips, self.bottleneck(skips[-1]))


def init_weights(model: nn.Module) -> None:
    """He init for convolutions and truncated normal for linear layers, leaving Mamba's own init intact."""
    protected = set()
    for m in model.modules():
        if isinstance(m, (MambaMixer, TokenMLP, GroupedMambaTorch)):
            protected.update(id(sub) for sub in m.modules())
    for m in model.modules():
        if id(m) in protected:
            continue
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, a=1e-2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def build_model(
    variant: str,
    strides,
    kernels,
    in_channels: int = 2,
    num_classes: int = 3,
    deep_supervision: bool = True,
    **overrides,
) -> MambaLite3D:
    cfg = make_config(variant, **overrides)
    return MambaLite3D(in_channels, num_classes, strides, kernels, cfg, deep_supervision)


MODEL_OVERRIDE_KEYS = ("chunk_size", "n_cga_levels", "n_mamba_levels", "features", "bottleneck_features")


def model_overrides(model_cfg: dict) -> dict:
    """Architecture overrides taken from the `model` section of a run config (None entries are ignored)."""
    return {k: model_cfg[k] for k in MODEL_OVERRIDE_KEYS if model_cfg.get(k) is not None}
