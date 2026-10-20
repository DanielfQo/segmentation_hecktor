"""3D adaptations of MambaLiteUNet's AMF, LGFM and CGA, plus the MambaStage used at deep levels.

Equations follow the MambaLiteUNet paper (Rahman et al., arXiv 2604.20286) and its official code;
differences are listed in DEVIATIONS.md.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BasicResBlock, ChannelLayerNorm, GSC3D, norm3d
from .config import MambaLiteConfig
from .scans import flatten, unflatten
from .ssm import HAS_MAMBA_SSM, GroupedMambaTorch, make_mixer


class DirectionalMixer(nn.Module):
    """Flattens the volume in a given scan order, mixes tokens and restores the volume."""

    def __init__(self, channels: int, order: str, cfg: MambaLiteConfig):
        super().__init__()
        self.order = order
        self.mixer = make_mixer(
            channels, cfg.mixer, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand, chunk_size=cfg.chunk_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[2:]
        return unflatten(self.mixer(flatten(x, self.order)), self.order, shape)


class MultiDirectionalMixer(nn.Module):
    """One token mixer per stream, each stream scanned in its own order.

    Without mamba_ssm, all Mamba mixers of the block run as one GroupedMambaTorch call (same maths
    as separate layers, far fewer GPU kernel launches). Otherwise one DirectionalMixer per stream.
    """

    def __init__(self, channels: int, orders, cfg: MambaLiteConfig):
        super().__init__()
        self.orders = list(orders)
        if cfg.mixer == "mamba" and not HAS_MAMBA_SSM:
            self.grouped = GroupedMambaTorch(
                len(self.orders), channels, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand, chunk_size=cfg.chunk_size
            )
            self.mixers = None
        else:
            self.grouped = None
            self.mixers = nn.ModuleList([DirectionalMixer(channels, o, cfg) for o in self.orders])

    def forward(self, xs):
        if self.grouped is None:
            return [m(x) for m, x in zip(self.mixers, xs)]
        shape = xs[0].shape[2:]
        tokens = torch.stack([flatten(x, o) for x, o in zip(xs, self.orders)], dim=1)  # (B, G, L, C)
        out = self.grouped(tokens)
        return [unflatten(out[:, i], o, shape) for i, o in enumerate(self.orders)]


class AMF3D(nn.Module):
    """Adaptive multi-branch Mamba fusion (eqs. 1-3). Each channel group scans in its own direction."""

    def __init__(self, channels: int, cfg: MambaLiteConfig):
        super().__init__()
        if channels % cfg.groups:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({cfg.groups})")
        self.groups = cfg.groups
        cg = channels // cfg.groups
        self.norm = ChannelLayerNorm(channels)
        self.mixer = MultiDirectionalMixer(cg, cfg.group_orders(), cfg)
        self.alpha = nn.Parameter(torch.tensor(float(cfg.alpha_init)))
        self.s_dw = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.s_pw = nn.Conv3d(channels, channels, 1)
        self.t_dw = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.t_pw = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = torch.chunk(self.norm(x), self.groups, dim=1)
        z = torch.cat([zi + self.alpha * xi for zi, xi in zip(self.mixer(list(xs)), xs)], dim=1)
        s = torch.sigmoid(self.s_pw(self.s_dw(z)))  # spatial gate S (eq. 2)
        zs = s * z  # eq. 3
        t = self.t_pw(self.t_dw(zs)) + zs  # transform stage T with residual
        return t + x  # F_int = T + X


class LGFM3D(nn.Module):
    """Local-global feature mixing (eq. 4) as a residual unit: local depthwise conv + multi-head attention."""

    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.heads = heads
        self.local = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.attn_norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.fuse = nn.Conv3d(2 * channels, channels, 1)
        self.fuse_norm = ChannelLayerNorm(channels)
        self.out_dw = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = f.shape
        f_local = self.local(f)
        tokens = self.attn_norm(f.flatten(2).transpose(1, 2))  # (B, L, C)
        qkv = self.qkv(tokens).reshape(b, -1, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])  # (B, heads, L, C/heads)
        attn = attn.transpose(1, 2).reshape(b, -1, c)
        f_global = self.proj(attn).transpose(1, 2).reshape(b, c, d, h, w)
        f_lg = self.out_dw(F.gelu(self.fuse_norm(self.fuse(torch.cat([f_local, f_global], dim=1)))))
        return f + f_lg


def _dw_norm_relu(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(channels, channels, 3, padding=1, groups=channels), norm3d(channels), nn.ReLU(inplace=True))


class CGA3D(nn.Module):
    """Cross-gated attention on a skip connection (eqs. 5-7). x: encoder skip, g: decoder map."""

    def __init__(self, channels: int, cfg: MambaLiteConfig):
        super().__init__()
        if channels % cfg.groups:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({cfg.groups})")
        g_ = cfg.groups
        cg = channels // g_
        f_int = max(channels // 2, 4)
        orders = cfg.group_orders()
        self.groups = g_
        self.norm_x = nn.ModuleList([ChannelLayerNorm(cg) for _ in range(g_)])
        self.norm_g = nn.ModuleList([ChannelLayerNorm(cg) for _ in range(g_)])
        self.mixer = MultiDirectionalMixer(cg, orders + orders, cfg)  # x streams, then g streams
        self.dw_x = nn.ModuleList([_dw_norm_relu(cg) for _ in range(g_)])
        self.dw_g = nn.ModuleList([_dw_norm_relu(cg) for _ in range(g_)])
        self.branch_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(cg, cg, 3, padding=1, groups=cg),
                    norm3d(cg),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(cg, f_int, 1),
                    norm3d(f_int),
                    nn.ReLU(inplace=True),
                )
                for _ in range(g_)
            ]
        )
        self.combine = nn.Sequential(
            nn.Conv3d(g_ * f_int, f_int, 3, padding=1), norm3d(f_int), nn.ReLU(inplace=True), nn.Conv3d(f_int, 1, 1)
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        xs = torch.chunk(x, self.groups, dim=1)
        gs = torch.chunk(g, self.groups, dim=1)
        mixed = self.mixer([self.norm_x[i](xs[i]) for i in range(self.groups)] + [self.norm_g[i](gs[i]) for i in range(self.groups)])
        branches = []
        for i in range(self.groups):
            hx, hg = mixed[i], mixed[self.groups + i]
            cross = hx * torch.sigmoid(self.dw_g[i](hg)) + hg * torch.sigmoid(self.dw_x[i](hx))  # eq. 5
            branches.append(self.branch_proj[i](cross))
        psi = torch.sigmoid(self.combine(torch.cat(branches, dim=1)))  # eq. 6, one channel
        return x * psi  # eq. 7


class MambaStage(nn.Module):
    """Projection -> GSC3D -> AMF3D -> LGFM3D -> LayerNorm + projection (AMF's "LN & Proj").

    lgfm overrides cfg.use_lgfm for this stage (MambaLite3D drops the global attention on large maps)."""

    def __init__(self, cin: int, cout: int, cfg: MambaLiteConfig, lgfm: Optional[bool] = None):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv3d(cin, cout, 1), norm3d(cout))
        self.gsc = GSC3D(cout) if cfg.use_gsc else nn.Identity()
        self.amf = AMF3D(cout, cfg) if cfg.use_amf else BasicResBlock(cout, 3)
        use_lgfm = cfg.use_lgfm if lgfm is None else lgfm
        self.lgfm = LGFM3D(cout, cfg.heads) if use_lgfm else nn.Identity()
        self.out_norm = ChannelLayerNorm(cout)
        self.out_proj = nn.Conv3d(cout, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.gsc(x)
        x = self.amf(x)
        x = self.lgfm(x)
        return self.out_proj(self.out_norm(x))
