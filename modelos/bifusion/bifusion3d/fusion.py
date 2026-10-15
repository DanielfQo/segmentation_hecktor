"""Modulos de fusion PET/CT.

Paso 1  GatedFusion3D       compuerta de contexto canal x espacio con modulacion residual
                            (mapas de fusion por voxel de Kumar et al., IEEE TMI 2020; CGM de vMambaX, 2025).
Paso 2  CrossGroupedMamba   scan selectivo en el que cada grupo decodifica su estado con la matriz C de otro grupo
                            (Cross Mamba Block de Sigma, WACV 2025).
        CrossMambaFusion3D  Cross Mamba Block + Concat Mamba Block de Sigma, adaptados a 3D con los ordenes de AMF3D.
Paso 3  CGAFusion3D         CGA3D de MambaLite3D como compuerta bidireccional PET<->CT: psi * CT + (1 - psi) * PET
                            (interaccion bidireccional entre modalidades como en H2ASeg, MICCAI 2024, pero mas barata).
Paso 4  BiFusionStage       MambaStage con una rama CNN paralela fusionada como el BiFusion de TransFuse (MICCAI 2021):
                            SE sobre la rama global (Mamba + atencion), atencion espacial sobre la rama CNN, producto
                            elemento a elemento entre ambas y bloque residual sobre la concatenacion.

Todos reciben las caracteristicas de CT y de PET con la misma forma (B, C, D, H, W) y devuelven (B, C, D, H, W).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mambalite3d.models import ssm as _ssm
from mambalite3d.models.blocks import GSC3D, BasicResBlock, ChannelLayerNorm, norm3d
from mambalite3d.models.config import MambaLiteConfig
from mambalite3d.models.modules import AMF3D, CGA3D, LGFM3D
from mambalite3d.models.scans import flatten, unflatten

Kernel = Union[int, Sequence[int]]


def _pad(kernel: Kernel):
    if isinstance(kernel, int):
        return kernel // 2
    return [k // 2 for k in kernel]


# --------------------------------------------------------------------------------------------------------------------
# Paso 1: compuerta canal x espacio
# --------------------------------------------------------------------------------------------------------------------
class GatedFusion3D(nn.Module):
    """Fusion con compuerta de contexto.

    z = [CT, PET] (2C canales). La compuerta de canal g_c = sigmoid(MLP(GAP(z))) da un peso por canal y modalidad; la
    espacial g_s = sigmoid(conv(z)) da un mapa por modalidad. Cada modalidad se modula de forma residual,
    f * (1 + g_c * g_s): con compuertas en 0 queda la entrada tal cual y nunca se suprime por debajo de ella. Una
    proyeccion 1x1 + InstanceNorm + LeakyReLU devuelve C canales, con la misma convencion que los bloques del base.

    `kernel` es el de la convolucion espacial; la red usa el kernel del nivel ([1, 3, 3] en el nivel 0 de HECKTOR).
    """

    def __init__(self, channels: int, kernel: Kernel = 3, reduction: int = 4):
        super().__init__()
        c2 = 2 * channels
        hidden = max(c2 // reduction, 4)
        self.channels = channels
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c2, hidden, 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, c2, 1),
        )
        self.spatial_gate = nn.Conv3d(c2, 2, kernel, padding=_pad(kernel))
        self.proj = nn.Sequential(nn.Conv3d(c2, channels, 1), norm3d(channels), nn.LeakyReLU(0.01, inplace=True))

    def gates(self, ct: torch.Tensor, pet: torch.Tensor):
        """Compuertas (en [0, 1]) de CT y de PET, cada una (B, C, D, H, W) por difusion."""
        z = torch.cat([ct, pet], dim=1)
        g_c = torch.sigmoid(self.channel_gate(z))  # (B, 2C, 1, 1, 1)
        g_s = torch.sigmoid(self.spatial_gate(z))  # (B, 2, D, H, W)
        c = self.channels
        return g_c[:, :c] * g_s[:, :1], g_c[:, c:] * g_s[:, 1:]

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        g_ct, g_pet = self.gates(ct, pet)
        return self.proj(torch.cat([ct * (1 + g_ct), pet * (1 + g_pet)], dim=1))


# --------------------------------------------------------------------------------------------------------------------
# Paso 2: fusion con Mamba (Sigma: Cross Mamba Block + Concat Mamba Block)
# --------------------------------------------------------------------------------------------------------------------
class CrossGroupedMamba(_ssm.GroupedMambaTorch):
    """GroupedMambaTorch de mambalite3d en el que un grupo puede decodificar su estado con la C de otro grupo.

    En el scan selectivo y_t = C_t h_t + D x_t, la matriz C "lee" el estado oculto. En el Cross Mamba Block de Sigma
    cada modalidad conserva sus A, B y delta (y por tanto su propio estado), pero lo lee con la C calculada a partir de
    la otra modalidad, de forma parecida a una atencion cruzada. `c_from[i]` es el grupo cuya C usa el grupo i; sin
    `c_from` el resultado es exactamente el de GroupedMambaTorch (mismos parametros e inicializacion).

    El scan se llama como `ssm.selective_scan` (atributo del modulo), asi tools/efficiency puede contar sus operaciones.
    """

    def forward(self, x: torch.Tensor, c_from: Optional[Sequence[int]] = None) -> torch.Tensor:  # (B, G, L, d_model)
        bsz, g, seqlen, _ = x.shape
        di, n, r = self.d_inner, self.d_state, self.dt_rank
        xz = torch.einsum("bgld,ged->bgle", x, self.in_proj_w)
        xi, z = xz.chunk(2, dim=-1)  # (B, G, L, di)
        xi = xi.permute(0, 1, 3, 2).reshape(bsz, g * di, seqlen)
        xi = F.silu(F.conv1d(xi, self.conv_w, self.conv_b, padding=self.d_conv - 1, groups=g * di)[..., :seqlen])
        dbl = torch.einsum("bgdl,ged->bgle", xi.reshape(bsz, g, di, seqlen), self.x_proj_w)  # (B, G, L, r + 2n)
        dt, b_mat, c_mat = torch.split(dbl, [r, n, n], dim=-1)
        if c_from is not None:
            if len(c_from) != g:
                raise ValueError(f"c_from tiene {len(c_from)} entradas para {g} grupos")
            c_mat = c_mat[:, list(c_from)]  # el grupo i lee su estado con la C del grupo c_from[i]
        dt = torch.einsum("bglr,gdr->bgdl", dt, self.dt_proj_w).reshape(bsz, g * di, seqlen)
        work = torch.float64 if x.dtype == torch.float64 else torch.float32
        a = -torch.exp(self.A_log.to(work))  # (G, di, n)
        d_skip = self.D.to(work)
        bias = self.dt_proj_b.to(work)
        zz = z.permute(0, 1, 3, 2).reshape(bsz, g * di, seqlen)
        per_group = bsz * di * seqlen * min(self.chunk_size, seqlen) * n
        step = max(1, min(g, int(self.max_scan_elems // max(per_group, 1))))
        ys = []
        for s0 in range(0, g, step):
            s1 = min(g, s0 + step)
            ch = slice(s0 * di, s1 * di)
            ys.append(
                _ssm.selective_scan(
                    xi[:, ch],
                    dt[:, ch],
                    a[s0:s1].reshape(-1, n),
                    b_mat[:, s0:s1].transpose(2, 3),
                    c_mat[:, s0:s1].transpose(2, 3),
                    d_skip[s0:s1].reshape(-1),
                    z=zz[:, ch],
                    delta_bias=bias[s0:s1].reshape(-1),
                    delta_softplus=True,
                    chunk_size=self.chunk_size,
                )
            )
        y = ys[0] if len(ys) == 1 else torch.cat(ys, dim=1)
        y = y.reshape(bsz, g, di, seqlen)
        return torch.einsum("bgdl,ged->bgle", y, self.out_proj_w)


class CrossMambaFusion3D(nn.Module):
    """Fusion PET/CT con Mamba: Cross Mamba Block (CroMB) + Concat Mamba Block (ConMB) de Sigma, en 3D.

    Los canales se reparten en cfg.groups grupos y cada grupo recorre el volumen en su propio orden, como en AMF3D
    (fwd, rev, slice, ap). CT y PET usan el mismo orden en cada grupo, asi el token t de una modalidad y el de la otra
    son el mismo voxel.

    1. CroMB: cada modalidad escanea su secuencia con sus propias A, B y delta, pero lee su estado con la C de la otra
       (intercambio de C); se suma a la entrada (residual).
    2. ConMB: en cada grupo se concatena [CT; PET] (2L tokens) y se escanea hacia delante y, con otros parametros, la
       secuencia invertida: hacia delante el PET ve al CT y hacia atras el CT ve al PET. Se suman ambos sentidos, se
       separa por modalidad y se anade con un factor aprendido por modalidad (residual).
    3. Una proyeccion 1x1 lleva [CT, PET] (2C canales) a C canales.
    """

    def __init__(self, channels: int, cfg: MambaLiteConfig):
        super().__init__()
        k = cfg.groups
        if channels % k:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({k})")
        self.k = k
        self.orders = cfg.group_orders()
        cg = channels // k
        mamba_kw = dict(d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand, chunk_size=cfg.chunk_size)
        self.norm_ct = ChannelLayerNorm(channels)
        self.norm_pet = ChannelLayerNorm(channels)
        self.cross = CrossGroupedMamba(2 * k, cg, **mamba_kw)  # grupos [CT_1..CT_k, PET_1..PET_k]
        self.norm2_ct = ChannelLayerNorm(channels)
        self.norm2_pet = ChannelLayerNorm(channels)
        self.concat = _ssm.GroupedMambaTorch(2 * k, cg, **mamba_kw)  # k sentidos directos + k invertidos
        self.scale_ct = nn.Parameter(torch.ones(1))
        self.scale_pet = nn.Parameter(torch.ones(1))
        self.proj = nn.Conv3d(2 * channels, channels, 1)
        self.swap = list(range(k, 2 * k)) + list(range(k))  # CT_i lee con la C de PET_i y viceversa

    def _tokens(self, x: torch.Tensor):
        return [flatten(xi, o) for xi, o in zip(torch.chunk(x, self.k, dim=1), self.orders)]

    def _volume(self, tokens, shape) -> torch.Tensor:
        return torch.cat([unflatten(t, o, shape) for t, o in zip(tokens, self.orders)], dim=1)

    def cross_mix(self, x_ct: torch.Tensor, x_pet: torch.Tensor):
        """Salida del Cross Mamba Block (sin el residual) para CT y para PET."""
        shape = x_ct.shape[2:]
        t = torch.stack(self._tokens(self.norm_ct(x_ct)) + self._tokens(self.norm_pet(x_pet)), dim=1)  # (B, 2k, L, cg)
        y = self.cross(t, c_from=self.swap)
        return self._volume([y[:, i] for i in range(self.k)], shape), self._volume([y[:, self.k + i] for i in range(self.k)], shape)

    def concat_mix(self, m_ct: torch.Tensor, m_pet: torch.Tensor):
        """Salida del Concat Mamba Block (sin el residual) para CT y para PET."""
        shape = m_ct.shape[2:]
        length = math.prod(shape)
        seqs = [torch.cat([a, b], dim=1) for a, b in zip(self._tokens(m_ct), self._tokens(m_pet))]  # (B, 2L, cg)
        y = self.concat(torch.stack(seqs + [s.flip(1) for s in seqs], dim=1))  # (B, 2k, 2L, cg)
        both = [y[:, i] + y[:, self.k + i].flip(1) for i in range(self.k)]
        return self._volume([s[:, :length] for s in both], shape), self._volume([s[:, length:] for s in both], shape)

    def forward(self, x_ct: torch.Tensor, x_pet: torch.Tensor) -> torch.Tensor:
        y_ct, y_pet = self.cross_mix(x_ct, x_pet)
        h_ct, h_pet = x_ct + y_ct, x_pet + y_pet
        o_ct, o_pet = self.concat_mix(self.norm2_ct(h_ct), self.norm2_pet(h_pet))
        return self.proj(torch.cat([h_ct + self.scale_ct * o_ct, h_pet + self.scale_pet * o_pet], dim=1))


# --------------------------------------------------------------------------------------------------------------------
# Paso 3: CGA3D como compuerta PET <-> CT
# --------------------------------------------------------------------------------------------------------------------
class CGAFusion3D(CGA3D):
    """La CGA3D de MambaLite3D aplicada entre modalidades en lugar de entre skip y decoder.

    Con x = CT y g = PET, CGA3D ya calcula una compuerta en las dos direcciones,
    cross = h_ct * sigmoid(dw(h_pet)) + h_pet * sigmoid(dw(h_ct)), y de ella un mapa psi de un canal (ecs. 5-6 de
    MambaLiteUNet). En vez de devolver x * psi, aqui psi decide en cada voxel cuanto viene de cada modalidad:
    psi * CT + (1 - psi) * PET. Mismos parametros e inicializacion que CGA3D.
    """

    def gate(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """El mapa psi (B, 1, D, H, W) de CGA3D.forward(x, g), que devuelve x * psi."""
        xs = torch.chunk(x, self.groups, dim=1)
        gs = torch.chunk(g, self.groups, dim=1)
        mixed = self.mixer(
            [self.norm_x[i](xs[i]) for i in range(self.groups)] + [self.norm_g[i](gs[i]) for i in range(self.groups)]
        )
        branches = []
        for i in range(self.groups):
            hx, hg = mixed[i], mixed[self.groups + i]
            cross = hx * torch.sigmoid(self.dw_g[i](hg)) + hg * torch.sigmoid(self.dw_x[i](hx))
            branches.append(self.branch_proj[i](cross))
        return torch.sigmoid(self.combine(torch.cat(branches, dim=1)))

    def forward(self, x_ct: torch.Tensor, x_pet: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        psi = self.gate(x_ct, x_pet)
        return psi * x_ct + (1 - psi) * x_pet


# --------------------------------------------------------------------------------------------------------------------
# Paso 4: BiFusion CNN <-> Mamba dentro de MambaStage (TransFuse)
# --------------------------------------------------------------------------------------------------------------------
class SE3D(nn.Module):
    """Atencion de canal squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Conv3d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv3d(hidden, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.fc(x))


class SpatialAttention3D(nn.Module):
    """Atencion espacial de CBAM: un mapa a partir de la media y el maximo sobre canales (kernel 3, mapas pequenos)."""

    def __init__(self, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(s))


class ResidualFuse3D(nn.Module):
    """Bloque residual sobre la concatenacion [b, t, g] (3C -> C), con atajo 1x1 como en un bloque de ResNet."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(cin, cout, 1), norm3d(cout), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, groups=cout), norm3d(cout),
        )
        self.short = nn.Sequential(nn.Conv3d(cin, cout, 1), norm3d(cout))
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.short(x))


class BiFusionStage(nn.Module):
    """MambaStage con una rama CNN en paralelo, fusionadas con el BiFusion de TransFuse.

    MambaStage es una cadena proyeccion -> GSC3D -> AMF3D -> LGFM3D -> LN + proyeccion. Aqui, tras la proyeccion:
      rama global  t = LGFM3D(AMF3D(GSC3D(x)))   (la misma de MambaStage: Mamba multidireccional + atencion)
      rama local   g = BasicResBlock(x)          (convoluciones 3x3x3, como los niveles CNN de la red)
    y se fusionan como en TransFuse: t' = SE(t), g' = AtencionEspacial(g), b = Conv3(t W1 * g W2) y
    f = Residual([b, t', g']); al final, LN + proyeccion 1x1 como en MambaStage. Para no inflar la red, la conv 3x3x3
    de b y la del bloque residual son separables en profundidad.
    """

    def __init__(self, cin: int, cout: int, cfg: MambaLiteConfig):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv3d(cin, cout, 1), norm3d(cout))
        self.gsc = GSC3D(cout) if cfg.use_gsc else nn.Identity()
        self.amf = AMF3D(cout, cfg) if cfg.use_amf else BasicResBlock(cout, 3)
        self.lgfm = LGFM3D(cout, cfg.heads) if cfg.use_lgfm else nn.Identity()
        self.local = BasicResBlock(cout, 3)
        self.se = SE3D(cout)
        self.spatial = SpatialAttention3D(3)
        self.w_t = nn.Conv3d(cout, cout, 1)
        self.w_g = nn.Conv3d(cout, cout, 1)
        self.interact = nn.Sequential(
            nn.Conv3d(cout, cout, 3, padding=1, groups=cout), nn.Conv3d(cout, cout, 1), norm3d(cout),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.fuse = ResidualFuse3D(3 * cout, cout)
        self.out_norm = ChannelLayerNorm(cout)
        self.out_proj = nn.Conv3d(cout, cout, 1)

    def branches(self, x: torch.Tensor):
        """Ramas global (Mamba + atencion) y local (CNN) despues de la proyeccion de entrada."""
        x = self.proj(x)
        return self.lgfm(self.amf(self.gsc(x))), self.local(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, g = self.branches(x)
        b = self.interact(self.w_t(t) * self.w_g(g))
        f = self.fuse(torch.cat([b, self.se(t), self.spatial(g)], dim=1))
        return self.out_proj(self.out_norm(f))
