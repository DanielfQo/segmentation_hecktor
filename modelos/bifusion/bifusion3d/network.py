"""MambaLite3D con codificador siames PET/CT y fusion por nivel.

La red reutiliza las clases de mambalite3d (Encoder, Decoder, MambaStage, init_weights) sin modificarlas:

- Codificador siames con pesos compartidos: la entrada (B, 2, D, H, W) se pasa como (2B, 1, D, H, W), asi CT y PET
  recorren los mismos filtros por separado (la InstanceNorm normaliza cada muestra, o sea cada modalidad, por su
  cuenta) y se duplican los mapas de caracteristicas sin anadir parametros.
- En cada nivel siames un modulo de fusion combina CT y PET y su salida es el skip de ese nivel. Si el codificador
  siames no llega al fondo (variante ligera), el flujo sigue con la salida fusionada del ultimo nivel siames.
- Decoder, cuello de botella, deep supervision y nombres de parametros son los de MambaLite3D, de modo que con
  `FusionSpec()` (sin fusion) la red es identica al base y acepta sus pesos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from mambalite3d.models import network as base_network
from mambalite3d.models.config import MambaLiteConfig, make_config
from mambalite3d.models.modules import MambaStage
from mambalite3d.models.network import Decoder, Encoder, init_weights, stage_features

from .fusion import BiFusionStage, CGAFusion3D, CrossMambaFusion3D, GatedFusion3D


@dataclass(frozen=True)
class FusionSpec:
    """Que niveles son siameses y con que operador se fusionan.

    siamese_levels: numero de niveles, desde el 0, con codificador siames; "all" = todos.
    deep: operador en los niveles profundos ("gate" = la misma compuerta que en los niveles finos):
          "xmamba" = Cross/Concat Mamba (Sigma) en los niveles con Mamba del codificador;
          "cga" = CGA3D como compuerta PET<->CT en los niveles con CGA de MambaLite3D.
    """

    siamese_levels: Union[int, str] = 0
    deep: str = "gate"
    bifusion_stage: bool = False  # paso 4: cada MambaStage con una rama CNN paralela (BiFusion de TransFuse)
    description: str = ""


# variante -> (variante base de mambalite3d de la que hereda canales y modulos, especificacion de la fusion)
FUSION_VARIANTS: Dict[str, Tuple[str, FusionSpec]] = {
    "siam_light": ("base", FusionSpec(siamese_levels=2, description="paso 1: siames en niveles 0-1 + compuerta")),
    "siam_full": ("base", FusionSpec(siamese_levels="all", deep="xmamba",
                                     description="paso 2: siames completo; compuerta arriba, Cross/Concat Mamba abajo")),
    "siam_gate": ("base", FusionSpec(siamese_levels="all", deep="gate",
                                     description="control del paso 2: siames completo con compuertas en todos los niveles")),
    "siam_cga": ("base", FusionSpec(siamese_levels="all", deep="cga",
                                    description="paso 3: siames completo; compuerta arriba, CGA3D PET<->CT abajo")),
    "base_bifusion": ("base", FusionSpec(bifusion_stage=True,
                                         description="paso 4: base (fusion temprana) con BiFusion CNN<->Mamba en cada MambaStage")),
    "siam_full_bifusion": ("base", FusionSpec(siamese_levels="all", deep="xmamba", bifusion_stage=True,
                                              description="pasos 2 + 4: siam_full con BiFusion CNN<->Mamba")),
}

DEEP_OPS = ("gate", "xmamba", "cga")


class BiFusionNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        strides: Sequence[Sequence[int]],
        kernels: Sequence[Sequence[int]],
        cfg: Optional[MambaLiteConfig] = None,
        spec: Optional[FusionSpec] = None,
        deep_supervision: bool = True,
    ):
        super().__init__()
        cfg = cfg or MambaLiteConfig()
        spec = spec or FusionSpec()
        n = len(strides)
        if n < 2:
            raise ValueError("need at least 2 stages")
        if cfg.baseline:
            raise ValueError("las variantes de fusion parten de MambaLite3D, no de la U-Net de referencia")
        n_siam = n if spec.siamese_levels == "all" else int(spec.siamese_levels)
        if not 0 <= n_siam <= n:
            raise ValueError(f"siamese_levels={spec.siamese_levels} fuera de rango para {n} niveles")
        if n_siam and in_channels != 2:
            raise ValueError("el codificador siames necesita 2 canales de entrada (CT, PET)")
        if spec.deep not in DEEP_OPS:
            raise ValueError(f"operador de fusion profundo desconocido {spec.deep!r}; opciones {DEEP_OPS}")

        feats = stage_features(n, cfg.features, keep="last")
        mamba_levels = set(range(max(n - cfg.n_mamba_levels, 1), n))
        cga_levels = set(range(max(n - cfg.n_cga_levels, 0), n)) if cfg.use_cga else set()
        self.cfg = cfg
        self.spec = spec
        self.feats = feats
        self.mamba_levels = sorted(mamba_levels)
        self.cga_levels = sorted(cga_levels)
        self.strides = [list(s) for s in strides]
        self.n_siamese = n_siam

        # mismos modulos (y nombres de parametros) que MambaLite3D; el siames solo cambia el canal de entrada
        self.encoder = Encoder(1 if n_siam else in_channels, feats, strides, kernels, cfg, mamba_levels)
        self.bottleneck = MambaStage(feats[-1], cfg.bottleneck_features, cfg)
        self.decoder = Decoder(
            feats, cfg.bottleneck_features, num_classes, strides, kernels, cfg, mamba_levels, cga_levels, deep_supervision
        )
        if spec.bifusion_stage:
            self._use_bifusion_stages(cfg)

        deep_levels = self._deep_levels(spec.deep, mamba_levels, cga_levels)
        self.fusion_ops = [spec.deep if lvl in deep_levels else "gate" for lvl in range(n_siam)]
        self.fusion = nn.ModuleList(
            [self._make_fusion(op, feats[lvl], kernels[lvl], cfg) for lvl, op in enumerate(self.fusion_ops)]
        )
        init_weights(self)

    def _use_bifusion_stages(self, cfg: MambaLiteConfig) -> None:
        """Sustituye cada MambaStage (codificador, cuello de botella y decoder) por un BiFusionStage equivalente."""
        for lvl in self.mamba_levels:
            down, old = self.encoder.stages[lvl]
            if not isinstance(old, MambaStage):
                raise TypeError(f"se esperaba MambaStage en el nivel {lvl} del codificador")
            self.encoder.stages[lvl] = nn.Sequential(down, BiFusionStage(old.proj[0].in_channels, self.feats[lvl], cfg))
            if not isinstance(self.decoder.block[lvl], MambaStage):
                raise TypeError(f"se esperaba MambaStage en el nivel {lvl} del decoder")
            self.decoder.block[lvl] = BiFusionStage(self.feats[lvl], self.feats[lvl], cfg)
        self.bottleneck = BiFusionStage(self.feats[-1], cfg.bottleneck_features, cfg)

    @staticmethod
    def _deep_levels(op: str, mamba_levels, cga_levels):
        if op == "xmamba":
            return set(mamba_levels)  # donde el codificador ya tiene MambaStage (L <= 512 tokens con 32x128x128)
        if op == "cga":
            if not cga_levels:
                raise ValueError("la fusion 'cga' necesita niveles CGA (use_cga=True, n_cga_levels > 0)")
            return set(cga_levels)  # los mismos niveles en los que MambaLite3D usa CGA3D en los skips
        return set()

    @staticmethod
    def _make_fusion(op: str, channels: int, kernel, cfg: MambaLiteConfig) -> nn.Module:
        if op == "gate":
            return GatedFusion3D(channels, kernel)
        if op == "xmamba":
            return CrossMambaFusion3D(channels, cfg)
        if op == "cga":
            return CGAFusion3D(channels, cfg)
        raise ValueError(f"operador de fusion desconocido {op!r}")

    def forward(self, x: torch.Tensor):
        skips = []
        if self.n_siamese:
            b = x.shape[0]
            h = x.reshape(b * 2, 1, *x.shape[2:])  # [b0 CT, b0 PET, b1 CT, b1 PET, ...]
            for lvl, stage in enumerate(self.encoder.stages):
                h = stage(h)
                if lvl < self.n_siamese:
                    pair = h.reshape(b, 2, *h.shape[1:])
                    fused = self.fusion[lvl](pair[:, 0], pair[:, 1])
                    skips.append(fused)
                    if lvl == self.n_siamese - 1:
                        h = fused  # el resto del codificador (si queda) es de un solo flujo
                else:
                    skips.append(h)
        else:
            h = x
            for stage in self.encoder.stages:
                h = stage(h)
                skips.append(h)
        return self.decoder(skips, self.bottleneck(skips[-1]))


def build_model(
    variant: str,
    strides,
    kernels,
    in_channels: int = 2,
    num_classes: int = 3,
    deep_supervision: bool = True,
    **overrides,
) -> nn.Module:
    """Igual que mambalite3d.models.network.build_model, mas las variantes de FUSION_VARIANTS."""
    if variant in FUSION_VARIANTS:
        base_variant, spec = FUSION_VARIANTS[variant]
        cfg = make_config(base_variant, **overrides)
        return BiFusionNet(in_channels, num_classes, strides, kernels, cfg, spec, deep_supervision)
    cfg = make_config(variant, **overrides)  # base, wide, unet, ablaciones... exactamente como en mambalite3d
    return base_network.MambaLite3D(in_channels, num_classes, strides, kernels, cfg, deep_supervision)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
