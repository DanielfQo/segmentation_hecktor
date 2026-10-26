"""Ejecuta el pipeline de preprocesado real guardando cada etapa intermedia.

Usa LAS MISMAS funciones que el entrenamiento (`mambalite3d.data.io` y
`mambalite3d.data.augment`), no una reimplementacion: lo unico que hace este
modulo es quedarse con los arrays que `load_case` normalmente descarta. Si
cambias el preprocesado, esto cambia con el.

Etapas:
  0 crudo          CT y PET en su rejilla nativa (ya reorientados a LPS)
  1 ROI            la caja que encuentra `compute_roi`, sobre el volumen crudo
  2 remuestreado   recorte a la rejilla comun (target_spacing_xyz)
  3 normalizado    ventana del CT a [0,1] y z-score del PET
  4 patch          parche con contexto centrado en un voxel de tumor
  5+ aumentado     realizaciones del aumento de datos

Lo consume `visor_pipeline.py`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
RAIZ = BASE.parent
if str(RAIZ / "modelos" / "mamba") not in sys.path:
    sys.path.insert(0, str(RAIZ / "modelos" / "mamba"))

import SimpleITK as sitk  # noqa: E402
import torch  # noqa: E402

from mambalite3d.config import DEFAULTS, load_config  # noqa: E402
from mambalite3d.data import augment as maug  # noqa: E402
from mambalite3d.data import io as mio  # noqa: E402
from mambalite3d.data.sampler import _extract  # noqa: E402


@dataclass
class Etapa:
    """Una etapa del pipeline: lo que se ve y lo que se mide."""
    indice: int
    nombre: str
    descripcion: str
    ct: np.ndarray                      # (Z, Y, X)
    pet: np.ndarray                     # (Z, Y, X), puede tener otra forma que ct
    etiqueta: np.ndarray | None         # en la rejilla del ct de esta etapa
    espaciado_zyx: tuple                # mm
    rango_ct: tuple                     # (vmin, vmax) de visualizacion
    rango_pet: tuple
    unidad_ct: str
    unidad_pet: str
    caja: dict = field(default_factory=dict)   # recuadros a dibujar, por vista
    etiqueta_pet: np.ndarray | None = None     # la misma etiqueta en la rejilla del PET
    centro_tumor: tuple | None = None          # centroide zyx, para elegir cortes

    @property
    def forma(self):
        return tuple(self.ct.shape)

    @property
    def mm3_por_voxel(self) -> float:
        return float(np.prod(self.espaciado_zyx))

    def estadisticas(self) -> dict:
        d = {
            "etapa": f"{self.indice}. {self.nombre}",
            "forma_zyx": "x".join(str(s) for s in self.ct.shape),
            "espaciado_zyx_mm": tuple(round(float(s), 2) for s in self.espaciado_zyx),
            "voxeles": int(self.ct.size),
            "mm3_por_voxel": round(self.mm3_por_voxel, 3),
            "volumen_cm3": round(self.ct.size * self.mm3_por_voxel / 1000.0, 1),
            "ct_min": float(self.ct.min()), "ct_max": float(self.ct.max()),
            "ct_media": float(self.ct.mean()), "ct_sd": float(self.ct.std()),
            "pet_min": float(self.pet.min()), "pet_max": float(self.pet.max()),
            "pet_media": float(self.pet.mean()), "pet_sd": float(self.pet.std()),
        }
        if self.etiqueta is not None:
            m = self.etiqueta
            d["frac_tumor_pct"] = round(100 * float((m > 0).mean()), 4)
            d["voxeles_gtvp"] = int((m == 1).sum())
            d["voxeles_gtvn"] = int((m == 2).sum())
        else:
            d["frac_tumor_pct"] = float("nan")
            d["voxeles_gtvp"] = d["voxeles_gtvn"] = 0
        return d


def carga_config(ruta: Path | None) -> dict:
    if ruta and Path(ruta).exists():
        return load_config(Path(ruta))
    import copy
    return copy.deepcopy(DEFAULTS)


def _rango_pet(vol) -> tuple:
    """Rango de visualizacion robusto: el maximo del PET suele ser un unico
    voxel (vejiga, cerebro) que aplasta todo lo demas en la escala de color."""
    alto = float(np.percentile(vol[vol > 0], 99.5)) if (vol > 0).any() else 1.0
    return (0.0, max(alto, 1e-3))


def _caja_en_voxeles(roi_lo, roi_hi, origen, espaciado):
    """Caja fisica (LPS, direccion identidad) -> indices de voxel x, y, z."""
    lo = (np.asarray(roi_lo) - np.asarray(origen)) / np.asarray(espaciado)
    hi = (np.asarray(roi_hi) - np.asarray(origen)) / np.asarray(espaciado)
    return {"x": (lo[0], hi[0]), "y": (lo[1], hi[1]), "z": (lo[2], hi[2])}


def construye_etapas(files: mio.CaseFiles, cfg: dict, semilla_patch: int = 7,
                     semillas_aug=(1, 2), forzar_ultimo: bool = True) -> list[Etapa]:
    """Pasa un caso por el pipeline y devuelve la lista de etapas."""
    data_cfg, aug_cfg = cfg["data"], cfg["augment"]
    patch_zyx = tuple(int(v) for v in cfg["model"]["patch_size_zyx"])
    spacing_xyz = tuple(float(s) for s in data_cfg["target_spacing_xyz"])
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    w_lo, w_hi = data_cfg["ct_window"]

    ct_img = mio.read_lps(files.ct)
    pet_img = mio.read_lps(files.pet)
    lab_img = mio.read_lps(files.label) if files.label else None

    ct0 = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    pet0 = sitk.GetArrayFromImage(pet_img).astype(np.float32)
    sp_ct0 = tuple(float(s) for s in ct_img.GetSpacing()[::-1])
    # La etiqueta viene en su propia rejilla, que en varios casos no coincide ni
    # con la del CT (un corte de mas en Z) ni con la del PET. Para poder verla
    # encima del volumen crudo se lleva a cada rejilla por vecino mas cercano;
    # es solo para mirar, el pipeline no hace esto hasta la etapa 2.
    lab0 = lab0_pet = None
    if lab_img is not None:
        lab0 = sitk.GetArrayFromImage(
            sitk.Resample(lab_img, ct_img, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)).astype(np.uint8)
        lab0_pet = sitk.GetArrayFromImage(
            sitk.Resample(lab_img, pet_img, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)).astype(np.uint8)
    etapas = []

    def _centroide(m):
        if m is None or not (m > 0).any():
            return None
        return tuple(int(v) for v in np.argwhere(m > 0).mean(axis=0))

    etapas.append(Etapa(
        0, "crudo",
        "CT y PET tal y como vienen del escaner, reorientados a LPS. Cada "
        "modalidad esta en su propia rejilla: el PET es 3-4 veces mas grueso "
        "en plano.",
        ct0, pet0, lab0, sp_ct0, (-1000, 1000), _rango_pet(pet0),
        "HU", "SUV", etiqueta_pet=lab0_pet, centro_tumor=_centroide(lab0)))

    # --- 1. ROI -------------------------------------------------------------
    roi_lo, roi_hi = mio.compute_roi(ct_img, pet_img, data_cfg["roi"], spacing_xyz)
    caja = _caja_en_voxeles(roi_lo, roi_hi, ct_img.GetOrigin(), ct_img.GetSpacing())
    extension = np.asarray(roi_hi) - np.asarray(roi_lo)
    etapas.append(Etapa(
        1, "ROI detectado",
        "La caja se busca sin mirar la etiqueta: cuerpo = (CT > {:.0f} HU) o "
        "(PET > {:.1f} SUV), se queda la componente conexa del paciente (quita "
        "la camilla) y el techo es el corte mas craneal con SUV > {:.1f}; de ahi "
        "baja {:.0f} mm. Extension: {:.0f}x{:.0f}x{:.0f} mm.".format(
            data_cfg["roi"]["body_hu"], data_cfg["roi"]["body_suv"],
            data_cfg["roi"]["suv_anchor"], data_cfg["roi"]["down_mm"], *extension),
        ct0, pet0, lab0, sp_ct0, (-1000, 1000), _rango_pet(pet0),
        "HU", "SUV", caja={"axial": caja, "coronal": caja},
        etiqueta_pet=lab0_pet, centro_tumor=_centroide(lab0)))

    # --- 2. remuestreado ----------------------------------------------------
    size = np.maximum(np.floor((np.asarray(roi_hi) - np.asarray(roi_lo))
                               / np.asarray(spacing_xyz)).astype(int) + 1, 1)
    ref = mio._grid(roi_lo, spacing_xyz, size)
    ct2 = sitk.GetArrayFromImage(mio._resample(ct_img, ref, sitk.sitkLinear, -1024,
                                               sitk.sitkFloat32))
    pet2 = sitk.GetArrayFromImage(mio._resample(pet_img, ref, sitk.sitkLinear, 0,
                                                sitk.sitkFloat32))
    lab2 = (sitk.GetArrayFromImage(mio._resample(lab_img, ref,
                                                 sitk.sitkNearestNeighbor, 0,
                                                 sitk.sitkUInt8))
            if lab_img else None)
    etapas.append(Etapa(
        2, "remuestreado",
        "CT, PET y etiqueta caen de una sola vez sobre la rejilla comun de "
        f"{spacing_xyz} mm (trilineal para las imagenes, vecino mas cercano para "
        "la etiqueta). A partir de aqui las tres comparten voxel.",
        ct2, pet2, lab2, spacing_zyx, (-1000, 1000), _rango_pet(pet2), "HU", "SUV",
        centro_tumor=_centroide(lab2)))

    # --- 3. normalizado -----------------------------------------------------
    ct3, pet3 = mio._normalize(ct2, pet2, data_cfg)
    etapas.append(Etapa(
        3, "normalizado",
        f"CT recortado a la ventana [{w_lo:.0f}, {w_hi:.0f}] HU y llevado a "
        f"[0, 1]; PET {data_cfg['pet_norm']} dentro del ROI. El z-score hace el "
        "PET independiente de la escala de SUV, que no es uniforme entre centros.",
        ct3, pet3, lab2, spacing_zyx, (0, 1), (-1, 4), "norm.", "z",
        centro_tumor=_centroide(lab2)))

    # --- 4. patch con contexto ----------------------------------------------
    imagen = np.stack([ct3, pet3])
    ext = maug.context_size(patch_zyx, aug_cfg["context_margin"])
    rng = np.random.default_rng(semilla_patch)
    if lab2 is not None and (lab2 > 0).any():
        clases = [c for c in (1, 2) if (lab2 == c).any()]
        coords = np.argwhere(lab2 == clases[rng.integers(len(clases))])
        centro = coords[rng.integers(len(coords))]
        origen_centro = "centrado en un voxel de tumor (fg_fraction del sampler)"
    else:
        centro = np.array([s // 2 for s in ct3.shape])
        origen_centro = "centrado en el ROI (este caso no tiene etiqueta)"
    pad = (float(imagen[0].min()), float(imagen[1].min()))
    patch = _extract(imagen, centro, ext, pad)
    patch_lab = _extract(lab2, centro, ext, 0) if lab2 is not None else None
    etapas.append(Etapa(
        4, "patch + contexto",
        f"Parche de {tuple(ext)} voxeles (el de entrenamiento es {patch_zyx}; el "
        f"margen del {100*aug_cfg['context_margin']:.0f}% es el contexto que el "
        f"aumento consume al rotar y escalar), {origen_centro}.",
        patch[0], patch[1], patch_lab, spacing_zyx, (0, 1), (-1, 4), "norm.", "z",
        centro_tumor=_centroide(patch_lab)))

    # --- 5+. aumentos -------------------------------------------------------
    img_t = torch.from_numpy(patch.astype(np.float32))[None]
    lab_t = torch.from_numpy((patch_lab if patch_lab is not None
                              else np.zeros(ext, np.uint8)).astype(np.float32))[None, None]
    planes = [(s, dict(aug_cfg), False) for s in semillas_aug]
    if forzar_ultimo:
        forzado = dict(aug_cfg)
        for k in list(forzado):
            if k.startswith("p_"):
                forzado[k] = 1.0
        planes.append((semillas_aug[-1] + 1000, forzado, True))

    for k, (semilla, c, es_forzado) in enumerate(planes):
        gen = torch.Generator().manual_seed(int(semilla))
        a_img, a_lab = maug.augment_batch(img_t, lab_t, patch_zyx, spacing_zyx,
                                          c, gen, enabled=True)
        a_img = a_img[0].numpy()
        a_lab = a_lab[0, 0].numpy().astype(np.uint8)
        if es_forzado:
            nombre = "aumento forzado"
            desc = ("La misma realizacion pero con todas las probabilidades a 1, "
                    "para ver de que es capaz el aumento. En entrenamiento son "
                    f"p_rotacion={aug_cfg['p_rotation']}, p_escala={aug_cfg['p_scale']}, "
                    f"p_gamma={aug_cfg['p_gamma']}, p_ruido={aug_cfg['p_noise']}, "
                    f"p_desenfoque={aug_cfg['p_blur']}: por eso la mayoria de "
                    "parches salen casi intactos.")
        else:
            nombre = f"aumentado #{k + 1}"
            desc = (f"Rotacion, escala, espejo, brillo, contraste, gamma, ruido y "
                    f"desenfoque con las probabilidades reales del config "
                    f"(semilla {semilla}). Recorta a {patch_zyx}, que es lo que "
                    "entra en la red.")
        etapas.append(Etapa(5 + k, nombre, desc, a_img[0], a_img[1], a_lab,
                            spacing_zyx, (0, 1), (-1, 4), "norm.", "z",
                            centro_tumor=_centroide(a_lab)))
    return etapas


def tabla_etapas(etapas: list[Etapa], caso: str, centro: str):
    """Las estadisticas de todas las etapas, como DataFrame."""
    import pandas as pd
    filas = []
    previo = None
    for e in etapas:
        d = {"caso": caso, "centro": centro, **e.estadisticas()}
        if previo is not None:
            d["voxeles_x"] = round(d["voxeles"] / previo["voxeles"], 4)
        else:
            d["voxeles_x"] = 1.0
        filas.append(d)
        previo = d
    return pd.DataFrame(filas)


def elige_casos(raiz_datos: Path, salidas: Path, n: int) -> list[str]:
    """Casos variados: uno por tercil de complejidad y de centros distintos."""
    ruta = salidas / "indice_complejidad.csv"
    if ruta.exists():
        import pandas as pd
        d = pd.read_csv(ruta)
        elegidos, usados = [], set()
        for t in ("facil", "medio", "difficil"):
            sub = d[d["tercil"] == t]
            if sub.empty:
                continue
            cand = sub[~sub["centro"].isin(usados)]
            cand = cand if not cand.empty else sub
            fila = cand.iloc[len(cand) // 2]
            elegidos.append(str(fila["caso"]))
            usados.add(str(fila["centro"]))
        if elegidos:
            return elegidos[:n]
    centros, elegidos = set(), []
    for p in sorted(raiz_datos.iterdir()):
        if not p.is_dir():
            continue
        pref = p.name.rsplit("-", 1)[0]
        if pref not in centros:
            centros.add(pref)
            elegidos.append(p.name)
    return elegidos[:n]
