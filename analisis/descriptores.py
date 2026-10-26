"""Descriptores por caso del dataset HECKTOR 2025 (CT + PET + mascara GTV).

Modulo de calculo puro: no imprime ni escribe ficheros, para poder reutilizarlo
desde `caracterizacion_dataset.py` y desde los tests.

Convenciones del dataset (verificadas sobre HECKTOR2025_raw):
  - <CASO>/<CASO>__CT.nii.gz   CT en HU, rejilla fina (512x512xZ)
  - <CASO>/<CASO>__PT.nii.gz   PET en SUV, rejilla gruesa (128..256 en plano)
  - <CASO>/<CASO>.nii.gz       mascara, rejilla del CT, 1=GTVp 2=GTVn
  Las tres rejillas NO coinciden exactamente (el eje Z puede diferir en un
  corte y algunas mascaras son 512x513), asi que todo el muestreo cruzado se
  hace por afines, nunca asumiendo indices compartidos.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
from scipy import ndimage, stats

GTVP = 1
GTVN = 2

# Bordes fijos de histograma, iguales para todos los casos: es lo que permite
# comparar y promediar histogramas entre casos y entre centros.
CT_BINS = np.linspace(-1000.0, 1000.0, 65)
PET_BINS = np.linspace(0.0, 20.0, 65)

UMBRAL_CUERPO_HU = -500.0   # por encima de esto consideramos tejido, no aire
UMBRAL_METAL_HU = 2000.0    # artefacto metalico (protesis/empastes dentales)
SUV_FONDO_MIN = 0.1         # para excluir el aire del histograma PET


def centro_de(caso: str) -> str:
    """CHUM-001 -> CHUM."""
    return caso.rsplit("-", 1)[0]


def entropia_shannon(conteos: np.ndarray) -> float:
    """Entropia en bits del histograma (0 si esta vacio)."""
    total = conteos.sum()
    if total <= 0:
        return float("nan")
    p = conteos.astype(np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _muestrea_mascara(seg: np.ndarray, afin_seg, forma_destino, afin_destino,
                      recorte: tuple[slice, slice, slice]) -> np.ndarray:
    """Lleva `seg` a la rejilla destino (solo dentro de `recorte`), vecino mas cercano.

    Devuelve un array de la forma del recorte con las etiquetas de seg.
    """
    ejes = [np.arange(s.start, s.stop, dtype=np.float32) for s in recorte]
    gi, gj, gk = np.meshgrid(*ejes, indexing="ij")
    hom = np.stack([gi.ravel(), gj.ravel(), gk.ravel(),
                    np.ones(gi.size, dtype=np.float32)])
    # indices destino -> mundo -> indices de seg
    m = np.linalg.inv(afin_seg) @ afin_destino
    idx = (m @ hom)[:3]
    idx = np.rint(idx).astype(np.int64)
    dentro = np.ones(idx.shape[1], dtype=bool)
    for eje in range(3):
        dentro &= (idx[eje] >= 0) & (idx[eje] < seg.shape[eje])
    fuera = np.zeros(idx.shape[1], dtype=seg.dtype)
    idx = np.clip(idx, 0, np.array(seg.shape)[:, None] - 1)
    valores = seg[idx[0], idx[1], idx[2]]
    valores = np.where(dentro, valores, fuera)
    return valores.reshape(gi.shape)


def _recorte_destino(bbox_seg, afin_seg, afin_destino, forma_destino, margen_mm):
    """Caja del tumor expresada en la rejilla destino, con margen en mm."""
    (i0, i1), (j0, j1), (k0, k1) = bbox_seg
    esquinas = np.array([[i, j, k, 1.0]
                         for i in (i0, i1) for j in (j0, j1) for k in (k0, k1)]).T
    m = np.linalg.inv(afin_destino) @ afin_seg
    idx = (m @ esquinas)[:3]
    zooms = np.sqrt((afin_destino[:3, :3] ** 2).sum(axis=0))
    margen_vox = np.ceil(margen_mm / np.maximum(zooms, 1e-6)).astype(int)
    recorte = []
    for eje in range(3):
        lo = int(np.floor(idx[eje].min())) - margen_vox[eje]
        hi = int(np.ceil(idx[eje].max())) + margen_vox[eje] + 1
        lo = max(lo, 0)
        hi = min(hi, forma_destino[eje])
        if hi <= lo:  # el tumor cae fuera de esta rejilla: no deberia pasar
            lo, hi = 0, min(1, forma_destino[eje])
        recorte.append(slice(lo, hi))
    return tuple(recorte)


def _bbox(mascara: np.ndarray):
    ejes = []
    for eje in range(3):
        proy = mascara.any(axis=tuple(e for e in range(3) if e != eje))
        pos = np.flatnonzero(proy)
        ejes.append((int(pos[0]), int(pos[-1])))
    return tuple(ejes)


def _superficie_mm2(mascara: np.ndarray, zooms) -> float:
    """Area de la superficie por marching cubes (malla triangular).

    Contar caras expuestas sobrestima cualquier superficie curva por el efecto
    escalera (~1.5x en una esfera), hasta el punto de que una esfera saldria
    menos "esferica" que un cubo. Marching cubes interpola la superficie y es
    lo que usa la radiomica estandar (IBSI). Si falla -por una mascara
    diminuta- se cae al recuento de caras.
    """
    try:
        from skimage import measure
        # Relleno de un voxel para que la malla cierre en los bordes.
        m = np.pad(mascara.astype(np.uint8), 1)
        verts, faces, _, _ = measure.marching_cubes(m, level=0.5,
                                                    spacing=tuple(float(z) for z in zooms))
        return float(measure.mesh_surface_area(verts, faces))
    except Exception:
        return _superficie_caras_mm2(mascara, zooms)


def _superficie_caras_mm2(mascara: np.ndarray, zooms) -> float:
    """Area contando caras de voxel expuestas (anisotropia incluida)."""
    area = 0.0
    for eje in range(3):
        otros = [zooms[e] for e in range(3) if e != eje]
        area_cara = otros[0] * otros[1]
        m = np.moveaxis(mascara, eje, 0)
        caras = int((m[:-1] & ~m[1:]).sum() + (m[1:] & ~m[:-1]).sum())
        borde = int(m[0].sum() + m[-1].sum())  # caras contra el borde del volumen
        area += (caras + borde) * area_cara
    return area


def _geometria(mascara: np.ndarray, zooms) -> dict:
    """Volumen, componentes, esfericidad y extension de una mascara binaria."""
    vox_mm3 = float(np.prod(zooms))
    n = int(mascara.sum())
    if n == 0:
        return {"volumen_mm3": 0.0, "n_componentes": 0, "esfericidad": np.nan,
                "superficie_mm3": np.nan, "sup_vol_mm": np.nan,
                "n_cortes": 0, "extension_z_mm": 0.0,
                "vol_componente_mayor_frac": np.nan}
    etiquetas, n_comp = ndimage.label(mascara, structure=np.ones((3, 3, 3), bool))
    tams = np.bincount(etiquetas.ravel())[1:]
    volumen = n * vox_mm3
    area = _superficie_mm2(mascara, zooms)
    esfericidad = (36 * np.pi * volumen ** 2) ** (1 / 3) / area if area > 0 else np.nan
    cortes = np.flatnonzero(mascara.any(axis=(0, 1)))
    return {
        "volumen_mm3": volumen,
        "n_componentes": int(n_comp),
        "esfericidad": float(esfericidad),
        "superficie_mm3": float(area),
        "sup_vol_mm": float(area / volumen),
        "n_cortes": int(cortes.size),
        "extension_z_mm": float((cortes[-1] - cortes[0] + 1) * zooms[2]),
        "vol_componente_mayor_frac": float(tams.max() / n),
    }


def _auc_separabilidad(a: np.ndarray, b: np.ndarray, maximo=20000, semilla=0) -> float:
    """AUC = P(valor de tumor > valor de fondo). 0.5 = indistinguibles.

    Equivale al solapamiento de los dos histogramas, pero como un solo numero y
    sin suponer normalidad. Se calcula via el estadistico U de Mann-Whitney.
    """
    if a.size < 10 or b.size < 10:
        return float("nan")
    rng = np.random.default_rng(semilla)
    if a.size > maximo:
        a = rng.choice(a, maximo, replace=False)
    if b.size > maximo:
        b = rng.choice(b, maximo, replace=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (a.size * b.size))


def _ruido_inmerkaer(corte: np.ndarray, valido: np.ndarray) -> float:
    """Desviacion del ruido por filtro paso-alto (Immerkaer, 1996).

    La mascara laplaciana [[0,1,0],[1,-4,1],[0,1,0]] amplifica la varianza del
    ruido x20, asi que sigma ~= std(laplaciano) / sqrt(20).
    """
    if valido.sum() < 500:
        return float("nan")
    lap = ndimage.laplace(corte.astype(np.float32))
    interior = ndimage.binary_erosion(valido, iterations=2)
    if interior.sum() < 500:
        return float("nan")
    return float(np.std(lap[interior]) / np.sqrt(20.0))


def _estadisticos_globales_ct(ct: np.ndarray, zooms, z_tumor) -> tuple[dict, np.ndarray]:
    """Histograma, composicion de tejidos, artefacto metalico, ruido y nitidez."""
    # Submuestreo en plano: millones de voxeles siguen siendo suficientes y
    # recorta memoria y tiempo a la cuarta parte.
    sub = ct[::2, ::2, :]
    cuerpo = sub > UMBRAL_CUERPO_HU
    n_cuerpo = int(cuerpo.sum())
    vals = sub[cuerpo].astype(np.float32)
    hist, _ = np.histogram(np.clip(vals, CT_BINS[0], CT_BINS[-1]), bins=CT_BINS)
    out = {
        "ct_hu_media": float(vals.mean()) if n_cuerpo else np.nan,
        "ct_hu_std": float(vals.std()) if n_cuerpo else np.nan,
        "ct_hu_p05": float(np.percentile(vals, 5)) if n_cuerpo else np.nan,
        "ct_hu_p95": float(np.percentile(vals, 95)) if n_cuerpo else np.nan,
        "ct_hu_asimetria": float(stats.skew(vals)) if n_cuerpo else np.nan,
        "ct_hu_curtosis": float(stats.kurtosis(vals)) if n_cuerpo else np.nan,
        "ct_entropia_bits": entropia_shannon(hist),
        "ct_frac_cuerpo": float(n_cuerpo / sub.size),
        "ct_frac_grasa": float(((vals >= -190) & (vals < -30)).mean()) if n_cuerpo else np.nan,
        "ct_frac_blando": float(((vals >= -30) & (vals < 300)).mean()) if n_cuerpo else np.nan,
        "ct_frac_hueso": float((vals >= 300).mean()) if n_cuerpo else np.nan,
        "ct_frac_metal": float((vals >= UMBRAL_METAL_HU).mean()) if n_cuerpo else np.nan,
    }
    del vals

    # Ruido y nitidez sobre 5 cortes repartidos en la extension del tumor.
    z0, z1 = z_tumor
    zs = np.unique(np.linspace(z0, z1, 5).round().astype(int))
    zs = zs[(zs >= 0) & (zs < ct.shape[2])]
    ruidos, nitideces = [], []
    for z in zs:
        corte = ct[:, :, z].astype(np.float32)
        valido = corte > UMBRAL_CUERPO_HU
        r = _ruido_inmerkaer(corte, valido)
        if np.isfinite(r):
            ruidos.append(r)
        if valido.sum() >= 500:
            nitideces.append(float(np.var(ndimage.laplace(corte)[valido])))
    out["ct_ruido_hu"] = float(np.median(ruidos)) if ruidos else np.nan
    out["ct_nitidez_laplaciano"] = float(np.median(nitideces)) if nitideces else np.nan
    return out, hist


def _estadisticos_globales_pet(pet: np.ndarray) -> tuple[dict, np.ndarray]:
    cuerpo = pet > SUV_FONDO_MIN
    vals = pet[cuerpo].astype(np.float32)
    hist, _ = np.histogram(np.clip(vals, PET_BINS[0], PET_BINS[-1]), bins=PET_BINS)
    if vals.size == 0:
        return {k: np.nan for k in ("pet_suv_max", "pet_suv_media", "pet_suv_p95",
                                    "pet_suv_asimetria", "pet_entropia_bits")}, hist
    return {
        "pet_suv_max": float(pet.max()),
        "pet_suv_media": float(vals.mean()),
        "pet_suv_p95": float(np.percentile(vals, 95)),
        "pet_suv_asimetria": float(stats.skew(vals)),
        "pet_entropia_bits": entropia_shannon(hist),
    }, hist


def _contraste(volumen: np.ndarray, tumor: np.ndarray, zooms, anillo_mm, prefijo) -> dict:
    """Contraste tumor/anillo peritumoral, separabilidad y gradiente de borde."""
    nan = {f"{prefijo}_{k}": np.nan for k in
           ("tumor_media", "tumor_std", "tumor_max", "fondo_media", "fondo_std",
            "cnr", "fisher", "auc_separabilidad", "razon_tumor_fondo",
            "gradiente_borde")}
    if not tumor.any():
        return nan
    # Anillo exacto en mm usando la transformada de distancia anisotropica.
    dist = ndimage.distance_transform_edt(~tumor, sampling=zooms)
    anillo = (dist > 0) & (dist <= anillo_mm)
    vt = volumen[tumor].astype(np.float32)
    vf = volumen[anillo].astype(np.float32)
    if vf.size < 10:
        return nan
    mt, st = float(vt.mean()), float(vt.std())
    mf, sf = float(vf.mean()), float(vf.std())
    borde = tumor & ~ndimage.binary_erosion(tumor)
    grad = np.sqrt(sum(g ** 2 for g in np.gradient(volumen.astype(np.float32), *zooms)))
    return {
        f"{prefijo}_tumor_media": mt,
        f"{prefijo}_tumor_std": st,
        f"{prefijo}_tumor_max": float(vt.max()),
        f"{prefijo}_fondo_media": mf,
        f"{prefijo}_fondo_std": sf,
        f"{prefijo}_cnr": (mt - mf) / sf if sf > 1e-6 else np.nan,
        f"{prefijo}_fisher": (mt - mf) ** 2 / (st ** 2 + sf ** 2) if (st or sf) else np.nan,
        f"{prefijo}_auc_separabilidad": _auc_separabilidad(vt, vf),
        f"{prefijo}_razon_tumor_fondo": mt / mf if abs(mf) > 1e-6 else np.nan,
        f"{prefijo}_gradiente_borde": float(grad[borde].mean()) if borde.any() else np.nan,
    }


def caracterizar_caso(caso: str, directorio, margen_mm=30.0, anillo_mm=10.0) -> dict:
    """Vector de descriptores de un caso. Devuelve tambien los histogramas."""
    base = f"{directorio}/{caso}"
    img_ct = nib.load(f"{base}__CT.nii.gz")
    img_pt = nib.load(f"{base}__PT.nii.gz")
    img_sg = nib.load(f"{base}.nii.gz")

    seg = np.asarray(img_sg.dataobj)
    zooms_sg = np.array(img_sg.header.get_zooms()[:3], dtype=np.float64)
    zooms_ct = np.array(img_ct.header.get_zooms()[:3], dtype=np.float64)
    zooms_pt = np.array(img_pt.header.get_zooms()[:3], dtype=np.float64)

    fila: dict = {
        "caso": caso,
        "centro": centro_de(caso),
        "ct_nx": img_ct.shape[0], "ct_ny": img_ct.shape[1], "ct_nz": img_ct.shape[2],
        "ct_sx": zooms_ct[0], "ct_sy": zooms_ct[1], "ct_sz": zooms_ct[2],
        "pet_nx": img_pt.shape[0], "pet_ny": img_pt.shape[1], "pet_nz": img_pt.shape[2],
        "pet_sx": zooms_pt[0], "pet_sy": zooms_pt[1], "pet_sz": zooms_pt[2],
        "fov_z_mm": float(img_ct.shape[2] * zooms_ct[2]),
        "vol_total_mm3": float(np.prod(img_ct.shape) * np.prod(zooms_ct)),
    }

    # --- Geometria de las etiquetas (en su propia rejilla) --------------------
    m_gtvp = seg == GTVP
    m_gtvn = seg == GTVN
    for nombre, m in (("gtvp", m_gtvp), ("gtvn", m_gtvn)):
        for k, v in _geometria(m, zooms_sg).items():
            fila[f"{nombre}_{k}"] = v
    fila["gtvp_frac_positiva"] = (fila["gtvp_volumen_mm3"] / fila["vol_total_mm3"]
                                  if fila["vol_total_mm3"] else np.nan)
    fila["tiene_gtvp"] = bool(m_gtvp.any())
    fila["tiene_gtvn"] = bool(m_gtvn.any())

    m_union = m_gtvp | m_gtvn
    if not m_union.any():
        fila["error"] = "mascara vacia"
        return {"fila": fila, "hist_ct": np.zeros(64), "hist_pet": np.zeros(64)}

    bbox = _bbox(m_union)

    # --- Estadisticos globales (volumen completo) ----------------------------
    pet = np.asarray(img_pt.dataobj, dtype=np.float32)
    g_pet, hist_pet = _estadisticos_globales_pet(pet)
    fila.update(g_pet)

    ct = np.asarray(img_ct.dataobj, dtype=np.float32)
    # Extension Z del tumor pasada a la rejilla del CT, para elegir los cortes.
    m_ct2sg = np.linalg.inv(img_ct.affine) @ img_sg.affine
    z_tumor = sorted(int(round((m_ct2sg @ [bbox[0][0], bbox[1][0], k, 1.0])[2]))
                     for k in (bbox[2][0], bbox[2][1]))
    g_ct, hist_ct = _estadisticos_globales_ct(ct, zooms_ct, z_tumor)
    fila.update(g_ct)

    # --- Contraste y separabilidad, en cada rejilla nativa -------------------
    objetivo = m_gtvp if m_gtvp.any() else m_union
    rec_pt = _recorte_destino(bbox, img_sg.affine, img_pt.affine, img_pt.shape, margen_mm)
    seg_en_pt = _muestrea_mascara(objetivo.astype(np.uint8), img_sg.affine,
                                  img_pt.shape, img_pt.affine, rec_pt)
    fila.update(_contraste(pet[rec_pt], seg_en_pt.astype(bool), zooms_pt, anillo_mm, "pet"))
    del pet

    rec_ct = _recorte_destino(bbox, img_sg.affine, img_ct.affine, img_ct.shape, margen_mm)
    seg_en_ct = _muestrea_mascara(objetivo.astype(np.uint8), img_sg.affine,
                                  img_ct.shape, img_ct.affine, rec_ct)
    fila.update(_contraste(ct[rec_ct], seg_en_ct.astype(bool), zooms_ct, anillo_mm, "ct"))
    del ct

    fila["error"] = ""
    return {"fila": fila, "hist_ct": hist_ct, "hist_pet": hist_pet}
