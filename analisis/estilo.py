"""Estilo comun de las figuras.

Paleta y tinta tomadas de la paleta de referencia validada para daltonismo
(slots 1-3, los unicos que superan el control de todas las parejas: blue,
orange, aqua). Donde hay mas de tres grupos (7 centros) la identidad la lleva
la etiqueta del eje o un panel propio, nunca el color: asi no hace falta
inventar hues nuevos ni reciclarlos.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIE_1 = "#2a78d6"   # azul
SERIE_2 = "#eb6834"   # naranja
SERIE_3 = "#1baf7a"   # aqua
SERIES = (SERIE_1, SERIE_2, SERIE_3)

DIV_NEG = "#2a78d6"   # pareja divergente azul <-> rojo, gris en el centro
DIV_MED = "#f0efec"
DIV_POS = "#d03b3b"

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_MUTE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"


def aplicar_estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": TINTA,
        "axes.labelcolor": TINTA_2,
        "axes.edgecolor": EJE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.6,
        "xtick.color": TINTA_MUTE,
        "ytick.color": TINTA_MUTE,
        "xtick.labelcolor": TINTA_2,
        "ytick.labelcolor": TINTA_2,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def limpiar(ax, rejilla="y"):
    """Quita los marcos superior y derecho y deja la rejilla en un solo eje."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(False)
    if rejilla in ("x", "both"):
        ax.xaxis.grid(True)
    if rejilla in ("y", "both"):
        ax.yaxis.grid(True)
    return ax


def cmap_divergente():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MED, DIV_POS])
