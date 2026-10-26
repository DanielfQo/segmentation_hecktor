#!/usr/bin/env python3
"""Visor interactivo: como cambia una imagen al pasar por el preprocesado.

Abre una ventana con el caso a la izquierda (axial) y a la derecha (coronal),
y deja moverse entre las etapas del pipeline real y entre cortes. Debajo, las
estadisticas de la etapa activa y el cambio respecto a la anterior.

Controles:
  etapas            botones de la izquierda, o las flechas izquierda/derecha
  corte             deslizador de abajo, o las flechas arriba/abajo
  modalidad         botones CT / PET, o la tecla m
  nuevo aumento     boton "re-sortear", o la tecla a (vuelve a tirar los dados
                    del aumento de datos sin recalcular el resto del pipeline)
  guardar           boton "guardar PNG", o la tecla g
  siguiente caso    tecla n

Uso:
    analisis/.venv/bin/python analisis/visor_pipeline.py
    analisis/.venv/bin/python analisis/visor_pipeline.py --casos CHUM-001 MDA-001
    analisis/.venv/bin/python analisis/visor_pipeline.py --config .../full.yaml

Sin ventana (para un servidor o para dejarlo en el informe):
    analisis/.venv/bin/python analisis/visor_pipeline.py --tabla
    analisis/.venv/bin/python analisis/visor_pipeline.py --exportar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
RAIZ = BASE.parent
sys.path.insert(0, str(BASE))

import matplotlib  # noqa: E402


def elige_backend(preferido: str | None = None) -> str:
    """Primer backend interactivo que funcione, o un error que diga que hacer.

    Un venv recien creado no trae ninguno: matplotlib se queda en 'agg' y
    entonces plt.show() no abre nada y el programa termina en silencio, que es
    lo peor que puede pasar. Mejor fallar diciendo por que.
    """
    candidatos = [preferido] if preferido else ["qtagg", "tkagg", "gtk4agg", "webagg"]
    errores = []
    for b in candidatos:
        try:
            matplotlib.use(b, force=True)
            return b
        except Exception as e:  # backend no instalado o sin servidor grafico
            errores.append(f"  {b}: {type(e).__name__}: {str(e).splitlines()[0][:90]}")
    raise SystemExit(
        "No hay ningun backend grafico disponible, asi que la ventana no puede "
        "abrirse.\n" + "\n".join(errores) + "\n\n"
        "Instala uno en el venv (no hace falta root):\n"
        "    analisis/.venv/bin/pip install PyQt6\n"
        "o usa el backend de navegador:\n"
        "    analisis/.venv/bin/pip install tornado\n"
        "    analisis/.venv/bin/python analisis/visor_pipeline.py --backend webagg\n\n"
        "Y si no quieres ventana:  --tabla  o  --exportar")


import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.widgets import Button, RadioButtons, Slider  # noqa: E402

from estilo import (REJILLA, SERIE_2, SERIE_3, SUPERFICIE, TINTA,  # noqa: E402
                    TINTA_2, TINTA_MUTE, aplicar_estilo)
from pipeline_etapas import (carga_config, construye_etapas, elige_casos,  # noqa: E402
                             tabla_etapas)
from mambalite3d.data import io as mio  # noqa: E402

AZULES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Rampa secuencial de un solo tono. Arranca en blanco para que el fondo (SUV 0,
# que es la mayor parte del volumen) no tina la imagen entera de azul.
CMAP_PET = LinearSegmentedColormap.from_list("pet", ["#ffffff"] + AZULES)
COLOR_GTVP, COLOR_GTVN = SERIE_2, SERIE_3


class Visor:
    def __init__(self, casos, raiz, cfg, salidas, figuras, semilla=7):
        self.casos = list(casos)
        self.raiz = raiz
        self.cfg = cfg
        self.salidas = salidas
        self.figuras = figuras
        self.semilla = semilla
        self.i_caso = 0
        self.i_etapa = 0
        self.modalidad = "CT"
        self.frac = 0.5
        self.n_sorteo = 0
        self.todos = {f.case_id: f for f in mio.find_cases(raiz)}
        self._carga_caso()
        self._construye_ventana()

    # ------------------------------------------------------------- datos
    def _carga_caso(self):
        caso = self.casos[self.i_caso]
        if caso not in self.todos:
            raise SystemExit(f"{caso}: no esta en {self.raiz}")
        print(f"procesando {caso} ...", flush=True)
        semillas = (1 + 10 * self.n_sorteo, 2 + 10 * self.n_sorteo)
        self.etapas = construye_etapas(self.todos[caso], self.cfg,
                                       semilla_patch=self.semilla,
                                       semillas_aug=semillas)
        self.caso = caso
        self.centro = self.todos[caso].center
        self.i_etapa = min(self.i_etapa, len(self.etapas) - 1)

    @property
    def etapa(self):
        return self.etapas[self.i_etapa]

    def _volumen(self):
        """Volumen, rango de visualizacion, mapa de color, unidad y etiqueta."""
        e = self.etapa
        self._cache_fila = getattr(self, "_cache_fila", {})
        if self.modalidad == "CT":
            return e.ct, e.rango_ct, "gray", e.unidad_ct, e.etiqueta
        etiqueta = e.etiqueta_pet if e.etiqueta_pet is not None else e.etiqueta
        return e.pet, e.rango_pet, CMAP_PET, e.unidad_pet, etiqueta

    # ----------------------------------------------------------- ventana
    def _construye_ventana(self):
        aplicar_estilo()
        self.fig = plt.figure(figsize=(13.5, 7.6))
        self.fig.canvas.manager.set_window_title("Pipeline de preprocesado")
        self.ax_ax = self.fig.add_axes([0.17, 0.40, 0.33, 0.50])
        self.ax_cor = self.fig.add_axes([0.53, 0.40, 0.33, 0.50])
        for ax in (self.ax_ax, self.ax_cor):
            ax.set_xticks([]), ax.set_yticks([])

        self.ax_radio = self.fig.add_axes([0.012, 0.40, 0.145, 0.50])
        self.ax_radio.set_facecolor(SUPERFICIE)
        self.radio = RadioButtons(
            self.ax_radio, [f"{e.indice}. {e.nombre}" for e in self.etapas],
            active=self.i_etapa, activecolor=AZULES[4])
        for t in self.radio.labels:
            t.set_fontsize(8.5)
            t.set_color(TINTA_2)
        self.radio.on_clicked(self._cambia_etapa)

        self.ax_mod = self.fig.add_axes([0.88, 0.78, 0.10, 0.12])
        self.ax_mod.set_facecolor(SUPERFICIE)
        self.radio_mod = RadioButtons(self.ax_mod, ["CT", "PET"], active=0,
                                      activecolor=AZULES[4])
        for t in self.radio_mod.labels:
            t.set_fontsize(9)
            t.set_color(TINTA_2)
        self.radio_mod.on_clicked(self._cambia_modalidad)

        self.ax_slider = self.fig.add_axes([0.17, 0.335, 0.69, 0.022])
        self.slider = Slider(self.ax_slider, "corte", 0.0, 1.0, valinit=self.frac,
                             color=AZULES[3], track_color=REJILLA)
        self.slider.valtext.set_fontsize(8)
        self.slider.label.set_color(TINTA_2)
        self.slider.on_changed(self._cambia_corte)

        self.ax_b1 = self.fig.add_axes([0.88, 0.66, 0.10, 0.05])
        self.b_sorteo = Button(self.ax_b1, "re-sortear\naumento",
                               color=SUPERFICIE, hovercolor=REJILLA)
        self.b_sorteo.label.set_fontsize(8)
        self.b_sorteo.on_clicked(lambda _: self._resortea())

        self.ax_b2 = self.fig.add_axes([0.88, 0.59, 0.10, 0.05])
        self.b_guardar = Button(self.ax_b2, "guardar PNG", color=SUPERFICIE,
                                hovercolor=REJILLA)
        self.b_guardar.label.set_fontsize(8)
        self.b_guardar.on_clicked(lambda _: self._guarda())

        self.ax_b3 = self.fig.add_axes([0.88, 0.52, 0.10, 0.05])
        self.b_caso = Button(self.ax_b3, "siguiente caso", color=SUPERFICIE,
                             hovercolor=REJILLA)
        self.b_caso.label.set_fontsize(8)
        self.b_caso.on_clicked(lambda _: self._siguiente_caso())

        self.ax_txt = self.fig.add_axes([0.012, 0.015, 0.975, 0.30])
        self.ax_txt.axis("off")
        self.fig.canvas.mpl_connect("key_press_event", self._tecla)
        self._dibuja()

    # ----------------------------------------------------------- callbacks
    def _cambia_etapa(self, etiqueta):
        self.i_etapa = int(str(etiqueta).split(".", 1)[0])
        self._dibuja()

    def _cambia_modalidad(self, etiqueta):
        self.modalidad = etiqueta
        self._dibuja()

    def _cambia_corte(self, valor):
        self.frac = float(valor)
        self._dibuja()

    def _resortea(self):
        self.n_sorteo += 1
        self._carga_caso()
        self._dibuja()
        print(f"  aumento re-sorteado (tirada {self.n_sorteo})", flush=True)

    def _siguiente_caso(self):
        self.i_caso = (self.i_caso + 1) % len(self.casos)
        self._carga_caso()
        self._refresca_radio()
        self._dibuja()

    def _refresca_radio(self):
        self.ax_radio.clear()
        self.ax_radio.set_facecolor(SUPERFICIE)
        self.radio = RadioButtons(
            self.ax_radio, [f"{e.indice}. {e.nombre}" for e in self.etapas],
            active=self.i_etapa, activecolor=AZULES[4])
        for t in self.radio.labels:
            t.set_fontsize(8.5)
            t.set_color(TINTA_2)
        self.radio.on_clicked(self._cambia_etapa)

    def _guarda(self):
        self.figuras.mkdir(parents=True, exist_ok=True)
        ruta = (self.figuras /
                f"pipeline_{self.caso}_{self.i_etapa}_{self.modalidad}.png")
        self.fig.savefig(ruta, dpi=200)
        print(f"  guardado {ruta}", flush=True)

    def _tecla(self, evento):
        if evento.key in ("right", "down"):
            self.i_etapa = (self.i_etapa + 1) % len(self.etapas)
            self.radio.set_active(self.i_etapa)
        elif evento.key in ("left", "up"):
            self.i_etapa = (self.i_etapa - 1) % len(self.etapas)
            self.radio.set_active(self.i_etapa)
        elif evento.key == "m":
            self.modalidad = "PET" if self.modalidad == "CT" else "CT"
            self.radio_mod.set_active(0 if self.modalidad == "CT" else 1)
        elif evento.key == "a":
            self._resortea()
        elif evento.key == "g":
            self._guarda()
        elif evento.key == "n":
            self._siguiente_caso()

    # -------------------------------------------------------------- pintar
    def _contorno(self, ax, m2d, vista):
        if m2d is None or not m2d.any():
            return
        for valor, color in ((1, COLOR_GTVP), (2, COLOR_GTVN)):
            if (m2d == valor).any():
                ax.contour((m2d == valor).astype(float), levels=[0.5],
                           colors=[color], linewidths=1.2)

    def _recuadro(self, ax, caja, vista, forma_ref, forma_vista):
        if not caja:
            return
        # La caja viene en voxeles de la rejilla del CT crudo (forma_ref). Si lo
        # que se esta pintando es otra rejilla -el PET- hay que reescalarla por
        # el mismo eje que se dibuja, no por otro.
        if vista == "axial":            # imagen = vol[z], ejes (y, x)
            x, y = caja["x"], caja["y"]
            ex = forma_vista[2] / forma_ref[2]
            ey = forma_vista[1] / forma_ref[1]
        else:                           # imagen = vol[:, y, :], ejes (z, x)
            x, y = caja["x"], caja["z"]
            ex = forma_vista[2] / forma_ref[2]
            ey = forma_vista[0] / forma_ref[0]
        ax.add_patch(mpatches.Rectangle(
            (x[0] * ex, y[0] * ey), (x[1] - x[0]) * ex, (y[1] - y[0]) * ey,
            fill=False, edgecolor=SERIE_2, linewidth=1.8, linestyle="-"))

    def _fila_con_mas_tumor(self, etiqueta):
        clave = (self.i_etapa, self.modalidad, self.n_sorteo, self.i_caso)
        if clave not in self._cache_fila:
            if etiqueta is None or not (etiqueta > 0).any():
                self._cache_fila[clave] = etiqueta.shape[1] // 2
            else:
                perfil = (etiqueta > 0).sum(axis=(0, 2))
                self._cache_fila[clave] = int(perfil.argmax())
        return self._cache_fila[clave]

    def _dibuja(self):
        e = self.etapa
        vol, rango, cmap, unidad, etiqueta = self._volumen()
        nz, ny, nx = vol.shape
        z = int(round(self.frac * (nz - 1)))
        mismo_grid = etiqueta is not None and etiqueta.shape == vol.shape
        # El corte coronal se queda donde mas tumor hay, en la rejilla que se
        # esta pintando (el centroide de otra rejilla caeria fuera).
        y = self._fila_con_mas_tumor(etiqueta) if mismo_grid else ny // 2

        for ax, vista in ((self.ax_ax, "axial"), (self.ax_cor, "coronal")):
            ax.clear()
            if vista == "axial":
                img = vol[z]
                aspecto = e.espaciado_zyx[1] / e.espaciado_zyx[2]
                m2d = etiqueta[z] if mismo_grid else None
                titulo = f"axial  z = {z} / {nz - 1}"
            else:
                img = vol[:, y, :]
                aspecto = e.espaciado_zyx[0] / e.espaciado_zyx[2]
                m2d = etiqueta[:, y, :] if mismo_grid else None
                titulo = f"coronal  y = {y} / {ny - 1}"
            ax.imshow(img, cmap=cmap, vmin=rango[0], vmax=rango[1],
                      interpolation="nearest", aspect=aspecto)
            self._contorno(ax, m2d, vista)
            self._recuadro(ax, e.caja.get(vista), vista, self.etapas[0].ct.shape,
                           vol.shape)
            ax.set_title(titulo, fontsize=9, color=TINTA_2)
            ax.set_xticks([]), ax.set_yticks([])
            for lado in ax.spines.values():
                lado.set_color(TINTA_MUTE)
                lado.set_linewidth(0.6)

        self.fig.suptitle(
            f"{self.caso}  ({self.centro})   -   etapa {e.indice}: {e.nombre}"
            f"   -   {self.modalidad} en {unidad}",
            fontsize=12, color=TINTA, fontweight="bold", y=0.975)
        self._texto()
        self.fig.canvas.draw_idle()

    def _texto(self):
        e = self.etapa
        d = e.estadisticas()
        lineas = [e.descripcion, ""]
        forma = "x".join(str(s) for s in e.ct.shape)
        esp = " x ".join(f"{s:.2f}" for s in e.espaciado_zyx)
        lineas.append(
            f"forma (z,y,x) = {forma}    espaciado = {esp} mm    "
            f"voxeles = {e.ct.size:,}    volumen = {d['volumen_cm3']:,.0f} cm3")
        lineas.append(
            f"CT:  min {d['ct_min']:>9.1f}   max {d['ct_max']:>9.1f}   "
            f"media {d['ct_media']:>8.2f}   sd {d['ct_sd']:>7.2f}        "
            f"PET:  min {d['pet_min']:>7.2f}   max {d['pet_max']:>8.2f}   "
            f"media {d['pet_media']:>7.3f}   sd {d['pet_sd']:>6.3f}")
        if np.isfinite(d["frac_tumor_pct"]):
            lineas.append(
                f"tumor: {d['frac_tumor_pct']:.4f} % de los voxeles   "
                f"(GTVp {d['voxeles_gtvp']:,}, GTVn {d['voxeles_gtvn']:,})")

        if self.i_etapa > 0:
            p = self.etapas[self.i_etapa - 1].estadisticas()
            cambios = [f"voxeles x{d['voxeles'] / p['voxeles']:.3f}"]
            if np.isfinite(d["frac_tumor_pct"]) and np.isfinite(p["frac_tumor_pct"]) \
                    and p["frac_tumor_pct"] > 0:
                cambios.append("densidad de tumor x"
                               f"{d['frac_tumor_pct'] / p['frac_tumor_pct']:.2f}")
            cambios.append(f"media del CT {p['ct_media']:.2f} -> {d['ct_media']:.2f}")
            cambios.append(f"sd del PET {p['pet_sd']:.3f} -> {d['pet_sd']:.3f}")
            lineas += ["", "frente a la etapa anterior:  " + "    |    ".join(cambios)]

        if self.i_etapa <= 1 and e.etiqueta is not None:
            lineas.append(
                "(en las etapas crudas la etiqueta esta en su propia rejilla "
                f"{'x'.join(str(v) for v in e.etiqueta.shape)}, que no siempre "
                "coincide con la del CT ni con la del PET: aqui se remuestrea "
                "solo para poder verla. El pipeline no las pone en la misma "
                "rejilla hasta la etapa 2.)")
        lineas += ["", "flechas = etapa   |   deslizador = corte   |   m = CT/PET"
                      "   |   a = re-sortear aumento   |   g = guardar   |   "
                      "n = siguiente caso"]

        self.ax_txt.clear()
        self.ax_txt.axis("off")
        self.ax_txt.text(0.0, 1.0, "\n".join(lineas), va="top", ha="left",
                         fontsize=8.6, color=TINTA_2, family="monospace",
                         linespacing=1.55, wrap=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=RAIZ / "HECKTOR2025_raw")
    ap.add_argument("--config", type=Path,
                    default=RAIZ / "modelos" / "mamba" / "configs" / "exp_wide.yaml")
    ap.add_argument("--salidas", type=Path, default=BASE / "salidas")
    ap.add_argument("--figuras", type=Path, default=BASE / "figuras")
    ap.add_argument("--casos", nargs="*", default=None)
    ap.add_argument("-n", type=int, default=3,
                    help="cuantos casos cargar si no se dan --casos")
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--tabla", action="store_true",
                    help="sin ventana: imprime y guarda la tabla de etapas")
    ap.add_argument("--exportar", action="store_true",
                    help="sin ventana: guarda un PNG por etapa y modalidad")
    ap.add_argument("--backend", default=None,
                    help="backend de matplotlib (qtagg, tkagg, webagg); por "
                         "defecto el primero que funcione")
    args = ap.parse_args()

    if not (args.tabla or args.exportar):
        backend = elige_backend(args.backend)
        print(f"backend grafico: {backend}"
              + ("  (se abre en el navegador)" if backend == "webagg" else ""))

    cfg = carga_config(args.config)
    print(f"config {args.config.name}: spacing {cfg['data']['target_spacing_xyz']} mm, "
          f"ventana CT {cfg['data']['ct_window']} HU, "
          f"patch {cfg['model']['patch_size_zyx']}")
    casos = args.casos or elige_casos(args.raiz, args.salidas, args.n)
    print("casos:", ", ".join(casos))

    if args.tabla or args.exportar:
        matplotlib.use("Agg")
        todos = {f.case_id: f for f in mio.find_cases(args.raiz)}
        import pandas as pd
        tablas = []
        for caso in casos:
            if caso not in todos:
                print(f"  {caso}: no encontrado")
                continue
            print(f"  {caso} ...", flush=True)
            etapas = construye_etapas(todos[caso], cfg, semilla_patch=args.semilla)
            tablas.append(tabla_etapas(etapas, caso, todos[caso].center))
            if args.exportar:
                v = Visor.__new__(Visor)
                v.etapas, v.caso, v.centro = etapas, caso, todos[caso].center
                v.casos, v.i_caso, v.frac = [caso], 0, 0.5
                v.figuras, v.n_sorteo, v.semilla = args.figuras, 0, args.semilla
                v.i_etapa, v.modalidad = 0, "CT"
                v.raiz, v.cfg, v.salidas = args.raiz, cfg, args.salidas
                v._construye_ventana()
                args.figuras.mkdir(parents=True, exist_ok=True)
                for i in range(len(etapas)):
                    for modalidad in ("CT", "PET"):
                        v.i_etapa, v.modalidad = i, modalidad
                        v._dibuja()
                        v._guarda()
                plt.close(v.fig)
        if tablas:
            t = pd.concat(tablas, ignore_index=True)
            args.salidas.mkdir(parents=True, exist_ok=True)
            ruta = args.salidas / "pipeline_etapas.csv"
            t.to_csv(ruta, index=False)
            cols = ["caso", "etapa", "forma_zyx", "voxeles", "voxeles_x",
                    "volumen_cm3", "ct_media", "pet_sd", "frac_tumor_pct"]
            print("\n" + t[cols].to_string(index=False))
            print(f"\nTabla completa: {ruta}")
        return 0

    Visor(casos, args.raiz, cfg, args.salidas, args.figuras, args.semilla)
    import time
    t0 = time.time()
    plt.show()
    if time.time() - t0 < 1.0:
        print("\nAVISO: la ventana se cerro al instante, asi que seguramente no "
              "llego a abrirse.\n"
              "Prueba con el visor en navegador:  --backend webagg\n"
              "o sin ventana:  --tabla  /  --exportar", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
