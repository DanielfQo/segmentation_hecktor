from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class NiftiViewer(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("Visualizador NIfTI")
		self.geometry("1100x800")

		self.volume_img = None
		self.volume_arr = None
		self.current_slice = 0

		self.protocol("WM_DELETE_WINDOW", self.on_close)

		self._build_ui()

	def _build_ui(self):
		top = ttk.Frame(self)
		top.pack(fill="x", padx=8, pady=8)

		ttk.Label(top, text="Ruta .nii.gz:").pack(side="left")
		self.path_var = tk.StringVar()
		self.path_entry = ttk.Entry(top, textvariable=self.path_var, width=80)
		self.path_entry.pack(side="left", padx=6, fill="x", expand=True)

		ttk.Button(top, text="Buscar", command=self.browse_file).pack(side="left", padx=4)
		ttk.Button(top, text="Cargar", command=self.load_volume).pack(side="left", padx=4)

		controls = ttk.Frame(self)
		controls.pack(fill="x", padx=8, pady=(0, 8))

		ttk.Label(controls, text="Corte:").pack(side="left")
		self.slice_var = tk.IntVar(value=0)
		self.slice_label = ttk.Label(controls, text="0")
		self.slice_label.pack(side="right")

		self.slice_slider = ttk.Scale(
			controls,
			from_=0,
			to=0,
			orient="horizontal",
			command=self.on_slice_change,
		)
		self.slice_slider.pack(side="left", fill="x", expand=True, padx=8)

		self.fig, self.ax = plt.subplots(figsize=(8, 8))
		self.fig.tight_layout()
		self.canvas = FigureCanvasTkAgg(self.fig, master=self)
		self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

		info_frame = ttk.Frame(self)
		info_frame.pack(fill="x", padx=8, pady=(0, 8))
		ttk.Label(info_frame, text="Información:").pack(anchor="w")
		self.info_text = tk.Text(info_frame, height=5)
		self.info_text.pack(fill="x", expand=False)

	def browse_file(self):
		file_path = filedialog.askopenfilename(
			title="Selecciona un archivo NIfTI",
			filetypes=[("NIfTI files", "*.nii.gz *.nii"), ("All files", "*.*")],
		)
		if file_path:
			self.path_var.set(file_path)
			self.load_volume()

	def load_volume(self):
		file_path = self.path_var.get().strip()
		if not file_path:
			messagebox.showerror("Error", "Introduce una ruta válida.")
			return

		nifti_path = Path(file_path)
		if not nifti_path.exists():
			messagebox.showerror("Error", f"No existe el archivo:\n{nifti_path}")
			return

		try:
			self.volume_img = sitk.ReadImage(str(nifti_path))
			self.volume_arr = sitk.GetArrayFromImage(self.volume_img)

			if self.volume_arr.ndim == 2:
				self.volume_arr = self.volume_arr[np.newaxis, ...]

			if self.volume_arr.ndim != 3:
				raise ValueError(f"Solo se soportan volúmenes 2D/3D. Dimensiones detectadas: {self.volume_arr.ndim}")

			self.current_slice = self.volume_arr.shape[0] // 2
			self.slice_slider.config(from_=0, to=max(0, self.volume_arr.shape[0] - 1))
			self.slice_slider.set(self.current_slice)
			self.slice_label.config(text=str(self.current_slice))

			self.update_view()
			self.update_info()

		except Exception as exc:
			messagebox.showerror("Error al cargar", str(exc))

	def on_slice_change(self, value):
		if self.volume_arr is None:
			return
		self.current_slice = int(float(value))
		self.slice_label.config(text=str(self.current_slice))
		self.update_view()

	def normalize_for_display(self, slice_arr):
		slice_arr = slice_arr.astype(np.float32)
		vmin = float(np.percentile(slice_arr, 1))
		vmax = float(np.percentile(slice_arr, 99))

		if np.isclose(vmin, vmax):
			vmin = float(np.min(slice_arr))
			vmax = float(np.max(slice_arr))

		if np.isclose(vmin, vmax):
			return np.zeros_like(slice_arr, dtype=np.float32)

		slice_arr = np.clip(slice_arr, vmin, vmax)
		slice_arr = (slice_arr - vmin) / (vmax - vmin)
		return slice_arr

	def update_view(self):
		if self.volume_arr is None:
			return

		self.ax.clear()
		slice_arr = self.volume_arr[self.current_slice]
		display_arr = self.normalize_for_display(slice_arr)

		self.ax.imshow(display_arr, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
		self.ax.set_title(f"Corte {self.current_slice + 1}/{self.volume_arr.shape[0]}")
		self.ax.axis("off")
		self.canvas.draw_idle()

	def update_info(self):
		if self.volume_arr is None or self.volume_img is None:
			return

		arr = self.volume_arr.astype(np.float32)
		lines = [
			f"Archivo: {self.path_var.get().strip()}",
			f"Shape: {arr.shape}",
			f"Spacing: {self.volume_img.GetSpacing()}",
			f"Origin: {self.volume_img.GetOrigin()}",
			f"Direction: {self.volume_img.GetDirection()}",
			f"Dtype: {arr.dtype}",
			f"Min: {float(arr.min()):.4f}",
			f"Max: {float(arr.max()):.4f}",
		]

		self.info_text.delete("1.0", tk.END)
		self.info_text.insert(tk.END, "\n".join(lines))

	def on_close(self):
		try:
			plt.close(self.fig)
		except Exception:
			pass
		self.destroy()
		sys.exit(0)


if __name__ == "__main__":
	app = NiftiViewer()
	app.mainloop()
