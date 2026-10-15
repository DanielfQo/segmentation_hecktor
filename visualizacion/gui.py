import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sys


class ViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PET/CT Viewer")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        default_patient_path = Path(__file__).resolve().parent.parent / "HECKTOR 2025 Training Data" / "CHUM-001"

        self.preprocessed = False
        self.ct_img = None
        self.pet_img = None
        self.mask_img = None
        self.raw_ct_img = None
        self.raw_pet_img = None
        self.raw_mask_img = None
        self.ct_arr = None
        self.pet_arr = None
        self.mask_arr = None
        self.raw_ct_arr = None
        self.raw_pet_arr = None
        self.raw_mask_arr = None
        self.z = 0

        # Top frame: path entry and load / toggle
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text="Ruta (carpeta del paciente o archivo):").pack(side="left")
        self.path_var = tk.StringVar(value=str(default_patient_path))
        self.path_entry = ttk.Entry(top, textvariable=self.path_var, width=60)
        self.path_entry.pack(side="left", padx=6)

        ttk.Button(top, text="Cargar", command=self.load).pack(side="left", padx=6)
        self.toggle_btn = ttk.Button(top, text="Usar preprocesado: OFF", command=self.toggle_preprocessed)
        self.toggle_btn.pack(side="left", padx=6)

        # Notebook with image view and histogram view
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)

        self.view_tab = ttk.Frame(self.notebook)
        self.hist_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.view_tab, text="Vista")
        self.notebook.add(self.hist_tab, text="Histogramas")

        # Image view: CT, PET and mask
        self.fig, (self.ax_ct, self.ax_pet, self.ax_mask) = plt.subplots(1, 3, figsize=(12, 4))
        plt.subplots_adjust(wspace=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.view_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Histogram view: one histogram per image
        self.hist_fig, (self.ax_hist_ct, self.ax_hist_pet, self.ax_hist_mask) = plt.subplots(1, 3, figsize=(12, 4))
        plt.subplots_adjust(wspace=0.35)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.hist_tab)
        self.hist_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Slider for z axis
        slider_frame = ttk.Frame(self)
        slider_frame.pack(fill="x", padx=8, pady=6)
        ttk.Label(slider_frame, text="Eje Z:").pack(side="left")
        self.z_var = tk.IntVar()
        self.z_slider = ttk.Scale(slider_frame, from_=0, to=0, orient="horizontal", command=self.on_slider)
        self.z_slider.pack(fill="x", expand=True, side="left", padx=6)

        # Info panel
        info_frame = ttk.Frame(self)
        info_frame.pack(fill="x", padx=8, pady=6)
        self.info_text = tk.Text(info_frame, height=4)
        self.info_text.pack(fill="x")

    def resample_to_reference(self, image, reference_image, interpolator, default_pixel_value=0):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(default_pixel_value)
        return resampler.Execute(image)

    def infer_raw_case_dir(self, ct_file: Path):
        case_id = ct_file.name
        suffixes = [
            "_CT_preprocessed.nii.gz",
            "_PT_preprocessed.nii.gz",
            "_mask_preprocessed.nii.gz",
            "__CT.nii.gz",
            "__PT.nii.gz",
            "_CT.nii.gz",
            "_PT.nii.gz",
            ".nii.gz",
        ]

        for suffix in suffixes:
            if case_id.endswith(suffix):
                case_id = case_id[: -len(suffix)]
                break

        search_roots = [
            Path(__file__).resolve().parent.parent / "HECKTOR 2025 Training Data",
            Path(__file__).resolve().parent.parent,
            ct_file.parent.parent / "HECKTOR 2025 Training Data",
        ]

        for root in search_roots:
            candidate = root / case_id
            if candidate.exists():
                return candidate

        if ct_file.parent.name not in {"imagesTr", "labelsTr"}:
            return ct_file.parent

        return None

    def toggle_preprocessed(self):
        self.preprocessed = not self.preprocessed
        self.toggle_btn.config(text=f"Usar preprocesado: {'ON' if self.preprocessed else 'OFF'}")
        if self.path_var.get().strip():
            self.load()

    def find_files(self, base_path: Path):
        p = Path(base_path)
        if p.is_file():
            directory = p.parent
        else:
            directory = p

        def choose_first(paths):
            return paths[0] if paths else None

        def find_mask_file(directory: Path, prefer_preprocessed: bool):
            if prefer_preprocessed:
                preprocessed_masks = list(directory.glob("*_mask_preprocessed.nii.gz"))
                if preprocessed_masks:
                    return preprocessed_masks[0]

            candidate = directory / f"{directory.name}.nii.gz"
            if candidate.exists():
                return candidate

            masks = [
                f for f in directory.glob("*.nii.gz")
                if "__CT" not in f.name
                and "__PT" not in f.name
                and "_CT_preprocessed" not in f.name
                and "_PT_preprocessed" not in f.name
                and "_mask_preprocessed" not in f.name
            ]
            return choose_first(masks)

        # Patterns
        if self.preprocessed:
            ct_list = list(directory.glob("*_CT_preprocessed.nii.gz"))
            pet_list = list(directory.glob("*_PT_preprocessed.nii.gz"))
            mask_list = list(directory.glob("*_mask_preprocessed.nii.gz"))
        else:
            # Accept both double-underscore naming and single underscore fallback
            ct_list = list(directory.glob("*__CT.nii.gz"))
            if not ct_list:
                ct_list = list(directory.glob("*_CT.nii.gz"))

            pet_list = list(directory.glob("*__PT.nii.gz"))
            if not pet_list:
                pet_list = list(directory.glob("*_PT.nii.gz"))

        ct = choose_first(ct_list)
        pet = choose_first(pet_list)
        mask = find_mask_file(directory, prefer_preprocessed=self.preprocessed)

        # In normal mode, never fall back to the preprocessed mask.
        if not self.preprocessed and mask is None:
            mask = choose_first([
                f for f in directory.glob("*.nii.gz")
                if "_mask_preprocessed" not in f.name
            ])

        # In preprocessed mode, prefer the preprocessed mask, but allow a fallback
        # to any available mask if the expected file name is missing.
        if self.preprocessed and mask is None:
            mask = choose_first([
                f for f in directory.glob("*.nii.gz")
                if "__CT" not in f.name and "__PT" not in f.name
            ])

        return ct, pet, mask

    def load(self):
        path_str = self.path_var.get().strip()
        if not path_str:
            messagebox.showerror("Error", "Introduce una ruta válida.")
            return

        ct_file, pet_file, mask_file = self.find_files(Path(path_str))

        if not ct_file or not pet_file or not mask_file:
            messagebox.showerror("Error", f"No se encontraron archivos necesarios en {path_str}\nCT: {ct_file}\nPET: {pet_file}\nMask: {mask_file}")
            return

        try:
            self.ct_img = sitk.ReadImage(str(ct_file))
            self.pet_img = sitk.ReadImage(str(pet_file))
            self.mask_img = sitk.ReadImage(str(mask_file))

            raw_case_dir = self.infer_raw_case_dir(ct_file)
            if raw_case_dir is not None:
                raw_ct = raw_case_dir / f"{raw_case_dir.name}__CT.nii.gz"
                raw_pet = raw_case_dir / f"{raw_case_dir.name}__PT.nii.gz"
                raw_mask = raw_case_dir / f"{raw_case_dir.name}.nii.gz"

                if raw_ct.exists() and raw_pet.exists() and raw_mask.exists():
                    self.raw_ct_img = sitk.ReadImage(str(raw_ct))
                    self.raw_pet_img = sitk.ReadImage(str(raw_pet))
                    self.raw_mask_img = sitk.ReadImage(str(raw_mask))
                else:
                    self.raw_ct_img = self.ct_img
                    self.raw_pet_img = self.pet_img
                    self.raw_mask_img = self.mask_img
            else:
                self.raw_ct_img = self.ct_img
                self.raw_pet_img = self.pet_img
                self.raw_mask_img = self.mask_img

            # Use CT as the reference geometry so all three volumes share the same Z axis.
            self.pet_img = self.resample_to_reference(
                self.pet_img,
                self.ct_img,
                sitk.sitkLinear,
            )
            self.mask_img = self.resample_to_reference(
                self.mask_img,
                self.ct_img,
                sitk.sitkNearestNeighbor,
            )

            self.ct_arr = sitk.GetArrayFromImage(self.ct_img)
            self.pet_arr = sitk.GetArrayFromImage(self.pet_img)
            self.mask_arr = sitk.GetArrayFromImage(self.mask_img)

            self.raw_ct_arr = sitk.GetArrayFromImage(self.raw_ct_img)
            self.raw_pet_arr = sitk.GetArrayFromImage(self.raw_pet_img)
            self.raw_mask_arr = sitk.GetArrayFromImage(self.raw_mask_img)

            z_slices = self.ct_arr.shape[0]
            self.z = z_slices // 2
            self.z_slider.config(from_=0, to=max(0, z_slices - 1))
            self.z_slider.set(self.z)

            self.update_display()
            self.update_info()
            self.update_histograms()

        except Exception as e:
            messagebox.showerror("Error al cargar", str(e))

    def on_slider(self, val):
        try:
            self.z = int(float(val))
            self.update_display()
        except Exception:
            pass

    def update_display(self):
        if self.ct_arr is None:
            return

        self.ax_ct.clear()
        self.ax_pet.clear()
        self.ax_mask.clear()

        import matplotlib.colors as mcolors

        ct_slice = self.ct_arr[self.z]
        pet_slice = self.pet_arr[self.z]
        mask_slice = self.mask_arr[self.z]

        # Define custom colormap for the 3 labels (0: background, 1: primary tumor, 2: secondary tumor)
        cmap_mask = mcolors.ListedColormap(['black', 'red', 'yellow'])
        norm_mask = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_mask.N)

        self.ax_ct.imshow(ct_slice, cmap="gray")
        self.ax_ct.set_title(f"CT (z={self.z})")
        self.ax_ct.axis("off")

        self.ax_pet.imshow(pet_slice, cmap="hot")
        self.ax_pet.set_title(f"PET (z={self.z})")
        self.ax_pet.axis("off")

        self.ax_mask.imshow(mask_slice, cmap=cmap_mask, norm=norm_mask, interpolation="nearest")
        self.ax_mask.set_title(f"MASK (z={self.z})")
        self.ax_mask.axis("off")

        self.canvas.draw_idle()

    def update_info(self):
        lines = []
        def img_info(name, img, arr, raw_arr=None):
            real_arr = raw_arr if raw_arr is not None else arr
            real_arr = real_arr.astype(np.float32)
            size = img.GetSize()
            return (
                f"{name}: size={size}, array_shape={arr.shape}, spacing={img.GetSpacing()}, "
                f"dtype={arr.dtype}, raw_min={float(real_arr.min()):.4f}, raw_max={float(real_arr.max()):.4f}"
            )

        lines.append(img_info("CT", self.ct_img, self.ct_arr, self.raw_ct_arr))
        lines.append(img_info("PET", self.pet_img, self.pet_arr, self.raw_pet_arr))
        lines.append(img_info("MASK", self.mask_img, self.mask_arr, self.raw_mask_arr))

        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "\n".join(lines))

    def update_histograms(self):
        if self.ct_arr is None:
            return

        self.ax_hist_ct.clear()
        self.ax_hist_pet.clear()
        self.ax_hist_mask.clear()

        ct_values = self.ct_arr.astype(np.float32).ravel()
        pet_values = self.pet_arr.astype(np.float32).ravel()
        mask_values = self.mask_arr.astype(np.float32).ravel()

        self.ax_hist_ct.hist(ct_values, bins=60, color="steelblue", alpha=0.85)
        self.ax_hist_ct.set_title("Hist CT")
        self.ax_hist_ct.set_xlabel("Intensidad")
        self.ax_hist_ct.set_ylabel("Frecuencia")

        self.ax_hist_pet.hist(pet_values, bins=60, color="darkorange", alpha=0.85)
        self.ax_hist_pet.set_title("Hist PET")
        self.ax_hist_pet.set_xlabel("Intensidad")
        self.ax_hist_pet.set_ylabel("Frecuencia")

        self.ax_hist_mask.hist(mask_values, bins=20, color="seagreen", alpha=0.85)
        self.ax_hist_mask.set_title("Hist MASK")
        self.ax_hist_mask.set_xlabel("Valor")
        self.ax_hist_mask.set_ylabel("Frecuencia")

        self.hist_fig.tight_layout()
        self.hist_canvas.draw_idle()

    def on_close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass
        try:
            plt.close(self.hist_fig)
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    app = ViewerApp()
    app.mainloop()
