"""YAML configuration with defaults. Paths are relative to the project folder (modelos/mamba)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import yaml

DEFAULTS = {
    "experiment": "smoke",
    "seed": 12345,
    "data": {
        "root": "../../HECKTOR2025_raw",
        "centers": None,  # list of center prefixes, None = all
        "cases_per_center": None,  # subset size per center, None = all cases
        "split": {
            "mode": "stratified",  # "stratified" (70/20/10 per center) or "center" (held-out test institution)
            "fractions": [0.7, 0.2, 0.1],  # train / val / test
            "test_centers": None,  # for mode "center", e.g. ["CHUP"]
        },
        "target_spacing_xyz": [1.0, 1.0, 3.0],
        "ct_window": [-200.0, 400.0],
        "pet_norm": "zscore",  # "zscore" or "clip"
        "suv_clip": 5.0,
        "roi": {
            "body_hu": -500.0,
            "body_suv": 0.5,
            "suv_anchor": 2.0,
            "down_mm": 270.0,
            "margin_vox": [8, 8, 8],
            "min_size_vox": [96, 96, 32],
            "main_body": True,  # keep only the patient's main connected component (drops the table)
            "open_iters": 2,  # coarse-grid opening before the component search (2 x 4 mm)
        },
        "load_threads": 2,
        "max_cached_cases": None,  # rolling RAM cache size for training, None = keep all
        "refresh_per_epoch": 4,  # cases swapped per epoch when the cache is smaller than the dataset
    },
    "model": {
        "variant": "base",
        "patch_size_zyx": [32, 128, 128],
        "max_stages": 6,
        "min_feature_size": 4,
        "chunk_size": 8,
        "n_cga_levels": None,
        "n_mamba_levels": None,
    },
    "train": {
        "epochs": 5,
        "iters_per_epoch": 50,
        "batch_size": 2,
        "lr": 0.01,
        "momentum": 0.99,
        "weight_decay": 3.0e-5,
        "fg_fraction": 0.33,
        "amp": True,
        "grad_clip": 12.0,
        "num_classes": 3,
        "save_every": 1,
        "val_every": 0,  # epochs between validations on val cases (0 = only at the end)
        "val_cases_during_training": 4,
    },
    "augment": {
        "enabled": True,
        "context_margin": 0.25,
        "p_rotation": 0.2,
        "rotation_deg": 30.0,
        "rotation_oop_deg": 15.0,
        "p_scale": 0.2,
        "scale": [0.7, 1.4],
        "mirror_axes": [0, 1, 2],
        "p_brightness": 0.15,
        "brightness": [0.75, 1.25],
        "p_contrast": 0.15,
        "contrast": [0.75, 1.25],
        "p_gamma": 0.3,
        "gamma": [0.7, 1.5],
        "p_noise": 0.1,
        "noise_std": [0.0, 0.1],
        "p_blur": 0.2,
        "blur_sigma": [0.5, 1.0],
    },
    "infer": {"step_fraction": 0.5, "mirror": False, "batch_size": 4},
    "output": {"root": "runs"},
}


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        _deep_update(cfg, user)
    return cfg


def resolve(path: str, base: Optional[Path] = None) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base or Path.cwd()) / p
