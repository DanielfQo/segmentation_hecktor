# -*- coding: utf-8 -*-
"""
Script to compare best_model.pth and swinunetr_structured_pruned.pth on HECKTOR 2025 Preprocessed Data
and plot the segmentation comparison against Ground Truth.
"""

import os
import json
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib

from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    AsDiscrete,
)
from monai.data import decollate_batch

# Define paths
DATASET_ROOT = "cropped"
DATASET_JSON = os.path.join(DATASET_ROOT, "dataset.json")
MODEL_BEST_PATH = "best_model.pth"
MODEL_PRUNED_PATH = "swinunetr_structured_pruned.pth"
OUTPUT_PLOT_PATH = "comparativa_segmentacion.png"

# ROI matches the one from training
ROI = (96, 96, 32)
NUM_CLASSES = 3
IN_CHANNELS = 2

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models on preprocessed HECKTOR 2025 data.")
    parser.add_argument("--case_id", type=str, default=None, help="Specific Case ID to visualize (e.g., CHUM-001). If not provided, it auto-selects a case with visible tumor.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset list
    if not os.path.exists(DATASET_JSON):
        raise FileNotFoundError(f"Could not find dataset.json at {DATASET_JSON}")
    
    with open(DATASET_JSON, "r") as f:
        dataset_data = json.load(f)
    
    cases = dataset_data.get("training", [])
    if not cases:
        raise ValueError("No training cases found in dataset.json")
    
    # Initialize models
    print("Loading models...")
    model_best = SwinUNETR(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, feature_size=48).to(device)
    model_pruned = SwinUNETR(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, feature_size=48).to(device)

    # Load checkpoints
    if os.path.exists(MODEL_BEST_PATH):
        print(f"Loading {MODEL_BEST_PATH}...")
        ckpt_best = torch.load(MODEL_BEST_PATH, map_location=device, weights_only=False)
        state_dict_best = ckpt_best["model"] if isinstance(ckpt_best, dict) and "model" in ckpt_best else ckpt_best
        model_best.load_state_dict(state_dict_best)
    else:
        raise FileNotFoundError(f"Best model checkpoint not found at {MODEL_BEST_PATH}")

    if os.path.exists(MODEL_PRUNED_PATH):
        print(f"Loading {MODEL_PRUNED_PATH}...")
        ckpt_pruned = torch.load(MODEL_PRUNED_PATH, map_location=device, weights_only=False)
        state_dict_pruned = ckpt_pruned["model"] if isinstance(ckpt_pruned, dict) and "model" in ckpt_pruned else ckpt_pruned
        model_pruned.load_state_dict(state_dict_pruned)
    else:
        raise FileNotFoundError(f"Pruned model checkpoint not found at {MODEL_PRUNED_PATH}")

    model_best.eval()
    model_pruned.eval()

    # Preprocessing pipeline (validation transforms equivalent)
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])

    # Select case
    selected_case = None
    if args.case_id:
        for c in cases:
            label_name = os.path.basename(c["label"])
            case_name = label_name.split(".")[0]
            if args.case_id in case_name or case_name in args.case_id:
                selected_case = c
                break
        if selected_case is None:
            print(f"Case {args.case_id} not found. Falling back to automatic selection.")

    if selected_case is None:
        # Automatically select a case with a reasonable amount of tumor labels to visualize
        print("Searching for a case with a tumor...")
        # To speed up, we shuffle and check labels
        random.seed(65)
        shuffled_cases = list(cases)
        random.shuffle(shuffled_cases)
        for c in shuffled_cases:
            label_path = os.path.join(DATASET_ROOT, c["label"].lstrip("./"))
            lbl_img = nib.load(label_path)
            lbl_data = lbl_img.get_fdata()
            tumor_voxels = np.sum(lbl_data > 0)
            if tumor_voxels > 500: # Select a case with at least 500 tumor voxels
                selected_case = c
                print(f"Selected case: {c['label']} (with {tumor_voxels} tumor voxels)")
                break
        if selected_case is None:
            selected_case = cases[0]
            print(f"No case found with > 500 tumor voxels. Defaulting to first case: {selected_case['label']}")

    # Load and preprocess
    case_data = {
        "image": [os.path.join(DATASET_ROOT, p.lstrip("./")) for p in selected_case["image"]],
        "label": os.path.join(DATASET_ROOT, selected_case["label"].lstrip("./"))
    }
    
    print(f"Preprocessing case...")
    processed_data = transforms(case_data)
    
    # Add batch dimension [1, C, H, W, D]
    image_tensor = processed_data["image"].unsqueeze(0).to(device)
    label_tensor = processed_data["label"].unsqueeze(0).to(device)

    # Crop around the tumor to speed up CPU inference significantly
    gt_numpy = label_tensor[0, 0].cpu().numpy()
    gt_coords = np.argwhere(gt_numpy > 0)
    if len(gt_coords) > 0:
        center_h, center_w, center_d = gt_coords.mean(axis=0).astype(int)
    else:
        center_h, center_w, center_d = np.array(gt_numpy.shape) // 2
        
    crop_h = min(128, gt_numpy.shape[0])
    crop_w = min(128, gt_numpy.shape[1])
    crop_d = min(64, gt_numpy.shape[2])
    
    start_h = max(0, min(center_h - crop_h // 2, gt_numpy.shape[0] - crop_h))
    end_h = min(gt_numpy.shape[0], start_h + crop_h)
    start_h = max(0, end_h - crop_h)
    
    start_w = max(0, min(center_w - crop_w // 2, gt_numpy.shape[1] - crop_w))
    end_w = min(gt_numpy.shape[1], start_w + crop_w)
    start_w = max(0, end_w - crop_w)
    
    start_d = max(0, min(center_d - crop_d // 2, gt_numpy.shape[2] - crop_d))
    end_d = min(gt_numpy.shape[2], start_d + crop_d)
    start_d = max(0, end_d - crop_d)
    
    print(f"Cropping volume from {gt_numpy.shape} to ({end_h-start_h}, {end_w-start_w}, {end_d-start_d}) to speed up CPU inference...")
    image_tensor = image_tensor[:, :, start_h:end_h, start_w:end_w, start_d:end_d]
    label_tensor = label_tensor[:, :, start_h:end_h, start_w:end_w, start_d:end_d]

    print("Running predictions...")
    with torch.no_grad():
        # Inference using sliding window
        outputs_best = sliding_window_inference(
            inputs=image_tensor, roi_size=ROI, sw_batch_size=2,
            predictor=model_best, overlap=0.25, mode="gaussian"
        )
        outputs_pruned = sliding_window_inference(
            inputs=image_tensor, roi_size=ROI, sw_batch_size=2,
            predictor=model_pruned, overlap=0.25, mode="gaussian"
        )

        # Convert predictions to discrete labels via argmax
        pred_best = torch.argmax(outputs_best[0], dim=0).cpu().numpy()
        pred_pruned = torch.argmax(outputs_pruned[0], dim=0).cpu().numpy()
        gt_label = label_tensor[0, 0].cpu().numpy()

    # Get input image array for visualization (using CT channel 0)
    ct_volume = image_tensor[0, 0].cpu().numpy()
    pet_volume = image_tensor[0, 1].cpu().numpy()

    # Find the slice with the maximum tumor area in Ground Truth
    tumor_mask = (gt_label > 0)
    tumor_slice_sums = np.sum(tumor_mask, axis=(0, 1)) # Sum across X and Y
    best_slice_idx = np.argmax(tumor_slice_sums)
    max_voxels = tumor_slice_sums[best_slice_idx]

    print(f"Best slice index along Z axis: {best_slice_idx} (Ground truth tumor voxels in slice: {max_voxels})")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Slices to extract
    ct_slice = ct_volume[:, :, best_slice_idx]
    pet_slice = pet_volume[:, :, best_slice_idx]
    gt_slice = gt_label[:, :, best_slice_idx]
    best_slice = pred_best[:, :, best_slice_idx]
    pruned_slice = pred_pruned[:, :, best_slice_idx]

    # Row 1: Overlays on CT
    # Plot GT
    axes[0, 0].imshow(ct_slice, cmap="gray")
    # Mask background out for transparency
    masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
    axes[0, 0].imshow(masked_gt, cmap="autumn", alpha=0.6, interpolation="none")
    axes[0, 0].set_title("Ground Truth (on CT)")
    axes[0, 0].axis("off")

    # Plot Best Model
    axes[0, 1].imshow(ct_slice, cmap="gray")
    masked_best = np.ma.masked_where(best_slice == 0, best_slice)
    axes[0, 1].imshow(masked_best, cmap="autumn", alpha=0.6, interpolation="none")
    axes[0, 1].set_title("best_model.pth Prediction (on CT)")
    axes[0, 1].axis("off")

    # Plot Pruned Model
    axes[0, 2].imshow(ct_slice, cmap="gray")
    masked_pruned = np.ma.masked_where(pruned_slice == 0, pruned_slice)
    axes[0, 2].imshow(masked_pruned, cmap="autumn", alpha=0.6, interpolation="none")
    axes[0, 2].set_title("swinunetr_structured_pruned.pth (on CT)")
    axes[0, 2].axis("off")

    # Row 2: Overlays on PET
    axes[1, 0].imshow(pet_slice, cmap="hot")
    axes[1, 0].imshow(masked_gt, cmap="cool", alpha=0.4, interpolation="none")
    axes[1, 0].set_title("Ground Truth (on PET)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pet_slice, cmap="hot")
    axes[1, 1].imshow(masked_best, cmap="cool", alpha=0.4, interpolation="none")
    axes[1, 1].set_title("best_model.pth (on PET)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(pet_slice, cmap="hot")
    axes[1, 2].imshow(masked_pruned, cmap="cool", alpha=0.4, interpolation="none")
    axes[1, 2].set_title("swinunetr_structured_pruned.pth (on PET)")
    axes[1, 2].axis("off")

    case_id = os.path.basename(selected_case["label"]).split(".")[0]
    plt.suptitle(f"Segmentation Comparison for Case: {case_id} (Z-Slice: {best_slice_idx})", fontsize=16)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight", dpi=150)
    print(f"Plot saved successfully to {OUTPUT_PLOT_PATH}")
    
    # Try displaying
    try:
        plt.show()
    except Exception as e:
        print("Could not display plot interactively (probably running in headless mode). Plot was saved successfully.")

if __name__ == "__main__":
    main()
