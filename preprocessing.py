#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# preprocess_to_h5.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import h5py
import tifffile as tiff
import tensorflow as tf
from tqdm import tqdm
from scipy.ndimage import uniform_filter


# -----------------------------
# Parameters (edit these)
# -----------------------------
INPUT_FOLDER = Path("foldre_to_tiff_stacks")
SAVE_PATH = Path("train.h5")

PATCH_SIZE = 512
DOWNSAMPLE_TO = 1024
BATCH_SIZE = 200

# Channel selection (based on your code)
# x: DIC, Reflection
X_CHANNELS = (6, 4)
# y: DAPI, Phalloidin, Lectin
Y_CHANNELS = (0, 2, 3)


# -----------------------------
# Preprocessing helpers
# -----------------------------
def local_contrast_normalization(img: np.ndarray, window_size: int = 129, eps: float = 1e-8) -> np.ndarray:
    """Local z-score normalization using a uniform filter."""
    local_mean = uniform_filter(img, size=window_size)
    local_sqr_mean = uniform_filter(img**2, size=window_size)
    local_var = local_sqr_mean - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0.0))
    return (img - local_mean) / (local_std + eps)


def normalize_to_unit_range_stackwise(volume: np.ndarray, lower: float = 1, upper: float = 99) -> np.ndarray:
    """Percentile clipping then scaling to [0, 1]."""
    p1, p99 = np.percentile(volume, (lower, upper))
    volume_clipped = np.clip(volume, p1, p99)
    return (volume_clipped - p1) / (p99 - p1 + 1e-8)


def preprocess_x_data(x_data: np.ndarray) -> np.ndarray:
    """
    x_data: (Z, H, W, 2) where channel 0 = DIC, channel 1 = Reflection
    """
    Z, H, W, _ = x_data.shape
    x_norm = np.zeros_like(x_data, dtype=np.float32)

    dic_stack = np.zeros((Z, H, W), dtype=np.float32)
    for z in range(Z):
        dic_stack[z] = local_contrast_normalization(x_data[z, :, :, 0])
    x_norm[..., 0] = normalize_to_unit_range_stackwise(dic_stack)

    ref_stack = x_data[..., 1]
    x_norm[..., 1] = normalize_to_unit_range_stackwise(ref_stack)

    return x_norm.astype(np.float32)


def normalize_labels(
    y_data: np.ndarray,
    lower_percentile: float = 1,
    upper_percentile: float = 99.94,
    threshold: float = 0.15,
) -> np.ndarray:
    """
    y_data: (Z, H, W, 3) where channels are [DAPI, Phalloidin, Lectin]
    """
    Z, H, W, C = y_data.shape
    y_norm = np.zeros_like(y_data, dtype=np.float32)

    for c in range(C):
        stack = y_data[..., c]
        p1, p99 = np.percentile(stack, (lower_percentile, upper_percentile))
        norm_stack = np.clip((stack - p1) / (p99 - p1 + 1e-8), 0, 1)

        # Thresholding for DAPI and Lectin (channels 0 and 2)
        if c in (0, 2):
            norm_stack[norm_stack < threshold] = 0.0

        y_norm[..., c] = norm_stack

    return y_norm.astype(np.float32)


def list_tiffs(folder: Path) -> List[Path]:
    exts = {".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def read_and_reorder(path: Path) -> np.ndarray:
    """
    Reads a tif stack and returns array shaped (Z, H, W, C).
    Your original code assumed input was (Z, C, H, W) and transposed to (Z, H, W, C).
    """
    img = tiff.imread(str(path)).astype(np.float32)

    # If it is already (Z, H, W, C), leave it.
    # If it is (Z, C, H, W), transpose.
    if img.ndim == 4 and img.shape[1] < img.shape[2] and img.shape[1] < img.shape[3]:
        img = np.transpose(img, (0, 2, 3, 1))

    if img.ndim != 4:
        raise ValueError(f"Expected 4D stack, got shape {img.shape} for {path.name}")

    return img


def resize_stack(stack: np.ndarray, out_hw: int) -> np.ndarray:
    """Resize each z-slice to (out_hw, out_hw). stack is (Z, H, W, C)."""
    Z = stack.shape[0]
    out = np.zeros((Z, out_hw, out_hw, stack.shape[-1]), dtype=np.float32)
    for z in range(Z):
        out[z] = tf.image.resize(stack[z], (out_hw, out_hw), method="bilinear").numpy()
    return out


# -----------------------------
# Main pipeline
# -----------------------------
def main(
    input_folder: Path = INPUT_FOLDER,
    save_path: Path = SAVE_PATH,
    patch_size: int = PATCH_SIZE,
    downsample_to: int = DOWNSAMPLE_TO,
    batch_size: int = BATCH_SIZE,
) -> None:
    image_files = list_tiffs(input_folder)
    if not image_files:
        raise FileNotFoundError(f"No .tif/.tiff files found in: {input_folder}")

    # Preview Z from first file
    first = read_and_reorder(image_files[0])
    Z = first.shape[0]

    patches_per_image = (downsample_to // patch_size) ** 2
    total_patches = len(image_files) * patches_per_image

    print(f"Found {len(image_files)} tiff(s) in {input_folder}")
    print(f"Z={Z}, patch_size={patch_size}, downsample_to={downsample_to}")
    print(f"Patches per image: {patches_per_image} | Total patches: {total_patches}")
    print(f"Saving to: {save_path}")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_path, "w") as f:
        dset_x = f.create_dataset(
            "x_patches",
            shape=(total_patches, Z, patch_size, patch_size, 2),
            dtype=np.float32,
            compression=None,
            chunks=(1, Z, patch_size, patch_size, 2),
        )
        dset_y = f.create_dataset(
            "y_patches",
            shape=(total_patches, Z, patch_size, patch_size, 3),
            dtype=np.float32,
            compression=None,
            chunks=(1, Z, patch_size, patch_size, 3),
        )

        buffer_x, buffer_y = [], []
        idx = 0

        for path in tqdm(image_files, desc="Processing & saving"):
            img = read_and_reorder(path)  # (Z, H, W, C)

            # Select channels
            x_img = img[..., list(X_CHANNELS)]  # (Z, H, W, 2)
            y_img = img[..., list(Y_CHANNELS)]  # (Z, H, W, 3)

            # Normalize
            x_norm = preprocess_x_data(x_img)
            y_norm = normalize_labels(y_img)

            # Resize
            x_down = resize_stack(x_norm, downsample_to)
            y_down = resize_stack(y_norm, downsample_to)

            # Patchify
            for h in range(0, downsample_to, patch_size):
                for w in range(0, downsample_to, patch_size):
                    x_patch = x_down[:, h:h + patch_size, w:w + patch_size, :]
                    y_patch = y_down[:, h:h + patch_size, w:w + patch_size, :]

                    buffer_x.append(x_patch)
                    buffer_y.append(y_patch)

                    if len(buffer_x) == batch_size:
                        dset_x[idx:idx + batch_size] = np.asarray(buffer_x, dtype=np.float32)
                        dset_y[idx:idx + batch_size] = np.asarray(buffer_y, dtype=np.float32)
                        idx += batch_size
                        buffer_x.clear()
                        buffer_y.clear()

        # Final flush
        if buffer_x:
            n = len(buffer_x)
            dset_x[idx:idx + n] = np.asarray(buffer_x, dtype=np.float32)
            dset_y[idx:idx + n] = np.asarray(buffer_y, dtype=np.float32)
            idx += n

        print(f"Done. Wrote {idx} patches to {save_path}")


if __name__ == "__main__":
    main()

