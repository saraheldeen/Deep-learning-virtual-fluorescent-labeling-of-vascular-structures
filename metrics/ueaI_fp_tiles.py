#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.filters import threshold_otsu

from .common import discover_pairs, read_tiff, z_project, normalize_pair_to_01
from .image_quality import compute_metrics


# ------------------------------
# Tiling helpers (2x2)
# ------------------------------
def tile2x2(arr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Split 2D array into 2x2 non-overlapping tiles: TL/TR/BL/BR."""
    assert arr.ndim == 2, f"Expected 2D, got {arr.ndim}D"
    H, W = arr.shape
    H2 = (H // 2) * 2
    W2 = (W // 2) * 2
    arr = arr[:H2, :W2]
    h = H2 // 2
    w = W2 // 2
    return [
        ("TL", arr[0:h, 0:w]),
        ("TR", arr[0:h, w:W2]),
        ("BL", arr[h:H2, 0:w]),
        ("BR", arr[h:H2, w:W2]),
    ]


# ------------------------------
# GT vessel mask (adaptive percentile -> Otsu fallback)
# ------------------------------
def vessel_mask_from_gt_adaptive(
    gt_tile: np.ndarray,
    smooth_sigma: float = 1.0,
    percentiles_desc=(99, 98, 97, 96, 95, 94, 93, 92, 90),
    cov_range=(0.002, 0.05),  # 0.2%–5%
) -> Tuple[np.ndarray, float, float]:
    x = np.clip(gt_tile.astype(np.float32), 0, 1)
    x_s = gaussian_filter(x, sigma=smooth_sigma) if (smooth_sigma and smooth_sigma > 0) else x
    H, W = x_s.shape
    npx = max(1, H * W)

    for p in percentiles_desc:
        T = float(np.percentile(x_s, p))
        m = x_s >= T
        cov = float(m.sum()) / npx
        if cov_range[0] <= cov <= cov_range[1]:
            return m.astype(bool), T, cov

    if np.std(x_s) > 1e-6:
        try:
            T_otsu = float(threshold_otsu(x_s))
            m = x_s >= T_otsu
            cov = float(m.sum()) / npx
            return m.astype(bool), T_otsu, cov
        except Exception:
            pass

    # last resort
    p = percentiles_desc[-1]
    T = float(np.percentile(x_s, p))
    m = x_s >= T
    cov = float(m.sum()) / npx
    return m.astype(bool), T, cov


# ------------------------------
# Euclidean tolerance band -> background
# ------------------------------
def background_from_euclidean_tolerance(gt_mask: np.ndarray, radius_px: float = 2.0) -> np.ndarray:
    dist_to_gt = distance_transform_edt(~gt_mask)
    near_vessel = dist_to_gt <= float(radius_px)
    return ~near_vessel


# ------------------------------
# FP metrics
# ------------------------------
def compute_fp_metrics_refined(
    gt_tile: np.ndarray,
    pred_tile: np.ndarray,
    tol_radius_px: float = 2.0,
    smooth_sigma: float = 1.0,
) -> Dict[str, float]:
    eps = 1e-12
    gt = np.clip(gt_tile.astype(np.float32), 0, 1)
    pr = np.clip(pred_tile.astype(np.float32), 0, 1)

    # 1) GT vessel mask + threshold
    gt_mask, T_used, cov = vessel_mask_from_gt_adaptive(gt, smooth_sigma=smooth_sigma)

    # 2) Pred binary using same threshold
    pred_mask = pr >= T_used

    # 3) Strict confusion
    TP = float(np.logical_and(pred_mask, gt_mask).sum())
    FP = float(np.logical_and(pred_mask, ~gt_mask).sum())
    FN = float(np.logical_and(~pred_mask, gt_mask).sum())
    precision_strict = TP / (TP + FP + eps)
    recall_strict = TP / (TP + FN + eps)
    f1_strict = (2 * precision_strict * recall_strict) / (precision_strict + recall_strict + eps)

    # 4) Tolerance band background
    background = background_from_euclidean_tolerance(gt_mask, radius_px=tol_radius_px)

    # 5) Tolerant precision
    TP_tol = float(np.logical_and(pred_mask, ~background).sum())
    FP_tol = float(np.logical_and(pred_mask, background).sum())
    precision_tolerant = TP_tol / (TP_tol + FP_tol + eps)

    # 6) Continuous FP on background only (excess)
    diff = pr - gt
    excess = np.where(diff > 0, diff, 0.0)
    bg_excess = excess[background]

    FP_sum_bg = float(bg_excess.sum())
    FP_mean_bg = float(bg_excess.mean()) if bg_excess.size else 0.0
    FP_p95_bg = float(np.percentile(bg_excess, 95.0)) if bg_excess.size else 0.0
    total_pred = float(pr.sum()) + eps
    FP_fraction = FP_sum_bg / total_pred

    # 7) Binary FP area fractions on background
    FP_area_px = float(np.logical_and(pred_mask, background).sum())
    bg_px = int(background.sum())
    tile_px = int(gt.size)
    FP_area_frac_bg = (FP_area_px / bg_px) if bg_px > 0 else 0.0
    FP_area_frac_tile = FP_area_px / tile_px if tile_px > 0 else 0.0

    return {
        "Thr_used": float(T_used),
        "GT_mask_coverage": float(cov),
        "Precision_strict": float(precision_strict),
        "Recall_strict": float(recall_strict),
        "F1_strict": float(f1_strict),
        "Precision_tolerant": float(precision_tolerant),
        "FP_area_px": float(FP_area_px),
        "FP_area_frac_bg": float(FP_area_frac_bg),
        "FP_area_frac_tile": float(FP_area_frac_tile),
        "FP_sum_bg": float(FP_sum_bg),
        "FP_mean_bg": float(FP_mean_bg),
        "FP_p95_bg": float(FP_p95_bg),
        "FP_fraction": float(FP_fraction),
    }


def run(
    folder: str | Path,
    gt_mode: str = "avg",        # "avg" => AVG_gt_stack_*.tif, "plain" => gt_stack_*.tif
    zproj: str = "max",          # "max" or "mean"
    tol_radius_px: float = 2.0,
    smooth_sigma: float = 1.0,
) -> Path:
    folder = Path(folder)
    pairs = discover_pairs(folder, gt_mode=gt_mode)

    rows = []
    for idx, d in sorted(pairs.items(), key=lambda kv: int(kv[0]) if isinstance(kv[0], str) else kv[0]):
        gt = z_project(read_tiff(d["gt"]), zproj)
        pr = z_project(read_tiff(d["pred"]), zproj)
        gt01, pr01 = normalize_pair_to_01(gt, pr)

        gt_tiles = tile2x2(gt01)
        pr_tiles = tile2x2(pr01)

        for (tid_gt, gt_tile), (tid_pr, pr_tile) in zip(gt_tiles, pr_tiles):
            assert tid_gt == tid_pr
            base = compute_metrics(gt_tile, pr_tile)
            fp = compute_fp_metrics_refined(
                gt_tile, pr_tile,
                tol_radius_px=tol_radius_px,
                smooth_sigma=smooth_sigma,
            )
            rows.append({"Index": int(idx), "Tile": tid_gt, **base, **fp})

    df = pd.DataFrame(rows).sort_values(["Index", "Tile"]).reset_index(drop=True)

    out_dir = folder / "metrics_output"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "UEAI_metrics_tiles_refined.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    print(run("ueai_predictions_folder", gt_mode="avg", zproj="max"))

