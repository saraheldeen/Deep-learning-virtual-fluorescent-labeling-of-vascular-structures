#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import disk, closing, remove_small_objects

from .common import discover_pairs, read_tiff, z_project, normalize_pair_to_01


def tile2x2(arr: np.ndarray):
    assert arr.ndim == 2
    H, W = arr.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    arr = arr[:H2, :W2]
    h, w = H2 // 2, W2 // 2
    return [
        ("TL", arr[0:h, 0:w]),
        ("TR", arr[0:h, w:W2]),
        ("BL", arr[h:H2, 0:w]),
        ("BR", arr[h:H2, w:W2]),
    ]


def safe_closing_bool(mask_bool: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask_bool
    se = disk(int(radius))
    try:
        return closing(mask_bool, selem=se)  # older skimage
    except TypeError:
        try:
            return closing(mask_bool, se)
        except Exception:
            from scipy.ndimage import binary_closing
            return binary_closing(mask_bool, structure=np.asarray(se, dtype=bool))


def density(mask_bool: np.ndarray) -> float:
    total = mask_bool.size if mask_bool.size else 1
    return float(mask_bool.sum()) / float(total)


def make_gt_binary_global(
    gt01: np.ndarray,
    gt_p: float = 15.0,
    use_max_with_otsu: bool = True,
    remove_small: bool = True,
    min_size: int = 16,
):
    gt = np.clip(gt01.astype(np.float32), 0, 1)
    p15 = float(np.percentile(gt, gt_p))
    thr = p15
    if use_max_with_otsu and np.std(gt) > 1e-6:
        try:
            thr = max(p15, float(threshold_otsu(gt)))
        except Exception:
            thr = p15
    gt_bin = gt >= thr
    if remove_small:
        gt_bin = remove_small_objects(gt_bin, min_size=int(min_size))
    return gt_bin, float(thr), float(p15)


def binarize_pred(
    pr01: np.ndarray,
    policy: str = "otsu",          # {"otsu","percentile","fixed"}
    percentile: float = 95.0,
    fixed_t: float = 0.5,
    smooth_sigma: float = 0.8,
    close_radius: int = 1,
    fallback_t: float = 0.5,
):
    x = pr01.astype(np.float32)
    if smooth_sigma and smooth_sigma > 0:
        x = gaussian_filter(x, sigma=float(smooth_sigma))

    if policy == "otsu":
        T = float(threshold_otsu(x)) if np.std(x) > 1e-6 else float(fallback_t)
    elif policy == "percentile":
        T = float(np.percentile(x, float(percentile)))
    elif policy == "fixed":
        T = float(fixed_t)
    else:
        T = float(fallback_t)

    m = x >= T
    m = safe_closing_bool(m.astype(bool), int(close_radius))
    return m


def run(
    folder: str | Path,
    gt_mode: str = "avg",
    zproj: str = "max",
    use_tiling_2x2: bool = True,
    # GT
    gt_p: float = 15.0,
    gt_use_max_with_otsu: bool = True,
    gt_remove_small: bool = True,
    gt_min_size: int = 16,
    # Pred
    pred_policy: str = "otsu",
    pred_percentile: float = 95.0,
    pred_fixed_t: float = 0.5,
    pred_smooth_sigma: float = 0.8,
    pred_close_radius: int = 1,
    # Plots
    make_plots: bool = True,
) -> Path:
    folder = Path(folder)
    pairs = discover_pairs(folder, gt_mode=gt_mode)

    rows = []
    for idx, d in sorted(pairs.items(), key=lambda kv: int(kv[0]) if isinstance(kv[0], str) else kv[0]):
        gt = z_project(read_tiff(d["gt"]), zproj)
        pr = z_project(read_tiff(d["pred"]), zproj)
        gt01, pr01 = normalize_pair_to_01(gt, pr)

        gt_bin_global, thr_used, p_used = make_gt_binary_global(
            gt01,
            gt_p=gt_p,
            use_max_with_otsu=gt_use_max_with_otsu,
            remove_small=gt_remove_small,
            min_size=gt_min_size,
        )

        if use_tiling_2x2:
            gt_tiles = tile2x2(gt_bin_global.astype(np.uint8))
            pr_tiles = tile2x2(pr01)
            for (tid_gt, gt_tile_bin), (tid_pr, pr_tile) in zip(gt_tiles, pr_tiles):
                assert tid_gt == tid_pr
                pred_mask = binarize_pred(
                    pr_tile,
                    policy=pred_policy,
                    percentile=pred_percentile,
                    fixed_t=pred_fixed_t,
                    smooth_sigma=pred_smooth_sigma,
                    close_radius=pred_close_radius,
                    fallback_t=thr_used,
                )
                dens_gt = density(gt_tile_bin.astype(bool))
                dens_pr = density(pred_mask)
                rows.append({
                    "Index": int(idx),
                    "Tile": tid_gt,
                    "GT_density": dens_gt,
                    "Pred_density": dens_pr,
                    "Density_diff": float(dens_pr - dens_gt),
                    "GT_area_px": int(gt_tile_bin.sum()),
                    "Pred_area_px": int(pred_mask.sum()),
                    "Total_px": int(gt_tile_bin.size),
                    "GT_thr_used": float(thr_used),
                    "GT_p_global": float(p_used),
                })
        else:
            pred_mask = binarize_pred(
                pr01,
                policy=pred_policy,
                percentile=pred_percentile,
                fixed_t=pred_fixed_t,
                smooth_sigma=pred_smooth_sigma,
                close_radius=pred_close_radius,
                fallback_t=thr_used,
            )
            dens_gt = density(gt_bin_global)
            dens_pr = density(pred_mask)
            rows.append({
                "Index": int(idx),
                "GT_density": dens_gt,
                "Pred_density": dens_pr,
                "Density_diff": float(dens_pr - dens_gt),
                "GT_area_px": int(gt_bin_global.sum()),
                "Pred_area_px": int(pred_mask.sum()),
                "Total_px": int(gt_bin_global.size),
                "GT_thr_used": float(thr_used),
                "GT_p_global": float(p_used),
            })

    df = pd.DataFrame(rows).sort_values(["Index", *(["Tile"] if use_tiling_2x2 else [])]).reset_index(drop=True)

    out_dir = folder / "metrics_output"
    out_dir.mkdir(exist_ok=True, parents=True)
    gran = "tiles2x2" if use_tiling_2x2 else "wholeimage"
    out_csv = out_dir / f"UEAI_vessel_density_GT_vs_Pred_{gran}.csv"
    df.to_csv(out_csv, index=False)

    if make_plots:
        labels = (df["Index"].astype(str) + "_" + df["Tile"]).tolist() if use_tiling_2x2 else df["Index"].astype(str).tolist()
        x = np.arange(len(labels))
        width = 0.4

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 4))
        ax.bar(x - width/2, df["GT_density"].values, width, label="GT")
        ax.bar(x + width/2, df["Pred_density"].values, width, label="Prediction")
        ax.set_ylabel("Vessel density")
        ax.set_xlabel("Image" + ("_Tile" if use_tiling_2x2 else ""))
        ax.set_title("UEA I vessel density: GT vs Prediction")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"UEAI_vessel_density_{gran}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 3.5))
        ax.bar(labels, df["Density_diff"].values)
        ax.set_ylabel("Pred - GT")
        ax.set_xlabel("Image" + ("_Tile" if use_tiling_2x2 else ""))
        ax.set_title("UEA I vessel density difference")
        plt.xticks(rotation=90)
        fig.tight_layout()
        fig.savefig(out_dir / f"UEAI_vessel_density_diff_{gran}.png", dpi=150)
        plt.close(fig)

    return out_csv


if __name__ == "__main__":
    print(run("ueai_predictions_folder", gt_mode="avg", zproj="max"))

