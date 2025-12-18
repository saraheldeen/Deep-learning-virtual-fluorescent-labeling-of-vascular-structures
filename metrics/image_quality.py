#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

from .common import discover_pairs, read_tiff, z_project, normalize_pair_to_01, pearson_r, ccc

def ssim_for_volume(gt01: np.ndarray, pr01: np.ndarray) -> float:
    if gt01.ndim == 3:
        vals = [ssim(gt01[i], pr01[i], data_range=1.0, gaussian_weights=True) for i in range(gt01.shape[0])]
        return float(np.mean(vals))
    return float(ssim(gt01, pr01, data_range=1.0, gaussian_weights=True))

def compute_metrics(gt01: np.ndarray, pr01: np.ndarray) -> dict:
    diff = (gt01 - pr01).astype(np.float64)
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    r = pearson_r(gt01, pr01)
    c = ccc(gt01, pr01)
    s = ssim_for_volume(gt01, pr01)
    p = float(psnr(gt01, pr01, data_range=1.0))
    return {
        "MSE": mse, "RMSE": rmse, "MAE": mae,
        "Pearson_r": r, "CCC": c, "SSIM": s, "PSNR_dB": p
    }

def run(folder: str | Path, gt_mode: str = "avg", zproj: str = "max") -> Path:
    folder = Path(folder)
    pairs = discover_pairs(folder, gt_mode=gt_mode)
    rows = []
    for idx, d in sorted(pairs.items(), key=lambda kv: int(kv[0]) if isinstance(kv[0], str) else kv[0]):
        gt = read_tiff(d["gt"])
        pr = read_tiff(d["pred"])
        # projection for fair 2D comparison if needed
        gt2 = z_project(gt, zproj)
        pr2 = z_project(pr, zproj)
        gt01, pr01 = normalize_pair_to_01(gt2, pr2)
        rows.append({"Index": idx, **compute_metrics(gt01, pr01)})

    out_dir = folder / "metrics_output"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "image_quality_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv

if __name__ == "__main__":
    print(run("predictions_folder", gt_mode="avg", zproj="max"))

