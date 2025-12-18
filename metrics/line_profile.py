#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .common import discover_pairs, read_tiff, z_project, normalize_pair_to_01

def run_random_roi(
    folder: str | Path,
    gt_mode: str = "avg",
    zproj: str = "max",
    roi_w: int = 1024,
    roi_h: int = 256,
    seed: int | None = None,
    colors: dict | None = None,
):
    folder = Path(folder)
    pairs = discover_pairs(folder, gt_mode=gt_mode)
    rng = np.random.default_rng(seed)

    # default colors (edit as you like)
    if colors is None:
        colors = {"gt": "#0072B2", "pred": "#D55E00"}  # Okabe–Ito

    idx = rng.choice(sorted(pairs.keys(), key=lambda k: int(k) if isinstance(k, str) else k))
    gt_path = Path(pairs[idx]["gt"])
    pr_path = Path(pairs[idx]["pred"])

    out_dir = gt_path.parent / "metrics_output"
    out_dir.mkdir(exist_ok=True)

    gt = z_project(read_tiff(gt_path), zproj)
    pr = z_project(read_tiff(pr_path), zproj)
    gt01, pr01 = normalize_pair_to_01(gt, pr)

    H, W = gt01.shape
    rw, rh = min(roi_w, W), min(roi_h, H)
    x0 = 0 if W == rw else int(rng.integers(0, W - rw + 1))
    y0 = 0 if H == rh else int(rng.integers(0, H - rh + 1))
    x1, y1 = x0 + rw, y0 + rh

    gt_roi = gt01[y0:y1, x0:x1]
    pr_roi = pr01[y0:y1, x0:x1]
    x = np.arange(rw)
    gt_prof = gt_roi.mean(axis=0)
    pr_prof = pr_roi.mean(axis=0)
    yc = rh // 2

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gt_roi, cmap="gray")
    ax1.add_line(Line2D([0, rw-1], [yc, yc], color=colors["gt"], linewidth=3))
    ax1.set_title(f"GT ROI idx {idx} [x:{x0}:{x1}, y:{y0}:{y1}]")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pr_roi, cmap="gray")
    ax2.add_line(Line2D([0, rw-1], [yc, yc], color=colors["pred"], linewidth=3))
    ax2.set_title(f"Pred ROI idx {idx} [x:{x0}:{x1}, y:{y0}:{y1}]")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(gt01, cmap="gray")
    ax3.add_patch(plt.Rectangle((x0, y0), rw, rh, edgecolor=colors["gt"], facecolor="none", linewidth=2))
    ax3.add_line(Line2D([x0, x1-1], [y0+yc, y0+yc], color=colors["gt"], linewidth=2))
    ax3.set_title("GT (context)")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(pr01, cmap="gray")
    ax4.add_patch(plt.Rectangle((x0, y0), rw, rh, edgecolor=colors["pred"], facecolor="none", linewidth=2))
    ax4.add_line(Line2D([x0, x1-1], [y0+yc, y0+yc], color=colors["pred"], linewidth=2))
    ax4.set_title("Pred (context)")
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(x, gt_prof, color=colors["gt"], linewidth=3, label="GT")
    ax5.plot(x, pr_prof, color=colors["pred"], linewidth=3, label="Pred")
    ax5.set_xlabel("Position across ROI width (px)")
    ax5.set_ylabel("Normalized intensity")
    ax5.set_title(f"Line profiles (ROI {rw}×{rh})")
    ax5.legend()

    fig.tight_layout()
    out_png = out_dir / f"pair_{idx}_ROI_{rw}x{rh}_line_profile.png"
    fig.savefig(out_png, dpi=150)
    return out_png

if __name__ == "__main__":
    print(run_random_roi("predictions_folder"))

