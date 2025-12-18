#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations

import re, csv
from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
from skimage import morphology

from .common import discover_pairs, read_tiff, z_project


def minmax01(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, (0.5, 99.5))
    if not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(a)), float(np.max(a))
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def fast_otsu(img01: np.ndarray, downscale: int = 4) -> float:
    if downscale > 1:
        w = max(1, img01.shape[1] // downscale)
        h = max(1, img01.shape[0] // downscale)
        small = cv2.resize(img01, (w, h), interpolation=cv2.INTER_AREA)
    else:
        small = img01
    small_u8 = np.clip(small * 255, 0, 255).astype(np.uint8)
    ret, _ = cv2.threshold(small_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(ret) / 255.0


def binarize(img01: np.ndarray, mode: str = "otsu", percentile: float = 90.0, downscale_for_otsu: int = 4) -> np.ndarray:
    x = minmax01(img01)
    if mode == "percentile":
        t = float(np.percentile(x, float(percentile)))
    else:
        t = fast_otsu(x, downscale=downscale_for_otsu)
    m = x >= t
    return morphology.remove_small_objects(m, min_size=32)


def find_contours_cv(mask_bool: np.ndarray, approx_tol: float = 1.5):
    m = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if len(c) < 5:
            continue
        c = cv2.approxPolyDP(c, epsilon=float(approx_tol), closed=True)
        out.append(c.reshape(-1, 2))  # (x,y)
    return out


def draw_dashed_polyline(canvas, pts, value_or_color, line_width=2, dash_len=6, gap_len=4, s_step=2, color_mode=False):
    if len(pts) < 2:
        return

    diffs = np.diff(pts.astype(np.float32), axis=0, append=pts[:1])
    seglens = np.sqrt((diffs**2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seglens)])
    total = cum[-1]
    if total <= 0:
        return

    s_vals = np.arange(0, total, s_step)
    pattern = dash_len + gap_len
    on_mask = ((s_vals % pattern) < dash_len)

    seg_idx = np.searchsorted(cum[1:], s_vals, side="right")
    seg_idx = np.clip(seg_idx, 0, len(seglens) - 1)
    seg_s0 = cum[seg_idx]
    alpha = (s_vals - seg_s0) / (seglens[seg_idx] + 1e-12)

    p0 = pts[seg_idx]
    p1 = pts[(seg_idx + 1) % len(pts)]
    samp = (p0 + (p1 - p0) * alpha[:, None]).astype(np.int32)

    on_idx = np.where(on_mask)[0]
    if on_idx.size == 0:
        return
    splits = np.where(np.diff(on_idx) > 1)[0] + 1
    runs = np.split(on_idx, splits)

    for run in runs:
        if run.size == 1:
            if color_mode:
                cv2.circle(canvas, tuple(samp[run[0]]), max(line_width // 2, 1), value_or_color, thickness=-1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(canvas, tuple(samp[run[0]]), max(line_width // 2, 1), int(value_or_color), thickness=-1, lineType=cv2.LINE_AA)
        else:
            for i in range(run.size - 1):
                pA = tuple(samp[run[i]])
                pB = tuple(samp[run[i + 1]])
                if color_mode:
                    cv2.line(canvas, pA, pB, value_or_color, thickness=line_width, lineType=cv2.LINE_AA)
                else:
                    cv2.line(canvas, pA, pB, int(value_or_color), thickness=line_width, lineType=cv2.LINE_AA)


def dice_iou(a_bool: np.ndarray, b_bool: np.ndarray):
    inter = np.count_nonzero(a_bool & b_bool)
    sa = np.count_nonzero(a_bool)
    sb = np.count_nonzero(b_bool)
    denom = sa + sb
    dice = (2.0 * inter / denom) if denom > 0 else 1.0
    union = np.count_nonzero(a_bool | b_bool)
    iou = (inter / union) if union > 0 else 1.0
    return float(dice), float(iou), int(inter), int(sa), int(sb), int(union)


def run_batch(
    folder: str | Path,
    zproj: str = "max",
    thresh_mode: str = "otsu",            # "otsu" or "percentile"
    percentile_thresh: float = 90.0,
    background_source: str = "gt",        # "gt" | "pred" | "avg" for overlay only
    dash_len: int = 6,
    gap_len: int = 4,
    line_width: int = 3,
    approx_tol: float = 1.5,
    s_step: int = 2,
    downscale_for_otsu: int = 4,
    color_gt_bgr=(0, 255, 0),
    color_pr_bgr=(255, 0, 255),
) -> Path:
    folder = Path(folder)
    # nuclei uses gt_stack_*.tif + pred_stack_*.tif
    pairs = discover_pairs(folder, gt_mode="plain")
    out_dir = folder / "metrics_output"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for idx in sorted(pairs.keys()):
        gt = z_project(read_tiff(pairs[idx]["gt"]), zproj)
        pr = z_project(read_tiff(pairs[idx]["pred"]), zproj)

        gt01 = minmax01(gt)
        pr01 = minmax01(pr)

        mask_gt = binarize(gt01, mode=thresh_mode, percentile=percentile_thresh, downscale_for_otsu=downscale_for_otsu)
        mask_pr = binarize(pr01, mode=thresh_mode, percentile=percentile_thresh, downscale_for_otsu=downscale_for_otsu)

        contours_gt = find_contours_cv(mask_gt, approx_tol=approx_tol)
        contours_pr = find_contours_cv(mask_pr, approx_tol=approx_tol)

        H, W = gt01.shape
        gt_dashed = np.zeros((H, W), dtype=np.uint8)
        pr_dashed = np.zeros((H, W), dtype=np.uint8)

        for pts in contours_gt:
            draw_dashed_polyline(gt_dashed, pts, 255, line_width, dash_len, gap_len, s_step, color_mode=False)
        for pts in contours_pr:
            draw_dashed_polyline(pr_dashed, pts, 255, line_width, dash_len, gap_len, s_step, color_mode=False)

        tiff.imwrite((out_dir / f"gt_dashed_mask_pair_{idx}.tif").as_posix(), gt_dashed)
        tiff.imwrite((out_dir / f"pred_dashed_mask_pair_{idx}.tif").as_posix(), pr_dashed)

        dice, iou, inter, sa, sb, union = dice_iou(gt_dashed > 0, pr_dashed > 0)
        rows.append({
            "Index": int(idx),
            "Dice_dashed": dice,
            "IoU_dashed": iou,
            "GT_dashed_pixels": sa,
            "Pred_dashed_pixels": sb,
            "Intersect_pixels": inter,
            "Union_pixels": union,
        })

        # overlay preview
        if background_source == "gt":
            bg = gt01
        elif background_source == "pred":
            bg = pr01
        else:
            bg = np.clip(0.5 * (gt01 + pr01), 0, 1)

        bg_u8 = (bg * 255).astype(np.uint8)
        canvas_bgr = cv2.merge([bg_u8, bg_u8, bg_u8])

        for pts in contours_gt:
            draw_dashed_polyline(canvas_bgr, pts, color_gt_bgr, line_width, dash_len, gap_len, s_step, color_mode=True)
        for pts in contours_pr:
            draw_dashed_polyline(canvas_bgr, pts, color_pr_bgr, line_width, dash_len, gap_len, s_step, color_mode=True)

        overlay_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        plt.imsave((out_dir / f"dashed_overlay_pair_{idx}.png").as_posix(), overlay_rgb)

    df = pd.DataFrame(rows).sort_values("Index").reset_index(drop=True)
    csv_path = out_dir / "pair_metrics_dashed.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    import pandas as pd
    print(run_batch("dapi_metric_folder"))

