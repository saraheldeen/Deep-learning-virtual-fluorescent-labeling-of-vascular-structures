#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations
import re, math
from pathlib import Path
import numpy as np
import tifffile as tiff

_pair_gt_avg   = re.compile(r"^AVG_gt_stack_(\d+)\.tif$", re.IGNORECASE)
_pair_gt_plain = re.compile(r"^gt_stack_(\d+)\.tif$",     re.IGNORECASE)
_pair_pred     = re.compile(r"^pred_stack_(\d+)\.tif$",   re.IGNORECASE)

def discover_pairs(folder: Path, gt_mode: str = "avg") -> dict:
    """
    gt_mode:
      - "avg"   : matches AVG_gt_stack_<i>.tif
      - "plain" : matches gt_stack_<i>.tif
    Returns: {idx_str_or_int: {"gt": Path, "pred": Path}}
    """
    folder = Path(folder)
    pairs = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue

        if gt_mode == "avg":
            m = _pair_gt_avg.match(p.name)
            if m:
                idx = m.group(1)
                pairs.setdefault(idx, {})["gt"] = p
                continue
        else:
            m = _pair_gt_plain.match(p.name)
            if m:
                idx = int(m.group(1))
                pairs.setdefault(idx, {})["gt"] = p
                continue

        m = _pair_pred.match(p.name)
        if m:
            idx = m.group(1) if gt_mode == "avg" else int(m.group(1))
            pairs.setdefault(idx, {})["pred"] = p

    return {k: v for k, v in pairs.items() if "gt" in v and "pred" in v}

def read_tiff(path: Path) -> np.ndarray:
    return np.asarray(tiff.imread(str(path)), dtype=np.float32)

def z_project(vol: np.ndarray, method: str = "max") -> np.ndarray:
    """If vol is 3D (Z,H,W) do projection; if 2D returns as-is."""
    if vol.ndim == 2:
        return vol
    if vol.ndim == 3:
        if method == "mean":
            return np.mean(vol, axis=0)
        return np.max(vol, axis=0)
    raise ValueError(f"Expected 2D or 3D, got shape {vol.shape}")

def normalize_pair_to_01(gt: np.ndarray, pred: np.ndarray, percentiles=(0, 99.9)):
    combined = np.concatenate([gt.ravel(), pred.ravel()])
    lo, hi = np.percentile(combined, percentiles)
    if not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(combined)), float(np.max(combined))
    scale = max(hi - lo, 1e-6)
    gt_n = np.clip((gt - lo) / scale, 0, 1).astype(np.float32)
    pr_n = np.clip((pred - lo) / scale, 0, 1).astype(np.float32)
    return gt_n, pr_n

def pearson_r(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    am = a.mean(); bm = b.mean()
    num = np.sum((a-am)*(b-bm))
    den = math.sqrt(np.sum((a-am)**2) * np.sum((b-bm)**2)) + eps
    return float(num / den)

def ccc(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = np.mean((a - ma)*(b - mb))
    return float((2*cov) / (va + vb + (ma - mb)**2 + eps))

