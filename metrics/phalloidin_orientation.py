#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class OrientationPlotConfig:
    # where to look for the fibriltool output folder (default: alongside your GT/PRED stacks)
    out_subdir: str = "fibriltool_output"

    # input CSV pattern (your code used ALL_pairs_*_orientation_compare.csv)
    csv_glob: str = "ALL_pairs_*_orientation_compare.csv"

    # plotting
    ylim: float = 0.05
    dpi: int = 200

    # style
    font_size: int = 18
    title_size: int = 20
    label_size: int = 18
    legend_size: int = 16
    tick_size: int = 14


def nematic_stats(bin_centers_deg: np.ndarray, prob: np.ndarray):
    """
    Compute nematic mean direction mu in [0,180) and order parameter R in [0,1]
    from a distribution over 0..180 degrees.
    """
    centers = np.asarray(bin_centers_deg, dtype=np.float64)
    w = np.asarray(prob, dtype=np.float64)
    w = np.clip(w, 0, None)

    phi = np.deg2rad(centers)
    C = np.sum(w * np.cos(2 * phi))
    S = np.sum(w * np.sin(2 * phi))
    R = np.hypot(C, S) / (np.sum(w) + 1e-12)
    mu = (0.5 * np.degrees(np.arctan2(S, C))) % 180.0
    return float(mu), float(R)


def _apply_pub_style(cfg: OrientationPlotConfig):
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.size": cfg.font_size,
        "axes.titlesize": cfg.title_size,
        "axes.labelsize": cfg.label_size,
        "legend.fontsize": cfg.legend_size,
        "xtick.labelsize": cfg.tick_size,
        "ytick.labelsize": cfg.tick_size,
    })


def load_latest_orientation_csv(out_dir: Path, cfg: OrientationPlotConfig) -> Path:
    csvs = sorted(out_dir.glob(cfg.csv_glob))
    if not csvs:
        raise FileNotFoundError(
            f"No orientation CSV found in {out_dir} matching '{cfg.csv_glob}'. "
            f"Expected something like ALL_pairs_*_orientation_compare.csv"
        )
    return csvs[-1]


def run(
    folder_with_stacks: str | Path,
    cfg: OrientationPlotConfig = OrientationPlotConfig(),
) -> dict:
    """
    Looks for: <folder_with_stacks>/<cfg.out_subdir>/<latest matching csv>
    Expects CSV columns:
      - bin_center_deg
      - GT_prob
      - Pred_prob

    Saves:
      - ALL_pairs_orientation_compare_mathtext_ylim005.png
      - ALL_pairs_orientation_rose_mathtext_ylim005.png
      - ALL_pairs_orientation_summary.csv  (mu/R for GT and Pred)
    """
    folder_with_stacks = Path(folder_with_stacks)
    out_dir = folder_with_stacks / cfg.out_subdir
    out_dir.mkdir(exist_ok=True, parents=True)

    csv_path = load_latest_orientation_csv(out_dir, cfg)
    df = pd.read_csv(csv_path)

    needed = {"bin_center_deg", "GT_prob", "Pred_prob"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    centers = df["bin_center_deg"].to_numpy(dtype=np.float64)
    gt_prob = df["GT_prob"].to_numpy(dtype=np.float64)
    pr_prob = df["Pred_prob"].to_numpy(dtype=np.float64)

    mu_gt, R_gt = nematic_stats(centers, gt_prob)
    mu_pr, R_pr = nematic_stats(centers, pr_prob)

    _apply_pub_style(cfg)

    # ------------------------
    # Linear plot
    # ------------------------
    plt.figure(figsize=(9, 4.2))
    plt.plot(
        centers, gt_prob,
        label=fr"$\mathrm{{GT}}\ (\mu\approx {mu_gt:.1f}^\circ,\ R\approx {R_gt:.3f})$"
    )
    plt.plot(
        centers, pr_prob,
        label=fr"$\mathrm{{Pred}}\ (\mu\approx {mu_pr:.1f}^\circ,\ R\approx {R_pr:.3f})$"
    )
    plt.ylim(0, cfg.ylim)
    plt.xlabel(r"Orientation ($^\circ$, 0–180)")
    plt.ylabel("Probability")
    plt.title(r"$\mathrm{Aggregated\ Orientation\ Distribution}$")
    plt.legend()
    plt.tight_layout()
    lin_out = out_dir / "ALL_pairs_orientation_compare_mathtext_ylim005.png"
    plt.savefig(lin_out, dpi=cfg.dpi)
    plt.close()

    # ------------------------
    # Rose plot (0..pi; 0..180 deg)
    # ------------------------
    theta_rad = np.deg2rad(centers)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(theta_rad, gt_prob, label=r"$\mathrm{GT}$")
    ax.plot(theta_rad, pr_prob, label=r"$\mathrm{Pred}$")

    try:
        ax.set_rlim(0, cfg.ylim)
    except Exception:
        ax.set_rmax(cfg.ylim)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_title(r"$\mathrm{Rose:\ GT\ vs\ Pred}$")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    fig.tight_layout()

    rose_out = out_dir / "ALL_pairs_orientation_rose_mathtext_ylim005.png"
    fig.savefig(rose_out, dpi=cfg.dpi)
    plt.close(fig)

    # ------------------------
    # Save summary CSV
    # ------------------------
    summary = pd.DataFrame([{
        "source_csv": csv_path.name,
        "mu_gt_deg": mu_gt,
        "R_gt": R_gt,
        "mu_pred_deg": mu_pr,
        "R_pred": R_pr,
        "ylim_used": cfg.ylim,
    }])
    summary_csv = out_dir / "ALL_pairs_orientation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    return {
        "input_csv": csv_path,
        "linear_plot": lin_out,
        "rose_plot": rose_out,
        "summary_csv": summary_csv,
        "mu_gt_deg": mu_gt,
        "R_gt": R_gt,
        "mu_pred_deg": mu_pr,
        "R_pred": R_pr,
    }


if __name__ == "__main__":
    # Edit this path to the folder that contains your GT/PRED stacks and fibriltool_output/
    res = run(".")
    print("Wrote:")
    for k, v in res.items():
        if isinstance(v, Path):
            print(f"  {k}: {v}")

