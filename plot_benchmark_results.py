"""
plot_benchmark_results.py

Reads the per-design TSV produced by rfantibody_benchmark.py and generates
three publication-quality figures:

  1. Bar plot — mean ipTM per arm  (± SD error bars)
  2. Bar plot — mean DockQ per arm (± SD error bars)
  3. Scatter plot — ipTM vs DockQ, colour-coded by arm

Usage:
    python plot_benchmark_results.py --input results.tsv
    python plot_benchmark_results.py --input results.tsv --output_dir figures/
    python plot_benchmark_results.py --input results.tsv --format pdf --dpi 300
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IPTM_THRESHOLD  = 0.6
DOCKQ_THRESHOLD = 0.4

ARM_LABELS = {
    "A": "A: Vanilla",
    "B": "B: Anchored",
    "C": "C: Beam",
    "D": "D: Beam+Anchor",
}
ARM_COLORS = {
    "A": "#6366f1",   # indigo
    "B": "#f59e0b",   # amber
    "C": "#10b981",   # emerald
    "D": "#ef4444",   # red
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", na_values=["NA", "na", ""])
    required = {"arm", "iptm", "dockq", "success", "gpu_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TSV is missing required columns: {missing}")

    df["success"] = df["success"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)
    df["arm_label"] = df["arm"].map(ARM_LABELS).fillna(df["arm"])
    df["color"]     = df["arm"].map(ARM_COLORS).fillna("#888888")
    return df


def arm_summary(df: pd.DataFrame) -> pd.DataFrame:
    arms = df["arm"].unique()
    rows = []
    for arm in sorted(arms):
        sub = df[df["arm"] == arm]
        gpu_h = sub["gpu_seconds"].sum() / 3600.0
        n_succ = sub["success"].sum()
        rows.append({
            "arm":           arm,
            "label":         ARM_LABELS.get(arm, arm),
            "color":         ARM_COLORS.get(arm, "#888"),
            "n":             len(sub),
            "n_success":     int(n_succ),
            "gpu_hours":     gpu_h,
            "succ_per_gpuh": n_succ / gpu_h if gpu_h > 0 else 0.0,
            "mean_iptm":     sub["iptm"].mean(),
            "sd_iptm":       sub["iptm"].std(ddof=0),
            "mean_dockq":    sub["dockq"].mean(),
            "sd_dockq":      sub["dockq"].std(ddof=0),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar_plot(
    summary:    pd.DataFrame,
    df:         pd.DataFrame,
    metric:     str,           # "iptm" or "dockq"
    threshold:  float,
    ylabel:     str,
    title:      str,
    out_path:   str,
    fmt:        str,
    dpi:        int,
):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    x      = np.arange(len(summary))
    means  = summary[f"mean_{metric}"].values
    sds    = summary[f"sd_{metric}"].values
    colors = summary["color"].tolist()
    labels = summary["label"].tolist()
    ns     = summary["n"].tolist()
    arms   = summary["arm"].tolist()

    bars = ax.bar(x, means, yerr=sds, capsize=5, width=0.55,
                  color=colors, edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=1.5, ecolor="#475569", capthick=1.5),
                  zorder=3)

    # individual data points — jittered horizontally within each bar
    rng = np.random.default_rng(42)
    for xi, arm, color in zip(x, arms, colors):
        vals = df.loc[df["arm"] == arm, metric].dropna().values
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(xi + jitter, vals,
                   s=18, color=color, alpha=0.55,
                   edgecolors="white", linewidths=0.4,
                   zorder=4)

    # value labels above each bar
    for bar, m, sd in zip(bars, means, sds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + sd + 0.012,
                f"{m:.3f}",
                ha="center", va="bottom", fontsize=9, color="#1e293b")

    # threshold line
    ax.axhline(threshold, color="#ef4444", linewidth=1.2,
               linestyle="--", zorder=2)
    ax.text(len(summary) - 0.5, threshold + 0.008,
            f"{threshold}", color="#ef4444", fontsize=8, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0, max(means + sds) * 1.25)
    ax.yaxis.grid(True, linestyle=":", color="#e2e8f0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _scatter_plot(
    df:       pd.DataFrame,
    out_path: str,
    fmt:      str,
    dpi:      int,
):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    arms = sorted(df["arm"].unique())
    for arm in arms:
        sub     = df[df["arm"] == arm]
        color   = ARM_COLORS.get(arm, "#888")
        label   = ARM_LABELS.get(arm, arm)
        success = sub[sub["success"]]
        failed  = sub[~sub["success"]]

        # failed designs — small, faded
        if len(failed):
            ax.scatter(failed["iptm"], failed["dockq"],
                       s=35, color=color, alpha=0.35,
                       edgecolors="none", zorder=3)
        # successful designs — larger, opaque, white edge
        if len(success):
            ax.scatter(success["iptm"], success["dockq"],
                       s=70, color=color, alpha=0.9,
                       edgecolors="white", linewidths=0.8,
                       zorder=4, label=label)

    # threshold lines
    ax.axvline(IPTM_THRESHOLD,  color="#94a3b8", linewidth=1.0,
               linestyle="--", zorder=2)
    ax.axhline(DOCKQ_THRESHOLD, color="#94a3b8", linewidth=1.0,
               linestyle="--", zorder=2)
    ax.text(IPTM_THRESHOLD + 0.003, ax.get_ylim()[0] + 0.005,
            f"ipTM={IPTM_THRESHOLD}", color="#64748b", fontsize=8)
    ax.text(ax.get_xlim()[0] + 0.003, DOCKQ_THRESHOLD + 0.004,
            f"DockQ={DOCKQ_THRESHOLD}", color="#64748b", fontsize=8)

    # legend: arms (colour) + success indicator
    arm_handles = [
        mpatches.Patch(color=ARM_COLORS.get(a, "#888"), label=ARM_LABELS.get(a, a))
        for a in arms
    ]
    success_handle = plt.scatter([], [], s=60, color="#555", alpha=0.9,
                                  edgecolors="white", linewidths=0.8,
                                  label="Success")
    fail_handle    = plt.scatter([], [], s=30, color="#555", alpha=0.35,
                                  edgecolors="none", label="Failed")
    ax.legend(
        handles=arm_handles + [success_handle, fail_handle],
        fontsize=9, framealpha=0.9, loc="upper left",
        ncol=1, handlelength=1.5,
    )

    ax.set_xlabel("ipTM", fontsize=11)
    ax.set_ylabel("DockQ", fontsize=11)
    ax.set_title("ipTM vs DockQ — all designs", fontsize=12,
                 fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle=":", color="#e2e8f0", zorder=0)
    ax.xaxis.grid(True, linestyle=":", color="#e2e8f0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table (printed to stdout)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(summary: pd.DataFrame):
    print("\n" + "=" * 78)
    print("BENCHMARK SUMMARY  —  success: DockQ > 0.23 AND ipTM > 0.6")
    print("=" * 78)
    hdr = f"{'Arm':<18}  {'N':>4}  {'Succ':>4}  {'GPU-h':>7}  "
    hdr += f"{'Succ/GPU-h':>10}  {'ipTM (mean±SD)':>16}  {'DockQ (mean±SD)':>16}"
    print(hdr)
    print("-" * 78)
    for _, r in summary.iterrows():
        print(
            f"{r['label']:<18}  {r['n']:>4}  {r['n_success']:>4}  "
            f"{r['gpu_hours']:>7.2f}  {r['succ_per_gpuh']:>10.3f}  "
            f"{r['mean_iptm']:>6.3f} ± {r['sd_iptm']:.3f}  "
            f"{r['mean_dockq']:>6.3f} ± {r['sd_dockq']:.3f}"
        )
    print("=" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot RFantibody benchmark results from a per-design TSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",      required=True,
                   help="Per-design TSV produced by rfantibody_benchmark.py")
    p.add_argument("--output_dir", default="",
                   help="Directory for output figures (default: same as input)")
    p.add_argument("--format",     default="png",
                   choices=["png", "pdf", "svg", "eps"],
                   help="Output figure format (default: png)")
    p.add_argument("--dpi",        type=int, default=200,
                   help="Resolution for raster formats (default: 200)")
    p.add_argument("--stem",       default="",
                   help="Output filename stem (default: input file stem)")
    return p.parse_args()


def main():
    args = parse_args()

    tsv_path   = str(Path(args.input).resolve())
    out_dir    = str(Path(args.output_dir).resolve()) if args.output_dir \
                 else str(Path(tsv_path).parent)
    os.makedirs(out_dir, exist_ok=True)

    stem = args.stem or Path(tsv_path).stem
    fmt  = args.format
    dpi  = args.dpi

    print(f"[plot] Loading {tsv_path}")
    df      = load_results(tsv_path)
    summary = arm_summary(df)
    print_summary(summary)

    print("[plot] Generating figures…")

    _bar_plot(
        summary=summary, df=df,
        metric="iptm",
        threshold=IPTM_THRESHOLD,
        ylabel="Mean ipTM",
        title="Mean ipTM by Arm  (± SD)",
        out_path=os.path.join(out_dir, f"{stem}_iptm.{fmt}"),
        fmt=fmt, dpi=dpi,
    )

    _bar_plot(
        summary=summary, df=df,
        metric="dockq",
        threshold=DOCKQ_THRESHOLD,
        ylabel="Mean DockQ",
        title="Mean DockQ by Arm  (± SD)",
        out_path=os.path.join(out_dir, f"{stem}_dockq.{fmt}"),
        fmt=fmt, dpi=dpi,
    )

    _scatter_plot(
        df=df,
        out_path=os.path.join(out_dir, f"{stem}_scatter.{fmt}"),
        fmt=fmt, dpi=dpi,
    )

    print(f"[plot] Done. Figures written to: {out_dir}/")


if __name__ == "__main__":
    main()