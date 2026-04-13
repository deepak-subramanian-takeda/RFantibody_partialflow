"""
merge_dockq_results.py

Merges DockQ scores from run_dockq_batch.py output into the benchmark
results TSV produced by rfantibody_benchmark_parallel.py.

The DockQ TSV uses full filenames (e.g. vanilla_91_seq.pdb) as the model
column, while the benchmark TSV uses truncated design_ids (e.g. vanilla_91).
This script strips known suffixes (_seq.pdb, _seq, .pdb) from the DockQ
model column to align the two.

Usage:
    python merge_dockq_results.py \
        --benchmark  1n8z_benchmark_results.tsv \
        --dockq      dockq_results.tsv \
        --output     1n8z_benchmark_results_merged.tsv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Suffixes to strip from DockQ model filenames to produce a bare design_id.
# Order matters: longer/more-specific patterns first.
_STRIP_PATTERNS = [
    r"_seq\.pdb$",
    r"_seq$",
    r"\.pdb$",
]

def _strip_to_design_id(filename: str) -> str:
    """
    Strip path and known suffixes from a DockQ model filename.

    Examples:
        vanilla_91_seq.pdb          → vanilla_91
        cp04_n0067_rfd_seq.pdb      → cp04_n0067_rfd
        1n8z_hlt_anchored_T50_5_grafted_enforced_seq.pdb
                                    → 1n8z_hlt_anchored_T50_5_grafted_enforced
    """
    name = Path(filename).name   # drop any directory component
    for pat in _STRIP_PATTERNS:
        name = re.sub(pat, "", name)
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Main merge logic
# ─────────────────────────────────────────────────────────────────────────────

def merge(
    benchmark_tsv: str,
    dockq_tsv:     str,
    output_tsv:    str,
):
    # ── Load ──────────────────────────────────────────────────────────────────
    bench = pd.read_csv(benchmark_tsv, sep="\t", na_values=["NA", ""])
    dockq = pd.read_csv(dockq_tsv,     sep="\t", na_values=["NA", ""])

    print(f"[merge] Benchmark rows : {len(bench)}")
    print(f"[merge] DockQ rows     : {len(dockq)}")

    # ── Build design_id key from DockQ model column ───────────────────────────
    dockq["design_id"] = dockq["model"].apply(_strip_to_design_id)

    # Warn about any duplicate design_ids in the DockQ file (shouldn't happen
    # but protects against scoring the same file twice)
    dupes = dockq[dockq.duplicated("design_id", keep=False)]
    if len(dupes):
        print(f"[WARN] {len(dupes)} duplicate design_id(s) in DockQ TSV "
              "— keeping last occurrence:")
        for did in dupes["design_id"].unique():
            print(f"  {did}")
        dockq = dockq.drop_duplicates("design_id", keep="last")

    # ── Columns to bring in from DockQ ────────────────────────────────────────
    # dockq_new replaces the existing dockq column in the benchmark TSV;
    # fnat, irmsd, lrmsd are added as new columns if not already present.
    dockq_cols = ["design_id", "dockq", "fnat", "irmsd", "lrmsd"]
    dockq_sub  = dockq[[c for c in dockq_cols if c in dockq.columns]].copy()
    dockq_sub  = dockq_sub.rename(columns={
        "dockq": "dockq_new",
        "fnat":  "fnat",
        "irmsd": "irmsd",
        "lrmsd": "lrmsd",
    })

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = bench.merge(dockq_sub, on="design_id", how="left")

    # Replace benchmark dockq with the freshly computed value where available;
    # keep the original where DockQ has no match (e.g. structures that failed).
    if "dockq_new" in merged.columns:
        updated = merged["dockq_new"].notna().sum()
        merged["dockq"] = merged["dockq_new"].combine_first(merged["dockq"])
        merged = merged.drop(columns=["dockq_new"])
        print(f"[merge] Updated dockq for {updated}/{len(merged)} rows")
    else:
        print("[WARN] No 'dockq' column found in DockQ TSV — dockq unchanged")

    # Recompute success flag with updated dockq values
    if "iptm" in merged.columns and "dockq" in merged.columns:
        merged["success"] = (
            merged["iptm"].gt(0.6) & merged["dockq"].gt(0.23)
        )
        print(f"[merge] Recomputed success: "
              f"{merged['success'].sum()} / {len(merged)} designs")

    # ── Report unmatched rows ─────────────────────────────────────────────────
    bench_ids = set(bench["design_id"].dropna())
    dockq_ids = set(dockq_sub["design_id"].dropna())

    unmatched_bench = bench_ids - dockq_ids
    unmatched_dockq = dockq_ids - bench_ids

    if unmatched_bench:
        print(f"[merge] {len(unmatched_bench)} benchmark design_id(s) with no "
              "DockQ match (dockq kept as-is):")
        for did in sorted(unmatched_bench)[:10]:
            print(f"  {did}")
        if len(unmatched_bench) > 10:
            print(f"  ... and {len(unmatched_bench) - 10} more")

    if unmatched_dockq:
        print(f"[merge] {len(unmatched_dockq)} DockQ design_id(s) not found "
              "in benchmark TSV (ignored):")
        for did in sorted(unmatched_dockq)[:10]:
            print(f"  {did}")
        if len(unmatched_dockq) > 10:
            print(f"  ... and {len(unmatched_dockq) - 10} more")

    # ── Write ─────────────────────────────────────────────────────────────────
    merged.to_csv(output_tsv, sep="\t", index=False, na_rep="NA")
    print(f"[merge] Merged TSV written to: {output_tsv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Merge DockQ batch scores into the rfantibody benchmark results TSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--benchmark", required=True,
                   help="Benchmark results TSV from rfantibody_benchmark_parallel.py")
    p.add_argument("--dockq",     required=True,
                   help="DockQ batch results TSV from run_dockq_batch.py")
    p.add_argument("--output",    default="",
                   help="Output TSV path (default: overwrites --benchmark in-place)")
    return p.parse_args()


def main():
    args   = parse_args()
    output = args.output or args.benchmark
    merge(
        benchmark_tsv=args.benchmark,
        dockq_tsv=args.dockq,
        output_tsv=output,
    )


if __name__ == "__main__":
    main()