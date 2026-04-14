"""
filter_by_iptm.py

Parses a benchmark results TSV and prints the design IDs of any designs
with an ipTM value above a user-defined cutoff.

Usage:
    python filter_by_iptm.py --input results.tsv --cutoff 0.6
    python filter_by_iptm.py --input results.tsv --cutoff 0.7 --arm B
    python filter_by_iptm.py --input results.tsv --cutoff 0.6 --output hits.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def filter_by_iptm(
    input_tsv:  str,
    cutoff:     float,
    arm:        str  = "",
    output_txt: str  = "",
):
    df = pd.read_csv(input_tsv, sep="\t", na_values=["NA", ""])

    if "iptm" not in df.columns:
        raise ValueError("Column 'iptm' not found in input TSV.")
    if "design_id" not in df.columns:
        raise ValueError("Column 'design_id' not found in input TSV.")

    mask = df["iptm"].notna() & (df["iptm"] > cutoff)

    if arm:
        if "arm" not in df.columns:
            raise ValueError("Column 'arm' not found — cannot filter by arm.")
        mask &= df["arm"].str.upper() == arm.upper()

    hits = df[mask].copy()
    hits = hits.sort_values("iptm", ascending=False)

    arm_str = f" (arm {arm.upper()})" if arm else ""
    print(f"[filter] {len(hits)} design(s) with ipTM > {cutoff}{arm_str} "
          f"(of {len(df)} total):\n")

    if len(hits):
        col_w = max(len(str(did)) for did in hits["design_id"])
        print(f"  {'design_id':<{col_w}}  {'ipTM':>6}  {'arm':>4}")
        print(f"  {'-'*col_w}  {'------'}  {'----'}")
        for _, row in hits.iterrows():
            arm_val = str(row.get("arm", "")) if "arm" in hits.columns else ""
            print(f"  {str(row['design_id']):<{col_w}}  "
                  f"{row['iptm']:>6.3f}  {arm_val:>4}")
    else:
        print("  (none)")

    if output_txt:
        Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt, "w") as fh:
            for did in hits["design_id"]:
                fh.write(f"{did}\n")
        print(f"\n[filter] Design IDs written to: {output_txt}")

    return hits["design_id"].tolist()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter benchmark results by ipTM cutoff.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",   required=True,
                   help="Benchmark results TSV")
    p.add_argument("--cutoff",  type=float, required=True,
                   help="ipTM threshold (designs strictly above this are returned)")
    p.add_argument("--arm",     default="",
                   help="Optional: restrict to a single arm (A, B, C, or D)")
    p.add_argument("--output",  default="",
                   help="Optional: write matching design IDs to a text file "
                        "(one per line)")
    return p.parse_args()


def main():
    args = parse_args()
    filter_by_iptm(
        input_tsv=args.input,
        cutoff=args.cutoff,
        arm=args.arm,
        output_txt=args.output,
    )


if __name__ == "__main__":
    main()