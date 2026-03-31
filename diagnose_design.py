"""
diagnose_design.py

Quick structural sanity-check for a generated antibody design vs the
reference HLT complex. Prints everything needed to locate the source
of high ipAE without requiring PyMOL or ChimeraX.

Usage:
    python diagnose_design.py \
        --design  path/to/design_final.pdb \
        --ref     path/to/1n8z_hlt.pdb \
        --hotspots T570,T571,T572,T573
"""

import argparse
import sys
import numpy as np
from pathlib import Path


# ── minimal PDB parser ────────────────────────────────────────────────────────

def parse_ca(pdb_path, chains=None):
    coords = {}  # (chain, resnum) -> xyz
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain  = line[21]
            resnum = int(line[22:26])
            if chains and chain not in chains:
                continue
            xyz = np.array([float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54])])
            coords[(chain, resnum)] = xyz
    return coords


def centroid(coords_dict):
    pts = np.array(list(coords_dict.values()))
    return pts.mean(axis=0)


def dist(a, b):
    return float(np.linalg.norm(a - b))


# ── checks ────────────────────────────────────────────────────────────────────

def check_chain_presence(pdb_path):
    chains = set()
    resnums = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            ch = line[21]
            rn = int(line[22:26])
            chains.add(ch)
            resnums.setdefault(ch, []).append(rn)
    print(f"\n{'='*60}")
    print(f"Chain composition: {pdb_path}")
    print(f"{'='*60}")
    for ch in sorted(chains):
        nums = sorted(set(resnums[ch]))
        # detect gaps
        gaps = [nums[i] for i in range(1, len(nums)) if nums[i] != nums[i-1]+1]
        print(f"  Chain {ch}: {len(nums)} residues  "
              f"[{nums[0]}–{nums[-1]}]"
              + (f"  GAPS after: {gaps}" if gaps else "  (contiguous)"))
    return chains, resnums


def check_binder_target_distance(design_path, hotspot_str):
    hotspots = []
    for tok in hotspot_str.split(","):
        tok = tok.strip()
        ch, rn = tok[0], int(tok[1:])
        hotspots.append((ch, rn))

    binder_ca = parse_ca(design_path, chains={"H", "L"})
    target_ca = parse_ca(design_path, chains={"T"})

    if not binder_ca:
        print("\n[WARN] No binder (H/L) Cα atoms found in design PDB.")
        return
    if not target_ca:
        print("\n[WARN] No target (T) Cα atoms found in design PDB.")
        return

    b_cent = centroid(binder_ca)
    t_cent = centroid(target_ca)

    print(f"\n{'='*60}")
    print(f"Binder–Target distance check: {Path(design_path).name}")
    print(f"{'='*60}")
    print(f"  Binder centroid (H+L Cα): {b_cent.round(1)}")
    print(f"  Target centroid (T Cα):   {t_cent.round(1)}")
    print(f"  Centroid–centroid dist:   {dist(b_cent, t_cent):.1f} Å")

    # distance from binder centroid to each hotspot residue
    print(f"\n  Distances from binder centroid to hotspot residues:")
    all_found = True
    for ch, rn in hotspots:
        key = (ch, rn)
        if key in target_ca:
            d = dist(b_cent, target_ca[key])
            flag = "  ← FAR" if d > 30 else ""
            print(f"    {ch}{rn}: {d:.1f} Å{flag}")
        else:
            print(f"    {ch}{rn}: NOT FOUND in design PDB")
            all_found = False

    # minimum distance between any binder and any target Cα
    min_d = np.inf
    for bxyz in binder_ca.values():
        for txyz in target_ca.values():
            d = dist(bxyz, txyz)
            if d < min_d:
                min_d = d
    print(f"\n  Minimum Cα–Cα binder–target distance: {min_d:.1f} Å")
    if min_d > 15:
        print("  [WARN] Binder and target are NOT in contact (>15 Å). "
              "This will produce high ipAE regardless of sequence.")
    elif min_d > 8:
        print("  [WARN] Binder–target contact is loose (8–15 Å). "
              "Interface may be weak.")
    else:
        print("  [OK] Binder and target appear to be in contact.")


def check_hotspots_in_ref(ref_path, hotspot_str):
    print(f"\n{'='*60}")
    print(f"Hotspot residue check in reference: {Path(ref_path).name}")
    print(f"{'='*60}")
    ref_ca = parse_ca(ref_path)
    for tok in hotspot_str.split(","):
        tok = tok.strip()
        ch, rn = tok[0], int(tok[1:])
        key = (ch, rn)
        if key in ref_ca:
            print(f"  {ch}{rn}: PRESENT  {ref_ca[key].round(1)}")
        else:
            print(f"  {ch}{rn}: MISSING from reference PDB  ← hotspot conditioning will fail")


def check_rfd_output(design_path, ref_path):
    """
    Compare binder backbone in design vs reference. If RMSD is near zero
    the diffusion didn't move anything — suggests a passthrough bug.
    If it's huge the binder was generated in a completely different location.
    """
    des = parse_ca(design_path, chains={"H", "L"})
    ref = parse_ca(ref_path,    chains={"H", "L"})
    common = sorted(set(des) & set(ref))
    if not common:
        print("\n[WARN] No common binder residues between design and reference.")
        return
    P = np.array([des[k] for k in common])
    Q = np.array([ref[k] for k in common])
    rmsd = float(np.sqrt(((P - Q)**2).sum(axis=1).mean()))
    print(f"\n{'='*60}")
    print(f"Binder backbone RMSD vs reference: {rmsd:.2f} Å  ({len(common)} residues)")
    print(f"{'='*60}")
    if rmsd < 0.5:
        print("  [WARN] RMSD is near zero — design may be identical to reference input.")
    elif rmsd > 50:
        print("  [WARN] RMSD is very large — binder placed far from reference position.")
    else:
        print("  [OK] Binder backbone has moved meaningfully from reference.")


def check_contig_coverage(design_path, ref_path):
    """
    Verify that all T-chain residues in the reference are present in the
    design. Missing target residues indicate a contig gap problem.
    """
    ref_t = parse_ca(ref_path, chains={"T"})
    des_t = parse_ca(design_path, chains={"T"})
    missing = sorted(set(ref_t) - set(des_t))
    extra   = sorted(set(des_t)  - set(ref_t))
    print(f"\n{'='*60}")
    print(f"Target chain coverage check")
    print(f"{'='*60}")
    print(f"  Reference T residues: {len(ref_t)}")
    print(f"  Design    T residues: {len(des_t)}")
    if missing:
        print(f"  [WARN] {len(missing)} ref residues missing from design T chain: "
              f"{[f'T{r}' for _,r in missing[:10]]}{'...' if len(missing)>10 else ''}")
    else:
        print("  [OK] All reference T residues present in design.")
    if extra:
        print(f"  [INFO] {len(extra)} extra residues in design T chain "
              f"(not in reference): {[f'T{r}' for _,r in extra[:10]]}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design",   required=True)
    p.add_argument("--ref",      required=True)
    p.add_argument("--hotspots", required=True,
                   help="e.g. T570,T571,T572,T573")
    args = p.parse_args()

    for path in (args.design, args.ref):
        if not Path(path).exists():
            sys.exit(f"File not found: {path}")

    check_chain_presence(args.design)
    check_chain_presence(args.ref)
    check_hotspots_in_ref(args.ref, args.hotspots)
    check_binder_target_distance(args.design, args.hotspots)
    check_rfd_output(args.design, args.ref)
    check_contig_coverage(args.design, args.ref)


if __name__ == "__main__":
    main()