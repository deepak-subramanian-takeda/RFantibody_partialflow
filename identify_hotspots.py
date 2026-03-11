"""
identify_hotspots.py

Analyses an HLT-formatted antibody-antigen complex PDB and identifies
antigen (chain T) residues that are in contact with the antibody CDR loops,
suitable for use as --hotspots in Step 1 (partial diffusion).

Contact definition
------------------
A target residue is considered a hotspot if any of its heavy atoms are
within --distance Angstroms of any heavy atom on a CDR loop residue
(identified from HLT REMARK PDBinfo-LABEL lines).

Output
------
- Prints the hotspot string ready to paste into the Step 1 command
- Writes a JSON file with per-residue contact details for inspection
- Optionally filters to the top N hotspots by number of contacts

Usage
-----
    python identify_hotspots.py \\
        --input  scripts/examples/example_inputs/1n8z_hlt.pdb \\
        --output 1n8z_hotspots.json \\
        --distance 5.0 \\
        --top_n  10

    # Then use the printed hotspot string in Step 1:
    python partial_diffusion_maturation.py \\
        --input  ... \\
        --hotspots "T305,T456,T512" \\   <- from this script's output
        ...

Dependencies
------------
    Python >= 3.9, numpy (already required by RFantibody environment)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CDR_NAMES = ["H1", "H2", "H3", "L1", "L2", "L3"]

# Heavy atom names to include in distance calculations (excludes hydrogens)
# We use all non-hydrogen atoms by excluding names starting with H
HYDROGEN_RE = re.compile(r"^H")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AtomRecord:
    chain:   str
    resnum:  int
    resname: str
    atomname: str
    xyz:     np.ndarray   # shape (3,)


@dataclass
class HotspotRecord:
    chain:    str
    resnum:   int
    resname:  str
    n_contacts: int                        # number of CDR atoms within cutoff
    contacting_cdrs: List[str]             # which CDR loops make contact
    min_dist: float                        # closest CDR atom distance (Å)
    contact_details: List[Dict]            # per-contact atom pairs


# ---------------------------------------------------------------------------
# 1. PDB parsing
# ---------------------------------------------------------------------------

def parse_atoms(pdb_path: str, chains: List[str]) -> List[AtomRecord]:
    """
    Parse all heavy ATOM/HETATM records for the specified chains.
    Hydrogen atoms are excluded.
    """
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21].strip()
            if chain not in chains:
                continue
            atomname = line[12:16].strip()
            if HYDROGEN_RE.match(atomname):
                continue
            try:
                resnum  = int(line[22:26].strip())
                resname = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue
            atoms.append(AtomRecord(
                chain=chain,
                resnum=resnum,
                resname=resname,
                atomname=atomname,
                xyz=np.array([x, y, z], dtype=np.float32),
            ))
    return atoms


def parse_hlt_remarks(pdb_path: str) -> Dict[str, Set[int]]:
    """
    Parse REMARK PDBinfo-LABEL lines.
    Returns {cdr_name -> set of absolute pose indices}.
    """
    remark_re = re.compile(
        r"^REMARK\s+PDBinfo-LABEL:\s+(\d+)\s+(H[123]|L[123])\s*$"
    )
    cdr_positions: Dict[str, Set[int]] = {n: set() for n in CDR_NAMES}
    with open(pdb_path) as fh:
        for line in fh:
            m = remark_re.match(line.strip())
            if m:
                cdr_positions[m.group(2)].add(int(m.group(1)))
    return cdr_positions


def build_abs_index_map(pdb_path: str) -> Dict[Tuple[str, int], int]:
    """
    Build (chain, resnum) -> absolute 1-based pose index mapping,
    needed to cross-reference REMARK indices with residue identity.
    """
    seen = set()
    idx = 0
    mapping: Dict[Tuple[str, int], int] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21].strip()
            if chain not in ("H", "L", "T"):
                continue
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            key = (chain, resnum)
            if key not in seen:
                seen.add(key)
                idx += 1
                mapping[key] = idx
    return mapping


def get_cdr_residue_keys(
    cdr_positions: Dict[str, Set[int]],
    abs_index_map: Dict[Tuple[str, int], int],
) -> Dict[str, Set[Tuple[str, int]]]:
    """
    Convert CDR absolute indices back to (chain, resnum) sets,
    grouped by CDR name.
    """
    # Invert the abs_index_map
    inv_map: Dict[int, Tuple[str, int]] = {v: k for k, v in abs_index_map.items()}

    cdr_keys: Dict[str, Set[Tuple[str, int]]] = {n: set() for n in CDR_NAMES}
    for cdr_name, abs_idxs in cdr_positions.items():
        for idx in abs_idxs:
            key = inv_map.get(idx)
            if key is not None:
                cdr_keys[cdr_name].add(key)
    return cdr_keys


# ---------------------------------------------------------------------------
# 2. Contact detection
# ---------------------------------------------------------------------------

def find_hotspots(
    pdb_path:    str,
    distance:    float,
    top_n:       Optional[int],
    min_contacts: int,
) -> List[HotspotRecord]:
    """
    Identify target (chain T) residues in contact with CDR loop atoms.

    Parameters
    ----------
    pdb_path     : HLT-formatted PDB
    distance     : contact cutoff in Angstroms
    top_n        : if set, return only the top N hotspots by contact count
    min_contacts : minimum number of CDR atom contacts to be reported

    Returns
    -------
    List of HotspotRecord sorted by n_contacts descending, then min_dist.
    """
    # Parse atoms
    ab_atoms = parse_atoms(pdb_path, ["H", "L"])
    t_atoms  = parse_atoms(pdb_path, ["T"])

    if not ab_atoms:
        sys.exit("[ERROR] No H/L chain atoms found. Is this a valid HLT PDB?")
    if not t_atoms:
        sys.exit("[ERROR] No T chain atoms found. Is this a valid HLT PDB?")

    # Get CDR residue keys
    cdr_positions = parse_hlt_remarks(pdb_path)
    abs_index_map = build_abs_index_map(pdb_path)
    cdr_keys = get_cdr_residue_keys(cdr_positions, abs_index_map)

    # Build set of all CDR atom records
    cdr_atom_set: Set[Tuple[str, int]] = set()
    for keys in cdr_keys.values():
        cdr_atom_set.update(keys)

    # Filter antibody atoms to CDR-only
    cdr_atoms = [
        a for a in ab_atoms
        if (a.chain, a.resnum) in cdr_atom_set
    ]

    if not cdr_atoms:
        sys.exit(
            "[ERROR] No CDR atoms found. Check that the PDB has "
            "REMARK PDBinfo-LABEL lines."
        )

    print(f"[identify_hotspots] CDR atoms: {len(cdr_atoms)}")
    print(f"[identify_hotspots] Target atoms: {len(t_atoms)}")

    # Build CDR xyz matrix for vectorised distance computation
    cdr_xyz  = np.array([a.xyz for a in cdr_atoms],  dtype=np.float32)  # (N_cdr, 3)

    # Which CDR does each cdr_atom belong to?
    cdr_atom_cdr_name: List[str] = []
    for a in cdr_atoms:
        name = "unknown"
        for cdr_name, keys in cdr_keys.items():
            if (a.chain, a.resnum) in keys:
                name = cdr_name
                break
        cdr_atom_cdr_name.append(name)

    # Group target atoms by residue
    t_residue_atoms: Dict[Tuple[str, int, str], List[AtomRecord]] = defaultdict(list)
    for a in t_atoms:
        t_residue_atoms[(a.chain, a.resnum, a.resname)].append(a)

    # Compute contacts
    hotspots: List[HotspotRecord] = []
    dist_sq_cutoff = distance ** 2

    for (chain, resnum, resname), res_atoms in t_residue_atoms.items():
        res_xyz = np.array([a.xyz for a in res_atoms], dtype=np.float32)  # (M, 3)

        # Pairwise distances: (M, N_cdr)
        diff = res_xyz[:, None, :] - cdr_xyz[None, :, :]   # (M, N_cdr, 3)
        dist_sq = np.sum(diff ** 2, axis=-1)                # (M, N_cdr)
        dist    = np.sqrt(dist_sq)

        # Contact mask
        contact_mask = dist_sq <= dist_sq_cutoff            # (M, N_cdr)
        n_contacts   = int(contact_mask.sum())

        if n_contacts < min_contacts:
            continue

        min_dist = float(dist[contact_mask].min())

        # Which CDRs are involved?
        contacting_cdr_set: Set[str] = set()
        contact_details: List[Dict] = []
        t_atom_indices, cdr_atom_indices = np.where(contact_mask)
        for ti, ci in zip(t_atom_indices, cdr_atom_indices):
            cdr_name = cdr_atom_cdr_name[ci]
            contacting_cdr_set.add(cdr_name)
            contact_details.append({
                "target_atom":   res_atoms[ti].atomname,
                "cdr_chain":     cdr_atoms[ci].chain,
                "cdr_resnum":    cdr_atoms[ci].resnum,
                "cdr_resname":   cdr_atoms[ci].resname,
                "cdr_atom":      cdr_atoms[ci].atomname,
                "cdr_loop":      cdr_name,
                "distance_A":    round(float(dist[ti, ci]), 3),
            })

        hotspots.append(HotspotRecord(
            chain=chain,
            resnum=resnum,
            resname=resname,
            n_contacts=n_contacts,
            contacting_cdrs=sorted(contacting_cdr_set),
            min_dist=min_dist,
            contact_details=sorted(contact_details,
                                   key=lambda x: x["distance_A"]),
        ))

    # Sort by contact count descending, then min distance ascending
    hotspots.sort(key=lambda h: (-h.n_contacts, h.min_dist))

    if top_n is not None:
        hotspots = hotspots[:top_n]

    return hotspots


# ---------------------------------------------------------------------------
# 3. Output formatting
# ---------------------------------------------------------------------------

def format_hotspot_string(hotspots: List[HotspotRecord]) -> str:
    """
    Format hotspots as a comma-separated string for --hotspots argument.
    e.g. "T305,T456,T512"
    """
    return ",".join(f"{h.chain}{h.resnum}" for h in hotspots)


def print_summary(hotspots: List[HotspotRecord], distance: float) -> None:
    print(f"\n{'='*65}")
    print(f"  Hotspot Residues  (contact cutoff: {distance:.1f} Å)")
    print(f"{'='*65}")
    print(f"  {'Residue':<10} {'Resname':<8} {'Contacts':<10} "
          f"{'Min dist (Å)':<14} {'CDR loops'}")
    print(f"  {'-'*60}")
    for h in hotspots:
        res_id   = f"{h.chain}{h.resnum}"
        cdr_str  = ",".join(h.contacting_cdrs)
        print(f"  {res_id:<10} {h.resname:<8} {h.n_contacts:<10} "
              f"{h.min_dist:<14.2f} {cdr_str}")
    print(f"{'='*65}")
    print(f"\n  --hotspots argument:")
    print(f'  "{format_hotspot_string(hotspots)}"')
    print()


def write_json(hotspots: List[HotspotRecord], output_path: str) -> None:
    data = {
        "hotspots": [
            {
                "residue":         f"{h.chain}{h.resnum}",
                "chain":           h.chain,
                "resnum":          h.resnum,
                "resname":         h.resname,
                "n_contacts":      h.n_contacts,
                "min_dist_A":      round(h.min_dist, 3),
                "contacting_cdrs": h.contacting_cdrs,
                "contact_details": h.contact_details,
            }
            for h in hotspots
        ],
        "hotspot_string": format_hotspot_string(hotspots),
    }
    with open(output_path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"[identify_hotspots] Detailed contact data written to: {output_path}")


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Identify antigen hotspot residues from an HLT complex PDB "
            "for use as --hotspots in Step 1 (partial diffusion)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="HLT-formatted antibody-antigen complex PDB")
    p.add_argument("--output", default="",
                   help="Path to write JSON output (default: "
                        "<input_stem>_hotspots.json)")
    p.add_argument("--distance", type=float, default=5.0,
                   help="Heavy-atom contact cutoff in Angstroms (default: 5.0). "
                        "4.0–5.0 Å is typical for direct contact; "
                        "6.0–8.0 Å captures near-contact residues.")
    p.add_argument("--top_n", type=int, default=None,
                   help="Return only the top N hotspots by contact count. "
                        "RFantibody recommends 3–10 hotspots. (default: all)")
    p.add_argument("--min_contacts", type=int, default=1,
                   help="Minimum number of CDR atom contacts for a residue "
                        "to be reported (default: 1). Increase to 3–5 to "
                        "focus on strongly contacted residues.")
    p.add_argument("--cdrs", default="",
                   help="Comma-separated CDR loops to consider for contact "
                        "detection, e.g. 'H1,H2,H3' (default: all six loops). "
                        "Useful for nanobodies where L-chain loops are absent.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_pdb = str(Path(args.input).resolve())
    if not os.path.isfile(input_pdb):
        sys.exit(f"[ERROR] Input PDB not found: {input_pdb}")

    output_json = args.output or str(
        Path(input_pdb).parent / (Path(input_pdb).stem + "_hotspots.json")
    )

    print(f"[identify_hotspots] Input  : {input_pdb}")
    print(f"[identify_hotspots] Cutoff : {args.distance:.1f} Å")
    if args.top_n:
        print(f"[identify_hotspots] Top N  : {args.top_n}")
    if args.min_contacts > 1:
        print(f"[identify_hotspots] Min contacts: {args.min_contacts}")

    # Optionally restrict which CDRs are considered
    global CDR_NAMES
    if args.cdrs:
        requested = [c.strip() for c in args.cdrs.split(",")]
        invalid = [c for c in requested if c not in CDR_NAMES]
        if invalid:
            sys.exit(f"[ERROR] Unknown CDR names: {invalid}. "
                     f"Valid: {CDR_NAMES}")
        CDR_NAMES = requested

    hotspots = find_hotspots(
        pdb_path=input_pdb,
        distance=args.distance,
        top_n=args.top_n,
        min_contacts=args.min_contacts,
    )

    if not hotspots:
        print(f"\n[identify_hotspots] No hotspots found at "
              f"{args.distance:.1f} Å cutoff.")
        print("  Try increasing --distance or reducing --min_contacts.")
        sys.exit(0)

    print(f"\n[identify_hotspots] Found {len(hotspots)} hotspot residue(s)")
    print_summary(hotspots, args.distance)
    write_json(hotspots, output_json)


if __name__ == "__main__":
    main()