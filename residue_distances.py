"""
residue_distances.py

For each PDB in a directory, computes the minimum heavy-atom distance
between a specified query residue (e.g. H105) and every residue on the
target chain (T), then reports and optionally filters by distance cutoff.

Useful for verifying that anchor residues remain close to their target
contacts after partial diffusion, or for inspecting how designed sequences
affect interface geometry after RF2 prediction.

Usage
-----
    # Distance from H105 to all T-chain residues across all PDBs:
    python residue_distances.py \\
        --input_dir  1n8z_rf2_inputs/ \\
        --query      H105 \\
        --output     H105_distances.json \\
        --cutoff     8.0

    # Multiple query residues:
    python residue_distances.py \\
        --input_dir  1n8z_rf2_inputs/ \\
        --query      H105 H57 L214 \\
        --cutoff     6.0 \\
        --output     anchor_distances.json

    # Single PDB:
    python residue_distances.py \\
        --input_dir  1n8z_rf2_inputs/ \\
        --pdb        1n8z_hlt_partial_T15_0_sample1.pdb \\
        --query      H105 \\
        --cutoff     8.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AtomRecord:
    chain:    str
    resnum:   int
    resname:  str
    atomname: str
    xyz:      np.ndarray


@dataclass
class ResidueDistance:
    target_chain:   str
    target_resnum:  int
    target_resname: str
    min_dist:       float     # minimum heavy-atom distance to query residue
    query_atom:     str       # query atom involved in closest contact
    target_atom:    str       # target atom involved in closest contact


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def parse_atoms(pdb_path: str) -> List[AtomRecord]:
    """Parse all heavy ATOM/HETATM records, excluding hydrogens."""
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atomname = line[12:16].strip()
            if atomname.startswith("H"):
                continue
            try:
                chain   = line[21].strip()
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


def parse_query(query_str: str) -> Tuple[str, int]:
    """
    Parse a query residue string like 'H105' into (chain, resnum).
    Accepts formats: H105, H:105, H 105.
    """
    import re
    m = re.match(r"^([A-Za-z]+):?(\d+)$", query_str.strip())
    if not m:
        sys.exit(f"[ERROR] Cannot parse query residue '{query_str}'. "
                 "Expected format: H105 or H:105")
    return m.group(1).upper(), int(m.group(2))


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def compute_distances(
    atoms:        List[AtomRecord],
    query_chain:  str,
    query_resnum: int,
    target_chain: str,
    cutoff:       Optional[float],
) -> List[ResidueDistance]:
    """
    Compute minimum heavy-atom distance from the query residue to every
    residue on target_chain.

    Returns a list of ResidueDistance sorted by min_dist ascending,
    optionally filtered to those within cutoff Angstroms.
    """
    # Gather query atoms
    query_atoms = [
        a for a in atoms
        if a.chain == query_chain and a.resnum == query_resnum
    ]
    if not query_atoms:
        return []

    # Group target atoms by (resnum, resname)
    target_residues: Dict[Tuple[int, str], List[AtomRecord]] = defaultdict(list)
    for a in atoms:
        if a.chain == target_chain:
            target_residues[(a.resnum, a.resname)].append(a)

    if not target_residues:
        return []

    query_xyz   = np.array([a.xyz for a in query_atoms],  dtype=np.float32)
    query_names = [a.atomname for a in query_atoms]

    results: List[ResidueDistance] = []

    for (resnum, resname), res_atoms in target_residues.items():
        target_xyz   = np.array([a.xyz for a in res_atoms], dtype=np.float32)
        target_names = [a.atomname for a in res_atoms]

        # Pairwise distances (n_query, n_target)
        diff    = query_xyz[:, None, :] - target_xyz[None, :, :]
        dists   = np.sqrt(np.sum(diff ** 2, axis=-1))
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_d   = float(dists[min_idx])

        if cutoff is not None and min_d > cutoff:
            continue

        results.append(ResidueDistance(
            target_chain=target_chain,
            target_resnum=resnum,
            target_resname=resname,
            min_dist=round(min_d, 3),
            query_atom=query_names[min_idx[0]],
            target_atom=target_names[min_idx[1]],
        ))

    results.sort(key=lambda r: r.min_dist)
    return results


# ---------------------------------------------------------------------------
# Per-PDB processing
# ---------------------------------------------------------------------------

def process_pdb(
    pdb_path:      str,
    queries:       List[Tuple[str, int]],
    target_chain:  str,
    cutoff:        Optional[float],
) -> Dict:
    """
    Process one PDB file and return a result dict.
    """
    atoms = parse_atoms(pdb_path)
    pdb_name = Path(pdb_path).stem

    result = {"pdb": pdb_name, "queries": {}}

    for query_chain, query_resnum in queries:
        query_id = f"{query_chain}{query_resnum}"

        # Check query residue exists
        query_atoms = [
            a for a in atoms
            if a.chain == query_chain and a.resnum == query_resnum
        ]
        if not query_atoms:
            result["queries"][query_id] = {
                "error": f"Residue {query_id} not found in PDB"
            }
            continue

        query_resname = query_atoms[0].resname
        dists = compute_distances(
            atoms=atoms,
            query_chain=query_chain,
            query_resnum=query_resnum,
            target_chain=target_chain,
            cutoff=cutoff,
        )

        result["queries"][query_id] = {
            "query_resname": query_resname,
            "target_chain":  target_chain,
            "cutoff_A":      cutoff,
            "n_contacts":    len(dists),
            "contacts": [
                {
                    "target_residue": f"{d.target_chain}{d.target_resnum}",
                    "target_resname": d.target_resname,
                    "min_dist_A":     d.min_dist,
                    "query_atom":     d.query_atom,
                    "target_atom":    d.target_atom,
                }
                for d in dists
            ],
        }

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(all_results: List[Dict], queries: List[Tuple[str, int]]) -> None:
    for res in all_results:
        print(f"\n{'='*60}")
        print(f"  PDB: {res['pdb']}")
        print(f"{'='*60}")
        for query_chain, query_resnum in queries:
            query_id = f"{query_chain}{query_resnum}"
            q = res["queries"].get(query_id, {})

            if "error" in q:
                print(f"  {query_id}: {q['error']}")
                continue

            print(f"\n  Query: {query_id} ({q['query_resname']})  "
                  f"-> chain {q['target_chain']}  "
                  f"cutoff={q['cutoff_A']}Å  "
                  f"contacts={q['n_contacts']}")
            print(f"  {'Target':<12} {'Resname':<8} {'Min dist (Å)':<14} "
                  f"{'Query atom':<12} {'Target atom'}")
            print(f"  {'-'*55}")
            for c in q["contacts"]:
                print(f"  {c['target_residue']:<12} "
                      f"{c['target_resname']:<8} "
                      f"{c['min_dist_A']:<14.3f} "
                      f"{c['query_atom']:<12} "
                      f"{c['target_atom']}")
            if not q["contacts"]:
                print(f"  (no contacts within cutoff)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute distances from query antibody residue(s) to all "
            "target chain residues across a directory of PDB files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_dir", required=True,
                   help="Directory containing PDB files to analyse")
    p.add_argument("--query", nargs="+", required=True,
                   help="Query residue(s), e.g. H105 or H105 H57 L214")
    p.add_argument("--output", default="",
                   help="Path to write JSON output (default: "
                        "<input_dir>/residue_distances.json)")
    p.add_argument("--cutoff", type=float, default=None,
                   help="Only report target residues within this distance "
                        "in Angstroms (default: report all)")
    p.add_argument("--target_chain", default="T",
                   help="Chain letter of the antigen/target (default: T)")
    p.add_argument("--pdb", default="",
                   help="If set, analyse only this specific PDB filename "
                        "within --input_dir rather than all PDBs")
    p.add_argument("--sort_by", choices=["dist", "resnum"], default="dist",
                   help="Sort contacts by distance or residue number "
                        "(default: dist)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = str(Path(args.input_dir).resolve())
    if not os.path.isdir(input_dir):
        sys.exit(f"[ERROR] Directory not found: {input_dir}")

    # Collect PDB files
    if args.pdb:
        pdb_files = [Path(input_dir) / args.pdb]
        if not pdb_files[0].exists():
            sys.exit(f"[ERROR] PDB not found: {pdb_files[0]}")
    else:
        pdb_files = sorted(Path(input_dir).glob("*.pdb"))
        if not pdb_files:
            sys.exit(f"[ERROR] No .pdb files found in {input_dir}")

    # Parse query residues
    queries = [parse_query(q) for q in args.query]
    query_ids = [f"{c}{r}" for c, r in queries]

    output_path = args.output or os.path.join(input_dir, "residue_distances.json")

    print(f"[residue_distances] PDBs       : {len(pdb_files)}")
    print(f"[residue_distances] Queries    : {', '.join(query_ids)}")
    print(f"[residue_distances] Target     : chain {args.target_chain}")
    if args.cutoff:
        print(f"[residue_distances] Cutoff     : {args.cutoff:.1f} Å")

    all_results = []
    for pdb_path in pdb_files:
        res = process_pdb(
            pdb_path=str(pdb_path),
            queries=queries,
            target_chain=args.target_chain,
            cutoff=args.cutoff,
        )
        all_results.append(res)

    # Optionally re-sort by resnum
    if args.sort_by == "resnum":
        for res in all_results:
            for q in res["queries"].values():
                if "contacts" in q:
                    q["contacts"].sort(
                        key=lambda c: int(c["target_residue"][1:])
                    )

    print_results(all_results, queries)

    with open(output_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n[residue_distances] Results written to: {output_path}")


if __name__ == "__main__":
    main()