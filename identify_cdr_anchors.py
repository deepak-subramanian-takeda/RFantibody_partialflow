"""

Given a filtered HLT-format antibody-antigen complex PDB, this script:
  1. Parses CDR residue ranges from HLT REMARK annotations
  2. Runs FastRelax (optional but recommended) to resolve clashes before scoring
  3. Scores per-residue interface energies with PyRosetta InterfaceAnalyzerMover
  4. Identifies CDR residues below an energy threshold as "anchors"
  5. Outputs:
       - A JSON file mapping CDR loop -> list of anchor PDB residue IDs
       - A fixed_positions.jsonl compatible with ProteinMPNN
       - A human-readable summary TSV

Usage:
    python identify_cdr_anchors.py \
        --input design_0001.pdb \
        --output_dir anchors/ \
        --energy_threshold -5.0 \
        --relax \
        --interface HLT          # "HLT" = HL-side vs T-side

Dependencies:
    pyrosetta (install via: pip install pyrosetta-installer && python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()")
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CDR_NAMES = ["H1", "H2", "H3", "L1", "L2", "L3"]

@dataclass
class CdrRange:
    """1-indexed absolute residue positions as stored in HLT REMARK lines."""
    name: str
    start: int  # 1-indexed, absolute (not per-chain)
    end: int

@dataclass
class AnchorResidue:
    """A single CDR residue identified as an energetic anchor."""
    cdr: str
    pose_idx: int          # Rosetta 1-indexed pose number
    pdb_chain: str         # e.g. 'H'
    pdb_resnum: int        # PDB residue number
    resname: str           # three-letter amino acid code
    interface_energy: float


# ---------------------------------------------------------------------------
# 1. HLT file parsing
# ---------------------------------------------------------------------------

def parse_hlt_remarks(pdb_path: str) -> Dict[str, CdrRange]:
    """
    Read REMARK PDBinfo-LABEL lines from an HLT-format PDB file.

    RFantibody writes remarks like:
        REMARK PDBinfo-LABEL:   32 H1
        REMARK PDBinfo-LABEL:   39 H1        <- last H1 residue also labelled
        REMARK PDBinfo-LABEL:   52 H2
        ...

    Each labelled residue carries its CDR name. We collect the min and max
    absolute residue index for each CDR to get the full range.

    Returns
    -------
    dict mapping CDR name (e.g. "H1") -> CdrRange
    """
    # Maps cdr_name -> set of absolute 1-indexed positions
    cdr_positions: Dict[str, List[int]] = {n: [] for n in CDR_NAMES}

    remark_re = re.compile(
        r"^REMARK\s+PDBinfo-LABEL:\s+(\d+)\s+(H[123]|L[123])\s*$"
    )

    with open(pdb_path) as fh:
        for line in fh:
            m = remark_re.match(line.strip())
            if m:
                abs_idx = int(m.group(1))
                cdr_name = m.group(2)
                if cdr_name in cdr_positions:
                    cdr_positions[cdr_name].append(abs_idx)

    ranges: Dict[str, CdrRange] = {}
    for name, positions in cdr_positions.items():
        if positions:
            ranges[name] = CdrRange(
                name=name,
                start=min(positions),
                end=max(positions),
            )

    missing = [n for n in CDR_NAMES if n not in ranges]
    if missing:
        # Nanobody designs will only have H1/H2/H3 — L loops are absent
        print(f"[INFO] CDR loops not found in REMARK lines: {missing} "
              "(expected for nanobody designs)")

    if not ranges:
        raise ValueError(
            f"No CDR REMARK lines found in {pdb_path}. "
            "Ensure this is a valid HLT-annotated file from RFantibody."
        )

    return ranges


def get_pose_chain_residue(pose, pose_idx: int) -> Tuple[str, int]:
    """Return (pdb_chain, pdb_resnum) for a given Rosetta pose index."""
    pdb_info = pose.pdb_info()
    chain = pdb_info.chain(pose_idx)
    resnum = pdb_info.number(pose_idx)
    return chain, resnum


# ---------------------------------------------------------------------------
# 2. Optional relax before scoring
# ---------------------------------------------------------------------------

def fast_relax_interface(pose, sfxn, interface: str = "HL_T"):
    """
    Run a single round of FastRelax constrained to interface residues only.
    This resolves clashes introduced by RFdiffusion without moving the whole
    structure. Operates on a clone so the original pose is unchanged.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector, OrResidueSelector, NeighborhoodResidueSelector,
        InterGroupInterfaceByVectorSelector,
    )
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.kinematics import MoveMap

    relaxed = pose.clone()

    # Select interface residues from both sides
    sides = interface.split("_")
    ab_chains  = list(sides[0])  # e.g. ['H', 'L']
    ag_chains  = list(sides[1])  # e.g. ['T']

    def chain_sel(chains):
        if len(chains) == 1:
            return ChainSelector(chains[0])
        sel = OrResidueSelector()
        for ch in chains:
            sel.add_residue_selector(ChainSelector(ch))
        return sel

    interface_sel = InterGroupInterfaceByVectorSelector()
    interface_sel.group1_selector(chain_sel(ab_chains))
    interface_sel.group2_selector(chain_sel(ag_chains))

    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    subset = interface_sel.apply(relaxed)
    for i in range(1, relaxed.size() + 1):
        if subset[i]:
            mm.set_chi(i, True)
            mm.set_bb(i, True)

    fr = FastRelax(sfxn, 1)   # 1 cycle is enough for clash removal
    fr.set_movemap(mm)
    fr.apply(relaxed)
    return relaxed


# ---------------------------------------------------------------------------
# 3. Per-residue interface energy via InterfaceAnalyzerMover
# ---------------------------------------------------------------------------

def score_per_residue_interface(
    pose,
    sfxn,
    interface: str,
    pack_input: bool = True,
    pack_separated: bool = True,
) -> Dict[int, float]:
    """
    Run InterfaceAnalyzerMover and return a dict of:
        { pose_residue_index -> weighted_total_energy_delta }

    The 'weighted_energy' values come from get_all_per_residue_data(), which
    stores the difference in residue energy between the complexed and
    separated states — i.e., the contribution of each residue to binding.

    Parameters
    ----------
    interface : str
        Interface string in Rosetta convention, e.g. "HL_T" for chains
        H+L vs T.  For nanobodies use "H_T".
    """
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    ia = InterfaceAnalyzerMover(interface)
    ia.set_pack_input(pack_input)
    ia.set_pack_separated(pack_separated)
    ia.set_scorefunction(sfxn)
    ia.set_use_tracer(False)
    ia.apply(pose)

    per_res_data = ia.get_all_per_residue_data()

    # per_res_data.weighted_total is a vector1<Real> indexed 1..nres
    # It holds the per-residue dG contribution (complexed - separated)
    energies: Dict[int, float] = {}
    wt = per_res_data.weighted_total
    for i in range(1, pose.size() + 1):
        energies[i] = wt[i]

    return energies


# ---------------------------------------------------------------------------
# 4. Main anchor identification logic
# ---------------------------------------------------------------------------

def identify_anchors(
    pdb_path: str,
    output_dir: str,
    energy_threshold: float = -5.0,
    interface: str = "HL_T",
    do_relax: bool = False,
    relax_sfxn_name: str = "ref2015",
) -> List[AnchorResidue]:
    """
    Full pipeline: parse HLT -> (optionally relax) -> score -> filter anchors.

    Returns list of AnchorResidue objects below the energy threshold that
    fall within a CDR loop.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import get_score_function

    # ---- Init PyRosetta (quiet) -------------------------------------------
    pyrosetta.init(
        "-mute all "
        "-ignore_unrecognized_res true "
        "-ignore_zero_occupancy false",
        silent=True,
    )

    sfxn = get_score_function(True)   # ref2015 by default

    # ---- Load pose --------------------------------------------------------
    print(f"[Step 0] Loading pose from {pdb_path}")
    pose = pyrosetta.pose_from_pdb(pdb_path)
    print(f"         Pose has {pose.size()} residues, "
          f"{pose.num_chains()} chains")

    # ---- Parse CDR ranges from HLT REMARK lines --------------------------
    print("[Step 0] Parsing CDR ranges from HLT REMARK annotations...")
    cdr_ranges = parse_hlt_remarks(pdb_path)
    for name, r in sorted(cdr_ranges.items()):
        print(f"         {name}: absolute residues {r.start}–{r.end} "
              f"({r.end - r.start + 1} residues)")

    # Build a fast lookup: pose_idx -> CDR name (for CDR residues only)
    cdr_lookup: Dict[int, str] = {}
    for cdr_name, r in cdr_ranges.items():
        for idx in range(r.start, r.end + 1):
            cdr_lookup[idx] = cdr_name

    # ---- Optional relax ---------------------------------------------------
    if do_relax:
        print("[Step 0] Running FastRelax on interface residues...")
        pose = fast_relax_interface(pose, sfxn, interface)
        relaxed_path = os.path.join(
            output_dir,
            Path(pdb_path).stem + "_relaxed.pdb"
        )
        pose.dump_pdb(relaxed_path)
        print(f"         Relaxed structure saved to {relaxed_path}")
    else:
        print("[Step 0] Skipping relax (use --relax to enable)")

    # ---- Score interface --------------------------------------------------
    print(f"[Step 0] Scoring per-residue interface energies "
          f"(interface={interface})...")
    energies = score_per_residue_interface(pose, sfxn, interface)

    # ---- Filter CDR anchors ----------------------------------------------
    print(f"[Step 0] Identifying CDR anchors below {energy_threshold} REU...")
    anchors: List[AnchorResidue] = []

    for pose_idx, energy in energies.items():
        if pose_idx not in cdr_lookup:
            continue   # skip framework and target residues
        if energy > energy_threshold:
            continue   # not energetically favourable enough

        cdr_name = cdr_lookup[pose_idx]
        chain, resnum = get_pose_chain_residue(pose, pose_idx)
        resname = pose.residue(pose_idx).name3().strip()

        anchors.append(AnchorResidue(
            cdr=cdr_name,
            pose_idx=pose_idx,
            pdb_chain=chain,
            pdb_resnum=resnum,
            resname=resname,
            interface_energy=energy,
        ))

    anchors.sort(key=lambda a: a.interface_energy)   # most favourable first

    if anchors:
        print(f"         Found {len(anchors)} anchor residue(s):")
        for a in anchors:
            print(f"           {a.cdr:3s}  {a.pdb_chain}{a.pdb_resnum:4d} "
                  f"{a.resname}  dG={a.interface_energy:+.2f} REU")
    else:
        print("         No anchor residues found below threshold. "
              "Consider loosening --energy_threshold.")

    return anchors


# ---------------------------------------------------------------------------
# 5. Output writers
# ---------------------------------------------------------------------------

def write_outputs(
    anchors: List[AnchorResidue],
    all_cdr_ranges: Dict[str, CdrRange],
    pose,  # needed to get all CDR residue IDs for fixed_positions
    pdb_path: str,
    output_dir: str,
    energy_threshold: float,
):
    """
    Write three output files into output_dir:

    1. anchors.json          — machine-readable anchor residue data
    2. fixed_positions.jsonl — ProteinMPNN fixed_positions_jsonl format
    3. anchors_summary.tsv   — human-readable table for inspection
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(pdb_path).stem

    # --- 1. anchors.json --------------------------------------------------
    anchor_json = {
        "source_pdb": str(pdb_path),
        "energy_threshold_reu": energy_threshold,
        "anchors_by_cdr": {},
        "all_anchor_residues": [],
    }
    for a in anchors:
        anchor_json["anchors_by_cdr"].setdefault(a.cdr, []).append(
            {"chain": a.pdb_chain, "resnum": a.pdb_resnum,
             "resname": a.resname, "dG_reu": round(a.interface_energy, 3)}
        )
        anchor_json["all_anchor_residues"].append(
            f"{a.pdb_chain}{a.pdb_resnum}"
        )

    json_path = os.path.join(output_dir, f"{stem}_anchors.json")
    with open(json_path, "w") as fh:
        json.dump(anchor_json, fh, indent=2)
    print(f"[Step 0] Written: {json_path}")

    # --- 2. fixed_positions.jsonl -----------------------------------------
    # ProteinMPNN expects a JSON-lines file where each line is:
    #   { "pdb_name": { "chain_id": [list_of_residue_numbers_to_fix] } }
    #
    # Anchor residues should be FIXED (sequence preserved).
    # All other CDR residues remain designable.
    fixed: Dict[str, List[int]] = {}
    for a in anchors:
        fixed.setdefault(a.pdb_chain, []).append(a.pdb_resnum)

    fixed_positions_record = {stem: fixed}
    jsonl_path = os.path.join(output_dir, f"{stem}_fixed_positions.jsonl")
    with open(jsonl_path, "w") as fh:
        fh.write(json.dumps(fixed_positions_record) + "\n")
    print(f"[Step 0] Written: {jsonl_path}")

    # --- 3. anchors_summary.tsv -------------------------------------------
    tsv_path = os.path.join(output_dir, f"{stem}_anchors_summary.tsv")
    with open(tsv_path, "w") as fh:
        fh.write("cdr\tchain\tresnum\tresname\tpose_idx\tdG_reu\tis_anchor\n")
        # Write all CDR residues with their energies, flagging anchors
        anchor_set = {(a.pdb_chain, a.pdb_resnum) for a in anchors}
        # Collect all CDR pose indices in order
        for cdr_name, r in sorted(all_cdr_ranges.items()):
            for pose_idx in range(r.start, r.end + 1):
                try:
                    chain, resnum = get_pose_chain_residue(pose, pose_idx)
                    resname = pose.residue(pose_idx).name3().strip()
                    # Energy may be 0 for non-interface residues
                    from_energies = _energy_cache.get(pose_idx, 0.0)
                    is_anchor = (chain, resnum) in anchor_set
                    fh.write(
                        f"{cdr_name}\t{chain}\t{resnum}\t{resname}\t"
                        f"{pose_idx}\t{from_energies:.3f}\t{is_anchor}\n"
                    )
                except Exception:
                    pass   # pose_idx out of range for nanobodies etc.
    print(f"[Step 0] Written: {tsv_path}")


# Module-level cache so write_outputs can access energies without re-running
_energy_cache: Dict[int, float] = {}


# ---------------------------------------------------------------------------
# 6. CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Identify energetic CDR anchor residues from an HLT PDB."
    )
    p.add_argument("--input", required=True,
                   help="Path to HLT-annotated antibody-antigen complex PDB")
    p.add_argument("--output_dir", default="anchors",
                   help="Directory to write output files (default: anchors/)")
    p.add_argument("--energy_threshold", type=float, default=-5.0,
                   help="Per-residue interface energy cutoff in REU "
                        "(default: -5.0; more negative = stricter)")
    p.add_argument("--interface", default="HL_T",
                   help="Rosetta interface string. Use 'HL_T' for full "
                        "antibody, 'H_T' for nanobody (default: HL_T)")
    p.add_argument("--relax", action="store_true",
                   help="Run FastRelax on interface residues before scoring "
                        "(recommended for fresh RFdiffusion outputs)")
    p.add_argument("--pack_input", action="store_true", default=True,
                   help="Repack input side chains before analysis "
                        "(default: True)")
    p.add_argument("--no_pack_input", dest="pack_input",
                   action="store_false")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run anchor identification
    anchors = identify_anchors(
        pdb_path=args.input,
        output_dir=args.output_dir,
        energy_threshold=args.energy_threshold,
        interface=args.interface,
        do_relax=args.relax,
    )

    # Re-score to populate the energy cache for the TSV writer
    # (anchors already computed; this just fills _energy_cache)
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import get_score_function

    pose = pyrosetta.pose_from_pdb(args.input)
    sfxn = get_score_function(True)
    global _energy_cache
    _energy_cache = score_per_residue_interface(
        pose, sfxn, args.interface,
        pack_input=args.pack_input,
        pack_separated=True,
    )

    cdr_ranges = parse_hlt_remarks(args.input)

    write_outputs(
        anchors=anchors,
        all_cdr_ranges=cdr_ranges,
        pose=pose,
        pdb_path=args.input,
        output_dir=args.output_dir,
        energy_threshold=args.energy_threshold,
    )

    print(f"\n[Step 0] Complete. {len(anchors)} anchor residue(s) identified.")
    print(f"         Outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()