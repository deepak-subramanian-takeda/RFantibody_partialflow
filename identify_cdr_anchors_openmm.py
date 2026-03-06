"""
identify_cdr_anchors.py

Step 0 of the PPIFlow-style maturation workflow for RFantibody.

Given a filtered HLT-format antibody-antigen complex PDB, this script:
  1. Prepares the structure (pdbfixer: missing atoms, hydrogens)
  2. Energy-minimises the prepared structure (OpenMM, Amber ff14SB)
  3. Scores per-residue interface energies using the OpenMM scorer
     (side-chain non-bonded interactions across the antibody/antigen interface)
  4. Identifies CDR residues below an energy threshold as "anchors"
  5. Outputs:
       - A JSON file mapping CDR loop -> list of anchor PDB residue IDs
       - A fixed_positions.jsonl compatible with ProteinMPNN
       - A human-readable summary TSV

Usage:
    python identify_cdr_anchors.py \\
        --input  design_0001.pdb \\
        --output_dir anchors/ \\
        --energy_threshold -50.0 \\
        --source_chains "HL" \\
        --target_chains "T" \\
        --interface_distance 4.0

    # Nanobody (H-chain only vs T):
    python identify_cdr_anchors.py \\
        --input  nb_design_0001.pdb \\
        --output_dir anchors/ \\
        --source_chains "H" \\
        --target_chains "T" \\
        --nanobody

Dependencies:
    openmm, pdbfixer, numpy
    (conda install -c conda-forge openmm pdbfixer numpy)

THRESHOLD NOTE
--------------
The OpenMM scorer returns inter-chain non-bonded energies in kJ/mol,
not Rosetta Energy Units. A reasonable starting threshold is around
-50 kJ/mol (roughly equivalent to -5 REU in magnitude), but this should
be tuned empirically against your target system. Use --dry_run_threshold
to print the full per-residue energy distribution before committing to
a cutoff.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dependency check — give a clear message if OpenMM is missing
# ---------------------------------------------------------------------------

try:
    import numpy as np
except ImportError:
    sys.exit("[ERROR] numpy is required: conda install numpy")

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from pdbfixer import PDBFixer
    _OPENMM_AVAILABLE = True
except ImportError:
    _OPENMM_AVAILABLE = False

# Import the scoring functions from the provided module.
# score_residues.py must be on sys.path or co-located with this file.
#
# OpenMM 8.2+ compatibility note
# --------------------------------
# score_residues.py uses the old-style `getPositions=True` / `getEnergy=True`
# keyword arguments on Context.getState() and the serialize/deserialize round-
# trip in _clone_force().  Both still work in OpenMM 8.4 (the old kwargs are
# deprecated but not yet removed), however the patched versions below replace
# them with the preferred modern equivalents to suppress DeprecationWarnings
# and future-proof the code:
#
#   getState(getPositions=True)  →  getState(positions=True)    (8.2+)
#   getState(getEnergy=True)     →  getState(energy=True)       (8.2+)
#   XmlSerializer.serialize/deserialize  →  XmlSerializer.clone()  (7.7+)
#
# Python 3.13 compatibility note
# --------------------------------
# No changes required: all stdlib modules used here (argparse, collections,
# dataclasses, json, os, pathlib, re, statistics, sys, tempfile, typing)
# are unaffected by the Python 3.13 removals (PEP 594 dead batteries) or
# any other breaking changes in that release.
try:
    from score_residues import prepare_structure, minimize_structure
    import score_residues as _score_residues_mod
    _SCORER_AVAILABLE = True
except ImportError:
    _SCORER_AVAILABLE = False


def score_residues(
    input_pdb: str,
    output_tsv: Optional[str] = None,
    forcefield: str = "amber14-all",
    source_chains: Optional[str] = None,
    target_chains: Optional[str] = None,
    interface_only: bool = True,
    interface_distance: float = 4.0,
) -> str:
    """
    Thin wrapper around score_residues.score_residues() that patches the two
    OpenMM 8.2+ deprecations before delegating to the original function.

    Patches applied
    ---------------
    1. _attribute_bond_energy, _attribute_angle_energy, _attribute_torsion_energy:
       context.getState(getPositions=True) → context.getState(positions=True)
    2. _clone_force:
       XmlSerializer.serialize/deserialize → XmlSerializer.clone()
    """
    if not _SCORER_AVAILABLE:
        raise RuntimeError("score_residues module could not be imported.")

    import openmm
    import types

    # ── Patch 1: getState keyword arguments ──────────────────────────────
    # Replace every helper that calls context.getState(getPositions=True).
    # We monkey-patch at function level so the original module file is
    # never touched on disk.

    _orig_bond    = _score_residues_mod._attribute_bond_energy
    _orig_angle   = _score_residues_mod._attribute_angle_energy
    _orig_torsion = _score_residues_mod._attribute_torsion_energy

    def _bond_patched(force, context, atom_to_residue, residue_energies, skipped):
        state = context.getState(positions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(
            _score_residues_mod.unit.nanometer
        )
        for i in range(force.getNumBonds()):
            p1, p2, length, k = force.getBondParameters(i)
            res1 = atom_to_residue.get(p1)
            res2 = atom_to_residue.get(p2)
            if res1 in skipped and res2 in skipped:
                continue
            r = np.linalg.norm(positions[p1] - positions[p2])
            r0 = length.value_in_unit(_score_residues_mod.unit.nanometer)
            k_val = k.value_in_unit(
                _score_residues_mod.unit.kilojoules_per_mole
                / _score_residues_mod.unit.nanometer**2
            )
            energy = 0.5 * k_val * (r - r0) ** 2
            if res1 not in skipped and res2 not in skipped:
                if res1 == res2:
                    residue_energies[res1] += energy
                else:
                    residue_energies[res1] += energy / 2
                    residue_energies[res2] += energy / 2
            elif res1 not in skipped:
                residue_energies[res1] += energy
            elif res2 not in skipped:
                residue_energies[res2] += energy

    def _angle_patched(force, context, atom_to_residue, residue_energies, skipped):
        state = context.getState(positions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(
            _score_residues_mod.unit.nanometer
        )
        for i in range(force.getNumAngles()):
            p1, p2, p3, angle0, k = force.getAngleParameters(i)
            res_central = atom_to_residue.get(p2)
            if res_central in skipped:
                continue
            v1 = positions[p1] - positions[p2]
            v2 = positions[p3] - positions[p2]
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angle0_val = angle0.value_in_unit(_score_residues_mod.unit.radian)
            k_val = k.value_in_unit(
                _score_residues_mod.unit.kilojoules_per_mole
                / _score_residues_mod.unit.radian**2
            )
            residue_energies[res_central] += 0.5 * k_val * (angle - angle0_val) ** 2

    def _torsion_patched(force, context, atom_to_residue, residue_energies, skipped):
        state = context.getState(positions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(
            _score_residues_mod.unit.nanometer
        )
        for i in range(force.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
            res2 = atom_to_residue.get(p2)
            res3 = atom_to_residue.get(p3)
            if res2 in skipped and res3 in skipped:
                continue
            b1 = positions[p2] - positions[p1]
            b2 = positions[p3] - positions[p2]
            b3 = positions[p4] - positions[p3]
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            if n1_norm < 1e-10 or n2_norm < 1e-10:
                continue
            n1 = n1 / n1_norm
            n2 = n2 / n2_norm
            m1 = np.cross(n1, b2 / np.linalg.norm(b2))
            x = np.dot(n1, n2)
            y = np.dot(m1, n2)
            dihedral = np.arctan2(y, x)
            phase_val = phase.value_in_unit(_score_residues_mod.unit.radian)
            k_val = k.value_in_unit(_score_residues_mod.unit.kilojoules_per_mole)
            energy = k_val * (1 + np.cos(periodicity * dihedral - phase_val))
            if res2 not in skipped and res3 not in skipped:
                if res2 == res3:
                    residue_energies[res2] += energy
                else:
                    residue_energies[res2] += energy / 2
                    residue_energies[res3] += energy / 2
            elif res2 not in skipped:
                residue_energies[res2] += energy
            elif res3 not in skipped:
                residue_energies[res3] += energy

    # ── Patch 2: _clone_force — use XmlSerializer.clone() ────────────────
    def _clone_force_patched(force):
        try:
            return openmm.XmlSerializer.clone(force)
        except Exception:
            return None

    # Apply patches
    _score_residues_mod._attribute_bond_energy    = _bond_patched
    _score_residues_mod._attribute_angle_energy   = _angle_patched
    _score_residues_mod._attribute_torsion_energy = _torsion_patched
    _score_residues_mod._clone_force              = _clone_force_patched

    try:
        result = _score_residues_mod.score_residues(
            input_pdb=input_pdb,
            output_tsv=output_tsv,
            forcefield=forcefield,
            source_chains=source_chains,
            target_chains=target_chains,
            interface_only=interface_only,
            interface_distance=interface_distance,
        )
    finally:
        # Restore originals so other callers of the module are unaffected
        _score_residues_mod._attribute_bond_energy    = _orig_bond
        _score_residues_mod._attribute_angle_energy   = _orig_angle
        _score_residues_mod._attribute_torsion_energy = _orig_torsion
        _score_residues_mod._clone_force              = _score_residues_mod._clone_force

    return result

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CDR_NAMES = ["H1", "H2", "H3", "L1", "L2", "L3"]

@dataclass
class CdrRange:
    """1-indexed absolute residue positions as stored in HLT REMARK lines."""
    name:  str
    start: int   # 1-indexed, absolute (not per-chain)
    end:   int

@dataclass
class AnchorResidue:
    """A single CDR residue identified as an energetic anchor."""
    cdr:              str
    pdb_chain:        str    # e.g. 'H'
    pdb_resnum:       str    # PDB residue number (kept as str to preserve insertion codes)
    resname:          str    # three-letter amino acid code
    interface_energy: float  # kJ/mol (OpenMM inter-chain non-bonded)

# ---------------------------------------------------------------------------
# 1. HLT REMARK parsing
# ---------------------------------------------------------------------------

def parse_hlt_remarks(pdb_path: str) -> Dict[str, CdrRange]:
    """
    Read REMARK PDBinfo-LABEL lines from an HLT-format PDB and return
    a dict mapping CDR name -> CdrRange (with 1-indexed absolute positions).
    """
    remark_re = re.compile(
        r"^REMARK\s+PDBinfo-LABEL:\s+(\d+)\s+(H[123]|L[123])\s*$"
    )
    cdr_positions: Dict[str, List[int]] = {n: [] for n in CDR_NAMES}

    with open(pdb_path) as fh:
        for line in fh:
            m = remark_re.match(line.strip())
            if m:
                abs_idx  = int(m.group(1))
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
        print(f"[INFO] CDR loops not found in REMARK lines (expected for nanobodies): {missing}")

    if not ranges:
        raise ValueError(
            f"No CDR REMARK lines found in {pdb_path}. "
            "Ensure this is a valid HLT-annotated file from RFantibody."
        )
    return ranges


# ---------------------------------------------------------------------------
# 2. Build a pose-index -> (chain, resnum, resname) map from the PDB
# ---------------------------------------------------------------------------

def read_pdb_residue_map(pdb_path: str) -> Dict[int, Tuple[str, str, str]]:
    """
    Walk ATOM records and return a dict mapping 1-indexed absolute pose index
    (counting unique chain+resnum pairs in file order) to (chain, resnum, resname).

    resnum is kept as a string to preserve insertion codes (e.g. '100A').
    """
    seen: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    ordered: List[Tuple[str, str]] = []   # preserves encounter order

    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain  = line[21].strip()
            resnum = line[22:27].strip()   # includes insertion code
            resname = line[17:20].strip()
            key = (chain, resnum)
            if key not in seen:
                seen[key] = (chain, resnum, resname)
                ordered.append(key)

    return {i + 1: seen[k] for i, k in enumerate(ordered)}


# ---------------------------------------------------------------------------
# 3. Prepare → minimise → score pipeline
# ---------------------------------------------------------------------------

def prepare_and_minimize(
    pdb_path: str,
    work_dir: str,
    skip_minimize: bool = False,
) -> str:
    """
    Run prepare_structure() then minimize_structure() on the input PDB,
    writing intermediate files to work_dir.

    Returns the path to the minimized (or just fixed) PDB.
    """
    stem  = Path(pdb_path).stem
    fixed = os.path.join(work_dir, f"{stem}_fixed.pdb")
    mini  = os.path.join(work_dir, f"{stem}_minimized.pdb")

    print("[Step 0]   Preparing structure (pdbfixer)...")
    prepare_structure(
        input_pdb=pdb_path,
        output_pdb=fixed,
        add_hydrogens=True,
        ph=7.0,
        remove_heterogens=True,
        keep_water=False,
        add_missing_residues=False,  # preserves original residue numbering
    )

    if skip_minimize:
        print("[Step 0]   Skipping minimization (--skip_minimize set)")
        return fixed

    print("[Step 0]   Minimizing structure (OpenMM Amber ff14SB)...")
    minimize_structure(
        input_pdb=fixed,
        output_pdb=mini,
        forcefield="amber14-all",
        max_iterations=1000,
        tolerance=10.0,
    )
    return mini


def run_openmm_scoring(
    minimized_pdb:      str,
    work_dir:           str,
    source_chains:      str,
    target_chains:      str,
    interface_distance: float,
) -> str:
    """
    Call score_residues() on the minimised PDB and return the path to
    the output TSV.
    """
    stem     = Path(minimized_pdb).stem
    tsv_path = os.path.join(work_dir, f"{stem}_scores.tsv")

    print("[Step 0]   Scoring interface energies (OpenMM)...")
    score_residues(
        input_pdb=minimized_pdb,
        output_tsv=tsv_path,
        forcefield="amber14-all",
        source_chains=source_chains,
        target_chains=target_chains,
        interface_only=True,
        interface_distance=interface_distance,
    )
    return tsv_path


def parse_score_tsv(tsv_path: str) -> Dict[Tuple[str, str], float]:
    """
    Read the TSV produced by score_residues() and return a dict mapping
    (chain_id, residue_id) -> total_energy (kJ/mol).

    The TSV has columns: chain_id, residue_id, residue_name, total_energy
    """
    energies: Dict[Tuple[str, str], float] = {}
    with open(tsv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("chain_id"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            chain, resnum, _resname, energy_str = parts[:4]
            try:
                energies[(chain, resnum)] = float(energy_str)
            except ValueError:
                continue
    return energies


# ---------------------------------------------------------------------------
# 4. Main anchor identification logic
# ---------------------------------------------------------------------------

def identify_anchors(
    pdb_path:           str,
    output_dir:         str,
    energy_threshold:   float = -50.0,
    source_chains:      str   = "HL",
    target_chains:      str   = "T",
    interface_distance: float = 4.0,
    skip_minimize:      bool  = False,
) -> List[AnchorResidue]:
    """
    Full pipeline: parse HLT → prepare → minimize → score → filter anchors.

    Parameters
    ----------
    pdb_path           : HLT-annotated antibody-antigen complex
    output_dir         : where to write intermediate and output files
    energy_threshold   : per-residue interface energy cutoff in kJ/mol.
                         Residues below this value (more negative = more
                         stabilising) are flagged as anchors.
    source_chains      : chains to score, e.g. "HL" for scFv, "H" for nanobody
    target_chains      : chains to compute interactions against, e.g. "T"
    interface_distance : heavy-atom distance (Å) for interface residue detection
    skip_minimize      : bypass energy minimisation (faster, less accurate)

    Returns
    -------
    List of AnchorResidue objects below the energy threshold, sorted most
    favourable first.
    """
    if not _SCORER_AVAILABLE:
        raise RuntimeError(
            "score_residues.py could not be imported. "
            "Ensure it is on sys.path or co-located with this script."
        )

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(pdb_path).stem

    # Use a per-design subdirectory for intermediate files so multiple
    # designs can be processed in the same output_dir without collision.
    work_dir = os.path.join(output_dir, f"_work_{stem}")
    os.makedirs(work_dir, exist_ok=True)

    print(f"[Step 0] Processing {stem}")

    # ── Parse CDR ranges from HLT REMARKs ─────────────────────────────────
    print("[Step 0]   Parsing CDR ranges from HLT REMARK annotations...")
    cdr_ranges = parse_hlt_remarks(pdb_path)
    for name, r in sorted(cdr_ranges.items()):
        print(f"           {name}: absolute residues {r.start}–{r.end} "
              f"({r.end - r.start + 1} residues)")

    # Build a fast lookup: absolute pose index -> CDR name
    cdr_lookup: Dict[int, str] = {}
    for cdr_name, r in cdr_ranges.items():
        for idx in range(r.start, r.end + 1):
            cdr_lookup[idx] = cdr_name

    # Build absolute-index -> (chain, resnum, resname) map from original PDB
    residue_map = read_pdb_residue_map(pdb_path)

    # ── Prepare + minimise ─────────────────────────────────────────────────
    minimized_pdb = prepare_and_minimize(pdb_path, work_dir, skip_minimize)

    # ── Score with OpenMM ──────────────────────────────────────────────────
    tsv_path = run_openmm_scoring(
        minimized_pdb=minimized_pdb,
        work_dir=work_dir,
        source_chains=source_chains,
        target_chains=target_chains,
        interface_distance=interface_distance,
    )

    # ── Parse TSV energies ─────────────────────────────────────────────────
    energy_map = parse_score_tsv(tsv_path)  # (chain, resnum_str) -> kJ/mol

    # ── Filter: keep only CDR residues below threshold ─────────────────────
    print(f"[Step 0]   Identifying CDR anchors below {energy_threshold} kJ/mol...")
    anchors: List[AnchorResidue] = []

    for abs_idx, cdr_name in cdr_lookup.items():
        if abs_idx not in residue_map:
            continue
        chain, resnum, resname = residue_map[abs_idx]

        energy = energy_map.get((chain, resnum))
        if energy is None:
            # Residue not scored (e.g. not at interface) — skip
            continue
        if energy > energy_threshold:
            continue

        anchors.append(AnchorResidue(
            cdr=cdr_name,
            pdb_chain=chain,
            pdb_resnum=resnum,
            resname=resname,
            interface_energy=energy,
        ))

    anchors.sort(key=lambda a: a.interface_energy)  # most favourable first

    if anchors:
        print(f"           Found {len(anchors)} anchor residue(s):")
        for a in anchors:
            print(f"             {a.cdr:3s}  {a.pdb_chain}{a.pdb_resnum:>5} "
                  f"{a.resname}  dG={a.interface_energy:+.1f} kJ/mol")
    else:
        print("           No anchor residues found below threshold. "
              "Consider loosening --energy_threshold.")

    return anchors


# ---------------------------------------------------------------------------
# 5. Output writers
# ---------------------------------------------------------------------------

def write_outputs(
    anchors:        List[AnchorResidue],
    all_cdr_ranges: Dict[str, CdrRange],
    pdb_path:       str,
    output_dir:     str,
    energy_threshold: float,
    tsv_path:       Optional[str] = None,
) -> None:
    """
    Write three output files into output_dir:

    1. *_anchors.json           — structured anchor data for Step 1
    2. *_fixed_positions.jsonl  — ProteinMPNN fixed_positions_jsonl format
    3. *_anchors_summary.tsv    — human-readable per-CDR-residue table
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(pdb_path).stem

    # ── 1. anchors.json ────────────────────────────────────────────────────
    anchor_json: Dict = {
        "source_pdb":           str(pdb_path),
        "energy_threshold_kj":  energy_threshold,
        "scorer":               "OpenMM Amber ff14SB inter-chain non-bonded",
        "anchors_by_cdr":       {},
        "all_anchor_residues":  [],
    }
    for a in anchors:
        anchor_json["anchors_by_cdr"].setdefault(a.cdr, []).append({
            "chain":   a.pdb_chain,
            "resnum":  a.pdb_resnum,
            "resname": a.resname,
            "dG_kj":   round(a.interface_energy, 2),
        })
        anchor_json["all_anchor_residues"].append(
            f"{a.pdb_chain}{a.pdb_resnum}"
        )

    json_path = os.path.join(output_dir, f"{stem}_anchors.json")
    with open(json_path, "w") as fh:
        json.dump(anchor_json, fh, indent=2)
    print(f"[Step 0] Written: {json_path}")

    # ── 2. fixed_positions.jsonl ───────────────────────────────────────────
    # ProteinMPNN fixed_positions_jsonl: one JSON-lines record per design.
    # Format: { "pdb_name": { "chain_id": [resnum1, resnum2, ...] } }
    # Residue numbers here are *PDB residue numbers* (strings), not
    # sequential indices — the Step 2 script converts them to 1-based
    # sequential indices as required by ProteinMPNN.
    fixed: Dict[str, List[str]] = {}
    for a in anchors:
        fixed.setdefault(a.pdb_chain, []).append(a.pdb_resnum)

    jsonl_path = os.path.join(output_dir, f"{stem}_fixed_positions.jsonl")
    with open(jsonl_path, "w") as fh:
        fh.write(json.dumps({stem: fixed}) + "\n")
    print(f"[Step 0] Written: {jsonl_path}")

    # ── 3. anchors_summary.tsv ─────────────────────────────────────────────
    # One row per CDR residue showing its energy and anchor status.
    # We reconstruct this from the TSV produced by OpenMM scoring if available,
    # otherwise just write the anchors themselves.
    tsv_out_path = os.path.join(output_dir, f"{stem}_anchors_summary.tsv")
    anchor_set = {(a.pdb_chain, a.pdb_resnum) for a in anchors}

    # If the OpenMM score TSV is accessible, use it for full coverage
    full_energy_map: Dict[Tuple[str, str], Tuple[str, float]] = {}
    if tsv_path and os.path.isfile(tsv_path):
        with open(tsv_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("chain_id"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 4:
                    chain, resnum, resname, energy_str = parts[:4]
                    try:
                        full_energy_map[(chain, resnum)] = (
                            resname, float(energy_str)
                        )
                    except ValueError:
                        pass

    residue_map = read_pdb_residue_map(pdb_path)
    with open(tsv_out_path, "w") as fh:
        fh.write("cdr\tchain\tresnum\tresname\tabs_pose_idx\t"
                 "energy_kj\tis_anchor\n")
        for cdr_name, r in sorted(all_cdr_ranges.items()):
            for abs_idx in range(r.start, r.end + 1):
                if abs_idx not in residue_map:
                    continue
                chain, resnum, resname = residue_map[abs_idx]
                key = (chain, resnum)
                if key in full_energy_map:
                    resname = full_energy_map[key][0]
                    energy  = full_energy_map[key][1]
                    energy_str = f"{energy:.2f}"
                else:
                    energy_str = "NA"
                is_anchor = key in anchor_set
                fh.write(f"{cdr_name}\t{chain}\t{resnum}\t{resname}\t"
                         f"{abs_idx}\t{energy_str}\t{is_anchor}\n")

    print(f"[Step 0] Written: {tsv_out_path}")


# ---------------------------------------------------------------------------
# 6. CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Identify energetic CDR anchor residues from an HLT PDB "
            "using OpenMM Amber ff14SB inter-chain non-bonded scoring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  required=True,
                   help="Path to HLT-annotated antibody-antigen complex PDB")
    p.add_argument("--output_dir", default="anchors",
                   help="Directory to write output files (default: anchors/)")
    p.add_argument("--energy_threshold", type=float, default=-50.0,
                   help="Per-residue interface energy cutoff in kJ/mol "
                        "(default: -50.0; more negative = stricter). "
                        "OpenMM reports in kJ/mol, not REU.")
    p.add_argument("--source_chains", default="HL",
                   help="Chains to score, e.g. 'HL' for scFv, 'H' for "
                        "nanobody (default: HL)")
    p.add_argument("--target_chains", default="T",
                   help="Chains to compute interactions against (default: T)")
    p.add_argument("--interface_distance", type=float, default=4.0,
                   help="Side-chain heavy-atom distance (Å) for interface "
                        "residue detection (default: 4.0)")
    p.add_argument("--skip_minimize", action="store_true",
                   help="Skip energy minimization (faster but less accurate)")
    p.add_argument("--nanobody", action="store_true",
                   help="Shorthand for --source_chains H (overrides "
                        "--source_chains if set)")
    p.add_argument("--dry_run_threshold", action="store_true",
                   help="Score and print the full per-CDR-residue energy "
                        "distribution without applying the threshold, to "
                        "help calibrate --energy_threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.nanobody:
        args.source_chains = "H"

    os.makedirs(args.output_dir, exist_ok=True)
    stem = Path(args.input).stem

    # Build work_dir path so we can pass the TSV to write_outputs
    work_dir = os.path.join(args.output_dir, f"_work_{stem}")

    if args.dry_run_threshold:
        # Run prepare + minimize + score, then print the distribution
        # without applying the threshold so the user can calibrate it.
        minimized_pdb = prepare_and_minimize(
            args.input, work_dir, args.skip_minimize
        )
        tsv_path = run_openmm_scoring(
            minimized_pdb=minimized_pdb,
            work_dir=work_dir,
            source_chains=args.source_chains,
            target_chains=args.target_chains,
            interface_distance=args.interface_distance,
        )
        energy_map = parse_score_tsv(tsv_path)
        cdr_ranges = parse_hlt_remarks(args.input)
        cdr_lookup: Dict[int, str] = {}
        for cdr_name, r in cdr_ranges.items():
            for idx in range(r.start, r.end + 1):
                cdr_lookup[idx] = cdr_name
        residue_map = read_pdb_residue_map(args.input)

        print("\n─── CDR residue energy distribution ───")
        print(f"{'CDR':4s}  {'Res':8s}  {'kJ/mol':>10s}")
        print("─" * 30)
        rows = []
        for abs_idx, cdr_name in sorted(cdr_lookup.items()):
            if abs_idx not in residue_map:
                continue
            chain, resnum, resname = residue_map[abs_idx]
            e = energy_map.get((chain, resnum))
            if e is not None:
                rows.append((cdr_name, f"{chain}{resnum}", resname, e))
        rows.sort(key=lambda x: x[3])
        for cdr_name, res_id, resname, e in rows:
            marker = " ◀" if e < args.energy_threshold else ""
            print(f"{cdr_name:4s}  {res_id:8s}  {e:10.1f}{marker}")
        print("─" * 30)
        energies = [r[3] for r in rows]
        if energies:
            import statistics
            print(f"n={len(energies)}  "
                  f"min={min(energies):.1f}  "
                  f"mean={statistics.mean(energies):.1f}  "
                  f"max={max(energies):.1f} kJ/mol")
            print(f"Current threshold: {args.energy_threshold} kJ/mol  "
                  f"→ would select {sum(1 for e in energies if e < args.energy_threshold)} anchor(s)")
        return

    anchors = identify_anchors(
        pdb_path=args.input,
        output_dir=args.output_dir,
        energy_threshold=args.energy_threshold,
        source_chains=args.source_chains,
        target_chains=args.target_chains,
        interface_distance=args.interface_distance,
        skip_minimize=args.skip_minimize,
    )

    # Reconstruct tsv_path to pass to write_outputs
    minimized_stem = f"{stem}_fixed_minimized"
    tsv_path = os.path.join(work_dir, f"{minimized_stem}_scores.tsv")
    if not os.path.isfile(tsv_path):
        # Fallback: look for any *_scores.tsv in work_dir
        candidates = list(Path(work_dir).glob("*_scores.tsv"))
        tsv_path = str(candidates[0]) if candidates else None

    cdr_ranges = parse_hlt_remarks(args.input)
    write_outputs(
        anchors=anchors,
        all_cdr_ranges=cdr_ranges,
        pdb_path=args.input,
        output_dir=args.output_dir,
        energy_threshold=args.energy_threshold,
        tsv_path=tsv_path,
    )

    print(f"\n[Step 0] Complete. {len(anchors)} anchor residue(s) identified.")
    print(f"         Outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()