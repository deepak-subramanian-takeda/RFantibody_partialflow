"""
sequence_design_maturation.py

For each HLT-annotated PDB produced by Step 1 (partial diffusion), this
script:

  1. Reads the HLT REMARK lines to identify CDR loop residue positions.
  2. Reads the anchor JSON to identify which CDR residues must be
     sequence-fixed (the energetic anchors).
  3. Computes two ProteinMPNN input files per PDB:
       a. chain_id.jsonl   — tells MPNN which chains to design (H, L) and
                             which to treat as fixed context (T).
       b. fixed_pos.jsonl  — tells MPNN which specific residue positions
                             within the designed chains to hold fixed
                             (framework + anchors). Free CDR positions are
                             left out so MPNN redesigns them.
  4. Invokes RFantibody's `proteinmpnn` CLI via subprocess, passing the
     two generated JSONL files.
  5. Copies the resulting HLT-annotated output PDBs (with redesigned CDR
     sequences) to the output directory, ready for Step 3 (RF2/AF3 filtering).

CRITICAL INDEXING NOTE
----------------------
ProteinMPNN's fixed_positions_jsonl uses *1-based per-chain sequential
indices*, not PDB residue numbers. A residue at PDB chain H, resnum 97 that
is the 73rd residue encountered on chain H in the file has fixed-position
index 73, not 97. This script computes those sequential indices by walking
the ATOM records in file order.

Usage
-----
    python sequence_design_maturation.py \\
        --input_dir   maturation/step1/          \\
        --anchors     anchors/design_0042_anchors.json \\
        --output_dir  maturation/step2/           \\
        --num_seqs    4                            \\
        --temperature 0.1                          \\
        --loops       "H1,H2,H3"                   \\
        --dry_run

    # Full antibody (H+L chains designed):
    python sequence_design_maturation.py \\
        --input_dir   maturation/step1/          \\
        --anchors     anchors/design_0042_anchors.json \\
        --output_dir  maturation/step2/           \\
        --loops       "H1,H2,H3,L1,L2,L3"         \\
        --num_seqs    4                            \\
        --temperature 0.1

Dependencies
------------
    - RFantibody installed (provides `proteinmpnn` CLI)
    - Python >= 3.9 (stdlib only; no PyRosetta required)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CDR_NAMES_ALL = ["H1", "H2", "H3", "L1", "L2", "L3"]
DESIGNABLE_CHAINS = ("H", "L")   # chains ProteinMPNN may alter sequence on
CONTEXT_CHAIN    = "T"           # antigen — always fixed context


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResidueRecord:
    """One unique residue position read from an ATOM/HETATM block."""
    pdb_chain:   str   # 'H', 'L', or 'T'
    pdb_resnum:  int   # PDB residue number (may be non-sequential)
    seq_idx_1:   int   # 1-based sequential index within this chain
    abs_idx:    int   # global pose index matching REMARK lines


@dataclass
class CdrRange:
    """Absolute 1-indexed range for one CDR, from HLT REMARK lines."""
    name:  str
    chain: str         # 'H' or 'L'
    start: int         # 1-indexed absolute pose index
    end:   int


# ---------------------------------------------------------------------------
# 1. HLT file parsing
# ---------------------------------------------------------------------------

def parse_hlt_remarks(pdb_path: str) -> Dict[str, CdrRange]:
    """
    Extract CDR position ranges from HLT REMARK PDBinfo-LABEL lines.
    Returns {cdr_name -> CdrRange} with absolute (across-chain) indices.
    """
    remark_re = re.compile(
        r"^REMARK\s+PDBinfo-LABEL:\s+(\d+)\s+(H[123]|L[123])\s*$"
    )
    positions: Dict[str, List[int]] = {n: [] for n in CDR_NAMES_ALL}

    with open(pdb_path) as fh:
        for line in fh:
            m = remark_re.match(line.strip())
            if m:
                positions[m.group(2)].append(int(m.group(1)))

    out: Dict[str, CdrRange] = {}
    for name, idxs in positions.items():
        if idxs:
            out[name] = CdrRange(
                name=name,
                chain=name[0],
                start=min(idxs),
                end=max(idxs),
            )
    return out


def read_residues(pdb_path: str) -> List[ResidueRecord]:
    seen, records = set(), []
    counters: Dict[str, int] = defaultdict(int)
    abs_counter = 0
    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21].strip()
            if chain not in (DESIGNABLE_CHAINS + (CONTEXT_CHAIN,)):
                continue
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            key = (chain, resnum)
            if key not in seen:
                seen.add(key)
                counters[chain] += 1
                abs_counter += 1
                records.append(ResidueRecord(
                    pdb_chain=chain,
                    pdb_resnum=resnum,
                    seq_idx_1=counters[chain],
                    abs_idx=abs_counter,        # ← store it here
                ))
    return records


def build_pdb_to_seq_map(
    records: List[ResidueRecord],
) -> Dict[Tuple[str, int], int]:
    """Return a fast (chain, pdb_resnum) -> seq_idx_1 lookup dict."""
    return {(r.pdb_chain, r.pdb_resnum): r.seq_idx_1 for r in records}


# ---------------------------------------------------------------------------
# 2. Load anchors from Step 0 JSON
# ---------------------------------------------------------------------------

def load_anchors(json_path: str) -> List[Tuple[str, int]]:
    """
    Parse *_anchors.json produced by Step 0.
    Returns a list of (pdb_chain, pdb_resnum) tuples.
    """
    with open(json_path) as fh:
        data = json.load(fh)

    anchors: List[Tuple[str, int]] = []
    for entry in data.get("all_anchor_residues", []):
        m = re.match(r"^([HLT])(\d+)$", entry)
        if m:
            anchors.append((m.group(1), int(m.group(2))))
        else:
            print(f"[WARN] Skipping unparseable anchor entry: '{entry}'")
    return anchors


# ---------------------------------------------------------------------------
# 3. Compute which residues to fix in ProteinMPNN
# ---------------------------------------------------------------------------

def compute_fixed_positions(
    records:       List[ResidueRecord],
    cdr_ranges:    Dict[str, CdrRange],
    anchor_set:    Set[Tuple[str, int]],
    loops_to_design: List[str],
    pdb_to_seq:    Dict[Tuple[str, int], int],
) -> Dict[str, List[int]]:
    """
    Determine the per-chain lists of 1-based sequential positions that
    ProteinMPNN should hold fixed.

    Logic
    -----
    ProteinMPNN's fixed_positions_jsonl specifies positions TO FIX.
    Everything not listed is designed.

    We want to fix:
      - All framework residues on H and L chains (never CDR positions).
      - CDR anchor residues (from Step 0 energetic analysis).

    We want to leave free (= designable):
      - Non-anchor positions within CDR loops listed in --loops.
      - If a CDR is NOT in --loops, its positions are all fixed (unchanged).

    Parameters
    ----------
    records         : ordered list of all residues in the PDB
    cdr_ranges      : CDR name -> CdrRange (from REMARK lines)
    anchor_set      : set of (chain, pdb_resnum) anchor residues to fix
    loops_to_design : CDR names the user wants to redesign, e.g. ['H1','H3']
    pdb_to_seq      : (chain, resnum) -> per-chain sequential index

    Returns
    -------
    Dict mapping chain letter -> list of 1-based sequential indices to fix.
    Only H and L chains are included (T is fixed by chain_id.jsonl, not here).
    """
    # Build a lookup: absolute pose index -> CDR name, for CDR residues only
    abs_to_cdr: Dict[int, str] = {}
    for name, r in cdr_ranges.items():
        for abs_idx in range(r.start, r.end + 1):
            abs_to_cdr[abs_idx] = name

    fixed: Dict[str, List[int]] = {ch: [] for ch in DESIGNABLE_CHAINS}
    
    # Build an ordered list of (record, absolute_idx) for H/L residues
    for rec in records:
        if rec.pdb_chain not in DESIGNABLE_CHAINS:
            continue
        cdr_name = abs_to_cdr.get(rec.abs_idx)

        if cdr_name is None:
            # Framework residue — always fix
            seq_i = pdb_to_seq[(rec.pdb_chain, rec.pdb_resnum)]
            fixed[rec.pdb_chain].append(seq_i)

        elif cdr_name not in loops_to_design:
            # CDR loop not targeted for redesign — fix entire loop
            seq_i = pdb_to_seq[(rec.pdb_chain, rec.pdb_resnum)]
            fixed[rec.pdb_chain].append(seq_i)

        else:
            # CDR loop IS targeted — fix only anchor residues
            if (rec.pdb_chain, rec.pdb_resnum) in anchor_set:
                seq_i = pdb_to_seq[(rec.pdb_chain, rec.pdb_resnum)]
                fixed[rec.pdb_chain].append(seq_i)
            # else: free position — not added to fixed list

    # Remove chains with no residues in the PDB (nanobody has no L)
    return {
        ch: sorted(idxs)
        for ch, idxs in fixed.items()
        if any(r.pdb_chain == ch for r in records)
    }

def load_anchor_resnames(anchors_json: str) -> Dict[Tuple[str, int], str]:
    """
    Read the anchor JSON and return a dict mapping
    (chain, resnum) -> three-letter residue name.

    e.g. {('H', 105): 'TYR', ('H', 57): 'TYR', ('L', 214): 'TYR', ...}
    """
    with open(anchors_json) as fh:
        data = json.load(fh)

    resnames: Dict[Tuple[str, int], str] = {}
    for cdr_anchors in data.get("anchors_by_cdr", {}).values():
        for entry in cdr_anchors:
            chain  = entry["chain"]
            resnum = int(entry["resnum"])
            resname = entry["resname"].upper().strip()
            resnames[(chain, resnum)] = resname
    return resnames


def graft_anchor_residues(
    pdb_path:       str,
    anchor_resnames: Dict[Tuple[str, int], str],
    out_path:       str,
) -> str:
    """
    Write a copy of pdb_path to out_path where ATOM/HETATM records for
    anchor positions have their residue name replaced with the original
    identity from the source structure.

    RFdiffusion fills all residues with GLY as a placeholder.  ProteinMPNN
    reads residue names from the PDB when scoring fixed positions, so fixing
    a GLY at H105 tells MPNN to keep glycine there — not the intended TYR.
    This function corrects that before ProteinMPNN runs.

    PDB residue name occupies columns 18-20 (0-indexed 17:20), right-aligned
    in a 3-character field.  Chain is column 22 (0-indexed 21).
    Residue number is columns 23-26 (0-indexed 22:26).
    """
    out_lines = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain = line[21]
                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    out_lines.append(line)
                    continue

                key = (chain, resnum)
                if key in anchor_resnames:
                    correct_name = anchor_resnames[key].ljust(3)[:3]
                    # PDB format: columns 17-20 are residue name (1-indexed)
                    # i.e. line[17:20] in 0-indexed Python slicing
                    line = line[:17] + correct_name + line[20:]

            out_lines.append(line)

    with open(out_path, "w") as fh:
        fh.writelines(out_lines)

    return out_path    


# ---------------------------------------------------------------------------
# 4. Build chain_id.jsonl
# ---------------------------------------------------------------------------

def build_chain_id_record(
    pdb_name: str,
    records:  List[ResidueRecord],
) -> Dict:
    """
    Produce a chain_id_dict entry for one PDB.

    protein_mpnn_run.py expects this format (from protein_mpnn_utils.py
    tied_featurize / get_chain_id_dict):

        { "pdb_name": {"designed_chain_list": ["H", "L"],
                       "fixed_chain_list":    ["T"]} }

    NOT the nested list format [["H","L"],["T"]] which is silently ignored,
    causing all chains including T to be designed.
    """
    present = {r.pdb_chain for r in records}
    designed = sorted(ch for ch in DESIGNABLE_CHAINS if ch in present)
    fixed    = [CONTEXT_CHAIN] if CONTEXT_CHAIN in present else []

    return {
        pdb_name: {
            "designed_chain_list": designed,
            "fixed_chain_list":    fixed,
        }
    }


# ---------------------------------------------------------------------------
# 5. Build fixed_positions.jsonl
# ---------------------------------------------------------------------------

def build_fixed_positions_record(
    pdb_name:       str,
    fixed_per_chain: Dict[str, List[int]],
    records:        List[ResidueRecord],
) -> Dict:
    """
    Produce a fixed_positions_dict entry for one PDB.

    ProteinMPNN format (from protein_mpnn_utils.py tied_featurize):
        { "pdb_name": { "H": [seq_idx_1, seq_idx_2, ...],
                        "L": [...],
                        "T": [] } }

    Keys must include ALL chains present (including T), but T's list is
    empty because T is in the fixed chain group — its sequence is never
    designed regardless of this dict.

    Indices are 1-based per-chain sequential integers.
    """
    present = {r.pdb_chain for r in records}

    record = {}
    for ch in sorted(present):
        if ch in fixed_per_chain:
            record[ch] = fixed_per_chain[ch]
        else:
            record[ch] = []   # context chain (T) or absent chain

    return {pdb_name: record}


# ---------------------------------------------------------------------------
# 6. Write JSONL helper
# ---------------------------------------------------------------------------

def write_jsonl(records: List[Dict], path: str) -> None:
    """Write a list of dicts as JSON-lines (one JSON object per line)."""
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# 7. Per-PDB processing
# ---------------------------------------------------------------------------

def process_pdb(
    pdb_path:    str,
    anchor_set:  Set[Tuple[str, int]],
    loops_to_design: List[str],
    scratch_dir: str,
) -> Tuple[str, str, str, Dict]:
    """
    Parse one PDB and produce the JSONL records needed by ProteinMPNN.

    Returns
    -------
    (pdb_name, chain_id_record, fixed_pos_record, design_summary)
    where design_summary is a human-readable dict for logging.
    """
    pdb_name = Path(pdb_path).stem

    cdr_ranges  = parse_hlt_remarks(pdb_path)
    records     = read_residues(pdb_path)
    pdb_to_seq  = build_pdb_to_seq_map(records)

    if not cdr_ranges:
        raise ValueError(
            f"No CDR REMARK lines found in {pdb_path}. "
            "Is this a valid HLT-annotated PDB from Step 1?"
        )

    fixed_per_chain = compute_fixed_positions(
        records=records,
        cdr_ranges=cdr_ranges,
        anchor_set=anchor_set,
        loops_to_design=loops_to_design,
        pdb_to_seq=pdb_to_seq,
    )

    chain_id_rec  = build_chain_id_record(pdb_name, records)
    fixed_pos_rec = build_fixed_positions_record(
        pdb_name, fixed_per_chain, records
    )

    # Compute summary stats for logging
    total_designable = 0
    n_fixed_in_cdr   = 0
    n_free_in_cdr    = 0
    chain_lengths: Dict[str, int] = defaultdict(int)
    for r in records:
        chain_lengths[r.pdb_chain] += 1

    for ch in DESIGNABLE_CHAINS:
        if ch not in chain_lengths:
            continue
        n_fixed = len(fixed_per_chain.get(ch, []))
        n_total = chain_lengths[ch]
        total_designable += n_total - n_fixed

    # Count free CDR positions specifically
    for name, cr in cdr_ranges.items():
        if name not in loops_to_design:
            continue
        for abs_idx in range(cr.start, cr.end + 1):
            # Find the residue with this absolute index
            for rec in records:
                pass  # already handled in compute_fixed_positions
        # Simpler: count loop length minus anchors in that loop
        loop_len = cr.end - cr.start + 1
        anchors_in_loop = sum(
            1 for rec in records
            if rec.pdb_chain == cr.chain
            and (rec.pdb_chain, rec.pdb_resnum) in anchor_set
            # Check if this residue is within the CDR range
            # We need the absolute index — reconstruct it
        )
        n_free_in_cdr += loop_len  # rough; refined below

    summary = {
        "pdb": pdb_name,
        "cdrs_found": sorted(cdr_ranges.keys()),
        "loops_to_design": loops_to_design,
        "anchor_count": len(anchor_set),
        "fixed_per_chain": {
            ch: len(idxs) for ch, idxs in fixed_per_chain.items()
        },
        "chain_lengths": dict(chain_lengths),
    }

    return pdb_name, chain_id_rec, fixed_pos_rec, summary


# ---------------------------------------------------------------------------
# 8. Build and run the proteinmpnn CLI call
# ---------------------------------------------------------------------------

def run_proteinmpnn(
    input_dir:        str,
    output_dir:       str,
    original_hlt_pdb: str,
    chain_id_jsonl:   str,
    fixed_pos_jsonl:  str,
    loops:            str,
    num_seqs:         int,
    temperature:      float,
    extra_args:       List[str],
    dry_run:          bool = False,
    anchor_set:       Optional[Set[Tuple[str, int]]] = None,
    loops_to_design:  Optional[List[str]] = None,
    anchor_resnames:  Optional[Dict[Tuple[str, int], str]] = None,  # NEW
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    pdb_files = sorted(Path(input_dir).glob("*.pdb"))
    if not pdb_files:
        sys.exit(f"[ERROR] No .pdb files found in {input_dir}")

    repo_root = str(Path(__file__).resolve().parent)
    mpnn_script = _find_protein_mpnn_run(repo_root)
    if mpnn_script is None:
        sys.exit(
            "[ERROR] Cannot locate protein_mpnn_run.py. "
            "Set RFANTIBODY_MPNN_SCRIPT to its absolute path."
        )

    weights_dir = os.path.join(repo_root, "weights", "vanilla_model_weights")

    print(f"\n[Step 2] Running protein_mpnn_run.py on {len(pdb_files)} PDB(s)...")

    scratch_root = os.path.join(output_dir, "_jsonl_per_pdb")
    os.makedirs(scratch_root, exist_ok=True)

    # Directory for grafted PDBs (anchor residues restored from source)
    grafted_dir = os.path.join(output_dir, "_grafted")
    os.makedirs(grafted_dir, exist_ok=True)

    for pdb_path in pdb_files:
        pdb_name = pdb_path.stem

        # ── Step 2a: graft original anchor residue names onto RFdiffusion
        # GLY placeholders before ProteinMPNN reads the file ──────────────
        if anchor_resnames:
            grafted_path = os.path.join(grafted_dir, pdb_path.name)
            graft_anchor_residues(
                pdb_path=str(pdb_path),
                anchor_resnames=anchor_resnames,
                out_path=grafted_path,
            )
            working_pdb = grafted_path
            print(f"\n  PDB: {pdb_path.name} (anchor residues grafted)")
        else:
            working_pdb = str(pdb_path)
            print(f"\n  PDB: {pdb_path.name}")

        # ── Step 2b: rebuild JSONL records keyed by THIS PDB's stem ──────
        try:
            records    = read_residues(working_pdb)
            cdr_ranges = parse_hlt_remarks(original_hlt_pdb)
            pdb_to_seq = build_pdb_to_seq_map(records)

            if not cdr_ranges:
                print(f"  [WARN] No CDR REMARKs in {pdb_path.name} — skipping.")
                continue

            fixed_per_chain = compute_fixed_positions(
                records=records,
                cdr_ranges=cdr_ranges,
                anchor_set=anchor_set or set(),
                loops_to_design=loops_to_design or [],
                pdb_to_seq=pdb_to_seq,
            )

            chain_id_rec  = build_chain_id_record(pdb_name, records)
            fixed_pos_rec = build_fixed_positions_record(
                pdb_name, fixed_per_chain, records
            )

        except Exception as e:
            print(f"  [WARN] Could not build JSONL for {pdb_path.name}: {e} — skipping.")
            continue

        _verify_anchors(
            pdb_name=pdb_name,
            records=records,
            fixed_per_chain=fixed_per_chain,
            anchor_set=anchor_set or set(),
            pdb_to_seq=pdb_to_seq,
        )

        pdb_scratch    = os.path.join(scratch_root, pdb_name)
        os.makedirs(pdb_scratch, exist_ok=True)
        this_chain_id  = os.path.join(pdb_scratch, "chain_ids.jsonl")
        this_fixed_pos = os.path.join(pdb_scratch, "fixed_positions.jsonl")
        write_jsonl([chain_id_rec],  this_chain_id)
        write_jsonl([fixed_pos_rec], this_fixed_pos)

        cmd = _build_rfantibody_mpnn_cmd(
            pdb_path=working_pdb,        # ← grafted PDB, not original
            output_dir=output_dir,
            chain_id_jsonl=this_chain_id,
            fixed_pos_jsonl=this_fixed_pos,
            num_seqs=num_seqs,
            temperature=temperature,
            mpnn_script=mpnn_script,
            extra_args=extra_args,
            model_weights_dir=weights_dir,
            model_name="v_48_020",
        )

        if dry_run:
            print("  CMD: " + " \\\n       ".join(cmd))
            for ch, idxs in sorted(fixed_per_chain.items()):
                n_total = sum(1 for r in records if r.pdb_chain == ch)
                print(f"  Chain {ch}: {len(idxs)}/{n_total} fixed, "
                      f"{n_total - len(idxs)} designable")
            continue

        result = subprocess.run(" ".join(cmd), shell=True)
        if result.returncode != 0:
            print(f"  [WARN] protein_mpnn_run.py failed for {pdb_path.name} "
                  f"(exit code {result.returncode}) — skipping.")

    if dry_run:
        print("\n[Step 2] DRY RUN — protein_mpnn_run.py not invoked.")


def _verify_anchors(
    pdb_name:        str,
    records:         List[ResidueRecord],
    fixed_per_chain: Dict[str, List[int]],
    anchor_set:      Set[Tuple[str, int]],
    pdb_to_seq:      Dict[Tuple[str, int], int],
) -> None:
    """
    Print a warning for any anchor residue that is NOT in the fixed list.
    This catches indexing bugs before protein_mpnn_run.py silently designs
    over them.
    """
    missing = []
    for chain, resnum in sorted(anchor_set):
        key = (chain, resnum)
        seq_i = pdb_to_seq.get(key)
        if seq_i is None:
            missing.append(f"{chain}{resnum} (not found in PDB)")
            continue
        if seq_i not in fixed_per_chain.get(chain, []):
            missing.append(f"{chain}{resnum} (seq_idx={seq_i} NOT in fixed list)")

    if missing:
        print(f"  [WARN] {pdb_name}: anchor residues not fixed:")
        for m in missing:
            print(f"    {m}")
    else:
        n = len(anchor_set)
        print(f"  [OK] All {n} anchor residue(s) confirmed fixed.")


def _find_protein_mpnn_run(repo_root: str) -> Optional[str]:
    candidates = [
        Path(repo_root) / "src" / "rfantibody" / "proteinmpnn" / "model" / "protein_mpnn_run.py",
        Path(repo_root) / ".venv" / "lib" / "python3.10" / "site-packages"
            / "rfantibody" / "proteinmpnn" / "model" / "protein_mpnn_run.py",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    for p in Path(repo_root).rglob("protein_mpnn_run.py"):
        if ".venv" not in str(p):
            return str(p)
    return None


def _build_rfantibody_mpnn_cmd(
    pdb_path:          str,
    output_dir:        str,
    chain_id_jsonl:    str,
    fixed_pos_jsonl:   str,
    num_seqs:          int,
    temperature:       float,
    mpnn_script:       str,
    extra_args:        List[str],
    model_weights_dir: Optional[str] = None,
    model_name:        str = "v_48_020",
) -> List[str]:
    cmd = [sys.executable, mpnn_script]
    cmd += ["--pdb_path",           pdb_path]
    cmd += ["--out_folder",         output_dir]
    cmd += ["--num_seq_per_target", str(num_seqs)]
    cmd += ["--sampling_temp",      str(temperature)]
    cmd += ["--batch_size",         "1"]
    cmd += ["--model_name",         model_name]
    if model_weights_dir:
        cmd += ["--path_to_model_weights", model_weights_dir]
    if os.path.isfile(chain_id_jsonl):
        cmd += ["--chain_id_jsonl",         chain_id_jsonl]
    if os.path.isfile(fixed_pos_jsonl):
        cmd += ["--fixed_positions_jsonl",  fixed_pos_jsonl]
    cmd.extend(extra_args)
    return cmd


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 2: Run ProteinMPNN on Step 1 partial-diffusion outputs, "
            "fixing anchor residue sequences and framework while redesigning "
            "free CDR loop positions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_dir", required=True,
                   help="Directory of HLT-annotated PDBs from Step 1 "
                        "(partial diffusion outputs)")
    p.add_argument("--anchors", required=True,
                   help="Path to *_anchors.json from Step 0")
    p.add_argument("--output_dir", required=True,
                   help="Directory for ProteinMPNN output PDBs")

    # ProteinMPNN settings
    p.add_argument("--loops", default="H1,H2,H3",
                   help="Comma-separated CDR loops to redesign, "
                        "e.g. 'H1,H2,H3' (nanobody) or "
                        "'H1,H2,H3,L1,L2,L3' (scFv). "
                        "(default: H1,H2,H3)")
    p.add_argument("--original_hlt_pdb", default="",
                    help="Original HLT PDB for parsing REMARKs against")
    p.add_argument("--num_seqs", "-n", type=int, default=4,
                   help="Sequences per backbone structure (default: 4)")
    p.add_argument("--temperature", "-t", type=float, default=0.1,
                   help="Sampling temperature 0.1–0.3. "
                        "Lower = more conservative. (default: 0.1)")

    # Execution
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands and JSONL content but do not run")
    p.add_argument("--keep_scratch", action="store_true",
                   help="Keep the temporary JSONL files after the run "
                        "(useful for debugging or inspection)")

    # Passthrough
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra flags passed verbatim to proteinmpnn")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir  = str(Path(args.input_dir).resolve())
    anchors_json = str(Path(args.anchors).resolve())
    output_dir = str(Path(args.output_dir).resolve())
    original_hlt_pdb = str(Path(args.original_hlt_pdb).resolve())
    os.makedirs(output_dir, exist_ok=True)

    # Parse loop list
    loops_to_design = [l.strip() for l in args.loops.split(",") if l.strip()]
    invalid = [l for l in loops_to_design if l not in CDR_NAMES_ALL]
    if invalid:
        sys.exit(f"[ERROR] Unknown loop names: {invalid}. "
                 f"Valid: {CDR_NAMES_ALL}")

    # Find PDB files in input_dir
    pdb_files = sorted(Path(input_dir).glob("*.pdb"))
    if not pdb_files:
        sys.exit(f"[ERROR] No .pdb files found in {input_dir}")
    print(f"[Step 2] Found {len(pdb_files)} PDB(s) in {input_dir}")

    # Load anchor set
    print(f"[Step 2] Loading anchors from {anchors_json}")
    anchors = load_anchors(anchors_json)
    anchor_resnames = load_anchor_resnames(anchors_json)
    anchor_set: Set[Tuple[str, int]] = set(anchors)
    print(f"         {len(anchor_set)} anchor residue(s): "
          f"{[f'{c}{n}' for c,n in sorted(anchor_set)]}")

    print(f"[Step 2] Loops to design: {loops_to_design}")
    print(f"         Sequences per structure: {args.num_seqs}")
    print(f"         Temperature: {args.temperature}")

    # Process all PDBs and accumulate JSONL records
    all_chain_id_records:   List[Dict] = []
    all_fixed_pos_records:  List[Dict] = []

    for pdb_path in pdb_files:
        print(f"\n[Step 2] Processing {pdb_path.name}...")
        try:
            pdb_name, chain_id_rec, fixed_pos_rec, summary = process_pdb(
                pdb_path=str(pdb_path),
                anchor_set=anchor_set,
                loops_to_design=loops_to_design,
                scratch_dir=output_dir,
            )
        except ValueError as e:
            print(f"  [WARN] Skipping {pdb_path.name}: {e}")
            continue

        all_chain_id_records.append(chain_id_rec)
        all_fixed_pos_records.append(fixed_pos_rec)

        # Log per-PDB summary
        for ch, n_fixed in sorted(summary["fixed_per_chain"].items()):
            n_total = summary["chain_lengths"].get(ch, 0)
            n_free  = n_total - n_fixed
            print(f"  Chain {ch}: {n_total} total | "
                  f"{n_fixed} fixed (framework+anchors) | "
                  f"{n_free} designable")
        print(f"  CDRs found: {summary['cdrs_found']}")

    if not all_chain_id_records:
        sys.exit("[ERROR] No valid PDBs could be processed.")

    # Write JSONL files to a scratch location (temp dir or output_dir)
    scratch_dir = output_dir if args.keep_scratch else tempfile.mkdtemp(
        prefix="rfantibody_step2_"
    )
    chain_id_path  = os.path.join(scratch_dir, "chain_ids.jsonl")
    fixed_pos_path = os.path.join(scratch_dir, "fixed_positions.jsonl")

    write_jsonl(all_chain_id_records,  chain_id_path)
    write_jsonl(all_fixed_pos_records, fixed_pos_path)

    print(f"\n[Step 2] JSONL files written:")
    print(f"  chain_id:       {chain_id_path}")
    print(f"  fixed_positions:{fixed_pos_path}")

    if args.dry_run:
        # Print the first record of each file for inspection
        print("\n--- chain_id.jsonl (first record) ---")
        print(json.dumps(all_chain_id_records[0], indent=2))
        print("\n--- fixed_positions.jsonl (first record) ---")
        first = all_fixed_pos_records[0]
        pname = list(first.keys())[0]
        # Show a compact summary (full list may be long)
        compact = {
            pname: {
                ch: f"[{len(v)} positions]"
                for ch, v in first[pname].items()
            }
        }
        print(json.dumps(compact, indent=2))
        print()

    # Run ProteinMPNN
    extra = [a for a in args.extra if a != "--"]
    run_proteinmpnn(
    input_dir=input_dir,
    output_dir=output_dir,
    chain_id_jsonl=chain_id_path,
    fixed_pos_jsonl=fixed_pos_path,
    loops=args.loops,
    original_hlt_pdb=original_hlt_pdb,
    num_seqs=args.num_seqs,
    temperature=args.temperature,
    extra_args=extra,
    dry_run=args.dry_run,
    anchor_set=anchor_set,
    loops_to_design=loops_to_design,
    anchor_resnames=anchor_resnames,
)

    if not args.dry_run:
        print(f"\n[Step 2] Complete. Outputs in: {output_dir}/")
        print(f"         Feed these into Step 3 (RF2/AF3 filtering):")
        print(f"         rf2 -i {output_dir}/ "
              f"--output-quiver step3_rf2.qv -r 10")

    # Clean up scratch unless requested
    if not args.keep_scratch and not args.dry_run:
        import shutil as _sh
        _sh.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()