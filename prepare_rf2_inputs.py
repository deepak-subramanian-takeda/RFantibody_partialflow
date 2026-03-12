"""
prepare_rf2_inputs.py

Threads ProteinMPNN-designed sequences (from Step 2 FASTA files) onto the
corresponding backbone PDB files (from Step 1 / _grafted/), producing
HLT-annotated PDB files ready for RF2 scoring.

For each .fa file in --fasta_dir:
  - Parses all sample sequences (skipping the first header which is the
    backbone/input sequence with GLY placeholders)
  - Finds the matching backbone PDB in --backbone_dir by stem name
  - For each sample, writes a new PDB where ATOM residue names are replaced
    with the designed amino acid, preserving all backbone coordinates and
    HLT REMARK annotations
  - Output PDBs are written to --output_dir as:
      <stem>_sample<N>.pdb

Usage
-----
    python prepare_rf2_inputs.py \\
        --fasta_dir   1n8z_step2/seqs/ \\
        --backbone_dir 1n8z_step2/_grafted/ \\
        --output_dir  1n8z_rf2_inputs/ \\
        --chains      H,L,T

    # Then run RF2:
    rfantibody rf2 \\
        input.pdb_dir=1n8z_rf2_inputs/ \\
        output.pdb_dir=1n8z_rf2_out/

Dependencies
------------
    Python >= 3.9, stdlib only.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Three-letter to one-letter amino acid code
AA3TO1: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "HSD": "H", "HSE": "H", "HSP": "H",
}

# One-letter to three-letter amino acid code
AA1TO3: Dict[str, str] = {v: k for k, v in AA3TO1.items() if k in {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
}}


# ---------------------------------------------------------------------------
# 1. FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta(fa_path: str) -> List[Tuple[str, str, int]]:
    """
    Parse a ProteinMPNN FASTA file.

    Returns a list of (header, sequence, sample_index) tuples for all
    designed samples, skipping the first record (which is the backbone
    input sequence containing GLY placeholders).

    The sequence has chains separated by '/' — we strip those to get a
    flat sequence string, then split back by chain length when threading.
    """
    records = []
    header = None
    seq_lines = []

    with open(fa_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            records.append((header, "".join(seq_lines)))

    # Skip the first record (backbone GLY sequence)
    samples = []
    sample_idx = 0
    for i, (hdr, seq) in enumerate(records):
        if i == 0:
            continue   # backbone input — skip
        sample_idx += 1
        samples.append((hdr, seq, sample_idx))

    return samples


def split_sequence_by_chain(
    flat_seq: str,
    chain_order: List[str],
    chain_lengths: Dict[str, int],
) -> Dict[str, str]:
    """
    Split a flat (or slash-separated) sequence string into per-chain
    sequences, given the expected chain order and lengths.

    ProteinMPNN outputs chains separated by '/' in the same order as the
    chains appear in the PDB (H, L, T for a full antibody complex).
    """
    # Try splitting on '/' first
    parts = flat_seq.split("/")
    if len(parts) == len(chain_order):
        return {ch: parts[i] for i, ch in enumerate(chain_order)}

    # Fallback: use chain lengths to slice the flat sequence
    result = {}
    pos = 0
    for ch in chain_order:
        n = chain_lengths[ch]
        result[ch] = flat_seq[pos:pos + n]
        pos += n
    return result


# ---------------------------------------------------------------------------
# 2. PDB parsing
# ---------------------------------------------------------------------------

def read_pdb_chain_order(pdb_path: str, chains: List[str]) -> List[str]:
    """Return chain letters in the order they first appear in the PDB."""
    seen = []
    seen_set = set()
    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            ch = line[21]
            if ch in chains and ch not in seen_set:
                seen.append(ch)
                seen_set.add(ch)
    return seen


def read_pdb_chain_lengths(pdb_path: str, chains: List[str]) -> Dict[str, int]:
    """Return the number of unique residues per chain."""
    seen: Dict[str, set] = {ch: set() for ch in chains}
    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            ch = line[21]
            if ch not in chains:
                continue
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            seen[ch].add(resnum)
    return {ch: len(resnums) for ch, resnums in seen.items()}


def build_resnum_to_seqpos(
    pdb_path: str,
    chains: List[str],
) -> Dict[Tuple[str, int], int]:
    """
    Build a mapping from (chain, pdb_resnum) -> 0-based position in the
    chain's sequence string.

    This is the key lookup used when threading: for each ATOM record we
    find which position in the designed sequence corresponds to it.
    """
    counters: Dict[str, int] = {ch: -1 for ch in chains}
    seen: Dict[str, set] = {ch: set() for ch in chains}
    mapping: Dict[Tuple[str, int], int] = {}

    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            ch = line[21]
            if ch not in chains:
                continue
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            key = (ch, resnum)
            if key not in seen[ch]:
                seen[ch].add(key)
                counters[ch] += 1
                mapping[key] = counters[ch]

    return mapping


# ---------------------------------------------------------------------------
# 3. Sequence threading
# ---------------------------------------------------------------------------

def thread_sequence_onto_pdb(
    pdb_path:        str,
    chain_seqs:      Dict[str, str],    # chain -> one-letter sequence
    resnum_to_seqpos: Dict[Tuple[str, int], int],
    designed_chains: List[str],         # chains where sequence should change
    out_path:        str,
) -> None:
    """
    Write a new PDB to out_path where ATOM residue names for designed_chains
    are replaced with the amino acids from chain_seqs.

    Chains not in designed_chains (e.g. T) are copied verbatim.
    REMARK lines (HLT CDR annotations) are preserved unchanged.
    """
    out_lines = []

    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                out_lines.append(line)
                continue

            ch = line[21]
            if ch not in designed_chains:
                out_lines.append(line)
                continue

            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                out_lines.append(line)
                continue

            key = (ch, resnum)
            seqpos = resnum_to_seqpos.get(key)
            if seqpos is None:
                out_lines.append(line)
                continue

            # Get the designed amino acid at this position
            seq = chain_seqs.get(ch, "")
            if seqpos >= len(seq):
                out_lines.append(line)
                continue

            aa1 = seq[seqpos]
            aa3 = AA1TO3.get(aa1.upper())
            if aa3 is None:
                # Unknown AA — keep original
                out_lines.append(line)
                continue

            # Replace residue name in columns 17-20 (0-indexed)
            new_line = line[:17] + aa3.ljust(3)[:3] + line[20:]
            out_lines.append(new_line)

    with open(out_path, "w") as fh:
        fh.writelines(out_lines)

def restore_hlt_remarks(pdb_path, original_pdb, out_path):
    original_remarks = []
    with open(original_pdb) as fh:
        for line in fh:
            if re.match(r"^REMARK\s+PDBinfo-LABEL:", line):
                original_remarks.append(line if line.endswith("\n") else line + "\n")

    out_lines = []
    remarks_inserted = False
    with open(pdb_path) as fh:
        for line in fh:
            if re.match(r"^REMARK\s+PDBinfo-LABEL:", line):
                if not remarks_inserted:
                    out_lines.extend(original_remarks)
                    remarks_inserted = True
                # Always skip the existing incomplete line (don't conditionally drop)
            else:
                out_lines.append(line)

    # Insert before first ATOM if no REMARK block existed
    if not remarks_inserted:
        final_lines = []
        inserted = False
        for line in out_lines:
            if not inserted and (line.startswith("ATOM") or line.startswith("HETATM")):
                final_lines.extend(original_remarks)
                inserted = True
            final_lines.append(line)
        out_lines = final_lines

    with open(out_path, "w") as fh:
        fh.writelines(out_lines)
    return out_path


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Thread ProteinMPNN-designed sequences onto backbone PDBs "
            "to produce HLT-annotated PDB files for RF2 input."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--fasta_dir", required=True)
    p.add_argument("--backbone_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--original_pdb", required=True,          # NEW
                   help="Original HLT complex PDB used as Step 1 input "
                        "(e.g. scripts/examples/example_inputs/1n8z_hlt.pdb). "
                        "Its complete REMARK PDBinfo-LABEL lines are restored "
                        "in every output PDB so RF2 can parse CDR positions.")
    p.add_argument("--chains", default="H,L,T")
    p.add_argument("--designed_chains", default="H,L")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    fasta_dir    = str(Path(args.fasta_dir).resolve())
    backbone_dir = str(Path(args.backbone_dir).resolve())
    output_dir   = str(Path(args.output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    chains          = [c.strip() for c in args.chains.split(",")]
    designed_chains = [c.strip() for c in args.designed_chains.split(",")]

    fa_files = sorted(Path(fasta_dir).glob("*.fa"))
    if not fa_files:
        sys.exit(f"[ERROR] No .fa files found in {fasta_dir}")
    print(f"[prepare_rf2] Found {len(fa_files)} FASTA file(s)")

    n_written = 0
    n_skipped = 0

    for fa_path in fa_files:
        stem = fa_path.stem   # e.g. '1n8z_hlt_partial_T15_10'

        # Find matching backbone PDB
        backbone_pdb = Path(backbone_dir) / f"{stem}.pdb"
        if not backbone_pdb.exists():
            print(f"  [WARN] No backbone PDB for {stem} in {backbone_dir} — skipping.")
            n_skipped += 1
            continue

        # Parse chain order and lengths from backbone
        chain_order  = read_pdb_chain_order(str(backbone_pdb), chains)
        chain_lengths = read_pdb_chain_lengths(str(backbone_pdb), chains)
        resnum_to_seqpos = build_resnum_to_seqpos(str(backbone_pdb), chains)

        # Parse designed sequences
        samples = parse_fasta(str(fa_path))
        if not samples:
            print(f"  [WARN] No samples found in {fa_path.name} — skipping.")
            n_skipped += 1
            continue

        print(f"  {stem}: {len(samples)} sample(s)")

        for hdr, flat_seq, sample_idx in samples:
            # Split sequence by chain
            try:
                chain_seqs = split_sequence_by_chain(
                    flat_seq, chain_order, chain_lengths
                )
            except Exception as e:
                print(f"    [WARN] Sample {sample_idx}: sequence split failed: {e}")
                n_skipped += 1
                continue

            # Validate lengths
            length_ok = True
            for ch in designed_chains:
                if ch not in chain_seqs:
                    continue
                expected = chain_lengths.get(ch, 0)
                got = len(chain_seqs[ch])
                if got != expected:
                    print(f"    [WARN] Sample {sample_idx} chain {ch}: "
                          f"expected {expected} residues, got {got} — skipping.")
                    length_ok = False
                    break
            if not length_ok:
                n_skipped += 1
                continue

            # Write threaded PDB
            out_name = f"{stem}_sample{sample_idx}.pdb"
            out_path = os.path.join(output_dir, out_name)

            if not args.dry_run:
                # Step A: thread designed sequence onto backbone
                threaded_tmp = out_path + ".tmp.pdb"
                thread_sequence_onto_pdb(
                    pdb_path=str(backbone_pdb),
                    chain_seqs=chain_seqs,
                    resnum_to_seqpos=resnum_to_seqpos,
                    designed_chains=designed_chains,
                    out_path=threaded_tmp,
                )
                # Step B: restore complete REMARK lines from original PDB
                restore_hlt_remarks(
                    pdb_path=threaded_tmp,
                    original_pdb=args.original_pdb,
                    out_path=out_path,
                )
                os.remove(threaded_tmp)
            n_written += 1

    print(f"\n[prepare_rf2] Done.")
    print(f"  Written : {n_written} PDB(s) -> {output_dir}/")
    print(f"  Skipped : {n_skipped}")

    if not args.dry_run and n_written > 0:
        print(f"\n[prepare_rf2] Run RF2 with:")
        print(f"  rfantibody rf2 \\")
        print(f"      input.pdb_dir={output_dir}/ \\")
        print(f"      output.pdb_dir=1n8z_rf2_out/")


if __name__ == "__main__":
    main()