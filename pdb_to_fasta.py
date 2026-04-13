"""
pdb_to_fasta.py

Reads all PDB files matching *_seq.pdb in a folder and writes a single
multi-sequence FASTA file.  Each chain in each PDB becomes one FASTA entry.

FASTA header format:
    >{design_id}_{chain}
    e.g. >vanilla_91_H, >vanilla_91_L, >vanilla_91_T

Usage:
    python pdb_to_fasta.py --input_dir /path/to/pdbs/ --output designs.fasta
    python pdb_to_fasta.py --input_dir /path/to/pdbs/ --output designs.fasta \
        --chains H L          # only extract specific chains
        --per_file            # write one FASTA file per PDB instead of one combined
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

# Standard one-letter amino acid code lookup
_AA3_TO_1: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # non-standard / modified — map to X
    "MSE": "M", "SEC": "U", "PYL": "O", "UNK": "X",
}


# ─────────────────────────────────────────────────────────────────────────────
# PDB parsing
# ─────────────────────────────────────────────────────────────────────────────

def extract_sequences(pdb_path: str) -> Dict[str, str]:
    """
    Parse a PDB file and return {chain_id: one_letter_sequence}.

    Uses only ATOM records (not HETATM) and visits each residue only once
    (keyed on chain + residue number + insertion code).
    """
    seen: Dict[str, set] = {}       # chain -> set of (resnum, icode)
    seqs: Dict[str, List[str]] = {} # chain -> list of one-letter codes
    chain_order: List[str] = []

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            chain  = line[21]
            resnum = line[22:26].strip()
            icode  = line[26].strip()
            resn   = line[17:20].strip()
            key    = (resnum, icode)

            if chain not in seen:
                seen[chain]       = set()
                seqs[chain]       = []
                chain_order.append(chain)

            if key not in seen[chain]:
                seen[chain].add(key)
                aa = _AA3_TO_1.get(resn, "X")
                seqs[chain].append(aa)

    return {ch: "".join(seqs[ch]) for ch in chain_order}


# ─────────────────────────────────────────────────────────────────────────────
# FASTA writer
# ─────────────────────────────────────────────────────────────────────────────

def write_fasta(records: List[tuple[str, str]], out_path: str, line_width: int = 60):
    """Write (header, sequence) pairs to a FASTA file."""
    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), line_width):
                fh.write(seq[i : i + line_width] + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def pdb_dir_to_fasta(
    input_dir:  str,
    output:     str,
    chains:     Optional[List[str]] = None,
    per_file:   bool = False,
    glob:       str  = "*_seq.pdb",
    line_width: int  = 60,
):
    input_dir = str(Path(input_dir).resolve())
    pdb_files = sorted(Path(input_dir).glob(glob))

    if not pdb_files:
        print(f"[pdb_to_fasta] No files matching '{glob}' found in {input_dir}")
        return

    print(f"[pdb_to_fasta] Found {len(pdb_files)} PDB file(s)")

    all_records: List[tuple[str, str]] = []
    n_written = 0
    n_skipped = 0

    for pdb in pdb_files:
        # Design ID: strip _seq.pdb suffix
        design_id = pdb.name
        for suffix in ("_seq.pdb", "_seq", ".pdb"):
            if design_id.endswith(suffix):
                design_id = design_id[: -len(suffix)]
                break

        try:
            seqs = extract_sequences(str(pdb))
        except Exception as e:
            print(f"  [WARN] Could not parse {pdb.name}: {e}")
            n_skipped += 1
            continue

        if not seqs:
            print(f"  [WARN] No ATOM records found in {pdb.name}, skipping")
            n_skipped += 1
            continue

        # Filter to requested chains if specified
        if chains:
            seqs = {ch: seq for ch, seq in seqs.items() if ch in chains}
            if not seqs:
                print(f"  [WARN] None of chains {chains} found in {pdb.name}, skipping")
                n_skipped += 1
                continue

        records = [(f"{design_id}_{ch}", seq) for ch, seq in seqs.items()]

        if per_file:
            out_path = str(Path(output).parent / f"{design_id}.fasta")
            write_fasta(records, out_path, line_width)
            print(f"  {pdb.name:55s} → {Path(out_path).name} "
                  f"({len(records)} chain(s))")
        else:
            all_records.extend(records)

        n_written += 1

    if not per_file:
        write_fasta(all_records, output, line_width)
        print(f"\n[pdb_to_fasta] Wrote {len(all_records)} sequence(s) "
              f"from {n_written} PDB(s) → {output}")
    else:
        print(f"\n[pdb_to_fasta] Wrote {n_written} FASTA file(s) "
              f"to {Path(output).parent}/")

    if n_skipped:
        print(f"[pdb_to_fasta] Skipped {n_skipped} file(s) — see warnings above")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write a FASTA file from all *_seq.pdb files in a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input_dir", required=True,
                   help="Directory containing *_seq.pdb files")
    p.add_argument("--output",    required=True,
                   help="Output FASTA file path (or output directory if --per_file)")
    p.add_argument("--chains",    nargs="+", default=None,
                   help="Only extract these chain IDs (e.g. --chains H L). "
                        "Default: all chains")
    p.add_argument("--per_file",  action="store_true",
                   help="Write one FASTA file per PDB instead of one combined file")
    p.add_argument("--glob",      default="*_seq.pdb",
                   help="Glob pattern for PDB files (default: *_seq.pdb)")
    p.add_argument("--line_width", type=int, default=60,
                   help="Sequence line width in FASTA output (default: 60)")
    return p.parse_args()


def main():
    args = parse_args()
    pdb_dir_to_fasta(
        input_dir=args.input_dir,
        output=args.output,
        chains=args.chains,
        per_file=args.per_file,
        glob=args.glob,
        line_width=args.line_width,
    )


if __name__ == "__main__":
    main()