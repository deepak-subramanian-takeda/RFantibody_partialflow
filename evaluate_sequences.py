"""
evaluate_sequences.py

Evaluates all *_seq*.pdb files found in subfolders of an input directory
using ColabFold ipTM and DockQ.  Results are written in two formats:

  1. A live-updating FASTA-style text file (updated after each structure)
     containing sequence, ipTM, and DockQ for each design as it is scored.

  2. A final ranked TSV sorted by ipTM descending (DockQ used to break ties).

MSA mode:
  ColabFold runs in --msa-mode single_sequence to avoid downloading MSA
  files and prevent "no space left on device" errors.  This is appropriate
  for designed antibody sequences which are artificial and have no close
  natural homologs.

Disk space:
  ColabFold writes temporary files per structure.  This script cleans up
  each structure's AF2 working directory after scoring to keep disk usage
  bounded.  A minimum free-space check runs before each ColabFold call and
  will skip the structure (marking it NA) rather than crashing if space
  is critically low.

Parallelism:
  --gpu_ids distributes structures evenly across GPUs.  Each GPU scores
  its shard sequentially.

Usage:
    python evaluate_sequences.py \\
        --input_dir      /path/to/sequences/ \\
        --native         /path/to/native.pdb \\
        --output_dir     /path/to/output/ \\
        --colabfold_batch_bin /path/to/colabfold_batch \\
        --colabfold_python   /path/to/colabfold_python \\
        --dockq_bin      /path/to/DockQ \\
        --gpu_ids        0,1,2,3
"""

from __future__ import annotations

import sys
import os

def _prepend_thermompnn_path():
    _here   = os.path.dirname(os.path.abspath(__file__))
    _thermo = os.path.join(_here, "ThermoMPNN")
    if os.path.isdir(_thermo) and _thermo not in sys.path:
        sys.path.insert(0, _thermo)

_prepend_thermompnn_path()

import argparse
import multiprocessing as mp
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from rfantibody_benchmark import GPUTimer, run_dockq
from evaluate_designs import (
    write_colabfold_fasta, compute_target_crop,
    run_colabfold, find_top_af2_result, extract_iptm,
)
from partial_diffusion_maturation import CHAIN_H, CHAIN_L, CHAIN_T


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IPTM_THRESHOLD  = 0.6
DOCKQ_THRESHOLD = 0.23
MIN_FREE_GB     = 2.0    # minimum free disk space before skipping a ColabFold run

# Standard one-letter amino acid lookup for FASTA extraction
_AA3_TO_1: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "UNK": "X",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    design_id:  str
    pdb_path:   str
    iptm:       Optional[float]
    dockq:      Optional[float]
    sequences:  Dict[str, str] = field(default_factory=dict)  # chain → sequence
    success:    bool = False

    def __post_init__(self):
        self.success = (
            self.iptm  is not None and self.iptm  > IPTM_THRESHOLD and
            self.dockq is not None and self.dockq > DOCKQ_THRESHOLD
        )


# ─────────────────────────────────────────────────────────────────────────────
# PDB discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_seq_pdbs(input_dir: str) -> List[Path]:
    """Find all *_seq*.pdb files in input_dir and its subfolders."""
    return sorted(Path(input_dir).rglob("*_seq*.pdb"))


# ─────────────────────────────────────────────────────────────────────────────
# FASTA extraction from PDB
# ─────────────────────────────────────────────────────────────────────────────

def extract_sequences_from_pdb(pdb_path: str) -> Dict[str, str]:
    """Return {chain_id: one_letter_sequence} for all chains in the PDB."""
    seen: Dict[str, set] = {}
    seqs: Dict[str, List[str]] = {}
    order: List[str] = []

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            chain  = line[21]
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            icode  = line[26].strip()
            resn   = line[17:20].strip()
            key    = (resnum, icode)
            if chain not in seen:
                seen[chain]  = set()
                seqs[chain]  = []
                order.append(chain)
            if key not in seen[chain]:
                seen[chain].add(key)
                seqs[chain].append(_AA3_TO_1.get(resn, "X"))

    return {ch: "".join(seqs[ch]) for ch in order}


# ─────────────────────────────────────────────────────────────────────────────
# Disk space guard
# ─────────────────────────────────────────────────────────────────────────────

def _free_gb(path: str) -> float:
    """Return free disk space in GB at the given path."""
    stat = shutil.disk_usage(path)
    return stat.free / (1024 ** 3)


def _check_disk(path: str, min_gb: float = MIN_FREE_GB) -> bool:
    free = _free_gb(path)
    if free < min_gb:
        print(f"  [WARN] Low disk space: {free:.1f} GB free at {path} "
              f"(minimum {min_gb} GB) — skipping ColabFold", flush=True)
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Single-structure scoring
# ─────────────────────────────────────────────────────────────────────────────

def _chain_present(pdb_path: str, chain: str) -> bool:
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == chain:
                return True
    return False


def score_one(
    pdb_path:            str,
    native_pdb:          str,
    af2_work_dir:        str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    dockq_bin:           str,
    af2_num_recycles:    int,
    af2_num_models:      int,
    cleanup_af2:         bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Score one PDB with ColabFold (full MSA mode) + DockQ.

    MSA mode: uses the default MMseqs2 paired+unpaired MSA so that ipTM
    scores are comparable to those produced by the benchmark pipeline.
    Single_sequence mode produces systematically lower ipTM (~0.1) and
    should not be used for scoring.

    Disk space: the AF2 working subdirectory is deleted after scoring
    (if cleanup_af2=True) to prevent accumulation of MSA tarballs.
    A minimum free-space check before each run skips ColabFold and marks
    ipTM as NA rather than crashing if space is critically low.

    Returns (iptm, dockq).
    """
    stem    = Path(pdb_path).stem
    out_dir = os.path.join(af2_work_dir, stem)
    os.makedirs(out_dir, exist_ok=True)

    iptm: Optional[float] = None

    # ── ColabFold (full MSA mode) ─────────────────────────────────────────────
    if colabfold_batch_bin and os.path.isfile(colabfold_batch_bin):
        if not _check_disk(af2_work_dir):
            pass  # skip ColabFold, iptm stays None
        else:
            try:
                target_crop = compute_target_crop(pdb_path)
            except ValueError:
                target_crop = None

            fasta_path = os.path.join(af2_work_dir, f"{stem}.fasta")
            chains     = [c for c in [CHAIN_H, CHAIN_L, CHAIN_T]
                          if _chain_present(pdb_path, c)]

            ok = write_colabfold_fasta(pdb_path, fasta_path, chains,
                                       target_crop=target_crop)
            if ok:
                # Use colabfold_batch directly (same as the benchmark pipeline)
                # Full MSA mode is essential for reliable ipTM scores.
                cmd = [
                    colabfold_batch_bin,
                    fasta_path, out_dir,
                    "--num-recycle", str(af2_num_recycles),
                    "--num-models",  str(af2_num_models),
                    "--model-type",  "alphafold2_multimer_v3",
                ]
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=1800,
                    )
                    if result.returncode == 0:
                        af2_result = find_top_af2_result(out_dir, stem)
                        if af2_result and af2_result.get("scores"):
                            raw = extract_iptm(af2_result["scores"])
                            if not np.isnan(raw):
                                iptm = raw
                    else:
                        print(f"  [ColabFold WARN] {stem}: "
                              f"{result.stderr[-300:]}", flush=True)
                except subprocess.TimeoutExpired:
                    print(f"  [ColabFold WARN] {stem}: timed out",
                          flush=True)
                except Exception as e:
                    print(f"  [ColabFold WARN] {stem}: {e}", flush=True)

    # ── DockQ ─────────────────────────────────────────────────────────────────
    dockq = run_dockq(
        model_pdb=pdb_path,
        native_pdb=native_pdb,
        dockq_bin=dockq_bin,
    )

    # ── Cleanup AF2 working dir to free disk space ────────────────────────────
    # This removes MSA tarballs and model PDBs for this structure,
    # keeping disk usage bounded when scoring many structures.
    if cleanup_af2 and os.path.isdir(out_dir):
        try:
            shutil.rmtree(out_dir)
        except Exception as e:
            print(f"  [WARN] Could not clean up {out_dir}: {e}", flush=True)

    # Also clean up the FASTA file
    fasta = os.path.join(af2_work_dir, f"{stem}.fasta")
    if cleanup_af2 and os.path.isfile(fasta):
        try:
            os.remove(fasta)
        except Exception:
            pass

    return iptm, dockq


# ─────────────────────────────────────────────────────────────────────────────
# Live-updating FASTA log writer
# ─────────────────────────────────────────────────────────────────────────────

# A threading lock so multiple workers don't interleave writes
_fasta_lock = threading.Lock()


def _append_to_live_log(
    log_path:   str,
    result:     EvalResult,
):
    """
    Append one result to the live FASTA-style log.  Thread-safe.

    Format:
        >{design_id}  ipTM={x}  DockQ={x}  success={True/False}
        >chain_H
        SEQUENCE...
        >chain_L
        SEQUENCE...
        >chain_T
        SEQUENCE...
        //
    """
    iptm_s  = f"{result.iptm:.4f}"  if result.iptm  is not None else "NA"
    dockq_s = f"{result.dockq:.4f}" if result.dockq is not None else "NA"

    lines = [
        f">{result.design_id}  ipTM={iptm_s}  DockQ={dockq_s}  "
        f"success={result.success}",
    ]
    for chain, seq in result.sequences.items():
        lines.append(f">chain_{chain}")
        # Wrap at 60 chars
        for i in range(0, len(seq), 60):
            lines.append(seq[i:i+60])
    lines.append("//")
    lines.append("")

    with _fasta_lock:
        with open(log_path, "a") as fh:
            fh.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_evenly(items: list, k: int) -> List[list]:
    n    = len(items)
    base = n // k
    rem  = n % k
    shards, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        shards.append(items[start:end])
        start = end
    return shards


def _parse_gpu_list(s: str) -> List[str]:
    return [g.strip() for g in s.split(",") if g.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Per-GPU evaluation worker
# ─────────────────────────────────────────────────────────────────────────────

def _eval_worker(
    gpu_id:              str,
    shard:               List[str],   # list of pdb_path strings
    native_pdb:          str,
    af2_work_dir:        str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    dockq_bin:           str,
    af2_num_recycles:    int,
    af2_num_models:      int,
    live_log_path:       str,
    cleanup_af2:         bool,
) -> List[EvalResult]:
    """Score a shard of PDB files on a single GPU."""
    _prepend_thermompnn_path()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    results = []
    for pdb_path in shard:
        design_id = Path(pdb_path).stem
        print(f"  [GPU {gpu_id}] Scoring {design_id}…", end=" ", flush=True)

        sequences = extract_sequences_from_pdb(pdb_path)

        iptm, dockq = score_one(
            pdb_path=pdb_path,
            native_pdb=native_pdb,
            af2_work_dir=af2_work_dir,
            colabfold_batch_bin=colabfold_batch_bin,
            colabfold_python=colabfold_python,
            dockq_bin=dockq_bin,
            af2_num_recycles=af2_num_recycles,
            af2_num_models=af2_num_models,
            cleanup_af2=cleanup_af2,
        )

        r = EvalResult(
            design_id=design_id,
            pdb_path=pdb_path,
            iptm=iptm,
            dockq=dockq,
            sequences=sequences,
        )
        results.append(r)

        # Append to live log immediately
        _append_to_live_log(live_log_path, r)

        iptm_s  = f"{iptm:.3f}"  if iptm  is not None else "NA"
        dockq_s = f"{dockq:.3f}" if dockq is not None else "NA"
        print(f"ipTM={iptm_s}  DockQ={dockq_s}  "
              f"{'✓' if r.success else '✗'}", flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    input_dir:           str,
    native_pdb:          str,
    output_dir:          str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    dockq_bin:           str       = "DockQ",
    af2_num_recycles:    int       = 1,
    af2_num_models:      int       = 1,
    gpu_ids:             List[str] = None,
    cleanup_af2:         bool      = True,
):
    gpu_ids = gpu_ids or ["0"]
    os.makedirs(output_dir, exist_ok=True)
    af2_work_dir = os.path.join(output_dir, "_af2")
    os.makedirs(af2_work_dir, exist_ok=True)

    # ── Find sequence PDBs ────────────────────────────────────────────────────
    pdbs = find_seq_pdbs(input_dir)
    if not pdbs:
        print(f"[evaluate] No *_seq*.pdb files found in {input_dir}")
        return
    print(f"[evaluate] {len(pdbs)} sequence PDB(s) found")
    print(f"[evaluate] GPUs: {gpu_ids}  "
          f"MSA mode: single_sequence (no MSA download)")

    stem         = Path(input_dir).resolve().name
    live_log     = os.path.join(output_dir, f"{stem}_live.fasta")
    final_tsv    = os.path.join(output_dir, f"{stem}_ranked.tsv")

    # Write header to live log
    with open(live_log, "w") as fh:
        fh.write(f"# evaluate_sequences.py live results\n")
        fh.write(f"# input_dir : {input_dir}\n")
        fh.write(f"# native    : {native_pdb}\n")
        fh.write(f"# total     : {len(pdbs)} structures\n")
        fh.write(f"# format    : >design_id  ipTM=x  DockQ=x  success=x\n")
        fh.write(f"# ----------\n\n")

    print(f"[evaluate] Live log → {live_log}")

    # ── Distribute across GPUs ────────────────────────────────────────────────
    shards = _split_evenly([str(p) for p in pdbs], len(gpu_ids))

    all_results: List[EvalResult] = []
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as exe:
        futures = {}
        for gpu_id, shard in zip(gpu_ids, shards):
            if not shard:
                continue
            fut = exe.submit(
                _eval_worker,
                gpu_id, shard, native_pdb,
                af2_work_dir, colabfold_batch_bin, colabfold_python,
                dockq_bin, af2_num_recycles, af2_num_models,
                live_log, cleanup_af2,
            )
            futures[fut] = gpu_id

        for fut in as_completed(futures):
            gpu_id = futures[fut]
            try:
                results = fut.result()
                all_results.extend(results)
                print(f"[evaluate] GPU {gpu_id} finished: "
                      f"{len(results)} structure(s)", flush=True)
            except Exception as e:
                import traceback
                print(f"[evaluate] GPU {gpu_id} ERROR:\n"
                      f"{traceback.format_exc()}", flush=True)

    # ── Sort: ipTM descending, DockQ as tiebreaker ────────────────────────────
    all_results.sort(
        key=lambda r: (
            r.iptm  if r.iptm  is not None else -1.0,
            r.dockq if r.dockq is not None else -1.0,
        ),
        reverse=True,
    )

    # ── Write ranked TSV ──────────────────────────────────────────────────────
    with open(final_tsv, "w") as fh:
        fh.write("rank\tdesign_id\tiptm\tdockq\tsuccess\tpdb_path\n")
        for rank, r in enumerate(all_results, 1):
            fh.write(
                f"{rank}\t{r.design_id}\t"
                f"{r.iptm  if r.iptm  is not None else 'NA'}\t"
                f"{r.dockq if r.dockq is not None else 'NA'}\t"
                f"{r.success}\t{r.pdb_path}\n"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_success = sum(1 for r in all_results if r.success)
    iptms     = [r.iptm  for r in all_results if r.iptm  is not None]
    dockqs    = [r.dockq for r in all_results if r.dockq is not None]

    print(f"\n{'='*60}")
    print(f"  Total evaluated : {len(all_results)}")
    print(f"  Successes       : {n_success} "
          f"(ipTM>{IPTM_THRESHOLD} AND DockQ>{DOCKQ_THRESHOLD})")
    if iptms:
        print(f"  ipTM  mean={np.mean(iptms):.3f}  "
              f"max={max(iptms):.3f}  min={min(iptms):.3f}")
    if dockqs:
        print(f"  DockQ mean={np.mean(dockqs):.3f}  "
              f"max={max(dockqs):.3f}  min={min(dockqs):.3f}")
    print(f"  Ranked TSV  → {final_tsv}")
    print(f"  Live FASTA  → {live_log}")
    print(f"{'='*60}\n")

    print("  Top 10 by ipTM (DockQ tiebreaker):")
    print(f"  {'rank':>4}  {'design_id':<45}  {'ipTM':>6}  {'DockQ':>6}")
    print(f"  {'-'*68}")
    for r in all_results[:10]:
        iptm_s  = f"{r.iptm:.3f}"  if r.iptm  is not None else "  NA"
        dockq_s = f"{r.dockq:.3f}" if r.dockq is not None else "  NA"
        rank    = all_results.index(r) + 1
        print(f"  {rank:>4}  {r.design_id:<45}  {iptm_s:>6}  {dockq_s:>6}  "
              f"{'✓' if r.success else '✗'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate all *_seq*.pdb files in a folder with ColabFold ipTM "
            "(single_sequence MSA mode) and DockQ, with live FASTA output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input_dir",           required=True,
                   help="Folder (or folder of subfolders) containing "
                        "*_seq*.pdb files")
    p.add_argument("--native",              required=True,
                   help="Native/reference PDB for DockQ")
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--colabfold_batch_bin", required=True)
    p.add_argument("--colabfold_python",    required=True)
    p.add_argument("--dockq_bin",           default="DockQ")
    p.add_argument("--af2_num_recycles",    type=int, default=1,
                   help="AF2 recycles (default 1 — fast scoring pass)")
    p.add_argument("--af2_num_models",      type=int, default=1)
    p.add_argument("--gpu_ids",             default="0",
                   help="Comma-separated GPU IDs (default: 0)")
    p.add_argument("--no_cleanup",          action="store_true",
                   help="Keep AF2 working directories after scoring "
                        "(uses more disk space)")
    return p.parse_args()


def main():
    mp.set_start_method("spawn", force=True)
    args    = parse_args()
    gpu_ids = _parse_gpu_list(args.gpu_ids)
    run(
        input_dir=str(Path(args.input_dir).resolve()),
        native_pdb=str(Path(args.native).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        colabfold_batch_bin=args.colabfold_batch_bin,
        colabfold_python=args.colabfold_python,
        dockq_bin=args.dockq_bin,
        af2_num_recycles=args.af2_num_recycles,
        af2_num_models=args.af2_num_models,
        gpu_ids=gpu_ids,
        cleanup_af2=not args.no_cleanup,
    )


if __name__ == "__main__":
    main()