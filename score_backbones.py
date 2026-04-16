"""
score_backbones.py

For every backbone PDB in a folder (excluding *_seq.pdb and *_grafted.pdb),
generate N sequences with ProteinMPNN, score each with ColabFold ipTM and
DockQ, and write results sorted by ipTM descending.

Pipeline per backbone:
  1. ProteinMPNN sequence design (N sequences → N *_seq_<i>.pdb files)
  2. ColabFold ipTM scoring
  3. DockQ scoring against a native/reference PDB
  4. Results merged and sorted by ipTM descending → TSV output

Usage:
    python score_backbones.py \
        --input_dir      /path/to/backbones/ \
        --native         /path/to/native.pdb \
        --output_dir     /path/to/output/ \
        --n_seqs         10 \
        --mpnn_weights   /path/to/v_48_020.pt \
        --colabfold_batch_bin /path/to/colabfold_batch \
        --colabfold_python   /path/to/colabfold_python \
        --dockq_bin      DockQ \
        --device         cuda
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── shared utilities from the RFantibody pipeline ────────────────────────────
from rfantibody_benchmark import (
    DesignResult, GPUTimer, score_design, IPTM_SUCCESS_THRESHOLD,
)
from partial_diffusion_maturation import (
    split_hlt_complex, parse_hlt_remarks, read_pdb_residues,
    CHAIN_H, CHAIN_L, CHAIN_T,
)
from smc_denovo_maturation import (
    build_cdr_mask, load_epitope_ca, load_proteinmpnn,
)
from beam_denovo_maturation_complexa import _apply_sequence_and_anchors


# ─────────────────────────────────────────────────────────────────────────────
# PDB discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_backbone_pdbs(input_dir: str) -> List[Path]:
    """
    Return all .pdb files in input_dir that are NOT sequence designs
    (_seq.pdb), grafted structures (_grafted.pdb), or trajectory files.

    Also excludes common RFdiffusion auxiliary outputs:
        *_traj.pdb, *_pX0*.pdb
    """
    excluded = re.compile(
        r"(_seq\.pdb|_seq_\d+\.pdb|_grafted\.pdb|_traj\.pdb|_pX0.*\.pdb)$"
    )
    pdbs = sorted(
        p for p in Path(input_dir).rglob("*.pdb")
        if not excluded.search(p.name)
    )
    return pdbs


# ─────────────────────────────────────────────────────────────────────────────
# Sequence generation: ProteinMPNN → N designed PDBs per backbone
# ─────────────────────────────────────────────────────────────────────────────

def generate_sequences(
    backbone_pdb: str,
    n_seqs:       int,
    mpnn,
    cdr_mask,
    output_dir:   str,
    device:       str,
    temperature:  float = 0.2,
) -> List[str]:
    """
    Run ProteinMPNN N times on backbone_pdb and return paths to the
    designed PDB files (one per sequence).
    """
    stem     = Path(backbone_pdb).stem
    seq_dir  = os.path.join(output_dir, "_sequences", stem)
    os.makedirs(seq_dir, exist_ok=True)

    designed_pdbs = []
    for i in range(n_seqs):
        out_prefix = os.path.join(seq_dir, f"{stem}_seq_{i:03d}")
        out_pdb    = out_prefix + ".pdb"

        if os.path.exists(out_pdb):
            # Resume: skip if already generated
            designed_pdbs.append(out_pdb)
            continue

        result = _apply_sequence_and_anchors(
            pdb_path=backbone_pdb,
            out_prefix=out_prefix,
            mpnn=mpnn,
            cdr_mask=cdr_mask,
            anchor_residues=[],
            ref_pdb=backbone_pdb,
            device=device,
        )
        if result and os.path.exists(result):
            designed_pdbs.append(result)
        else:
            print(f"  [WARN] Sequence {i} failed for {Path(backbone_pdb).name}")

    return designed_pdbs


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_pdb(
    pdb_path:            str,
    native_pdb:          str,
    af2_work_dir:        str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    dockq_bin:           str,
    af2_num_recycles:    int,
    af2_num_models:      int,
    device:              str,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (iptm, dockq) for a single PDB."""
    timer = GPUTimer()
    iptm, dockq = score_design(
        pdb_path=pdb_path,
        af2_work_dir=af2_work_dir,
        native_pdb=native_pdb,
        colabfold_batch_bin=colabfold_batch_bin,
        colabfold_python=colabfold_python,
        timer=timer,
        af2_num_recycles=af2_num_recycles,
        af2_num_models=af2_num_models,
        device=device,
    )
    return iptm, dockq


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class BackboneResult:
    backbone:    str           # backbone PDB stem
    seq_idx:     int           # sequence index (0-based)
    design_id:   str           # backbone_stem + seq index
    pdb_path:    str
    iptm:        Optional[float]
    dockq:       Optional[float]
    success:     bool = False

    def __post_init__(self):
        self.success = (
            self.iptm  is not None and self.iptm  > 0.6 and
            self.dockq is not None and self.dockq > 0.23
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(
    input_dir:           str,
    native_pdb:          str,
    output_dir:          str,
    n_seqs:              int,
    mpnn_weights:        str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    dockq_bin:           str        = "DockQ",
    af2_num_recycles:    int        = 3,
    af2_num_models:      int        = 1,
    device:              str        = "cuda",
    temperature:         float      = 0.2,
    framework_pdb:       str        = "",
    hotspots:            str        = "",
):
    os.makedirs(output_dir, exist_ok=True)
    af2_work_dir = os.path.join(output_dir, "_af2")
    os.makedirs(af2_work_dir, exist_ok=True)

    # ── Find backbone PDBs ────────────────────────────────────────────────────
    backbones = find_backbone_pdbs(input_dir)
    if not backbones:
        print(f"[score_backbones] No backbone PDBs found in {input_dir}")
        return
    print(f"[score_backbones] Found {len(backbones)} backbone PDB(s)")

    # ── Load ProteinMPNN and CDR mask ─────────────────────────────────────────
    print("[score_backbones] Loading ProteinMPNN…")
    mpnn = load_proteinmpnn(mpnn_weights, device)

    # Build CDR mask from framework PDB if provided, else from first backbone
    mask_source = framework_pdb if framework_pdb else str(backbones[0])
    cdr_mask = build_cdr_mask(mask_source)

    # ── Process each backbone ─────────────────────────────────────────────────
    all_results: List[BackboneResult] = []
    total_backbones = len(backbones)

    for b_idx, backbone in enumerate(backbones, 1):
        b_stem = backbone.stem
        print(f"\n[{b_idx:>4}/{total_backbones}] Backbone: {backbone.name}")

        # Generate N sequences
        print(f"  Generating {n_seqs} sequence(s)…")
        designed_pdbs = generate_sequences(
            backbone_pdb=str(backbone),
            n_seqs=n_seqs,
            mpnn=mpnn,
            cdr_mask=cdr_mask,
            output_dir=output_dir,
            device=device,
            temperature=temperature,
        )
        print(f"  {len(designed_pdbs)} sequence(s) generated")

        # Score each designed sequence
        for seq_idx, pdb in enumerate(designed_pdbs):
            design_id = f"{b_stem}_seq_{seq_idx:03d}"
            print(f"  Scoring {design_id}…", end=" ", flush=True)

            iptm, dockq = score_pdb(
                pdb_path=pdb,
                native_pdb=native_pdb,
                af2_work_dir=af2_work_dir,
                colabfold_batch_bin=colabfold_batch_bin,
                colabfold_python=colabfold_python,
                dockq_bin=dockq_bin,
                af2_num_recycles=af2_num_recycles,
                af2_num_models=af2_num_models,
                device=device,
            )

            r = BackboneResult(
                backbone=b_stem,
                seq_idx=seq_idx,
                design_id=design_id,
                pdb_path=pdb,
                iptm=iptm,
                dockq=dockq,
            )
            all_results.append(r)

            iptm_s  = f"{iptm:.3f}"  if iptm  is not None else "NA"
            dockq_s = f"{dockq:.3f}" if dockq is not None else "NA"
            print(f"ipTM={iptm_s}  DockQ={dockq_s}  "
                  f"{'✓' if r.success else '✗'}")

    # ── Sort by ipTM descending ───────────────────────────────────────────────
    all_results.sort(
        key=lambda r: r.iptm if r.iptm is not None else -1.0,
        reverse=True,
    )

    # ── Write TSV ─────────────────────────────────────────────────────────────
    stem      = Path(input_dir).resolve().name
    tsv_path  = os.path.join(output_dir, f"{stem}_scored.tsv")
    with open(tsv_path, "w") as fh:
        fh.write("design_id\tbackbone\tseq_idx\tiptm\tdockq\tsuccess\tpdb_path\n")
        for r in all_results:
            fh.write(
                f"{r.design_id}\t{r.backbone}\t{r.seq_idx}\t"
                f"{r.iptm if r.iptm is not None else 'NA'}\t"
                f"{r.dockq if r.dockq is not None else 'NA'}\t"
                f"{r.success}\t{r.pdb_path}\n"
            )

    # ── Print summary ─────────────────────────────────────────────────────────
    n_success = sum(1 for r in all_results if r.success)
    iptms     = [r.iptm  for r in all_results if r.iptm  is not None]
    dockqs    = [r.dockq for r in all_results if r.dockq is not None]

    print(f"\n{'='*60}")
    print(f"  Total designs scored : {len(all_results)}")
    print(f"  Successes            : {n_success} "
          f"(ipTM>0.6 AND DockQ>0.23)")
    if iptms:
        print(f"  ipTM   mean={np.mean(iptms):.3f}  "
              f"max={max(iptms):.3f}  min={min(iptms):.3f}")
    if dockqs:
        print(f"  DockQ  mean={np.mean(dockqs):.3f}  "
              f"max={max(dockqs):.3f}  min={min(dockqs):.3f}")
    print(f"  Results → {tsv_path}")
    print(f"{'='*60}\n")

    # ── Print top 10 ──────────────────────────────────────────────────────────
    print("  Top designs by ipTM:")
    print(f"  {'design_id':<40}  {'ipTM':>6}  {'DockQ':>6}  {'success':>7}")
    print(f"  {'-'*65}")
    for r in all_results[:10]:
        iptm_s  = f"{r.iptm:.3f}"  if r.iptm  is not None else "  NA"
        dockq_s = f"{r.dockq:.3f}" if r.dockq is not None else "  NA"
        print(f"  {r.design_id:<40}  {iptm_s:>6}  {dockq_s:>6}  "
              f"{'✓' if r.success else '✗':>7}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Score backbone PDBs: generate N sequences per backbone with "
            "ProteinMPNN, then score with ColabFold ipTM and DockQ."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input_dir",           required=True,
                   help="Folder to search for backbone PDB files")
    p.add_argument("--native",              required=True,
                   help="Native/reference PDB for DockQ scoring")
    p.add_argument("--output_dir",          required=True,
                   help="Output directory for sequences and results TSV")
    p.add_argument("--n_seqs",              type=int, default=10,
                   help="Number of sequences to generate per backbone (default: 10)")
    p.add_argument("--mpnn_weights",        required=True,
                   help="Path to ProteinMPNN weights (.pt)")
    p.add_argument("--colabfold_batch_bin", required=True,
                   help="Path to colabfold_batch executable")
    p.add_argument("--colabfold_python",    required=True,
                   help="Python interpreter inside ColabFold conda env")
    p.add_argument("--dockq_bin",           default="DockQ",
                   help="DockQ executable (default: DockQ on PATH)")
    p.add_argument("--af2_num_recycles",    type=int, default=3,
                   help="AF2 recycles per structure (default: 3)")
    p.add_argument("--af2_num_models",      type=int, default=1,
                   help="Number of AF2 models to run (default: 1)")
    p.add_argument("--device",              default="cuda")
    p.add_argument("--temperature",         type=float, default=0.2,
                   help="ProteinMPNN sampling temperature (default: 0.2)")
    p.add_argument("--framework_pdb",       default="",
                   help="Optional: PDB to build CDR mask from. "
                        "Defaults to first backbone found.")
    p.add_argument("--hotspots",            default="",
                   help="Comma-separated hotspot residues (e.g. T45,T67). "
                        "Used for epitope_ca loading if needed.")
    return p.parse_args()


def main():
    args = parse_args()
    run(
        input_dir=args.input_dir,
        native_pdb=str(Path(args.native).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        n_seqs=args.n_seqs,
        mpnn_weights=args.mpnn_weights,
        colabfold_batch_bin=args.colabfold_batch_bin,
        colabfold_python=args.colabfold_python,
        dockq_bin=args.dockq_bin,
        af2_num_recycles=args.af2_num_recycles,
        af2_num_models=args.af2_num_models,
        device=args.device,
        temperature=args.temperature,
        framework_pdb=args.framework_pdb,
        hotspots=args.hotspots,
    )


if __name__ == "__main__":
    main()