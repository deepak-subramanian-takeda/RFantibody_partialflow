"""
rfantibody_benchmark.py

Direct comparison of four RFantibody design strategies:

  A) Vanilla RFantibody (baseline)
  B) RFantibody + anchor conservation  (partial_T=50, provide_seq lock)
  C) Beam search (ipTM + ThermoMPNN DDG), no anchor conservation
  D) Beam search (ipTM + ThermoMPNN DDG) + anchor conservation

Success metric: DockQ > 0.23  AND  ipTM > 0.6
Efficiency metric: successes per GPU-hour

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design choices and rationale
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Vanilla: full de novo RFdiffusion + ProteinMPNN sequence design.
  No constraints; provides the reference success rate and GPU cost.

- Anchored (partial_T=50): uses provide_seq to fix framework + anchor
  backbone frames.  partial_T=50 means the full noise schedule is used,
  but anchored positions are never updated — effectively a constrained
  de novo run.  Same GPU cost as vanilla per design.

- Beam (no anchors): the Complexa-style beam search from snippet 2.
  Each rollout is a full RFdiffusion run; reward = w_iptm*ipTM +
  w_thermo*(-DDG).  Requires N*L ColabFold calls per checkpoint, so
  GPU cost per *successful* design is higher but quality is better
  gated.

- Beam + anchors: beam search with provide_seq anchor locking.
  Combines the quality filtering of beam search with the conservation
  of experimentally confirmed anchor contacts.

GPU timing: wall-clock seconds are captured around each subprocess call
(RFdiffusion, ColabFold) via time.perf_counter().  At the end, GPU
allocation time is converted to GPU-hours assuming a single GPU is in use
throughout (i.e. wall-clock ≈ GPU time).  If multiple GPUs are used in
parallel the caller should divide by the number of GPUs.

DockQ: calls the external DockQ binary.  Requires DockQ installed and
available on PATH (https://github.com/bjornwallner/DockQ).

ipTM: extracted from ColabFold rank_001 scores JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── imports from the two source files provided ────────────────────────────────
# partial_diffusion_maturation.py (snippet 1)
from partial_diffusion_maturation import (
    CHAIN_H, CHAIN_L, CHAIN_T,
    CdrRange, ResidueInfo,
    parse_hlt_remarks,
    read_pdb_residues,
    build_residue_lookup,
    split_hlt_complex,
    build_contig_string,
    build_provide_seq,
    parse_free_loops,
    load_anchors,
    mask_anchors_in_hlt,
    graft_target_sequence,
    build_rfdiffusion_command,
)

# beam_denovo_maturation_complexa.py (snippet 2)
from beam_denovo_maturation_complexa import (
    BeamNode,
    RANKING_MODES,
    score_complexa_reward,
    _rollout_and_score,
    _apply_sequence_and_anchors,
    _print_beam,
    write_renumbered_pdb,
    run_af2_multimer,
    IPTM_SUCCESS_THRESHOLD,
)

# smc_denovo_maturation.py (shared utilities used by snippet 2)
from smc_denovo_maturation import (
    build_denovo_contig,
    build_cdr_mask,
    load_epitope_ca,
    load_thermompnn,
    load_proteinmpnn,
    design_sequence_onto_backbone,
    graft_anchor_identities,
    pack_sidechains,
    run_denovo_round,
)

# evaluate_designs.py (ColabFold runner shared by both snippets)
from evaluate_designs import (
    write_colabfold_fasta,
    _get_chain_sequence_range,
    compute_target_crop,
    run_colabfold,
    find_top_af2_result,
    extract_plddt,
    extract_iptm,
    BINDER_CHAINS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DesignResult:
    arm:          str            # A / B / C / D
    design_id:    str
    pdb_path:     str
    iptm:         Optional[float]
    dockq:        Optional[float]
    ddg:          Optional[float]
    gpu_seconds:  float          # wall-clock GPU seconds consumed for this design
    success:      bool = False   # DockQ > 0.23 AND ipTM > 0.6

    def __post_init__(self):
        self.success = (
            self.iptm  is not None and self.iptm  > 0.6 and
            self.dockq is not None and self.dockq > 0.23
        )


@dataclass
class ArmSummary:
    arm:               str
    n_designs:         int
    n_success:         int
    total_gpu_hours:   float
    success_per_gpu_h: float
    mean_iptm:         float
    mean_dockq:        float
    results:           List[DesignResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# GPU-wall-clock timer context manager
# ─────────────────────────────────────────────────────────────────────────────

class GPUTimer:
    """Accumulates wall-clock time used as a proxy for single-GPU hours."""

    def __init__(self):
        self._total: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self._start is not None:
            self._total += time.perf_counter() - self._start
            self._start = None

    def tick(self) -> float:
        """Return seconds elapsed in the current (open) segment."""
        if self._start is not None:
            return time.perf_counter() - self._start
        return 0.0

    @property
    def total_seconds(self) -> float:
        return self._total

    @property
    def total_hours(self) -> float:
        return self._total / 3600.0


# ─────────────────────────────────────────────────────────────────────────────
# DockQ wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_dockq(
    model_pdb:    str,
    native_pdb:   str,
    binder_chain: str = CHAIN_H,
    target_chain: str = CHAIN_T,
    dockq_bin:    str = "DockQ",
) -> Optional[float]:
    """
    Run DockQ v2 and return the scalar DockQ score for the
    binder–target interface.

    DockQ v2 CLI (pip install DockQ):
        DockQ <model> <native> [--mapping MODELCHAINS:NATIVECHAINS]

    Chain mapping:
      - For a full Fv (H+L vs T): --mapping HLT:HLT
      - For a nanobody (H vs T):  --mapping HT:HT
      The mapping is the same for model and native because RFantibody
      preserves chain IDs.  DockQ v2 will auto-detect interfaces within
      those chains (H–T and L–T) and return a GlobalDockQ.

    Output (--short) format varies by v2 minor version; we try several
    patterns to be robust across releases.
    """
    # Build the chain mapping string from whichever chains are present
    # in the model.  We always include T (target); add H and L if present.
    present = []
    for ch in [binder_chain, CHAIN_L, target_chain]:
        if ch != binder_chain and ch != target_chain and ch == CHAIN_L:
            # only include L if it's actually present in the model PDB
            if _chain_present(model_pdb, ch):
                present.append(ch)
        else:
            present.append(ch)

    # Build ordered chain string: H (+ L if present) + T
    chain_order = ""
    for ch in [CHAIN_H, CHAIN_L, CHAIN_T]:
        if ch == CHAIN_L:
            if _chain_present(model_pdb, CHAIN_L):
                chain_order += ch
        else:
            chain_order += ch
    # mapping is identical for model and native (chain IDs preserved)
    mapping = f"{chain_order}:{chain_order}"

    cmd = [dockq_bin, model_pdb, native_pdb, "--short", "--mapping", mapping]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                      text=True, timeout=120)
    except subprocess.CalledProcessError as e:
        print(f"  [DockQ WARN] {e}\n  stdout: {e.output[:300]}")
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  [DockQ WARN] {e}")
        return None

    # v2 --short output patterns (varies across minor releases):
    #   "Total DockQ over N native interfaces: 0.XXXX ..."
    #   "GlobalDockQ 0.XXXX"
    #   "DockQ 0.XXXX"                          (single interface)
    for pattern in [
        r"Total DockQ[^:]*:\s*([0-9.]+)",
        r"GlobalDockQ\s+([0-9.]+)",
        r"DockQ\s+([0-9.]+)",
    ]:
        m = re.search(pattern, out)
        if m:
            return float(m.group(1))

    print(f"  [DockQ WARN] Could not parse score from output:\n{out[:400]}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ipTM extraction helper (thin wrapper around evaluate_designs.py)
# ─────────────────────────────────────────────────────────────────────────────

def score_design(
    pdb_path:            str,
    af2_work_dir:        str,
    native_pdb:          str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    timer:               GPUTimer,
    af2_num_recycles:    int = 3,
    af2_num_models:      int = 1,
    use_gpu:             bool = True,
    binder_chain:        str = CHAIN_H,
    target_chain:        str = CHAIN_T,
    dockq_bin:           str = "DockQ",
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (iptm, dockq).

    GPU timer is open while ColabFold runs; DockQ is CPU-only and is not
    counted against GPU time (though it is fast and the effect is negligible).
    """
    stem = Path(pdb_path).stem
    out_dir = os.path.join(af2_work_dir, stem)
    os.makedirs(out_dir, exist_ok=True)

    # ── ipTM via ColabFold ────────────────────────────────────────────────────
    iptm: Optional[float] = None
    if colabfold_batch_bin and os.path.isfile(colabfold_batch_bin):
        try:
            target_crop = compute_target_crop(pdb_path)
        except ValueError:
            target_crop = None

        fasta_path = os.path.join(af2_work_dir, f"{stem}.fasta")
        chains = [c for c in [CHAIN_H, CHAIN_L, CHAIN_T]
                  if _chain_present(pdb_path, c)]
        ok = write_colabfold_fasta(pdb_path, fasta_path, chains,
                                   target_crop=target_crop)
        if ok:
            with timer:
                success = run_colabfold(
                    fasta_path=fasta_path,
                    af2_out_dir=out_dir,
                    colabfold_batch_bin=colabfold_batch_bin,
                    colabfold_python=colabfold_python,
                    num_recycles=af2_num_recycles,
                    num_models=af2_num_models,
                    use_gpu=use_gpu,
                )
            if success:
                result = find_top_af2_result(out_dir, stem)
                if result and result["scores"]:
                    raw = extract_iptm(result["scores"])
                    iptm = None if np.isnan(raw) else raw

    # ── DockQ (CPU) ───────────────────────────────────────────────────────────
    dockq = run_dockq(
        model_pdb=pdb_path,
        native_pdb=native_pdb,
        binder_chain=binder_chain,
        target_chain=target_chain,
        dockq_bin=dockq_bin,
    )

    return iptm, dockq


def _chain_present(pdb_path: str, chain: str) -> bool:
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == chain:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Shared RFdiffusion launcher (returns list of output PDB paths)
# ─────────────────────────────────────────────────────────────────────────────

def _run_rfdiffusion(
    cmd:         List[str],
    output_dir:  str,
    output_prefix: str,
    timer:       GPUTimer,
) -> List[str]:
    """Launch rfdiffusion subprocess under GPU timer; return output PDBs."""
    shell_cmd = " ".join(cmd)
    print(f"  $ {shell_cmd[:160]}{'…' if len(shell_cmd) > 160 else ''}")
    with timer:
        result = subprocess.run(shell_cmd, shell=True)
    if result.returncode != 0:
        print(f"  [WARN] rfdiffusion exited with code {result.returncode}")
    name = Path(output_prefix).name
    return sorted(Path(output_dir).glob(f"{name}*.pdb"))


# ─────────────────────────────────────────────────────────────────────────────
# ProteinMPNN sequence design helper (shared by arms A and B)
# ─────────────────────────────────────────────────────────────────────────────

def _design_and_graft(
    backbone_pdb:    str,
    mpnn,
    cdr_mask,
    anchor_residues: list,
    ref_pdb:         str,
    out_prefix:      str,
    device:          str,
) -> str:
    """ProteinMPNN sequence design + re-graft anchor identities."""
    return _apply_sequence_and_anchors(
        pdb_path=backbone_pdb,
        out_prefix=out_prefix,
        mpnn=mpnn,
        cdr_mask=cdr_mask,
        anchor_residues=anchor_residues,
        ref_pdb=ref_pdb,
        device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ARM A — Vanilla RFantibody
# ─────────────────────────────────────────────────────────────────────────────

def run_arm_A(
    input_pdb:           str,
    output_dir:          str,
    hotspots:            str,
    model_weights:       str,
    num_designs:         int,
    native_pdb:          str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    af2_work_dir:        str,
    mpnn,
    cdr_mask,
    framework_pdb:       str,
    extra_args:          List[str],
    device:              str,
    dockq_bin:           str = "DockQ",
    af2_num_recycles:    int = 3,
    af2_num_models:      int = 1,
    nanobody:            bool = False,
    free_loops:          Dict = None,
) -> List[DesignResult]:
    """
    Vanilla RFantibody: full de novo generation with no anchor constraints.

    Pipeline per design:
      1. RFdiffusion (full noise schedule, no provide_seq)
      2. ProteinMPNN sequence design
      3. ColabFold ipTM  +  DockQ
    """
    free_loops = free_loops or {}
    arm_dir    = os.path.join(output_dir, "arm_A_vanilla")
    os.makedirs(arm_dir, exist_ok=True)
    timer      = GPUTimer()
    results: List[DesignResult] = []

    print("\n" + "=" * 60)
    print("ARM A — Vanilla RFantibody")
    print("=" * 60)

    cdr_ranges = parse_hlt_remarks(input_pdb)
    residues   = read_pdb_residues(input_pdb)

    # No anchor constraints for vanilla arm
    anchor_residues: List = []
    contig_string = build_contig_string(
        residues=residues,
        cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues,
        free_loop_overrides=free_loops,
        nanobody=nanobody,
    )
    print(f"  Contig: {contig_string}")

    output_prefix = os.path.join(arm_dir, "vanilla")
    cmd = build_rfdiffusion_command(
        input_pdb=input_pdb,
        target_pdb="",
        framework_pdb=framework_pdb,
        contig_string=contig_string,
        provide_seq="",          # ← no provide_seq
        hotspots=hotspots,
        output_prefix=output_prefix,
        partial_T=50,            # full noise schedule
        num_designs=num_designs,
        model_weights=model_weights,
        extra_args=extra_args,
    )

    out_pdbs = _run_rfdiffusion(cmd, arm_dir, output_prefix, timer)
    rfd_gpu_s = timer.total_seconds

    # Distribute RFdiffusion cost equally across all designs
    rfd_per_design = rfd_gpu_s / max(len(out_pdbs), 1)

    for pdb in out_pdbs:
        t0   = time.perf_counter()
        dsgn = _design_and_graft(
            backbone_pdb=str(pdb),
            mpnn=mpnn, cdr_mask=cdr_mask,
            anchor_residues=[],
            ref_pdb=input_pdb,
            out_prefix=str(pdb).replace(".pdb", ""),
            device=device,
        )
        mpnn_s = time.perf_counter() - t0

        af2_timer = GPUTimer()
        iptm, dockq = score_design(
            pdb_path=dsgn, af2_work_dir=af2_work_dir,
            native_pdb=native_pdb,
            colabfold_batch_bin=colabfold_batch_bin,
            colabfold_python=colabfold_python,
            timer=af2_timer,
            af2_num_recycles=af2_num_recycles,
            af2_num_models=af2_num_models,
            dockq_bin=dockq_bin,
        )

        gpu_s = rfd_per_design + mpnn_s + af2_timer.total_seconds
        r = DesignResult(
            arm="A", design_id=Path(pdb).stem,
            pdb_path=dsgn, iptm=iptm, dockq=dockq,
            ddg=None, gpu_seconds=gpu_s,
        )
        results.append(r)
        _log_result(r)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ARM B — Anchored RFantibody  (partial_T = 50, provide_seq locked)
# ─────────────────────────────────────────────────────────────────────────────

def run_arm_B(
    input_pdb:           str,
    anchors_json:        str,
    output_dir:          str,
    hotspots:            str,
    model_weights:       str,
    num_designs:         int,
    native_pdb:          str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    af2_work_dir:        str,
    mpnn,
    cdr_mask,
    framework_pdb:       str,
    extra_args:          List[str],
    device:              str,
    dockq_bin:           str = "DockQ",
    af2_num_recycles:    int = 3,
    af2_num_models:      int = 1,
    nanobody:            bool = False,
    free_loops:          Dict = None,
) -> List[DesignResult]:
    """
    Anchored RFantibody: full noise schedule (partial_T=50) with
    provide_seq locking framework + anchor positions.

    Mirrors the logic in partial_diffusion_maturation.py (snippet 1).
    The anchor REMARK lines are masked so AbSampler sees anchors as
    framework; provide_seq then enforces their backbone frames.
    """
    free_loops = free_loops or {}
    arm_dir    = os.path.join(output_dir, "arm_B_anchored")
    os.makedirs(arm_dir, exist_ok=True)
    timer      = GPUTimer()
    results: List[DesignResult] = []

    print("\n" + "=" * 60)
    print("ARM B — Anchored RFantibody  (partial_T=50 + provide_seq)")
    print("=" * 60)

    cdr_ranges      = parse_hlt_remarks(input_pdb)
    residues        = read_pdb_residues(input_pdb)
    anchor_residues = load_anchors(anchors_json)
    print(f"  Anchors ({len(anchor_residues)}): "
          f"{[f'{c}{n}' for c, n in anchor_residues]}")

    # Mask anchor REMARK lines (see snippet 1 rationale)
    stem       = Path(input_pdb).stem
    masked_pdb = os.path.join(arm_dir, f"{stem}_anchors_masked.pdb")
    mask_anchors_in_hlt(
        input_pdb=input_pdb,
        anchor_residues=anchor_residues,
        out_path=masked_pdb,
    )

    provide_seq   = build_provide_seq(
        residues=residues, cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues, nanobody=nanobody,
    )
    contig_string = build_contig_string(
        residues=residues, cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues,
        free_loop_overrides=free_loops, nanobody=nanobody,
    )
    print(f"  Contig     : {contig_string}")
    n_fixed = len(provide_seq.split(",")) if provide_seq else 0
    print(f"  provide_seq: {n_fixed} residue(s) fixed")

    output_prefix = os.path.join(arm_dir, f"{stem}_anchored_T50")
    cmd = build_rfdiffusion_command(
        input_pdb=masked_pdb,
        target_pdb="",
        framework_pdb=framework_pdb,
        contig_string=contig_string,
        provide_seq=provide_seq,   # ← anchor + framework locked
        hotspots=hotspots,
        output_prefix=output_prefix,
        partial_T=50,              # full noise schedule (de novo with constraints)
        num_designs=num_designs,
        model_weights=model_weights,
        extra_args=extra_args,
    )

    out_pdbs  = _run_rfdiffusion(cmd, arm_dir, output_prefix, timer)
    rfd_gpu_s = timer.total_seconds
    rfd_per   = rfd_gpu_s / max(len(out_pdbs), 1)

    for pdb in out_pdbs:
        # Graft original target chain back (anchor positions may have drifted)
        grafted = str(pdb).replace(".pdb", "_grafted.pdb")
        graft_target_sequence(
            rfdiffusion_pdb=str(pdb),
            original_target=input_pdb,
            input_pdb=masked_pdb,
            out_path=grafted,
        )

        t0   = time.perf_counter()
        dsgn = _design_and_graft(
            backbone_pdb=grafted,
            mpnn=mpnn, cdr_mask=cdr_mask,
            anchor_residues=anchor_residues,
            ref_pdb=input_pdb,
            out_prefix=grafted.replace(".pdb", ""),
            device=device,
        )
        mpnn_s = time.perf_counter() - t0

        af2_timer = GPUTimer()
        iptm, dockq = score_design(
            pdb_path=dsgn, af2_work_dir=af2_work_dir,
            native_pdb=native_pdb,
            colabfold_batch_bin=colabfold_batch_bin,
            colabfold_python=colabfold_python,
            timer=af2_timer,
            af2_num_recycles=af2_num_recycles,
            af2_num_models=af2_num_models,
            dockq_bin=dockq_bin,
        )

        gpu_s = rfd_per + mpnn_s + af2_timer.total_seconds
        r = DesignResult(
            arm="B", design_id=Path(pdb).stem,
            pdb_path=dsgn, iptm=iptm, dockq=dockq,
            ddg=None, gpu_seconds=gpu_s,
        )
        results.append(r)
        _log_result(r)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ARM C — Beam search, no anchors
# ─────────────────────────────────────────────────────────────────────────────

def run_arm_C(
    input_pdb:           str,
    output_dir:          str,
    hotspots:            str,
    model_weights:       str,
    native_pdb:          str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    af2_work_dir:        str,
    mpnn,
    cdr_mask,
    framework_pdb:       str,   # accepted from eval_kw but not used directly
    thermo,
    epitope_ca,
    extra_args:          List[str],
    device:              str,
    beam_width:          int   = 4,
    branch_factor:       int   = 4,
    n_checkpoints:       int   = 4,
    w_iptm:              float = 1.0,
    w_thermo:            float = 0.5,
    iptm_threshold:      float = IPTM_SUCCESS_THRESHOLD,
    ranking_mode:        str   = "cumulative",
    dockq_bin:           str   = "DockQ",
    af2_num_recycles:    int   = 1,    # 1 recycle during beam scoring for speed
    af2_num_models:      int   = 1,
    nanobody:            bool  = False,
    free_loops:          Dict  = None,
    stem:                str   = "",
) -> List[DesignResult]:
    """
    Complexa-style beam search WITHOUT anchor conservation.

    GPU timer wraps both RFdiffusion and ColabFold calls.
    Final evaluation uses a higher-quality AF2 run (3 recycles) to
    compute the official ipTM; DockQ is computed against the native.
    """
    free_loops   = free_loops or {}
    arm_dir      = os.path.join(output_dir, "arm_C_beam_no_anchor")
    work_dir     = os.path.join(arm_dir, "_beam_work")
    os.makedirs(work_dir, exist_ok=True)
    timer        = GPUTimer()
    results: List[DesignResult] = []
    use_gpu      = device.lower() == "cuda"

    print("\n" + "=" * 60)
    print("ARM C — Beam search (no anchor conservation)")
    print("=" * 60)

    cdr_ranges = parse_hlt_remarks(input_pdb)
    residues   = read_pdb_residues(input_pdb)

    if not nanobody and not any(r.pdb_chain == CHAIN_L for r in residues):
        nanobody = True

    renumbered_pdb = os.path.join(work_dir, "input_renumbered.pdb")
    resnum_mapping = write_renumbered_pdb(input_pdb, renumbered_pdb)

    anchor_residues: list = []    # ← no anchors
    contig_string = build_denovo_contig(
        residues, cdr_ranges, anchor_residues, free_loops, nanobody
    )

    def _remap_hotspots(hs, mapping):
        out = []
        for tok in hs.split(","):
            ch, rn = tok.strip()[0], int(tok.strip()[1:])
            out.append(f"{ch}{mapping[(ch, rn)]}")
        return ",".join(out)

    remapped_hotspots = _remap_hotspots(hotspots, resnum_mapping)
    renumbered_res    = read_pdb_residues(renumbered_pdb)
    remapped_contig   = build_denovo_contig(
        renumbered_res, cdr_ranges, anchor_residues, free_loops, nanobody
    )
    print(f"  Contig (remapped): {remapped_contig}")

    rank_fn      = RANKING_MODES[ranking_mode]
    node_counter = 0
    af2_bwork    = os.path.join(work_dir, "_af2")
    os.makedirs(af2_bwork, exist_ok=True)

    rollout_kw = dict(
        work_dir=work_dir, model_weights=model_weights,
        input_pdb=input_pdb, renumbered_pdb=renumbered_pdb,
        contig_string=remapped_contig, hotspots=remapped_hotspots,
        anchor_residues=[], cdr_ranges=cdr_ranges,
        extra_args=extra_args,
        mpnn=mpnn, cdr_mask=cdr_mask, thermo=thermo,
        epitope_ca=epitope_ca, w_iptm=w_iptm, w_thermo=w_thermo,
        iptm_threshold=iptm_threshold,
        af2_work_dir=af2_bwork,
        colabfold_batch_bin=colabfold_batch_bin,
        colabfold_python=colabfold_python,
        af2_num_recycles=af2_num_recycles,
        af2_num_models=af2_num_models,
        use_gpu=use_gpu, device=device,
    )

    # ── Initialise beam ──────────────────────────────────────────────────────
    n_seeds = beam_width * branch_factor
    print(f"\n  [C] Initialising beam with {n_seeds} seeds…")
    root = BeamNode(idx=-1, pdb_path=renumbered_pdb, parent_idx=None,
                    checkpoint_born=-1)

    seeds: List[BeamNode] = []
    for i in range(n_seeds):
        with timer:
            node = _rollout_and_score(
                parent_node=root, child_idx=i,
                checkpoint_idx=0, node_counter=node_counter,
                **rollout_kw,
            )
        if node is not None:
            node.gpu_seconds = timer.total_seconds  # cumulative up to this seed
            seeds.append(node)
        node_counter += 1

    seeds.sort(key=rank_fn, reverse=True)
    beam: List[BeamNode] = seeds[:beam_width]
    _print_beam(beam, rank_fn, ranking_mode, "C: Initial beam")

    # ── Beam checkpoints ─────────────────────────────────────────────────────
    for cp in range(1, n_checkpoints + 1):
        print(f"\n  [C] Checkpoint {cp}/{n_checkpoints}")
        candidates: List[BeamNode] = []
        for parent in beam:
            for b in range(branch_factor):
                with timer:
                    node = _rollout_and_score(
                        parent_node=parent, child_idx=b,
                        checkpoint_idx=cp, node_counter=node_counter,
                        **rollout_kw,
                    )
                if node is not None:
                    candidates.append(node)
                node_counter += 1

        if candidates:
            candidates.sort(key=rank_fn, reverse=True)
            beam = candidates[:beam_width]
        _print_beam(beam, rank_fn, ranking_mode, f"C: Beam after cp {cp}")

    # ── Final evaluation of beam survivors ───────────────────────────────────
    print(f"\n  [C] Final evaluation of {len(beam)} beam survivors…")
    final_af2_dir = os.path.join(arm_dir, "_final_af2")
    os.makedirs(final_af2_dir, exist_ok=True)

    for node in beam:
        af2_timer = GPUTimer()
        iptm, dockq = score_design(
            pdb_path=node.pdb_path, af2_work_dir=final_af2_dir,
            native_pdb=native_pdb,
            colabfold_batch_bin=colabfold_batch_bin,
            colabfold_python=colabfold_python,
            timer=af2_timer,
            af2_num_recycles=3,       # higher quality for final eval
            af2_num_models=af2_num_models,
            dockq_bin=dockq_bin,
        )
        # GPU cost = total beam time allocated to this survivor
        #   (entire timer / beam survivors is an approximation; for a more
        #    precise attribution divide by n_checkpoints*branch_factor)
        total_beam_gpu_s = timer.total_seconds / max(len(beam), 1)
        gpu_s = total_beam_gpu_s + af2_timer.total_seconds

        last_h = node.reward_history[-1] if node.reward_history else {}
        r = DesignResult(
            arm="C", design_id=f"c_node{node.idx:04d}",
            pdb_path=node.pdb_path,
            iptm=iptm, dockq=dockq,
            ddg=last_h.get("ddg"), gpu_seconds=gpu_s,
        )
        results.append(r)
        _log_result(r)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ARM D — Beam search + anchor conservation
# ─────────────────────────────────────────────────────────────────────────────

def run_arm_D(
    input_pdb:           str,
    anchors_json:        str,
    output_dir:          str,
    hotspots:            str,
    model_weights:       str,
    native_pdb:          str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    af2_work_dir:        str,
    mpnn,
    cdr_mask,
    framework_pdb:       str,   # accepted from eval_kw but not used directly
    thermo,
    epitope_ca,
    extra_args:          List[str],
    device:              str,
    beam_width:          int   = 4,
    branch_factor:       int   = 4,
    n_checkpoints:       int   = 4,
    w_iptm:              float = 1.0,
    w_thermo:            float = 0.5,
    iptm_threshold:      float = IPTM_SUCCESS_THRESHOLD,
    ranking_mode:        str   = "cumulative",
    dockq_bin:           str   = "DockQ",
    af2_num_recycles:    int   = 1,
    af2_num_models:      int   = 1,
    nanobody:            bool  = False,
    free_loops:          Dict  = None,
    stem:                str   = "",
) -> List[DesignResult]:
    """
    Beam search WITH anchor conservation.

    Identical to Arm C but:
      - provide_seq locking applied via build_provide_seq
      - anchor REMARK lines masked before RFdiffusion
      - anchor identities re-grafted after ProteinMPNN
    """
    free_loops = free_loops or {}
    arm_dir    = os.path.join(output_dir, "arm_D_beam_anchored")
    work_dir   = os.path.join(arm_dir, "_beam_work")
    os.makedirs(work_dir, exist_ok=True)
    timer      = GPUTimer()
    results: List[DesignResult] = []
    use_gpu    = device.lower() == "cuda"

    print("\n" + "=" * 60)
    print("ARM D — Beam search + anchor conservation")
    print("=" * 60)

    cdr_ranges      = parse_hlt_remarks(input_pdb)
    residues        = read_pdb_residues(input_pdb)
    anchor_residues = load_anchors(anchors_json)
    print(f"  Anchors ({len(anchor_residues)}): "
          f"{[f'{c}{n}' for c, n in anchor_residues]}")

    if not nanobody and not any(r.pdb_chain == CHAIN_L for r in residues):
        nanobody = True

    # Mask anchors from REMARK lines (same as snippet 1)
    _stem      = stem or Path(input_pdb).stem
    masked_pdb = os.path.join(arm_dir, f"{_stem}_anchors_masked.pdb")
    mask_anchors_in_hlt(input_pdb, anchor_residues, masked_pdb)

    renumbered_pdb = os.path.join(work_dir, "input_renumbered.pdb")
    resnum_mapping = write_renumbered_pdb(masked_pdb, renumbered_pdb)

    provide_seq = build_provide_seq(
        residues=residues, cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues, nanobody=nanobody,
    )

    contig_string = build_contig_string(
        residues=residues, cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues,
        free_loop_overrides=free_loops, nanobody=nanobody,
    )

    def _remap_hotspots(hs, mapping):
        out = []
        for tok in hs.split(","):
            ch, rn = tok.strip()[0], int(tok.strip()[1:])
            out.append(f"{ch}{mapping.get((ch, rn), rn)}")
        return ",".join(out)

    remapped_hotspots = _remap_hotspots(hotspots, resnum_mapping)
    renumbered_res    = read_pdb_residues(renumbered_pdb)
    remapped_contig   = build_contig_string(
        residues=renumbered_res, cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues,
        free_loop_overrides=free_loops, nanobody=nanobody,
    )

    # provide_seq is passed via extra_args so run_denovo_round (used inside
    # _rollout_and_score) can forward it to RFdiffusion
    provide_extra = [f"'contigmap.provide_seq=[{provide_seq}]'"]

    rank_fn      = RANKING_MODES[ranking_mode]
    node_counter = 0
    af2_bwork    = os.path.join(work_dir, "_af2")
    os.makedirs(af2_bwork, exist_ok=True)

    rollout_kw = dict(
        work_dir=work_dir, model_weights=model_weights,
        input_pdb=masked_pdb, renumbered_pdb=renumbered_pdb,
        contig_string=remapped_contig, hotspots=remapped_hotspots,
        anchor_residues=anchor_residues,
        cdr_ranges=cdr_ranges,
        extra_args=extra_args + provide_extra,   # inject provide_seq
        mpnn=mpnn, cdr_mask=cdr_mask, thermo=thermo,
        epitope_ca=epitope_ca, w_iptm=w_iptm, w_thermo=w_thermo,
        iptm_threshold=iptm_threshold,
        af2_work_dir=af2_bwork,
        colabfold_batch_bin=colabfold_batch_bin,
        colabfold_python=colabfold_python,
        af2_num_recycles=af2_num_recycles,
        af2_num_models=af2_num_models,
        use_gpu=use_gpu, device=device,
    )

    # ── Initialise beam ──────────────────────────────────────────────────────
    n_seeds = beam_width * branch_factor
    print(f"\n  [D] Initialising beam with {n_seeds} seeds…")
    root = BeamNode(idx=-1, pdb_path=renumbered_pdb, parent_idx=None,
                    checkpoint_born=-1)

    seeds: List[BeamNode] = []
    for i in range(n_seeds):
        with timer:
            node = _rollout_and_score(
                parent_node=root, child_idx=i,
                checkpoint_idx=0, node_counter=node_counter,
                **rollout_kw,
            )
        if node is not None:
            seeds.append(node)
        node_counter += 1

    seeds.sort(key=rank_fn, reverse=True)
    beam: List[BeamNode] = seeds[:beam_width]
    _print_beam(beam, rank_fn, ranking_mode, "D: Initial beam")

    # ── Beam checkpoints ─────────────────────────────────────────────────────
    for cp in range(1, n_checkpoints + 1):
        print(f"\n  [D] Checkpoint {cp}/{n_checkpoints}")
        candidates: List[BeamNode] = []
        for parent in beam:
            for b in range(branch_factor):
                with timer:
                    node = _rollout_and_score(
                        parent_node=parent, child_idx=b,
                        checkpoint_idx=cp, node_counter=node_counter,
                        **rollout_kw,
                    )
                if node is not None:
                    candidates.append(node)
                node_counter += 1

        if candidates:
            candidates.sort(key=rank_fn, reverse=True)
            beam = candidates[:beam_width]
        _print_beam(beam, rank_fn, ranking_mode, f"D: Beam after cp {cp}")

    # ── Final evaluation ─────────────────────────────────────────────────────
    print(f"\n  [D] Final evaluation of {len(beam)} beam survivors…")
    final_af2_dir = os.path.join(arm_dir, "_final_af2")
    os.makedirs(final_af2_dir, exist_ok=True)

    for node in beam:
        # Re-graft original T-chain after diffusion
        grafted = node.pdb_path.replace(".pdb", "_grafted.pdb")
        graft_target_sequence(
            rfdiffusion_pdb=node.pdb_path,
            original_target=input_pdb,
            input_pdb=masked_pdb,
            out_path=grafted,
        )

        af2_timer = GPUTimer()
        iptm, dockq = score_design(
            pdb_path=grafted, af2_work_dir=final_af2_dir,
            native_pdb=native_pdb,
            colabfold_batch_bin=colabfold_batch_bin,
            colabfold_python=colabfold_python,
            timer=af2_timer,
            af2_num_recycles=3,
            af2_num_models=af2_num_models,
            dockq_bin=dockq_bin,
        )
        total_beam_gpu_s = timer.total_seconds / max(len(beam), 1)
        gpu_s = total_beam_gpu_s + af2_timer.total_seconds

        last_h = node.reward_history[-1] if node.reward_history else {}
        r = DesignResult(
            arm="D", design_id=f"d_node{node.idx:04d}",
            pdb_path=grafted,
            iptm=iptm, dockq=dockq,
            ddg=last_h.get("ddg"), gpu_seconds=gpu_s,
        )
        results.append(r)
        _log_result(r)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Logging + reporting
# ─────────────────────────────────────────────────────────────────────────────

def _log_result(r: DesignResult):
    iptm_s  = f"{r.iptm:.3f}"  if r.iptm  is not None else "N/A"
    dockq_s = f"{r.dockq:.3f}" if r.dockq is not None else "N/A"
    ddg_s   = f"{r.ddg:+.3f}"  if r.ddg   is not None else "N/A"
    print(
        f"  [{r.arm}] {r.design_id:30s}  "
        f"ipTM={iptm_s}  DockQ={dockq_s}  DDG={ddg_s}  "
        f"GPU={r.gpu_seconds/3600:.4f} h  "
        f"{'✓ SUCCESS' if r.success else '✗'}"
    )


def summarise(all_results: Dict[str, List[DesignResult]]) -> Dict[str, ArmSummary]:
    summaries: Dict[str, ArmSummary] = {}
    for arm, results in all_results.items():
        n       = len(results)
        n_succ  = sum(1 for r in results if r.success)
        gpu_h   = sum(r.gpu_seconds for r in results) / 3600.0
        eff     = n_succ / gpu_h if gpu_h > 0 else 0.0
        iptms   = [r.iptm  for r in results if r.iptm  is not None]
        dockqs  = [r.dockq for r in results if r.dockq is not None]
        summaries[arm] = ArmSummary(
            arm=arm, n_designs=n, n_success=n_succ,
            total_gpu_hours=gpu_h,
            success_per_gpu_h=eff,
            mean_iptm=float(np.mean(iptms))   if iptms  else float("nan"),
            mean_dockq=float(np.mean(dockqs)) if dockqs else float("nan"),
            results=results,
        )
    return summaries


def print_report(summaries: Dict[str, ArmSummary]):
    arm_labels = {
        "A": "Vanilla RFantibody",
        "B": "Anchored (partial_T=50, provide_seq)",
        "C": "Beam search (no anchors)",
        "D": "Beam search + anchors",
    }
    print("\n" + "=" * 78)
    print("BENCHMARK RESULTS  —  successes defined as DockQ > 0.23 AND ipTM > 0.6")
    print("=" * 78)
    print(
        f"{'Arm':<4}  {'Description':<38}  "
        f"{'N':>4}  {'Succ':>4}  "
        f"{'GPUh':>7}  {'Succ/GPUh':>10}  "
        f"{'mean_ipTM':>9}  {'mean_DockQ':>10}"
    )
    print("-" * 78)
    for arm, s in sorted(summaries.items()):
        desc = arm_labels.get(arm, arm)
        print(
            f"{arm:<4}  {desc:<38}  "
            f"{s.n_designs:>4}  {s.n_success:>4}  "
            f"{s.total_gpu_hours:>7.3f}  {s.success_per_gpu_h:>10.3f}  "
            f"{s.mean_iptm:>9.3f}  {s.mean_dockq:>10.3f}"
        )
    print("=" * 78)


def save_results(
    summaries: Dict[str, ArmSummary],
    output_dir: str,
    stem: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # Per-design TSV
    tsv_path = os.path.join(output_dir, f"{stem}_benchmark_results.tsv")
    with open(tsv_path, "w") as fh:
        fh.write("arm\tdesign_id\tiptm\tdockq\tddg\tgpu_seconds\tsuccess\tpdb_path\n")
        for s in sorted(summaries.values(), key=lambda x: x.arm):
            for r in s.results:
                fh.write(
                    f"{r.arm}\t{r.design_id}\t"
                    f"{r.iptm if r.iptm is not None else 'NA'}\t"
                    f"{r.dockq if r.dockq is not None else 'NA'}\t"
                    f"{r.ddg if r.ddg is not None else 'NA'}\t"
                    f"{r.gpu_seconds:.2f}\t{r.success}\t{r.pdb_path}\n"
                )
    print(f"\n[Benchmark] Per-design TSV  → {tsv_path}")

    # Summary JSON
    json_path = os.path.join(output_dir, f"{stem}_benchmark_summary.json")
    out = {}
    for arm, s in summaries.items():
        out[arm] = {
            "n_designs":         s.n_designs,
            "n_success":         s.n_success,
            "total_gpu_hours":   round(s.total_gpu_hours, 4),
            "success_per_gpu_h": round(s.success_per_gpu_h, 4),
            "mean_iptm":         round(s.mean_iptm,   4),
            "mean_dockq":        round(s.mean_dockq,  4),
        }
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[Benchmark] Summary JSON    → {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark four RFantibody strategies and report successes per GPU-hour.\n"
            "Success: DockQ > 0.23 AND ipTM > 0.6.\n\n"
            "Arms:\n"
            "  A — Vanilla RFantibody\n"
            "  B — Anchored (partial_T=50 + provide_seq)\n"
            "  C — Beam search (no anchors)\n"
            "  D — Beam search + anchors"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── required ─────────────────────────────────────────────────────────────
    p.add_argument("--input",             required=True,
                   help="Input HLT PDB (H+L+T chains with CDR REMARKs)")
    p.add_argument("--native",            required=True,
                   help="Native/reference PDB for DockQ scoring")
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--hotspots",          required=True,
                   help="Comma-separated hotspot residues, e.g. T45,T67,T102")
    p.add_argument("--model_weights",     required=True,
                   help="RFdiffusion model weights (.pt)")
    p.add_argument("--colabfold_batch_bin", required=True,
                   help="Path to colabfold_batch executable")
    p.add_argument("--colabfold_python",    required=True,
                   help="Python interpreter inside ColabFold conda env")
    p.add_argument("--thermo_local_yaml",   required=True)
    p.add_argument("--thermo_model_yaml",   required=True)
    p.add_argument("--thermo_checkpoint",   required=True)
    p.add_argument("--mpnn_weights",        required=True)
    p.add_argument("--dockq_bin",    default="DockQ",
                   help="DockQ executable (default: DockQ, must be on PATH)")
    # ── anchors (required for arms B and D) ──────────────────────────────────
    p.add_argument("--anchors",
                   help="Anchor residues JSON (required for arms B and D)")
    # ── arm selection ─────────────────────────────────────────────────────────
    p.add_argument("--arms", default="A,B,C,D",
                   help="Comma-separated list of arms to run (default: A,B,C,D)")
    # ── vanilla / anchored design count ───────────────────────────────────────
    p.add_argument("--num_designs", type=int, default=50,
                   help="Designs per run for arms A and B (default: 50)")
    # ── beam hyperparameters (arms C and D) ───────────────────────────────────
    p.add_argument("--beam_width",     type=int,   default=4)
    p.add_argument("--branch_factor",  type=int,   default=4)
    p.add_argument("--n_checkpoints",  type=int,   default=4)
    p.add_argument("--ranking_mode",   default="cumulative",
                   choices=["cumulative", "latest", "average"])
    p.add_argument("--w_iptm",         type=float, default=1.0)
    p.add_argument("--w_thermo",       type=float, default=0.5)
    p.add_argument("--iptm_threshold", type=float, default=IPTM_SUCCESS_THRESHOLD)
    p.add_argument("--af2_num_recycles_beam", type=int, default=1,
                   help="AF2 recycles during beam scoring (default 1 for speed)")
    p.add_argument("--af2_num_recycles_eval", type=int, default=3,
                   help="AF2 recycles for final evaluation (default 3)")
    p.add_argument("--af2_num_models",  type=int,  default=1)
    # ── other ─────────────────────────────────────────────────────────────────
    p.add_argument("--free_loops",  default="",
                   help="Free-loop length overrides, e.g. H3:5-13,L3:4-10")
    p.add_argument("--nanobody",    action="store_true")
    p.add_argument("--name",        default="",
                   help="Output file stem (default: input PDB stem)")
    p.add_argument("--device",      default="cuda")
    p.add_argument("extra",         nargs=argparse.REMAINDER,
                   help="Extra args forwarded verbatim to RFdiffusion")
    return p.parse_args()


def main():
    args  = parse_args()
    extra = [a for a in (args.extra or []) if a != "--"]
    arms  = [a.strip().upper() for a in args.arms.split(",")]

    if any(arm in ("B", "D") for arm in arms) and not args.anchors:
        sys.exit("[ERROR] --anchors is required when running arms B or D.")

    input_pdb  = str(Path(args.input).resolve())
    native_pdb = str(Path(args.native).resolve())
    output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    stem = args.name or Path(input_pdb).stem
    free_loops = parse_free_loops(args.free_loops)
    device     = args.device

    # ── Shared scoring infrastructure (loaded once; reused across arms) ───────
    print("\n[Benchmark] Loading shared scoring infrastructure…")

    split_dir  = os.path.join(output_dir, "_split")
    target_pdb, framework_pdb = split_hlt_complex(input_pdb, split_dir)

    cdr_mask   = build_cdr_mask(framework_pdb)
    epitope_ca = load_epitope_ca(target_pdb, args.hotspots, device)

    print("[Benchmark] Loading ThermoMPNN…")
    thermo = load_thermompnn(
        config_yaml=args.thermo_model_yaml,
        local_yaml=args.thermo_local_yaml,
        checkpoint=args.thermo_checkpoint,
        device=device,
    )
    print("[Benchmark] Loading ProteinMPNN…")
    mpnn = load_proteinmpnn(args.mpnn_weights, device)

    af2_work_dir = os.path.join(output_dir, "_af2_eval")
    os.makedirs(af2_work_dir, exist_ok=True)

    # ── Common kwargs shared by arms that use AF2 evaluation ─────────────────
    eval_kw = dict(
        native_pdb=native_pdb,
        colabfold_batch_bin=args.colabfold_batch_bin,
        colabfold_python=args.colabfold_python,
        af2_work_dir=af2_work_dir,
        mpnn=mpnn, cdr_mask=cdr_mask,
        framework_pdb=framework_pdb,
        extra_args=extra,
        device=device,
        dockq_bin=args.dockq_bin,
        af2_num_models=args.af2_num_models,
        nanobody=args.nanobody,
        free_loops=free_loops,
    )

    beam_kw = dict(
        thermo=thermo, epitope_ca=epitope_ca,
        beam_width=args.beam_width,
        branch_factor=args.branch_factor,
        n_checkpoints=args.n_checkpoints,
        w_iptm=args.w_iptm, w_thermo=args.w_thermo,
        iptm_threshold=args.iptm_threshold,
        ranking_mode=args.ranking_mode,
        af2_num_recycles=args.af2_num_recycles_beam,
        stem=stem,
    )

    # ── Run arms ──────────────────────────────────────────────────────────────
    all_results: Dict[str, List[DesignResult]] = {}

    if "A" in arms:
        all_results["A"] = run_arm_A(
            input_pdb=input_pdb,
            output_dir=output_dir,
            hotspots=args.hotspots,
            model_weights=args.model_weights,
            num_designs=args.num_designs,
            af2_num_recycles=args.af2_num_recycles_eval,
            **eval_kw,
        )

    if "B" in arms:
        all_results["B"] = run_arm_B(
            input_pdb=input_pdb,
            anchors_json=str(Path(args.anchors).resolve()),
            output_dir=output_dir,
            hotspots=args.hotspots,
            model_weights=args.model_weights,
            num_designs=args.num_designs,
            af2_num_recycles=args.af2_num_recycles_eval,
            **eval_kw,
        )

    if "C" in arms:
        all_results["C"] = run_arm_C(
            input_pdb=input_pdb,
            output_dir=output_dir,
            hotspots=args.hotspots,
            model_weights=args.model_weights,
            **eval_kw, **beam_kw,
        )

    if "D" in arms:
        all_results["D"] = run_arm_D(
            input_pdb=input_pdb,
            anchors_json=str(Path(args.anchors).resolve()),
            output_dir=output_dir,
            hotspots=args.hotspots,
            model_weights=args.model_weights,
            **eval_kw, **beam_kw,
        )

    # ── Report ────────────────────────────────────────────────────────────────
    summaries = summarise(all_results)
    print_report(summaries)
    save_results(summaries, output_dir, stem)


if __name__ == "__main__":
    main()