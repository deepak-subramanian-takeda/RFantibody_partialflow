"""
fk_denovo_maturation.py

Feynman-Kac (FK) steered de novo antibody design with motif-anchored CDRs.

Feynman-Kac vs SMC
──────────────────
SMC uses hard resampling: at each round, low-weight particles are discarded
and high-weight particles are duplicated, maintaining a fixed population.

Feynman-Kac steering uses soft multiplicative weighting without resampling:
every trajectory is kept, but its contribution to the final ensemble is
proportional to the product of its potential scores across all rounds:

    w_i = exp( γ · Σ_r G(x_i^r) )

where G is the composite potential (ThermoMPNN + BSA) and γ is the
guidance scale.  The final output is drawn by sampling from this weighted
distribution — equivalent to importance sampling from a tilted version of
the rfdiffusion generative distribution.

This has several advantages over SMC for a subprocess-based pipeline:
  - No population collapse: all N trajectories survive all R rounds
  - No diversity loss from resampling-induced duplication
  - The weight history gives a full picture of how each trajectory evolved
  - Final selection is by weighted sampling, not just top-k

The tradeoff is that without resampling, low-scoring trajectories are
never pruned, so compute is "wasted" on them.  This is acceptable when
N is small (8–32) and R is small (3–5), which is typical here.

Relationship to Feynman-Kac path measures
──────────────────────────────────────────
The FK path measure over trajectories x^{1:R} is:

    Q(x^{1:R}) ∝ P(x^{1:R}) · Π_r exp(γ · G(x^r))

where P is the rfdiffusion prior.  Importance weights for a sample drawn
from P are exactly w_i above.  The normalised weights define a probability
distribution over trajectories from which we draw the final designs.

All structural helpers (contig builder, anchor grafting, Kabsch alignment,
ThermoMPNN/ProteinMPNN scoring) are shared with smc_denovo_maturation.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

# ── Re-use all helpers from smc_denovo_maturation.py ─────────────────────────
# Assumes smc_denovo_maturation.py is on PYTHONPATH or in the same directory.
from smc_denovo_maturation import (
    # env / path setup is handled by importing this module
    CHAIN_H, CHAIN_L, CHAIN_T,
    AA1_TO_3, AA3_TO_1,
    TransferModel, ProteinMPNN,
    parse_PDB, tied_featurize, _S_to_seq,
    build_denovo_contig,
    build_cdr_mask,
    load_epitope_ca,
    load_thermompnn,
    load_proteinmpnn,
    parse_pdb_for_thermo,
    score_thermompnn,
    score_bsa,
    score_composite,
    _build_per_chain_fixed_positions,
    design_sequence_onto_backbone,
    graft_anchor_identities,
    pack_sidechains,
    copy_target_chain,
    build_denovo_rfdiffusion_cmd,
    run_denovo_round,
)

from partial_diffusion_maturation import (
    CdrRange, ResidueInfo,
    load_anchors,
    parse_free_loops,
    parse_hlt_remarks,
    read_pdb_residues,
    split_hlt_complex,
)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """One FK trajectory — kept alive for all R rounds."""
    idx:           int
    pdb_path:      str                      # current structure on disk
    log_weight:    float = 0.0              # cumulative Σ_r γ·G(x^r)
    score_history: List[Dict] = field(default_factory=list)
    round_pdbs:    List[str]  = field(default_factory=list)  # one per round


# ─────────────────────────────────────────────────────────────────────────────
# FK weighting utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalised_weights(trajectories: List[Trajectory]) -> np.ndarray:
    """
    Compute normalised importance weights from log-weights.
    Returns array of shape (N,) summing to 1.
    """
    lw = np.array([t.log_weight for t in trajectories], dtype=np.float64)
    lw -= lw.max()                    # numerical stability
    w   = np.exp(lw)
    return w / w.sum()


def effective_sample_size(trajectories: List[Trajectory]) -> float:
    """
    ESS = 1 / Σ w_i^2  (ranges from 1 to N).
    Low ESS means the weight distribution is highly concentrated —
    consider increasing N or reducing guidance_scale.
    """
    w = normalised_weights(trajectories)
    return float(1.0 / (w ** 2).sum())


def weighted_sample(
    trajectories: List[Trajectory],
    n_samples:    int,
    rng:          np.random.Generator,
) -> List[Trajectory]:
    """
    Draw n_samples trajectories without replacement from the FK-weighted
    distribution.  This is the final selection step.
    """
    w = normalised_weights(trajectories)
    indices = rng.choice(len(trajectories), size=n_samples,
                         replace=False, p=w)
    return [trajectories[i] for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# Annealing schedule
# ─────────────────────────────────────────────────────────────────────────────

def annealing_schedule(
    n_rounds:       int,
    schedule:       str = "linear",
    gamma_final:    float = 1.0,
) -> List[float]:
    """
    Return per-round guidance scale γ_r for r = 1..n_rounds.

    Annealing the guidance scale (starting low, ending at gamma_final)
    prevents the potential from dominating too early when structures are
    still rough, and concentrates pressure in later rounds when structures
    are more refined.

    Schedules:
      "linear"      : γ_r = gamma_final * r / n_rounds
      "constant"    : γ_r = gamma_final  (no annealing)
      "geometric"   : γ_r = gamma_final * (r/n_rounds)^2  (slow start)
      "reverse"     : γ_r = gamma_final * (1 - (n_rounds-r)/n_rounds)
                      (heavier weighting on early rounds)
    """
    r = np.arange(1, n_rounds + 1, dtype=float)
    if schedule == "linear":
        gammas = gamma_final * r / n_rounds
    elif schedule == "constant":
        gammas = np.full(n_rounds, gamma_final)
    elif schedule == "geometric":
        gammas = gamma_final * (r / n_rounds) ** 2
    elif schedule == "reverse":
        gammas = gamma_final * (1.0 - (n_rounds - r) / n_rounds)
    else:
        raise ValueError(f"Unknown schedule '{schedule}'. "
                         "Choose: linear, constant, geometric, reverse")
    return gammas.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Main FK loop
# ─────────────────────────────────────────────────────────────────────────────

def run_fk_denovo(
    # ── structure inputs ────────────────────────────────────────────────────
    input_pdb:         str,
    anchors_json:      str,
    output_dir:        str,
    hotspots:          str,
    model_weights:     str,
    # ── FK hyperparameters ───────────────────────────────────────────────────
    n_trajectories:    int   = 16,       # population size (no resampling)
    n_rounds:          int   = 4,        # number of rfdiffusion rounds
    guidance_scale:    float = 1.0,      # γ_final — total potential strength
    annealing:         str   = "linear", # schedule for γ_r
    n_output:          int   = 5,        # designs to draw from final weights
    # ── ThermoMPNN ──────────────────────────────────────────────────────────
    thermo_local_yaml: str   = "",
    thermo_model_yaml: str   = "",
    thermo_checkpoint: str   = "",
    # ── ProteinMPNN ─────────────────────────────────────────────────────────
    mpnn_weights:      str   = "",
    # ── potential weights ────────────────────────────────────────────────────
    w_thermo:          float = 1.0,
    w_bsa:             float = 0.5,
    # ── other ───────────────────────────────────────────────────────────────
    free_loops_spec:   str   = "",
    nanobody:          bool  = False,
    name:              str   = "",
    extra_args:        Optional[List[str]] = None,
    device:            str   = "cuda",
    seed:              int   = 42,
) -> List[Trajectory]:
    """
    Feynman-Kac steered de novo antibody design.

    Returns all trajectories sorted by log_weight (best first).
    The top n_output weighted samples are written to final_designs/.
    """
    extra_args = extra_args or []
    rng        = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, "_fk_work")
    os.makedirs(work_dir, exist_ok=True)
    stem = name or Path(input_pdb).stem

    # ── 1. Parse HLT structure ───────────────────────────────────────────────
    print(f"[FK] Parsing HLT: {input_pdb}")
    cdr_ranges = parse_hlt_remarks(input_pdb)
    residues   = read_pdb_residues(input_pdb)

    n_l = sum(1 for r in residues if r.pdb_chain == CHAIN_L)
    if not nanobody and n_l == 0:
        print("[INFO] No L-chain — treating as nanobody.")
        nanobody = True

    split_dir = os.path.join(work_dir, "_split")
    target_pdb, framework_pdb = split_hlt_complex(input_pdb, split_dir)

    # ── 2. Anchors and contig ────────────────────────────────────────────────
    anchor_residues = load_anchors(anchors_json)
    print(f"[FK] {len(anchor_residues)} anchor(s): "
          f"{[f'{c}{n}' for c, n in anchor_residues]}")

    free_loops    = parse_free_loops(free_loops_spec)
    contig_string = build_denovo_contig(
        residues, cdr_ranges, anchor_residues, free_loops, nanobody
    )
    print(f"[FK] Contig: {contig_string}")

    # ── 3. Masks and scoring setup ───────────────────────────────────────────
    cdr_mask   = build_cdr_mask(framework_pdb)
    epitope_ca = load_epitope_ca(target_pdb, hotspots, device)

    print("[FK] Loading ThermoMPNN...")
    thermo = load_thermompnn(
        config_yaml=thermo_model_yaml,
        local_yaml=thermo_local_yaml,
        checkpoint=thermo_checkpoint,
        device=device,
    )
    print("[FK] Loading ProteinMPNN...")
    mpnn = load_proteinmpnn(mpnn_weights, device)

    # ── 4. Annealing schedule ────────────────────────────────────────────────
    gammas = annealing_schedule(n_rounds, annealing, guidance_scale)
    print(f"[FK] Annealing schedule ({annealing}): "
          f"{[f'{g:.3f}' for g in gammas]}")

    # ── 5. Initialise trajectories ───────────────────────────────────────────
    print(f"[FK] Initialising {n_trajectories} trajectories...")
    trajectories: List[Trajectory] = []
    for i in range(n_trajectories):
        init_path = os.path.join(work_dir, f"r00_t{i:03d}_ref.pdb")
        shutil.copy2(input_pdb, init_path)
        trajectories.append(Trajectory(idx=i, pdb_path=init_path))

    # ── 6. FK loop ───────────────────────────────────────────────────────────
    for rnd, gamma_r in enumerate(gammas, start=1):
        print(f"\n[FK] ── Round {rnd}/{n_rounds}  γ_r={gamma_r:.3f} ──")

        # 6a. Run rfdiffusion for every trajectory (no pruning)
        for t in trajectories:
            out_prefix = os.path.join(
                work_dir, f"r{rnd:02d}_t{t.idx:03d}_rfd"
            )
            print(f"  t{t.idx:03d}: rfdiffusion...", end=" ", flush=True)
            out = run_denovo_round(
                ref_pdb=t.pdb_path,
                contig_string=contig_string,
                hotspots=hotspots,
                output_prefix=out_prefix,
                model_weights=model_weights,
                original_pdb=input_pdb,
                anchor_residues=anchor_residues,
                cdr_ranges=cdr_ranges,
                extra_args=extra_args,
            )
            if out is not None:
                seq_pdb = out_prefix + "_seq.pdb"
                out = design_sequence_onto_backbone(
                    mpnn_model=mpnn,
                    backbone_pdb=out,
                    cdr_mask=cdr_mask,
                    out_pdb=seq_pdb,
                    temperature=0.1,
                    device=device,
                ) or out
                if anchor_residues:
                    enforced = out_prefix + "_enforced.pdb"
                    graft_anchor_identities(
                        rfdiffusion_pdb=out,
                        ref_pdb=t.pdb_path,
                        anchor_residues=anchor_residues,
                        out_path=enforced,
                    )
                    out = enforced
                # Keep the round output as next-round reference
                ref_next = out_prefix + "_ref.pdb"
                shutil.copy2(out, ref_next)
                t.pdb_path = ref_next
                t.round_pdbs.append(out)
                print("done")
            else:
                # Failed run — penalise but do not remove
                t.round_pdbs.append(None)
                print("FAILED")

        # 6b. Score and update log-weights with γ_r
        # KEY FK DIFFERENCE vs SMC: weights are accumulated multiplicatively
        # across rounds using the annealed γ_r, with NO resampling step.
        print(f"  Scoring {n_trajectories} trajectories...")
        for t in trajectories:
            current_pdb = t.round_pdbs[-1]
            if current_pdb is None:
                t.log_weight -= 1e6   # penalise failed runs
                continue
            score, bd = score_composite(
                pdb_path=current_pdb,
                thermo=thermo,
                cdr_mask=cdr_mask,
                epitope_ca=epitope_ca,
                w_thermo=w_thermo,
                w_bsa=w_bsa,
                device=device,
            )
            # Accumulate γ_r · G(x^r) — the FK path weight product
            t.log_weight += gamma_r * score
            t.score_history.append({
                "round": rnd, "gamma_r": gamma_r,
                "score": score, **bd,
                "cumulative_log_weight": t.log_weight,
            })
            print(f"    t{t.idx:03d}  score={score:+.3f}  "
                  f"thermo={bd['thermo_neg_ddg']:+.3f}  "
                  f"bsa={bd['bsa_contacts']:.0f}  "
                  f"log_w={t.log_weight:+.3f}")

        # 6c. Diagnostics — ESS and weight concentration
        # In FK there is no resampling, but we monitor ESS to warn the user
        # if the weight distribution has collapsed (suggesting guidance_scale
        # is too high or the potential is too noisy).
        ess = effective_sample_size(trajectories)
        w   = normalised_weights(trajectories)
        print(f"  ESS = {ess:.1f} / {n_trajectories}  "
              f"(max_w={w.max():.3f}  min_w={w.min():.3f})")
        if ess < 0.25 * n_trajectories:
            print(f"  [WARN] ESS is very low ({ess:.1f}). Consider reducing "
                  f"guidance_scale or switching to SMC (smc_denovo_maturation.py).")

    # ── 7. Final weighted sampling and output ────────────────────────────────
    trajectories.sort(key=lambda t: t.log_weight, reverse=True)
    n_draw   = min(n_output, n_trajectories)
    selected = weighted_sample(trajectories, n_draw, rng)

    final_dir = os.path.join(output_dir, "final_designs")
    os.makedirs(final_dir, exist_ok=True)

    print(f"\n[FK] Drawing {n_draw} designs from FK-weighted distribution...")
    w_norm = normalised_weights(trajectories)
    for rank, t in enumerate(selected):
        dst = os.path.join(final_dir, f"{stem}_fk_rank{rank:03d}.pdb")
        if t.pdb_path and os.path.isfile(t.pdb_path):
            shutil.copy2(t.pdb_path, dst)

        packed = os.path.join(final_dir, f"{stem}_fk_rank{rank:03d}_packed.pdb")
        pack_sidechains(
            pdb_path=dst,
            anchor_residues=anchor_residues,
            mpnn_model=mpnn,
            cdr_mask=cdr_mask,
            out_path=packed,
            device=device,
        )
        final = os.path.join(final_dir, f"{stem}_fk_rank{rank:03d}_final.pdb")
        graft_anchor_identities(
            rfdiffusion_pdb=packed,
            ref_pdb=input_pdb,
            anchor_residues=anchor_residues,
            out_path=final,
        )
        traj_weight = w_norm[trajectories.index(t)]
        print(f"  rank {rank:03d}  traj={t.idx:03d}  "
              f"log_w={t.log_weight:+.3f}  norm_w={traj_weight:.4f}"
              f"  → {Path(final).name}")

    # ── 8. Save full weight summary ──────────────────────────────────────────
    summary = [
        {
            "trajectory":    t.idx,
            "log_weight":    t.log_weight,
            "norm_weight":   float(w_norm[i]),
            "final_pdb":     os.path.basename(t.pdb_path) if t.pdb_path else None,
            "score_history": t.score_history,
        }
        for i, t in enumerate(trajectories)
    ]
    summary_path = os.path.join(output_dir, f"{stem}_fk_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[FK] Weight summary → {summary_path}")

    return trajectories


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Feynman-Kac steered de novo antibody design. "
            "No resampling — all trajectories survive all rounds."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",             required=True)
    p.add_argument("--anchors",           required=True)
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--hotspots",          required=True)
    p.add_argument("--model_weights",     required=True)
    p.add_argument("--thermo_local_yaml", required=True)
    p.add_argument("--thermo_model_yaml", required=True)
    p.add_argument("--thermo_checkpoint", required=True)
    p.add_argument("--mpnn_weights",      required=True)
    p.add_argument("--n_trajectories", type=int,   default=16)
    p.add_argument("--n_rounds",       type=int,   default=4)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--annealing",      default="linear",
                   choices=["linear", "constant", "geometric", "reverse"],
                   help="Annealing schedule for guidance scale across rounds")
    p.add_argument("--n_output",       type=int,   default=5,
                   help="Number of final designs to draw from FK distribution")
    p.add_argument("--w_thermo",       type=float, default=1.0)
    p.add_argument("--w_bsa",          type=float, default=0.5)
    p.add_argument("--free_loops",    default="")
    p.add_argument("--nanobody",      action="store_true")
    p.add_argument("--name",          default="")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("extra",           nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    args  = parse_args()
    extra = [a for a in (args.extra or []) if a != "--"]
    run_fk_denovo(
        input_pdb=str(Path(args.input).resolve()),
        anchors_json=str(Path(args.anchors).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        hotspots=args.hotspots,
        model_weights=args.model_weights,
        n_trajectories=args.n_trajectories,
        n_rounds=args.n_rounds,
        guidance_scale=args.guidance_scale,
        annealing=args.annealing,
        n_output=args.n_output,
        thermo_local_yaml=args.thermo_local_yaml,
        thermo_model_yaml=args.thermo_model_yaml,
        thermo_checkpoint=args.thermo_checkpoint,
        mpnn_weights=args.mpnn_weights,
        w_thermo=args.w_thermo,
        w_bsa=args.w_bsa,
        free_loops_spec=args.free_loops,
        nanobody=args.nanobody,
        name=args.name,
        extra_args=extra,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()