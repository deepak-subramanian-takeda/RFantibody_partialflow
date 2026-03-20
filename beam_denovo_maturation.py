"""
beam_denovo_maturation.py

Beam search guided de novo antibody design with motif-anchored CDRs.

Beam Search vs SMC vs Feynman-Kac
───────────────────────────────────
SMC:         Weighted population with periodic hard resampling.
             Maintains diversity through stochastic resampling events.

Feynman-Kac: All trajectories survive; soft multiplicative weighting
             with no resampling. Final output drawn from weight distribution.

Beam Search: Hard pruning at every round — only the top-K trajectories
             survive. Each survivor is expanded into B children (branching
             factor), giving K*B candidates which are scored and pruned
             back to K. Total population stays constant at K*B during
             scoring, K after pruning.

Beam search is more aggressive than SMC or FK: it ruthlessly discards
low-scoring structures early, concentrating all compute on the most
promising regions of structure space. This is efficient when the scoring
function is reliable, but risks premature convergence if the potential
is noisy or the landscape is multimodal.

Beam parameters:
  beam_width  K : number of survivors kept after each pruning step
  branch_factor B : children generated per survivor per round
  Total candidates scored per round: K * B

Scoring uses the same composite potential as SMC/FK:
  G(x) = w_thermo * (-DDG) + w_bsa * BSA_contacts

where DDG is predicted by ThermoMPNN on the ProteinMPNN-designed sequence.

All structural helpers are imported from smc_denovo_maturation.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from smc_denovo_maturation import (
    CHAIN_H, CHAIN_L, CHAIN_T,
    AA1_TO_3, AA3_TO_1,
    TransferModel, ProteinMPNN,
    build_denovo_contig,
    build_cdr_mask,
    load_epitope_ca,
    load_thermompnn,
    load_proteinmpnn,
    score_composite,
    design_sequence_onto_backbone,
    graft_anchor_identities,
    pack_sidechains,
    copy_target_chain,
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
# Beam node dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeamNode:
    """One node in the beam — a structure with its cumulative score."""
    idx:            int               # unique node index across all rounds
    pdb_path:       str               # current best structure on disk
    parent_idx:     Optional[int]     # idx of parent node (None for root)
    round_born:     int               # round this node was created
    cumulative_score: float = 0.0    # sum of per-round scores
    score_history:  List[Dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Beam search scoring modes
# ─────────────────────────────────────────────────────────────────────────────

def cumulative_score(node: BeamNode) -> float:
    """Sum of all per-round scores — rewards consistently good trajectories."""
    return node.cumulative_score


def latest_score(node: BeamNode) -> float:
    """Score from the most recent round only — greedy local search."""
    if node.score_history:
        return node.score_history[-1]["score"]
    return 0.0


def average_score(node: BeamNode) -> float:
    """Mean per-round score — normalises for nodes born at different rounds."""
    if node.score_history:
        return np.mean([h["score"] for h in node.score_history])
    return 0.0


SCORING_MODES = {
    "cumulative": cumulative_score,
    "latest":     latest_score,
    "average":    average_score,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main beam search loop
# ─────────────────────────────────────────────────────────────────────────────

def run_beam_denovo(
    # ── structure inputs ────────────────────────────────────────────────────
    input_pdb:         str,
    anchors_json:      str,
    output_dir:        str,
    hotspots:          str,
    model_weights:     str,
    # ── beam search hyperparameters ─────────────────────────────────────────
    beam_width:        int   = 4,    # K: survivors kept after each round
    branch_factor:     int   = 4,    # B: children per survivor per round
    n_rounds:          int   = 4,    # number of expand-score-prune cycles
    scoring_mode:      str   = "cumulative",
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
) -> List[BeamNode]:
    """
    Beam search guided de novo antibody design.

    Each round:
      1. Expand  — generate B children from each of the K beam survivors
      2. Score   — evaluate all K*B candidates with ThermoMPNN + BSA
      3. Prune   — keep the top-K by scoring_mode, discard the rest

    Returns the final beam (K nodes) sorted by score descending.
    All intermediate structures are kept in output_dir/_beam_work/ for
    inspection of the search trajectory.
    """
    extra_args = extra_args or []
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, "_beam_work")
    os.makedirs(work_dir, exist_ok=True)
    stem = name or Path(input_pdb).stem

    score_fn = SCORING_MODES.get(scoring_mode)
    if score_fn is None:
        raise ValueError(f"Unknown scoring_mode '{scoring_mode}'. "
                         f"Choose: {list(SCORING_MODES)}")

    # ── 1. Parse HLT structure ───────────────────────────────────────────────
    print(f"[Beam] Parsing HLT: {input_pdb}")
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
    print(f"[Beam] {len(anchor_residues)} anchor(s): "
          f"{[f'{c}{n}' for c, n in anchor_residues]}")

    free_loops    = parse_free_loops(free_loops_spec)
    contig_string = build_denovo_contig(
        residues, cdr_ranges, anchor_residues, free_loops, nanobody
    )
    print(f"[Beam] Contig: {contig_string}")

    # ── 3. Scoring setup ─────────────────────────────────────────────────────
    cdr_mask   = build_cdr_mask(framework_pdb)
    epitope_ca = load_epitope_ca(target_pdb, hotspots, device)

    print("[Beam] Loading ThermoMPNN...")
    thermo = load_thermompnn(
        config_yaml=thermo_model_yaml,
        local_yaml=thermo_local_yaml,
        checkpoint=thermo_checkpoint,
        device=device,
    )
    print("[Beam] Loading ProteinMPNN...")
    mpnn = load_proteinmpnn(mpnn_weights, device)

    # ── 4. Initialise beam ───────────────────────────────────────────────────
    # Round 0: generate beam_width * branch_factor initial candidates from
    # scratch, score them, and seed the beam with the top beam_width.
    print(f"\n[Beam] ── Round 0 (initialise beam, "
          f"generating {beam_width * branch_factor} seeds) ──")

    node_counter = 0
    init_candidates: List[BeamNode] = []

    for i in range(beam_width * branch_factor):
        out_prefix = os.path.join(work_dir, f"r00_n{node_counter:04d}_rfd")
        print(f"  seed {i+1}/{beam_width * branch_factor}: "
              f"rfdiffusion...", end=" ", flush=True)
        out = run_denovo_round(
            ref_pdb=input_pdb,
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
            out = _apply_sequence_and_anchors(
                out, out_prefix, mpnn, cdr_mask, anchor_residues,
                input_pdb, device
            )
            score, bd = score_composite(
                pdb_path=out, thermo=thermo, cdr_mask=cdr_mask,
                epitope_ca=epitope_ca, w_thermo=w_thermo,
                w_bsa=w_bsa, device=device,
            )
            node = BeamNode(
                idx=node_counter, pdb_path=out,
                parent_idx=None, round_born=0,
                cumulative_score=score,
                score_history=[{"round": 0, "score": score, **bd}],
            )
            init_candidates.append(node)
            print(f"score={score:+.3f}")
        else:
            print("FAILED")
        node_counter += 1

    if not init_candidates:
        raise RuntimeError("All initial rfdiffusion runs failed.")

    # Seed the beam with top-K
    init_candidates.sort(key=score_fn, reverse=True)
    beam: List[BeamNode] = init_candidates[:beam_width]
    print(f"\n[Beam] Initial beam ({len(beam)} nodes):")
    for rank, node in enumerate(beam):
        print(f"  rank {rank}  node={node.idx}  "
              f"score={score_fn(node):+.3f}")

    # ── 5. Beam search rounds ────────────────────────────────────────────────
    for rnd in range(1, n_rounds + 1):
        print(f"\n[Beam] ── Round {rnd}/{n_rounds}  "
              f"beam={len(beam)}  branch={branch_factor}  "
              f"candidates={len(beam)*branch_factor} ──")

        # 5a. Expand: generate B children from each beam node
        candidates: List[BeamNode] = []
        for parent in beam:
            for b in range(branch_factor):
                out_prefix = os.path.join(
                    work_dir, f"r{rnd:02d}_n{node_counter:04d}_rfd"
                )
                print(f"  parent={parent.idx} child={b+1}/{branch_factor}: "
                      f"rfdiffusion...", end=" ", flush=True)
                out = run_denovo_round(
                    ref_pdb=parent.pdb_path,
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
                    out = _apply_sequence_and_anchors(
                        out, out_prefix, mpnn, cdr_mask, anchor_residues,
                        parent.pdb_path, device
                    )
                    # 5b. Score
                    score, bd = score_composite(
                        pdb_path=out, thermo=thermo, cdr_mask=cdr_mask,
                        epitope_ca=epitope_ca, w_thermo=w_thermo,
                        w_bsa=w_bsa, device=device,
                    )
                    node = BeamNode(
                        idx=node_counter,
                        pdb_path=out,
                        parent_idx=parent.idx,
                        round_born=rnd,
                        cumulative_score=parent.cumulative_score + score,
                        score_history=parent.score_history + [
                            {"round": rnd, "score": score, **bd}
                        ],
                    )
                    candidates.append(node)
                    print(f"score={score:+.3f}  "
                          f"cumulative={node.cumulative_score:+.3f}")
                else:
                    print("FAILED")
                node_counter += 1

        if not candidates:
            print(f"  [WARN] All expansions failed in round {rnd} — "
                  f"keeping previous beam.")
            continue

        # 5c. Prune: keep top-K by scoring_mode
        candidates.sort(key=score_fn, reverse=True)
        beam = candidates[:beam_width]

        print(f"\n  Beam after pruning ({len(beam)} survivors):")
        for rank, node in enumerate(beam):
            print(f"    rank {rank}  node={node.idx}  "
                  f"parent={node.parent_idx}  "
                  f"score(latest)={node.score_history[-1]['score']:+.3f}  "
                  f"score({scoring_mode})={score_fn(node):+.3f}")

        # Log diversity: how many distinct parents survived pruning
        unique_parents = len({n.parent_idx for n in beam})
        print(f"  Diversity: {unique_parents}/{len(beam)} unique parents in beam")
        if unique_parents == 1:
            print("  [WARN] Beam has collapsed to a single lineage. "
                  "Consider increasing beam_width or branch_factor.")

    # ── 6. Write final outputs ────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final_designs")
    os.makedirs(final_dir, exist_ok=True)

    print(f"\n[Beam] Writing {len(beam)} final designs to {final_dir}/")
    for rank, node in enumerate(beam):
        dst = os.path.join(final_dir, f"{stem}_beam_rank{rank:03d}.pdb")
        if os.path.isfile(node.pdb_path):
            shutil.copy2(node.pdb_path, dst)

        packed = os.path.join(
            final_dir, f"{stem}_beam_rank{rank:03d}_packed.pdb"
        )
        pack_sidechains(
            pdb_path=dst, anchor_residues=anchor_residues,
            mpnn_model=mpnn, cdr_mask=cdr_mask,
            out_path=packed, device=device,
        )
        final = os.path.join(
            final_dir, f"{stem}_beam_rank{rank:03d}_final.pdb"
        )
        graft_anchor_identities(
            rfdiffusion_pdb=packed, ref_pdb=input_pdb,
            anchor_residues=anchor_residues, out_path=final,
        )
        print(f"  rank {rank:03d}  node={node.idx}  "
              f"{scoring_mode}_score={score_fn(node):+.3f}  "
              f"→ {Path(final).name}")

    # ── 7. Save search tree summary ──────────────────────────────────────────
    summary = [
        {
            "rank":             rank,
            "node_idx":         node.idx,
            "parent_idx":       node.parent_idx,
            "round_born":       node.round_born,
            "cumulative_score": node.cumulative_score,
            f"{scoring_mode}_score": score_fn(node),
            "final_pdb":        os.path.basename(node.pdb_path),
            "score_history":    node.score_history,
        }
        for rank, node in enumerate(beam)
    ]
    summary_path = os.path.join(output_dir, f"{stem}_beam_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Beam] Summary → {summary_path}")

    return beam


# ─────────────────────────────────────────────────────────────────────────────
# Helper: sequence design + anchor enforcement (shared across rounds)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_sequence_and_anchors(
    pdb_path:        str,
    out_prefix:      str,
    mpnn:            ProteinMPNN,
    cdr_mask:        torch.Tensor,
    anchor_residues: list,
    ref_pdb:         str,
    device:          str,
) -> str:
    """Run ProteinMPNN sequence design and re-graft anchor identities."""
    seq_pdb = out_prefix + "_seq.pdb"
    result  = design_sequence_onto_backbone(
        mpnn_model=mpnn, backbone_pdb=pdb_path, cdr_mask=cdr_mask,
        out_pdb=seq_pdb, temperature=0.1, device=device,
    )
    out = result or pdb_path
    if anchor_residues:
        enforced = out_prefix + "_enforced.pdb"
        graft_anchor_identities(
            rfdiffusion_pdb=out, ref_pdb=ref_pdb,
            anchor_residues=anchor_residues, out_path=enforced,
        )
        out = enforced
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Beam search guided de novo antibody design. "
            "Keeps top-K structures at each round; expands each into B children."
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
    p.add_argument("--beam_width",     type=int,   default=4,
                   help="K: survivors kept after each round")
    p.add_argument("--branch_factor",  type=int,   default=4,
                   help="B: children generated per survivor per round")
    p.add_argument("--n_rounds",       type=int,   default=4)
    p.add_argument("--scoring_mode",   default="cumulative",
                   choices=["cumulative", "latest", "average"],
                   help="How to rank candidates for pruning")
    p.add_argument("--w_thermo",       type=float, default=1.0)
    p.add_argument("--w_bsa",          type=float, default=0.5)
    p.add_argument("--free_loops",    default="")
    p.add_argument("--nanobody",      action="store_true")
    p.add_argument("--name",          default="")
    p.add_argument("--device",        default="cuda")
    p.add_argument("extra",           nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    args  = parse_args()
    extra = [a for a in (args.extra or []) if a != "--"]
    run_beam_denovo(
        input_pdb=str(Path(args.input).resolve()),
        anchors_json=str(Path(args.anchors).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        hotspots=args.hotspots,
        model_weights=args.model_weights,
        beam_width=args.beam_width,
        branch_factor=args.branch_factor,
        n_rounds=args.n_rounds,
        scoring_mode=args.scoring_mode,
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
    )


if __name__ == "__main__":
    main()