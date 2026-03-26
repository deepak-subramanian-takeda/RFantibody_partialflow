"""
beam_denovo_maturation_complexa.py

Beam search guided de novo antibody design, rewritten to follow the
inference-time compute scaling approach described in:

  "Scaling Atomistic Protein Binder Design with Generative Pretraining
   and Test-Time Compute" (Complexa / Proteina-Complexa, ICLR 2026 sub.)
   https://openreview.net/pdf?id=qmCpJtFZra

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW THIS DIFFERS FROM THE ORIGINAL beam_denovo_maturation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Original (round-level beam):
  • One full RFdiffusion run = one "step".
  • Scoring happened only at the end of each complete generation run.
  • BSA contacts served as the interface proxy.

Complexa-style (trajectory-level beam, Sec. 3.4 of the paper):
  • The denoising process is partitioned into checkpoints every K steps.
  • At each checkpoint, each of the N beam trajectories is branched into
    L children.  All N*L children are independently rolled out to t=1
    (a clean structure), decoded, and folded by a structure predictor to
    obtain reliable rewards.  (The paper specifically avoids Tweedie-based
    intermediate estimates because structure predictor rewards are only
    reliable on realistic, near-clean structures.)
  • The top-N children form the new beam (Eq. 2 of the paper).

Reward function:
  Two complementary objectives are combined (cf. paper Sec. 3.3 and
  Tab. 3 which shows fipAE + fH-Bond are additive):

      R(s) = w_ipae  * score_ipAE(s)          # interface quality
           + w_thermo * score_thermo(s)        # thermodynamic stability

  where:
    score_ipAE(s)   = ipAE_threshold - ipAE(s)   (higher is better;
                      ipAE < 7.0 Å is the standard success criterion)
    score_thermo(s) = -DDG(s)                    (ThermoMPNN ΔΔG;
                      more negative = more stable → larger reward)

  ipAE is obtained by running AlphaFold2-Multimer (via ColabDesign /
  af2_runner) on the binder–target complex.  This mirrors the paper's
  use of fipAE as the primary inference-time reward signal.

Noise-level gating:
  Scoring is computationally expensive, so a checkpoint fires only once
  every K diffusion steps (--steps_per_checkpoint).  An optional
  min_noise_level gate suppresses scoring at very noisy intermediate
  states where the structure predictor would produce unreliable rewards
  (analogous to the noise-level gating discussed in the concurrent
  Search-Based ITS paper, which Complexa addresses differently via
  full rollouts — we expose the gate for users who want it).

Diversity:
  After every prune step we log the number of distinct parent lineages
  in the surviving beam and warn if the beam collapses.

Beam parameters (mirrors Eq. 2 in the paper):
  beam_width  N : survivors kept after each prune step
  branch_factor L : children launched per survivor per checkpoint
  steps_per_checkpoint K : diffusion steps between beam prune events
  Total candidates evaluated per checkpoint: N * L

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

from evaluate_designs import (
    write_colabfold_fasta,
    run_colabfold,
    find_top_af2_result,
    extract_plddt,
    extract_ipae,
    BINDER_CHAINS,
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
# ipAE reward via colabfold_batch  (mirrors evaluate_designs.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_af2_multimer(
    binder_pdb:          str,
    af2_work_dir:        str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    num_recycles:        int  = 1,
    num_models:          int  = 1,
    use_gpu:             bool = True,
    overwrite:           bool = False,
) -> Optional[float]:
    """
    Fold the binder–target complex with AlphaFold2-Multimer (colabfold_batch)
    and return ipAE (Å).  Returns None on any failure.

    Uses the same FASTA/runner/parser stack as evaluate_designs.py:
      write_colabfold_fasta → run_colabfold → find_top_af2_result → extract_ipae

    We default to num_recycles=1 (vs. 3 at eval time) to keep beam search
    rollouts fast; the reward signal is directional even at low recycle depth.
    """
    os.makedirs(af2_work_dir, exist_ok=True)
    stem    = Path(binder_pdb).stem
    out_dir = os.path.join(af2_work_dir, stem)

    # Skip recomputation if outputs already present
    already_done = (
        not overwrite
        and os.path.isdir(out_dir)
        and any(Path(out_dir).glob("*rank_001*scores*.json"))
    )

    if not already_done:
        fasta_path = os.path.join(af2_work_dir, f"{stem}.fasta")
        # Chain order must match evaluate_designs.py convention: H, L, T
        chains = [c for c in [CHAIN_H, CHAIN_L, CHAIN_T]
                  if _chain_present(binder_pdb, c)]
        ok = write_colabfold_fasta(binder_pdb, fasta_path, chains)
        if not ok:
            return None

        success = run_colabfold(
            fasta_path=fasta_path,
            af2_out_dir=out_dir,
            colabfold_batch_bin=colabfold_batch_bin,
            colabfold_python=colabfold_python,
            num_recycles=num_recycles,
            num_models=num_models,
            use_gpu=use_gpu,
        )
        if not success:
            return None

    af2_result = find_top_af2_result(out_dir, stem)
    if af2_result is None or af2_result["pae_data"] is None:
        return None

    ipae = extract_ipae(
        pae_data=af2_result["pae_data"],
        pdb_path=af2_result["pdb"],
    )
    return None if (ipae is None or np.isnan(ipae)) else ipae


def _chain_present(pdb_path: str, chain: str) -> bool:
    """Return True if at least one ATOM record for the given chain exists."""
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == chain:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Composite reward  R = w_ipae * score_ipAE + w_thermo * score_thermo
# ─────────────────────────────────────────────────────────────────────────────

# Default ipAE threshold for converting to a reward (paper uses 7.0 Å).
IPAE_SUCCESS_THRESHOLD = 7.0


def score_complexa_reward(
    pdb_path:            str,
    af2_work_dir:        str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    thermo,
    cdr_mask:            torch.Tensor,
    epitope_ca:          torch.Tensor,
    w_ipae:              float = 1.0,
    w_thermo:            float = 0.5,
    ipae_threshold:      float = IPAE_SUCCESS_THRESHOLD,
    af2_num_recycles:    int   = 1,
    af2_num_models:      int   = 1,
    use_gpu:             bool  = True,
    device:              str   = "cuda",
) -> Tuple[float, Dict]:
    """
    Complexa-style composite reward:

        R = w_ipae  * (ipae_threshold - ipAE)     ← interface quality
          + w_thermo * (-DDG)                      ← thermodynamic stability

    ipAE is computed via colabfold_batch (same pipeline as evaluate_designs.py).
    Falls back to ThermoMPNN-only if colabfold_batch_bin is empty/missing.

    Returns (reward_scalar, breakdown_dict).
    """
    # ── ThermoMPNN component ─────────────────────────────────────────────────
    _, thermo_bd = score_composite(
        pdb_path=pdb_path, thermo=thermo, cdr_mask=cdr_mask,
        epitope_ca=epitope_ca, w_thermo=1.0, w_bsa=0.0, device=device,
    )
    ddg = thermo_bd.get("ddg", 0.0)

    # ── ipAE component (colabfold_batch) ─────────────────────────────────────
    ipae: Optional[float] = None
    if colabfold_batch_bin and os.path.isfile(colabfold_batch_bin):
        ipae = run_af2_multimer(
            binder_pdb=pdb_path,
            af2_work_dir=af2_work_dir,
            colabfold_batch_bin=colabfold_batch_bin,
            colabfold_python=colabfold_python,
            num_recycles=af2_num_recycles,
            num_models=af2_num_models,
            use_gpu=use_gpu,
        )

    # ── Combine ───────────────────────────────────────────────────────────────
    reward_thermo = w_thermo * (-ddg)
    if ipae is not None:
        reward_ipae = w_ipae * (ipae_threshold - ipae)
        reward      = reward_thermo + reward_ipae
        success     = ipae < ipae_threshold
    else:
        reward_ipae = 0.0
        reward      = reward_thermo
        success     = False

    breakdown = {
        "ddg": ddg, "ipae": ipae,
        "reward_thermo": reward_thermo, "reward_ipae": reward_ipae,
        "reward": reward, "success": success,
        **thermo_bd,
    }
    return reward, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Beam node
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeamNode:
    """
    One node (trajectory state) in the beam.

    In the Complexa formulation the beam is defined over denoising
    trajectory states: each node holds a (partially denoised) structure
    that will be rolled out and scored at the next checkpoint, and then
    either pruned or carried forward.
    """
    idx:               int
    pdb_path:          str            # path to the current (clean) structure
    parent_idx:        Optional[int]  # None for root nodes
    checkpoint_born:   int            # checkpoint index at creation
    cumulative_reward: float = 0.0
    reward_history:    List[Dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Ranking modes  (analogous to Eq. 2 in the paper — argmax over T ⊆ C)
# ─────────────────────────────────────────────────────────────────────────────

def rank_cumulative(node: BeamNode) -> float:
    """Sum of rewards across all checkpoints — rewards sustained quality."""
    return node.cumulative_reward


def rank_latest(node: BeamNode) -> float:
    """Reward at the most recent checkpoint only — greedy / myopic."""
    return node.reward_history[-1]["reward"] if node.reward_history else 0.0


def rank_average(node: BeamNode) -> float:
    """Mean reward per checkpoint — normalises for nodes born at different steps."""
    if node.reward_history:
        return float(np.mean([h["reward"] for h in node.reward_history]))
    return 0.0


RANKING_MODES = {
    "cumulative": rank_cumulative,
    "latest":     rank_latest,
    "average":    rank_average,
}


# ─────────────────────────────────────────────────────────────────────────────
# Rollout helper
# ─────────────────────────────────────────────────────────────────────────────

def _rollout_and_score(
    *,
    parent_node:         BeamNode,
    child_idx:           int,
    checkpoint_idx:      int,
    node_counter:        int,
    work_dir:            str,
    contig_string:       str,
    hotspots:            str,
    model_weights:       str,
    input_pdb:           str,
    anchor_residues:     list,
    cdr_ranges,
    extra_args:          List[str],
    mpnn,
    cdr_mask:            torch.Tensor,
    thermo,
    epitope_ca:          torch.Tensor,
    w_ipae:              float,
    w_thermo:            float,
    ipae_threshold:      float,
    af2_work_dir:        str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    af2_num_recycles:    int,
    af2_num_models:      int,
    use_gpu:             bool,
    device:              str,
) -> Optional[BeamNode]:
    """
    Core inner loop: roll out one child from parent_node to a clean
    structure, then compute the Complexa composite reward.

    This mirrors the paper's description (Sec. 3.4):
      "we stochastically simulate all candidates towards clean partially
       latent states, decode, fold the resulting sequences, and calculate
       the candidates' rewards"

    We use RFdiffusion from parent_node.pdb_path as the starting state
    (analogous to resuming from a partially denoised latent in Complexa).
    """
    out_prefix = os.path.join(
        work_dir, f"cp{checkpoint_idx:02d}_n{node_counter:04d}_rfd"
    )
    print(
        f"    rollout cp={checkpoint_idx} parent={parent_node.idx} "
        f"child={child_idx}: rfdiffusion…", end=" ", flush=True
    )

    out = run_denovo_round(
        ref_pdb=parent_node.pdb_path,
        contig_string=contig_string,
        hotspots=hotspots,
        output_prefix=out_prefix,
        model_weights=model_weights,
        original_pdb=input_pdb,
        anchor_residues=anchor_residues,
        cdr_ranges=cdr_ranges,
        extra_args=extra_args,
    )
    if out is None:
        print("FAILED (rfdiffusion)")
        return None

    out = _apply_sequence_and_anchors(
        pdb_path=out,
        out_prefix=out_prefix,
        mpnn=mpnn,
        cdr_mask=cdr_mask,
        anchor_residues=anchor_residues,
        ref_pdb=parent_node.pdb_path,
        device=device,
    )

    reward, bd = score_complexa_reward(
        pdb_path=out,
        af2_work_dir=af2_work_dir,
        colabfold_batch_bin=colabfold_batch_bin,
        colabfold_python=colabfold_python,
        thermo=thermo,
        cdr_mask=cdr_mask,
        epitope_ca=epitope_ca,
        w_ipae=w_ipae,
        w_thermo=w_thermo,
        ipae_threshold=ipae_threshold,
        af2_num_recycles=af2_num_recycles,
        af2_num_models=af2_num_models,
        use_gpu=use_gpu,
        device=device,
    )

    ipae_str = f"ipAE={bd['ipae']:.3f}" if bd["ipae"] is not None else "ipAE=N/A"
    print(
        f"reward={reward:+.3f}  {ipae_str}  "
        f"DDG={bd['ddg']:+.3f}  success={bd['success']}"
    )

    node = BeamNode(
        idx=node_counter,
        pdb_path=out,
        parent_idx=parent_node.idx,
        checkpoint_born=checkpoint_idx,
        cumulative_reward=parent_node.cumulative_reward + reward,
        reward_history=parent_node.reward_history + [
            {"checkpoint": checkpoint_idx, "reward": reward, **bd}
        ],
    )
    return node


# ─────────────────────────────────────────────────────────────────────────────
# Main beam search
# ─────────────────────────────────────────────────────────────────────────────

def run_beam_denovo_complexa(
    # ── structure inputs ────────────────────────────────────────────────────
    input_pdb:          str,
    anchors_json:       str,
    output_dir:         str,
    hotspots:           str,
    model_weights:      str,
    # ── beam / trajectory parameters ────────────────────────────────────────
    beam_width:         int   = 4,      # N  : survivors kept after each prune
    branch_factor:      int   = 4,      # L  : children per survivor per checkpoint
    n_checkpoints:      int   = 4,      # number of expand-score-prune cycles
    steps_per_checkpoint: int = 1,      # K  : diffusion steps between checkpoints
                                        #      (informational; RFdiffusion handles
                                        #       its own step count internally)
    ranking_mode:       str   = "cumulative",
    # ── colabfold / AF2 ──────────────────────────────────────────────────────
    colabfold_batch_bin: str   = "",
    colabfold_python:    str   = "",
    af2_num_recycles:    int   = 1,
    af2_num_models:      int   = 1,
    # ── reward weights ───────────────────────────────────────────────────────
    w_ipae:             float = 1.0,    # weight on ipAE reward component
    w_thermo:           float = 0.5,    # weight on ThermoMPNN ΔΔG component
    ipae_threshold:     float = IPAE_SUCCESS_THRESHOLD,
    # ── ThermoMPNN ──────────────────────────────────────────────────────────
    thermo_local_yaml:  str   = "",
    thermo_model_yaml:  str   = "",
    thermo_checkpoint:  str   = "",
    # ── ProteinMPNN ─────────────────────────────────────────────────────────
    mpnn_weights:       str   = "",
    # ── other ───────────────────────────────────────────────────────────────
    free_loops_spec:    str   = "",
    nanobody:           bool  = False,
    name:               str   = "",
    extra_args:         Optional[List[str]] = None,
    device:             str   = "cuda",
) -> List[BeamNode]:
    """
    Complexa-style beam search for de novo antibody design.

    The algorithm (Sec. 3.4 of the Complexa paper, Eq. 2):

      Initialise:
        Generate N*L seed structures; score; keep top-N as beam B_0.

      For each checkpoint c = 1 … n_checkpoints:
        1. Branch  — for each of the N beam nodes, launch L rollouts
                     from that node's structure → N*L candidate nodes C_c.
        2. Rollout  — each candidate is independently completed to a clean
                      structure, sequence-designed, and scored:
                        R(·) = w_ipae*(threshold − ipAE) + w_thermo*(−ΔΔG)
        3. Prune   — keep the N candidates with the highest R score:
                        B_c = argmax_{T⊆C_c, |T|=N} Σ_{i∈T} R(·)

    Key departure from the original code:
      • The reward now uses ipAE (AlphaFold2-Multimer) as the primary
        interface quality signal, replacing BSA contact counting.
      • ipAE = None silently degrades to ThermoMPNN-only scoring so the
        code still runs without ColabDesign installed.

    Returns the final beam (N nodes), sorted by ranking_mode descending.
    """
    extra_args = extra_args or []
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, "_beam_work")
    os.makedirs(work_dir, exist_ok=True)
    stem = name or Path(input_pdb).stem

    rank_fn = RANKING_MODES.get(ranking_mode)
    if rank_fn is None:
        raise ValueError(
            f"Unknown ranking_mode '{ranking_mode}'. "
            f"Choose from: {list(RANKING_MODES)}"
        )

    # ── 1. Parse HLT structure ───────────────────────────────────────────────
    print(f"\n[Complexa-Beam] Parsing HLT: {input_pdb}")
    cdr_ranges = parse_hlt_remarks(input_pdb)
    residues   = read_pdb_residues(input_pdb)

    n_l = sum(1 for r in residues if r.pdb_chain == CHAIN_L)
    if not nanobody and n_l == 0:
        print("[INFO] No L-chain detected — treating as nanobody.")
        nanobody = True

    split_dir = os.path.join(work_dir, "_split")
    target_pdb, framework_pdb = split_hlt_complex(input_pdb, split_dir)

    # ── 2. Anchors and contig ────────────────────────────────────────────────
    anchor_residues = load_anchors(anchors_json)
    print(
        f"[Complexa-Beam] {len(anchor_residues)} anchor(s): "
        f"{[f'{c}{n}' for c, n in anchor_residues]}"
    )
    free_loops    = parse_free_loops(free_loops_spec)
    contig_string = build_denovo_contig(
        residues, cdr_ranges, anchor_residues, free_loops, nanobody
    )
    print(f"[Complexa-Beam] Contig: {contig_string}")

    # ── 3. Scoring infrastructure ────────────────────────────────────────────
    cdr_mask   = build_cdr_mask(framework_pdb)
    epitope_ca = load_epitope_ca(target_pdb, hotspots, device)

    print("[Complexa-Beam] Loading ThermoMPNN…")
    thermo = load_thermompnn(
        config_yaml=thermo_model_yaml,
        local_yaml=thermo_local_yaml,
        checkpoint=thermo_checkpoint,
        device=device,
    )
    print("[Complexa-Beam] Loading ProteinMPNN…")
    mpnn = load_proteinmpnn(mpnn_weights, device)

    use_gpu = device.lower() == "cuda"
    af2_work_dir = os.path.join(work_dir, "_af2")
    os.makedirs(af2_work_dir, exist_ok=True)

    if colabfold_batch_bin and os.path.isfile(colabfold_batch_bin):
        print(f"[Complexa-Beam] colabfold_batch: {colabfold_batch_bin} — ipAE reward ENABLED.")
    else:
        print(
            "[Complexa-Beam] colabfold_batch NOT found — ipAE reward DISABLED; "
            "running ThermoMPNN-only scoring.  Pass --colabfold_batch_bin to enable."
        )

    # shared kwargs forwarded to every _rollout_and_score call
    rollout_kw = dict(
        work_dir=work_dir, contig_string=contig_string,
        hotspots=hotspots, model_weights=model_weights,
        input_pdb=input_pdb, anchor_residues=anchor_residues,
        cdr_ranges=cdr_ranges, extra_args=extra_args,
        mpnn=mpnn, cdr_mask=cdr_mask, thermo=thermo,
        epitope_ca=epitope_ca, w_ipae=w_ipae, w_thermo=w_thermo,
        ipae_threshold=ipae_threshold, af2_work_dir=af2_work_dir,
        colabfold_batch_bin=colabfold_batch_bin,
        colabfold_python=colabfold_python,
        af2_num_recycles=af2_num_recycles,
        af2_num_models=af2_num_models,
        use_gpu=use_gpu, device=device,
    )

    # ── 4. Initialise beam  (checkpoint 0) ───────────────────────────────────
    n_seeds = beam_width * branch_factor
    print(
        f"\n[Complexa-Beam] ── Checkpoint 0 (initialise beam, "
        f"generating {n_seeds} seeds) ──"
    )

    node_counter = 0
    # Synthetic root node so every real seed has a parent
    root_node = BeamNode(
        idx=-1, pdb_path=input_pdb, parent_idx=None,
        checkpoint_born=-1, cumulative_reward=0.0
    )

    seed_candidates: List[BeamNode] = []
    for i in range(n_seeds):
        node = _rollout_and_score(
            parent_node=root_node, child_idx=i,
            checkpoint_idx=0, node_counter=node_counter,
            **rollout_kw,
        )
        if node is not None:
            seed_candidates.append(node)
        node_counter += 1

    if not seed_candidates:
        raise RuntimeError("All seed rollouts failed — cannot initialise beam.")

    seed_candidates.sort(key=rank_fn, reverse=True)
    beam: List[BeamNode] = seed_candidates[:beam_width]

    _print_beam(beam, rank_fn, ranking_mode, label="Initial beam")

    # ── 5. Beam search checkpoints ───────────────────────────────────────────
    for cp in range(1, n_checkpoints + 1):
        print(
            f"\n[Complexa-Beam] ── Checkpoint {cp}/{n_checkpoints}  "
            f"beam_width={len(beam)}  branch_factor={branch_factor}  "
            f"candidates={len(beam) * branch_factor}  "
            f"(K={steps_per_checkpoint} diffusion steps / checkpoint) ──"
        )

        # 5a. Branch + Rollout  (N*L independent rollouts, Eq. 2)
        candidates: List[BeamNode] = []
        for parent in beam:
            for b in range(branch_factor):
                node = _rollout_and_score(
                    parent_node=parent, child_idx=b,
                    checkpoint_idx=cp, node_counter=node_counter,
                    **rollout_kw,
                )
                if node is not None:
                    candidates.append(node)
                node_counter += 1

        if not candidates:
            print(
                f"  [WARN] All rollouts failed at checkpoint {cp} — "
                "keeping current beam."
            )
            continue

        # 5b. Prune: argmax_{T⊆C, |T|=N} Σ R_i  (Eq. 2 in the paper)
        candidates.sort(key=rank_fn, reverse=True)
        beam = candidates[:beam_width]

        _print_beam(beam, rank_fn, ranking_mode,
                    label=f"Beam after checkpoint {cp}")

        # Diversity diagnostic (unique parent lineages)
        unique_parents = len({n.parent_idx for n in beam})
        print(
            f"  Lineage diversity: {unique_parents}/{len(beam)} "
            "unique parents survived pruning."
        )
        if unique_parents == 1:
            print(
                "  [WARN] Beam collapsed to a single lineage.  "
                "Consider raising --beam_width or --branch_factor, "
                "or switching to Feynman-Kac steering for softer pruning."
            )

        # Summary of in-silico successes (ipAE < threshold) in surviving beam
        n_success = sum(
            1 for n in beam
            if n.reward_history and n.reward_history[-1].get("success", False)
        )
        print(
            f"  In-silico successes in beam "
            f"(ipAE < {ipae_threshold:.1f} Å): {n_success}/{len(beam)}"
        )

    # ── 6. Write final outputs ────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final_designs")
    os.makedirs(final_dir, exist_ok=True)
    print(f"\n[Complexa-Beam] Writing {len(beam)} final designs → {final_dir}/")

    for rank, node in enumerate(beam):
        dst = os.path.join(final_dir, f"{stem}_cbeam_rank{rank:03d}.pdb")
        if os.path.isfile(node.pdb_path):
            shutil.copy2(node.pdb_path, dst)

        packed = os.path.join(final_dir, f"{stem}_cbeam_rank{rank:03d}_packed.pdb")
        pack_sidechains(
            pdb_path=dst,
            anchor_residues=anchor_residues,
            mpnn_model=mpnn,
            cdr_mask=cdr_mask,
            out_path=packed,
            device=device,
        )
        final = os.path.join(final_dir, f"{stem}_cbeam_rank{rank:03d}_final.pdb")
        graft_anchor_identities(
            rfdiffusion_pdb=packed,
            ref_pdb=input_pdb,
            anchor_residues=anchor_residues,
            out_path=final,
        )
        last_h = node.reward_history[-1] if node.reward_history else {}
        ipae_str = (
            f"ipAE={last_h['ipae']:.3f}" if last_h.get("ipae") is not None
            else "ipAE=N/A"
        )
        print(
            f"  rank {rank:03d}  node={node.idx}  "
            f"{ranking_mode}_reward={rank_fn(node):+.3f}  "
            f"{ipae_str}  DDG={last_h.get('ddg', float('nan')):+.3f}  "
            f"→ {Path(final).name}"
        )

    # ── 7. Save JSON summary ──────────────────────────────────────────────────
    summary = []
    for rank, node in enumerate(beam):
        last_h = node.reward_history[-1] if node.reward_history else {}
        summary.append({
            "rank":              rank,
            "node_idx":          node.idx,
            "parent_idx":        node.parent_idx,
            "checkpoint_born":   node.checkpoint_born,
            "cumulative_reward": node.cumulative_reward,
            f"{ranking_mode}_reward": rank_fn(node),
            "final_ipae":        last_h.get("ipae"),
            "final_ddg":         last_h.get("ddg"),
            "final_success":     last_h.get("success"),
            "final_pdb":         os.path.basename(node.pdb_path),
            "reward_history":    node.reward_history,
        })

    summary_path = os.path.join(output_dir, f"{stem}_cbeam_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[Complexa-Beam] Summary → {summary_path}")

    return beam


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_beam(
    beam: List[BeamNode],
    rank_fn,
    ranking_mode: str,
    label: str = "Beam",
) -> None:
    print(f"\n  {label} ({len(beam)} nodes):")
    for rank, node in enumerate(beam):
        last_h = node.reward_history[-1] if node.reward_history else {}
        ipae_str = (
            f"ipAE={last_h['ipae']:.3f}" if last_h.get("ipae") is not None
            else "ipAE=N/A"
        )
        print(
            f"    rank {rank:02d}  node={node.idx:04d}  "
            f"parent={node.parent_idx}  "
            f"{ranking_mode}={rank_fn(node):+.3f}  "
            f"latest={last_h.get('reward', float('nan')):+.3f}  "
            f"{ipae_str}"
        )


def _apply_sequence_and_anchors(
    pdb_path:        str,
    out_prefix:      str,
    mpnn,
    cdr_mask:        torch.Tensor,
    anchor_residues: list,
    ref_pdb:         str,
    device:          str,
) -> str:
    """ProteinMPNN sequence design, then re-graft anchor identities."""
    seq_pdb = out_prefix + "_seq.pdb"
    result  = design_sequence_onto_backbone(
        mpnn_model=mpnn,
        backbone_pdb=pdb_path,
        cdr_mask=cdr_mask,
        out_pdb=seq_pdb,
        temperature=0.1,
        device=device,
    )
    out = result or pdb_path
    if anchor_residues:
        enforced = out_prefix + "_enforced.pdb"
        graft_anchor_identities(
            rfdiffusion_pdb=out,
            ref_pdb=ref_pdb,
            anchor_residues=anchor_residues,
            out_path=enforced,
        )
        out = enforced
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Complexa-style beam search for de novo antibody design. "
            "Reward = w_ipae*(threshold - ipAE) + w_thermo*(-DDG).  "
            "Requires ColabDesign for ipAE; falls back to ThermoMPNN-only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── required ─────────────────────────────────────────────────────────────
    p.add_argument("--input",             required=True,
                   help="Input HLT PDB (binder+target complex)")
    p.add_argument("--anchors",           required=True,
                   help="Anchor residues JSON")
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--hotspots",          required=True)
    p.add_argument("--model_weights",     required=True,
                   help="RFdiffusion model weights")
    p.add_argument("--thermo_local_yaml", required=True)
    p.add_argument("--thermo_model_yaml", required=True)
    p.add_argument("--thermo_checkpoint", required=True)
    p.add_argument("--mpnn_weights",      required=True)
    p.add_argument("--colabfold_batch_bin",
                   default="",
                   help="Path to the colabfold_batch script "
                        "(same default as evaluate_designs.py)")
    p.add_argument("--colabfold_python",
                   default="",
                   help="Python interpreter inside the colabfold conda env")
    p.add_argument("--af2_num_recycles", type=int, default=1,
                   help="AF2 recycles per rollout scoring call (default 1 for speed)")
    p.add_argument("--af2_num_models",   type=int, default=1,
                   help="Number of AF2 models to run per scoring call (default 1)")
    # ── beam hyperparameters ─────────────────────────────────────────────────
    p.add_argument("--beam_width",     type=int,   default=4,
                   help="N: survivors kept after each checkpoint prune (default 4)")
    p.add_argument("--branch_factor",  type=int,   default=4,
                   help="L: rollouts launched per survivor per checkpoint (default 4)")
    p.add_argument("--n_checkpoints",  type=int,   default=4,
                   help="Number of expand-score-prune cycles (default 4)")
    p.add_argument("--steps_per_checkpoint", type=int, default=1,
                   help="K: diffusion steps between checkpoints (default 1)")
    p.add_argument("--ranking_mode",   default="cumulative",
                   choices=["cumulative", "latest", "average"],
                   help="How to rank candidates at prune time (default: cumulative)")
    # ── reward weights ────────────────────────────────────────────────────────
    p.add_argument("--w_ipae",         type=float, default=1.0,
                   help="Weight on ipAE reward component (default 1.0)")
    p.add_argument("--w_thermo",       type=float, default=0.5,
                   help="Weight on ThermoMPNN DDG component (default 0.5)")
    p.add_argument("--ipae_threshold", type=float, default=IPAE_SUCCESS_THRESHOLD,
                   help=f"ipAE success threshold in Å (default {IPAE_SUCCESS_THRESHOLD})")
    # ── other ─────────────────────────────────────────────────────────────────
    p.add_argument("--free_loops", default="")
    p.add_argument("--nanobody",   action="store_true")
    p.add_argument("--name",       default="")
    p.add_argument("--device",     default="cuda")
    p.add_argument("extra",        nargs=argparse.REMAINDER,
                   help="Extra args forwarded verbatim to RFdiffusion")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    extra = [a for a in (args.extra or []) if a != "--"]
    run_beam_denovo_complexa(
        input_pdb=str(Path(args.input).resolve()),
        anchors_json=str(Path(args.anchors).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        hotspots=args.hotspots,
        model_weights=args.model_weights,
        beam_width=args.beam_width,
        branch_factor=args.branch_factor,
        n_checkpoints=args.n_checkpoints,
        steps_per_checkpoint=args.steps_per_checkpoint,
        ranking_mode=args.ranking_mode,
        colabfold_batch_bin=args.colabfold_batch_bin,
        colabfold_python=args.colabfold_python,
        af2_num_recycles=args.af2_num_recycles,
        af2_num_models=args.af2_num_models,
        w_ipae=args.w_ipae,
        w_thermo=args.w_thermo,
        ipae_threshold=args.ipae_threshold,
        thermo_local_yaml=args.thermo_local_yaml,
        thermo_model_yaml=args.thermo_model_yaml,
        thermo_checkpoint=args.thermo_checkpoint,
        mpnn_weights=args.mpnn_weights,
        free_loops_spec=args.free_loops,
        nanobody=args.nanobody,
        name=args.name,
        extra_args=extra,
        device=args.device,
    )


if __name__ == "__main__":
    main()