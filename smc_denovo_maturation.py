"""
smc_denovo_maturation.py

SMC-guided de novo antibody design with motif-anchored CDR residues.

Replaces partial diffusion with full de novo RFdiffusion, using the
motif scaffolding mechanism to anchor specific CDR residues:

  - Anchor residues appear in contigmap.contigs as fixed segments
    (e.g. "H32-35") — their backbone frames are held exactly as in the
    input PDB throughout the full diffusion trajectory.
  - Non-anchored CDR positions appear as free-length ranges (e.g. "3-8").
  - No partial_T, no provide_seq.

SMC operates over complete trajectories (N parallel rfdiffusion runs),
scoring finished PDBs with ThermoMPNN + BSA and resampling survivors
between rounds. Each round is a fully independent de novo diffusion run
that starts from pure noise — the "inheritance" between rounds comes
entirely from the resampling step selecting high-scoring structures as
new reference inputs for motif placement.

NOTE ON ANCESTRY:
  Unlike partial diffusion, full de novo diffusion cannot literally
  continue from a previous structure. Instead, the winner PDBs from
  round r become the *reference PDB* (inference.input_pdb) for round
  r+1, so that the anchor residue coordinates are taken from the
  best structures found so far — a form of iterated enrichment.

Usage:
    python smc_denovo_maturation.py \
        --input   complex.pdb \
        --anchors anchors.json \
        --output_dir  out/ \
        --hotspots    "T305,T456" \
        --model_weights /path/to/AbModel.pt \
        --thermo_local_yaml ThermoMPNN/examples/configs/local.yaml \
        --thermo_model_yaml ThermoMPNN/examples/configs/single.yaml \
        --thermo_checkpoint ThermoMPNN/models/thermoMPNN_default.pt \
        --mpnn_weights include/proteinmpnn_weights/v_48_020.pt \
        --n_particles 16 --n_rounds 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

# ── Re-use helpers from partial_diffusion_maturation.py ──────────────────────
from partial_diffusion_maturation import (
    CHAIN_H, CHAIN_L, CHAIN_T,
    CDR_NAMES_ALL,
    CdrRange, ResidueInfo,
    build_residue_lookup,
    graft_target_sequence,
    load_anchors,
    mask_anchors_in_hlt,
    parse_free_loops,
    parse_hlt_remarks,
    read_pdb_residues,
    split_hlt_complex,
)

# ── ProteinMPNN (bundled with RFantibody) ─────────────────────────────────────
# protein_mpnn_utils.py lives at src/rfantibody/proteinmpnn/ inside the
# RFantibody clone. Set RFANTIBODY_ROOT to the repo root.
RFANTIBODY_ROOT = os.environ.get("RFANTIBODY_ROOT", "")
if not RFANTIBODY_ROOT:
    raise EnvironmentError(
        "RFANTIBODY_ROOT is not set. "
        "Export it before running, e.g.:\n"
        "  export RFANTIBODY_ROOT=/absolute/path/to/RFantibody"
    )
RFANTIBODY_ROOT = str(Path(RFANTIBODY_ROOT).resolve())
PMPNN_PATH = str(Path(RFANTIBODY_ROOT) / "src" / "rfantibody" / "proteinmpnn")
if not (Path(PMPNN_PATH) / "util_protein_mpnn.py").exists():
    raise FileNotFoundError(
        f"util_protein_mpnn not found at '{PMPNN_PATH}'. "
        "Check that RFANTIBODY_ROOT points to the cloned RFantibody directory."
    )
if PMPNN_PATH not in sys.path:
    sys.path.insert(0, PMPNN_PATH)

from protein_mpnn_utils import (
    parse_PDB,
    tied_featurize,
    ProteinMPNN,
    _S_to_seq,
)

# ── ThermoMPNN ────────────────────────────────────────────────────────────────
# ThermoMPNN is a flat repo — all modules live directly in the root directory:
#   transfer_model.py, datasets.py, model_utils.py, protein_mpnn_utils.py
# Set THERMOMPNN_ROOT to the absolute path of your cloned ThermoMPNN directory,
# e.g. export THERMOMPNN_ROOT=/home/user/ThermoMPNN
THERMOMPNN_ROOT = os.environ.get("THERMOMPNN_ROOT", "")
if not THERMOMPNN_ROOT:
    raise EnvironmentError(
        "THERMOMPNN_ROOT is not set. "
        "Export it before running, e.g.:\n"
        "  export THERMOMPNN_ROOT=/absolute/path/to/ThermoMPNN"
    )
THERMOMPNN_ROOT = str(Path(THERMOMPNN_ROOT).resolve())
if not (Path(THERMOMPNN_ROOT) / "transfer_model.py").exists():
    raise FileNotFoundError(
        f"transfer_model.py not found in THERMOMPNN_ROOT='{THERMOMPNN_ROOT}'. "
        "Check that THERMOMPNN_ROOT points to the cloned ThermoMPNN directory."
    )
if THERMOMPNN_ROOT not in sys.path:
    sys.path.insert(0, THERMOMPNN_ROOT)

from transfer_model import TransferModel            # plain nn.Module, not Lightning
from datasets import Mutation, parse_pdb_cached     # confirmed fields: position, wildtype, mutation, ddG


# ─────────────────────────────────────────────────────────────────────────────
# Contig builder — de novo with fixed anchor motifs
# ─────────────────────────────────────────────────────────────────────────────

def build_denovo_contig(
    residues:            List[ResidueInfo],
    cdr_ranges:          Dict[str, CdrRange],
    anchor_residues:     List[Tuple[str, int]],
    free_loop_overrides: Dict[str, Tuple[int, int]],
    nanobody:            bool = False,
) -> str:
    """
    Build a RFdiffusion contig string for full de novo diffusion where
    anchor residues are fixed motif segments.

    Rules:
      - Framework residues (non-CDR H/L): always fixed as "ChainStart-End"
      - CDR residues that are anchors: fixed as "ChainStart-End"
      - CDR residues that are NOT anchors: free-length range "min-max"
        (from free_loop_overrides or ±2 of the original loop length)
      - Target (chain T): always fixed as "TStart-End"
      - Chain breaks between antibody and target: "/0 "

    The resulting string is passed to contigmap.contigs=[...] and tells
    RFdiffusion to scaffold the free CDR segments around the fixed motifs
    from a pure-noise starting point.
    """
    anchor_set = {(c, r) for c, r in anchor_residues}

    # Map pose_idx → CDR name for fast lookup
    cdr_pose_idx_to_name: Dict[int, str] = {}
    for name, cr in cdr_ranges.items():
        for idx in range(cr.start, cr.end + 1):
            cdr_pose_idx_to_name[idx] = name

    h_res = [r for r in residues if r.pdb_chain == CHAIN_H]
    l_res = [r for r in residues if r.pdb_chain == CHAIN_L]
    t_res = [r for r in residues if r.pdb_chain == CHAIN_T]

    def chain_segments(chain_residues: List[ResidueInfo]) -> str:
        """Convert one chain's residues to a contig token sequence."""
        tokens: List[str] = []
        i = 0
        while i < len(chain_residues):
            r = chain_residues[i]
            cdr_name = cdr_pose_idx_to_name.get(r.pose_idx)

            if cdr_name is None:
                # Framework run — collect contiguous non-CDR residues
                run = []
                while (i < len(chain_residues) and
                       cdr_pose_idx_to_name.get(chain_residues[i].pose_idx) is None):
                    run.append(chain_residues[i])
                    i += 1
                # Fixed motif segment
                ch = run[0].pdb_chain
                tokens.append(f"{ch}{run[0].pdb_resnum}-{run[-1].pdb_resnum}")
            else:
                # CDR loop — collect all residues in this loop
                cr = cdr_ranges[cdr_name]
                loop_res = []
                while (i < len(chain_residues) and
                       chain_residues[i].pose_idx <= cr.end):
                    loop_res.append(chain_residues[i])
                    i += 1

                # Split loop into fixed-anchor runs and free gaps
                tokens.extend(
                    _cdr_to_contig_tokens(
                        loop_res, anchor_set, cdr_name,
                        len(loop_res), free_loop_overrides
                    )
                )

        return "/".join(tokens)

    h_seg = chain_segments(h_res)

    if t_res:
        t_first, t_last = t_res[0], t_res[-1]
        t_seg = f"{CHAIN_T}{t_first.pdb_resnum}-{t_last.pdb_resnum}"
    else:
        t_seg = ""

    if nanobody or not l_res:
        return f"{h_seg}/0 {t_seg}" if t_seg else h_seg

    l_seg = chain_segments(l_res)
    if t_seg:
        return f"{h_seg}/0 {l_seg}/0 {t_seg}"
    return f"{h_seg}/0 {l_seg}"


def _cdr_to_contig_tokens(
    loop_res:            List[ResidueInfo],
    anchor_set:          set,
    cdr_name:            str,
    orig_len:            int,
    free_loop_overrides: Dict[str, Tuple[int, int]],
) -> List[str]:
    """
    Convert a single CDR loop's residues into contig tokens.

    Contiguous anchor sub-runs become fixed motif tokens ("ChainS-E").
    Contiguous non-anchor sub-runs become free-length tokens ("min-max").

    The free-length range for the entire loop is from free_loop_overrides
    if provided, otherwise (max(1, orig_len-2), orig_len+2).
    The free residue budget is distributed proportionally across gaps.
    """
    tokens: List[str] = []
    min_len, max_len = free_loop_overrides.get(
        cdr_name, (max(1, orig_len - 2), orig_len + 2)
    )

    # Count total free (non-anchor) residues in this loop
    n_free = sum(1 for r in loop_res if (r.pdb_chain, r.pdb_resnum) not in anchor_set)
    n_free = max(n_free, 1)   # avoid division by zero

    i = 0
    while i < len(loop_res):
        r = loop_res[i]
        is_anchor = (r.pdb_chain, r.pdb_resnum) in anchor_set

        if is_anchor:
            # Collect contiguous anchor run
            run_start = r
            run_end   = r
            while (i + 1 < len(loop_res) and
                   (loop_res[i+1].pdb_chain, loop_res[i+1].pdb_resnum) in anchor_set):
                i += 1
                run_end = loop_res[i]
            tokens.append(f"{run_start.pdb_chain}"
                          f"{run_start.pdb_resnum}-{run_end.pdb_resnum}")
            i += 1
        else:
            # Collect contiguous free run; scale range proportionally
            gap_size = 0
            while (i < len(loop_res) and
                   (loop_res[i].pdb_chain, loop_res[i].pdb_resnum) not in anchor_set):
                gap_size += 1
                i += 1
            frac     = gap_size / n_free
            gap_min  = max(1, round(min_len * frac))
            gap_max  = max(gap_min, round(max_len * frac))
            tokens.append(f"{gap_min}-{gap_max}")

    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# rfdiffusion CLI command builder (de novo — no partial_T, no provide_seq)
# ─────────────────────────────────────────────────────────────────────────────

def _find_inference_script() -> str:
    """Locate rfdiffusion_inference.py using the same search as Step 1."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "src" / "rfantibody" / "rfdiffusion" / "rfdiffusion_inference.py",
        script_dir.parent / "src" / "rfantibody" / "rfdiffusion" / "rfdiffusion_inference.py",
        script_dir / "rfdiffusion_inference.py",
        Path(RFANTIBODY_ROOT) / "src" / "rfantibody" / "rfdiffusion" / "rfdiffusion_inference.py",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    env = os.environ.get("RFANTIBODY_INFERENCE_SCRIPT", "")
    if env and os.path.isfile(env):
        return env
    raise FileNotFoundError(
        "Cannot locate rfdiffusion_inference.py. "
        "Set RFANTIBODY_INFERENCE_SCRIPT to its absolute path."
    )


def build_denovo_rfdiffusion_cmd(
    input_pdb:      str,    # reference PDB (for motif coords)
    output_prefix:  str,
    contig_string:  str,
    hotspots:       str,
    model_weights:  str,
    num_designs:    int = 1,
    extra_args:     Optional[List[str]] = None,
) -> List[str]:
    """
    Build the rfdiffusion CLI command for full de novo diffusion with
    fixed anchor motifs.

    Key differences from partial diffusion:
      - No diffuser.partial_T
      - No contigmap.provide_seq
      - contigmap.contigs encodes anchors as fixed segments
    """
    inference_script = _find_inference_script()
    cmd = [
        sys.executable,
        inference_script,
        f"inference.input_pdb={input_pdb}",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={num_designs}",
    ]
    if model_weights:
        cmd.append(f"inference.ckpt_override_path={model_weights}")
    if contig_string:
        cmd.append(f"'contigmap.contigs=[{contig_string}]'")
    if hotspots:
        cmd.append(f"'ppi.hotspot_res=[{hotspots}]'")
    cmd.extend(extra_args or [])
    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# CDR mask for scoring (reads REMARK lines from framework PDB)
# ─────────────────────────────────────────────────────────────────────────────

def build_cdr_mask(pdb_path: str) -> torch.Tensor:
    remark_re = re.compile(
        r"^REMARK\s+PDBinfo-LABEL:\s+(\d+)\s+(H[123]|L[123])\s*$"
    )
    cdr_abs, n_res = [], 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    n_res = max(n_res, int(line[22:26]))
                except ValueError:
                    pass
            if line.startswith("REMARK PDBinfo-LABEL:"):
                m = remark_re.match(line.strip())
                if m:
                    cdr_abs.append(int(m.group(1)) - 1)   # 0-indexed
    mask = torch.zeros(n_res, dtype=torch.bool)
    for i in cdr_abs:
        if 0 <= i < n_res:
            mask[i] = True
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Epitope Ca loader
# ─────────────────────────────────────────────────────────────────────────────

def load_epitope_ca(pdb_path: str, hotspot_str: str,
                    device: str) -> torch.Tensor:
    hs_list = [h.strip() for h in hotspot_str.split(",") if h.strip()]
    ca_lookup: Dict[Tuple[str, int], List[float]] = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ch = line[21]; rn = int(line[22:26])
                ca_lookup[(ch, rn)] = [
                    float(line[30:38]), float(line[38:46]), float(line[46:54])
                ]
    coords = []
    for hs in hs_list:
        key = (hs[0], int(hs[1:]))
        if key in ca_lookup:
            coords.append(ca_lookup[key])
        else:
            print(f"  [WARN] Hotspot {hs} not found in {pdb_path}")
    if not coords:
        raise ValueError(f"No epitope Ca coords for '{hotspot_str}'")
    return torch.tensor(coords, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_thermompnn(
    config_yaml: str,    # ThermoMPNN/config.yaml  (model hyperparameters)
    local_yaml:  str,    # ThermoMPNN/local.yaml   (data/weight paths)
    checkpoint:  str,    # ThermoMPNN/models/thermoMPNN_default.pt
    device:      str,
) -> TransferModel:
    """
    Load ThermoMPNN. The model class is TransferModel(nn.Module) — a plain
    PyTorch module, not a Lightning module — so we use torch.load + 
    instantiate-then-load-state-dict rather than load_from_checkpoint.

    TransferModel.__init__ takes (cfg) and internally calls get_protein_mpnn(cfg)
    to build the ProteinMPNN backbone, so cfg must have the correct path to
    vanilla_model_weights/ set in local.yaml.
    """
    cfg = OmegaConf.merge(OmegaConf.load(local_yaml), OmegaConf.load(config_yaml))
    model = TransferModel(cfg)
    state = torch.load(checkpoint, map_location=device)
    # Checkpoint may be a raw state_dict or wrapped in a dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # Strip "model." prefix added by PyTorch Lightning when the checkpoint was
    # saved from a LightningModule that wrapped TransferModel as self.model
    if any(k.startswith("model.") for k in state):
        state = {k.removeprefix("model."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.eval().to(device)


def load_proteinmpnn(weights_path: str, device: str) -> ProteinMPNN:
    ckpt = torch.load(weights_path, map_location="cpu")
    m = ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=ckpt.get("num_edges", 48),
    )
    m.load_state_dict(ckpt["model_state_dict"])
    return m.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence design onto backbone + annotated PDB writer
# ─────────────────────────────────────────────────────────────────────────────

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA1_TO_3 = {v: k for k, v in AA3_TO_1.items()}


def design_sequence_onto_backbone(
    mpnn_model:  ProteinMPNN,
    backbone_pdb: str,
    cdr_mask:    torch.Tensor,   # (L,) bool over H+L residues in order
    out_pdb:     str,
    temperature: float = 0.1,
    device:      str   = "cuda",
) -> Optional[str]:
    """
    Run ProteinMPNN on a Gly-backbone PDB, design CDR residues, and write a
    new PDB with residue names updated to reflect the designed sequence.

    Steps:
      1. parse_PDB() → featurize with tied_featurize()
      2. model.sample() → one-letter sequence string
      3. Rewrite ATOM records in the backbone PDB, replacing residue names
         for CDR positions with the designed amino acids.

    The output PDB can then be passed directly to ThermoMPNN's load_pdb(),
    which reads residue names rather than coordinates to determine sequence.

    Returns out_pdb on success, None on failure.
    """
    try:
        pdb_dicts = parse_PDB(backbone_pdb, ca_only=False)
        if not pdb_dicts:
            raise ValueError("parse_PDB returned empty list")
        pdb_dict = pdb_dicts[0]
    except Exception as e:
        print(f"    [WARN] parse_PDB failed: {e}")
        return None

    # Identify antibody chains (H, L) and target chain (T)
    all_chains = sorted({
        item[-1:] for item in pdb_dict if item.startswith("seq_chain")
    })
    ab_chains  = [c for c in all_chains if c in ("H", "L")]
    tgt_chains = [c for c in all_chains if c == "T"]

    if not ab_chains:
        print("    [WARN] No H/L chains found in PDB for ProteinMPNN")
        return None

    # chain_id_dict: H and L are designable; T is fixed context
    chain_id_dict = {pdb_dict["name"]: (ab_chains, tgt_chains)}

    # Fix all framework (non-CDR) positions; CDR positions are designable.
    # cdr_mask is 0-indexed over the concatenated H+L sequence length.
    fixed_pos = [i for i in range(len(cdr_mask)) if not cdr_mask[i].item()]
    fixed_positions_dict = {
        pdb_dict["name"]: {c: fixed_pos for c in ab_chains}
    }

    try:
        (X, S, mask, lengths, chain_M, chain_encoding_all,
         chain_list_list, visible_list_list, masked_list_list,
         masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
         residue_idx, dihedral_mask, tied_pos_list_of_lists_list,
         pssm_coef, pssm_bias, pssm_log_odds_all,
         bias_by_res_all, tied_beta) = tied_featurize(
            [pdb_dict], device, chain_id_dict, fixed_positions_dict,
            omit_AA_dict=None, tied_positions_dict=None,
            pssm_dict=None, bias_by_res_dict=None,
        )
    except Exception as e:
        print(f"    [WARN] tied_featurize failed: {e}")
        return None

    randn = torch.randn(chain_M.shape, device=device)
    with torch.no_grad():
        sample = mpnn_model.sample(
            X, randn, S, chain_M, chain_encoding_all, residue_idx,
            mask=mask, temperature=temperature,
            omit_AAs_np=np.zeros(21), bias_AAs_np=np.zeros(21),
            chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
            pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0,
            pssm_log_odds_flag=False,
            pssm_log_odds_mask=(pssm_log_odds_all > 0).float(),
            pssm_bias_flag=False,
            tied_pos=tied_pos_list_of_lists_list[0],
            tied_beta=tied_beta, bias_by_res=bias_by_res_all,
        )
    seq_1letter: str = _S_to_seq(sample["S"][0], mask[0])

    # Build a lookup: (chain, resnum) → designed one-letter AA.
    # parse_PDB concatenates H then L residues in file order; we walk
    # the same order to assign sequence positions.
    chain_resnum_to_aa: Dict[Tuple[str, int], str] = {}
    seq_pos = 0
    seen: set = set()
    with open(backbone_pdb) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            ch = line[21]
            if ch not in ("H", "L"):
                continue
            try:
                rn = int(line[22:26])
            except ValueError:
                continue
            key = (ch, rn)
            if key not in seen:
                seen.add(key)
                if seq_pos < len(seq_1letter):
                    chain_resnum_to_aa[key] = seq_1letter[seq_pos]
                seq_pos += 1

    # Rewrite backbone PDB with updated residue names
    out_lines: List[str] = []
    with open(backbone_pdb) as fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                ch = line[21]
                try:
                    rn = int(line[22:26])
                except ValueError:
                    out_lines.append(line)
                    continue
                aa1 = chain_resnum_to_aa.get((ch, rn))
                if aa1 and aa1 in AA1_TO_3:
                    aa3 = AA1_TO_3[aa1].ljust(3)
                    line = line[:17] + aa3 + line[20:]
                out_lines.append(line)
            else:
                out_lines.append(line)

    try:
        with open(out_pdb, "w") as fh:
            fh.writelines(out_lines)
    except Exception as e:
        print(f"    [WARN] Could not write sequence-annotated PDB: {e}")
        return None

    return out_pdb


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

THERMO_AA = "ACDEFGHIKLMNPQRSTVWY"


def score_thermompnn(
    thermo:   TransferModel,
    thermo_cfg: object,         # OmegaConf cfg — passed to parse_pdb_cached
    pdb_path: str,
    cdr_mask: torch.Tensor,
    device:   str,
) -> Optional[float]:
    """
    Score a PDB with ThermoMPNN using the confirmed v1 API:

        TransferModelPL.forward(pdb, mutations, tied_feat=True)

    Confirmed from datasets.py grep:
      - parse_pdb_cached(cfg, pdb_file) → calls parse_PDB(pdb_file); cfg unused
      - Mutation fields: position (int), wildtype (str), mutation (str),
                         ddG (Optional[float] = None)   ← no pdb_file field

    We sweep all 19 non-WT amino acids at each CDR position and return the
    mean predicted DDG. Lower DDG = more stable; negated for the reward signal.
    """
    try:
        pdb = parse_pdb_cached(thermo_cfg, pdb_path)
    except Exception as e:
        print(f"    [WARN] parse_pdb_cached failed: {e}")
        return None

    seq = pdb.get("seq", "")
    if not seq:
        print(f"    [WARN] Empty sequence for {pdb_path}")
        return None

    cdr_positions = cdr_mask.nonzero(as_tuple=True)[0].tolist()
    mutations = []
    for pos in cdr_positions:
        if pos >= len(seq):
            continue
        wt = seq[pos]
        if wt == "-" or wt not in THERMO_AA:
            continue
        for mut in THERMO_AA:
            if mut == wt:
                continue
            mutations.append(Mutation(
                position=pos,
                wildtype=wt,
                mutation=mut,
                ddG=None,       # placeholder; model predicts this
            ))

    if not mutations:
        return None

    try:
        with torch.no_grad():
            preds = thermo(pdb, mutations, tied_feat=True)
        # forward() returns a tensor of shape (len(mutations),) or similar
        if isinstance(preds, dict):
            preds = preds.get("ddG", next(iter(preds.values())))
        ddgs = preds.detach().cpu().numpy().flatten()
        return float(np.mean(ddgs))
    except Exception as e:
        print(f"    [WARN] ThermoMPNN forward failed: {e}")
        return None


def score_bsa(pdb_path: str, cdr_mask: torch.Tensor,
              epitope_ca: torch.Tensor, dist: float = 8.0) -> float:
    """Ca-Ca contact count between CDR and epitope residues."""
    ca = []
    with open(pdb_path) as f:
        for line in f:
            if (line.startswith("ATOM") and
                    line[12:16].strip() == "CA" and
                    line[21] in (CHAIN_H, CHAIN_L)):
                ca.append([float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54])])
    if not ca:
        return 0.0
    ca_t = torch.tensor(ca, dtype=torch.float32)
    n = min(len(ca_t), len(cdr_mask))
    cdr_ca = ca_t[:n][cdr_mask[:n]]
    if cdr_ca.shape[0] == 0:
        return 0.0
    return float((torch.cdist(cdr_ca, epitope_ca.float().cpu()) < dist).sum())


def score_composite(
    pdb_path:   str,
    thermo:     TransferModel,
    thermo_cfg: object,          # OmegaConf cfg passed to parse_pdb_cached
    cdr_mask:   torch.Tensor,
    epitope_ca: torch.Tensor,
    w_thermo:   float,
    w_bsa:      float,
    device:     str,
) -> Tuple[float, Dict]:
    ddg = score_thermompnn(thermo, thermo_cfg, pdb_path, cdr_mask, device)
    s_thermo = (-ddg) if ddg is not None else 0.0
    s_bsa    = score_bsa(pdb_path, cdr_mask, epitope_ca)
    total    = w_thermo * s_thermo + w_bsa * s_bsa
    return total, {"thermo_neg_ddg": s_thermo, "bsa_contacts": s_bsa}


# ─────────────────────────────────────────────────────────────────────────────
# SMC utilities
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Particle:
    pdb_path:      str
    log_weight:    float = 0.0
    score_history: List[Dict] = field(default_factory=list)
    round_idx:     int = 0


def ess(particles: List[Particle]) -> float:
    lw = np.array([p.log_weight for p in particles], dtype=np.float64)
    lw -= lw.max()
    w = np.exp(lw); w /= w.sum()
    return float(1.0 / (w**2).sum())


def systematic_resample(
    particles: List[Particle],
    work_dir:  str,
    round_idx: int,
) -> List[Particle]:
    """
    Systematic resampling. Winner PDB files are copied to new paths so
    that each survivor can be used independently as the next-round
    reference input without overwriting its siblings.
    """
    n  = len(particles)
    lw = np.array([p.log_weight for p in particles], dtype=np.float64)
    lw -= np.logaddexp.reduce(lw)
    w  = np.exp(lw); cum = np.cumsum(w)

    u         = np.random.uniform(0, 1.0 / n)
    positions = u + np.arange(n) / n
    indices, j = [], 0
    for pos in positions:
        while j < n - 1 and cum[j] < pos:
            j += 1
        indices.append(j)

    out = []
    for new_i, old_i in enumerate(indices):
        src = particles[old_i]
        dst = os.path.join(work_dir, f"r{round_idx:02d}_p{new_i:03d}_ref.pdb")
        shutil.copy2(src.pdb_path, dst)
        out.append(Particle(
            pdb_path=dst,
            log_weight=0.0,                         # reset after resample
            score_history=deepcopy(src.score_history),
            round_idx=round_idx,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# One de novo rfdiffusion run for a single particle
# ─────────────────────────────────────────────────────────────────────────────

def graft_anchor_identities(
    rfdiffusion_pdb:  str,
    ref_pdb:          str,
    anchor_residues:  List[Tuple[str, int]],
    out_path:         str,
) -> str:
    """
    Restore residue names (amino acid identities) for anchor positions from
    ref_pdb into rfdiffusion_pdb.

    IMPORTANT LIMITATION — backbone-only representation:
      RFdiffusion outputs only backbone atoms (N, CA, C, O) and writes GLY
      for all designed residues.  This function corrects residue names so that
      downstream tools see the right amino acid identity, but it does NOT add
      sidechain atoms or repack sidechains.

      Consequence by consumer:
        - ThermoMPNN:   safe — uses backbone atoms only for GNN featurization
        - ProteinMPNN:  safe — uses backbone atoms only for sequence sampling
        - Visual inspection / Rosetta / AF2 validation: structures will be
          missing CB and all sidechain atoms at anchor positions.  Run a
          sidechain packing step (e.g. scwrl4, Rosetta FastRelax, or
          ProteinMPNN with pack_side_chains) on final top-ranked outputs
          before any structural analysis or experimental follow-up.

    For in-loop SMC scoring this representation is sufficient.  Sidechain
    packing is deferred to the post-SMC step to avoid the per-round cost.
    """
    anchor_set = {(chain, resnum) for chain, resnum in anchor_residues}

    # Build lookup: (chain, resnum) -> 3-letter residue name from ref_pdb
    ref_resnames: Dict[Tuple[str, int], str] = {}
    with open(ref_pdb) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                chain   = line[21]
                resnum  = int(line[22:26].strip())
                resname = line[17:20].strip()
            except (ValueError, IndexError):
                continue
            key = (chain, resnum)
            if key in anchor_set and key not in ref_resnames:
                ref_resnames[key] = resname

    n_restored = 0
    out_lines: List[str] = []
    with open(rfdiffusion_pdb) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    chain  = line[21]
                    resnum = int(line[22:26].strip())
                except (ValueError, IndexError):
                    out_lines.append(line)
                    continue
                key = (chain, resnum)
                if key in ref_resnames:
                    resname = ref_resnames[key].ljust(3)[:3]
                    line = line[:17] + resname + line[20:]
                    n_restored += 1
            out_lines.append(line)

    with open(out_path, "w") as f:
        f.writelines(out_lines)

    if n_restored == 0 and anchor_residues:
        print(f"    [WARN] graft_anchor_identities: no anchor residues found "
              f"in {rfdiffusion_pdb} — check chain/resnum match")
    else:
        print(f"    [INFO] Restored residue names for {len(ref_resnames)} "
              f"anchor position(s) (backbone-only; no sidechains added)")

    return out_path


def pack_sidechains(
    pdb_path:        str,
    anchor_residues: List[Tuple[str, int]],
    mpnn_model:      ProteinMPNN,
    cdr_mask:        torch.Tensor,
    out_path:        str,
    device:          str = "cuda",
) -> str:
    """
    Add sidechain atoms to anchor positions in a backbone-only PDB using
    ProteinMPNN's tied_featurize + fixed-sequence sampling.

    Strategy:
      - Anchor positions: sequence is fixed (original identity from graft),
        ProteinMPNN predicts the most likely rotamer by sampling with T→0
      - Non-anchor CDR positions: designed freely (as during SMC)
      - Framework + target: fixed

    This produces a physically plausible sidechain placement for anchor
    residues without requiring an external packing tool.

    NOTE: ProteinMPNN does not explicitly model all rotamer degrees of freedom;
    for high-accuracy sidechain placement use scwrl4 or Rosetta FastRelax on
    the output of this function.  For the purposes of structural inspection
    and downstream filtering this representation is sufficient.

    Returns out_path on success, pdb_path unchanged on failure.
    """
    try:
        pdb_dicts = parse_PDB(pdb_path, ca_only=False)
        if not pdb_dicts:
            raise ValueError("parse_PDB returned empty list")
        pdb_dict = pdb_dicts[0]
    except Exception as e:
        print(f"    [WARN] pack_sidechains parse_PDB failed: {e}")
        return pdb_path

    all_chains = sorted({
        item[-1:] for item in pdb_dict if item.startswith("seq_chain")
    })
    ab_chains  = [c for c in all_chains if c in ("H", "L")]
    tgt_chains = [c for c in all_chains if c == "T"]
    if not ab_chains:
        return pdb_path

    chain_id_dict = {pdb_dict["name"]: (ab_chains, tgt_chains)}

    # Fix anchor positions (preserve identity) AND framework positions.
    # Only free CDR non-anchor positions are left designable.
    anchor_set = {(ch, rn) for ch, rn in anchor_residues}
    seq = pdb_dict.get("seq", "")

    # Build fixed positions: framework (non-CDR) + anchor CDR positions
    # cdr_mask covers antibody residues in H-then-L file order
    ab_res_order: List[Tuple[str, int]] = []
    seen: set = set()
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            ch = line[21]
            if ch not in ("H", "L"):
                continue
            try:
                rn = int(line[22:26].strip())
            except ValueError:
                continue
            if (ch, rn) not in seen:
                seen.add((ch, rn))
                ab_res_order.append((ch, rn))

    fixed_pos = []
    for idx, (ch, rn) in enumerate(ab_res_order):
        is_cdr    = idx < len(cdr_mask) and cdr_mask[idx].item()
        is_anchor = (ch, rn) in anchor_set
        if not is_cdr or is_anchor:
            fixed_pos.append(idx)

    fixed_positions_dict = {
        pdb_dict["name"]: {c: fixed_pos for c in ab_chains}
    }

    try:
        (X, S, mask, lengths, chain_M, chain_encoding_all,
         chain_list_list, visible_list_list, masked_list_list,
         masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
         residue_idx, dihedral_mask, tied_pos_list_of_lists_list,
         pssm_coef, pssm_bias, pssm_log_odds_all,
         bias_by_res_all, tied_beta) = tied_featurize(
            [pdb_dict], device, chain_id_dict, fixed_positions_dict,
            omit_AA_dict=None, tied_positions_dict=None,
            pssm_dict=None, bias_by_res_dict=None,
        )
        randn = torch.randn(chain_M.shape, device=device)
        with torch.no_grad():
            # Very low temperature → near-deterministic, most likely sequence
            sample = mpnn_model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask, temperature=0.01,
                omit_AAs_np=np.zeros(21), bias_AAs_np=np.zeros(21),
                chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0,
                pssm_log_odds_flag=False,
                pssm_log_odds_mask=(pssm_log_odds_all > 0).float(),
                pssm_bias_flag=False,
                tied_pos=tied_pos_list_of_lists_list[0],
                tied_beta=tied_beta, bias_by_res=bias_by_res_all,
            )
        seq_1letter: str = _S_to_seq(sample["S"][0], mask[0])
    except Exception as e:
        print(f"    [WARN] pack_sidechains ProteinMPNN failed: {e}")
        return pdb_path

    # Rewrite residue names in the PDB using the sampled sequence
    chain_resnum_to_aa: Dict[Tuple[str, int], str] = {}
    for idx, (ch, rn) in enumerate(ab_res_order):
        if idx < len(seq_1letter):
            chain_resnum_to_aa[(ch, rn)] = seq_1letter[idx]

    out_lines: List[str] = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                ch = line[21]
                try:
                    rn = int(line[22:26].strip())
                except ValueError:
                    out_lines.append(line)
                    continue
                aa1 = chain_resnum_to_aa.get((ch, rn))
                if aa1 and aa1 in AA1_TO_3:
                    line = line[:17] + AA1_TO_3[aa1].ljust(3) + line[20:]
            out_lines.append(line)

    with open(out_path, "w") as f:
        f.writelines(out_lines)

    print(f"    [INFO] Sidechain packing written to {out_path} "
          f"(backbone-only; use scwrl4/Rosetta for full rotamer placement)")
    return out_path



    ref_pdb:          str,
    contig_string:    str,
    hotspots:         str,
    output_prefix:    str,
    model_weights:    str,
    original_pdb:     str,
    anchor_residues:  List[Tuple[str, int]],
    extra_args:       List[str],
) -> Optional[str]:
    """
    Run one full de novo diffusion starting from pure noise.

    Post-processing pipeline after rfdiffusion:
      1. Graft original target chain coordinates back (existing behaviour)
      2. Restore anchor residue identities from ref_pdb (new step)

    Step 2 is necessary because rfdiffusion writes GLY for all designed
    positions including anchors, even though their backbone geometry is
    correctly preserved by the contig string.
    """
    cmd = build_denovo_rfdiffusion_cmd(
        input_pdb=ref_pdb,
        output_prefix=output_prefix,
        contig_string=contig_string,
        hotspots=hotspots,
        model_weights=model_weights,
        num_designs=1,
        extra_args=extra_args,
    )
    result = subprocess.run(" ".join(cmd), shell=True,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [WARN] rfdiffusion failed:\n{result.stderr[-600:]}")
        return None

    # Locate output
    candidate = f"{output_prefix}_0.pdb"
    if not os.path.isfile(candidate):
        matches = sorted(Path(os.path.dirname(output_prefix)).glob(
            f"{Path(output_prefix).name}*.pdb"
        ))
        candidate = str(matches[-1]) if matches else None
    if not candidate or not os.path.isfile(candidate):
        print(f"    [WARN] No output PDB found at {output_prefix}*.pdb")
        return None

    # Step 1: graft original target chain coordinates back
    grafted = output_prefix + "_grafted.pdb"
    graft_target_sequence(
        rfdiffusion_pdb=candidate,
        original_target=original_pdb,
        input_pdb=ref_pdb,
        out_path=grafted,
        target_chain=CHAIN_T,
        source_chain=CHAIN_T,
        ref_chain=CHAIN_H,
    )

    # Step 2: restore anchor residue identities (backbone preserved by contig,
    # but rfdiffusion writes GLY for all designed positions including anchors)
    if anchor_residues:
        anchored = output_prefix + "_anchored.pdb"
        graft_anchor_identities(
            rfdiffusion_pdb=grafted,
            ref_pdb=ref_pdb,
            anchor_residues=anchor_residues,
            out_path=anchored,
        )
        return anchored

    return grafted


# ─────────────────────────────────────────────────────────────────────────────
# Main SMC loop
# ─────────────────────────────────────────────────────────────────────────────

def run_smc_denovo(
    # ── structure inputs ────────────────────────────────────────────────────
    input_pdb:       str,
    anchors_json:    str,
    output_dir:      str,
    hotspots:        str,
    model_weights:   str,
    # ── SMC ─────────────────────────────────────────────────────────────────
    n_particles:     int   = 16,
    n_rounds:        int   = 4,
    guidance_scale:  float = 1.0,
    ess_threshold:   float = 0.5,
    w_thermo:        float = 1.0,
    w_bsa:           float = 0.5,
    # ── ThermoMPNN ──────────────────────────────────────────────────────────
    thermo_local_yaml: str = "",
    thermo_model_yaml: str = "",
    thermo_checkpoint: str = "",
    # ── ProteinMPNN ─────────────────────────────────────────────────────────
    mpnn_weights:    str = "",
    # ── other ───────────────────────────────────────────────────────────────
    free_loops_spec: str  = "",
    nanobody:        bool = False,
    name:            str  = "",
    extra_args:      Optional[List[str]] = None,
    device:          str  = "cuda",
) -> List[Particle]:
    """
    SMC-guided de novo antibody design with motif-anchored CDRs.

    Each SMC round is a complete de novo rfdiffusion run (pure noise start).
    Anchor residues are kept fixed via the motif scaffolding contig mechanism.
    Resampling propagates high-scoring anchor coordinates to the next round.

    Returns Particle list sorted by log_weight (best first).
    """
    extra_args = extra_args or []
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, "_smc_work")
    os.makedirs(work_dir, exist_ok=True)
    stem = name or Path(input_pdb).stem

    # ── 1. Parse HLT structure ───────────────────────────────────────────────
    print(f"[SMC] Parsing HLT: {input_pdb}")
    cdr_ranges = parse_hlt_remarks(input_pdb)
    residues   = read_pdb_residues(input_pdb)

    n_l = sum(1 for r in residues if r.pdb_chain == CHAIN_L)
    if not nanobody and n_l == 0:
        print("[INFO] No L-chain residues — treating as nanobody.")
        nanobody = True

    # ── 2. Split target / framework (for grafting and REMARK mask) ───────────
    split_dir = os.path.join(work_dir, "_split")
    target_pdb, framework_pdb = split_hlt_complex(input_pdb, split_dir)

    # ── 3. Load anchor residues ──────────────────────────────────────────────
    anchor_residues = load_anchors(anchors_json)
    print(f"[SMC] {len(anchor_residues)} anchor residue(s): "
          f"{[f'{c}{n}' for c, n in anchor_residues]}")

    # ── 4. Build de novo contig string ───────────────────────────────────────
    #
    # IMPORTANT DIFFERENCE FROM PARTIAL DIFFUSION:
    #   We do NOT mask anchor REMARK lines here. The HLT REMARK annotations
    #   are read by the RFdiffusion antibody model to identify CDR positions.
    #   Masking them would cause the model to treat anchors as framework and
    #   potentially mis-classify loop regions.
    #
    #   Instead, anchoring is achieved purely via the contig string: anchor
    #   residue spans appear as "ChainStart-End" (fixed motif), so RFdiffusion
    #   scaffolds free CDR segments around them from pure noise.
    #
    free_loops    = parse_free_loops(free_loops_spec)
    contig_string = build_denovo_contig(
        residues, cdr_ranges, anchor_residues, free_loops, nanobody
    )
    print(f"[SMC] Contig: {contig_string}")

    # ── 5. CDR mask (from unmodified framework — all CDR positions) ──────────
    cdr_mask = build_cdr_mask(framework_pdb)

    # ── 6. Epitope Ca coords ─────────────────────────────────────────────────
    epitope_ca = load_epitope_ca(target_pdb, hotspots, device)

    # ── 7. Load scoring models ────────────────────────────────────────────────
    print("[SMC] Loading ThermoMPNN...")
    thermo = load_thermompnn(
        config_yaml=thermo_model_yaml,
        local_yaml=thermo_local_yaml,
        checkpoint=thermo_checkpoint,
        device=device,
    )
    # cfg needed to satisfy parse_pdb_cached(cfg, pdb_file) signature
    thermo_cfg = OmegaConf.merge(
        OmegaConf.load(thermo_local_yaml),
        OmegaConf.load(thermo_model_yaml),
    )
    print("[SMC] Loading ProteinMPNN...")
    mpnn = load_proteinmpnn(mpnn_weights, device)

    # ── 8. Initialise particles — all start from the input PDB ───────────────
    #    The input PDB provides anchor motif coordinates for round 1.
    particles: List[Particle] = []
    for i in range(n_particles):
        init_path = os.path.join(work_dir, f"r00_p{i:03d}_ref.pdb")
        shutil.copy2(input_pdb, init_path)
        particles.append(Particle(pdb_path=init_path))

    # ── 9. SMC loop ───────────────────────────────────────────────────────────
    for rnd in range(1, n_rounds + 1):
        print(f"\n[SMC] ── Round {rnd}/{n_rounds} ──")

        # 9a. Run de novo diffusion for every particle
        next_pdbs: List[Optional[str]] = []
        for i, p in enumerate(particles):
            out_prefix = os.path.join(
                work_dir, f"r{rnd:02d}_p{i:03d}_rfd"
            )
            print(f"  p{i:03d}: rfdiffusion (ref={Path(p.pdb_path).name})...",
                  end=" ", flush=True)
            out = run_denovo_round(
                ref_pdb=p.pdb_path,
                contig_string=contig_string,
                hotspots=hotspots,
                output_prefix=out_prefix,
                model_weights=model_weights,
                original_pdb=input_pdb,
                anchor_residues=anchor_residues,
                extra_args=extra_args,
            )
            next_pdbs.append(out)
            print("done" if out else "FAILED")

        # 9b. Score and update log-weights
        print(f"  Scoring {n_particles} particles...")
        for i, (p, out_pdb) in enumerate(zip(particles, next_pdbs)):
            if out_pdb is None:
                p.log_weight -= 1e6
                continue
            score, bd = score_composite(
                pdb_path=out_pdb,
                thermo=thermo,
                thermo_cfg=thermo_cfg,
                cdr_mask=cdr_mask,
                epitope_ca=epitope_ca,
                w_thermo=w_thermo,
                w_bsa=w_bsa,
                device=device,
            )
            p.log_weight += guidance_scale * score
            p.pdb_path    = out_pdb
            p.round_idx   = rnd
            p.score_history.append({"round": rnd, "score": score, **bd})
            print(f"    p{i:03d}  score={score:+.3f}  "
                  f"thermo={bd['thermo_neg_ddg']:+.3f}  "
                  f"bsa={bd['bsa_contacts']:.0f}  "
                  f"log_w={p.log_weight:+.3f}")

        # 9c. Resample if ESS too low
        cur_ess = ess(particles)
        print(f"  ESS = {cur_ess:.1f} / {n_particles}")
        if cur_ess < ess_threshold * n_particles:
            print("  Resampling...")
            particles = systematic_resample(particles, work_dir, rnd)
            # After resampling, winning PDBs become the reference input for
            # the next round — anchor coordinates are inherited from survivors.
        else:
            # Still update pdb_path for next round even without resampling
            for p in particles:
                if p.pdb_path and os.path.isfile(p.pdb_path):
                    new_ref = p.pdb_path.replace(".pdb", "_ref.pdb")
                    shutil.copy2(p.pdb_path, new_ref)
                    p.pdb_path = new_ref

    # ── 10. Write outputs ─────────────────────────────────────────────────────
    particles.sort(key=lambda p: p.log_weight, reverse=True)
    final_dir = os.path.join(output_dir, "final_designs")
    os.makedirs(final_dir, exist_ok=True)

    print(f"\n[SMC] Copying top designs to {final_dir}/ and packing sidechains...")
    for rank, p in enumerate(particles):
        dst = os.path.join(final_dir, f"{stem}_smc_rank{rank:03d}.pdb")
        if os.path.isfile(p.pdb_path):
            shutil.copy2(p.pdb_path, dst)
        # Pack sidechains on the copied final structure so that anchor
        # residues have physically plausible (though approximate) geometry
        packed = os.path.join(final_dir, f"{stem}_smc_rank{rank:03d}_packed.pdb")
        pack_sidechains(
            pdb_path=dst,
            anchor_residues=anchor_residues,
            mpnn_model=mpnn,
            cdr_mask=cdr_mask,
            out_path=packed,
            device=device,
        )
        print(f"  rank {rank:03d}  log_weight={p.log_weight:+.3f}"
              f"  packed → {Path(packed).name}")

    summary = [
        {"rank": i, "log_weight": p.log_weight,
         "pdb": os.path.basename(p.pdb_path),
         "score_history": p.score_history}
        for i, p in enumerate(particles)
    ]
    summary_path = os.path.join(output_dir, f"{stem}_smc_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SMC] Summary → {summary_path}")
    return particles


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "SMC-guided de novo antibody design with motif-anchored CDRs. "
            "No partial diffusion — anchoring is via contigmap.contigs only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",         required=True)
    p.add_argument("--anchors",       required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--hotspots",      required=True)
    p.add_argument("--model_weights", required=True)
    p.add_argument("--thermo_local_yaml", required=True)
    p.add_argument("--thermo_model_yaml", required=True)
    p.add_argument("--thermo_checkpoint", required=True)
    p.add_argument("--mpnn_weights",  required=True)
    p.add_argument("--n_particles",    type=int,   default=16)
    p.add_argument("--n_rounds",       type=int,   default=4)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--ess_threshold",  type=float, default=0.5)
    p.add_argument("--w_thermo",       type=float, default=1.0)
    p.add_argument("--w_bsa",          type=float, default=0.5)
    p.add_argument("--free_loops",    default="",
                   help="e.g. 'H3:5-13,L3:7-11'")
    p.add_argument("--nanobody",      action="store_true")
    p.add_argument("--name",          default="")
    p.add_argument("--device",        default="cuda")
    p.add_argument("extra",           nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    args  = parse_args()
    extra = [a for a in (args.extra or []) if a != "--"]
    run_smc_denovo(
        input_pdb=str(Path(args.input).resolve()),
        anchors_json=str(Path(args.anchors).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        hotspots=args.hotspots,
        model_weights=args.model_weights,
        n_particles=args.n_particles,
        n_rounds=args.n_rounds,
        guidance_scale=args.guidance_scale,
        ess_threshold=args.ess_threshold,
        w_thermo=args.w_thermo,
        w_bsa=args.w_bsa,
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