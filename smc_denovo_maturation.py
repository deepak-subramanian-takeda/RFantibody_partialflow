"""
smc_denovo_maturation.py

SMC-guided de novo antibody design with motif-anchored CDR residues.
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

# ── ThermoMPNN ────────────────────────────────────────────────────────────────
THERMOMPNN_ROOT = os.environ.get("THERMOMPNN_ROOT", "")
if not THERMOMPNN_ROOT:
    raise EnvironmentError(
        "THERMOMPNN_ROOT is not set. Export it before running, e.g.:\n"
        "  export THERMOMPNN_ROOT=/absolute/path/to/ThermoMPNN"
    )
THERMOMPNN_ROOT = str(Path(THERMOMPNN_ROOT).resolve())
if not (Path(THERMOMPNN_ROOT) / "transfer_model.py").exists():
    raise FileNotFoundError(
        f"transfer_model.py not found in THERMOMPNN_ROOT='{THERMOMPNN_ROOT}'."
    )
if THERMOMPNN_ROOT not in sys.path:
    sys.path.insert(0, THERMOMPNN_ROOT)

from transfer_model import TransferModel
from datasets import Mutation
from protein_mpnn_utils import parse_PDB as _thermo_parse_PDB

# ── ProteinMPNN (bundled with RFantibody) ─────────────────────────────────────
RFANTIBODY_ROOT = os.environ.get("RFANTIBODY_ROOT", "")
if not RFANTIBODY_ROOT:
    raise EnvironmentError(
        "RFANTIBODY_ROOT is not set. Export it before running, e.g.:\n"
        "  export RFANTIBODY_ROOT=/absolute/path/to/RFantibody"
    )
RFANTIBODY_ROOT = str(Path(RFANTIBODY_ROOT).resolve())

_pmpnn_candidates = [
    Path(RFANTIBODY_ROOT) / "src" / "rfantibody" / "proteinmpnn" / "model",
    Path(RFANTIBODY_ROOT) / "src" / "rfantibody" / "proteinmpnn",
    Path(RFANTIBODY_ROOT) / "scripts",
    Path(RFANTIBODY_ROOT) / "proteinmpnn",
    Path(RFANTIBODY_ROOT),
]
PMPNN_PATH = None
for _candidate in _pmpnn_candidates:
    if (_candidate / "protein_mpnn_utils.py").exists():
        PMPNN_PATH = str(_candidate)
        break
if PMPNN_PATH is None:
    _found = list(Path(RFANTIBODY_ROOT).rglob("protein_mpnn_utils.py"))
    if _found:
        PMPNN_PATH = str(_found[0].parent)
if PMPNN_PATH is None:
    raise FileNotFoundError(
        f"protein_mpnn_utils.py not found under RFANTIBODY_ROOT='{RFANTIBODY_ROOT}'."
    )
print(f"[ProteinMPNN] Found protein_mpnn_utils.py at: {PMPNN_PATH}")
if PMPNN_PATH not in sys.path:
    sys.path.insert(0, PMPNN_PATH)

from protein_mpnn_utils import (
    parse_PDB,
    tied_featurize,
    ProteinMPNN,
    _S_to_seq,
)

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA1_TO_3 = {v: k for k, v in AA3_TO_1.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Contig builder
# ─────────────────────────────────────────────────────────────────────────────

def build_denovo_contig(
    residues:            List[ResidueInfo],
    cdr_ranges:          Dict[str, CdrRange],
    anchor_residues:     List[Tuple[str, int]],
    free_loop_overrides: Dict[str, Tuple[int, int]],
    nanobody:            bool = False,
) -> str:
    anchor_set = {(c, r) for c, r in anchor_residues}
    cdr_pose_idx_to_name: Dict[int, str] = {}
    for name, cr in cdr_ranges.items():
        for idx in range(cr.start, cr.end + 1):
            cdr_pose_idx_to_name[idx] = name

    h_res = [r for r in residues if r.pdb_chain == CHAIN_H]
    l_res = [r for r in residues if r.pdb_chain == CHAIN_L]
    t_res = [r for r in residues if r.pdb_chain == CHAIN_T]

    def chain_segments(chain_residues: List[ResidueInfo]) -> str:
        tokens: List[str] = []
        i = 0
        while i < len(chain_residues):
            r = chain_residues[i]
            cdr_name = cdr_pose_idx_to_name.get(r.pose_idx)
            if cdr_name is None:
                run = []
                while (i < len(chain_residues) and
                       cdr_pose_idx_to_name.get(chain_residues[i].pose_idx) is None):
                    run.append(chain_residues[i])
                    i += 1
                ch = run[0].pdb_chain
                tokens.append(f"{ch}{run[0].pdb_resnum}-{run[-1].pdb_resnum}")
            else:
                cr = cdr_ranges[cdr_name]
                loop_res = []
                while (i < len(chain_residues) and
                       chain_residues[i].pose_idx <= cr.end):
                    loop_res.append(chain_residues[i])
                    i += 1
                tokens.extend(_cdr_to_contig_tokens(
                    loop_res, anchor_set, cdr_name,
                    len(loop_res), free_loop_overrides
                ))
        return "/".join(tokens)

    h_seg = chain_segments(h_res)
    if t_res:
        t_seg = f"{CHAIN_T}{t_res[0].pdb_resnum}-{t_res[-1].pdb_resnum}"
    else:
        t_seg = ""

    if nanobody or not l_res:
        return f"{h_seg}/0 {t_seg}" if t_seg else h_seg
    l_seg = chain_segments(l_res)
    return f"{h_seg}/0 {l_seg}/0 {t_seg}" if t_seg else f"{h_seg}/0 {l_seg}"


def _cdr_to_contig_tokens(
    loop_res:            List[ResidueInfo],
    anchor_set:          set,
    cdr_name:            str,
    orig_len:            int,
    free_loop_overrides: Dict[str, Tuple[int, int]],
) -> List[str]:
    tokens: List[str] = []
    min_len, max_len = free_loop_overrides.get(
        cdr_name, (max(1, orig_len - 2), orig_len + 2)
    )
    n_free = max(sum(1 for r in loop_res
                     if (r.pdb_chain, r.pdb_resnum) not in anchor_set), 1)
    i = 0
    while i < len(loop_res):
        r = loop_res[i]
        is_anchor = (r.pdb_chain, r.pdb_resnum) in anchor_set
        if is_anchor:
            run_start = run_end = r
            while (i + 1 < len(loop_res) and
                   (loop_res[i+1].pdb_chain, loop_res[i+1].pdb_resnum) in anchor_set):
                i += 1
                run_end = loop_res[i]
            tokens.append(f"{run_start.pdb_chain}"
                          f"{run_start.pdb_resnum}-{run_end.pdb_resnum}")
            i += 1
        else:
            gap_size = 0
            while (i < len(loop_res) and
                   (loop_res[i].pdb_chain, loop_res[i].pdb_resnum) not in anchor_set):
                gap_size += 1
                i += 1
            frac    = gap_size / n_free
            gap_min = max(1, round(min_len * frac))
            gap_max = max(gap_min, round(max_len * frac))
            tokens.append(f"{gap_min}-{gap_max}")
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# rfdiffusion CLI command builder
# ─────────────────────────────────────────────────────────────────────────────

def _find_inference_script() -> str:
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
    input_pdb:     str,
    output_prefix: str,
    contig_string: str,
    hotspots:      str,
    model_weights: str,
    num_designs:   int = 1,
    extra_args:    Optional[List[str]] = None,
) -> List[str]:
    inference_script = _find_inference_script()
    cmd = [
        sys.executable, inference_script,
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
# CDR mask
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
                    cdr_abs.append(int(m.group(1)) - 1)
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
    config_yaml: str,
    local_yaml:  str,
    checkpoint:  str,
    device:      str,
) -> TransferModel:
    cfg = OmegaConf.merge(OmegaConf.load(local_yaml), OmegaConf.load(config_yaml))
    model = TransferModel(cfg)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
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
# ThermoMPNN PDB parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdb_for_thermo(pdb_path: str) -> dict:
    result = _thermo_parse_PDB(pdb_path)
    if isinstance(result, list):
        return result[0] if result else {}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

THERMO_AA = "ACDEFGHIKLMNPQRSTVWY"


def score_thermompnn(
    thermo:   TransferModel,
    pdb_path: str,
    cdr_mask: torch.Tensor,
    device:   str,
) -> Optional[float]:
    try:
        pdb = parse_pdb_for_thermo(pdb_path)
    except Exception as e:
        print(f"    [WARN] parse_pdb_for_thermo failed: {e}")
        return None

    seq = pdb.get("seq", "")
    if not seq:
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
            mutations.append(Mutation(position=pos, wildtype=wt,
                                      mutation=mut, ddG=None))
    if not mutations:
        return None

    try:
        with torch.no_grad():
            preds = thermo([pdb], mutations, tied_feat=True)
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(preds, list):
            ddgs = np.array([
                d["ddG"].item() for d in preds
                if isinstance(d, dict) and "ddG" in d
            ])
        elif isinstance(preds, dict):
            ddgs = preds["ddG"].detach().cpu().numpy().flatten()
        else:
            ddgs = preds.detach().cpu().numpy().flatten()
        return float(np.mean(ddgs)) if len(ddgs) > 0 else None
    except Exception as e:
        import traceback
        print(f"    [WARN] ThermoMPNN forward failed: {e}")
        traceback.print_exc()
        return None


def score_bsa(pdb_path: str, cdr_mask: torch.Tensor,
              epitope_ca: torch.Tensor, dist: float = 8.0) -> float:
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
    cdr_mask:   torch.Tensor,
    epitope_ca: torch.Tensor,
    w_thermo:   float,
    w_bsa:      float,
    device:     str,
) -> Tuple[float, Dict]:
    ddg      = score_thermompnn(thermo, pdb_path, cdr_mask, device)
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
            log_weight=0.0,
            score_history=deepcopy(src.score_history),
            round_idx=round_idx,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-chain fixed positions helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_per_chain_fixed_positions(
    pdb_dict:  dict,
    ab_chains: List[str],
    cdr_mask:  torch.Tensor,
) -> dict:
    chain_lengths: Dict[str, int] = {}
    for ch in ab_chains:
        seq_key = f"seq_chain_{ch}"
        if seq_key in pdb_dict:
            chain_lengths[ch] = len(pdb_dict[seq_key])

    fixed_positions_dict: Dict[str, Dict[str, List[int]]] = {
        pdb_dict["name"]: {}
    }
    offset = 0
    for ch in ab_chains:
        ch_len = chain_lengths.get(ch, 0)
        ch_mask = cdr_mask[offset: offset + ch_len]
        fixed = [i for i in range(ch_len)
                 if i >= len(ch_mask) or not ch_mask[i].item()]
        fixed_positions_dict[pdb_dict["name"]][ch] = fixed
        offset += ch_len
    return fixed_positions_dict


# ─────────────────────────────────────────────────────────────────────────────
# Sequence design onto backbone
# ─────────────────────────────────────────────────────────────────────────────

def design_sequence_onto_backbone(
    mpnn_model:   ProteinMPNN,
    backbone_pdb: str,
    cdr_mask:     torch.Tensor,
    out_pdb:      str,
    temperature:  float = 0.1,
    device:       str   = "cuda",
) -> Optional[str]:
    try:
        pdb_dicts = parse_PDB(backbone_pdb, ca_only=False)
        if not pdb_dicts:
            raise ValueError("parse_PDB returned empty list")
        pdb_dict = pdb_dicts[0]
    except Exception as e:
        print(f"    [WARN] parse_PDB failed: {e}")
        return None

    all_chains = sorted({
        item[-1:] for item in pdb_dict if item.startswith("seq_chain")
    })
    ab_chains  = [c for c in all_chains if c in ("H", "L")]
    tgt_chains = [c for c in all_chains if c == "T"]
    if not ab_chains:
        return None

    chain_id_dict        = {pdb_dict["name"]: (ab_chains, tgt_chains)}
    fixed_positions_dict = _build_per_chain_fixed_positions(
        pdb_dict, ab_chains, cdr_mask
    )

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
            sample = mpnn_model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask, temperature=temperature,
                omit_AAs_np=np.zeros(21), bias_AAs_np=np.zeros(21),
                chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0,
                pssm_log_odds_flag=False,
                pssm_log_odds_mask=(pssm_log_odds_all > 0).float(),
                pssm_bias_flag=False, bias_by_res=bias_by_res_all,
            )
        seq_1letter: str = _S_to_seq(sample["S"][0], mask[0])
    except Exception as e:
        print(f"    [WARN] ProteinMPNN sampling failed: {e}")
        return None

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
                    line = line[:17] + AA1_TO_3[aa1].ljust(3) + line[20:]
            out_lines.append(line)

    with open(out_pdb, "w") as fh:
        fh.writelines(out_lines)
    return out_pdb


# ─────────────────────────────────────────────────────────────────────────────
# Anchor identity grafting
# ─────────────────────────────────────────────────────────────────────────────

def graft_anchor_identities(
    rfdiffusion_pdb:  str,
    ref_pdb:          str,
    anchor_residues:  List[Tuple[str, int]],
    out_path:         str,
) -> str:
    """
    Restore residue names for anchor positions from ref_pdb.
    Backbone-only: does not add sidechain atoms.
    Safe for ThermoMPNN and ProteinMPNN (backbone-only consumers).
    """
    anchor_set = {(ch, rn) for ch, rn in anchor_residues}
    ref_resnames: Dict[Tuple[str, int], str] = {}
    with open(ref_pdb) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                ch = line[21]; rn = int(line[22:26].strip())
                resname = line[17:20].strip()
            except (ValueError, IndexError):
                continue
            key = (ch, rn)
            if key in anchor_set and key not in ref_resnames:
                ref_resnames[key] = resname

    n_restored = 0
    out_lines: List[str] = []
    with open(rfdiffusion_pdb) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    ch = line[21]; rn = int(line[22:26].strip())
                except (ValueError, IndexError):
                    out_lines.append(line)
                    continue
                key = (ch, rn)
                if key in ref_resnames:
                    line = line[:17] + ref_resnames[key].ljust(3)[:3] + line[20:]
                    n_restored += 1
            out_lines.append(line)

    with open(out_path, "w") as f:
        f.writelines(out_lines)

    if n_restored == 0 and anchor_residues:
        print(f"    [WARN] graft_anchor_identities: no anchor residues matched")
    else:
        print(f"    [INFO] Restored residue names for {len(ref_resnames)} "
              f"anchor position(s)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Sidechain packing
# ─────────────────────────────────────────────────────────────────────────────

def pack_sidechains(
    pdb_path:        str,
    anchor_residues: List[Tuple[str, int]],
    mpnn_model:      ProteinMPNN,
    cdr_mask:        torch.Tensor,
    out_path:        str,
    device:          str = "cuda",
) -> str:
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

    anchor_set = {(ch, rn) for ch, rn in anchor_residues}
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

    chain_id_dict        = {pdb_dict["name"]: (ab_chains, tgt_chains)}
    fixed_positions_dict = _build_per_chain_fixed_positions(
        pdb_dict, ab_chains, cdr_mask
    )
    # Also fix anchor positions within CDRs
    offset = 0
    for ch in ab_chains:
        seq_key = f"seq_chain_{ch}"
        ch_len  = len(pdb_dict.get(seq_key, ""))
        existing = set(fixed_positions_dict[pdb_dict["name"]].get(ch, []))
        for idx, (ach, arn) in enumerate(ab_res_order):
            if ach == ch and (ach, arn) in anchor_set:
                ch_idx = idx - offset
                if 0 <= ch_idx < ch_len:
                    existing.add(ch_idx)
        fixed_positions_dict[pdb_dict["name"]][ch] = sorted(existing)
        offset += ch_len

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
            sample = mpnn_model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask, temperature=0.01,
                omit_AAs_np=np.zeros(21), bias_AAs_np=np.zeros(21),
                chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0,
                pssm_log_odds_flag=False,
                pssm_log_odds_mask=(pssm_log_odds_all > 0).float(),
                pssm_bias_flag=False, bias_by_res=bias_by_res_all,
            )
        seq_1letter: str = _S_to_seq(sample["S"][0], mask[0])
    except Exception as e:
        print(f"    [WARN] pack_sidechains ProteinMPNN failed: {e}")
        return pdb_path

    chain_resnum_to_aa = {
        (ch, rn): seq_1letter[idx]
        for idx, (ch, rn) in enumerate(ab_res_order)
        if idx < len(seq_1letter)
    }

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
    print(f"    [INFO] Sidechain packing written to {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Framework Ca extraction and Kabsch alignment
# ─────────────────────────────────────────────────────────────────────────────

def _get_framework_ca(
    pdb_path:   str,
    cdr_ranges: Dict[str, CdrRange],
    chain:      str = CHAIN_H,
) -> Dict[int, np.ndarray]:
    """
    Extract Ca coordinates for framework (non-CDR) residues of a given chain.
    CDR residues are identified from REMARK PDBinfo-LABEL lines.
    Returns {resnum: xyz}.
    """
    remark_re = re.compile(
        r"^REMARK\s+PDBinfo-LABEL:\s+(\d+)\s+(H[123]|L[123])\s*$"
    )
    # Build abs_idx -> (chain, resnum) mapping
    abs_to_res: Dict[int, Tuple[str, int]] = {}
    seen: set = set()
    abs_idx = 0
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            ch = line[21]
            if ch not in (CHAIN_H, CHAIN_L, CHAIN_T):
                continue
            try:
                rn = int(line[22:26].strip())
            except ValueError:
                continue
            key = (ch, rn)
            if key not in seen:
                seen.add(key)
                abs_idx += 1
                abs_to_res[abs_idx] = (ch, rn)

    # CDR abs indices from REMARK lines
    cdr_abs: set = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("REMARK PDBinfo-LABEL:"):
                m = remark_re.match(line.strip())
                if m:
                    cdr_abs.add(int(m.group(1)))

    cdr_keys = {res for idx, res in abs_to_res.items() if idx in cdr_abs}

    ca: Dict[int, np.ndarray] = {}
    with open(pdb_path) as f:
        for line in f:
            if (line.startswith("ATOM") and
                    line[12:16].strip() == "CA" and
                    line[21] == chain):
                try:
                    rn = int(line[22:26].strip())
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                except ValueError:
                    continue
                if (chain, rn) not in cdr_keys:
                    ca[rn] = np.array([x, y, z])
    return ca


def _kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch algorithm: find R, t such that R @ p + t ≈ q."""
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    H = (P - p_mean).T @ (Q - q_mean)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_mean - R @ p_mean
    return R, t


# ─────────────────────────────────────────────────────────────────────────────
# Target chain copy with framework alignment
# ─────────────────────────────────────────────────────────────────────────────

def copy_target_chain(
    rfdiffusion_pdb: str,
    ref_pdb:         str,
    out_path:        str,
    cdr_ranges:      Dict[str, CdrRange],
    target_chain:    str = CHAIN_T,
    framework_chain: str = CHAIN_H,
) -> str:
    """
    Copy the target chain from ref_pdb into rfdiffusion_pdb, applying a
    Kabsch superposition on framework (non-CDR) Ca atoms to align the
    antigen into the rfdiffusion output coordinate frame.

    RFdiffusion may rotate/translate the complex internally. Framework
    residues are preserved geometrically, so we use them to estimate the
    rigid-body transform and apply it to the antigen coordinates.
    """
    ref_fw  = _get_framework_ca(ref_pdb,         cdr_ranges, framework_chain)
    out_fw  = _get_framework_ca(rfdiffusion_pdb,  cdr_ranges, framework_chain)

    common = sorted(set(ref_fw) & set(out_fw))
    if len(common) < 3:
        print(f"    [WARN] copy_target_chain: only {len(common)} common "
              f"framework Ca — copying target without alignment")
        R, t = np.eye(3), np.zeros(3)
    else:
        P = np.array([ref_fw[r] for r in common])
        Q = np.array([out_fw[r] for r in common])
        R, t = _kabsch(P, Q)
        rmsd = float(np.sqrt((((R @ P.T).T + t - Q)**2).sum(axis=1).mean()))
        print(f"    [INFO] Framework alignment RMSD: {rmsd:.3f} Å "
              f"({len(common)} Ca)")

    # Transform T-chain coordinates from ref_pdb
    ref_t_lines: List[str] = []
    with open(ref_pdb) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[21] != target_chain:
                continue
            try:
                xyz = R @ np.array([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54])
                ]) + t
                line = (line[:30]
                        + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                        + line[54:])
            except ValueError:
                pass
            ref_t_lines.append(line)

    # Write: antibody from rfdiffusion, aligned T chain from ref
    out_lines: List[str] = []
    t_inserted = False
    with open(rfdiffusion_pdb) as f:
        for line in f:
            if line.startswith(("END", "MASTER")):
                if not t_inserted:
                    out_lines.extend(ref_t_lines)
                    t_inserted = True
                out_lines.append(line)
                continue
            if (line.startswith("ATOM") or line.startswith("HETATM")):
                if line[21] == target_chain:
                    if not t_inserted:
                        out_lines.extend(ref_t_lines)
                        t_inserted = True
                    continue
            out_lines.append(line)

    if not t_inserted:
        out_lines.extend(ref_t_lines)
    if not out_lines or not out_lines[-1].startswith("END"):
        out_lines.append("END\n")

    with open(out_path, "w") as f:
        f.writelines(out_lines)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# One de novo rfdiffusion run for a single particle
# ─────────────────────────────────────────────────────────────────────────────

def run_denovo_round(
    ref_pdb:          str,
    contig_string:    str,
    hotspots:         str,
    output_prefix:    str,
    model_weights:    str,
    original_pdb:     str,
    anchor_residues:  List[Tuple[str, int]],
    cdr_ranges:       Dict[str, CdrRange],
    extra_args:       List[str],
) -> Optional[str]:
    """
    Run one full de novo diffusion starting from pure noise.
    Post-processing:
      1. Align and copy target chain from original PDB (Kabsch on framework Ca)
      2. Restore anchor residue identities
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

    candidate = f"{output_prefix}_0.pdb"
    if not os.path.isfile(candidate):
        matches = sorted(Path(os.path.dirname(output_prefix)).glob(
            f"{Path(output_prefix).name}*.pdb"
        ))
        candidate = str(matches[-1]) if matches else None
    if not candidate or not os.path.isfile(candidate):
        print(f"    [WARN] No output PDB found at {output_prefix}*.pdb")
        return None

    # Step 1: align and copy target chain
    grafted = output_prefix + "_grafted.pdb"
    copy_target_chain(
        rfdiffusion_pdb=candidate,
        ref_pdb=original_pdb,
        out_path=grafted,
        cdr_ranges=cdr_ranges,
        target_chain=CHAIN_T,
        framework_chain=CHAIN_H,
    )

    # Step 2: restore anchor residue identities
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
    input_pdb:         str,
    anchors_json:      str,
    output_dir:        str,
    hotspots:          str,
    model_weights:     str,
    n_particles:       int   = 16,
    n_rounds:          int   = 4,
    guidance_scale:    float = 1.0,
    ess_threshold:     float = 0.5,
    w_thermo:          float = 1.0,
    w_bsa:             float = 0.5,
    thermo_local_yaml: str   = "",
    thermo_model_yaml: str   = "",
    thermo_checkpoint: str   = "",
    mpnn_weights:      str   = "",
    free_loops_spec:   str   = "",
    nanobody:          bool  = False,
    name:              str   = "",
    extra_args:        Optional[List[str]] = None,
    device:            str   = "cuda",
) -> List[Particle]:
    extra_args = extra_args or []
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, "_smc_work")
    os.makedirs(work_dir, exist_ok=True)
    stem = name or Path(input_pdb).stem

    print(f"[SMC] Parsing HLT: {input_pdb}")
    cdr_ranges = parse_hlt_remarks(input_pdb)
    residues   = read_pdb_residues(input_pdb)

    n_l = sum(1 for r in residues if r.pdb_chain == CHAIN_L)
    if not nanobody and n_l == 0:
        print("[INFO] No L-chain residues — treating as nanobody.")
        nanobody = True

    split_dir = os.path.join(work_dir, "_split")
    target_pdb, framework_pdb = split_hlt_complex(input_pdb, split_dir)

    anchor_residues = load_anchors(anchors_json)
    print(f"[SMC] {len(anchor_residues)} anchor residue(s): "
          f"{[f'{c}{n}' for c, n in anchor_residues]}")

    free_loops    = parse_free_loops(free_loops_spec)
    contig_string = build_denovo_contig(
        residues, cdr_ranges, anchor_residues, free_loops, nanobody
    )
    print(f"[SMC] Contig: {contig_string}")

    cdr_mask   = build_cdr_mask(framework_pdb)
    epitope_ca = load_epitope_ca(target_pdb, hotspots, device)

    print("[SMC] Loading ThermoMPNN...")
    thermo = load_thermompnn(
        config_yaml=thermo_model_yaml,
        local_yaml=thermo_local_yaml,
        checkpoint=thermo_checkpoint,
        device=device,
    )

    print("[SMC] Loading ProteinMPNN...")
    mpnn = load_proteinmpnn(mpnn_weights, device)

    print(f"[SMC] Initialising {n_particles} particles from {input_pdb}")
    particles: List[Particle] = []
    for i in range(n_particles):
        init_path = os.path.join(work_dir, f"r00_p{i:03d}_ref.pdb")
        shutil.copy2(input_pdb, init_path)
        particles.append(Particle(pdb_path=init_path))

    for rnd in range(1, n_rounds + 1):
        print(f"\n[SMC] ── Round {rnd}/{n_rounds} ──")

        next_pdbs: List[Optional[str]] = []
        for i, p in enumerate(particles):
            out_prefix = os.path.join(work_dir, f"r{rnd:02d}_p{i:03d}_rfd")
            print(f"  p{i:03d}: rfdiffusion...", end=" ", flush=True)
            out = run_denovo_round(
                ref_pdb=p.pdb_path,
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
                seq_pdb = out.replace("_anchored.pdb", "_seq.pdb") \
                             .replace("_grafted.pdb",  "_seq.pdb")
                out = design_sequence_onto_backbone(
                    mpnn_model=mpnn,
                    backbone_pdb=out,
                    cdr_mask=cdr_mask,
                    out_pdb=seq_pdb,
                    temperature=0.1,
                    device=device,
                ) or out
                if anchor_residues:
                    enforced_pdb = seq_pdb.replace("_seq.pdb", "_enforced.pdb")
                    graft_anchor_identities(
                        rfdiffusion_pdb=out,
                        ref_pdb=p.pdb_path,
                        anchor_residues=anchor_residues,
                        out_path=enforced_pdb,
                    )
                    out = enforced_pdb
                print("done")
            else:
                print("FAILED")
            next_pdbs.append(out)

        print(f"  Scoring {n_particles} particles...")
        for i, (p, out_pdb) in enumerate(zip(particles, next_pdbs)):
            if out_pdb is None:
                p.log_weight -= 1e6
                continue
            score, bd = score_composite(
                pdb_path=out_pdb,
                thermo=thermo,
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

        cur_ess = ess(particles)
        print(f"  ESS = {cur_ess:.1f} / {n_particles}")
        if cur_ess < ess_threshold * n_particles:
            print("  Resampling...")
            particles = systematic_resample(particles, work_dir, rnd)
        else:
            for p in particles:
                if p.pdb_path and os.path.isfile(p.pdb_path):
                    new_ref = p.pdb_path.replace(".pdb", "_ref.pdb")
                    shutil.copy2(p.pdb_path, new_ref)
                    p.pdb_path = new_ref

    particles.sort(key=lambda p: p.log_weight, reverse=True)
    final_dir = os.path.join(output_dir, "final_designs")
    os.makedirs(final_dir, exist_ok=True)

    print(f"\n[SMC] Copying top designs to {final_dir}/ and packing sidechains...")
    for rank, p in enumerate(particles):
        dst = os.path.join(final_dir, f"{stem}_smc_rank{rank:03d}.pdb")
        if os.path.isfile(p.pdb_path):
            shutil.copy2(p.pdb_path, dst)

        packed = os.path.join(final_dir, f"{stem}_smc_rank{rank:03d}_packed.pdb")
        pack_sidechains(
            pdb_path=dst,
            anchor_residues=anchor_residues,
            mpnn_model=mpnn,
            cdr_mask=cdr_mask,
            out_path=packed,
            device=device,
        )

        final = os.path.join(final_dir, f"{stem}_smc_rank{rank:03d}_final.pdb")
        graft_anchor_identities(
            rfdiffusion_pdb=packed,
            ref_pdb=input_pdb,
            anchor_residues=anchor_residues,
            out_path=final,
        )
        print(f"  rank {rank:03d}  log_weight={p.log_weight:+.3f}"
              f"  → {Path(final).name}")

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
    p.add_argument("--n_particles",    type=int,   default=16)
    p.add_argument("--n_rounds",       type=int,   default=4)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--ess_threshold",  type=float, default=0.5)
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