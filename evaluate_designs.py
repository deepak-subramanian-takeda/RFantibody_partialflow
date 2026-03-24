#!/usr/bin/env python3
"""
evaluate_designs.py

Evaluates final designs from a de novo maturation run by:
  1. Running AlphaFold2 (via colabfold_batch) on each _final.pdb
  2. Extracting per-residue pLDDT and interface pAE (ipAE) from AF2 outputs
  3. Computing Binder-RMSD: align full complex to reference, then measure
     Cα RMSD over H+L binder chains only

Usage:
    python evaluate_designs.py \
        --run_dir   /path/to/5y2l_B_fk_denovo_traj16_rounds4 \
        --reference /path/to/5y2l_hlt_B.pdb \
        --output    results.json \
        [--colabfold_conda  /path/to/conda/envs/colabfold] \
        [--af2_num_recycles 3] \
        [--af2_num_models   1] \
        [--device           cuda] \
        [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Chain constants (matches HLT convention) ──────────────────────────────────
CHAIN_H = "H"
CHAIN_L = "L"
CHAIN_T = "T"
BINDER_CHAINS = {CHAIN_H, CHAIN_L}


# ─────────────────────────────────────────────────────────────────────────────
# PDB parsing utilities
# ─────────────────────────────────────────────────────────────────────────────

def parse_ca_atoms(
    pdb_path: str,
    chains: Optional[set] = None,
) -> Dict[Tuple[str, int, str], np.ndarray]:
    """
    Parse Cα coordinates from a PDB file.

    Returns dict keyed by (chain_id, res_seq, ins_code) → xyz array.
    If chains is provided, only those chains are included.
    """
    ca_coords: Dict[Tuple[str, int, str], np.ndarray] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain   = line[21]
            res_seq = int(line[22:26])
            ins     = line[26].strip()
            x       = float(line[30:38])
            y       = float(line[38:46])
            z       = float(line[46:54])
            if chains and chain not in chains:
                continue
            ca_coords[(chain, res_seq, ins)] = np.array([x, y, z])
    return ca_coords


def get_chain_sequence(pdb_path: str, chain: str) -> str:
    """Extract one-letter sequence for a chain from a PDB file."""
    aa3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    seen, seq = set(), []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            res_seq = int(line[22:26])
            ins     = line[26].strip()
            key     = (res_seq, ins)
            if key in seen:
                continue
            seen.add(key)
            res_name = line[17:20].strip()
            seq.append(aa3to1.get(res_name, "X"))
    return "".join(seq)


# ─────────────────────────────────────────────────────────────────────────────
# Binder-RMSD
# ─────────────────────────────────────────────────────────────────────────────

def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Kabsch algorithm: returns rotation matrix R such that P @ R ≈ Q.
    Both P and Q must already be centred.
    """
    H   = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d   = np.linalg.det(Vt.T @ U.T)
    D   = np.diag([1.0, 1.0, d])
    R   = Vt.T @ D @ U.T
    return R


def compute_binder_rmsd(
    design_pdb: str,
    reference_pdb: str,
    binder_chains: set = BINDER_CHAINS,
    all_chains: set = BINDER_CHAINS | {CHAIN_T},
) -> float:
    """
    1. Align the full complex (binder + target Cα) of design onto reference
       using the Kabsch algorithm.
    2. After alignment, compute Cα RMSD over binder chains only.

    Returns RMSD in Ångströms, or NaN if insufficient shared residues.
    """
    # Parse full complex for alignment
    des_all = parse_ca_atoms(design_pdb,   chains=all_chains)
    ref_all = parse_ca_atoms(reference_pdb, chains=all_chains)

    # Common residues across full complex
    common_all = sorted(set(des_all) & set(ref_all))
    if len(common_all) < 3:
        print(f"  [WARN] Too few shared complex residues for alignment "
              f"({len(common_all)}). Skipping RMSD.")
        return float("nan")

    P_all = np.array([des_all[k] for k in common_all])
    Q_all = np.array([ref_all[k] for k in common_all])

    # Centre
    p_com = P_all.mean(axis=0)
    q_com = Q_all.mean(axis=0)
    P_c   = P_all - p_com
    Q_c   = Q_all - q_com

    R = _kabsch(P_c, Q_c)

    # Parse binder-only residues
    des_binder = parse_ca_atoms(design_pdb,    chains=binder_chains)
    ref_binder = parse_ca_atoms(reference_pdb, chains=binder_chains)
    common_binder = sorted(set(des_binder) & set(ref_binder))

    if len(common_binder) < 3:
        print(f"  [WARN] Too few shared binder residues ({len(common_binder)}). "
              f"Skipping RMSD.")
        return float("nan")

    P_b = np.array([des_binder[k] for k in common_binder])
    Q_b = np.array([ref_binder[k] for k in common_binder])

    # Apply alignment (translate to full-complex centre, rotate, translate back)
    P_b_aligned = (P_b - p_com) @ R + q_com

    diff = P_b_aligned - Q_b
    rmsd = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
    return rmsd


# ─────────────────────────────────────────────────────────────────────────────
# FASTA preparation for ColabFold
# ─────────────────────────────────────────────────────────────────────────────

def write_colabfold_fasta(
    pdb_path: str,
    out_fasta: str,
    chains_ordered: List[str],
) -> bool:
    """
    Write a multimer FASTA for colabfold_batch.
    Chains are joined with ':' as the ColabFold multimer separator.
    Returns False if any chain sequence is empty.
    """
    seqs = []
    for ch in chains_ordered:
        s = get_chain_sequence(pdb_path, ch)
        if not s:
            print(f"  [WARN] Chain {ch} has no residues in {pdb_path}")
            return False
        seqs.append(s)

    stem   = Path(pdb_path).stem
    header = f">{stem}"
    body   = ":".join(seqs)

    with open(out_fasta, "w") as f:
        f.write(f"{header}\n{body}\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ColabFold runner
# ─────────────────────────────────────────────────────────────────────────────

def run_colabfold(
    fasta_path:      str,
    af2_out_dir:     str,
    conda_env:       str,
    num_recycles:    int = 3,
    num_models:      int = 1,
    use_gpu:         bool = True,
) -> bool:
    """
    Run colabfold_batch inside a conda environment.
    Returns True on success.
    """
    os.makedirs(af2_out_dir, exist_ok=True)

    # Build the activation + colabfold_batch command as a single shell string
    # so conda activate works correctly in a non-interactive shell.
    activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {conda_env}"
    cf_cmd   = (
        f"colabfold_batch"
        f" --num-recycle {num_recycles}"
        f" --num-models {num_models}"
        f" --model-type alphafold2_multimer_v3"
        f" --rank iptm"
        + (" --use-gpu-relax" if use_gpu else "")
        + f" {fasta_path} {af2_out_dir}"
    )
    full_cmd = f"{activate} && {cf_cmd}"

    print(f"  [AF2] Running colabfold_batch → {af2_out_dir}")
    result = subprocess.run(
        full_cmd, shell=True, executable="/bin/bash",
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  [ERROR] colabfold_batch failed (exit {result.returncode})")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# AF2 output parsing
# ─────────────────────────────────────────────────────────────────────────────

def find_top_af2_result(af2_out_dir: str, stem: str) -> Optional[Dict]:
    """
    Locate the rank_001 JSON scores file and PDB from colabfold_batch output.
    ColabFold names files like: <stem>_scores_rank_001_*.json
    """
    out = Path(af2_out_dir)
    # Scores JSON
    score_files = sorted(out.glob(f"{stem}*rank_001*scores*.json"))
    if not score_files:
        # Try without stem prefix (ColabFold sometimes truncates)
        score_files = sorted(out.glob("*rank_001*scores*.json"))
    if not score_files:
        print(f"  [WARN] No rank_001 scores JSON found in {af2_out_dir}")
        return None

    score_file = score_files[0]
    with open(score_file) as f:
        scores = json.load(f)

    # Relaxed or unrelaxed PDB
    pdb_files = sorted(out.glob(f"*rank_001*.pdb"))
    pdb_file  = str(pdb_files[0]) if pdb_files else None

    return {"scores": scores, "pdb": pdb_file, "scores_json": str(score_file)}


def extract_plddt(scores: Dict) -> float:
    """
    Mean pLDDT over all residues from a ColabFold scores JSON.
    The JSON contains a 'plddt' list (per-residue, 0–100 scale).
    """
    plddt = scores.get("plddt", [])
    if not plddt:
        return float("nan")
    return float(np.mean(plddt))


def extract_ipae(
    scores:        Dict,
    pdb_path:      str,
    binder_chains: set = BINDER_CHAINS,
    target_chain:  str = CHAIN_T,
) -> float:
    """
    Interface pAE (ipAE): mean PAE between binder residues and target residues.

    ColabFold scores JSON contains 'pae' as a 2-D list of shape (N, N).
    We average the off-diagonal block: pae[binder_idx, target_idx] and
    pae[target_idx, binder_idx].

    Chain residue counts are inferred from the AF2 prediction PDB itself.
    """
    pae_matrix = scores.get("pae")
    if pae_matrix is None:
        return float("nan")

    pae = np.array(pae_matrix)

    # Infer per-chain residue counts from the AF2 PDB (multimer output)
    # ColabFold preserves chain order: H, L, T (same as FASTA input order)
    if pdb_path is None or not os.path.isfile(pdb_path):
        return float("nan")

    chain_sizes: Dict[str, int] = {}
    current_chain, current_res = None, None
    count = 0
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            ch      = line[21]
            res_seq = int(line[22:26])
            ins     = line[26].strip()
            key     = (res_seq, ins)
            if ch != current_chain:
                if current_chain is not None:
                    chain_sizes[current_chain] = count
                current_chain = ch
                current_res   = set()
                count         = 0
            if key not in current_res:
                current_res.add(key)
                count += 1
    if current_chain is not None:
        chain_sizes[current_chain] = count

    # Build residue index ranges per chain
    chain_order = list(chain_sizes.keys())
    ranges: Dict[str, Tuple[int, int]] = {}
    start = 0
    for ch in chain_order:
        n = chain_sizes[ch]
        ranges[ch] = (start, start + n)
        start      += n

    binder_idx = []
    for ch in binder_chains:
        if ch in ranges:
            s, e = ranges[ch]
            binder_idx.extend(range(s, e))

    target_idx = []
    if target_chain in ranges:
        s, e = ranges[target_chain]
        target_idx = list(range(s, e))

    if not binder_idx or not target_idx:
        print(f"  [WARN] Could not locate binder or target residues in AF2 PDB.")
        return float("nan")

    b = np.array(binder_idx)
    t = np.array(target_idx)

    # Average PAE in both directions across the interface
    ipae = float(np.mean([pae[np.ix_(b, t)].mean(),
                          pae[np.ix_(t, b)].mean()]))
    return ipae


# ─────────────────────────────────────────────────────────────────────────────
# Per-design evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_design(
    pdb_path:        str,
    reference_pdb:   str,
    af2_work_dir:    str,
    conda_env:       str,
    num_recycles:    int  = 3,
    num_models:      int  = 1,
    use_gpu:         bool = True,
    overwrite:       bool = False,
    chains_for_fasta: List[str] = None,
) -> Dict:
    """
    Full evaluation pipeline for one design PDB.
    Returns a dict of scores.
    """
    chains_for_fasta = chains_for_fasta or [CHAIN_H, CHAIN_L, CHAIN_T]
    stem    = Path(pdb_path).stem
    out_dir = os.path.join(af2_work_dir, stem)

    result: Dict = {
        "design":       stem,
        "pdb_path":     pdb_path,
        "plddt":        None,
        "ipae":         None,
        "binder_rmsd":  None,
        "af2_pdb":      None,
        "af2_scores_json": None,
        "error":        None,
    }

    # ── Binder-RMSD (no AF2 needed) ──────────────────────────────────────────
    try:
        rmsd = compute_binder_rmsd(pdb_path, reference_pdb)
        result["binder_rmsd"] = round(rmsd, 4) if not np.isnan(rmsd) else None
        print(f"  Binder-RMSD = {rmsd:.3f} Å")
    except Exception as e:
        print(f"  [ERROR] Binder-RMSD failed: {e}")
        result["error"] = str(e)

    # ── AlphaFold2 via colabfold_batch ───────────────────────────────────────
    already_done = (
        not overwrite
        and os.path.isdir(out_dir)
        and any(Path(out_dir).glob("*rank_001*scores*.json"))
    )

    if already_done:
        print(f"  [AF2] Existing results found, skipping rerun (use --overwrite to force).")
    else:
        fasta_path = os.path.join(af2_work_dir, f"{stem}.fasta")
        ok = write_colabfold_fasta(pdb_path, fasta_path, chains_for_fasta)
        if not ok:
            result["error"] = "Failed to write FASTA (empty chain?)"
            return result

        success = run_colabfold(
            fasta_path=fasta_path,
            af2_out_dir=out_dir,
            conda_env=conda_env,
            num_recycles=num_recycles,
            num_models=num_models,
            use_gpu=use_gpu,
        )
        if not success:
            result["error"] = "colabfold_batch failed"
            return result

    af2_result = find_top_af2_result(out_dir, stem)
    if af2_result is None:
        result["error"] = "AF2 outputs not found after run"
        return result

    scores                    = af2_result["scores"]
    result["af2_pdb"]         = af2_result["pdb"]
    result["af2_scores_json"] = af2_result["scores_json"]

    plddt = extract_plddt(scores)
    result["plddt"] = round(plddt, 4) if not np.isnan(plddt) else None
    print(f"  pLDDT  = {plddt:.2f}")

    ipae = extract_ipae(scores, af2_result["pdb"])
    result["ipae"] = round(ipae, 4) if not np.isnan(ipae) else None
    print(f"  ipAE   = {ipae:.3f}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Evaluate final designs: pLDDT, ipAE (AF2), Binder-RMSD."
    )
    p.add_argument("--run_dir",   required=True,
                   help="Root output directory of a maturation run "
                        "(contains final_designs/)")
    p.add_argument("--reference", required=True,
                   help="Original --input PDB used in the maturation run "
                        "(HLT-formatted; used as RMSD reference)")
    p.add_argument("--output",    default="evaluation_results.json",
                   help="Path for the output JSON (default: evaluation_results.json)")
    p.add_argument("--colabfold_conda", default="colabfold",
                   help="Name or path of the conda environment containing "
                        "colabfold_batch (default: 'colabfold')")
    p.add_argument("--af2_num_recycles", type=int, default=3,
                   help="Number of AF2 recycles (default: 3)")
    p.add_argument("--af2_num_models",   type=int, default=1,
                   help="Number of AF2 models to run (default: 1)")
    p.add_argument("--device", default="cuda",
                   help="'cuda' or 'cpu' (default: cuda)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run AF2 even if outputs already exist")
    args = p.parse_args()

    # ── Locate final designs ─────────────────────────────────────────────────
    final_dir = Path(args.run_dir) / "final_designs"
    if not final_dir.is_dir():
        sys.exit(f"[ERROR] final_designs/ not found under {args.run_dir}")

    designs = sorted(final_dir.glob("*_final.pdb"))
    if not designs:
        sys.exit(f"[ERROR] No *_final.pdb files found in {final_dir}")

    print(f"[eval] Found {len(designs)} design(s) in {final_dir}")
    print(f"[eval] Reference PDB: {args.reference}")

    # ── AF2 working directory ────────────────────────────────────────────────
    af2_work = Path(args.run_dir) / "_af2_eval"
    af2_work.mkdir(exist_ok=True)

    use_gpu = args.device.lower() == "cuda"

    # ── Evaluate each design ─────────────────────────────────────────────────
    all_results = []
    for i, pdb in enumerate(designs):
        print(f"\n[eval] ({i+1}/{len(designs)}) {pdb.name}")
        res = evaluate_design(
            pdb_path=str(pdb),
            reference_pdb=args.reference,
            af2_work_dir=str(af2_work),
            conda_env=args.colabfold_conda,
            num_recycles=args.af2_num_recycles,
            num_models=args.af2_num_models,
            use_gpu=use_gpu,
            overwrite=args.overwrite,
        )
        all_results.append(res)

    # ── Sort by pLDDT descending (None values last) ──────────────────────────
    all_results.sort(
        key=lambda r: r["plddt"] if r["plddt"] is not None else -1,
        reverse=True,
    )

    # ── Write JSON ───────────────────────────────────────────────────────────
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval] Results written to {out_path}")

    # ── Terminal summary ─────────────────────────────────────────────────────
    print(f"\n{'Design':<45} {'pLDDT':>8} {'ipAE':>8} {'BinderRMSD':>12}")
    print("-" * 77)
    for r in all_results:
        plddt = f"{r['plddt']:.2f}"  if r["plddt"]       is not None else "N/A"
        ipae  = f"{r['ipae']:.3f}"   if r["ipae"]        is not None else "N/A"
        rmsd  = f"{r['binder_rmsd']:.3f}" if r["binder_rmsd"] is not None else "N/A"
        print(f"{r['design']:<45} {plddt:>8} {ipae:>8} {rmsd:>12}")


if __name__ == "__main__":
    main()