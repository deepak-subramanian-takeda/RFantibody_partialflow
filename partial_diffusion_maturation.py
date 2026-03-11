"""
partial_diffusion_maturation.py  (patched)

Key change: anchor residues are now passed to both:
  1. contigmap.contigs  — as motif segments (backbone placement)
  2. contigmap.provide_seq — as fixed-sequence positions (sequence + structure lock)

This combination is required for RFdiffusion partial diffusion to actually
hold anchor positions constant.  The contig string alone is insufficient
because partial_T steps of noise are still added to motif coordinates;
provide_seq instructs the model to treat those positions as fully resolved
and not to update their backbone frames during the reverse diffusion trajectory.

NOTE: provide_seq and hotspot_res are incompatible when provide_seq covers
*all* residues.  Here provide_seq only covers anchor positions, so the two
can coexist safely.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CDR_NAMES_H = ["H1", "H2", "H3"]
CDR_NAMES_L = ["L1", "L2", "L3"]
CDR_NAMES_ALL = CDR_NAMES_H + CDR_NAMES_L

CHAIN_H = "H"
CHAIN_L = "L"
CHAIN_T = "T"

_PDB_COORD_RECORDS = frozenset({"ATOM", "HETATM", "TER"})
_PDB_KEEP_RECORDS  = frozenset({"REMARK", "HEADER", "TITLE", "CRYST1"})


@dataclass
class CdrRange:
    name: str
    chain: str
    start: int
    end: int
    pdb_resnums: List[int] = field(default_factory=list)


@dataclass
class ResidueInfo:
    pose_idx: int
    pdb_chain: str
    pdb_resnum: int


# ---------------------------------------------------------------------------
# 1. HLT parsing
# ---------------------------------------------------------------------------

def parse_hlt_remarks(pdb_path: str) -> Dict[str, CdrRange]:
    remark_re = re.compile(
        r"^REMARK\s+PDBinfo-LABEL:\s+(\d+)\s+(H[123]|L[123])\s*$"
    )
    cdr_positions: Dict[str, List[int]] = {n: [] for n in CDR_NAMES_ALL}

    with open(pdb_path) as fh:
        for line in fh:
            m = remark_re.match(line.strip())
            if m:
                abs_idx = int(m.group(1))
                cdr_name = m.group(2)
                if cdr_name in cdr_positions:
                    cdr_positions[cdr_name].append(abs_idx)

    ranges: Dict[str, CdrRange] = {}
    for name, positions in cdr_positions.items():
        if positions:
            chain = name[0]
            ranges[name] = CdrRange(
                name=name,
                chain=chain,
                start=min(positions),
                end=max(positions),
            )

    return ranges


def read_pdb_residues(pdb_path: str) -> List[ResidueInfo]:
    seen = set()
    residues: List[ResidueInfo] = []
    pose_idx = 0

    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21].strip()
            if chain not in (CHAIN_H, CHAIN_L, CHAIN_T):
                continue
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            key = (chain, resnum)
            if key not in seen:
                seen.add(key)
                pose_idx += 1
                residues.append(ResidueInfo(
                    pose_idx=pose_idx,
                    pdb_chain=chain,
                    pdb_resnum=resnum,
                ))

    return residues


def build_residue_lookup(
    residues: List[ResidueInfo],
) -> Dict[Tuple[str, int], ResidueInfo]:
    return {(r.pdb_chain, r.pdb_resnum): r for r in residues}


# ---------------------------------------------------------------------------
# 1b. Split HLT complex
# ---------------------------------------------------------------------------

def split_hlt_complex(
    complex_pdb: str,
    out_dir:     str,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(complex_pdb).stem

    target_path    = os.path.join(out_dir, f"{stem}_target.pdb")
    framework_path = os.path.join(out_dir, f"{stem}_framework.pdb")

    target_lines:    List[str] = []
    framework_lines: List[str] = []
    remark_lines:    List[str] = []

    with open(complex_pdb) as fh:
        for line in fh:
            record = line[:6].strip()

            if record == "REMARK":
                remark_lines.append(line)
                continue

            if record in _PDB_KEEP_RECORDS:
                target_lines.append(line)
                framework_lines.append(line)
                continue

            if record in _PDB_COORD_RECORDS:
                if record == "TER":
                    target_lines.append(line)
                    framework_lines.append(line)
                    continue

                chain = line[21] if len(line) > 21 else " "

                if chain == CHAIN_T:
                    target_lines.append(line)
                elif chain in (CHAIN_H, CHAIN_L):
                    framework_lines.append(line)

            elif record in {"END", "MASTER"}:
                target_lines.append(line)
                framework_lines.append(line)

    with open(target_path, "w") as fh:
        fh.writelines(target_lines)
        if not target_lines or not target_lines[-1].startswith("END"):
            fh.write("END\n")

    with open(framework_path, "w") as fh:
        header = [l for l in framework_lines
                  if l[:6].strip() in _PDB_KEEP_RECORDS]
        coord  = [l for l in framework_lines
                  if l[:6].strip() not in _PDB_KEEP_RECORDS]
        fh.writelines(header)
        fh.writelines(remark_lines)
        fh.writelines(coord)
        if not coord or not coord[-1].startswith("END"):
            fh.write("END\n")

    return target_path, framework_path


# ---------------------------------------------------------------------------
# 2. Contig string builder
# ---------------------------------------------------------------------------

def build_contig_string(
    residues: List[ResidueInfo],
    cdr_ranges: Dict[str, CdrRange],
    anchor_residues: List[Tuple[str, int]],
    free_loop_overrides: Dict[str, Tuple[int, int]],
    nanobody: bool = False,
) -> str:
    anchor_set = set(anchor_residues)

    cdr_pose_idx_to_name: Dict[int, str] = {}
    for name, r in cdr_ranges.items():
        for idx in range(r.start, r.end + 1):
            cdr_pose_idx_to_name[idx] = name

    h_residues = [r for r in residues if r.pdb_chain == CHAIN_H]
    l_residues = [r for r in residues if r.pdb_chain == CHAIN_L]
    t_residues = [r for r in residues if r.pdb_chain == CHAIN_T]

    def chain_to_segments(chain_residues: List[ResidueInfo]) -> str:
        tokens: List[str] = []
        i = 0
        while i < len(chain_residues):
            r = chain_residues[i]
            cdr_name = cdr_pose_idx_to_name.get(r.pose_idx)

            if cdr_name is None:
                run_start = r
                while (i < len(chain_residues) and
                       cdr_pose_idx_to_name.get(chain_residues[i].pose_idx) is None):
                    i += 1
                run_end = chain_residues[i - 1]
                tokens.append(f"{run_start.pdb_chain}{run_start.pdb_resnum}"
                               f"-{run_end.pdb_resnum}")
            else:
                cdr_r = cdr_ranges[cdr_name]
                cdr_res = [chain_residues[j]
                            for j in range(i, len(chain_residues))
                            if chain_residues[j].pose_idx <= cdr_r.end]
                i += len(cdr_res)

                loop_anchors = [(r2.pdb_chain, r2.pdb_resnum)
                                for r2 in cdr_res
                                if (r2.pdb_chain, r2.pdb_resnum) in anchor_set]

                if not loop_anchors:
                    orig_len = len(cdr_res)
                    min_l, max_l = free_loop_overrides.get(
                        cdr_name, (max(1, orig_len - 2), orig_len + 2)
                    )
                    tokens.append(f"{min_l}-{max_l}")
                else:
                    tokens.extend(_cdr_with_anchors(
                        cdr_res, loop_anchors, cdr_name,
                        free_loop_overrides, anchor_set
                    ))

        return "/".join(tokens)

    h_seg = chain_to_segments(h_residues)
    t_seg = chain_to_segments(t_residues)

    if nanobody or not l_residues:
        return f"{h_seg}/0 {t_seg}"
    else:
        l_seg = chain_to_segments(l_residues)
        return f"{h_seg}/0 {l_seg}/0 {t_seg}"


def _cdr_with_anchors(
    cdr_residues: List[ResidueInfo],
    loop_anchors: List[Tuple[str, int]],
    cdr_name: str,
    free_loop_overrides: Dict[str, Tuple[int, int]],
    anchor_set: set,
) -> List[str]:
    tokens: List[str] = []

    in_anchor_run = False
    anchor_run_start: Optional[ResidueInfo] = None
    anchor_run_end:   Optional[ResidueInfo] = None
    gap_count = 0

    def flush_gap(n: int):
        if n > 0:
            tokens.append(f"{n}-{n}")

    def flush_anchor_run():
        nonlocal in_anchor_run, anchor_run_start, anchor_run_end
        if in_anchor_run and anchor_run_start is not None:
            ch = anchor_run_start.pdb_chain
            tokens.append(f"{ch}{anchor_run_start.pdb_resnum}"
                           f"-{anchor_run_end.pdb_resnum}")
        in_anchor_run = False
        anchor_run_start = None
        anchor_run_end = None

    for r in cdr_residues:
        is_anchor = (r.pdb_chain, r.pdb_resnum) in anchor_set
        if is_anchor:
            if gap_count:
                flush_gap(gap_count)
                gap_count = 0
            if not in_anchor_run:
                in_anchor_run = True
                anchor_run_start = r
            anchor_run_end = r
        else:
            if in_anchor_run:
                flush_anchor_run()
            gap_count += 1

    if in_anchor_run:
        flush_anchor_run()
    flush_gap(gap_count)

    return tokens


# ---------------------------------------------------------------------------
# NEW: Build provide_seq string from anchor + framework residues
# ---------------------------------------------------------------------------

def build_provide_seq(
    residues: List[ResidueInfo],
    cdr_ranges: Dict[str, CdrRange],
    anchor_residues: List[Tuple[str, int]],
    nanobody: bool = False,
) -> str:
    """
    Build the contigmap.provide_seq list that tells RFdiffusion to treat
    certain residues as fully resolved (sequence AND backbone fixed) during
    the reverse diffusion trajectory.

    We include:
      - All framework residues on chains H/L (they are already fixed as motif
        segments in the contig string, but provide_seq reinforces backbone
        fixation under noise).
      - All anchor residues (the primary target of this fix).
      - All target chain T residues (antigen — always fixed).

    The provide_seq format is a comma-separated list of pose indices
    (1-indexed, Rosetta-style absolute across the whole pose):
        contigmap.provide_seq=[1,2,3,7,8,45]

    Parameters
    ----------
    residues
        Full ordered residue list from read_pdb_residues().
    cdr_ranges
        CDR name -> CdrRange from parse_hlt_remarks().
    anchor_residues
        (chain, resnum) pairs to fix; from load_anchors().
    nanobody
        If True, skip L-chain.

    Returns
    -------
    Comma-separated string of pose indices, e.g. "1,2,3,7,8,45,46,47"
    """
    anchor_set = set(anchor_residues)

    # Build a set of ALL pose indices that fall inside any CDR
    cdr_pose_indices: set = set()
    for r in cdr_ranges.values():
        for idx in range(r.start, r.end + 1):
            cdr_pose_indices.add(idx)

    fixed_pose_indices: List[int] = []

    for r in residues:
        if r.pdb_chain == CHAIN_T:
            # Antigen: always fixed
            fixed_pose_indices.append(r.pose_idx)
        elif r.pdb_chain in (CHAIN_H, CHAIN_L):
            if nanobody and r.pdb_chain == CHAIN_L:
                continue
            # Framework residue (not in any CDR): always fixed
            if r.pose_idx not in cdr_pose_indices:
                fixed_pose_indices.append(r.pose_idx)
            # CDR residue that is an anchor: fixed
            elif (r.pdb_chain, r.pdb_resnum) in anchor_set:
                fixed_pose_indices.append(r.pose_idx)
            # Non-anchor CDR residue: freely diffused — NOT added

    return ",".join(str(i) for i in sorted(fixed_pose_indices))


# ---------------------------------------------------------------------------
# 3. Parse free-loop overrides
# ---------------------------------------------------------------------------

def parse_free_loops(spec: str) -> Dict[str, Tuple[int, int]]:
    result: Dict[str, Tuple[int, int]] = {}
    if not spec:
        return result
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(H[123]|L[123]):(\d+)-(\d+)$", part)
        if not m:
            raise ValueError(
                f"Cannot parse free_loops spec '{part}'. "
                "Expected format: H3:5-13  (CDR:min-max)"
            )
        name, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        if lo > hi:
            raise ValueError(
                f"min_len ({lo}) > max_len ({hi}) for loop {name}"
            )
        result[name] = (lo, hi)
    return result


# ---------------------------------------------------------------------------
# 4. Load anchor data from Step 0 JSON
# ---------------------------------------------------------------------------

def load_anchors(json_path: str) -> List[Tuple[str, int]]:
    with open(json_path) as fh:
        data = json.load(fh)

    anchors: List[Tuple[str, int]] = []
    for entry in data.get("all_anchor_residues", []):
        m = re.match(r"^([HLT])(\d+)$", entry)
        if m:
            anchors.append((m.group(1), int(m.group(2))))
        else:
            print(f"[WARN] Cannot parse anchor residue '{entry}', skipping.")
    return anchors

def mask_anchors_in_hlt(
    input_pdb:       str,
    anchor_residues: List[Tuple[str, int]],
    out_path:        str,
) -> str:
    """
    Write a copy of input_pdb to out_path where HLT REMARK PDBinfo-LABEL
    lines for anchor residues have been removed.

    How AbSampler uses REMARK lines
    --------------------------------
    AbSampler.from_HLT() calls HLT_pdb_parser() which reads
    REMARK PDBinfo-LABEL lines to build loop_masks — boolean arrays that
    mark which residues belong to each CDR loop.  These loop_masks directly
    become the diffusion_mask: True = diffuse (free), False = keep fixed.

    Residues NOT annotated by any REMARK line are treated as framework and
    held fixed during partial diffusion.  By removing the REMARK lines for
    anchor residues, we demote them from "CDR / diffusible" to "framework /
    fixed", which is exactly what we want.

    The anchor residues identified by Step 0 are specified as absolute
    pose indices in the REMARK lines (e.g. "REMARK PDBinfo-LABEL:  105 H3").
    We need to convert (chain, pdb_resnum) anchor pairs to absolute pose
    indices to know which REMARK lines to drop.

    Parameters
    ----------
    input_pdb        : HLT-annotated complex PDB (Step 0 / original input)
    anchor_residues  : list of (chain, pdb_resnum) pairs from Step 0 JSON
    out_path         : path to write the modified PDB

    Returns
    -------
    out_path
    """
    # Build (chain, pdb_resnum) -> absolute_pose_index mapping
    # Absolute index = 1-based sequential count across ALL chains in H/L/T order
    seen = set()
    pose_idx = 0
    resnum_to_abs: Dict[Tuple[str, int], int] = {}

    with open(input_pdb) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21].strip()
            if chain not in ("H", "L", "T"):
                continue
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            key = (chain, resnum)
            if key not in seen:
                seen.add(key)
                pose_idx += 1
                resnum_to_abs[key] = pose_idx

    # Build set of absolute indices to suppress
    anchor_abs_indices: set = set()
    for chain, resnum in anchor_residues:
        abs_idx = resnum_to_abs.get((chain, resnum))
        if abs_idx is not None:
            anchor_abs_indices.add(abs_idx)
        else:
            print(f"  [WARN] Anchor {chain}{resnum} not found in PDB — "
                  "cannot remove from REMARK lines.")

    # REMARK line format: "REMARK PDBinfo-LABEL:  <abs_idx> <CDR_NAME>"
    remark_re = re.compile(
        r"^REMARK\s+PDBinfo-LABEL:\s+(\d+)\s+(H[123]|L[123])\s*$"
    )

    out_lines = []
    n_removed = 0

    with open(input_pdb) as fh:
        for line in fh:
            if line.startswith("REMARK"):
                m = remark_re.match(line.strip())
                if m and int(m.group(1)) in anchor_abs_indices:
                    n_removed += 1
                    continue   # drop this REMARK line
            out_lines.append(line)

    with open(out_path, "w") as fh:
        fh.writelines(out_lines)

    print(f"  [mask_anchors] Removed {n_removed} REMARK line(s) for "
          f"{len(anchor_abs_indices)} anchor position(s)")
    print(f"  [mask_anchors] Modified PDB written to: {out_path}")

    return out_path

# ---------------------------------------------------------------------------
# 5. Build rfdiffusion CLI command
# ---------------------------------------------------------------------------

def build_rfdiffusion_command(
    input_pdb:      str,
    target_pdb:     str,
    framework_pdb:  str,
    provide_seq:    str,       # ← NEW: comma-separated fixed pose indices
    hotspots:       str,
    output_prefix:  str,
    partial_T:      int,
    num_designs:    int,
    model_weights:  str,
    extra_args:     List[str],
) -> List[str]:
    script_dir = Path(__file__).resolve().parent
    inference_candidates = [
        script_dir / "src" / "rfantibody" / "rfdiffusion" / "rfdiffusion_inference.py",
        script_dir.parent / "src" / "rfantibody" / "rfdiffusion" / "rfdiffusion_inference.py",
        script_dir / "rfdiffusion_inference.py",
    ]
    inference_script: Optional[str] = None
    for candidate in inference_candidates:
        if candidate.is_file():
            inference_script = str(candidate)
            break

    if inference_script is None:
        inference_script = os.environ.get("RFANTIBODY_INFERENCE_SCRIPT", "/home/pymc/Deepak/RFantibody_partialflow/src/rfantibody/rfdiffusion/rfdiffusion_inference.py")
        if not inference_script or not os.path.isfile(inference_script):
            raise FileNotFoundError(
                "Cannot locate rfdiffusion_inference.py. "
                "Set RFANTIBODY_INFERENCE_SCRIPT to its absolute path."
            )

    cmd = [
        sys.executable,
        inference_script,
        f"inference.input_pdb={input_pdb}",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={num_designs}",
    ]

    if model_weights:
        cmd.append(f"inference.ckpt_override_path={model_weights}")

    # ── KEY FIX: provide_seq locks anchor + framework + target backbone ──
    # Without this, partial_T steps of added Gaussian noise shift the
    # "fixed" motif segments away from their input coordinates.
    # provide_seq instructs the model to keep these positions fully resolved
    # throughout the reverse diffusion trajectory.
    if provide_seq:
        cmd.append(f"'contigmap.provide_seq=[{provide_seq}]'")

    # Partial diffusion noise depth
    cmd.append(f"diffuser.partial_T={partial_T}")

    # Hotspot residues
    if hotspots:
        cmd.append(f"'ppi.hotspot_res=[{hotspots}]'")

    cmd.extend(extra_args)

    return cmd


# ---------------------------------------------------------------------------
# 6. Summary / dry-run printer
# ---------------------------------------------------------------------------

def print_summary(
    input_pdb:   str,
    anchors:     List[Tuple[str, int]],
    cdr_ranges:  Dict[str, CdrRange],
    free_loops:  Dict[str, Tuple[int, int]],
    provide_seq: str,
    cmd:         List[str],
    partial_T:   int,
    num_designs: int,
):
    print("\n" + "=" * 70)
    print("  Step 1 — Partial Diffusion Maturation")
    print("=" * 70)
    print(f"  Input complex : {input_pdb}")
    print(f"  partial_T     : {partial_T}  (out of 50 total steps)")
    print(f"  num_designs   : {num_designs}")
    print()

    print("  CDR ranges from HLT REMARKs:")
    for name in sorted(cdr_ranges):
        r = cdr_ranges[name]
        print(f"    {name}: absolute residues {r.start}–{r.end} "
              f"({r.end - r.start + 1} residues)")

    print()
    print(f"  Anchor residues ({len(anchors)} total):")
    if anchors:
        for ch, rn in sorted(anchors):
            print(f"    {ch}{rn}")
    else:
        print("    (none — all CDR loops will be freely diffused)")

    if free_loops:
        print()
        print("  Free-loop length overrides:")
        for name, (lo, hi) in sorted(free_loops.items()):
            print(f"    {name}: {lo}–{hi}")

    # Show how many positions are locked by provide_seq
    n_fixed = len(provide_seq.split(",")) if provide_seq else 0
    print()
    print(f"  provide_seq   : {n_fixed} residue(s) fixed "
          f"(framework + anchors + target)")
    print(f"    [{provide_seq[:80]}{'...' if len(provide_seq) > 80 else ''}]")

    print()
    print("  RFdiffusion command:")
    print("    " + " \\\n        ".join(cmd))
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# 7. CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 1: Build partial diffusion contig from Step 0 anchors "
            "and run RFantibody's rfdiffusion for in silico maturation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--input", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--hotspots", required=True)
    p.add_argument("--model_weights", required=True)

    p.add_argument("--partial_T", type=int, default=15)
    p.add_argument("--num_designs", type=int, default=20)
    p.add_argument("--free_loops", default="")
    p.add_argument("--nanobody", action="store_true")
    p.add_argument("--name", default="")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("extra", nargs=argparse.REMAINDER)

    return p.parse_args()


def main():
    args = parse_args()

    input_pdb    = str(Path(args.input).resolve())
    anchors_json = str(Path(args.anchors).resolve())
    output_dir   = str(Path(args.output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    stem = args.name or Path(input_pdb).stem
    output_prefix = os.path.join(output_dir, f"{stem}_partial_T{args.partial_T}")

    if not 1 <= args.partial_T <= 50:
        sys.exit(f"[ERROR] --partial_T must be between 1 and 50 "
                 f"(got {args.partial_T}).")

    print(f"[Step 1] Parsing HLT REMARK annotations from {input_pdb}")
    cdr_ranges = parse_hlt_remarks(input_pdb)
    if not cdr_ranges:
        sys.exit("[ERROR] No CDR REMARK lines found.")
    print(f"         Found CDRs: {', '.join(sorted(cdr_ranges))}")

    print(f"[Step 1] Reading residue list from PDB...")
    residues = read_pdb_residues(input_pdb)
    n_h = sum(1 for r in residues if r.pdb_chain == CHAIN_H)
    n_l = sum(1 for r in residues if r.pdb_chain == CHAIN_L)
    n_t = sum(1 for r in residues if r.pdb_chain == CHAIN_T)
    print(f"         H:{n_h}  L:{n_l}  T:{n_t}  total:{len(residues)}")

    if args.nanobody and n_l > 0:
        print("[WARN] --nanobody flag set but L-chain residues found.")
    if not args.nanobody and n_l == 0:
        print("[INFO] No L-chain residues detected; treating as nanobody.")
        args.nanobody = True

    print(f"[Step 1] Splitting HLT complex into target and framework PDBs...")
    split_dir = os.path.join(output_dir, "_split")
    target_pdb, framework_pdb = split_hlt_complex(input_pdb, split_dir)
    print(f"         Target   : {target_pdb}")
    print(f"         Framework: {framework_pdb}")

    t_residues  = [r for r in residues if r.pdb_chain == CHAIN_T]
    ab_residues = [r for r in residues if r.pdb_chain in (CHAIN_H, CHAIN_L)]
    if not t_residues:
        sys.exit("[ERROR] No chain T (target) residues found.")
    if not ab_residues:
        sys.exit("[ERROR] No H/L (antibody) residues found.")

    print(f"[Step 1] Loading anchor residues from {anchors_json}")
    anchor_residues = load_anchors(anchors_json)

    # Mask anchor positions in HLT REMARK lines so AbSampler treats them
    # as framework (fixed) rather than CDR (diffusible)
    masked_pdb_path = os.path.join(output_dir, f"{stem}_anchors_masked.pdb")
    input_pdb = mask_anchors_in_hlt(
        input_pdb=input_pdb,
        anchor_residues=anchor_residues,
        out_path=masked_pdb_path,
    )
    print(f"[Step 1] Using anchor-masked PDB: {input_pdb}")
    print(f"         {len(anchor_residues)} anchor(s): "
          f"{[f'{c}{n}' for c, n in anchor_residues]}")

    print(f"[Step 1] Parsing free-loop overrides: '{args.free_loops}'")
    try:
        free_loops = parse_free_loops(args.free_loops)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    # ── NEW: Build provide_seq string ──────────────────────────────────────
    print("[Step 1] Building provide_seq (fixed backbone positions)...")
    provide_seq = build_provide_seq(
        residues=residues,
        cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues,
        nanobody=args.nanobody,
    )
    n_fixed = len(provide_seq.split(",")) if provide_seq else 0
    print(f"         {n_fixed} residue(s) marked as fixed "
          f"(framework + anchors + antigen)")

    # ── Build CLI command ──────────────────────────────────────────────────
    extra = [a for a in args.extra if a != "--"]
    cmd = build_rfdiffusion_command(
        input_pdb=input_pdb,
        target_pdb=target_pdb,
        framework_pdb=framework_pdb,
        provide_seq=provide_seq,
        hotspots=args.hotspots,
        output_prefix=output_prefix,
        partial_T=args.partial_T,
        num_designs=args.num_designs,
        model_weights=args.model_weights,
        extra_args=extra,
    )

    print_summary(
        input_pdb=input_pdb,
        anchors=anchor_residues,
        cdr_ranges=cdr_ranges,
        free_loops=free_loops,
        provide_seq=provide_seq,
        cmd=cmd,
        partial_T=args.partial_T,
        num_designs=args.num_designs,
    )

    record_path = os.path.join(output_dir, f"{stem}_contig.json")
    with open(record_path, "w") as fh:
        json.dump({
            "input_pdb":        input_pdb,
            "target_pdb":       target_pdb,
            "framework_pdb":    framework_pdb,
            "partial_T":        args.partial_T,
            "provide_seq":      provide_seq,
            "anchor_residues":  [f"{c}{n}" for c, n in anchor_residues],
            "free_loop_overrides": {k: list(v) for k, v in free_loops.items()},
            "command":          cmd,
        }, fh, indent=2)
    print(f"[Step 1] Contig record saved to: {record_path}")

    if args.dry_run:
        print("[Step 1] DRY RUN — rfdiffusion not invoked.")
        return

    print("[Step 1] Launching rfdiffusion...")
    shell_cmd = " ".join(cmd)
    print(f"         $ {shell_cmd}\n")

    result = subprocess.run(shell_cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"[ERROR] rfdiffusion exited with code {result.returncode}")

    print(f"\n[Step 1] Complete. Outputs written to: {output_dir}/")
    print(f"         Feed these PDBs into Step 2 (ProteinMPNN) using:")
    print(f"         proteinmpnn -i {output_dir}/ "
          f"--output-quiver step2_mpnn.qv "
          f"--fixed-positions-jsonl anchors/{stem}_fixed_positions.jsonl")


if __name__ == "__main__":
    main()