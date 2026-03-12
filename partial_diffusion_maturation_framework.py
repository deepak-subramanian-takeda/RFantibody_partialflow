"""
partial_diffusion_maturation_framework.py

Variant of Step 1 that accepts a separate HLT-annotated framework PDB
(containing only H and L chains with CDR REMARK lines) and a separate
target PDB (containing only chain T), instead of a single pre-merged
HLT complex.

Key differences from partial_diffusion_maturation.py
------------------------------------------------------
1. --input is replaced by --framework (H/L only) and --target (T only).
2. The two files are merged in memory into a temporary HLT complex so
   that all downstream logic (REMARK parsing, residue indexing,
   provide_seq, contig building) operates on a single consistently
   ordered file, exactly as before.
3. split_hlt_complex() is no longer called — the split already exists.
4. graft_target_sequence() still runs post-diffusion, using --target as
   the source of original T-chain residue names.
5. The anchor-masking step writes to a masked copy of the merged file.
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
# Constants
# ---------------------------------------------------------------------------

CDR_NAMES_H   = ["H1", "H2", "H3"]
CDR_NAMES_L   = ["L1", "L2", "L3"]
CDR_NAMES_ALL = CDR_NAMES_H + CDR_NAMES_L

CHAIN_H = "H"
CHAIN_L = "L"
CHAIN_T = "T"

_PDB_COORD_RECORDS = frozenset({"ATOM", "HETATM", "TER"})
_PDB_KEEP_RECORDS  = frozenset({"REMARK", "HEADER", "TITLE", "CRYST1"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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
# 1. Merge framework + target into a single HLT file
# ---------------------------------------------------------------------------

def merge_framework_and_target(
    framework_pdb:   str,          # H/L chains + CDR REMARK lines
    target_pdb:      str,          # target chain only
    out_path:        str,
    target_chain_in: str = "T",    # chain ID as it appears in target_pdb
) -> str:
    """
    Write a merged HLT PDB to out_path with chain order H → L → T.

    Strategy
    --------
    - All REMARK PDBinfo-LABEL lines are taken from framework_pdb and
      written first (they carry absolute pose indices that will be correct
      once T-chain residues are appended at the end).
    - All other header/metadata REMARKs from both files are preserved.
    - ATOM/HETATM/TER records for H and L come from framework_pdb.
    - ATOM/HETATM/TER records for T come from target_pdb.
    - A single END is written at the very end.

    The absolute pose indices in the REMARK lines are computed against the
    framework alone (H+L residues only), so they remain valid in the merged
    file because T residues are appended after all H/L residues.

    Parameters
    ----------
    framework_pdb : HLT-annotated PDB with H and L chains only
    target_pdb    : PDB with T chain only (no REMARK annotations needed)
    out_path      : path for the merged output PDB

    Returns
    -------
    out_path
    """
    remark_label_lines: List[str] = []   # CDR annotation REMARKs
    other_remark_lines: List[str] = []   # non-label REMARKs from framework
    header_lines:       List[str] = []   # HEADER/TITLE/CRYST1 from framework
    hl_coord_lines:     List[str] = []   # ATOM/HETATM/TER for H and L
    t_coord_lines:      List[str] = []   # ATOM/HETATM/TER for T

    label_re = re.compile(r"^REMARK\s+PDBinfo-LABEL:")

    # --- Read framework (H/L) ---
    with open(framework_pdb) as fh:
        for line in fh:
            record = line[:6].strip()
            if record == "REMARK":
                if label_re.match(line):
                    remark_label_lines.append(
                        line if line.endswith("\n") else line + "\n"
                    )
                else:
                    other_remark_lines.append(
                        line if line.endswith("\n") else line + "\n"
                    )
            elif record in {"HEADER", "TITLE", "CRYST1"}:
                header_lines.append(line if line.endswith("\n") else line + "\n")
            elif record in _PDB_COORD_RECORDS:
                if record == "TER":
                    hl_coord_lines.append(
                        line if line.endswith("\n") else line + "\n"
                    )
                    continue
                chain = line[21] if len(line) > 21 else " "
                if chain in (CHAIN_H, CHAIN_L):
                    hl_coord_lines.append(
                        line if line.endswith("\n") else line + "\n"
                    )
                # silently skip any unexpected chains in the framework file
            # END / MASTER are dropped; we write a single END at the finish

    if not hl_coord_lines:
        raise ValueError(
            f"No H or L chain ATOM/HETATM records found in {framework_pdb}"
        )
    if not remark_label_lines:
        raise ValueError(
            f"No CDR REMARK PDBinfo-LABEL lines found in {framework_pdb}. "
            "Is this a valid HLT-annotated framework PDB?"
        )

    # --- Read target ---
    with open(target_pdb) as fh:
        for line in fh:
            record = line[:6].strip()
            if record in _PDB_COORD_RECORDS:
                if record == "TER":
                    t_coord_lines.append(
                        line if line.endswith("\n") else line + "\n"
                    )
                    continue
                chain = line[21] if len(line) > 21 else " "
                if chain == target_chain_in:
                    # Remap to T if the source uses a different chain letter
                    if target_chain_in != CHAIN_T:
                        line = line[:21] + CHAIN_T + line[22:]
                    t_coord_lines.append(
                        line if line.endswith("\n") else line + "\n"
                    )
                else:
                    print(
                        f"  [WARN] merge_framework_and_target: unexpected chain "
                        f"'{chain}' in target PDB {target_pdb} — skipping line."
                    )

    if not t_coord_lines:
        raise ValueError(
            f"No T chain ATOM/HETATM records found in {target_pdb}"
        )

    # --- Write merged file ---
    with open(out_path, "w") as fh:
        # Header / metadata
        fh.writelines(header_lines)
        fh.writelines(other_remark_lines)
        # CDR annotations — must come before ATOM records for some parsers
        fh.writelines(remark_label_lines)
        # Coordinates: H/L then T
        fh.writelines(hl_coord_lines)
        fh.writelines(t_coord_lines)
        fh.write("END\n")

    n_label = len(remark_label_lines)
    n_hl    = sum(1 for l in hl_coord_lines if l[:6].strip() in {"ATOM","HETATM"})
    n_t     = sum(1 for l in t_coord_lines  if l[:6].strip() in {"ATOM","HETATM"})
    print(f"  [merge] {n_label} CDR REMARK line(s), "
          f"{n_hl} H/L ATOM record(s), {n_t} T ATOM record(s) -> {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# 2. HLT parsing  (unchanged from original)
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
                abs_idx  = int(m.group(1))
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
    seen     = set()
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
# 3. Contig string builder  (unchanged from original)
# ---------------------------------------------------------------------------

def build_contig_string(
    residues:           List[ResidueInfo],
    cdr_ranges:         Dict[str, CdrRange],
    anchor_residues:    List[Tuple[str, int]],
    free_loop_overrides: Dict[str, Tuple[int, int]],
    nanobody:           bool = False,
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
            r        = chain_residues[i]
            cdr_name = cdr_pose_idx_to_name.get(r.pose_idx)

            if cdr_name is None:
                run_start = r
                while (i < len(chain_residues) and
                       cdr_pose_idx_to_name.get(chain_residues[i].pose_idx) is None):
                    i += 1
                run_end = chain_residues[i - 1]
                tokens.append(
                    f"{run_start.pdb_chain}{run_start.pdb_resnum}"
                    f"-{run_end.pdb_resnum}"
                )
            else:
                cdr_r   = cdr_ranges[cdr_name]
                cdr_res = [chain_residues[j]
                           for j in range(i, len(chain_residues))
                           if chain_residues[j].pose_idx <= cdr_r.end]
                i += len(cdr_res)

                loop_anchors = [
                    (r2.pdb_chain, r2.pdb_resnum)
                    for r2 in cdr_res
                    if (r2.pdb_chain, r2.pdb_resnum) in anchor_set
                ]

                if not loop_anchors:
                    orig_len = len(cdr_res)
                    min_l, max_l = free_loop_overrides.get(
                        cdr_name, (max(1, orig_len - 2), orig_len + 2)
                    )
                    tokens.append(f"{min_l}-{max_l}")
                else:
                    tokens.extend(_cdr_with_anchors(
                        cdr_res, loop_anchors, cdr_name,
                        free_loop_overrides, anchor_set,
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
    cdr_residues:       List[ResidueInfo],
    loop_anchors:       List[Tuple[str, int]],
    cdr_name:           str,
    free_loop_overrides: Dict[str, Tuple[int, int]],
    anchor_set:         set,
) -> List[str]:
    tokens: List[str] = []
    in_anchor_run      = False
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
            tokens.append(
                f"{ch}{anchor_run_start.pdb_resnum}"
                f"-{anchor_run_end.pdb_resnum}"
            )
        in_anchor_run     = False
        anchor_run_start  = None
        anchor_run_end    = None

    for r in cdr_residues:
        is_anchor = (r.pdb_chain, r.pdb_resnum) in anchor_set
        if is_anchor:
            if gap_count:
                flush_gap(gap_count)
                gap_count = 0
            if not in_anchor_run:
                in_anchor_run    = True
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
# 4. provide_seq builder  (unchanged from original)
# ---------------------------------------------------------------------------

def build_provide_seq(
    residues:        List[ResidueInfo],
    cdr_ranges:      Dict[str, CdrRange],
    anchor_residues: List[Tuple[str, int]],
    nanobody:        bool = False,
) -> str:
    anchor_set = set(anchor_residues)

    cdr_pose_indices: set = set()
    for r in cdr_ranges.values():
        for idx in range(r.start, r.end + 1):
            cdr_pose_indices.add(idx)

    fixed_pose_indices: List[int] = []

    for r in residues:
        if r.pdb_chain == CHAIN_T:
            fixed_pose_indices.append(r.pose_idx)
        elif r.pdb_chain in (CHAIN_H, CHAIN_L):
            if nanobody and r.pdb_chain == CHAIN_L:
                continue
            if r.pose_idx not in cdr_pose_indices:
                fixed_pose_indices.append(r.pose_idx)
            elif (r.pdb_chain, r.pdb_resnum) in anchor_set:
                fixed_pose_indices.append(r.pose_idx)

    return ",".join(str(i) for i in sorted(fixed_pose_indices))


# ---------------------------------------------------------------------------
# 5. Parse free-loop overrides  (unchanged from original)
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
# 6. Load anchor data  (unchanged from original)
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


# ---------------------------------------------------------------------------
# 7. Mask anchor REMARK lines  (unchanged from original)
# ---------------------------------------------------------------------------

def mask_anchors_in_hlt(
    input_pdb:       str,
    anchor_residues: List[Tuple[str, int]],
    out_path:        str,
) -> str:
    seen     = set()
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

    anchor_abs_indices: set = set()
    for chain, resnum in anchor_residues:
        abs_idx = resnum_to_abs.get((chain, resnum))
        if abs_idx is not None:
            anchor_abs_indices.add(abs_idx)
        else:
            print(f"  [WARN] Anchor {chain}{resnum} not found in PDB — "
                  "cannot remove from REMARK lines.")

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
                    continue
            out_lines.append(line)

    with open(out_path, "w") as fh:
        fh.writelines(out_lines)

    print(f"  [mask_anchors] Removed {n_removed} REMARK line(s) for "
          f"{len(anchor_abs_indices)} anchor position(s)")
    print(f"  [mask_anchors] Modified PDB written to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 8. Graft original T-chain sequence onto RFdiffusion outputs
# ---------------------------------------------------------------------------

def graft_target_sequence(
    rfdiffusion_pdb: str,
    original_target: str,
    out_path:        str,
    target_chain:    str = CHAIN_T,
    source_chain:    str = CHAIN_T,
) -> str:
    """
    Replace the entire set of ATOM records for the target chain in an
    RFdiffusion output PDB with the original ATOM records from the input
    structure, remapping the chain letter if needed.

    This restores both residue names AND side-chain atoms/coordinates,
    which RFdiffusion drops entirely since it only models backbone frames.
    """
    # Read all original ATOM records for source_chain, keyed by resnum
    # preserving full line content (coordinates, B-factors, etc.)
    original_records: Dict[int, List[str]] = {}
    with open(original_target) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[21] != source_chain:
                continue
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            # Remap chain letter to target_chain in the stored line
            if source_chain != target_chain:
                line = line[:21] + target_chain + line[22:]
            original_records.setdefault(resnum, []).append(
                line if line.endswith("\n") else line + "\n"
            )

    if not original_records:
        print(f"  [WARN] graft_target_sequence: no chain '{source_chain}' "
              f"residues found in {original_target} — skipping graft.")
        import shutil
        shutil.copy(rfdiffusion_pdb, out_path)
        return out_path

    # Write output, replacing target_chain ATOM records wholesale
    # with the original full-atom records
    out_lines:         List[str] = []
    inserted_resnums:  set       = set()
    n_replaced = 0

    with open(rfdiffusion_pdb) as fh:
        for line in fh:
            if ((line.startswith("ATOM") or line.startswith("HETATM"))
                    and line[21] == target_chain):
                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    out_lines.append(line)
                    continue
                # On first encounter of each resnum, emit all original
                # atoms for that residue; skip subsequent backbone-only
                # lines from RFdiffusion for the same residue
                if resnum not in inserted_resnums:
                    inserted_resnums.add(resnum)
                    orig = original_records.get(resnum)
                    if orig:
                        out_lines.extend(orig)
                        n_replaced += 1
                    else:
                        # Residue not in original (shouldn't happen) —
                        # keep RFdiffusion line as fallback
                        out_lines.append(line)
                # else: skip duplicate backbone atoms for same resnum
            else:
                out_lines.append(line)

    with open(out_path, "w") as fh:
        fh.writelines(out_lines)

    print(f"  [graft_target_sequence] Replaced {n_replaced} residue(s) "
          f"on chain {target_chain} with original full-atom records")
    print(f"  [graft_target_sequence] Written to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 9. Build RFdiffusion CLI command
# ---------------------------------------------------------------------------

def build_rfdiffusion_command(
    input_pdb:      str,    # merged HLT complex — required for partial diffusion
    target_pdb:     str,    # antigen — passed as antibody.target_pdb
    framework_pdb:  str,    # H/L scaffold — passed as antibody.framework_pdb
    contig_string:  str,
    provide_seq:    str,
    hotspots:       str,
    output_prefix:  str,
    partial_T:      int,
    num_designs:    int,
    model_weights:  str,
    extra_args:     List[str],
) -> List[str]:
    """
    Build the RFantibody rfdiffusion CLI command.

    For partial diffusion, inference.input_pdb is REQUIRED — it is the
    starting structure that noise is added to. This must be the merged HLT
    complex (H+L+T) so that RFdiffusion knows the starting coordinates for
    all chains.

    Additionally, the antibody-finetuned model uses two further arguments:
        antibody.target_pdb    — antigen chain(s), held fully fixed;
                                 provides the target template track
        antibody.framework_pdb — H/L scaffold with CDR REMARK annotations;
                                 provides the framework template track

    All three are needed simultaneously for partial diffusion with the
    antibody model. The T-chain sequence mutation issue is addressed by
    graft_target_sequence post-run, not by omitting input_pdb.
    """
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
        inference_script = os.environ.get(
            "RFANTIBODY_INFERENCE_SCRIPT",
            "/home/pymc/Deepak/RFantibody_partialflow/src/rfantibody/"
            "rfdiffusion/rfdiffusion_inference.py",
        )
        if not inference_script or not os.path.isfile(inference_script):
            raise FileNotFoundError(
                "Cannot locate rfdiffusion_inference.py. "
                "Set RFANTIBODY_INFERENCE_SCRIPT to its absolute path."
            )

    cmd = [
        sys.executable,
        inference_script,
        "--config-name", "antibody",
        f"inference.input_pdb={input_pdb}",         # required for partial diffusion
        f"antibody.target_pdb={target_pdb}",        # antigen template track
        f"antibody.framework_pdb={framework_pdb}",  # H/L template track
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={num_designs}",
    ]

    if model_weights:
        cmd.append(f"inference.ckpt_override_path={model_weights}")

    if contig_string:
        cmd.append(f"'contigmap.contigs=[{contig_string}]'")

    if provide_seq:
        cmd.append(f"'contigmap.provide_seq=[{provide_seq}]'")

    cmd.append(f"diffuser.partial_T={partial_T}")

    if hotspots:
        cmd.append(f"'ppi.hotspot_res=[{hotspots}]'")

    cmd.extend(extra_args)
    return cmd


# ---------------------------------------------------------------------------
# 10. Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    framework_pdb: str,
    target_pdb:    str,
    anchors:       List[Tuple[str, int]],
    cdr_ranges:    Dict[str, CdrRange],
    free_loops:    Dict[str, Tuple[int, int]],
    contig_string: str,
    provide_seq:   str,
    cmd:           List[str],
    partial_T:     int,
    num_designs:   int,
) -> None:
    print("\n" + "=" * 70)
    print("  Step 1 — Partial Diffusion Maturation (framework + target inputs)")
    print("=" * 70)
    print(f"  Framework PDB : {framework_pdb}")
    print(f"  Target PDB    : {target_pdb}")
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

    print()
    print(f"  Contig string : {contig_string}")

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
# 11. Extract H/L-only PDB from a merged HLT file
# ---------------------------------------------------------------------------

def _write_framework_only(merged_pdb: str, out_path: str) -> None:
    """
    Write a copy of merged_pdb to out_path containing only H and L chain
    records (ATOM/HETATM/TER) plus all REMARK lines.

    This is needed because build_rfdiffusion_command now passes the framework
    as antibody.framework_pdb, which must not contain the T chain.  The
    input to this function is the anchor-masked merged PDB so that the REMARK
    lines for anchor positions are already removed.
    """
    out_lines: List[str] = []
    with open(merged_pdb) as fh:
        for line in fh:
            record = line[:6].strip()
            if record == "REMARK":
                out_lines.append(line)
            elif record in {"HEADER", "TITLE", "CRYST1"}:
                out_lines.append(line)
            elif record in _PDB_COORD_RECORDS:
                if record == "TER":
                    out_lines.append(line)
                    continue
                chain = line[21] if len(line) > 21 else " "
                if chain in (CHAIN_H, CHAIN_L):
                    out_lines.append(line)
                # T-chain lines are dropped
            elif record in {"END", "MASTER"}:
                pass  # written once at the end

    out_lines.append("END\n")
    with open(out_path, "w") as fh:
        fh.writelines(out_lines)


# ---------------------------------------------------------------------------
# 12. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 1 (framework variant): Build partial diffusion contig "
            "from a separate HLT-annotated framework PDB (H/L chains) and "
            "a separate target PDB (T chain), then run RFantibody's "
            "rfdiffusion for in silico maturation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── NEW: separate framework and target instead of a single --input ──
    p.add_argument(
        "--framework", required=True,
        help="HLT-annotated PDB containing only H (and optionally L) chains "
             "with CDR REMARK PDBinfo-LABEL annotations.",
    )
    p.add_argument(
        "--target", required=True,
        help="PDB containing only the T (antigen) chain.",
    )
    p.add_argument("--anchors",     required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--hotspots",    required=True)
    p.add_argument("--model_weights", required=True)

    p.add_argument(
        "--target_chain", default="T",
        help="Chain ID used in the target PDB (default: T). Will be "
             "remapped to T in the merged file used for parsing and "
             "as input to RFdiffusion.",
    )
    p.add_argument("--partial_T",   type=int, default=15)
    p.add_argument("--num_designs", type=int, default=20)
    p.add_argument("--free_loops",  default="")
    p.add_argument("--nanobody",    action="store_true")
    p.add_argument("--name",        default="")
    p.add_argument("--dry_run",     action="store_true")
    p.add_argument("extra",         nargs=argparse.REMAINDER)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    framework_pdb = str(Path(args.framework).resolve())
    target_pdb    = str(Path(args.target).resolve())
    anchors_json  = str(Path(args.anchors).resolve())
    output_dir    = str(Path(args.output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    stem          = args.name or Path(framework_pdb).stem
    output_prefix = os.path.join(
        output_dir, f"{stem}_partial_T{args.partial_T}"
    )

    if not 1 <= args.partial_T <= 50:
        sys.exit(
            f"[ERROR] --partial_T must be between 1 and 50 "
            f"(got {args.partial_T})."
        )

    # ── Step A: merge framework + target into a single HLT complex ────────
    print(f"[Step 1] Merging framework ({framework_pdb}) "
          f"and target ({target_pdb}, chain '{args.target_chain}') ...")
    merged_dir = os.path.join(output_dir, "_merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged_pdb = os.path.join(merged_dir, f"{stem}_merged.pdb")
    try:
        merge_framework_and_target(
            framework_pdb,
            target_pdb,
            merged_pdb,
            target_chain_in=args.target_chain,
        )
    except ValueError as e:
        sys.exit(f"[ERROR] Failed to merge framework and target: {e}")

    # All subsequent logic operates on the merged file, exactly as in the
    # original script that received a pre-merged HLT complex via --input.
    working_pdb = merged_pdb

    # ── Step B: parse CDR REMARKs and residues ────────────────────────────
    print(f"[Step 1] Parsing HLT REMARK annotations from {working_pdb}")
    cdr_ranges = parse_hlt_remarks(working_pdb)
    if not cdr_ranges:
        sys.exit("[ERROR] No CDR REMARK lines found in merged PDB.")
    print(f"         Found CDRs: {', '.join(sorted(cdr_ranges))}")

    print("[Step 1] Reading residue list from merged PDB ...")
    residues = read_pdb_residues(working_pdb)
    n_h = sum(1 for r in residues if r.pdb_chain == CHAIN_H)
    n_l = sum(1 for r in residues if r.pdb_chain == CHAIN_L)
    n_t = sum(1 for r in residues if r.pdb_chain == CHAIN_T)
    print(f"         H:{n_h}  L:{n_l}  T:{n_t}  total:{len(residues)}")

    if args.nanobody and n_l > 0:
        print("[WARN] --nanobody flag set but L-chain residues found.")
    if not args.nanobody and n_l == 0:
        print("[INFO] No L-chain residues detected; treating as nanobody.")
        args.nanobody = True

    t_residues  = [r for r in residues if r.pdb_chain == CHAIN_T]
    ab_residues = [r for r in residues if r.pdb_chain in (CHAIN_H, CHAIN_L)]
    if not t_residues:
        sys.exit("[ERROR] No chain T (target) residues found after merge.")
    if not ab_residues:
        sys.exit("[ERROR] No H/L (antibody) residues found after merge.")

    # ── Step C: load anchors ───────────────────────────────────────────────
    print(f"[Step 1] Loading anchor residues from {anchors_json}")
    anchor_residues = load_anchors(anchors_json)

    # Save the original merged PDB path before masking reassigns working_pdb
    original_merged_pdb = working_pdb

    # ── Step D: mask anchor REMARK lines ──────────────────────────────────
    masked_pdb = os.path.join(output_dir, f"{stem}_anchors_masked.pdb")
    working_pdb = mask_anchors_in_hlt(
        input_pdb=working_pdb,
        anchor_residues=anchor_residues,
        out_path=masked_pdb,
    )
    print(f"[Step 1] Using anchor-masked PDB: {working_pdb}")
    print(f"         {len(anchor_residues)} anchor(s): "
          f"{[f'{c}{n}' for c, n in anchor_residues]}")

    # ── Step E: free-loop overrides ────────────────────────────────────────
    print(f"[Step 1] Parsing free-loop overrides: '{args.free_loops}'")
    try:
        free_loops = parse_free_loops(args.free_loops)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    # ── Step F: build contig string and provide_seq ────────────────────────
    # Use original_merged_pdb residues (not masked) so pose indices are
    # consistent with the REMARK lines.
    print("[Step 1] Building contig string ...")
    contig_string = build_contig_string(
        residues=residues,
        cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues,
        free_loop_overrides=free_loops,
        nanobody=args.nanobody,
    )
    print(f"         {contig_string}")

    print("[Step 1] Building provide_seq (fixed backbone positions) ...")
    provide_seq = build_provide_seq(
        residues=residues,
        cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues,
        nanobody=args.nanobody,
    )
    n_fixed = len(provide_seq.split(",")) if provide_seq else 0
    print(f"         {n_fixed} residue(s) marked as fixed "
          f"(framework + anchors + antigen)")

    # ── Step G: build CLI command ──────────────────────────────────────────
    # Pass the original --target and the anchor-masked --framework as
    # separate arguments. The merged PDB is only used for parsing above.
    # The masked framework is used (not the original) so that AbSampler
    # treats anchor positions as framework-fixed rather than CDR-diffusible.
    masked_framework_pdb = os.path.join(
        output_dir, f"{stem}_framework_anchors_masked.pdb"
    )
    _write_framework_only(working_pdb, masked_framework_pdb)

    extra = [a for a in args.extra if a != "--"]
    cmd = build_rfdiffusion_command(
        input_pdb=working_pdb,              # anchor-masked merged HLT (H+L+T)
        target_pdb=target_pdb,             # original antigen
        framework_pdb=masked_framework_pdb, # H/L only, anchors masked
        contig_string=contig_string,
        provide_seq=provide_seq,
        hotspots=args.hotspots,
        output_prefix=output_prefix,
        partial_T=args.partial_T,
        num_designs=args.num_designs,
        model_weights=args.model_weights,
        extra_args=extra,
    )

    print_summary(
        framework_pdb=framework_pdb,
        target_pdb=target_pdb,
        anchors=anchor_residues,
        cdr_ranges=cdr_ranges,
        free_loops=free_loops,
        contig_string=contig_string,
        provide_seq=provide_seq,
        cmd=cmd,
        partial_T=args.partial_T,
        num_designs=args.num_designs,
    )

    # Save contig record
    record_path = os.path.join(output_dir, f"{stem}_contig.json")
    with open(record_path, "w") as fh:
        json.dump(
            {
                "framework_pdb":          framework_pdb,
                "target_pdb":             target_pdb,
                "merged_pdb":             original_merged_pdb,
                "masked_merged_pdb":      working_pdb,
                "masked_framework_pdb":   masked_framework_pdb,
                "partial_T":              args.partial_T,
                "contig_string":          contig_string,
                "provide_seq":            provide_seq,
                "anchor_residues":        [f"{c}{n}" for c, n in anchor_residues],
                "free_loop_overrides":    {
                    k: list(v) for k, v in free_loops.items()
                },
                "command": cmd,
            },
            fh,
            indent=2,
        )
    print(f"[Step 1] Contig record saved to: {record_path}")

    if args.dry_run:
        print("[Step 1] DRY RUN — rfdiffusion not invoked.")
        return

    # ── Step H: run RFdiffusion ────────────────────────────────────────────
    print("[Step 1] Launching rfdiffusion ...")
    shell_cmd = " ".join(cmd)
    print(f"         $ {shell_cmd}\n")

    result = subprocess.run(shell_cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"[ERROR] rfdiffusion exited with code {result.returncode}")

    # ── Step I: graft original T-chain sequence onto outputs ──────────────
    print("\n[Step 1] Grafting original target sequence onto outputs ...")
    output_pdbs = sorted(
        Path(output_dir).glob(f"{Path(output_prefix).name}*.pdb")
    )
    if not output_pdbs:
        print("  [WARN] No output PDBs found to graft — check output_prefix.")
    for out_pdb in output_pdbs:
        graft_target_sequence(
            rfdiffusion_pdb=str(out_pdb),
            original_target=target_pdb,
            out_path=str(out_pdb),
            target_chain=CHAIN_T,           # RFdiffusion always outputs T
            source_chain=args.target_chain, # original file may use e.g. 'C'
        )

    print(f"\n[Step 1] Complete. Outputs written to: {output_dir}/")
    print(f"         Feed these PDBs into Step 2 (ProteinMPNN).")


if __name__ == "__main__":
    main()