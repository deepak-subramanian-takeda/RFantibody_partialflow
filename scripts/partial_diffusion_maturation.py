"""

Reads the anchor JSON produced by Step 0, parses the HLT-format complex PDB
to reconstruct the full residue map, builds the contigmap string that fixes
anchor residues as motif segments and leaves free CDR residues as diffusible,
then shells out to RFantibody's `rfdiffusion` CLI with diffuser.partial_T set.

The resulting outputs are HLT-annotated PDB files that can be fed directly
into ProteinMPNN.

Usage
-----
    python partial_diffusion_maturation.py \\
        --input       design_0042.pdb         \\
        --anchors     anchors/design_0042_anchors.json  \\
        --output_dir  maturation/step1/        \\
        --partial_T   15                       \\
        --num_designs 20                       \\
        --hotspots    "T305,T456"              \\
        --free_loops  "H3:5-13"               \\
        --model_weights path/to/antibody.ckpt  \\
        --dry_run

    # Nanobody (no L-chain):
    python partial_diffusion_maturation.py \\
        --input       nb_design_0001.pdb      \\
        --anchors     anchors/nb_design_0001_anchors.json \\
        --output_dir  maturation/step1/        \\
        --partial_T   15                       \\
        --num_designs 20                       \\
        --hotspots    "T101,T135"              \\
        --free_loops  "H3:9-21"               \\
        --nanobody                             \\
        --model_weights path/to/nanobody.ckpt

Dependencies
------------
    - Python >= 3.9 (no GPU required for the builder; GPU required for diffusion)
    - RFantibody installed via `uv sync` (provides the `rfdiffusion` CLI)
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
# Data structures (mirrors Step 0 types without importing it)
# ---------------------------------------------------------------------------

CDR_NAMES_H = ["H1", "H2", "H3"]
CDR_NAMES_L = ["L1", "L2", "L3"]
CDR_NAMES_ALL = CDR_NAMES_H + CDR_NAMES_L

# RFantibody chains
CHAIN_H = "H"
CHAIN_L = "L"
CHAIN_T = "T"


@dataclass
class CdrRange:
    """Absolute 1-indexed residue range for one CDR from HLT REMARK lines."""
    name: str
    chain: str          # 'H' or 'L'
    start: int          # 1-indexed pose index (absolute across all chains)
    end: int
    pdb_resnums: List[int] = field(default_factory=list)  # per-residue PDB nums


@dataclass
class ResidueInfo:
    """Minimal info per residue needed for contig building."""
    pose_idx: int     # 1-indexed Rosetta-style absolute index
    pdb_chain: str    # 'H', 'L', or 'T'
    pdb_resnum: int   # PDB residue number


# ---------------------------------------------------------------------------
# 1. HLT parsing (duplicated lightly from Step 0 so this script is standalone)
# ---------------------------------------------------------------------------

def parse_hlt_remarks(pdb_path: str) -> Dict[str, CdrRange]:
    """
    Parse REMARK PDBinfo-LABEL lines from an HLT PDB file.

    Returns a dict mapping CDR name -> CdrRange with absolute pose indices
    and the PDB residue numbers for each member residue.
    """
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
            chain = name[0]   # 'H' or 'L'
            ranges[name] = CdrRange(
                name=name,
                chain=chain,
                start=min(positions),
                end=max(positions),
            )

    return ranges


def read_pdb_residues(pdb_path: str) -> List[ResidueInfo]:
    """
    Read ATOM/HETATM lines and return one ResidueInfo per unique
    (chain, resnum) pair, in file order.

    Only returns residues on chains H, L, T (ignores anything else).
    """
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
    """Return a fast (chain, resnum) -> ResidueInfo lookup dict."""
    return {(r.pdb_chain, r.pdb_resnum): r for r in residues}


# ---------------------------------------------------------------------------
# 2. Contig string builder
# ---------------------------------------------------------------------------

def build_contig_string(
    residues: List[ResidueInfo],
    cdr_ranges: Dict[str, CdrRange],
    anchor_residues: List[Tuple[str, int]],   # (chain, resnum) pairs
    free_loop_overrides: Dict[str, Tuple[int, int]],  # cdr_name -> (min_len, max_len)
    nanobody: bool = False,
) -> str:
    """
    Construct the contigmap.contigs string for partial diffusion.

    Design principles
    -----------------
    The contig string encodes three classes of residues:

    1. TARGET (chain T) — always fully fixed as a motif:
           e.g.  T1-150

    2. FRAMEWORK (H/L, not in any CDR) — fully fixed as motif:
           e.g.  H1-25  (N-terminal framework segment before H1)

    3. CDR residues — two sub-cases:
       a. ANCHOR: one or more anchor residues inside a loop are specified
          as fixed motif sub-segments (their chain/resnum from the input PDB).
          The non-anchor residues around them in the same loop are replaced
          with a free-length range drawn from free_loop_overrides or the
          original loop length ±2.
       b. NO ANCHOR: the whole loop is replaced with its free-length range.

    Chain breaks between H and L (or H and T, L and T) are represented as /0.

    The antibody-finetuned RFdiffusion model expects:
        [<H-chain-segments> /0 <L-chain-segments> /0 <T-chain-segments>]
    For nanobodies (no L chain):
        [<H-chain-segments> /0 <T-chain-segments>]

    Important constraint: partial diffusion with fixed motif segments and
    hotspot_res can coexist; the incompatible combination is provide_seq +
    hotspot_res.  We therefore use motif-fixation rather than provide_seq
    to preserve anchor backbone positions.

    Parameters
    ----------
    residues
        Ordered list of all residues in the pose (H then L then T).
    cdr_ranges
        CDR name -> CdrRange mapping from REMARK lines.
    anchor_residues
        List of (chain, resnum) tuples for the anchor positions to fix.
    free_loop_overrides
        CDR name -> (min_len, max_len) from --free_loops argument.
        If a CDR has no override, its length defaults to [orig-2, orig+2].
    nanobody
        If True, skip L-chain segments.

    Returns
    -------
    str  e.g. "H1-25/H27-31/2-4/H35-40/0 T1-150"
    """
    anchor_set = set(anchor_residues)   # fast O(1) lookup

    # Build a set of pose indices that belong to a CDR
    cdr_pose_idx_to_name: Dict[int, str] = {}
    for name, r in cdr_ranges.items():
        for idx in range(r.start, r.end + 1):
            cdr_pose_idx_to_name[idx] = name

    # Separate residues by chain
    h_residues = [r for r in residues if r.pdb_chain == CHAIN_H]
    l_residues = [r for r in residues if r.pdb_chain == CHAIN_L]
    t_residues = [r for r in residues if r.pdb_chain == CHAIN_T]

    def chain_to_segments(chain_residues: List[ResidueInfo]) -> str:
        """
        Walk residues on one chain and emit contig tokens.

        Consecutive non-CDR residues are collapsed into range tokens (e.g. H1-25).
        CDR residues are split into fixed (anchor) sub-segments and free-length
        gaps between them.
        """
        tokens: List[str] = []
        i = 0
        while i < len(chain_residues):
            r = chain_residues[i]
            cdr_name = cdr_pose_idx_to_name.get(r.pose_idx)

            if cdr_name is None:
                # ── Framework residue: start a run of consecutive fixed residues
                run_start = r
                while (i < len(chain_residues) and
                       cdr_pose_idx_to_name.get(chain_residues[i].pose_idx) is None):
                    i += 1
                run_end = chain_residues[i - 1]
                tokens.append(f"{run_start.pdb_chain}{run_start.pdb_resnum}"
                               f"-{run_end.pdb_resnum}")
            else:
                # ── CDR residue: collect the whole CDR loop
                cdr_r = cdr_ranges[cdr_name]
                cdr_residues = [chain_residues[j]
                                 for j in range(i, len(chain_residues))
                                 if chain_residues[j].pose_idx <= cdr_r.end]
                # advance i past the CDR
                i += len(cdr_residues)

                # Anchors in this loop?
                loop_anchors = [(r2.pdb_chain, r2.pdb_resnum)
                                for r2 in cdr_residues
                                if (r2.pdb_chain, r2.pdb_resnum) in anchor_set]

                if not loop_anchors:
                    # No anchors — emit a free-length range for the whole loop
                    orig_len = len(cdr_residues)
                    min_l, max_l = free_loop_overrides.get(
                        cdr_name, (max(1, orig_len - 2), orig_len + 2)
                    )
                    tokens.append(f"{min_l}-{max_l}")
                else:
                    # Anchors present — interleave fixed sub-segments with
                    # free-length tokens for the gaps between them.
                    tokens.extend(_cdr_with_anchors(
                        cdr_residues, loop_anchors, cdr_name,
                        free_loop_overrides, anchor_set
                    ))

        return "/".join(tokens)

    # Build per-chain segment strings
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
    """
    Return a list of contig tokens representing one CDR loop that contains
    one or more anchor residues.

    Strategy
    --------
    Walk the CDR from N- to C-terminus.  Consecutive anchor residues are
    collapsed into a single fixed motif token (e.g. H97-101).  Gaps between
    fixed segments (or at either end of the loop) are emitted as integer
    free-length ranges.

    Example (H3 loop residues H96-H111, anchors at H99-H101 and H106-H107):
        gap  H96-H98   -> "3-3"   (3 free residues before first anchor)
        fix  H99-H101  -> "H99-101"
        gap  H102-H105 -> "4-4"   (4 free residues between fixed segments)
        fix  H106-H107 -> "H106-107"
        gap  H108-H111 -> "4-4"   (4 free residues after last anchor)

    For gap regions we use a fixed range equal to the actual number of
    residues (±0).  The caller can relax this by expanding free_loop_overrides
    at a per-CDR level, but keeping the gaps exact maintains the total loop
    length constraint and prevents dramatic length changes.
    """
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

    # Flush any trailing state
    if in_anchor_run:
        flush_anchor_run()
    flush_gap(gap_count)

    return tokens


# ---------------------------------------------------------------------------
# 3. Parse free-loop overrides from CLI string
# ---------------------------------------------------------------------------

def parse_free_loops(spec: str) -> Dict[str, Tuple[int, int]]:
    """
    Parse a loop-length specification like "H3:5-13,H1:7-7,L3:9-11".
    Returns a dict mapping CDR name -> (min_len, max_len).
    """
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
    """
    Load anchor residue list from the Step 0 JSON file.
    Returns a list of (pdb_chain, pdb_resnum) tuples.
    """
    with open(json_path) as fh:
        data = json.load(fh)

    anchors: List[Tuple[str, int]] = []
    for entry in data.get("all_anchor_residues", []):
        # Format is "H97", "T305", etc.
        m = re.match(r"^([HLT])(\d+)$", entry)
        if m:
            anchors.append((m.group(1), int(m.group(2))))
        else:
            print(f"[WARN] Cannot parse anchor residue '{entry}', skipping.")
    return anchors


# ---------------------------------------------------------------------------
# 5. Build and emit the rfdiffusion CLI command
# ---------------------------------------------------------------------------

def build_rfdiffusion_command(
    input_pdb: str,
    contig_str: str,
    hotspots: str,
    output_prefix: str,
    partial_T: int,
    num_designs: int,
    model_weights: str,
    extra_args: List[str],
) -> List[str]:
    """
    Build the list of shell tokens for the `rfdiffusion` CLI call.

    RFantibody (new CLI, uv-installed) exposes `rfdiffusion` as a console
    script that wraps run_inference.py via Hydra.  We pass Hydra-style
    dot-notation overrides as extra positional arguments.

    Key design decisions
    --------------------
    - `inference.input_pdb`     : the existing HLT-annotated complex
    - `contigmap.contigs`       : our synthesised anchor+free contig string
    - `diffuser.partial_T`      : the noise depth (10–25 for refinement)
    - `ppi.hotspot_res`         : antigen hotspots carried over from Step 0
    - `antibody.design_loops`   : intentionally NOT set here — the contig
                                  string already encodes all loop lengths;
                                  passing design_loops on top would conflict.
    - `inference.ckpt_override_path` : point at the antibody/nanobody weights

    NOTE: `ppi.hotspot_res` and `contigmap.provide_seq` are mutually
    exclusive in RFdiffusion.  We use motif-fixation (chain+resnum anchors
    in the contig) rather than provide_seq, so hotspot_res is safe to use.
    """
    cmd = ["rfdiffusion"]

    # Core inference settings
    cmd += [
        f"inference.input_pdb={input_pdb}",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={num_designs}",
    ]

    # Model weights override (antibody vs nanobody checkpoint)
    if model_weights:
        cmd.append(f"inference.ckpt_override_path={model_weights}")

    # Contig map — must be quoted as a list literal for Hydra
    # The contig string itself must not have unescaped brackets.
    cmd.append(f"'contigmap.contigs=[{contig_str}]'")

    # Partial diffusion depth
    cmd.append(f"diffuser.partial_T={partial_T}")

    # Hotspot residues on the antigen
    if hotspots:
        # hotspots expected as "T305,T456" — wrap in Hydra list syntax
        cmd.append(f"'ppi.hotspot_res=[{hotspots}]'")

    # Any extra passthrough args (e.g. potentials, denoiser settings)
    cmd.extend(extra_args)

    return cmd


# ---------------------------------------------------------------------------
# 6. Summary / dry-run printer
# ---------------------------------------------------------------------------

def print_summary(
    input_pdb: str,
    anchors: List[Tuple[str, int]],
    cdr_ranges: Dict[str, CdrRange],
    free_loops: Dict[str, Tuple[int, int]],
    contig_str: str,
    cmd: List[str],
    partial_T: int,
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

    print()
    print(f"  Generated contig string:")
    print(f"    [{contig_str}]")
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

    # Required
    p.add_argument("--input", required=True,
                   help="HLT-annotated antibody-antigen complex PDB "
                        "(output of the original RFantibody design pipeline)")
    p.add_argument("--anchors", required=True,
                   help="Path to *_anchors.json from Step 0")
    p.add_argument("--output_dir", required=True,
                   help="Directory for rfdiffusion output PDBs")
    p.add_argument("--hotspots", required=True,
                   help="Antigen hotspot residues, e.g. 'T305,T456'")
    p.add_argument("--model_weights", required=True,
                   help="Path to RFantibody model checkpoint "
                        "(antibody.ckpt or nanobody.ckpt)")

    # Partial diffusion settings
    p.add_argument("--partial_T", type=int, default=15,
                   help="Diffusion noise depth (1–50). "
                        "10–15: gentle refinement. "
                        "20–25: moderate redesign. "
                        "(default: 15)")
    p.add_argument("--num_designs", type=int, default=20,
                   help="Number of designs to generate (default: 20)")

    # Loop length control
    p.add_argument("--free_loops", default="",
                   help="Comma-separated CDR length overrides for non-anchor "
                        "positions, e.g. 'H3:5-13,L3:9-11'. "
                        "CDRs not listed default to [orig_len-2, orig_len+2].")

    # Antibody type
    p.add_argument("--nanobody", action="store_true",
                   help="Input is a nanobody (H-chain only, no L-chain)")

    # Output prefix stem
    p.add_argument("--name", default="",
                   help="Optional name tag appended to output_prefix "
                        "(default: stem of input filename)")

    # Dry-run / execution
    p.add_argument("--dry_run", action="store_true",
                   help="Print the command and contig string but do not run")

    # Passthrough to rfdiffusion
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra Hydra-style overrides passed verbatim to "
                        "rfdiffusion, e.g. "
                        "'potentials.guiding_potentials=interface_ncontacts'")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────
    input_pdb = str(Path(args.input).resolve())
    anchors_json = str(Path(args.anchors).resolve())
    output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    stem = args.name or Path(input_pdb).stem
    output_prefix = os.path.join(output_dir, f"{stem}_partial_T{args.partial_T}")

    # ── Validate partial_T ─────────────────────────────────────────────────
    if not 1 <= args.partial_T <= 50:
        sys.exit(f"[ERROR] --partial_T must be between 1 and 50 "
                 f"(got {args.partial_T}). "
                 "RFantibody uses T=50 total steps.")

    # ── Parse inputs ───────────────────────────────────────────────────────
    print(f"[Step 1] Parsing HLT REMARK annotations from {input_pdb}")
    cdr_ranges = parse_hlt_remarks(input_pdb)
    if not cdr_ranges:
        sys.exit("[ERROR] No CDR REMARK lines found. "
                 "Is this a valid HLT-annotated PDB?")
    print(f"         Found CDRs: {', '.join(sorted(cdr_ranges))}")

    print(f"[Step 1] Reading residue list from PDB...")
    residues = read_pdb_residues(input_pdb)
    n_h = sum(1 for r in residues if r.pdb_chain == CHAIN_H)
    n_l = sum(1 for r in residues if r.pdb_chain == CHAIN_L)
    n_t = sum(1 for r in residues if r.pdb_chain == CHAIN_T)
    print(f"         H:{n_h}  L:{n_l}  T:{n_t}  total:{len(residues)}")

    if args.nanobody and n_l > 0:
        print("[WARN] --nanobody flag set but L-chain residues found. "
              "L-chain will be excluded from contig.")
    if not args.nanobody and n_l == 0:
        print("[INFO] No L-chain residues detected; treating as nanobody.")
        args.nanobody = True

    print(f"[Step 1] Loading anchor residues from {anchors_json}")
    anchor_residues = load_anchors(anchors_json)
    print(f"         {len(anchor_residues)} anchor(s) loaded: "
          f"{[f'{c}{n}' for c,n in anchor_residues]}")

    print(f"[Step 1] Parsing free-loop overrides: '{args.free_loops}'")
    try:
        free_loops = parse_free_loops(args.free_loops)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")
    if free_loops:
        for name, (lo, hi) in sorted(free_loops.items()):
            print(f"         {name}: {lo}–{hi}")
    else:
        print("         (none set — will use orig_len ±2 for free positions)")

    # ── Build contig string ────────────────────────────────────────────────
    print("[Step 1] Building contig string...")
    contig_str = build_contig_string(
        residues=residues,
        cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues,
        free_loop_overrides=free_loops,
        nanobody=args.nanobody,
    )
    print(f"         Contig: [{contig_str}]")

    # ── Build CLI command ──────────────────────────────────────────────────
    # Strip leading '--' that argparse may prepend to REMAINDER
    extra = [a for a in args.extra if a != "--"]

    cmd = build_rfdiffusion_command(
        input_pdb=input_pdb,
        contig_str=contig_str,
        hotspots=args.hotspots,
        output_prefix=output_prefix,
        partial_T=args.partial_T,
        num_designs=args.num_designs,
        model_weights=args.model_weights,
        extra_args=extra,
    )

    # ── Print summary ──────────────────────────────────────────────────────
    print_summary(
        input_pdb=input_pdb,
        anchors=anchor_residues,
        cdr_ranges=cdr_ranges,
        free_loops=free_loops,
        contig_str=contig_str,
        cmd=cmd,
        partial_T=args.partial_T,
        num_designs=args.num_designs,
    )

    # ── Write contig record ────────────────────────────────────────────────
    record_path = os.path.join(output_dir, f"{stem}_contig.json")
    with open(record_path, "w") as fh:
        json.dump({
            "input_pdb": input_pdb,
            "partial_T": args.partial_T,
            "contig_string": contig_str,
            "anchor_residues": [f"{c}{n}" for c, n in anchor_residues],
            "free_loop_overrides": {k: list(v) for k, v in free_loops.items()},
            "command": cmd,
        }, fh, indent=2)
    print(f"[Step 1] Contig record saved to: {record_path}")

    # ── Execute or dry-run ─────────────────────────────────────────────────
    if args.dry_run:
        print("[Step 1] DRY RUN — rfdiffusion not invoked.")
        print("         To run for real, drop the --dry_run flag.")
        return

    print("[Step 1] Launching rfdiffusion...")
    # Join into a single shell string so that the quoted Hydra overrides
    # (e.g. 'contigmap.contigs=[...]') are passed correctly.
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