"""
maturation_pipeline.py

Unified orchestrator for the PPIFlow-style in silico affinity maturation
workflow, integrating the logic from Steps 0, 1, and 2 into a single
script that runs after the standard RFantibody generation pipeline.

Standard RFantibody flow (unchanged):
    rfdiffusion → proteinmpnn → rf2 → [filtered .pdb files]

This script adds a maturation pass on top of those filtered designs:

    Step 0  identify_cdr_anchors      Rosetta interface decomposition
    Step 1  partial_diffusion         Rebuild free CDR regions around anchors
    Step 2  sequence_design           ProteinMPNN with anchor residues fixed

Usage
-----
    # Minimal: maturation of every PDB in a directory
    python maturation_pipeline.py \\
        --input_dir        rf2_filtered/ \\
        --output_dir       maturation/ \\
        --hotspots         "T305,T456" \\
        --model_weights    ckpt/nanobody.ckpt \\
        --nanobody

    # Full control
    python maturation_pipeline.py \\
        --input_dir        rf2_filtered/ \\
        --output_dir       maturation/ \\
        --hotspots         "T305,T456" \\
        --model_weights    ckpt/antibody.ckpt \\
        --loops            "H1,H2,H3,L1,L2,L3" \\
        --free_loops       "H3:5-13,L3:9-11" \\
        --energy_threshold -5.0 \\
        --partial_T        15 \\
        --num_diffusion    20 \\
        --num_seqs         4 \\
        --temperature      0.1 \\
        --interface        "HL_T" \\
        --relax \\
        --dry_run

    # Multi-round maturation (iterate 3 times on best designs)
    python maturation_pipeline.py \\
        --input_dir     rf2_filtered/ \\
        --output_dir    maturation/ \\
        --hotspots      "T305,T456" \\
        --model_weights ckpt/nanobody.ckpt \\
        --nanobody \\
        --rounds        3 \\
        --top_n         5

Structure of output_dir after a single round
---------------------------------------------
    maturation/
    ├── round_1/
    │   ├── step0_anchors/          # *_anchors.json, *_fixed_positions.jsonl
    │   ├── step1_diffusion/        # partial-diffusion HLT PDBs
    │   └── step2_sequences/        # ProteinMPNN output PDBs
    └── pipeline_config.json        # full run record

Dependencies
------------
    - RFantibody installed via `uv sync` (provides rfdiffusion, proteinmpnn)
    - PyRosetta (for Step 0; install separately)
    - Python >= 3.9
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Inline imports from the three step modules ────────────────────────────
# We import the step functions directly rather than shelling out, so the
# pipeline is one coherent Python process with shared data structures and
# no redundant file I/O.  The three scripts (identify_cdr_anchors.py,
# partial_diffusion_maturation.py, sequence_design_maturation.py) must be
# on sys.path, or co-located with this file.

try:
    from identify_cdr_anchors_openmm import (
        identify_anchors,
        parse_hlt_remarks as _parse_hlt_remarks_s0,
        write_outputs as _write_anchor_outputs,
        read_pdb_residue_map as _read_residue_map_s0,
    )
    from partial_diffusion_maturation import (
        build_contig_string,
        build_rfdiffusion_command,
        parse_free_loops,
        read_pdb_residues as _read_residues_s1,
        parse_hlt_remarks as _parse_hlt_remarks_s1,
    )
    from sequence_design_maturation import (
        load_anchors as _load_anchors_s2,
        process_pdb as _process_pdb_s2,
        write_jsonl,
        run_proteinmpnn,
    )
    _STEPS_AVAILABLE = True
except ImportError as _e:
    _STEPS_AVAILABLE = False
    _IMPORT_ERROR = str(_e)


# ── Pipeline-level dataclasses ────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Complete configuration record written to pipeline_config.json."""
    input_dir:        str
    output_dir:       str
    hotspots:         str
    model_weights:    str
    loops:            str
    free_loops:       str
    energy_threshold: float
    partial_T:        int
    num_diffusion:    int
    num_seqs:         int
    temperature:      float
    interface:        str
    nanobody:         bool
    rounds:           int
    top_n:            Optional[int]
    dry_run:          bool
    relax:            bool = False   # unused; kept for schema stability
    skip_minimize:    bool = False


@dataclass
class DesignRecord:
    """Tracks processing state for one input PDB through the pipeline."""
    pdb_path:      str
    stem:          str
    anchors:       List[Tuple[str, int]] = field(default_factory=list)
    anchor_json:   str = ""               # path to *_anchors.json
    fixed_jsonl:   str = ""               # path to *_fixed_positions.jsonl
    contig_str:    str = ""
    diffusion_dir: str = ""               # directory of partial-diff outputs
    sequence_dir:  str = ""               # directory of MPNN outputs
    skipped:       bool = False
    skip_reason:   str = ""


# ── Directory layout helpers ──────────────────────────────────────────────

def round_dirs(output_dir: str, round_n: int) -> Tuple[str, str, str]:
    """Return (step0_dir, step1_dir, step2_dir) for a given round number."""
    base = os.path.join(output_dir, f"round_{round_n}")
    return (
        os.path.join(base, "step0_anchors"),
        os.path.join(base, "step1_diffusion"),
        os.path.join(base, "step2_sequences"),
    )


def makedirs_for_round(output_dir: str, round_n: int) -> Tuple[str, str, str]:
    s0, s1, s2 = round_dirs(output_dir, round_n)
    for d in (s0, s1, s2):
        os.makedirs(d, exist_ok=True)
    return s0, s1, s2


# ── Step 0 integration ────────────────────────────────────────────────────

def run_step0(
    pdb_path:         str,
    step0_dir:        str,
    config:           PipelineConfig,
) -> DesignRecord:
    """
    Run Rosetta interface decomposition on one filtered design PDB.

    This calls identify_anchors() from identify_cdr_anchors.py directly
    (no subprocess), then writes the JSON and JSONL outputs to step0_dir.

    Returns a DesignRecord with anchor data populated, or with skipped=True
    if PyRosetta is unavailable or the PDB has no CDR REMARK lines.
    """
    stem = Path(pdb_path).stem
    rec  = DesignRecord(pdb_path=pdb_path, stem=stem)

    print(f"\n  [Step 0] {stem}")

    try:
        anchors = identify_anchors(
            pdb_path=pdb_path,
            output_dir=step0_dir,
            energy_threshold=config.energy_threshold,
            source_chains=config.interface.split("_")[0],  # e.g. "HL_T" -> "HL"
            target_chains=config.interface.split("_")[1],  # e.g. "HL_T" -> "T"
            interface_distance=4.0,
            skip_minimize=False,
        )
    except Exception as exc:
        rec.skipped     = True
        rec.skip_reason = f"Step 0 failed: {exc}"
        print(f"    SKIP: {exc}")
        return rec

    # Resolve the score TSV path written by identify_anchors into
    # _work_{stem}/ so write_outputs can populate the full summary table.
    work_dir  = os.path.join(step0_dir, f"_work_{stem}")
    tsv_candidates = list(Path(work_dir).glob("*_scores.tsv"))
    tsv_path  = str(tsv_candidates[0]) if tsv_candidates else None

    cdr_ranges = _parse_hlt_remarks_s0(pdb_path)
    _write_anchor_outputs(
        anchors=anchors,
        all_cdr_ranges=cdr_ranges,
        pdb_path=pdb_path,
        output_dir=step0_dir,
        energy_threshold=config.energy_threshold,
        tsv_path=tsv_path,
    )

    rec.anchors    = [(a.pdb_chain, a.pdb_resnum) for a in anchors]
    rec.anchor_json = os.path.join(step0_dir, f"{stem}_anchors.json")
    rec.fixed_jsonl = os.path.join(step0_dir, f"{stem}_fixed_positions.jsonl")

    print(f"    {len(rec.anchors)} anchor(s): "
          f"{[f'{c}{n}' for c,n in rec.anchors]}")
    return rec


# ── Step 1 integration ────────────────────────────────────────────────────

def run_step1(
    rec:      DesignRecord,
    step1_dir: str,
    config:   PipelineConfig,
) -> DesignRecord:
    """
    Build the contig string from Step 0 anchors and launch partial diffusion
    via the rfdiffusion CLI.

    The contig-building logic is imported directly from
    partial_diffusion_maturation.py; the actual rfdiffusion call is a
    subprocess so GPU memory is released between designs.
    """
    if rec.skipped:
        return rec

    print(f"\n  [Step 1] {rec.stem}")

    # Parse residue map of the input PDB
    cdr_ranges   = _parse_hlt_remarks_s1(rec.pdb_path)
    residues     = _read_residues_s1(rec.pdb_path)
    free_loops   = parse_free_loops(config.free_loops)
    anchor_set   = set(rec.anchors)

    try:
        contig_str = build_contig_string(
            residues=residues,
            cdr_ranges=cdr_ranges,
            anchor_residues=rec.anchors,
            free_loop_overrides=free_loops,
            nanobody=config.nanobody,
        )
    except Exception as exc:
        rec.skipped     = True
        rec.skip_reason = f"Contig build failed: {exc}"
        print(f"    SKIP: {exc}")
        return rec

    rec.contig_str = contig_str
    print(f"    Contig: [{contig_str}]")

    output_prefix = os.path.join(step1_dir, f"{rec.stem}_partial_T{config.partial_T}")

    cmd = build_rfdiffusion_command(
        input_pdb=rec.pdb_path,
        contig_str=contig_str,
        hotspots=config.hotspots,
        output_prefix=output_prefix,
        partial_T=config.partial_T,
        num_designs=config.num_diffusion,
        model_weights=config.model_weights,
        extra_args=[],
    )

    print(f"    $ {' '.join(cmd)}")

    if not config.dry_run:
        result = subprocess.run(" ".join(cmd), shell=True)
        if result.returncode != 0:
            rec.skipped     = True
            rec.skip_reason = f"rfdiffusion exited {result.returncode}"
            print(f"    FAIL: rfdiffusion returned {result.returncode}")
            return rec

    rec.diffusion_dir = step1_dir

    # Save the contig record alongside the outputs
    contig_record = {
        "source_pdb":     rec.pdb_path,
        "partial_T":      config.partial_T,
        "contig_string":  contig_str,
        "anchor_residues": [f"{c}{n}" for c, n in rec.anchors],
        "command":        cmd,
    }
    with open(os.path.join(step1_dir, f"{rec.stem}_contig.json"), "w") as fh:
        json.dump(contig_record, fh, indent=2)

    return rec


# ── Step 2 integration ────────────────────────────────────────────────────

def run_step2(
    rec:       DesignRecord,
    step2_dir: str,
    config:    PipelineConfig,
) -> DesignRecord:
    """
    For every PDB produced by Step 1, compute per-PDB chain_id and
    fixed_positions JSONL dicts (using the anchor data from Step 0), then
    call the proteinmpnn CLI once with all PDBs batched.

    The JSONL-building logic is imported from sequence_design_maturation.py.
    The actual CLI call is a subprocess.
    """
    if rec.skipped:
        return rec

    print(f"\n  [Step 2] {rec.stem}")

    step1_pdbs = sorted(Path(rec.diffusion_dir).glob(
        f"{rec.stem}_partial_T{config.partial_T}_*.pdb"
    ))

    if not step1_pdbs:
        if config.dry_run:
            # Dry run: no PDBs exist yet; synthesise a placeholder count
            print(f"    (dry run — would process "
                  f"{config.num_diffusion} PDBs from Step 1)")
            rec.sequence_dir = step2_dir
            return rec
        rec.skipped     = True
        rec.skip_reason = "No Step 1 PDBs found"
        print(f"    SKIP: no PDBs in {rec.diffusion_dir}")
        return rec

    print(f"    {len(step1_pdbs)} PDB(s) to sequence-design")

    loops_list = [l.strip() for l in config.loops.split(",") if l.strip()]
    anchor_set: Set[Tuple[str, int]] = set(rec.anchors)

    all_chain_id:  List[Dict] = []
    all_fixed_pos: List[Dict] = []

    for pdb_path in step1_pdbs:
        try:
            pdb_name, chain_id_rec, fixed_pos_rec, summary = _process_pdb_s2(
                pdb_path=str(pdb_path),
                anchor_set=anchor_set,
                loops_to_design=loops_list,
                scratch_dir=step2_dir,
            )
        except Exception as exc:
            print(f"    WARN: skipping {pdb_path.name}: {exc}")
            continue

        all_chain_id.append(chain_id_rec)
        all_fixed_pos.append(fixed_pos_rec)

    if not all_chain_id:
        rec.skipped     = True
        rec.skip_reason = "All Step 1 PDBs failed JSONL build"
        return rec

    # Write JSONL files to a per-design temp location within step2_dir
    scratch = os.path.join(step2_dir, f"jsonl_{rec.stem}")
    os.makedirs(scratch, exist_ok=True)
    chain_id_path  = os.path.join(scratch, "chain_ids.jsonl")
    fixed_pos_path = os.path.join(scratch, "fixed_positions.jsonl")
    write_jsonl(all_chain_id,  chain_id_path)
    write_jsonl(all_fixed_pos, fixed_pos_path)

    # Run ProteinMPNN on the Step 1 PDB directory
    run_proteinmpnn(
        input_dir=rec.diffusion_dir,
        output_dir=step2_dir,
        chain_id_jsonl=chain_id_path,
        fixed_pos_jsonl=fixed_pos_path,
        loops=config.loops,
        num_seqs=config.num_seqs,
        temperature=config.temperature,
        extra_args=[],
        dry_run=config.dry_run,
    )

    rec.sequence_dir = step2_dir
    return rec


# ── Multi-round logic ─────────────────────────────────────────────────────

def collect_input_pdbs(input_dir: str) -> List[str]:
    """Return sorted list of .pdb paths in input_dir."""
    pdbs = sorted(str(p) for p in Path(input_dir).glob("*.pdb"))
    if not pdbs:
        sys.exit(f"[ERROR] No .pdb files found in {input_dir}")
    return pdbs


def select_top_n_from_step2(
    step2_dir: str,
    top_n: Optional[int],
) -> List[str]:
    """
    Return the top N PDBs from step2_dir to use as input for the next round.

    Selection is based on ProteinMPNN score (MPNN log-likelihood) embedded
    in the FASTA headers written by proteinmpnn, or falls back to
    alphabetical order if scores are not parseable.

    MPNN writes FASTA headers like:
        >design_0042_0, score=1.234, fixed_chains=[T], designed_chains=[H,L]
    Lower score = better (score = -mean log-probability).
    """
    pdb_paths = sorted(Path(step2_dir).glob("*.pdb"))
    if not pdb_paths:
        return []

    # Try to read MPNN scores from companion .fa / seqs/ directory
    seqs_dir = os.path.join(step2_dir, "seqs")
    score_map: Dict[str, float] = {}

    if os.path.isdir(seqs_dir):
        for fa in Path(seqs_dir).glob("*.fa"):
            with open(fa) as fh:
                for line in fh:
                    if not line.startswith(">"):
                        continue
                    m = re.search(r"score=([\d.]+)", line)
                    if m:
                        # Strip sample suffix to get base PDB name
                        base = fa.stem
                        score = float(m.group(1))
                        # Keep the minimum (best) score per base name
                        if base not in score_map or score < score_map[base]:
                            score_map[base] = score

    # Sort PDBs by MPNN score (ascending = better), fall back to name
    def sort_key(p: Path) -> Tuple[float, str]:
        return (score_map.get(p.stem, 999.0), p.name)

    ranked = sorted(pdb_paths, key=sort_key)
    selected = ranked[:top_n] if top_n else ranked
    return [str(p) for p in selected]


def copy_pdbs_to_round_input(
    pdb_paths: List[str],
    dest_dir:  str,
) -> List[str]:
    """Copy a list of PDBs into dest_dir and return the new paths."""
    os.makedirs(dest_dir, exist_ok=True)
    new_paths = []
    for src in pdb_paths:
        dst = os.path.join(dest_dir, Path(src).name)
        shutil.copy2(src, dst)
        new_paths.append(dst)
    return new_paths


# ── Main orchestrator ─────────────────────────────────────────────────────

def run_pipeline(config: PipelineConfig) -> None:
    """
    Run the full maturation pipeline for the requested number of rounds.

    Round N uses the Step 2 outputs of Round N-1 as inputs (or the original
    input_dir for Round 1). Within each round, every input PDB passes
    through Steps 0 → 1 → 2 sequentially.
    """
    if not _STEPS_AVAILABLE:
        sys.exit(
            f"[ERROR] Could not import step modules: {_IMPORT_ERROR}\n"
            "Ensure identify_cdr_anchors.py, partial_diffusion_maturation.py,\n"
            "and sequence_design_maturation.py are on sys.path or co-located\n"
            "with this script."
        )

    os.makedirs(config.output_dir, exist_ok=True)

    # Write pipeline config for reproducibility
    config_path = os.path.join(config.output_dir, "pipeline_config.json")
    with open(config_path, "w") as fh:
        json.dump(asdict(config), fh, indent=2)
    print(f"[Pipeline] Config saved to {config_path}")

    current_input_dir = config.input_dir

    for round_n in range(1, config.rounds + 1):
        print(f"\n{'='*60}")
        print(f"  MATURATION ROUND {round_n} of {config.rounds}")
        print(f"{'='*60}")

        s0_dir, s1_dir, s2_dir = makedirs_for_round(config.output_dir, round_n)

        input_pdbs = collect_input_pdbs(current_input_dir)
        print(f"[Round {round_n}] {len(input_pdbs)} input PDB(s)")

        records: List[DesignRecord] = []

        for pdb_path in input_pdbs:
            print(f"\n--- Design: {Path(pdb_path).stem} ---")

            rec = run_step0(pdb_path, s0_dir, config)
            rec = run_step1(rec, s1_dir, config)
            rec = run_step2(rec, s2_dir, config)
            records.append(rec)

        # Summary for this round
        n_ok   = sum(1 for r in records if not r.skipped)
        n_skip = sum(1 for r in records if r.skipped)
        print(f"\n[Round {round_n}] Completed: {n_ok} ok, {n_skip} skipped")

        for r in records:
            if r.skipped:
                print(f"  SKIPPED {r.stem}: {r.skip_reason}")

        # Write round summary
        summary_path = os.path.join(
            config.output_dir, f"round_{round_n}", "round_summary.json"
        )
        with open(summary_path, "w") as fh:
            json.dump(
                [
                    {
                        "stem":          r.stem,
                        "anchors":       [f"{c}{n}" for c, n in r.anchors],
                        "contig":        r.contig_str,
                        "skipped":       r.skipped,
                        "skip_reason":   r.skip_reason,
                        "diffusion_dir": r.diffusion_dir,
                        "sequence_dir":  r.sequence_dir,
                    }
                    for r in records
                ],
                fh, indent=2,
            )

        # Prepare input for the next round
        if round_n < config.rounds:
            top_pdbs = select_top_n_from_step2(s2_dir, config.top_n)
            if not top_pdbs:
                print(f"[Round {round_n}] No outputs for next round; stopping.")
                break
            next_input = os.path.join(
                config.output_dir, f"round_{round_n}_to_{round_n+1}_inputs"
            )
            copy_pdbs_to_round_input(top_pdbs, next_input)
            current_input_dir = next_input
            print(f"[Round {round_n}] {len(top_pdbs)} PDB(s) "
                  f"selected for round {round_n+1}")

    print(f"\n[Pipeline] All rounds complete. Outputs in: {config.output_dir}/")
    print(f"[Pipeline] Final designs: "
          f"{os.path.join(config.output_dir, f'round_{config.rounds}', 'step2_sequences')}/")


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Maturation pipeline: Steps 0–2 applied to RFantibody designs.\n"
            "Runs Rosetta anchor identification, partial diffusion, and "
            "anchor-constrained ProteinMPNN in one command."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────
    g = p.add_argument_group("required")
    g.add_argument("--input_dir",    required=True,
                   help="Directory of RF2-filtered HLT PDBs to mature")
    g.add_argument("--output_dir",   required=True,
                   help="Root output directory (will be created)")
    g.add_argument("--hotspots",     required=True,
                   help="Antigen hotspot residues, e.g. 'T305,T456'")
    g.add_argument("--model_weights", required=True,
                   help="Path to RFantibody model checkpoint "
                        "(antibody.ckpt or nanobody.ckpt)")

    # ── Step 0 ────────────────────────────────────────────────────────────
    g0 = p.add_argument_group("Step 0 — anchor identification")
    g0.add_argument("--energy_threshold", type=float, default=-5.0,
                    help="Per-residue interface energy cutoff in REU "
                         "(default: -5.0)")
    g0.add_argument("--interface", default="HL_T",
                    help="Interface string in 'source_T' format. "
                         "'HL_T' for scFv, 'H_T' for nanobody (default: HL_T). "
                         "The part before '_' is passed as --source_chains "
                         "and the part after as --target_chains to the OpenMM scorer.")
    g0.add_argument("--skip_minimize", action="store_true",
                    help="Skip OpenMM energy minimization in Step 0 "
                         "(faster but less accurate)")

    # ── Step 1 ────────────────────────────────────────────────────────────
    g1 = p.add_argument_group("Step 1 — partial diffusion")
    g1.add_argument("--partial_T", type=int, default=15,
                    help="Diffusion noise depth 1–50 (default: 15). "
                         "10–15 = gentle refinement, 20–25 = moderate redesign")
    g1.add_argument("--num_diffusion", type=int, default=20,
                    help="Designs to generate per input PDB (default: 20)")
    g1.add_argument("--free_loops", default="",
                    help="CDR length overrides for non-anchor positions, "
                         "e.g. 'H3:5-13,L3:9-11'")
    g1.add_argument("--nanobody", action="store_true",
                    help="Input is nanobody (H-chain only)")

    # ── Step 2 ────────────────────────────────────────────────────────────
    g2 = p.add_argument_group("Step 2 — sequence design")
    g2.add_argument("--loops", default="H1,H2,H3",
                    help="CDR loops to redesign, e.g. 'H1,H2,H3' or "
                         "'H1,H2,H3,L1,L2,L3' (default: H1,H2,H3)")
    g2.add_argument("--num_seqs", type=int, default=4,
                    help="Sequences per backbone (default: 4)")
    g2.add_argument("--temperature", type=float, default=0.1,
                    help="ProteinMPNN sampling temperature (default: 0.1)")

    # ── Pipeline control ──────────────────────────────────────────────────
    gp = p.add_argument_group("pipeline control")
    gp.add_argument("--rounds", type=int, default=1,
                    help="Number of maturation iterations (default: 1)")
    gp.add_argument("--top_n", type=int, default=None,
                    help="For multi-round runs: best N designs to carry "
                         "forward per round (default: all)")
    gp.add_argument("--dry_run", action="store_true",
                    help="Print all commands and skip all GPU calls")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig(
        input_dir        = str(Path(args.input_dir).resolve()),
        output_dir       = str(Path(args.output_dir).resolve()),
        hotspots         = args.hotspots,
        model_weights    = str(Path(args.model_weights).resolve()),
        loops            = args.loops,
        free_loops       = args.free_loops,
        energy_threshold = args.energy_threshold,
        partial_T        = args.partial_T,
        num_diffusion    = args.num_diffusion,
        num_seqs         = args.num_seqs,
        temperature      = args.temperature,
        interface        = args.interface,
        nanobody         = args.nanobody,
        relax            = False,
        skip_minimize    = args.skip_minimize,
        rounds           = args.rounds,
        top_n            = args.top_n,
        dry_run          = args.dry_run,
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()