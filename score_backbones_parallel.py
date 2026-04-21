"""
score_backbones_parallel.py

Multi-GPU parallel version of score_backbones.py.

Backbone PDBs are split evenly across the available GPUs.  Each GPU runs
ProteinMPNN sequence generation on its shard simultaneously.  After all
generation workers finish (or the GPU budget expires), all generated
structures are evaluated for ipTM (ColabFold) and DockQ in the main process.
Results are written sorted by ipTM descending.

GPU assignment:
    --gpu_ids 0,1,2,3    (all four GPUs generate in parallel)

GPU budget (--max_gpu_hours):
    Applies to sequence generation only.  When the budget expires, workers
    are terminated and every structure produced so far is evaluated.

Usage:
    python score_backbones_parallel.py \
        --input_dir      /path/to/backbones/ \
        --native         /path/to/native.pdb \
        --output_dir     /path/to/output/ \
        --n_seqs         10 \
        --mpnn_weights   /path/to/igdesign_acvr2b_holdout.ckpt \
        --colabfold_batch_bin /path/to/colabfold_batch \
        --colabfold_python   /path/to/colabfold_python \
        --dockq_bin      /path/to/DockQ \
        --gpu_ids        0,1,2,3 \
        --max_gpu_hours  4.0
"""

from __future__ import annotations

import sys
import os

# ── Ensure ThermoMPNN's protein_mpnn_utils is found before the installed
#    RFantibody package version — required for multiprocessing.spawn.
def _prepend_thermompnn_path():
    _here   = os.path.dirname(os.path.abspath(__file__))
    _thermo = os.path.join(_here, "ThermoMPNN")
    if os.path.isdir(_thermo) and _thermo not in sys.path:
        sys.path.insert(0, _thermo)

_prepend_thermompnn_path()

import argparse
import multiprocessing as mp
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from rfantibody_benchmark import DesignResult, GPUTimer, score_design
from partial_diffusion_maturation import (
    parse_hlt_remarks, read_pdb_residues,
    CHAIN_H, CHAIN_L, CHAIN_T,
)
from smc_denovo_maturation import build_cdr_mask, load_proteinmpnn
from beam_denovo_maturation_complexa import _apply_sequence_and_anchors


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeneratedDesign:
    backbone:    str
    seq_idx:     int
    design_id:   str
    pdb_path:    str
    gpu_seconds: float


@dataclass
class BackboneResult:
    backbone:  str
    seq_idx:   int
    design_id: str
    pdb_path:  str
    iptm:      Optional[float]
    dockq:     Optional[float]
    success:   bool = False

    def __post_init__(self):
        self.success = (
            self.iptm  is not None and self.iptm  > 0.6 and
            self.dockq is not None and self.dockq > 0.23
        )


# ─────────────────────────────────────────────────────────────────────────────
# PDB discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_backbone_pdbs(input_dir: str) -> List[Path]:
    """
    Return all backbone .pdb files in input_dir, excluding sequence designs,
    grafted structures, and RFdiffusion trajectory outputs.

    Special rule for arm_C folders:
        Beam search checkpoints are named cp<NN>_*.pdb.  For any folder
        whose path contains 'arm_C', only PDBs from the highest-numbered
        checkpoint are kept, discarding earlier intermediate checkpoints.
        This avoids scoring redundant intermediate structures when only
        the final beam survivors are of interest.
    """
    excluded = re.compile(
        r"(_seq\.pdb|_seq_\d+\.pdb|_grafted\.pdb|_traj\.pdb|_pX0.*\.pdb)$"
    )
    cp_re = re.compile(r"cp(\d+)_")

    all_pdbs = [
        p for p in Path(input_dir).rglob("*.pdb")
        if not excluded.search(p.name)
    ]

    # Separate arm_C PDBs from the rest
    arm_c_pdbs  = [p for p in all_pdbs if "arm_C" in str(p)]
    other_pdbs  = [p for p in all_pdbs if "arm_C" not in str(p)]

    # For arm_C: group by parent folder, keep only the highest checkpoint
    filtered_arm_c: List[Path] = []
    by_folder: Dict[Path, List[Path]] = {}
    for p in arm_c_pdbs:
        by_folder.setdefault(p.parent, []).append(p)

    for folder, pdbs in by_folder.items():
        # Find the highest checkpoint number present in this folder
        max_cp = -1
        for p in pdbs:
            m = cp_re.search(p.name)
            if m:
                max_cp = max(max_cp, int(m.group(1)))

        if max_cp == -1:
            # No checkpoint pattern found — include all as-is
            filtered_arm_c.extend(pdbs)
        else:
            # Keep only PDBs from the highest checkpoint
            kept = [
                p for p in pdbs
                if (m := cp_re.search(p.name)) and int(m.group(1)) == max_cp
            ]
            filtered_arm_c.extend(kept)
            skipped = len(pdbs) - len(kept)
            if skipped:
                print(f"  [find_backbone_pdbs] arm_C folder {folder.name}: "
                      f"keeping cp{max_cp:02d} ({len(kept)} PDB(s)), "
                      f"skipping {skipped} earlier checkpoint(s)")

    result = sorted(other_pdbs + filtered_arm_c)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_evenly(items: list, k: int) -> List[list]:
    """Split a list into k sublists as evenly as possible."""
    n    = len(items)
    base = n // k
    rem  = n % k
    shards, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        shards.append(items[start:end])
        start = end
    return shards


def _parse_gpu_list(gpu_ids_str: str) -> List[str]:
    return [g.strip() for g in gpu_ids_str.split(",") if g.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Per-GPU generation worker
# ─────────────────────────────────────────────────────────────────────────────

def _generation_worker(
    gpu_id:       str,
    shard_idx:    int,
    backbones:    List[str],   # paths as strings
    n_seqs:       int,
    mpnn_weights: str,
    output_dir:   str,
    temperature:  float,
    framework_pdb: str,
) -> List[GeneratedDesign]:
    """
    Run ProteinMPNN on a shard of backbone PDBs on a single GPU.
    Returns a list of GeneratedDesign for every structure produced.
    """
    _prepend_thermompnn_path()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = "cuda"

    import time as _t

    mpnn = load_proteinmpnn(mpnn_weights, device)

    # Build CDR mask from framework_pdb if provided, else first backbone
    mask_src = framework_pdb if framework_pdb else backbones[0]
    cdr_mask = build_cdr_mask(mask_src)

    results = []

    for backbone in backbones:
        b_stem  = Path(backbone).stem
        seq_dir = os.path.join(output_dir, "_sequences", b_stem)
        os.makedirs(seq_dir, exist_ok=True)

        for i in range(n_seqs):
            out_prefix = os.path.join(seq_dir, f"{b_stem}_seq_{i:03d}")
            out_pdb    = out_prefix + ".pdb"

            if os.path.exists(out_pdb):
                results.append(GeneratedDesign(
                    backbone=b_stem, seq_idx=i,
                    design_id=f"{b_stem}_seq_{i:03d}",
                    pdb_path=out_pdb, gpu_seconds=0.0,
                ))
                continue

            t0 = _t.perf_counter()
            try:
                result = _apply_sequence_and_anchors(
                    pdb_path=backbone,
                    out_prefix=out_prefix,
                    mpnn=mpnn,
                    cdr_mask=cdr_mask,
                    anchor_residues=[],
                    ref_pdb=backbone,
                    device=device,
                )
            except Exception as e:
                print(f"  [GPU {gpu_id}] WARN seq {i} of {b_stem}: {e}",
                      flush=True)
                continue
            gpu_s = _t.perf_counter() - t0

            if result and os.path.exists(result):
                results.append(GeneratedDesign(
                    backbone=b_stem, seq_idx=i,
                    design_id=f"{b_stem}_seq_{i:03d}",
                    pdb_path=result, gpu_seconds=gpu_s,
                ))
            else:
                print(f"  [GPU {gpu_id}] WARN no output for seq {i} of "
                      f"{b_stem}", flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Arm worker process — one per GPU group, puts GeneratedDesign onto queue
# ─────────────────────────────────────────────────────────────────────────────

def _shard_worker(
    gpu_id:          str,
    shard_idx:       int,
    backbones:       List[str],
    n_seqs:          int,
    mpnn_weights:    str,
    output_dir:      str,
    temperature:     float,
    framework_pdb:   str,
    generated_queue: mp.Queue,
):
    """
    Wraps _generation_worker and puts results onto the shared queue.
    Installs SIGTERM handler so partial results are flushed on budget expiry.
    """
    import signal

    def _sigterm(signum, frame):
        print(f"[Shard {shard_idx} GPU {gpu_id}] SIGTERM — flushing partial "
              "results.", flush=True)
        generated_queue.put((shard_idx, None))   # sentinel
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm)

    print(f"[Shard {shard_idx} GPU {gpu_id}] Starting: "
          f"{len(backbones)} backbone(s) × {n_seqs} seq(s) "
          f"(PID {os.getpid()})", flush=True)
    try:
        designs = _generation_worker(
            gpu_id=gpu_id,
            shard_idx=shard_idx,
            backbones=backbones,
            n_seqs=n_seqs,
            mpnn_weights=mpnn_weights,
            output_dir=output_dir,
            temperature=temperature,
            framework_pdb=framework_pdb,
        )
        for d in designs:
            generated_queue.put(d)
        print(f"[Shard {shard_idx} GPU {gpu_id}] Done: "
              f"{len(designs)} structure(s)", flush=True)
    except SystemExit:
        pass
    except Exception:
        import traceback
        print(f"[Shard {shard_idx} GPU {gpu_id}] ERROR:\n"
              f"{traceback.format_exc()}", flush=True)

    generated_queue.put((shard_idx, None))   # sentinel


# ─────────────────────────────────────────────────────────────────────────────
# Post-budget evaluation
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Post-generation evaluation — parallelised across GPUs
# ─────────────────────────────────────────────────────────────────────────────

def _eval_worker(
    gpu_id:              str,
    shard:               List[GeneratedDesign],
    native_pdb:          str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    af2_work_dir:        str,
    dockq_bin:           str,
    af2_num_recycles:    int,
    af2_num_models:      int,
) -> List[BackboneResult]:
    """
    Score a shard of GeneratedDesigns on a single GPU.
    ColabFold uses CUDA_VISIBLE_DEVICES to select the GPU.
    """
    _prepend_thermompnn_path()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    results = []
    for g in shard:
        timer = GPUTimer()
        iptm, dockq = score_design(
            pdb_path=g.pdb_path,
            af2_work_dir=af2_work_dir,
            native_pdb=native_pdb,
            colabfold_batch_bin=colabfold_batch_bin,
            colabfold_python=colabfold_python,
            timer=timer,
            af2_num_recycles=af2_num_recycles,
            af2_num_models=af2_num_models,
            dockq_bin=dockq_bin,
        )
        r = BackboneResult(
            backbone=g.backbone, seq_idx=g.seq_idx,
            design_id=g.design_id, pdb_path=g.pdb_path,
            iptm=iptm, dockq=dockq,
        )
        results.append(r)
        iptm_s  = f"{iptm:.3f}"  if iptm  is not None else "NA"
        dockq_s = f"{dockq:.3f}" if dockq is not None else "NA"
        print(f"  [GPU {gpu_id}] {g.design_id}  "
              f"ipTM={iptm_s}  DockQ={dockq_s}  "
              f"{'✓' if r.success else '✗'}", flush=True)
    return results


def _evaluate_generated(
    generated:           List[GeneratedDesign],
    native_pdb:          str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    af2_work_dir:        str,
    dockq_bin:           str,
    af2_num_recycles:    int,
    af2_num_models:      int,
    gpu_ids:             List[str],
) -> List[BackboneResult]:
    """
    Run ColabFold ipTM + DockQ on every GeneratedDesign, parallelised
    across gpu_ids.  Designs are split evenly across GPUs; each GPU runs
    its shard sequentially in a separate process.
    """
    total = len(generated)
    print(f"\n[Eval] Evaluating {total} structure(s) across "
          f"{len(gpu_ids)} GPU(s): {gpu_ids}")

    shards = _split_evenly(generated, len(gpu_ids))

    all_results: List[BackboneResult] = []
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as exe:
        futures = {}
        for gpu_id, shard in zip(gpu_ids, shards):
            if not shard:
                continue
            fut = exe.submit(
                _eval_worker,
                gpu_id, shard, native_pdb,
                colabfold_batch_bin, colabfold_python,
                af2_work_dir, dockq_bin,
                af2_num_recycles, af2_num_models,
            )
            futures[fut] = gpu_id

        for fut in as_completed(futures):
            gpu_id = futures[fut]
            try:
                results = fut.result()
                all_results.extend(results)
                print(f"[Eval] GPU {gpu_id} finished: "
                      f"{len(results)} structure(s)", flush=True)
            except Exception as e:
                print(f"[Eval] GPU {gpu_id} ERROR: {e}", flush=True)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Parallel generation scheduler
# ─────────────────────────────────────────────────────────────────────────────

def run_parallel_generation(
    backbones:           List[Path],
    gpu_ids:             List[str],
    n_seqs:              int,
    mpnn_weights:        str,
    output_dir:          str,
    temperature:         float,
    framework_pdb:       str,
    max_gpu_hours:       Optional[float],
) -> List[GeneratedDesign]:
    """
    Split backbones evenly across gpu_ids, launch one subprocess per GPU,
    collect GeneratedDesign items from the shared queue.
    Optional watchdog terminates workers after max_gpu_hours wall-clock time.
    """
    shards  = _split_evenly([str(p) for p in backbones], len(gpu_ids))
    queue: mp.Queue = mp.Queue()

    processes = []
    for shard_idx, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
        if not shard:
            continue
        p = mp.Process(
            target=_shard_worker,
            args=(gpu_id, shard_idx, shard, n_seqs, mpnn_weights,
                  output_dir, temperature, framework_pdb, queue),
            daemon=False,
        )
        p.start()
        processes.append(p)
        print(f"[Parallel] Launched PID {p.pid} on GPU {gpu_id}: "
              f"{len(shard)} backbone(s)", flush=True)

    if max_gpu_hours is not None:
        print(f"[Parallel] Generation budget: {max_gpu_hours:.2f} GPU-hour(s)")

    # ── Watchdog ──────────────────────────────────────────────────────────────
    stop_event  = threading.Event()
    wall_start  = time.perf_counter()

    def _watchdog():
        if max_gpu_hours is None:
            return
        budget_s = max_gpu_hours * 3600.0
        while not stop_event.wait(timeout=30):
            if time.perf_counter() - wall_start >= budget_s:
                print(
                    f"\n[Parallel] ⏰ Generation budget ({max_gpu_hours:.2f} h) "
                    "reached — terminating workers.", flush=True,
                )
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                return

    watchdog = threading.Thread(target=_watchdog, daemon=True)
    watchdog.start()

    # ── Collect ───────────────────────────────────────────────────────────────
    all_generated: List[GeneratedDesign] = []
    sentinels_expected = len(processes)
    sentinels_received = 0

    while sentinels_received < sentinels_expected and \
          any(p.is_alive() for p in processes):
        try:
            item = queue.get(timeout=5)
        except Exception:
            continue
        if isinstance(item, GeneratedDesign):
            all_generated.append(item)
            print(f"[Parallel] Generated: {item.design_id}", flush=True)
        elif isinstance(item, tuple) and item[1] is None:
            sentinels_received += 1
            print(f"[Parallel] Shard {item[0]} finished.", flush=True)

    # Final drain
    while not queue.empty():
        try:
            item = queue.get_nowait()
            if isinstance(item, GeneratedDesign):
                all_generated.append(item)
        except Exception:
            break

    stop_event.set()
    watchdog.join(timeout=5)
    for p in processes:
        p.join()

    print(f"\n[Parallel] Generation complete: "
          f"{len(all_generated)} structure(s) collected.")
    return all_generated


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    input_dir:           str,
    native_pdb:          str,
    output_dir:          str,
    n_seqs:              int,
    mpnn_weights:        str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    dockq_bin:           str        = "DockQ",
    af2_num_recycles:    int        = 3,
    af2_num_models:      int        = 1,
    device:              str        = "cuda",
    temperature:         float      = 0.2,
    framework_pdb:       str        = "",
    gpu_ids:             List[str]  = None,
    max_gpu_hours:       Optional[float] = None,
):
    gpu_ids = gpu_ids or ["0"]
    os.makedirs(output_dir, exist_ok=True)
    af2_work_dir = os.path.join(output_dir, "_af2")
    os.makedirs(af2_work_dir, exist_ok=True)

    # ── Find backbones ────────────────────────────────────────────────────────
    backbones = find_backbone_pdbs(input_dir)
    if not backbones:
        print(f"[score_backbones] No backbone PDBs found in {input_dir}")
        return
    print(f"[score_backbones] {len(backbones)} backbone(s) found, "
          f"{len(gpu_ids)} GPU(s): {gpu_ids}")

    # ── Parallel generation ───────────────────────────────────────────────────
    all_generated = run_parallel_generation(
        backbones=backbones,
        gpu_ids=gpu_ids,
        n_seqs=n_seqs,
        mpnn_weights=mpnn_weights,
        output_dir=output_dir,
        temperature=temperature,
        framework_pdb=framework_pdb,
        max_gpu_hours=max_gpu_hours,
    )

    if not all_generated:
        print("[score_backbones] No structures generated — exiting.")
        return

    # ── Evaluation (main process, not subject to GPU budget) ──────────────────
    all_results = _evaluate_generated(
        generated=all_generated,
        native_pdb=native_pdb,
        colabfold_batch_bin=colabfold_batch_bin,
        colabfold_python=colabfold_python,
        af2_work_dir=af2_work_dir,
        dockq_bin=dockq_bin,
        af2_num_recycles=af2_num_recycles,
        af2_num_models=af2_num_models,
        gpu_ids=gpu_ids,
    )

    # ── Sort by ipTM descending ───────────────────────────────────────────────
    all_results.sort(
        key=lambda r: r.iptm if r.iptm is not None else -1.0,
        reverse=True,
    )

    # ── Write TSV ─────────────────────────────────────────────────────────────
    stem     = Path(input_dir).resolve().name
    tsv_path = os.path.join(output_dir, f"{stem}_scored.tsv")
    with open(tsv_path, "w") as fh:
        fh.write("design_id\tbackbone\tseq_idx\tiptm\tdockq\tsuccess\tpdb_path\n")
        for r in all_results:
            fh.write(
                f"{r.design_id}\t{r.backbone}\t{r.seq_idx}\t"
                f"{r.iptm  if r.iptm  is not None else 'NA'}\t"
                f"{r.dockq if r.dockq is not None else 'NA'}\t"
                f"{r.success}\t{r.pdb_path}\n"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_success = sum(1 for r in all_results if r.success)
    iptms     = [r.iptm  for r in all_results if r.iptm  is not None]
    dockqs    = [r.dockq for r in all_results if r.dockq is not None]

    print(f"\n{'='*60}")
    print(f"  Total scored   : {len(all_results)}")
    print(f"  Successes      : {n_success} (ipTM>0.6 AND DockQ>0.23)")
    if iptms:
        print(f"  ipTM  mean={np.mean(iptms):.3f}  "
              f"max={max(iptms):.3f}  min={min(iptms):.3f}")
    if dockqs:
        print(f"  DockQ mean={np.mean(dockqs):.3f}  "
              f"max={max(dockqs):.3f}  min={min(dockqs):.3f}")
    print(f"  Results → {tsv_path}")
    print(f"{'='*60}\n")

    print("  Top designs by ipTM:")
    print(f"  {'design_id':<45}  {'ipTM':>6}  {'DockQ':>6}  {'ok':>4}")
    print(f"  {'-'*67}")
    for r in all_results[:10]:
        iptm_s  = f"{r.iptm:.3f}"  if r.iptm  is not None else "  NA"
        dockq_s = f"{r.dockq:.3f}" if r.dockq is not None else "  NA"
        print(f"  {r.design_id:<45}  {iptm_s:>6}  {dockq_s:>6}  "
              f"{'✓' if r.success else '✗':>4}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Multi-GPU backbone scoring: ProteinMPNN sequence generation "
            "parallelised across GPUs, followed by ColabFold ipTM + DockQ "
            "evaluation in the main process."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input_dir",           required=True,
                   help="Folder to search for backbone PDB files")
    p.add_argument("--native",              required=True,
                   help="Native/reference PDB for DockQ scoring")
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--n_seqs",              type=int, default=10,
                   help="Sequences per backbone (default: 10)")
    p.add_argument("--mpnn_weights",        required=True,
                   help="ProteinMPNN or IgDesign fine-tuned weights (.pt/.ckpt)")
    p.add_argument("--colabfold_batch_bin", required=True)
    p.add_argument("--colabfold_python",    required=True)
    p.add_argument("--dockq_bin",           default="DockQ")
    p.add_argument("--af2_num_recycles",    type=int, default=3)
    p.add_argument("--af2_num_models",      type=int, default=1)
    p.add_argument("--device",              default="cuda",
                   help="Device for evaluation (default: cuda)")
    p.add_argument("--temperature",         type=float, default=0.2)
    p.add_argument("--framework_pdb",       default="",
                   help="Optional PDB to build CDR mask from")
    p.add_argument("--gpu_ids",             default="0",
                   help="Comma-separated GPU IDs for parallel generation "
                        "(default: 0). e.g. --gpu_ids 0,1,2,3")
    p.add_argument("--max_gpu_hours",       type=float, default=None,
                   help="Optional wall-clock budget in hours for generation "
                        "phase (default: no limit)")
    return p.parse_args()


def main():
    mp.set_start_method("spawn", force=True)
    args     = parse_args()
    gpu_ids  = _parse_gpu_list(args.gpu_ids)
    run(
        input_dir=str(Path(args.input_dir).resolve()),
        native_pdb=str(Path(args.native).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        n_seqs=args.n_seqs,
        mpnn_weights=str(Path(args.mpnn_weights).resolve()),
        colabfold_batch_bin=args.colabfold_batch_bin,
        colabfold_python=args.colabfold_python,
        dockq_bin=args.dockq_bin,
        af2_num_recycles=args.af2_num_recycles,
        af2_num_models=args.af2_num_models,
        device=args.device,
        temperature=args.temperature,
        framework_pdb=args.framework_pdb,
        gpu_ids=gpu_ids,
        max_gpu_hours=args.max_gpu_hours,
    )


if __name__ == "__main__":
    main()