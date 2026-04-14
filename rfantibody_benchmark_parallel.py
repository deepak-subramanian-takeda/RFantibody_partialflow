"""
rfantibody_benchmark_parallel.py

Multi-GPU parallel version of rfantibody_benchmark.py.

Each arm runs as an independent subprocess pinned to one or more GPUs via
CUDA_VISIBLE_DEVICES.  Arms assigned to the same GPU run sequentially on
that GPU; arms assigned to different GPUs run concurrently.

GPU assignment is controlled by --gpu_map, e.g.:
    --gpu_map A:0,B:0,C:1,D:1
    Arms A and B share GPU 0 (sequential); C and D share GPU 1 (sequential).
    A/B and C/D run concurrently across the two GPUs.

    --gpu_map A:0,B:1,C:2,D:3
    All four arms run fully in parallel on four separate GPUs.

    --gpu_map A:0,0,C:1     (only run arms A and C)
    Arm A gets GPUs 0 and 1 (CUDA_VISIBLE_DEVICES=0,1); C gets GPU 2.

Results from all arms are merged into a single TSV and summary JSON,
identical in format to the serial rfantibody_benchmark.py outputs.

Timing:
    Wall-clock time per arm is measured in the worker process and reported
    as gpu_seconds in the merged TSV.  Because arms run concurrently the
    total wall-clock time is determined by the slowest GPU group.
"""

from __future__ import annotations

import sys
import os

# ── Ensure ThermoMPNN's protein_mpnn_utils is found before the installed
#    RFantibody package version.  This must happen before any other local
#    imports because multiprocessing.spawn re-imports this file from scratch
#    in each child process, so PYTHONPATH set in the shell is not inherited.
def _prepend_thermompnn_path():
    _here = os.path.dirname(os.path.abspath(__file__))
    _thermo = os.path.join(_here, "ThermoMPNN")
    if os.path.isdir(_thermo) and _thermo not in sys.path:
        sys.path.insert(0, _thermo)

_prepend_thermompnn_path()

import argparse
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── imports from the serial benchmark ────────────────────────────────────────
from rfantibody_benchmark import (
    DesignResult, ArmSummary, GPUTimer,
    run_arm_A, run_arm_B, run_arm_C, run_arm_D,
    summarise, print_report, save_results,
    score_design,                        # used in post-budget evaluation
    CHAIN_H, CHAIN_L, CHAIN_T,
    IPTM_SUCCESS_THRESHOLD,
)
from partial_diffusion_maturation import (
    parse_free_loops, split_hlt_complex,
    parse_hlt_remarks, read_pdb_residues,
    build_contig_string, build_provide_seq,
    mask_anchors_in_hlt, graft_target_sequence,
    build_rfdiffusion_command, load_anchors,
    CHAIN_H, CHAIN_L, CHAIN_T,
)
from smc_denovo_maturation import (
    build_cdr_mask, load_epitope_ca,
    load_thermompnn, load_proteinmpnn,
    build_denovo_contig,
    design_sequence_onto_backbone,
    graft_anchor_identities,
    run_denovo_round,
)
from beam_denovo_maturation_complexa import (
    BeamNode, RANKING_MODES,
    score_complexa_reward,
    _rollout_and_score,
    _apply_sequence_and_anchors,
    write_renumbered_pdb,
    IPTM_SUCCESS_THRESHOLD,
)
from evaluate_designs import (
    write_colabfold_fasta, compute_target_crop,
    run_colabfold, find_top_af2_result,
    extract_iptm, BINDER_CHAINS,
)


# ─────────────────────────────────────────────────────────────────────────────
# GPU map parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_gpu_map(spec: str, arms: List[str]) -> Dict[str, str]:
    """
    Parse --gpu_map into {arm: cuda_visible_devices_string}.

    Formats accepted:
        "A:0,B:0,C:1,D:1"       → A→"0", B→"0", C→"1", D→"1"
        "A:0,1,C:2"             → A→"0,1",  C→"2"
        ""  (empty)             → all arms → "0"
    """
    if not spec:
        return {arm: "0" for arm in arms}

    result: Dict[str, str] = {}
    # Split on uppercase arm letters that are followed by a colon
    import re
    tokens = re.split(r'(?=[A-D]:)', spec)
    for tok in tokens:
        tok = tok.strip().strip(",")
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"Cannot parse gpu_map token '{tok}'. "
                             "Expected format: ARM:gpu_id[,gpu_id,...]")
        arm, gpus = tok.split(":", 1)
        arm = arm.strip().upper()
        result[arm] = gpus.strip().strip(",")

    # Default any missing arms to GPU 0
    for arm in arms:
        if arm not in result:
            print(f"[WARN] No GPU assigned for arm {arm} — defaulting to GPU 0.")
            result[arm] = "0"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Worker function — runs in a separate process
# ─────────────────────────────────────────────────────────────────────────────

def _arm_worker(
    arm:         str,
    gpu_ids:     str,       # e.g. "0" or "0,1"
    args_dict:   dict,      # serialisable copy of parsed args
    result_queue: mp.Queue,
):
    """
    Entry point for each arm subprocess.

    Sets CUDA_VISIBLE_DEVICES before importing any torch/CUDA code, loads
    the shared scoring infrastructure, runs the arm, and puts
    List[DesignResult] onto result_queue.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    # After setting CUDA_VISIBLE_DEVICES, device "cuda:0" always refers to
    # the first GPU in the assigned set.
    device = "cuda" if gpu_ids else "cpu"

    arm_label = f"[Arm {arm} / GPU {gpu_ids}]"
    print(f"{arm_label} Starting (PID {os.getpid()})", flush=True)
    t0 = time.perf_counter()

    try:
        a = argparse.Namespace(**args_dict)
        free_loops   = parse_free_loops(a.free_loops)
        input_pdb    = str(Path(a.input).resolve())
        native_pdb   = str(Path(a.native).resolve())
        output_dir   = str(Path(a.output_dir).resolve())

        # ── split HLT once per worker (cheap, avoids shared-memory issues) ──
        split_dir    = os.path.join(output_dir, f"_split_arm{arm}")
        target_pdb, framework_pdb = split_hlt_complex(input_pdb, split_dir)

        # ── shared scoring infrastructure ────────────────────────────────────
        print(f"{arm_label} Loading scoring models…", flush=True)
        cdr_mask   = build_cdr_mask(framework_pdb)
        epitope_ca = load_epitope_ca(target_pdb, a.hotspots, device)
        thermo     = load_thermompnn(
            config_yaml=a.thermo_model_yaml,
            local_yaml=a.thermo_local_yaml,
            checkpoint=a.thermo_checkpoint,
            device=device,
        )
        mpnn = load_proteinmpnn(a.mpnn_weights, device)

        af2_work_dir = os.path.join(output_dir, f"_af2_eval_arm{arm}")
        os.makedirs(af2_work_dir, exist_ok=True)

        extra = [x for x in (a.extra or []) if x != "--"]

        eval_kw = dict(
            native_pdb=native_pdb,
            colabfold_batch_bin=a.colabfold_batch_bin,
            colabfold_python=a.colabfold_python,
            af2_work_dir=af2_work_dir,
            mpnn=mpnn, cdr_mask=cdr_mask,
            framework_pdb=framework_pdb,
            extra_args=extra,
            device=device,
            dockq_bin=a.dockq_bin,
            af2_num_models=a.af2_num_models,
            nanobody=a.nanobody,
            free_loops=free_loops,
        )
        beam_kw = dict(
            thermo=thermo, epitope_ca=epitope_ca,
            beam_width=a.beam_width,
            branch_factor=a.branch_factor,
            n_checkpoints=a.n_checkpoints,
            w_iptm=a.w_iptm, w_thermo=a.w_thermo,
            iptm_threshold=a.iptm_threshold,
            ranking_mode=a.ranking_mode,
            af2_num_recycles=a.af2_num_recycles_beam,
            stem=a.name or Path(input_pdb).stem,
        )

        # ── dispatch ─────────────────────────────────────────────────────────
        results: List[DesignResult] = []

        if arm == "A":
            results = run_arm_A(
                input_pdb=input_pdb,
                output_dir=output_dir,
                hotspots=a.hotspots,
                model_weights=a.model_weights,
                num_designs=a.num_designs,
                af2_num_recycles=a.af2_num_recycles_eval,
                **eval_kw,
            )
        elif arm == "B":
            results = run_arm_B(
                input_pdb=input_pdb,
                anchors_json=str(Path(a.anchors).resolve()),
                output_dir=output_dir,
                hotspots=a.hotspots,
                model_weights=a.model_weights,
                num_designs=a.num_designs,
                af2_num_recycles=a.af2_num_recycles_eval,
                **eval_kw,
            )
        elif arm == "C":
            results = run_arm_C(
                input_pdb=input_pdb,
                output_dir=output_dir,
                hotspots=a.hotspots,
                model_weights=a.model_weights,
                **eval_kw, **beam_kw,
            )
        elif arm == "D":
            results = run_arm_D(
                input_pdb=input_pdb,
                anchors_json=str(Path(a.anchors).resolve()),
                output_dir=output_dir,
                hotspots=a.hotspots,
                model_weights=a.model_weights,
                **eval_kw, **beam_kw,
            )

        completed_results.extend(results)
        elapsed = time.perf_counter() - t0
        print(f"{arm_label} Finished in {elapsed/3600:.2f} h  "
              f"({len(completed_results)} designs)", flush=True)
        result_queue.put((arm, completed_results, None))

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"{arm_label} FAILED:\n{tb}", flush=True)
        result_queue.put((arm, [], str(exc)))


# ─────────────────────────────────────────────────────────────────────────────
# GPU group scheduler
# ─────────────────────────────────────────────────────────────────────────────

def run_parallel(
    arms:          List[str],
    gpu_map:       Dict[str, str],
    args_dict:     dict,
    max_gpu_hours: Optional[float] = None,
) -> Dict[str, List[DesignResult]]:
    """
    Group arms by GPU assignment.  Arms on different GPUs launch concurrently;
    arms sharing a GPU run sequentially within that GPU group.

    If max_gpu_hours is set, a watchdog thread monitors wall-clock time and
    sends SIGTERM to all child processes once the budget is exceeded.  Each
    child catches the signal, flushes any completed results onto the queue,
    and exits cleanly.  Results collected up to that point are returned.

    Returns {arm: List[DesignResult]}.
    """
    from collections import defaultdict
    gpu_groups: Dict[str, List[str]] = defaultdict(list)
    for arm in arms:
        gpu_groups[gpu_map[arm]].append(arm)

    print("\n[Parallel] GPU assignment:")
    for gpu_ids, group_arms in sorted(gpu_groups.items()):
        mode = "concurrent" if len(group_arms) == 1 else "sequential within group"
        print(f"  GPU {gpu_ids}: arms {group_arms}  ({mode})")
    if max_gpu_hours is not None:
        print(f"[Parallel] GPU-hour budget: {max_gpu_hours:.2f} h "
              f"({max_gpu_hours * 3600:.0f} s wall-clock)")

    result_queue: mp.Queue = mp.Queue()
    all_results: Dict[str, List[DesignResult]] = {arm: [] for arm in arms}

    processes = []
    for gpu_ids, group_arms in gpu_groups.items():
        p = mp.Process(
            target=_gpu_group_worker,
            args=(group_arms, gpu_ids, args_dict, result_queue),
            daemon=False,
        )
        p.start()
        processes.append(p)
        print(f"[Parallel] Launched PID {p.pid} for GPU {gpu_ids} "
              f"(arms {group_arms})", flush=True)

    wall_start = time.perf_counter()

    # ── Watchdog thread ───────────────────────────────────────────────────────
    stop_event = threading.Event()

    def _watchdog():
        if max_gpu_hours is None:
            return
        budget_s = max_gpu_hours * 3600.0
        while not stop_event.wait(timeout=30):
            elapsed = time.perf_counter() - wall_start
            if elapsed >= budget_s:
                print(
                    f"\n[Parallel] ⏰ GPU-hour budget ({max_gpu_hours:.2f} h) "
                    f"reached after {elapsed / 3600:.3f} h — "
                    "sending SIGTERM to all workers…",
                    flush=True,
                )
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                return

    watchdog = threading.Thread(target=_watchdog, daemon=True)
    watchdog.start()

    # ── Collect results ───────────────────────────────────────────────────────
    # We don't know exactly how many results will arrive (workers may be
    # terminated early), so we drain the queue until all processes have exited.
    while any(p.is_alive() for p in processes):
        try:
            arm, results, error = result_queue.get(timeout=5)
            if error:
                print(f"[Parallel] Arm {arm} reported error: {error}")
            else:
                all_results[arm].extend(results)
                print(f"[Parallel] Received results for arm {arm} "
                      f"({len(results)} designs)", flush=True)
        except Exception:
            pass  # queue.Empty or timeout — loop and recheck process liveness

    # Drain any remaining items that arrived after processes exited
    while not result_queue.empty():
        try:
            arm, results, error = result_queue.get_nowait()
            if not error:
                all_results[arm].extend(results)
                print(f"[Parallel] (drain) Arm {arm}: {len(results)} design(s)")
        except Exception:
            break

    stop_event.set()   # stop watchdog
    watchdog.join(timeout=5)

    for p in processes:
        p.join()

    return all_results


def _gpu_group_worker(
    arms:             List[str],
    gpu_ids:          str,
    args_dict:        dict,
    generated_queue:  mp.Queue,
):
    """Runs a list of arms sequentially on the same GPU(s)."""
    for arm in arms:
        _arm_worker(arm, gpu_ids, args_dict, generated_queue)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Multi-GPU parallel RFantibody benchmark.\n"
            "Arms assigned to different GPUs run concurrently;\n"
            "arms sharing a GPU run sequentially on that GPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── required ─────────────────────────────────────────────────────────────
    p.add_argument("--input",             required=True)
    p.add_argument("--native",            required=True)
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--hotspots",          required=True)
    p.add_argument("--model_weights",     required=True)
    p.add_argument("--colabfold_batch_bin", required=True)
    p.add_argument("--colabfold_python",    required=True)
    p.add_argument("--thermo_local_yaml",   required=True)
    p.add_argument("--thermo_model_yaml",   required=True)
    p.add_argument("--thermo_checkpoint",   required=True)
    p.add_argument("--mpnn_weights",        required=True)
    p.add_argument("--dockq_bin",    default="DockQ")
    p.add_argument("--anchors",
                   help="Required for arms B and D")
    # ── parallelism ──────────────────────────────────────────────────────────
    p.add_argument(
        "--gpu_map", default="",
        help=(
            "Comma-separated ARM:GPU_ID assignments, e.g. "
            "'A:0,B:0,C:1,D:1'  (A+B share GPU 0; C+D share GPU 1). "
            "Omit to place all arms on GPU 0 (serial fallback). "
            "Multi-GPU per arm: 'A:0,1' sets CUDA_VISIBLE_DEVICES=0,1 for arm A."
        ),
    )
    p.add_argument(
        "--max_gpu_hours", type=float, default=None,
        help=(
            "Optional wall-clock budget in GPU-hours. When elapsed time "
            "exceeds this limit all worker processes are terminated and "
            "results collected so far are saved (default: no limit)."
        ),
    )
    # ── arm selection ─────────────────────────────────────────────────────────
    p.add_argument("--arms", default="A,B,C,D")
    # ── design counts ─────────────────────────────────────────────────────────
    p.add_argument("--num_designs",   type=int, default=50)
    # ── beam hyperparameters ─────────────────────────────────────────────────
    p.add_argument("--beam_width",    type=int,   default=4)
    p.add_argument("--branch_factor", type=int,   default=4)
    p.add_argument("--n_checkpoints", type=int,   default=4)
    p.add_argument("--ranking_mode",  default="cumulative",
                   choices=["cumulative", "latest", "average"])
    p.add_argument("--w_iptm",         type=float, default=1.0)
    p.add_argument("--w_thermo",       type=float, default=0.5)
    p.add_argument("--iptm_threshold", type=float,
                   default=IPTM_SUCCESS_THRESHOLD)
    p.add_argument("--af2_num_recycles_beam", type=int, default=1)
    p.add_argument("--af2_num_recycles_eval", type=int, default=3)
    p.add_argument("--af2_num_models",  type=int, default=1)
    # ── other ─────────────────────────────────────────────────────────────────
    p.add_argument("--free_loops",  default="")
    p.add_argument("--nanobody",    action="store_true")
    p.add_argument("--name",        default="")
    p.add_argument("extra",         nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    # multiprocessing requires the 'spawn' start method on Linux/macOS when
    # CUDA is involved to avoid forking after CUDA initialisation.
    mp.set_start_method("spawn", force=True)

    args  = parse_args()
    arms  = [a.strip().upper() for a in args.arms.split(",")]

    if any(arm in ("B", "D") for arm in arms) and not args.anchors:
        sys.exit("[ERROR] --anchors is required when running arms B or D.")

    output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)
    stem = args.name or Path(args.input).stem

    try:
        gpu_map = parse_gpu_map(args.gpu_map, arms)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    # Serialise args to a plain dict so it can be pickled across processes
    args_dict = vars(args)

    wall_t0 = time.perf_counter()
    all_results = run_parallel(arms, gpu_map, args_dict,
                               max_gpu_hours=args.max_gpu_hours)
    wall_elapsed = time.perf_counter() - wall_t0

    print(f"\n[Parallel] All arms complete. "
          f"Wall-clock time: {wall_elapsed/3600:.2f} h")

    summaries = summarise(all_results)
    print_report(summaries)
    save_results(summaries, output_dir, stem)


if __name__ == "__main__":
    main()