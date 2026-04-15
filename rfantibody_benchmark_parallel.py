"""
rfantibody_benchmark_parallel.py

Multi-GPU parallel version of rfantibody_benchmark.py.

GPU assignment is controlled by --gpu_map, e.g.:
    --gpu_map A:0,1,2,3,C:4,5,6,7

Within each arm, available GPUs are used in parallel:

  Arms A/B:
    num_designs is split evenly across the assigned GPUs.
    One RFdiffusion subprocess runs per GPU simultaneously, then
    ProteinMPNN sequence design is applied on each GPU's outputs.
    All outputs are merged before evaluation.

  Arms C/D (beam search):
    At each checkpoint the N×L rollouts are distributed across GPUs,
    each GPU running its assigned subset in parallel.  All candidates
    are gathered to a single coordinator process for scoring and pruning
    (BeamNode selection), then the surviving beam nodes are redistributed
    across GPUs for the next checkpoint's rollouts.

GPU budget (--max_gpu_hours):
    Applies to the generation phase only (RFdiffusion + beam rollouts).
    When the budget expires, all generation workers are terminated and
    every structure produced so far is evaluated in the main process
    (ColabFold ipTM + DockQ).
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── imports from the serial benchmark ────────────────────────────────────────
from rfantibody_benchmark import (
    DesignResult, ArmSummary, GPUTimer,
    summarise, print_report, save_results,
    score_design,
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
)
from beam_denovo_maturation_complexa import (
    BeamNode, RANKING_MODES,
    _rollout_and_score,
    _apply_sequence_and_anchors,
    write_renumbered_pdb,
)
from evaluate_designs import (
    write_colabfold_fasta, compute_target_crop,
    run_colabfold, find_top_af2_result,
    extract_iptm, BINDER_CHAINS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Generated design dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeneratedDesign:
    arm:         str
    design_id:   str
    pdb_path:    str
    gpu_seconds: float   # generation wall-clock time for this design


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_evenly(n: int, k: int) -> List[int]:
    """Split n items into k buckets as evenly as possible."""
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def _parse_gpu_list(gpu_map_value: str) -> List[str]:
    """Turn the GPU string for one arm into a list of individual GPU IDs.
    e.g. "0,1,2" → ["0", "1", "2"]
    """
    return [g.strip() for g in gpu_map_value.split(",") if g.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# GPU map parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_gpu_map(spec: str, arms: List[str]) -> Dict[str, str]:
    """
    Parse --gpu_map into {arm: comma_separated_gpu_ids_string}.

    e.g. "A:0,1,2,3,C:4,5,6,7" → {"A": "0,1,2,3", "C": "4,5,6,7"}

    All GPUs listed for an arm are used in parallel within that arm:
      - Arms A/B: designs split evenly across the GPUs
      - Arms C/D: rollouts distributed round-robin across the GPUs
    """
    if not spec:
        return {arm: "0" for arm in arms}

    import re
    result: Dict[str, str] = {}
    tokens = re.split(r'(?=[A-D]:)', spec)
    for tok in tokens:
        tok = tok.strip().strip(",")
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"Cannot parse gpu_map token '{tok}'. "
                             "Expected format: ARM:gpu_id[,gpu_id,...]")
        arm, gpus = tok.split(":", 1)
        result[arm.strip().upper()] = gpus.strip().strip(",")

    for arm in arms:
        if arm not in result:
            print(f"[WARN] No GPU assigned for arm {arm} — defaulting to GPU 0.")
            result[arm] = "0"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-GPU generation workers (called via ProcessPoolExecutor)
# _rfd_worker_A / _rfd_worker_B : one shard of designs on one GPU
# _beam_rollout_worker           : one beam rollout on one GPU
# _generate_arm_AB_parallel      : splits num_designs across gpu_ids
# _generate_arm_beam_parallel    : distributes rollouts across gpu_ids
# ─────────────────────────────────────────────────────────────────────────────

def _rfd_worker_A(
    gpu_id:        str,
    shard_idx:     int,
    n_designs:     int,
    input_pdb:     str,
    output_dir:    str,
    framework_pdb: str,
    hotspots:      str,
    model_weights: str,
    free_loops:    dict,
    nanobody:      bool,
    mpnn_weights:  str,
    extra:         List[str],
) -> List[GeneratedDesign]:
    """Generate n_designs structures on a single GPU for arm A."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = "cuda"

    import subprocess
    import time as _t
    from pathlib import Path as _P

    cdr_ranges = parse_hlt_remarks(input_pdb)
    residues   = read_pdb_residues(input_pdb)
    contig     = build_contig_string(
        residues=residues, cdr_ranges=cdr_ranges,
        anchor_residues=[], free_loop_overrides=free_loops,
        nanobody=nanobody,
    )
    shard_dir     = os.path.join(output_dir, "arm_A_vanilla",
                                 f"shard_{shard_idx:02d}")
    os.makedirs(shard_dir, exist_ok=True)
    output_prefix = os.path.join(shard_dir, "vanilla")

    cmd = build_rfdiffusion_command(
        input_pdb=input_pdb, target_pdb="", framework_pdb=framework_pdb,
        contig_string=contig, provide_seq="", hotspots=hotspots,
        output_prefix=output_prefix, partial_T=50,
        num_designs=n_designs, model_weights=model_weights,
        extra_args=extra,
    )
    t0 = _t.perf_counter()
    subprocess.run(" ".join(cmd), shell=True)
    rfd_s = _t.perf_counter() - t0

    out_pdbs = sorted(_P(shard_dir).glob("vanilla*.pdb"))
    rfd_per  = rfd_s / max(len(out_pdbs), 1)

    cdr_mask = build_cdr_mask(framework_pdb)
    mpnn     = load_proteinmpnn(mpnn_weights, device)

    results = []
    for pdb in out_pdbs:
        t1   = _t.perf_counter()
        dsgn = _apply_sequence_and_anchors(
            pdb_path=str(pdb),
            out_prefix=str(pdb).replace(".pdb", ""),
            mpnn=mpnn, cdr_mask=cdr_mask,
            anchor_residues=[], ref_pdb=input_pdb, device=device,
        )
        mpnn_s = _t.perf_counter() - t1
        results.append(GeneratedDesign(
            arm="A", design_id=_P(pdb).stem,
            pdb_path=dsgn, gpu_seconds=rfd_per + mpnn_s,
        ))
    return results


def _rfd_worker_B(
    gpu_id:        str,
    shard_idx:     int,
    n_designs:     int,
    input_pdb:     str,
    output_dir:    str,
    framework_pdb: str,
    hotspots:      str,
    model_weights: str,
    anchors_json:  str,
    free_loops:    dict,
    nanobody:      bool,
    mpnn_weights:  str,
    extra:         List[str],
) -> List[GeneratedDesign]:
    """Generate n_designs structures on a single GPU for arm B."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = "cuda"

    import subprocess
    import time as _t
    from pathlib import Path as _P

    cdr_ranges      = parse_hlt_remarks(input_pdb)
    residues        = read_pdb_residues(input_pdb)
    anchor_residues = load_anchors(anchors_json)
    stem            = _P(input_pdb).stem

    shard_dir = os.path.join(output_dir, "arm_B_anchored",
                             f"shard_{shard_idx:02d}")
    os.makedirs(shard_dir, exist_ok=True)

    masked_pdb = os.path.join(shard_dir, f"{stem}_anchors_masked.pdb")
    mask_anchors_in_hlt(input_pdb, anchor_residues, masked_pdb)

    provide_seq = build_provide_seq(
        residues=residues, cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues, nanobody=nanobody,
    )
    contig = build_contig_string(
        residues=residues, cdr_ranges=cdr_ranges,
        anchor_residues=anchor_residues,
        free_loop_overrides=free_loops, nanobody=nanobody,
    )
    output_prefix = os.path.join(shard_dir, f"{stem}_anchored_T50")
    cmd = build_rfdiffusion_command(
        input_pdb=masked_pdb, target_pdb="", framework_pdb=framework_pdb,
        contig_string=contig, provide_seq=provide_seq,
        hotspots=hotspots, output_prefix=output_prefix,
        partial_T=50, num_designs=n_designs,
        model_weights=model_weights, extra_args=extra,
    )
    t0 = _t.perf_counter()
    subprocess.run(" ".join(cmd), shell=True)
    rfd_s = _t.perf_counter() - t0

    out_pdbs = sorted(_P(shard_dir).glob(f"{stem}_anchored_T50*.pdb"))
    rfd_per  = rfd_s / max(len(out_pdbs), 1)

    cdr_mask = build_cdr_mask(framework_pdb)
    mpnn     = load_proteinmpnn(mpnn_weights, device)

    results = []
    for pdb in out_pdbs:
        grafted = str(pdb).replace(".pdb", "_grafted.pdb")
        graft_target_sequence(
            rfdiffusion_pdb=str(pdb), original_target=input_pdb,
            input_pdb=masked_pdb, out_path=grafted,
        )
        t1   = _t.perf_counter()
        dsgn = _apply_sequence_and_anchors(
            pdb_path=grafted,
            out_prefix=grafted.replace(".pdb", ""),
            mpnn=mpnn, cdr_mask=cdr_mask,
            anchor_residues=anchor_residues,
            ref_pdb=input_pdb, device=device,
        )
        mpnn_s = _t.perf_counter() - t1
        results.append(GeneratedDesign(
            arm="B", design_id=_P(pdb).stem,
            pdb_path=dsgn, gpu_seconds=rfd_per + mpnn_s,
        ))
    return results


def _beam_rollout_worker(
    gpu_id:         str,
    parent_node:    BeamNode,
    child_idx:      int,
    node_counter:   int,
    checkpoint_idx: int,
    rollout_kw:     dict,
    mpnn_weights:   str,
    thermo_local_yaml: str,
    thermo_model_yaml: str,
    thermo_checkpoint: str,
) -> Optional[BeamNode]:
    """
    Run a single beam rollout on the specified GPU.

    Models are loaded here rather than passed via pickle, because
    PyTorch model objects loaded via importlib (thermompnn_protein_mpnn_utils)
    cannot be pickled across process boundaries.
    """
    _prepend_thermompnn_path()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = "cuda"

    # Load models fresh in this worker process
    mpnn   = load_proteinmpnn(mpnn_weights, device)
    thermo = load_thermompnn(
        config_yaml=thermo_model_yaml,
        local_yaml=thermo_local_yaml,
        checkpoint=thermo_checkpoint,
        device=device,
    )

    # Inject freshly loaded models into rollout_kw copy
    kw = dict(rollout_kw)
    kw["mpnn"]   = mpnn
    kw["thermo"] = thermo
    kw["device"] = device

    return _rollout_and_score(
        parent_node=parent_node,
        child_idx=child_idx,
        checkpoint_idx=checkpoint_idx,
        node_counter=node_counter,
        **kw,
    )


def _generate_arm_AB_parallel(
    arm:             str,
    a:               argparse.Namespace,
    input_pdb:       str,
    output_dir:      str,
    framework_pdb:   str,
    free_loops:      dict,
    nanobody:        bool,
    extra:           List[str],
    gpu_ids:         List[str],
    generated_queue: mp.Queue,
):
    """
    Split num_designs evenly across gpu_ids, run one RFdiffusion process
    per GPU in parallel via ProcessPoolExecutor, then collect all results.
    """
    counts    = _split_evenly(a.num_designs, len(gpu_ids))
    worker_fn = _rfd_worker_A if arm == "A" else _rfd_worker_B

    common = dict(
        input_pdb=input_pdb, output_dir=output_dir,
        framework_pdb=framework_pdb, hotspots=a.hotspots,
        model_weights=a.model_weights, free_loops=free_loops,
        nanobody=nanobody, mpnn_weights=a.mpnn_weights,
        extra=extra,
    )
    if arm == "B":
        common["anchors_json"] = str(Path(a.anchors).resolve())

    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as exe:
        futures = {}
        for shard_idx, (gpu_id, n) in enumerate(zip(gpu_ids, counts)):
            if n == 0:
                continue
            fut = exe.submit(worker_fn, gpu_id, shard_idx, n, **common)
            futures[fut] = (gpu_id, shard_idx)

        for fut in as_completed(futures):
            gpu_id, shard_idx = futures[fut]
            try:
                designs = fut.result()
                for d in designs:
                    generated_queue.put(d)
                print(f"  [Arm {arm} shard {shard_idx} GPU {gpu_id}] "
                      f"{len(designs)} structure(s) done.", flush=True)
            except Exception as e:
                print(f"  [Arm {arm} shard {shard_idx} GPU {gpu_id}] "
                      f"ERROR: {e}", flush=True)


def _generate_arm_beam_parallel(
    arm:             str,
    a:               argparse.Namespace,
    input_pdb:       str,
    output_dir:      str,
    framework_pdb:   str,
    free_loops:      dict,
    nanobody:        bool,
    extra:           List[str],
    gpu_ids:         List[str],
    generated_queue: mp.Queue,
    anchor_residues: list = None,
):
    """
    Multi-GPU beam search for arms C/D.

    Rollouts at each checkpoint are distributed across gpu_ids via
    ProcessPoolExecutor (each GPU runs its assigned subset in parallel).
    After all rollouts complete, scoring and pruning happen in this process
    (coordinator on gpu_ids[0]), then survivors are redistributed for the
    next checkpoint.
    """
    anchor_residues = anchor_residues or []
    arm_dir  = os.path.join(
        output_dir,
        "arm_C_beam_no_anchor" if arm == "C" else "arm_D_beam_anchored",
    )
    work_dir = os.path.join(arm_dir, "_beam_work")
    os.makedirs(work_dir, exist_ok=True)

    # Coordinator runs on first GPU in the list
    coord_gpu = gpu_ids[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = coord_gpu
    device = "cuda"

    cdr_ranges = parse_hlt_remarks(input_pdb)
    residues   = read_pdb_residues(input_pdb)
    if not nanobody and not any(r.pdb_chain == CHAIN_L for r in residues):
        nanobody = True

    renumbered_pdb = os.path.join(work_dir, "input_renumbered.pdb")
    resnum_mapping = write_renumbered_pdb(input_pdb, renumbered_pdb)

    def _remap_hs(hs, mapping):
        out = []
        for tok in hs.split(","):
            ch, rn = tok.strip()[0], int(tok.strip()[1:])
            out.append(f"{ch}{mapping.get((ch, rn), rn)}")
        return ",".join(out)

    remapped_hs    = _remap_hs(a.hotspots, resnum_mapping)
    renumbered_res = read_pdb_residues(renumbered_pdb)
    remapped_contig = build_denovo_contig(
        renumbered_res, cdr_ranges, anchor_residues, free_loops, nanobody,
    )

    provide_extra = []
    if arm == "D":
        provide_seq = build_provide_seq(
            residues=residues, cdr_ranges=cdr_ranges,
            anchor_residues=anchor_residues, nanobody=nanobody,
        )
        provide_extra = [f"'contigmap.provide_seq=[{provide_seq}]'"]

    # Load scoring infrastructure on coordinator GPU
    split_dir  = os.path.join(output_dir, f"_split_arm{arm}")
    target_pdb, _ = split_hlt_complex(input_pdb, split_dir)

    mpnn       = load_proteinmpnn(a.mpnn_weights, device)
    thermo     = load_thermompnn(
        config_yaml=a.thermo_model_yaml,
        local_yaml=a.thermo_local_yaml,
        checkpoint=a.thermo_checkpoint,
        device=device,
    )
    epitope_ca = load_epitope_ca(target_pdb, a.hotspots, device)
    cdr_mask   = build_cdr_mask(framework_pdb)

    af2_bwork = os.path.join(work_dir, "_af2")
    os.makedirs(af2_bwork, exist_ok=True)

    rollout_kw = dict(
        work_dir=work_dir, model_weights=a.model_weights,
        input_pdb=input_pdb, renumbered_pdb=renumbered_pdb,
        contig_string=remapped_contig, hotspots=remapped_hs,
        anchor_residues=anchor_residues, cdr_ranges=cdr_ranges,
        extra_args=extra + provide_extra,
        # models are intentionally omitted here — each worker loads them
        # fresh to avoid pickling errors with thermompnn_protein_mpnn_utils
        cdr_mask=cdr_mask,
        epitope_ca=epitope_ca,
        w_iptm=a.w_iptm, w_thermo=a.w_thermo,
        iptm_threshold=a.iptm_threshold,
        af2_work_dir=af2_bwork,
        colabfold_batch_bin=a.colabfold_batch_bin,
        colabfold_python=a.colabfold_python,
        af2_num_recycles=a.af2_num_recycles_beam,
        af2_num_models=a.af2_num_models,
        use_gpu=True, device=device,
    )

    # Weight paths forwarded to each worker for local model loading
    worker_model_kw = dict(
        mpnn_weights=a.mpnn_weights,
        thermo_local_yaml=a.thermo_local_yaml,
        thermo_model_yaml=a.thermo_model_yaml,
        thermo_checkpoint=a.thermo_checkpoint,
    )

    rank_fn      = RANKING_MODES[a.ranking_mode]
    node_counter = 0

    def _run_rollouts_parallel(parents, branch_factor, checkpoint_idx):
        """Distribute N×L rollouts across gpu_ids, return all candidates."""
        nonlocal node_counter
        tasks = [
            (parent, b, node_counter + i)
            for i, (parent, b) in enumerate(
                [(p, b) for p in parents for b in range(branch_factor)]
            )
        ]
        node_counter += len(tasks)

        candidates = []
        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as exe:
            futures = {}
            for task_i, (parent, b, nc) in enumerate(tasks):
                gpu_id = gpu_ids[task_i % len(gpu_ids)]
                fut = exe.submit(
                    _beam_rollout_worker,
                    gpu_id, parent, b, nc, checkpoint_idx,
                    rollout_kw, **worker_model_kw,
                )
                futures[fut] = gpu_id
            for fut in as_completed(futures):
                node = fut.result()
                if node is not None:
                    candidates.append(node)
        return candidates

    # Initialise beam
    n_seeds = a.beam_width * a.branch_factor
    root    = BeamNode(idx=-1, pdb_path=renumbered_pdb,
                       parent_idx=None, checkpoint_born=-1)
    print(f"  [Arm {arm}] Initialising beam: {n_seeds} seeds across "
          f"{len(gpu_ids)} GPU(s)…", flush=True)
    seeds = _run_rollouts_parallel([root], n_seeds, checkpoint_idx=0)
    seeds.sort(key=rank_fn, reverse=True)
    beam  = seeds[:a.beam_width]

    # Beam checkpoints
    for cp in range(1, a.n_checkpoints + 1):
        print(f"  [Arm {arm}] Checkpoint {cp}/{a.n_checkpoints} — "
              f"{len(beam) * a.branch_factor} rollout(s) across "
              f"{len(gpu_ids)} GPU(s)…", flush=True)
        candidates = _run_rollouts_parallel(beam, a.branch_factor,
                                            checkpoint_idx=cp)
        if candidates:
            candidates.sort(key=rank_fn, reverse=True)
            beam = candidates if cp == a.n_checkpoints \
                   else candidates[:a.beam_width]

    total_gpu_s = sum(
        sum(h.get("gpu_seconds", 0) for h in n.reward_history)
        for n in beam
    ) / max(len(beam), 1)

    for node in beam:
        pdb = node.pdb_path
        if arm == "D":
            stem_inp  = Path(input_pdb).stem
            arm_dir_d = os.path.join(output_dir, "arm_D_beam_anchored")
            masked_pb = os.path.join(arm_dir_d,
                                     f"{stem_inp}_anchors_masked.pdb")
            if not os.path.exists(masked_pb):
                mask_anchors_in_hlt(input_pdb, anchor_residues, masked_pb)
            grafted = pdb.replace(".pdb", "_grafted.pdb")
            graft_target_sequence(
                rfdiffusion_pdb=pdb, original_target=input_pdb,
                input_pdb=masked_pb, out_path=grafted,
            )
            pdb = grafted

        did = f"{arm.lower()}_node{node.idx:04d}"
        generated_queue.put(GeneratedDesign(
            arm=arm, design_id=did, pdb_path=pdb,
            gpu_seconds=total_gpu_s,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Top-level arm worker (one per arm group, runs in its own process)
# ─────────────────────────────────────────────────────────────────────────────

def _arm_worker(
    arm:             str,
    gpu_ids:         List[str],
    args_dict:       dict,
    generated_queue: mp.Queue,
):
    """
    Entry point for each arm subprocess.  Dispatches to the appropriate
    multi-GPU generation function.  On SIGTERM, stops generating and exits;
    structures already queued are preserved for evaluation.
    """
    import signal

    def _sigterm_handler(signum, frame):
        print(f"[Arm {arm}] SIGTERM — stopping generation.", flush=True)
        generated_queue.put((arm, None))   # sentinel
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    print(f"[Arm {arm}] Starting on GPU(s) {gpu_ids} (PID {os.getpid()})",
          flush=True)

    try:
        a          = argparse.Namespace(**args_dict)
        free_loops = parse_free_loops(a.free_loops)
        input_pdb  = str(Path(a.input).resolve())
        output_dir = str(Path(a.output_dir).resolve())
        nanobody   = a.nanobody
        extra      = [x for x in (a.extra or []) if x != "--"]

        split_dir           = os.path.join(output_dir, f"_split_arm{arm}")
        target_pdb, fwk_pdb = split_hlt_complex(input_pdb, split_dir)

        if not nanobody:
            residues = read_pdb_residues(input_pdb)
            if not any(r.pdb_chain == CHAIN_L for r in residues):
                nanobody = True

        anchor_residues = []
        if arm in ("B", "D") and a.anchors:
            anchor_residues = load_anchors(str(Path(a.anchors).resolve()))

        if arm in ("A", "B"):
            _generate_arm_AB_parallel(
                arm=arm, a=a,
                input_pdb=input_pdb, output_dir=output_dir,
                framework_pdb=fwk_pdb, free_loops=free_loops,
                nanobody=nanobody, extra=extra,
                gpu_ids=gpu_ids,
                generated_queue=generated_queue,
            )
        elif arm in ("C", "D"):
            _generate_arm_beam_parallel(
                arm=arm, a=a,
                input_pdb=input_pdb, output_dir=output_dir,
                framework_pdb=fwk_pdb, free_loops=free_loops,
                nanobody=nanobody, extra=extra,
                gpu_ids=gpu_ids,
                generated_queue=generated_queue,
                anchor_residues=anchor_residues,
            )

        print(f"[Arm {arm}] Generation complete.", flush=True)

    except SystemExit:
        pass
    except Exception:
        import traceback
        print(f"[Arm {arm}] ERROR:\n{traceback.format_exc()}", flush=True)

    generated_queue.put((arm, None))   # sentinel


def _gpu_group_worker(
    arms:            List[str],
    gpu_ids:         List[str],
    args_dict:       dict,
    generated_queue: mp.Queue,
):
    """Runs a list of arms sequentially, all sharing the same GPU pool."""
    for arm in arms:
        _arm_worker(arm, gpu_ids, args_dict, generated_queue)


# ─────────────────────────────────────────────────────────────────────────────
# Post-budget evaluation: ColabFold ipTM + DockQ on all generated structures
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_generated(
    generated:           List[GeneratedDesign],
    native_pdb:          str,
    colabfold_batch_bin: str,
    colabfold_python:    str,
    af2_work_dir:        str,
    dockq_bin:           str,
    af2_num_recycles:    int,
    af2_num_models:      int,
    device:              str,
) -> List[DesignResult]:
    """
    Run ColabFold ipTM + DockQ on every GeneratedDesign and return
    List[DesignResult].  Runs in the main process after all generation
    workers have finished or been terminated.
    """
    results = []
    total   = len(generated)
    print(f"\n[Eval] Evaluating {total} generated structure(s)…")

    for i, g in enumerate(generated, 1):
        print(f"  [{i:>4}/{total}] {g.design_id}", flush=True)
        af2_timer = GPUTimer()
        iptm, dockq = score_design(
            pdb_path=g.pdb_path,
            af2_work_dir=af2_work_dir,
            native_pdb=native_pdb,
            colabfold_batch_bin=colabfold_batch_bin,
            colabfold_python=colabfold_python,
            timer=af2_timer,
            af2_num_recycles=af2_num_recycles,
            af2_num_models=af2_num_models,
            device=device,
        )
        r = DesignResult(
            arm=g.arm,
            design_id=g.design_id,
            pdb_path=g.pdb_path,
            iptm=iptm,
            dockq=dockq,
            ddg=None,
            gpu_seconds=g.gpu_seconds + af2_timer.total_seconds,
        )
        results.append(r)
        tag    = "✓ SUCCESS" if r.success else "✗"
        iptm_s = f"{iptm:.3f}"  if iptm  is not None else "NA"
        dq_s   = f"{dockq:.3f}" if dockq is not None else "NA"
        print(f"         ipTM={iptm_s}  DockQ={dq_s}  {tag}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# GPU group scheduler + watchdog
# ─────────────────────────────────────────────────────────────────────────────

def run_parallel(
    arms:          List[str],
    gpu_map:       Dict[str, str],
    args_dict:     dict,
    max_gpu_hours: Optional[float] = None,
) -> Dict[str, List[DesignResult]]:
    """
    Launch generation workers, apply optional GPU budget watchdog, collect
    all generated structures, then run evaluation in the main process.
    """
    from collections import defaultdict
    gpu_groups: Dict[str, List[str]] = defaultdict(list)
    for arm in arms:
        gpu_groups[gpu_map[arm]].append(arm)

    print("\n[Parallel] GPU assignment:")
    for gpu_ids_str, group_arms in sorted(gpu_groups.items()):
        gpu_ids = _parse_gpu_list(gpu_ids_str)
        mode    = "concurrent" if len(group_arms) == 1 \
                  else "sequential within group"
        print(f"  GPU(s) {gpu_ids}: arms {group_arms}  "
              f"({len(gpu_ids)} GPU(s) per arm, {mode})")
    if max_gpu_hours is not None:
        print(f"[Parallel] Generation budget: {max_gpu_hours:.2f} GPU-hour(s)")
    else:
        print("[Parallel] No GPU budget — generation runs to completion")

    generated_queue: mp.Queue = mp.Queue()

    processes = []
    for gpu_ids_str, group_arms in gpu_groups.items():
        gpu_ids = _parse_gpu_list(gpu_ids_str)
        p = mp.Process(
            target=_gpu_group_worker,
            args=(group_arms, gpu_ids, args_dict, generated_queue),
            daemon=False,
        )
        p.start()
        processes.append(p)
        print(f"[Parallel] Launched PID {p.pid} for GPU(s) {gpu_ids} "
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
                    f"\n[Parallel] ⏰ Generation budget ({max_gpu_hours:.2f} h) "
                    f"reached after {elapsed/3600:.3f} h — terminating workers.",
                    flush=True,
                )
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                return

    watchdog = threading.Thread(target=_watchdog, daemon=True)
    watchdog.start()

    # ── Collect generated structures ──────────────────────────────────────────
    # Workers put GeneratedDesign items as structures finish, and a
    # (arm, None) sentinel when they exit.  Drain until all sentinels arrive
    # or all processes die.
    all_generated: List[GeneratedDesign] = []
    sentinels_expected = len(arms)
    sentinels_received = 0

    while sentinels_received < sentinels_expected and \
          any(p.is_alive() for p in processes):
        try:
            item = generated_queue.get(timeout=5)
        except Exception:
            continue

        if isinstance(item, GeneratedDesign):
            all_generated.append(item)
            print(f"[Parallel] Generated: [{item.arm}] {item.design_id}",
                  flush=True)
        elif isinstance(item, tuple) and item[1] is None:
            sentinels_received += 1
            print(f"[Parallel] Arm {item[0]} generation finished.", flush=True)

    # Final drain
    while not generated_queue.empty():
        try:
            item = generated_queue.get_nowait()
            if isinstance(item, GeneratedDesign):
                all_generated.append(item)
        except Exception:
            break

    stop_event.set()
    watchdog.join(timeout=5)
    for p in processes:
        p.join()

    print(f"\n[Parallel] Generation phase complete. "
          f"{len(all_generated)} structure(s) collected.")

    # ── Post-budget evaluation ────────────────────────────────────────────────
    a            = argparse.Namespace(**args_dict)
    output_dir   = str(Path(a.output_dir).resolve())
    af2_work_dir = os.path.join(output_dir, "_af2_eval")
    os.makedirs(af2_work_dir, exist_ok=True)

    eval_results = _evaluate_generated(
        generated=all_generated,
        native_pdb=str(Path(a.native).resolve()),
        colabfold_batch_bin=a.colabfold_batch_bin,
        colabfold_python=a.colabfold_python,
        af2_work_dir=af2_work_dir,
        dockq_bin=a.dockq_bin,
        af2_num_recycles=a.af2_num_recycles_eval,
        af2_num_models=a.af2_num_models,
        device=getattr(a, "device", "cuda"),
    )

    all_results: Dict[str, List[DesignResult]] = {arm: [] for arm in arms}
    for r in eval_results:
        all_results[r.arm].append(r)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Multi-GPU parallel RFantibody benchmark.\n"
            "GPUs listed per arm are used in parallel within that arm;\n"
            "different arm groups run concurrently."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── required ─────────────────────────────────────────────────────────────
    p.add_argument("--input",               required=True)
    p.add_argument("--native",              required=True)
    p.add_argument("--output_dir",          required=True)
    p.add_argument("--hotspots",            required=True)
    p.add_argument("--model_weights",       required=True)
    p.add_argument("--colabfold_batch_bin", required=True)
    p.add_argument("--colabfold_python",    required=True)
    p.add_argument("--thermo_local_yaml",   required=True)
    p.add_argument("--thermo_model_yaml",   required=True)
    p.add_argument("--thermo_checkpoint",   required=True)
    p.add_argument("--mpnn_weights",        required=True)
    p.add_argument("--dockq_bin",           default="DockQ")
    p.add_argument("--anchors",
                   help="Required for arms B and D")
    # ── parallelism ──────────────────────────────────────────────────────────
    p.add_argument(
        "--gpu_map", default="",
        help=(
            "ARM:GPU_IDs assignments, e.g. 'A:0,1,2,3,C:4,5,6,7'. "
            "GPUs listed per arm are used in parallel within that arm. "
            "Omit to place all arms on GPU 0."
        ),
    )
    p.add_argument(
        "--max_gpu_hours", type=float, default=None,
        help=(
            "Optional wall-clock budget in GPU-hours for the generation "
            "phase.  When reached, workers are terminated and all structures "
            "generated so far are evaluated (default: no limit)."
        ),
    )
    # ── arm selection ─────────────────────────────────────────────────────────
    p.add_argument("--arms",          default="A,B,C,D")
    p.add_argument("--num_designs",   type=int,   default=50)
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
    p.add_argument("--af2_num_models",        type=int, default=1)
    # ── other ─────────────────────────────────────────────────────────────────
    p.add_argument("--free_loops", default="")
    p.add_argument("--nanobody",   action="store_true")
    p.add_argument("--name",       default="")
    p.add_argument("--device",     default="cuda")
    p.add_argument("extra",        nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    mp.set_start_method("spawn", force=True)

    args = parse_args()
    arms = [a.strip().upper() for a in args.arms.split(",")]

    if any(arm in ("B", "D") for arm in arms) and not args.anchors:
        sys.exit("[ERROR] --anchors is required when running arms B or D.")

    output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)
    stem = args.name or Path(args.input).stem

    try:
        gpu_map = parse_gpu_map(args.gpu_map, arms)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    args_dict = vars(args)

    wall_t0      = time.perf_counter()
    all_results  = run_parallel(arms, gpu_map, args_dict,
                                max_gpu_hours=args.max_gpu_hours)
    wall_elapsed = time.perf_counter() - wall_t0

    print(f"\n[Parallel] All arms complete. "
          f"Wall-clock time: {wall_elapsed/3600:.2f} h")

    summaries = summarise(all_results)
    print_report(summaries)
    save_results(summaries, output_dir, stem)


if __name__ == "__main__":
    main()