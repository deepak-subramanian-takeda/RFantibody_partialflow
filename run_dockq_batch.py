"""
run_dockq_batch.py

Runs DockQ on every PDB file in a folder against a single native/reference PDB
and writes results to a TSV file.

Usage:
    python run_dockq_batch.py \
        --models_dir  /path/to/designed_pdbs/ \
        --native      /path/to/native.pdb \
        --output      results_dockq.tsv \
        --dockq_bin   /path/to/.venv/bin/DockQ \
        --mapping     HLT:HLT \
        --workers     4

Notes:
    - Files that DockQ cannot parse (wrong chains, no interface, etc.) are
      skipped and logged with a WARN rather than crashing the whole run.
    - --mapping is optional; if omitted DockQ v2 auto-detects chain mapping.
    - --workers controls parallel DockQ processes (default: 1, i.e. serial).
      Set to the number of CPU cores you want to use.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# DockQ runner
# ─────────────────────────────────────────────────────────────────────────────

def run_dockq_single(
    model_pdb:  str,
    native_pdb: str,
    dockq_bin:  str,
    mapping:    str,
    timeout:    int,
) -> dict:
    """
    Run DockQ on one model PDB and return a result dict.

    Returns:
        {
            "model":   filename,
            "dockq":   float or None,
            "fnat":    float or None,
            "irmsd":   float or None,
            "lrmsd":   float or None,
            "mapping": str or None,   # reported chain mapping
            "error":   str or None,
        }
    """
    name = Path(model_pdb).name
    cmd  = [dockq_bin, model_pdb, native_pdb, "--short"]
    if mapping:
        cmd += ["--mapping", mapping]

    result = {
        "model":   name,
        "dockq":   None,
        "fnat":    None,
        "irmsd":   None,
        "lrmsd":   None,
        "mapping": None,
        "error":   None,
    }

    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout
        )
    except subprocess.CalledProcessError as e:
        result["error"] = e.output.strip().splitlines()[-1] if e.output else str(e)
        return result
    except subprocess.TimeoutExpired:
        result["error"] = f"timed out after {timeout}s"
        return result
    except FileNotFoundError:
        result["error"] = f"DockQ binary not found: {dockq_bin}"
        return result

    # ── Parse --short output ──────────────────────────────────────────────────
    # DockQ v2 --short produces lines like:
    #   Total DockQ over N native interfaces: 0.XXXX with AB:AB model:native mapping
    #   DockQ 0.XXXX Fnat 0.XXXX iRMSD 0.XXXX LRMSD 0.XXXX mapping AB:AB ...
    #   GlobalDockQ 0.XXXX

    # GlobalDockQ / Total DockQ
    for pat in [
        r"Total DockQ[^:]*:\s*([0-9.]+)",
        r"GlobalDockQ\s+([0-9.]+)",
        r"^DockQ\s+([0-9.]+)",
    ]:
        m = re.search(pat, out, re.MULTILINE)
        if m:
            result["dockq"] = float(m.group(1))
            break

    # Per-interface metrics (from first interface line if multi-chain)
    m = re.search(r"Fnat\s+([0-9.]+)", out)
    if m:
        result["fnat"] = float(m.group(1))

    m = re.search(r"iRMSD\s+([0-9.]+)", out)
    if m:
        result["irmsd"] = float(m.group(1))

    m = re.search(r"LRMSD\s+([0-9.]+)", out)
    if m:
        result["lrmsd"] = float(m.group(1))

    # Reported mapping
    m = re.search(r"mapping\s+([A-Za-z]+:[A-Za-z]+)", out)
    if m:
        result["mapping"] = m.group(1)

    if result["dockq"] is None:
        result["error"] = f"Could not parse DockQ score. Output: {out[:200]}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_dockq_batch(
    models_dir: str,
    native_pdb: str,
    output_tsv: str,
    dockq_bin:  str  = "DockQ",
    mapping:    str  = "",
    workers:    int  = 1,
    timeout:    int  = 120,
            glob:       str  = "*_seq*.pdb",
):
    models_dir = str(Path(models_dir).resolve())
    native_pdb = str(Path(native_pdb).resolve())

    pdb_files = sorted(Path(models_dir).glob(glob))
    if not pdb_files:
        print(f"[WARN] No PDB files found in {models_dir} matching '{glob}'")
        return

    print(f"[DockQ batch] {len(pdb_files)} PDB file(s) found in {models_dir}")
    print(f"[DockQ batch] Native : {native_pdb}")
    print(f"[DockQ batch] Mapping: {mapping or '(auto)'}")
    print(f"[DockQ batch] Workers: {workers}")
    print(f"[DockQ batch] Output : {output_tsv}")

    results = []

    if workers <= 1:
        # Serial
        for i, pdb in enumerate(pdb_files, 1):
            r = run_dockq_single(str(pdb), native_pdb, dockq_bin, mapping, timeout)
            _print_result(i, len(pdb_files), r)
            results.append(r)
    else:
        # Parallel
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as exe:
            for pdb in pdb_files:
                fut = exe.submit(
                    run_dockq_single,
                    str(pdb), native_pdb, dockq_bin, mapping, timeout,
                )
                futures[fut] = pdb.name

            completed = 0
            for fut in as_completed(futures):
                completed += 1
                r = fut.result()
                _print_result(completed, len(pdb_files), r)
                results.append(r)

    # Sort by filename for reproducible output
    results.sort(key=lambda r: r["model"])

    # ── Write TSV ──────────────────────────────────────────────────────────────
    n_ok   = sum(1 for r in results if r["dockq"] is not None)
    n_fail = len(results) - n_ok

    os.makedirs(Path(output_tsv).parent, exist_ok=True)
    with open(output_tsv, "w") as fh:
        fh.write("model\tdockq\tfnat\tirmsd\tlrmsd\tmapping\terror\n")
        for r in results:
            fh.write(
                f"{r['model']}\t"
                f"{r['dockq'] if r['dockq'] is not None else 'NA'}\t"
                f"{r['fnat']  if r['fnat']  is not None else 'NA'}\t"
                f"{r['irmsd'] if r['irmsd'] is not None else 'NA'}\t"
                f"{r['lrmsd'] if r['lrmsd'] is not None else 'NA'}\t"
                f"{r['mapping'] or 'NA'}\t"
                f"{r['error']   or ''}\n"
            )

    print(f"\n[DockQ batch] Done. {n_ok} scored, {n_fail} failed/skipped.")
    print(f"[DockQ batch] Results written to: {output_tsv}")

    if n_ok:
        dockqs = [r["dockq"] for r in results if r["dockq"] is not None]
        import statistics
        print(f"[DockQ batch] DockQ  mean={statistics.mean(dockqs):.3f}  "
              f"stdev={statistics.stdev(dockqs) if len(dockqs) > 1 else 0:.3f}  "
              f"min={min(dockqs):.3f}  max={max(dockqs):.3f}")


def _print_result(i: int, total: int, r: dict):
    if r["error"] and r["dockq"] is None:
        print(f"  [{i:>4}/{total}] WARN  {r['model']:50s}  {r['error'][:80]}")
    else:
        dq = f"{r['dockq']:.3f}" if r["dockq"] is not None else "NA"
        print(f"  [{i:>4}/{total}]  {r['model']:50s}  DockQ={dq}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch DockQ scoring of all PDB files in a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--models_dir", required=True,
                   help="Directory containing model PDB files to score")
    p.add_argument("--native",     required=True,
                   help="Native/reference PDB to score against")
    p.add_argument("--output",     default="dockq_results.tsv",
                   help="Output TSV file (default: dockq_results.tsv)")
    p.add_argument("--dockq_bin",  default="DockQ",
                   help="Path to DockQ executable (default: DockQ on PATH)")
    p.add_argument("--mapping",    default="",
                   help="Chain mapping string, e.g. HLT:HLT "
                        "(default: auto-detect)")
    p.add_argument("--workers",    type=int, default=1,
                   help="Number of parallel DockQ processes (default: 1)")
    p.add_argument("--timeout",    type=int, default=120,
                   help="Per-structure timeout in seconds (default: 120)")
    p.add_argument("--glob",       default="*_seq*.pdb",
                   help="Glob pattern for PDB files (default: *_seq*.pdb)")
    return p.parse_args()


def main():
    args = parse_args()
    run_dockq_batch(
        models_dir=args.models_dir,
        native_pdb=args.native,
        output_tsv=args.output,
        dockq_bin=args.dockq_bin,
        mapping=args.mapping,
        workers=args.workers,
        timeout=args.timeout,
        glob=args.glob,
    )


if __name__ == "__main__":
    main()