#!/usr/bin/env bash
# run_evaluate_sequences.sh
#
# Runs evaluate_sequences.py on a remote server and keeps it alive
# after you disconnect using nohup + output logging.
#
# Usage:
#   bash run_evaluate_sequences.sh          # start the job
#   bash run_evaluate_sequences.sh status   # check if running
#   bash run_evaluate_sequences.sh log      # tail the live log
#   bash run_evaluate_sequences.sh stop     # kill the job

set -euo pipefail

export PYTHONPATH="/home/sagemaker-user/RFantibody_partialflow/ThermoMPNN:${PYTHONPATH:-}"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

export RFANTIBODY_ROOT="/home/sagemaker-user/RFantibody_partialflow"

PYTHON="${RFANTIBODY_ROOT}/.venv/bin/python"
SCRIPT="${RFANTIBODY_ROOT}/evaluate_sequences.py"

# ── Required inputs ───────────────────────────────────────────────────────────
# Folder (or folder of subfolders) containing *_seq*.pdb files to evaluate.
INPUT_DIR="/home/sagemaker-user/RFantibody_partialflow/sequences/"

# Native/reference PDB for DockQ scoring.
# If scoring redesigns of a known binder, this is often the same as the
# original HLT input PDB.
NATIVE_PDB="/home/sagemaker-user/RFantibody_partialflow/scripts/examples/example_inputs/1n8z_hlt.pdb"

OUTPUT_DIR="/home/sagemaker-user/RFantibody_partialflow/1n8z_evaluated"

# ── ColabFold / AF2 ───────────────────────────────────────────────────────────
# ColabFold runs in --msa-mode single_sequence to avoid MSA downloads and
# prevent "no space left on device" errors.
COLABFOLD_BATCH_BIN="/home/sagemaker-user/.conda/envs/colabfold/bin/colabfold_batch"
COLABFOLD_PYTHON="/home/sagemaker-user/.conda/envs/colabfold/bin/python"

# Lower recycle count is appropriate here since single_sequence mode is
# already a fast scoring pass.  Increase for higher-quality final evaluation.
AF2_NUM_RECYCLES=1
AF2_NUM_MODELS=1

# ── DockQ ─────────────────────────────────────────────────────────────────────
DOCKQ_BIN="${RFANTIBODY_ROOT}/.venv/bin/DockQ"

# ── GPU assignment ────────────────────────────────────────────────────────────
# Comma-separated GPU IDs.  Structures are split evenly across all listed
# GPUs and evaluation runs simultaneously on each.
# Examples:
#   All 8 GPUs:  GPU_IDS="0,1,2,3,4,5,6,7"
#   4 GPUs:      GPU_IDS="0,1,2,3"
#   Single GPU:  GPU_IDS="0"
GPU_IDS="0,1,2,3,4,5,6,7"

# ── Disk management ───────────────────────────────────────────────────────────
# By default, AF2 working directories are cleaned up after each structure
# is scored to keep disk usage bounded.  Set to "--no_cleanup" to keep them.
CLEANUP_FLAG=""   # leave empty for cleanup (default), set to "--no_cleanup" to keep

# ── Job name ──────────────────────────────────────────────────────────────────
RUN_NAME="evaluate_sequences_run"

# ─────────────────────────────────────────────────────────────────────────────
# Internal — do not edit below here
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}.log"
PID_FILE="${OUTPUT_DIR}/${RUN_NAME}.pid"

CMD=(
    "$PYTHON" "$SCRIPT"
    --input_dir           "$INPUT_DIR"
    --native              "$NATIVE_PDB"
    --output_dir          "$OUTPUT_DIR"
    --colabfold_batch_bin "$COLABFOLD_BATCH_BIN"
    --colabfold_python    "$COLABFOLD_PYTHON"
    --dockq_bin           "$DOCKQ_BIN"
    --af2_num_recycles    "$AF2_NUM_RECYCLES"
    --af2_num_models      "$AF2_NUM_MODELS"
    --gpu_ids             "$GPU_IDS"
)

[[ -n "$CLEANUP_FLAG" ]] && CMD+=("$CLEANUP_FLAG")

# ── Subcommands ───────────────────────────────────────────────────────────────

status() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[status] Job is running (PID $PID)"
            echo "[status] Log: $LOG_FILE"
            echo "[status] Child processes:"
            pgrep -P "$PID" | while read -r cpid; do
                echo "  PID $cpid: $(ps -p "$cpid" -o args= 2>/dev/null | cut -c1-100)"
            done
        else
            echo "[status] Job is NOT running (stale PID $PID)"
        fi
    else
        echo "[status] No PID file found — job may not have been started."
    fi
}

log() {
    if [[ -f "$LOG_FILE" ]]; then
        tail -f "$LOG_FILE"
    else
        echo "[log] Log file not found: $LOG_FILE"
    fi
}

stop() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill -- -"$PID" 2>/dev/null || kill "$PID"
            echo "[stop] Sent SIGTERM to PID $PID and its children"
            rm -f "$PID_FILE"
        else
            echo "[stop] Process $PID is not running"
            rm -f "$PID_FILE"
        fi
    else
        echo "[stop] No PID file found"
    fi
}

start() {
    mkdir -p "$OUTPUT_DIR"

    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[start] Job is already running (PID $PID). Use 'stop' first."
            exit 1
        fi
    fi

    echo "[start] Launching sequence evaluation…"
    echo "[start] Input dir  : $INPUT_DIR"
    echo "[start] Native PDB : $NATIVE_PDB"
    echo "[start] Output dir : $OUTPUT_DIR"
    echo "[start] GPUs       : $GPU_IDS"
    echo "[start] MSA mode   : single_sequence (no MSA download)"
    echo "[start] AF2 recycles: $AF2_NUM_RECYCLES"
    echo "[start] Cleanup    : ${CLEANUP_FLAG:-enabled (default)}"
    echo "[start] Log        → $LOG_FILE"
    echo "[start] Command:"
    printf "  %s\n" "${CMD[@]}"
    echo ""

    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    JOB_PID=$!
    echo "$JOB_PID" > "$PID_FILE"
    echo "[start] Started with PID $JOB_PID"
    echo "[start] Monitor with:  bash $0 log"
    echo "[start] Check status:  bash $0 status"
    echo "[start] Stop job:      bash $0 stop"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

case "${1:-start}" in
    start)  start  ;;
    status) status ;;
    log)    log    ;;
    stop)   stop   ;;
    *)
        echo "Usage: bash $0 [start|status|log|stop]"
        exit 1
        ;;
esac