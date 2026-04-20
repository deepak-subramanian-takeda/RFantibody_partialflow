#!/usr/bin/env bash
# run_score_backbones_parallel.sh
#
# Runs score_backbones_parallel.py on a remote server and keeps it alive
# after you disconnect using nohup + output logging.
#
# Usage:
#   bash run_score_backbones_parallel.sh          # start the job
#   bash run_score_backbones_parallel.sh status   # check if running
#   bash run_score_backbones_parallel.sh log      # tail the live log
#   bash run_score_backbones_parallel.sh stop     # kill the job

set -euo pipefail

export PYTHONPATH="/home/sagemaker-user/RFantibody_partialflow/ThermoMPNN:${PYTHONPATH:-}"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

export RFANTIBODY_ROOT="/home/sagemaker-user/RFantibody_partialflow"
export THERMOMPNN_ROOT="/home/sagemaker-user/RFantibody_partialflow/ThermoMPNN"

PYTHON="${RFANTIBODY_ROOT}/.venv/bin/python"
SCRIPT="${RFANTIBODY_ROOT}/score_backbones_parallel.py"

# ── Required inputs ───────────────────────────────────────────────────────────
# Folder containing backbone PDB files to score (e.g. RFdiffusion outputs).
# Files matching *_seq.pdb, *_grafted.pdb, *_traj.pdb are excluded automatically.
INPUT_DIR="/home/sagemaker-user/RFantibody_partialflow/IL1RAP_5I1A_parallel/arm_C_beam_no_anchor/_beam_work"
NATIVE_PDB="/home/sagemaker-user/RFantibody_partialflow/scripts/examples/example_inputs/IL1RAP_5I1A_hlt.pdb"
OUTPUT_DIR="/home/sagemaker-user/RFantibody_partialflow/IL1RAP_5I1A_parallel/arm_C_beam_no_anchor/arm_C_beam_no_anchor/generated_and_scored"

# ── ProteinMPNN / IgDesign weights ───────────────────────────────────────────
# Pass vanilla ProteinMPNN weights OR IgDesign fine-tuned weights here.
# Both .pt and .ckpt formats are handled automatically.
MPNN_WEIGHTS="${RFANTIBODY_ROOT}/igdesign/ckpts/igmpnn_acvr2b_holdout.ckpt"

# Optional: PDB to build the CDR mask from.
# If left empty, the first backbone found is used.
FRAMEWORK_PDB=""

# ── ColabFold / AF2 ───────────────────────────────────────────────────────────
COLABFOLD_BATCH_BIN="/home/sagemaker-user/.conda/envs/colabfold/bin/colabfold_batch"
COLABFOLD_PYTHON="/home/sagemaker-user/.conda/envs/colabfold/bin/python"
AF2_NUM_RECYCLES=3
AF2_NUM_MODELS=1

# ── DockQ ─────────────────────────────────────────────────────────────────────
DOCKQ_BIN="${RFANTIBODY_ROOT}/.venv/bin/DockQ"

# ── Sequence generation ───────────────────────────────────────────────────────
N_SEQS=25            # sequences generated per backbone
TEMPERATURE=0.2      # ProteinMPNN sampling temperature

# ── GPU assignment ────────────────────────────────────────────────────────────
# Comma-separated GPU IDs.  Backbone PDBs are split evenly across all listed
# GPUs and generation runs simultaneously on each.
# Examples:
#   All 8 GPUs:  GPU_IDS="0,1,2,3,4,5,6,7"
#   4 GPUs:      GPU_IDS="0,1,2,3"
#   Single GPU:  GPU_IDS="0"
GPU_IDS="0,1,2,3,4,5,6,7"

# ── GPU budget ────────────────────────────────────────────────────────────────
# Maximum wall-clock time (hours) for the GENERATION phase only.
# Evaluation (ColabFold + DockQ) always runs to completion afterwards.
# Set to "" for no limit.
MAX_GPU_HOURS=""

# ── Other ─────────────────────────────────────────────────────────────────────
DEVICE="cuda"
RUN_NAME="score_backbones_run"

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
    --n_seqs              "$N_SEQS"
    --mpnn_weights        "$MPNN_WEIGHTS"
    --colabfold_batch_bin "$COLABFOLD_BATCH_BIN"
    --colabfold_python    "$COLABFOLD_PYTHON"
    --dockq_bin           "$DOCKQ_BIN"
    --af2_num_recycles    "$AF2_NUM_RECYCLES"
    --af2_num_models      "$AF2_NUM_MODELS"
    --device              "$DEVICE"
    --temperature         "$TEMPERATURE"
    --gpu_ids             "$GPU_IDS"
)

[[ -n "$FRAMEWORK_PDB"  ]] && CMD+=(--framework_pdb  "$FRAMEWORK_PDB")
[[ -n "$MAX_GPU_HOURS"  ]] && CMD+=(--max_gpu_hours  "$MAX_GPU_HOURS")

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

    echo "[start] Launching parallel backbone scoring…"
    echo "[start] Input dir : $INPUT_DIR"
    echo "[start] Native    : $NATIVE_PDB"
    echo "[start] Output    : $OUTPUT_DIR"
    echo "[start] GPUs      : $GPU_IDS"
    echo "[start] Seqs/bb   : $N_SEQS"
    echo "[start] Budget    : ${MAX_GPU_HOURS:-unlimited} GPU-hours"
    echo "[start] Log       → $LOG_FILE"
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