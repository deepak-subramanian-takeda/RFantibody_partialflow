#!/usr/bin/env bash
# run_rfantibody_benchmark.sh
#
# Runs rfantibody_benchmark.py on a remote server and keeps it alive after
# you disconnect using nohup + output logging.
#
# Usage:
#   bash run_rfantibody_benchmark.sh          # start the job
#   bash run_rfantibody_benchmark.sh status   # check if running
#   bash run_rfantibody_benchmark.sh log      # tail the live log
#   bash run_rfantibody_benchmark.sh stop     # kill the job

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Paths to your cloned repos
export RFANTIBODY_ROOT="/home/sagemaker-user/RFantibody_partialflow"
export THERMOMPNN_ROOT="/home/sagemaker-user/RFantibody_partialflow/ThermoMPNN"

# Python interpreter inside the RFantibody venv
PYTHON="${RFANTIBODY_ROOT}/.venv/bin/python"

# Script location
SCRIPT="${RFANTIBODY_ROOT}/rfantibody_benchmark.py"

# ── Required inputs ───────────────────────────────────────────────────────────
INPUT_PDB="/home/sagemaker-user/RFantibody_partialflow/scripts/examples/example_inputs/1n8z_hlt.pdb"
# Native/reference PDB for DockQ scoring.
# If benchmarking redesigns of a known binder, this is often the same as
# INPUT_PDB.  Set to a different path if your input was pre-processed
# (renumbered, chains renamed, etc.) relative to the deposited structure.
NATIVE_PDB="/home/sagemaker-user/RFantibody_partialflow/scripts/examples/example_inputs/1n8z_hlt.pdb"
ANCHORS_JSON="/home/sagemaker-user/RFantibody_partialflow/1n8z_anchors/1n8z_hlt_anchors.json"
OUTPUT_DIR="/home/sagemaker-user/RFantibody_partialflow/1n8z_benchmark"
HOTSPOTS="T570,T571,T572,T573"
MODEL_WEIGHTS="${RFANTIBODY_ROOT}/weights/RFdiffusion_Ab.pt"

# ── ThermoMPNN config ─────────────────────────────────────────────────────────
THERMO_LOCAL_YAML="${THERMOMPNN_ROOT}/local.yaml"
THERMO_MODEL_YAML="${THERMOMPNN_ROOT}/config.yaml"
THERMO_CHECKPOINT="${THERMOMPNN_ROOT}/models/thermoMPNN_default.pt"
MPNN_WEIGHTS="${THERMOMPNN_ROOT}/vanilla_model_weights/v_48_020.pt"

# ── ColabFold / AF2-Multimer ──────────────────────────────────────────────────
# Required for ipTM scoring in all arms.
# Leave COLABFOLD_BATCH_BIN empty to fall back to ThermoMPNN-only scoring
# (arms A and B will still run; arms C and D will have degraded beam rewards).
COLABFOLD_BATCH_BIN="/home/sagemaker-user/.conda/envs/colabfold/bin/colabfold_batch"
COLABFOLD_PYTHON="/home/sagemaker-user/.conda/envs/colabfold/bin/python"
# Higher recycles for final success evaluation (arms A, B, and final eval of C, D)
AF2_NUM_RECYCLES_EVAL=3
# Lower recycles during beam rollout scoring (arms C and D internal scoring)
AF2_NUM_RECYCLES_BEAM=1
AF2_NUM_MODELS=1

# ── DockQ ─────────────────────────────────────────────────────────────────────
# DockQ must be installed and on PATH, or set an absolute path here.
DOCKQ_BIN="DockQ"

# ── Arms to run ───────────────────────────────────────────────────────────────
# A = Vanilla RFantibody
# B = Anchored (partial_T=50 + provide_seq)
# C = Beam search, no anchors
# D = Beam search + anchors
# Remove any arm you don't want to run.
ARMS="A,B,C,D"

# ── Arms A and B: number of designs per run ───────────────────────────────────
NUM_DESIGNS=50

# ── Beam search hyperparameters (arms C and D) ────────────────────────────────
BEAM_WIDTH=4           # N: survivors kept after each checkpoint prune
BRANCH_FACTOR=4        # L: rollouts launched per survivor per checkpoint
N_CHECKPOINTS=4        # number of expand-score-prune cycles
RANKING_MODE="cumulative"   # cumulative | latest | average

# ── Reward weights (arms C and D) ────────────────────────────────────────────
W_IPTM=2.0             # weight on ipTM reward component
W_THERMO=0.5           # weight on ThermoMPNN DDG component
IPTM_THRESHOLD=0.6     # beam success criterion; also used for final reporting

# ── Optional ──────────────────────────────────────────────────────────────────
FREE_LOOPS=""           # e.g. "H3:5-13,L3:4-10" or leave empty
NANOBODY_FLAG=""        # set to "--nanobody" for nanobody design
DEVICE="cuda"
RUN_NAME="benchmark_1n8z"

# ─────────────────────────────────────────────────────────────────────────────
# Internal — do not edit below here
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}.log"
PID_FILE="${OUTPUT_DIR}/${RUN_NAME}.pid"

CMD=(
    "$PYTHON" "$SCRIPT"
    --input                  "$INPUT_PDB"
    --native                 "$NATIVE_PDB"
    --anchors                "$ANCHORS_JSON"
    --output_dir             "$OUTPUT_DIR"
    --hotspots               "$HOTSPOTS"
    --model_weights          "$MODEL_WEIGHTS"
    --mpnn_weights           "$MPNN_WEIGHTS"
    --thermo_local_yaml      "$THERMO_LOCAL_YAML"
    --thermo_model_yaml      "$THERMO_MODEL_YAML"
    --thermo_checkpoint      "$THERMO_CHECKPOINT"
    --dockq_bin              "$DOCKQ_BIN"
    --arms                   "$ARMS"
    --num_designs            "$NUM_DESIGNS"
    --beam_width             "$BEAM_WIDTH"
    --branch_factor          "$BRANCH_FACTOR"
    --n_checkpoints          "$N_CHECKPOINTS"
    --ranking_mode           "$RANKING_MODE"
    --w_iptm                 "$W_IPTM"
    --w_thermo               "$W_THERMO"
    --iptm_threshold         "$IPTM_THRESHOLD"
    --af2_num_recycles_beam  "$AF2_NUM_RECYCLES_BEAM"
    --af2_num_recycles_eval  "$AF2_NUM_RECYCLES_EVAL"
    --af2_num_models         "$AF2_NUM_MODELS"
    --device                 "$DEVICE"
    --name                   "$RUN_NAME"
)

# Append colabfold paths only if the binary exists
if [[ -n "$COLABFOLD_BATCH_BIN" && -f "$COLABFOLD_BATCH_BIN" ]]; then
    CMD+=(
        --colabfold_batch_bin "$COLABFOLD_BATCH_BIN"
        --colabfold_python    "$COLABFOLD_PYTHON"
    )
else
    echo "[warn] COLABFOLD_BATCH_BIN not set or not found — ipTM scoring will be disabled."
fi

# Append optional flags only if set
[[ -n "$FREE_LOOPS"    ]] && CMD+=(--free_loops    "$FREE_LOOPS")
[[ -n "$NANOBODY_FLAG" ]] && CMD+=("$NANOBODY_FLAG")

# ── Subcommands ───────────────────────────────────────────────────────────────

status() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[status] Job is running (PID $PID)"
            echo "[status] Log: $LOG_FILE"
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
            kill "$PID"
            echo "[stop] Sent SIGTERM to PID $PID"
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

    echo "[start] Launching RFantibody benchmark (arms: ${ARMS})..."
    echo "[start] Input  : $INPUT_PDB"
    echo "[start] Native : $NATIVE_PDB"
    echo "[start] Output : $OUTPUT_DIR"
    echo "[start] Log    → $LOG_FILE"
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