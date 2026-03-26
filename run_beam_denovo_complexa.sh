#!/usr/bin/env bash
# run_beam_denovo_complexa.sh
#
# Runs beam_denovo_maturation_complexa.py on a remote server and keeps it
# alive after you disconnect using nohup + output logging.
#
# Usage:
#   bash run_beam_denovo_complexa.sh          # start the job
#   bash run_beam_denovo_complexa.sh status   # check if running
#   bash run_beam_denovo_complexa.sh log      # tail the live log
#   bash run_beam_denovo_complexa.sh stop     # kill the job

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Paths to your cloned repos
export RFANTIBODY_ROOT="/home/pymc/Deepak/RFantibody_partialflow"
export THERMOMPNN_ROOT="/home/pymc/Deepak/RFantibody_partialflow/ThermoMPNN"

# Python interpreter inside the RFantibody venv
PYTHON="${RFANTIBODY_ROOT}/.venv/bin/python"

# Script location
SCRIPT="${RFANTIBODY_ROOT}/beam_denovo_maturation_complexa.py"

# ── Required inputs ───────────────────────────────────────────────────────────
INPUT_PDB="/home/pymc/Deepak/RFantibody_partialflow/scripts/examples/example_inputs/1n8z_hlt.pdb"
ANCHORS_JSON="/home/pymc/Deepak/RFantibody_partialflow/1n8z_anchors/1n8z_hlt_anchors.json"
OUTPUT_DIR="/home/pymc/Deepak/RFantibody_partialflow/1n8z_cbeam_denovo_width16_cp4"
HOTSPOTS="T570,T571,T572,T573"
MODEL_WEIGHTS="${RFANTIBODY_ROOT}/weights/RFdiffusion_Ab.pt"
MPNN_WEIGHTS="${THERMOMPNN_ROOT}/vanilla_model_weights/v_48_020.pt"

# ── ThermoMPNN config ─────────────────────────────────────────────────────────
THERMO_LOCAL_YAML="${THERMOMPNN_ROOT}/local.yaml"
THERMO_MODEL_YAML="${THERMOMPNN_ROOT}/config.yaml"
THERMO_CHECKPOINT="${THERMOMPNN_ROOT}/models/thermoMPNN_default.pt"

# ── ColabFold / AF2-Multimer (for ipAE reward) ────────────────────────────────
# Set these to enable the full Complexa reward (ipAE + ThermoMPNN).
# Leave COLABFOLD_BATCH_BIN empty to run ThermoMPNN-only scoring.
COLABFOLD_BATCH_BIN="/home/pymc/miniconda3/envs/colabfold_env/bin/colabfold_batch"
COLABFOLD_PYTHON="/home/pymc/miniconda3/envs/colabfold_env/bin/python"
AF2_NUM_RECYCLES=1    # low recycle count keeps beam rollouts fast
AF2_NUM_MODELS=1

# ── Beam search hyperparameters ───────────────────────────────────────────────
BEAM_WIDTH=4          # N: survivors kept after each checkpoint prune
BRANCH_FACTOR=4       # L: rollouts launched per survivor per checkpoint
N_CHECKPOINTS=4       # number of expand-score-prune cycles
STEPS_PER_CHECKPOINT=1
RANKING_MODE="cumulative"   # cumulative | latest | average

# ── Reward weights ────────────────────────────────────────────────────────────
W_IPAE=1.0            # weight on ipAE reward component
W_THERMO=0.5          # weight on ThermoMPNN DDG component
IPAE_THRESHOLD=7.0    # success criterion: ipAE < threshold (Å)

# ── Optional ──────────────────────────────────────────────────────────────────
FREE_LOOPS=""          # e.g. "H3:5-13" or leave empty
NANOBODY_FLAG=""       # set to "--nanobody" for nanobody design
DEVICE="cuda"
RUN_NAME="cbeam_run_1n8z"

# ─────────────────────────────────────────────────────────────────────────────
# Internal — do not edit below here
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}.log"
PID_FILE="${OUTPUT_DIR}/${RUN_NAME}.pid"

CMD=(
    "$PYTHON" "$SCRIPT"
    --input               "$INPUT_PDB"
    --anchors             "$ANCHORS_JSON"
    --output_dir          "$OUTPUT_DIR"
    --hotspots            "$HOTSPOTS"
    --model_weights       "$MODEL_WEIGHTS"
    --mpnn_weights        "$MPNN_WEIGHTS"
    --thermo_local_yaml   "$THERMO_LOCAL_YAML"
    --thermo_model_yaml   "$THERMO_MODEL_YAML"
    --thermo_checkpoint   "$THERMO_CHECKPOINT"
    --beam_width          "$BEAM_WIDTH"
    --branch_factor       "$BRANCH_FACTOR"
    --n_checkpoints       "$N_CHECKPOINTS"
    --steps_per_checkpoint "$STEPS_PER_CHECKPOINT"
    --ranking_mode        "$RANKING_MODE"
    --w_ipae              "$W_IPAE"
    --w_thermo            "$W_THERMO"
    --ipae_threshold      "$IPAE_THRESHOLD"
    --af2_num_recycles    "$AF2_NUM_RECYCLES"
    --af2_num_models      "$AF2_NUM_MODELS"
    --device              "$DEVICE"
    --name                "$RUN_NAME"
)

# Append colabfold paths only if the binary exists
if [[ -n "$COLABFOLD_BATCH_BIN" && -f "$COLABFOLD_BATCH_BIN" ]]; then
    CMD+=(
        --colabfold_batch_bin "$COLABFOLD_BATCH_BIN"
        --colabfold_python    "$COLABFOLD_PYTHON"
    )
else
    echo "[warn] COLABFOLD_BATCH_BIN not set or not found — ipAE reward will be disabled."
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

    echo "[start] Launching Complexa-style beam search de novo maturation..."
    echo "[start] Log → $LOG_FILE"
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