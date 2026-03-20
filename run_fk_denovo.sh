#!/usr/bin/env bash
# run_fk_denovo.sh
#
# Runs fk_denovo_maturation.py on a remote server and keeps it alive
# after you disconnect using nohup + output logging.
#
# Usage:
#   bash run_fk_denovo.sh          # start the job
#   bash run_fk_denovo.sh status   # check if running
#   bash run_fk_denovo.sh log      # tail the live log
#   bash run_fk_denovo.sh stop     # kill the job

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
SCRIPT="${RFANTIBODY_ROOT}/fk_denovo_maturation.py"

# ── Required inputs ───────────────────────────────────────────────────────────
INPUT_PDB="/home/pymc/Deepak/RFantibody_partialflow/scripts/examples/example_inputs/5y2l_hlt_B.pdb"          # HLT-formatted complex
ANCHORS_JSON="/home/pymc/Deepak/RFantibody_partialflow/5y2l_B_anchors/5y2l_hlt_B_anchors.json"      # Step 0 anchor output
OUTPUT_DIR="/home/pymc/Deepak/RFantibody_partialflow/5y2l_B_fk_denovo_traj16_rounds4"
HOTSPOTS="T49,T42,T53,T19,T38,T20"                           # e.g. "T305,T456"
MODEL_WEIGHTS="${RFANTIBODY_ROOT}/weights/RFdiffusion_Ab.pt"
MPNN_WEIGHTS="${THERMOMPNN_ROOT}/vanilla_model_weights/v_48_020.pt"

# ── ThermoMPNN config ─────────────────────────────────────────────────────────
THERMO_LOCAL_YAML="${THERMOMPNN_ROOT}/local.yaml"
THERMO_MODEL_YAML="${THERMOMPNN_ROOT}/config.yaml"
THERMO_CHECKPOINT="${THERMOMPNN_ROOT}/models/thermoMPNN_default.pt"

# ── FK hyperparameters ────────────────────────────────────────────────────────
N_TRAJECTORIES=16      # population size (all survive — no resampling)
N_ROUNDS=4
GUIDANCE_SCALE=1.0
ANNEALING="linear"     # linear | constant | geometric | reverse
N_OUTPUT=5             # designs drawn from final FK-weighted distribution
W_THERMO=1.0
W_BSA=0.5
SEED=42

# ── Optional ──────────────────────────────────────────────────────────────────
FREE_LOOPS=""          # e.g. "H3:5-13" or leave empty
NANOBODY_FLAG=""       # set to "--nanobody" for nanobody design
DEVICE="cuda"
RUN_NAME="fk_run_5y2l_B"      # used for log/pid file names

# ─────────────────────────────────────────────────────────────────────────────
# Internal — do not edit below here
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}.log"
PID_FILE="${OUTPUT_DIR}/${RUN_NAME}.pid"

CMD=(
    "$PYTHON" "$SCRIPT"
    --input             "$INPUT_PDB"
    --anchors           "$ANCHORS_JSON"
    --output_dir        "$OUTPUT_DIR"
    --hotspots          "$HOTSPOTS"
    --model_weights     "$MODEL_WEIGHTS"
    --mpnn_weights      "$MPNN_WEIGHTS"
    --thermo_local_yaml "$THERMO_LOCAL_YAML"
    --thermo_model_yaml "$THERMO_MODEL_YAML"
    --thermo_checkpoint "$THERMO_CHECKPOINT"
    --n_trajectories    "$N_TRAJECTORIES"
    --n_rounds          "$N_ROUNDS"
    --guidance_scale    "$GUIDANCE_SCALE"
    --annealing         "$ANNEALING"
    --n_output          "$N_OUTPUT"
    --w_thermo          "$W_THERMO"
    --w_bsa             "$W_BSA"
    --seed              "$SEED"
    --device            "$DEVICE"
    --name              "$RUN_NAME"
)

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

    echo "[start] Launching Feynman-Kac de novo maturation..."
    echo "[start] Log → $LOG_FILE"
    echo "[start] Command:"
    printf "  %s\n" "${CMD[@]}"
    echo ""

    # nohup keeps the process alive after disconnect.
    # stdout and stderr both go to the log file.
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