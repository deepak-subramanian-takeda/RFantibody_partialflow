#!/usr/bin/env bash
# run_rfantibody_benchmark_parallel.sh
#
# Runs rfantibody_benchmark_parallel.py on a remote server and keeps it
# alive after you disconnect using nohup + output logging.
#
# Usage:
#   bash run_rfantibody_benchmark_parallel.sh          # start the job
#   bash run_rfantibody_benchmark_parallel.sh status   # check if running
#   bash run_rfantibody_benchmark_parallel.sh log      # tail the live log
#   bash run_rfantibody_benchmark_parallel.sh stop     # kill the job

set -euo pipefail

export PYTHONPATH="/home/sagemaker-user/RFantibody_partialflow/ThermoMPNN:${PYTHONPATH:-}"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

export RFANTIBODY_ROOT="/home/sagemaker-user/RFantibody_partialflow"
export THERMOMPNN_ROOT="/home/sagemaker-user/RFantibody_partialflow/ThermoMPNN"

PYTHON="${RFANTIBODY_ROOT}/.venv/bin/python"
SCRIPT="${RFANTIBODY_ROOT}/rfantibody_benchmark_parallel.py"

# ── Required inputs ───────────────────────────────────────────────────────────
INPUT_PDB="/home/sagemaker-user/RFantibody_partialflow/scripts/examples/example_inputs/IL1RAP_5I1A_hlt.pdb"
NATIVE_PDB="/home/sagemaker-user/RFantibody_partialflow/scripts/examples/example_inputs/IL1RAP_5I1A_hlt.pdb"
ANCHORS_JSON="/home/sagemaker-user/RFantibody_partialflow/1n8z_anchors/1n8z_hlt_anchors.json"
OUTPUT_DIR="/home/sagemaker-user/RFantibody_partialflow/IL1RAP_5I1A_parallel_additionalA"
HOTSPOTS="T162,T165,T166,T170,T219,T287"
MODEL_WEIGHTS="${RFANTIBODY_ROOT}/weights/RFdiffusion_Ab.pt"
MPNN_WEIGHTS="${THERMOMPNN_ROOT}/vanilla_model_weights/v_48_020.pt"

# ── ThermoMPNN config ─────────────────────────────────────────────────────────
THERMO_LOCAL_YAML="${THERMOMPNN_ROOT}/local.yaml"
THERMO_MODEL_YAML="${THERMOMPNN_ROOT}/config.yaml"
THERMO_CHECKPOINT="${THERMOMPNN_ROOT}/models/thermoMPNN_default.pt"

# ── ColabFold / AF2-Multimer ──────────────────────────────────────────────────
COLABFOLD_BATCH_BIN="/home/sagemaker-user/.conda/envs/colabfold/bin/colabfold_batch"
COLABFOLD_PYTHON="/home/sagemaker-user/.conda/envs/colabfold/bin/python"
AF2_NUM_RECYCLES_EVAL=3
AF2_NUM_RECYCLES_BEAM=1
AF2_NUM_MODELS=1

# ── DockQ ─────────────────────────────────────────────────────────────────────
DOCKQ_BIN="/home/sagemaker-user/RFantibody_partialflow/.venv/bin/DockQ"

# ── Arms to run ───────────────────────────────────────────────────────────────
ARMS="A"

# ── GPU assignment ────────────────────────────────────────────────────────────
# Format: ARM:GPU_ID  (comma-separated, no spaces)
#
# Examples:
#   4 GPUs — fully parallel:
#     GPU_MAP="A:0,B:1,C:2,D:3"
#
#   2 GPUs — A+B share GPU 0 (sequential), C+D share GPU 1 (sequential),
#             the two GPU groups run concurrently:
#     GPU_MAP="A:0,B:0,C:1,D:1"
#
#   2 GPUs — A on GPU 0, C on GPU 1, skip B and D:
#     ARMS="A,C"
#     GPU_MAP="A:0,C:1"
#
#   1 GPU — all arms sequential (same as serial script):
#     GPU_MAP="A:0,B:0,C:0,D:0"
#
#   Arm A across two GPUs (CUDA_VISIBLE_DEVICES=0,1), C on GPU 2:
#     ARMS="A,C"
#     GPU_MAP="A:0,1,C:2"
GPU_MAP="A:0"

# ── Arms A and B: number of designs per run ───────────────────────────────────
NUM_DESIGNS=300

# ── Beam search hyperparameters (arms C and D) ────────────────────────────────
BEAM_WIDTH=10
BRANCH_FACTOR=10
N_CHECKPOINTS=2
RANKING_MODE="cumulative"

# ── Reward weights ────────────────────────────────────────────────────────────
W_IPTM=2.0
W_THERMO=0.5
IPTM_THRESHOLD=0.6
MAX_GPU_HOURS=21

# ── Optional ──────────────────────────────────────────────────────────────────
FREE_LOOPS=""
NANOBODY_FLAG=""
RUN_NAME="benchmark_parallel_il1rap_5i1a"

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
    --gpu_map                "$GPU_MAP"
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
    --name                   "$RUN_NAME"
)

[[ -n "$MAX_GPU_HOURS" ]] && CMD+=(--max_gpu_hours "$MAX_GPU_HOURS")

if [[ -n "$COLABFOLD_BATCH_BIN" && -f "$COLABFOLD_BATCH_BIN" ]]; then
    CMD+=(
        --colabfold_batch_bin "$COLABFOLD_BATCH_BIN"
        --colabfold_python    "$COLABFOLD_PYTHON"
    )
else
    echo "[warn] COLABFOLD_BATCH_BIN not found — ipTM scoring will be disabled."
fi

[[ -n "$FREE_LOOPS"    ]] && CMD+=(--free_loops    "$FREE_LOOPS")
[[ -n "$NANOBODY_FLAG" ]] && CMD+=("$NANOBODY_FLAG")

# ── Subcommands ───────────────────────────────────────────────────────────────

status() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[status] Job is running (PID $PID)"
            echo "[status] Log: $LOG_FILE"
            # Show child processes (one per GPU group)
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
            # Kill the process group to catch all spawned children
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

    echo "[start] Launching parallel RFantibody benchmark..."
    echo "[start] Arms   : ${ARMS}"
    echo "[start] GPU map: ${GPU_MAP}"
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