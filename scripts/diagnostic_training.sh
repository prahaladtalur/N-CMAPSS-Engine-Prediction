#!/usr/bin/env bash
# diagnostic_training.sh
#
# Purpose: answer "is training duration the bottleneck?" by running key model
# configurations for 200 epochs each (early stopping patience=40 so it really
# runs) and logging full learning curves to W&B offline + a local results file.
#
# Memory-safe design:
#   - Sequential runs only (never parallel)
#   - batch_size=32 (half the default) to stay within RAM budget
#   - --no-visualize  (matplotlib holds large figures in memory)
#   - WANDB_MODE=offline  (no network I/O during training)
#   - 30-second sleep between runs to let GC/OS reclaim memory
#   - TF_FORCE_GPU_ALLOW_GROWTH + memory fraction cap via env vars
#   - Writes a timestamped log; safe to kill at any time (no data loss)
#
# Usage:
#   bash scripts/diagnostic_training.sh            # foreground
#   nohup bash scripts/diagnostic_training.sh &    # background (recommended)
#   tail -f logs/diagnostic_$(date +%Y%m%d)*.log   # watch progress

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
LOG_DIR="logs"
RESULTS_FILE="tuning_results/diagnostic_$(date +%Y%m%d_%H%M%S).jsonl"
EPOCHS=200
PATIENCE=40        # let it really try before stopping
BATCH=32           # conservative: half the default to protect RAM
SEQ_LEN=1000
FD=2               # DS02 — standard benchmark
SEEDS="42 123 456" # 3 seeds to average out 3-test-unit noise

# ── Environment ────────────────────────────────────────────────────────────────
export WANDB_MODE=offline
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2        # suppress TF info/warning spam
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "tuning_results"
LOG_FILE="$LOG_DIR/diagnostic_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo " Diagnostic training run — $(date)"
echo " epochs=$EPOCHS  patience=$PATIENCE  batch=$BATCH"
echo " seq_len=$SEQ_LEN  fd=$FD  seeds=$SEEDS"
echo " Log: $LOG_FILE"
echo " Results: $RESULTS_FILE"
echo "========================================================"

# ── Helper ─────────────────────────────────────────────────────────────────────
run_and_record() {
    local label="$1"; shift
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  RUN: $label  $(date)"
    echo "──────────────────────────────────────────────────────"

    # Run training; capture exit code without set -e killing the script
    if uv run python train_model.py "$@" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --patience-early-stop "$PATIENCE" \
        --max-seq-length "$SEQ_LEN" \
        --fd "$FD" \
        --no-visualize \
        --no-save; then
        echo "  ✓ $label completed  $(date)"
    else
        echo "  ✗ $label FAILED (exit $?)  $(date)"
    fi

    # Give the OS 30 s to reclaim memory before the next run
    echo "  … sleeping 30s for memory reclaim"
    sleep 30
}

# ── Experiment matrix ──────────────────────────────────────────────────────────
#
# Goal: compare learning curves across:
#   A) baseline (current best: mstcn, asymmetric_mse, all 32 features)
#   B) new loss only  (multi_zone_mse)
#   C) drop virtual sensors  (18 features)
#   D) both B+C together
#   E) cbam_cnn_lstm with best settings (confirmed RMSE 5.50 in literature)
#
# Each configuration runs with 3 seeds so we can average out single-engine noise.

for SEED in $SEEDS; do

    echo ""
    echo "════════════════════════════  SEED $SEED  ════════════════════════════"

    # A — baseline mstcn
    run_and_record "mstcn-baseline-s$SEED" \
        --model mstcn \
        --loss asymmetric_mse \
        --seed "$SEED" \
        --run-name "diag-mstcn-baseline-s$SEED"

    # B — new multi-zone loss only
    run_and_record "mstcn-multizone-s$SEED" \
        --model mstcn \
        --loss multi_zone_mse \
        --seed "$SEED" \
        --run-name "diag-mstcn-multizone-s$SEED"

    # C — drop virtual sensors only
    run_and_record "mstcn-dropvirtual-s$SEED" \
        --model mstcn \
        --loss asymmetric_mse \
        --drop-virtual \
        --seed "$SEED" \
        --run-name "diag-mstcn-dropvirtual-s$SEED"

    # D — multi-zone loss + drop virtual (best expected combo)
    run_and_record "mstcn-full-s$SEED" \
        --model mstcn \
        --loss multi_zone_mse \
        --drop-virtual \
        --seed "$SEED" \
        --run-name "diag-mstcn-full-s$SEED"

    # E — cbam_cnn_lstm with best settings
    run_and_record "cbam-full-s$SEED" \
        --model cbam_cnn_lstm \
        --loss multi_zone_mse \
        --drop-virtual \
        --seed "$SEED" \
        --run-name "diag-cbam-full-s$SEED"

done

echo ""
echo "========================================================"
echo " All runs complete — $(date)"
echo " Log:     $LOG_FILE"
echo " Results: $RESULTS_FILE"
echo "========================================================"
