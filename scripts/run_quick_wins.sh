#!/bin/bash
# Quick wins experiment runner

echo "=========================================="
echo "Starting SOTA Quick Wins Experiments"
echo "=========================================="
echo ""
echo "Model: MSTCN (current best SOTA model)"
echo "Experiments: 5 variations"
echo "Expected time: 2-3 hours"
echo ""
echo "Tests:"
echo "  1. Baseline (epochs=50, batch=32, lr=0.001)"
echo "  2. Longer training (epochs=100)"
echo "  3. Smaller batch (batch=16)"
echo "  4. Higher LR (lr=0.002)"
echo "  5. Lower LR (lr=0.0005)"
echo ""
echo "=========================================="
echo ""

# Keep W&B local-only for quick experiments
export WANDB_MODE=offline

# Create log file
LOG_FILE="tuning_results/quick_wins_$(date +%Y%m%d_%H%M%S).log"
mkdir -p tuning_results

echo "Starting experiment at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the tuning script
WANDB_MODE=offline uv run python scripts/tune_for_sota.py --experiment quick_wins --model mstcn --offline 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=========================================="  | tee -a "$LOG_FILE"
echo "Experiment completed at $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: tuning_results/" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
