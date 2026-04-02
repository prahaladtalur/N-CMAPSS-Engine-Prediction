#!/bin/bash
# Quick wins with full W&B tracking

echo "=========================================="
echo "SOTA Quick Wins - W&B Enabled"
echo "=========================================="
echo ""

# Check if W&B is configured
if ! uv run python - <<'PY'
import wandb
raise SystemExit(0 if getattr(wandb.api, "api_key", None) else 1)
PY
then
    echo "⚠️  W&B not configured!"
    echo "Please run: uv run wandb login"
    echo ""
    read -p "Continue without W&B? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please set up W&B first."
        exit 1
    fi
    export WANDB_MODE=offline
    echo "Running with W&B offline tracking..."
fi

echo "Model: MSTCN"
echo "Experiments: 5 variations"
echo "Expected time: 2-3 hours"
echo ""
echo "W&B Project: sota-tuning"
echo "Dashboard: https://wandb.ai/your-username/sota-tuning"
echo ""
echo "=========================================="
echo ""

# Create log file
LOG_FILE="tuning_results/quick_wins_wandb_$(date +%Y%m%d_%H%M%S).log"
mkdir -p tuning_results

echo "Starting experiment at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run with W&B
uv run python scripts/tune_for_sota.py --experiment quick_wins --model mstcn 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Experiment completed at $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: tuning_results/" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "W&B Dashboard: https://wandb.ai" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
