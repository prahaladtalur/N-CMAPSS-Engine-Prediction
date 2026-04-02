#!/bin/bash
# W&B Setup Helper

echo "=========================================="
echo "Weights & Biases Setup"
echo "=========================================="
echo ""
echo "W&B tracks all your training experiments in one place."
echo ""
echo "Setup options:"
echo "  1. Create account: https://wandb.ai/signup"
echo "  2. Get API key: https://wandb.ai/authorize"
echo "  3. Login: wandb login"
echo ""
echo "Or to keep runs local only:"
echo "  export WANDB_MODE=offline"
echo ""
echo "=========================================="

# Check if wandb is logged in
if uv run python - <<'PY'
import wandb
raise SystemExit(0 if getattr(wandb.api, "api_key", None) else 1)
PY
then
    echo "✅ W&B is already configured!"
else
    echo "⚠️  W&B not configured yet"
    echo ""
    read -p "Do you want to login now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        uv run wandb login
    else
        echo "Skipping W&B setup. Set WANDB_MODE=offline to keep tracking local."
    fi
fi
