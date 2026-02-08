#!/bin/bash
set -e

export WANDB_MODE=offline

models=("transformer" "inception_lstm" "cnn_gru")

for model in "${models[@]}"; do
    echo "========================================"
    echo "Starting training: $model"
    echo "========================================"
    .venv/bin/python train_model.py \
        --model $model \
        --epochs 30 \
        --batch-size 32 \
        --max-seq-length 1000 \
        --run-name ${model}-production
    echo ""
    echo "✓ Completed: $model"
    echo ""
done

echo "========================================"
echo "ALL TRAINING COMPLETE!"
echo "========================================"
