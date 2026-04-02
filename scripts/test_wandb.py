"""Quick test to verify W&B is working."""

import wandb
import time

print("Testing W&B integration...")
print("-" * 50)

# Initialize a test run
run = wandb.init(
    project="n-cmapss-test",
    name="wandb_test",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "test",
    }
)

print(f"✅ W&B initialized successfully!")
print(f"   Project: {run.project}")
print(f"   Run name: {run.name}")
print(f"   Run URL: {run.url}")
print("-" * 50)

# Log some fake metrics
for epoch in range(5):
    wandb.log({
        "epoch": epoch,
        "train_loss": 1.0 / (epoch + 1),
        "val_loss": 1.2 / (epoch + 1),
        "accuracy": 0.5 + (epoch * 0.08),
    })
    time.sleep(0.1)

print("\n✅ Test metrics logged successfully!")
print(f"\nView your dashboard: {run.url}")
print("-" * 50)

wandb.finish()

print("\n🎉 W&B is working perfectly!")
print(f"\nYou can now run the tuning experiments with full tracking.")
print(f"\nNext step: ./scripts/run_quick_wins_wandb.sh")
