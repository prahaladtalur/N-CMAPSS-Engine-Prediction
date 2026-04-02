#!/usr/bin/env python3
"""
List local training runs, checkpoints, and W&B run state.

This script is meant to answer:
- what ran locally
- when it ran
- which model it used
- what RMSE it achieved
- where its local files live
- whether the local W&B run has been synced
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - wandb normally brings PyYAML
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
WANDB_DIR = PROJECT_ROOT / "wandb"


@dataclass
class RunRecord:
    source: str
    started_at: Optional[datetime]
    model_name: str
    run_name: str
    rmse: Optional[float]
    synced: Optional[bool]
    primary_path: Path
    results_path: Optional[Path] = None
    wandb_path: Optional[Path] = None
    notes: str = ""


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        return {}
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    if value.endswith("Z"):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_model_timestamp_from_dir(path: Path) -> Optional[datetime]:
    parts = path.name.rsplit("_", 2)
    if len(parts) < 3:
        return None
    return parse_datetime(f"{parts[-2]}_{parts[-1]}")


def parse_wandb_timestamp_from_dir(path: Path) -> Optional[datetime]:
    tokens = path.name.split("-")
    if len(tokens) < 2:
        return None
    return parse_datetime(tokens[-2])


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_results_path(run_name: str) -> Optional[Path]:
    path = RESULTS_DIR / run_name
    return path if path.exists() else None


def iter_checkpoint_runs() -> Iterable[RunRecord]:
    if not MODELS_DIR.exists():
        return

    for metadata_path in sorted(MODELS_DIR.glob("*/metadata.json")):
        model_dir = metadata_path.parent
        metadata = load_json(metadata_path)
        metrics_path = model_dir / "metrics.json"
        metrics = load_json(metrics_path) if metrics_path.exists() else {}

        run_name = metadata.get("run_name", model_dir.name)
        yield RunRecord(
            source="checkpoint",
            started_at=parse_datetime(metadata.get("datetime"))
            or parse_datetime(metadata.get("timestamp"))
            or parse_model_timestamp_from_dir(model_dir),
            model_name=metadata.get("model_name", "unknown"),
            run_name=run_name,
            rmse=maybe_float(metrics.get("rmse") or metadata.get("performance", {}).get("rmse")),
            synced=None,
            primary_path=model_dir,
            results_path=find_results_path(run_name),
            notes="Saved model checkpoint with metadata and metrics.",
        )


def parse_wandb_summary(summary_path: Path) -> Dict[str, Any]:
    if not summary_path.exists():
        return {}
    return load_json(summary_path)


def parse_wandb_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    return load_yaml(config_path)


def config_value(config: Dict[str, Any], key: str) -> Optional[Any]:
    value = config.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def infer_run_name_from_summary(summary: Dict[str, Any]) -> Optional[str]:
    checkpoint_dir = summary.get("checkpoint/directory") or summary.get("checkpoint/saved_path")
    if not checkpoint_dir:
        return None
    metadata_path = Path(checkpoint_dir) / "metadata.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        return metadata.get("run_name")
    return None


def iter_wandb_runs() -> Iterable[RunRecord]:
    if not WANDB_DIR.exists():
        return

    run_dirs = sorted(list(WANDB_DIR.glob("run-*")) + list(WANDB_DIR.glob("offline-run-*")))
    for run_dir in run_dirs:
        files_dir = run_dir / "files"
        metadata_path = files_dir / "wandb-metadata.json"
        summary_path = files_dir / "wandb-summary.json"
        config_path = files_dir / "config.yaml"

        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        summary = parse_wandb_summary(summary_path)
        config = parse_wandb_config(config_path)

        model_name = (
            config_value(config, "model_name")
            or config_value(config, "model")
            or "unknown"
        )
        run_name = (
            infer_run_name_from_summary(summary)
            or config_value(config, "run_name")
            or run_dir.name
        )
        rmse = maybe_float(
            summary.get("test/rmse")
            or summary.get("results/best_rmse")
            or summary.get("rmse")
        )

        synced = any(run_dir.glob("*.wandb.synced"))
        mode = "offline" if run_dir.name.startswith("offline-run-") else "online"
        notes = f"W&B {mode} run"
        checkpoint_dir = summary.get("checkpoint/directory")
        results_path = find_results_path(run_name)
        if checkpoint_dir and not results_path:
            metadata_path = Path(checkpoint_dir) / "metadata.json"
            if metadata_path.exists():
                checkpoint_metadata = load_json(metadata_path)
                results_path = find_results_path(checkpoint_metadata.get("run_name", ""))

        yield RunRecord(
            source="wandb",
            started_at=parse_datetime(metadata.get("startedAt"))
            or parse_wandb_timestamp_from_dir(run_dir),
            model_name=str(model_name),
            run_name=str(run_name),
            rmse=rmse,
            synced=synced,
            primary_path=run_dir,
            results_path=results_path,
            wandb_path=run_dir,
            notes=notes,
        )


def dedupe_checkpoint_runs(records: List[RunRecord]) -> List[RunRecord]:
    seen = set()
    deduped: List[RunRecord] = []
    for record in records:
        key = (record.source, record.primary_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def sort_records(records: List[RunRecord]) -> List[RunRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.started_at or datetime.min,
            record.model_name,
            record.run_name,
        ),
        reverse=True,
    )


def format_dt(value: Optional[datetime]) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value else "-"


def format_rmse(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "-"


def format_synced(value: Optional[bool]) -> str:
    if value is None:
        return "-"
    return "yes" if value else "no"


def print_section(title: str, records: List[RunRecord]) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    if not records:
        print("No runs found.")
        return

    for record in records:
        print(
            " | ".join(
                [
                    format_dt(record.started_at),
                    f"model={record.model_name}",
                    f"run={record.run_name}",
                    f"rmse={format_rmse(record.rmse)}",
                    f"synced={format_synced(record.synced)}",
                ]
            )
        )
        print(f"  primary: {record.primary_path}")
        if record.results_path:
            print(f"  results: {record.results_path}")
        if record.wandb_path and record.wandb_path != record.primary_path:
            print(f"  wandb:   {record.wandb_path}")
        if record.notes:
            print(f"  notes:   {record.notes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List local training runs and paths.")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum records to print per section (default: 20).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the human-readable report.",
    )
    return parser.parse_args()


def record_to_dict(record: RunRecord) -> Dict[str, Any]:
    return {
        "source": record.source,
        "started_at": record.started_at.isoformat() if record.started_at else None,
        "model_name": record.model_name,
        "run_name": record.run_name,
        "rmse": record.rmse,
        "synced": record.synced,
        "primary_path": str(record.primary_path),
        "results_path": str(record.results_path) if record.results_path else None,
        "wandb_path": str(record.wandb_path) if record.wandb_path else None,
        "notes": record.notes,
    }


def main() -> None:
    args = parse_args()

    checkpoint_runs = sort_records(dedupe_checkpoint_runs(list(iter_checkpoint_runs())))
    wandb_runs = sort_records(list(iter_wandb_runs()))

    if args.limit > 0:
        checkpoint_runs = checkpoint_runs[: args.limit]
        wandb_runs = wandb_runs[: args.limit]

    if args.json:
        payload = {
            "checkpoint_runs": [record_to_dict(record) for record in checkpoint_runs],
            "wandb_runs": [record_to_dict(record) for record in wandb_runs],
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"Project root: {PROJECT_ROOT}")
    print_section("Checkpointed Runs", checkpoint_runs)
    print_section("W&B Local Runs", wandb_runs)


if __name__ == "__main__":
    main()
