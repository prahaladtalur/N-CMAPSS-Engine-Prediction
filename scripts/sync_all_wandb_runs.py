#!/usr/bin/env python
"""Sync offline W&B runs, repairing seeded artifact client IDs when needed."""

from __future__ import annotations

import secrets
import shutil
import string
import subprocess
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore


ROOT = Path(__file__).resolve().parent.parent
WANDB_DIR = ROOT / "wandb"


def random_id(length: int) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def iter_offline_runs() -> list[Path]:
    return sorted(WANDB_DIR.glob("offline-run-*"))


def run_file_path(run_dir: Path) -> Path | None:
    files = list(run_dir.glob("run-*.wandb"))
    return files[0] if files else None


def synced_marker(run_file: Path) -> Path:
    return run_file.with_name(run_file.name + ".synced")


def unsynced_runs() -> list[Path]:
    runs: list[Path] = []
    for run_dir in iter_offline_runs():
        run_file = run_file_path(run_dir)
        if run_file and not synced_marker(run_file).exists():
            runs.append(run_dir)
    return runs


def repair_artifact_client_ids(run_dir: Path) -> int:
    run_file = run_file_path(run_dir)
    if run_file is None:
        return 0

    backup_file = run_file.with_suffix(run_file.suffix + ".bak")
    source_file = backup_file if backup_file.exists() else run_file
    if not backup_file.exists():
        shutil.copy2(run_file, backup_file)

    reader = datastore.DataStore()
    reader.open_for_scan(str(source_file))

    rewritten = run_file.with_suffix(".rewritten")
    if rewritten.exists():
        rewritten.unlink()

    writer = datastore.DataStore()
    writer.open_for_write(str(rewritten))

    repaired = 0
    try:
        while True:
            data = reader.scan_data()
            if data is None:
                break

            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)

            if record.WhichOneof("record_type") == "artifact":
                client_len = len(record.artifact.client_id) or 128
                sequence_len = len(record.artifact.sequence_client_id) or 128
                record.artifact.client_id = random_id(client_len)
                record.artifact.sequence_client_id = random_id(sequence_len)
                repaired += 1

            writer._write_data(record.SerializeToString())
    finally:
        if reader._fp:
            reader._fp.close()
        if writer._fp:
            writer._fp.flush()
            writer._fp.close()

    shutil.move(rewritten, run_file)
    return repaired


def sync_run(run_dir: Path) -> bool:
    run_file = run_file_path(run_dir)
    if run_file is None:
        return False
    proc = subprocess.run(["wandb", "sync", "--mark-synced", str(run_dir)], cwd=ROOT)
    if proc.returncode != 0:
        return False
    marker = synced_marker(run_file)
    marker.touch(exist_ok=True)
    return True


def main() -> None:
    runs = unsynced_runs()
    print(f"Unsynced offline runs: {len(runs)}")

    synced = 0
    repaired = 0
    failed: list[str] = []

    for index, run_dir in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {run_dir}")
        if sync_run(run_dir):
            synced += 1
            continue

        repaired_records = repair_artifact_client_ids(run_dir)
        if repaired_records:
            repaired += repaired_records
            if sync_run(run_dir):
                synced += 1
                continue

        failed.append(str(run_dir))

    print("")
    print(f"Synced runs: {synced}")
    print(f"Repaired artifact records: {repaired}")
    print(f"Failed runs: {len(failed)}")
    for run_dir in failed:
        print(run_dir)


if __name__ == "__main__":
    main()
