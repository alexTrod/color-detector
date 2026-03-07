#!/usr/bin/env python3
"""Orchestrate end-to-end pipeline stages."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"-> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-downloads", action="store_true")
    parser.add_argument("--skip-zuna", action="store_true")
    args = parser.parse_args()

    run(["python3", "scripts/size_inventory.py"])
    if not args.skip_downloads:
        run(["python3", "scripts/download_sample_subjects.py"])
    run(["python3", "scripts/build_unified_manifest.py"])
    run(["python3", "scripts/preprocess_eeg.py"])
    run(["python3", "scripts/train_baseline.py"])
    if not args.skip_zuna:
        print("ZUNA stage requires .fif inputs; run scripts/run_zuna_augmentation.py with your FIF directory.")
    run(["python3", "scripts/train_labram.py", "--epochs", "5"])
    print("Pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
