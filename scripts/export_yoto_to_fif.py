#!/usr/bin/env python3
"""Export YOTO preprocessed continuous data to FIF for Zuna. One FIF per task-task recording."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))
from preprocess_yoto import (
    CONFIG_PREPROC,
    load_config,
    load_raw,
    preprocess_chain,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = ROOT / "data/manifests/unified_manifest.csv"
RAW_ROOT = ROOT / "data/raw_samples"
OUT_FIF_DIR = ROOT / "data/processed/fif_for_zuna"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_CSV))
    parser.add_argument("--dataset", type=str, default="ds005815")
    parser.add_argument("--task-contains", type=str, default="task-task")
    parser.add_argument("--out-dir", type=str, default=str(OUT_FIF_DIR))
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-ica", action="store_true")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PREPROC)
    if args.skip_ica:
        cfg.setdefault("preprocessing", {})["ica_enabled"] = False
    if args.skip_asr:
        cfg.setdefault("preprocessing", {})["asr_enabled"] = False

    if Path(args.manifest).exists():
        df = pd.read_csv(args.manifest)
        df = df[df["file_ext"] == ".vhdr"]
        df = df[df["dataset_id"] == args.dataset]
        df = df[df["file_path"].str.contains(args.task_contains, case=False, na=False)]
        vhdr_paths = [Path(p) for p in df["file_path"].unique()]
    else:
        vhdr_paths = list((RAW_ROOT / args.dataset).rglob("*task-task*eeg.vhdr"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for vhdr in vhdr_paths:
        if not vhdr.is_file():
            continue
        raw = load_raw(vhdr)
        if raw is None:
            continue
        preprocess_chain(raw, cfg)
        parts = vhdr.parts
        subject_id = next((p for p in parts if p.startswith("sub-")), "unknown")
        stem = f"{args.dataset}__{subject_id}__{vhdr.stem}"
        fif_path = out_dir / f"{stem}.fif"
        raw.save(fif_path, overwrite=True, verbose=False)
        print(f"Wrote {fif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
