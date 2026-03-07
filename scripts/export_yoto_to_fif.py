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
from preprocess_yoto import CONFIG_PREPROC, load_raw, preprocess_chain
from yoto_utils import load_config

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = ROOT / "data/manifests/unified_manifest.csv"
RAW_ROOT = ROOT / "data/raw_samples"
OUT_FIF_DIR = ROOT / "data/processed/fif_for_zuna"


def export_yoto_fif(
    manifest_path: Path = MANIFEST_CSV,
    dataset_id: str = "ds005815",
    task_contains: str = "task-task",
    out_dir: Path = OUT_FIF_DIR,
    skip_asr: bool = False,
    skip_ica: bool = False,
    config_preproc_path: Path = CONFIG_PREPROC,
    raw_root: Path = RAW_ROOT,
) -> dict:
    cfg = load_config(config_preproc_path)
    if skip_ica:
        cfg.setdefault("preprocessing", {})["ica_enabled"] = False
    if skip_asr:
        cfg.setdefault("preprocessing", {})["asr_enabled"] = False

    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        df = df[df["file_ext"] == ".vhdr"]
        df = df[df["dataset_id"] == dataset_id]
        df = df[df["file_path"].str.contains(task_contains, case=False, na=False)]
        vhdr_paths = [Path(p) for p in df["file_path"].unique()]
    else:
        vhdr_paths = list((raw_root / dataset_id).rglob(f"*{task_contains}*eeg.vhdr"))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_files: list[str] = []
    for vhdr in vhdr_paths:
        if not vhdr.is_file():
            continue
        raw = load_raw(vhdr)
        if raw is None:
            continue
        preprocess_chain(raw, cfg)
        parts = vhdr.parts
        subject_id = next((p for p in parts if p.startswith("sub-")), "unknown")
        stem = f"{dataset_id}__{subject_id}__{vhdr.stem}"
        fif_path = out_dir / f"{stem}.fif"
        raw.save(fif_path, overwrite=True, verbose=False)
        out_files.append(str(fif_path))
        print(f"Wrote {fif_path}")
    return {"dataset_id": dataset_id, "n_fif": len(out_files), "out_dir": str(out_dir)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_CSV))
    parser.add_argument("--dataset", type=str, default="ds005815")
    parser.add_argument("--task-contains", type=str, default="task-task")
    parser.add_argument("--out-dir", type=str, default=str(OUT_FIF_DIR))
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-ica", action="store_true")
    args = parser.parse_args()

    export_yoto_fif(
        manifest_path=Path(args.manifest),
        dataset_id=args.dataset,
        task_contains=args.task_contains,
        out_dir=Path(args.out_dir),
        skip_asr=args.skip_asr,
        skip_ica=args.skip_ica,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
