#!/usr/bin/env python3
"""Export YOTO preprocessed continuous data to FIF for Zuna. One FIF per task-task recording."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

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


def export_yoto_fif(
    manifest: str | Path = MANIFEST_CSV,
    datasets: str | list[str] = "ds005815",
    task_contains: str = "task-task",
    out_dir: str | Path = OUT_FIF_DIR,
    skip_asr: bool = False,
    skip_ica: bool = False,
    config_preproc: Path = CONFIG_PREPROC,
) -> list[Path]:
    cfg = load_config(config_preproc)
    if skip_ica:
        cfg.setdefault("preprocessing", {})["ica_enabled"] = False
    if skip_asr:
        cfg.setdefault("preprocessing", {})["asr_enabled"] = False

    datasets_list = [datasets] if isinstance(datasets, str) else list(datasets)
    file_to_dataset: dict[str, str] = {}
    if Path(manifest).exists():
        df = pd.read_csv(manifest)
        df = df[df["file_ext"] == ".vhdr"]
        df = df[df["dataset_id"].isin(datasets_list)]
        df = df[df["file_path"].str.contains(task_contains, case=False, na=False)]
        vhdr_paths = [Path(p) for p in df["file_path"].unique()]
        file_to_dataset = dict(zip(df["file_path"], df["dataset_id"]))
    else:
        vhdr_paths = []
        for dataset_id in datasets_list:
            vhdr_paths.extend((RAW_ROOT / dataset_id).rglob(f"*{task_contains}*eeg.vhdr"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for vhdr in vhdr_paths:
        if not vhdr.is_file():
            continue
        raw = load_raw(vhdr)
        if raw is None:
            continue
        preprocess_chain(raw, cfg)
        parts = vhdr.parts
        subject_id = next((p for p in parts if p.startswith("sub-")), "unknown")
        dataset_id = file_to_dataset.get(str(vhdr)) or next(
            (p for p in parts if p.startswith("ds")), datasets_list[0]
        )
        stem = f"{dataset_id}__{subject_id}__{vhdr.stem}"
        fif_path = out_dir / f"{stem}.fif"
        raw.save(fif_path, overwrite=True, verbose=False)
        outputs.append(fif_path)
        print(f"Wrote {fif_path}")
    return outputs


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
        manifest=args.manifest,
        datasets=args.dataset,
        task_contains=args.task_contains,
        out_dir=args.out_dir,
        skip_asr=args.skip_asr,
        skip_ica=args.skip_ica,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
