#!/usr/bin/env python3
"""Build long-format epoch index for Zuna-augmented YOTO tone epochs. Epoch Zuna FIF at event onsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from yoto_utils import extract_tone_onsets, get_event_mapping, load_config, load_events_tsv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PREPROC = ROOT / "configs/yoto_preprocessing.yaml"
CONFIG_EVENTS = ROOT / "configs/yoto_events.yaml"
EPOCH_INDEX_YOTO = ROOT / "data/manifests/epoch_index_yoto_tones.csv"
AUG_EPOCH_DIR = ROOT / "data/processed/epochs_aug"
OUT_INDEX = ROOT / "data/manifests/epoch_index_yoto_tones_zuna.csv"


def build_zuna_epoch_index(
    zuna_fif_dir: str | Path,
    epoch_index: str | Path = EPOCH_INDEX_YOTO,
    out_dir: str | Path = AUG_EPOCH_DIR,
    out_index: str | Path = OUT_INDEX,
    config_preproc: Path = CONFIG_PREPROC,
    config_events: Path = CONFIG_EVENTS,
) -> list[dict]:
    cfg = load_config(config_preproc)
    cfg_events = load_config(config_events)
    event_mapping = get_event_mapping(cfg_events)
    epoch_cfg = cfg.get("epoch", {})
    tmin = float(epoch_cfg.get("tmin", -0.2))
    tmax = float(epoch_cfg.get("tmax", 0.8))

    df = pd.read_csv(epoch_index)
    if "source_file" not in df.columns:
        print("Epoch index must have source_file column (long-format).")
        return []
    unique_sources = df["source_file"].unique()
    zuna_dir = Path(zuna_fif_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []

    for source_file in unique_sources:
        vhdr = Path(source_file)
        if not vhdr.exists():
            continue
        events_df = load_events_tsv(vhdr)
        if events_df is None:
            continue
        onsets_stim = extract_tone_onsets(events_df, event_mapping)
        if not onsets_stim:
            continue
        parts = vhdr.parts
        subject_id = next((p for p in parts if p.startswith("sub-")), "unknown")
        dataset_id = df[df["source_file"] == source_file]["dataset_id"].iloc[0]
        stem = f"{dataset_id}__{subject_id}__{vhdr.stem}"
        fif_path = zuna_dir / f"{stem}.fif"
        if not fif_path.exists():
            continue
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        n_samp_pre = int(round(-tmin * sfreq))
        n_samp_post = int(round(tmax * sfreq))
        n_samp = n_samp_pre + n_samp_post
        data = raw.get_data()
        n_ch = data.shape[0]
        valid = []
        for onset_sec, stim_id in onsets_stim:
            onset_samp = int(round(onset_sec * sfreq))
            start = onset_samp - n_samp_pre
            end = onset_samp + n_samp_post
            if start < 0 or end > data.shape[1]:
                continue
            valid.append((start, stim_id))
        if not valid:
            continue
        epochs = np.zeros((len(valid), n_ch, n_samp), dtype=np.float32)
        for i, (start, _) in enumerate(valid):
            epochs[i] = data[:, start : start + n_samp]
        out_npy = out_dir / f"zuna_{stem}.npy"
        np.save(out_npy, epochs)
        for epoch_idx, (_, stim_id) in enumerate(valid):
            index_rows.append({
                "dataset_id": f"{dataset_id}_zuna_aug",
                "subject_id": subject_id,
                "source_file": str(fif_path),
                "epochs_file": str(out_npy),
                "epoch_idx": epoch_idx,
                "stimulus_id": stim_id,
                "n_channels": n_ch,
                "n_samples": n_samp,
                "pipeline_source": "zuna",
            })

    out_index_path = Path(out_index)
    out_index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_rows:
        pd.DataFrame(index_rows).to_csv(out_index_path, index=False)
        out_index_path.with_suffix(".json").write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
        print(f"Wrote {len(index_rows)} rows to {out_index_path}")
    return index_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zuna-fif-dir", required=True, help="ZUNA 4_fif_output directory")
    parser.add_argument("--epoch-index", type=str, default=str(EPOCH_INDEX_YOTO))
    parser.add_argument("--out-dir", type=str, default=str(AUG_EPOCH_DIR))
    parser.add_argument("--out-index", type=str, default=str(OUT_INDEX))
    args = parser.parse_args()

    build_zuna_epoch_index(
        zuna_fif_dir=args.zuna_fif_dir,
        epoch_index=args.epoch_index,
        out_dir=args.out_dir,
        out_index=args.out_index,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
