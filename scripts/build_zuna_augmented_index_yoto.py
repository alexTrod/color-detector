#!/usr/bin/env python3
"""Build long-format epoch index for Zuna-augmented YOTO tone epochs. Epoch Zuna FIF at event onsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PREPROC = ROOT / "configs/yoto_preprocessing.yaml"
CONFIG_EVENTS = ROOT / "configs/yoto_events.yaml"
EPOCH_INDEX_YOTO = ROOT / "data/manifests/epoch_index_yoto_tones.csv"
AUG_EPOCH_DIR = ROOT / "data/processed/epochs_aug"
OUT_INDEX = ROOT / "data/manifests/epoch_index_yoto_tones_zuna.csv"


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def get_event_mapping(config_events: dict) -> dict[int | str, str]:
    out = {}
    for k, v in config_events.get("event_value_to_stimulus", {}).items():
        out[int(k)] = str(v)
    for k, v in config_events.get("trial_type_to_stimulus", {}).items():
        out[str(k)] = str(v)
    return out


def load_events_tsv(vhdr_path: Path) -> pd.DataFrame | None:
    stem = vhdr_path.stem.replace("_eeg", "")
    tsv = vhdr_path.parent / (stem + "_events.tsv")
    if not tsv.exists():
        tsv = vhdr_path.with_suffix(".tsv")
    if not tsv.exists():
        return None
    return pd.read_csv(tsv, sep="\t")


def extract_tone_onsets(
    events_df: pd.DataFrame,
    event_mapping: dict,
    onset_col: str = "onset",
    value_col: str = "value",
) -> list[tuple[float, str]]:
    out = []
    for _, row in events_df.iterrows():
        stim_id = None
        if value_col in events_df.columns and pd.notna(row.get(value_col)):
            try:
                v = int(float(row[value_col]))
                stim_id = event_mapping.get(v)
            except (ValueError, TypeError):
                pass
        if stim_id not in ("tone_C", "tone_D", "tone_E"):
            continue
        out.append((float(row[onset_col]), stim_id))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zuna-fif-dir", required=True, help="ZUNA 4_fif_output directory")
    parser.add_argument("--epoch-index", type=str, default=str(EPOCH_INDEX_YOTO))
    parser.add_argument("--out-dir", type=str, default=str(AUG_EPOCH_DIR))
    parser.add_argument("--out-index", type=str, default=str(OUT_INDEX))
    args = parser.parse_args()

    cfg = load_config(CONFIG_PREPROC)
    cfg_events = load_config(CONFIG_EVENTS)
    event_mapping = get_event_mapping(cfg_events)
    epoch_cfg = cfg.get("epoch", {})
    tmin = float(epoch_cfg.get("tmin", -0.2))
    tmax = float(epoch_cfg.get("tmax", 0.8))

    df = pd.read_csv(args.epoch_index)
    if "source_file" not in df.columns:
        print("Epoch index must have source_file column (long-format).")
        return 1
    unique_sources = df["source_file"].unique()
    zuna_dir = Path(args.zuna_fif_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

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

    out_index_path = Path(args.out_index)
    out_index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_rows:
        pd.DataFrame(index_rows).to_csv(out_index_path, index=False)
        out_index_path.with_suffix(".json").write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
        print(f"Wrote {len(index_rows)} rows to {out_index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
