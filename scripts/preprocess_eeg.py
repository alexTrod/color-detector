#!/usr/bin/env python3
"""Preprocess harmonized EEG files into fixed-length epochs for training."""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = ROOT / "data/manifests/unified_manifest.csv"
OUT_DIR = ROOT / "data/processed/epochs"
OUT_INDEX = ROOT / "data/manifests/epoch_index.csv"

TARGET_SFREQ = 256
HP = 0.5
LP = 45.0
EPOCH_SECONDS = 5.0


def load_raw(path: Path) -> mne.io.BaseRaw | None:
    ext = path.suffix.lower()
    try:
        if ext == ".vhdr":
            return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        if ext == ".edf":
            return mne.io.read_raw_edf(path, preload=True, verbose=False)
        if ext == ".bdf":
            return mne.io.read_raw_bdf(path, preload=True, verbose=False)
        if ext == ".set":
            return mne.io.read_raw_eeglab(path, preload=True, verbose=False)
        if ext == ".fif":
            return mne.io.read_raw_fif(path, preload=True, verbose=False)
    except Exception:  # noqa: BLE001
        return None
    return None


def epoch_raw(raw: mne.io.BaseRaw) -> np.ndarray:
    raw.resample(TARGET_SFREQ, verbose=False)
    raw.filter(HP, LP, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    data = raw.get_data()
    n_samples = data.shape[1]
    win = int(TARGET_SFREQ * EPOCH_SECONDS)
    if n_samples < win:
        return np.empty((0, data.shape[0], win), dtype=np.float32)
    n_epochs = n_samples // win
    out = np.zeros((n_epochs, data.shape[0], win), dtype=np.float32)
    for i in range(n_epochs):
        out[i] = data[:, i * win : (i + 1) * win]
    return out


def main() -> int:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_CSV}")
    df = pd.read_csv(MANIFEST_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    epoch_rows = []
    for row in df.itertuples(index=False):
        p = Path(row.file_path)
        raw = load_raw(p)
        if raw is None:
            continue
        epochs = epoch_raw(raw)
        if epochs.size == 0:
            continue
        stem = f"{row.dataset_id}__{row.subject_id}__{p.stem}"
        out_path = OUT_DIR / f"{stem}.npy"
        np.save(out_path, epochs)
        meta = {
            "dataset_id": row.dataset_id,
            "subject_id": row.subject_id,
            "source_file": str(p),
            "epochs_file": str(out_path),
            "n_epochs": int(epochs.shape[0]),
            "n_channels": int(epochs.shape[1]),
            "n_samples": int(epochs.shape[2]),
            "modality": row.modality,
            "task_label": row.task_label,
        }
        epoch_rows.append(meta)

    pd.DataFrame(epoch_rows).to_csv(OUT_INDEX, index=False)
    (ROOT / "data/manifests/epoch_index.json").write_text(json.dumps(epoch_rows, indent=2), encoding="utf-8")
    print(f"Saved {len(epoch_rows)} preprocessed epoch files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
