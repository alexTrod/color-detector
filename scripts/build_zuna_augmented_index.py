#!/usr/bin/env python3
"""Create augmented epoch index by adding ZUNA reconstructed FIF epochs."""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_INDEX = ROOT / "data/manifests/epoch_index.csv"
ZUNA_FIF = ROOT / "runs/zuna_smoke/4_fif_output/sub-01_ses-1_task-rest1_eeg.fif"
AUG_EPOCH_DIR = ROOT / "data/processed/epochs_aug"
OUT_INDEX = ROOT / "data/manifests/epoch_index_with_zuna.csv"


def epoch_fif(path: Path, out_path: Path) -> tuple[int, int, int]:
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    raw.resample(256, verbose=False)
    raw.filter(0.5, 45.0, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    data = raw.get_data()
    win = 256 * 5
    n_epochs = data.shape[1] // win
    epochs = np.zeros((n_epochs, data.shape[0], win), dtype=np.float32)
    for i in range(n_epochs):
        epochs[i] = data[:, i * win : (i + 1) * win]
    np.save(out_path, epochs)
    return n_epochs, data.shape[0], win


def main() -> int:
    df = pd.read_csv(BASE_INDEX)
    AUG_EPOCH_DIR.mkdir(parents=True, exist_ok=True)

    out_npy = AUG_EPOCH_DIR / "zuna_sub-01_rest1.npy"
    n_epochs, n_channels, n_samples = epoch_fif(ZUNA_FIF, out_npy)
    aug_row = {
        "dataset_id": "ds005815_zuna_aug",
        "subject_id": "sub-01",
        "source_file": str(ZUNA_FIF),
        "epochs_file": str(out_npy),
        "n_epochs": n_epochs,
        "n_channels": n_channels,
        "n_samples": n_samples,
        "modality": "auditory",
        "task_label": "rest_zuna_aug",
    }

    pd.concat([df, pd.DataFrame([aug_row])], ignore_index=True).to_csv(OUT_INDEX, index=False)
    print(f"Wrote {OUT_INDEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
