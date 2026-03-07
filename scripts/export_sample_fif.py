#!/usr/bin/env python3
"""Export a small FIF sample from downloaded BrainVision data for ZUNA."""
from __future__ import annotations

from pathlib import Path

import mne

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw_samples/ds005815_sample/sub-01/ses-1/eeg/sub-01_ses-1_task-rest1_eeg.vhdr"
OUT_DIR = ROOT / "data/processed/fif_inputs"
OUT = OUT_DIR / "sub-01_ses-1_task-rest1_eeg.fif"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = mne.io.read_raw_brainvision(SRC, preload=True, verbose=False)
    # Keep only first 120 seconds to speed up local smoke testing.
    raw.crop(tmin=0, tmax=min(raw.times[-1], 120))
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
    raw.save(OUT, overwrite=True)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
