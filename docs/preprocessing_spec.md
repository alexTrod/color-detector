# Preprocessing Spec

Pipeline implementation: `scripts/preprocess_eeg.py`

## Current transform chain

1. Load raw EEG (`.vhdr`, `.edf`, `.bdf`, `.fif`).
2. Resample to `256 Hz`.
3. Bandpass filter `0.5-45 Hz`.
4. Re-reference to average.
5. Split into non-overlapping `5s` windows.
6. Save epoch tensors to `.npy`.

## Notes

- This aligns with ZUNA expectations (`256 Hz`, `5s` windows) from the official package docs and paper.
- `.mat` files are listed in the manifest but not parsed by current preprocessing script; add dataset-specific loaders when needed.
- Subject-wise splitting is performed downstream to avoid leakage.

