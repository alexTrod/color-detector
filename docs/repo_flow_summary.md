# Repo Flow Summary

## Purpose
- Build a reproducible EEG pipeline for color/sound attention classification, Labram fine-tuning, and channel augmentation (ZUNA).

## High-level dataflow
1. Register available datasets in `eeg_datasets_for_labram.csv`.
2. Compute sizes with `scripts/size_inventory.py`.
3. Download one-subject samples using `scripts/download_sample_subjects.py`.
4. Build a unified manifest: `scripts/build_unified_manifest.py`.
5. Preprocess raw EEG into epochs (resample, filter, re-reference, window): `scripts/preprocess_eeg.py`.
6. (Optional) Run ZUNA augmentation on FIF data: `scripts/run_zuna_augmentation.py`.
7. Train baseline classifier: `scripts/train_baseline.py`.
8. Fine-tune Labram scaffold: `scripts/train_labram.py`.

## Key artifacts & locations
- Raw samples: `data/raw_samples/`  
- Manifest files / epoch index: `data/manifests/`, `data/processed/epochs/`  
- Model & metrics outputs: `runs/` (e.g. `runs/baseline_metrics.json`, `runs/labram_metrics.json`)

## Quickstart (smoke test)
1. Install deps: `pip install -r requirements.txt`  
2. `python3 scripts/size_inventory.py`  
3. `python3 scripts/download_sample_subjects.py`  
4. `python3 scripts/build_unified_manifest.py` && `python3 scripts/preprocess_eeg.py`  
5. `python3 scripts/train_baseline.py` and `python3 scripts/train_labram.py --epochs 5`  
6. (Optional) `python3 scripts/run_zuna_augmentation.py --input-fif-dir data/raw_samples/ds005815 --work-dir runs/zuna_ds005815`

## Notes & safeguards
- ZUNA outputs are imputed signals — apply only to training split and track provenance.  
- Manifests and `epoch_index` drive notebook exploration and reproducible runs.  
- Notebooks (e.g. `notebooks/00_yoto_data_exploration.ipynb`) are exploratory; for automation use the `scripts/` entry points.

For details see `docs/pipeline_overview.md`, `docs/preprocessing_spec.md`, and `docs/augmentation_zuna.md`.

