# Pipeline Overview

## Objectives

- Build a modality/color/sound EEG classifier from harmonized public datasets.
- Fine-tune a Labram-style model on the same unified schema.
- Add train-time augmentation via ZUNA channel infilling/superresolution.

## Dataflow

1. `eeg_datasets_for_labram.csv` as registry.
2. `scripts/size_inventory.py` computes total and per-subject sizes.
3. `scripts/download_sample_subjects.py` downloads 1-subject samples for:
   - `ds005815` (YOTO)
   - `ds004621` (Nencki-Symfonia)
   - `j289e` (Hajonides OSF bundle)
4. `scripts/build_unified_manifest.py` creates a common file manifest.
5. `scripts/preprocess_eeg.py` performs filtering/resampling/epoching.
6. `scripts/train_baseline.py` trains baseline classifier.
7. `scripts/run_zuna_augmentation.py` runs ZUNA pipeline on FIF data.
8. `scripts/train_labram.py` runs a fine-tuning scaffold for Labram.

## Artifact locations

- Raw samples: `data/raw_samples/`
- Manifest files: `data/manifests/`
- Processed epochs: `data/processed/epochs/`
- Outputs: `runs/`
