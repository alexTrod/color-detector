# EEG Color-Sound Pipeline

This project builds a reproducible EEG pipeline for:

1. Classification on color/sound attention data.
2. Fine-tuning a Labram-style model on harmonized data.
3. Augmenting/imputing channels via the ZUNA library.

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 scripts/size_inventory.py
python3 scripts/download_sample_subjects.py
python3 scripts/build_unified_manifest.py
python3 scripts/preprocess_eeg.py
python3 scripts/train_baseline.py
python3 scripts/train_labram.py --epochs 5
```

## ZUNA integration

This repository uses ZUNA as a real dependency (not placeholder code):

- Repo: `https://github.com/Zyphra/zuna`
- Paper: `https://arxiv.org/html/2602.18478v1`

Run augmentation stage:

```bash
python3 scripts/run_zuna_augmentation.py \
  --input-fif-dir data/raw_samples/ds005815 \
  --work-dir runs/zuna_ds005815
```

## Docs

- `docs/pipeline_overview.md`
- `docs/dataset_registry_guide.md`
- `docs/preprocessing_spec.md`
- `docs/training_playbook.md`
- `docs/augmentation_zuna.md`
- `docs/quickstart_3_subject_smoke_test.md`
