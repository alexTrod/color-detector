# Quickstart: 3-Subject Smoke Test

This smoke test verifies that the full pipeline wiring works on one-subject samples from:

- YOTO (`ds005815`)
- Nencki-Symfonia (`ds004621`)
- Hajonides (`osf.io/j289e`)

## 1) Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

## 2) Compute dataset sizes and refresh registry

```bash
python3 scripts/size_inventory.py
```

## 3) Download sample subjects (raw + metadata/events where available)

```bash
python3 scripts/download_sample_subjects.py
```

## 4) Build manifest and preprocess

```bash
python3 scripts/build_unified_manifest.py
python3 scripts/preprocess_eeg.py
```

## 5) Train baseline and Labram scaffold

```bash
python3 scripts/train_baseline.py
python3 scripts/train_labram.py --epochs 5
```

## 6) Optional ZUNA augmentation stage

```bash
python3 scripts/run_zuna_augmentation.py \
  --input-fif-dir data/raw_samples/ds005815 \
  --work-dir runs/zuna_ds005815
```

## Expected outputs

- `data/manifests/sample_download_manifest.json`
- `data/manifests/unified_manifest.csv`
- `data/manifests/epoch_index.csv`
- `runs/baseline_metrics.json`
- `runs/labram_metrics.json`
