# ZUNA Augmentation

ZUNA is integrated as an actual library dependency:

- Repo: `https://github.com/Zyphra/zuna`
- Paper: `https://arxiv.org/html/2602.18478v1`
- Package: `pip install zuna`

## Why ZUNA here

ZUNA reconstructs or infills missing channels and supports arbitrary electrode positions, making it useful for harmonizing heterogeneous montages before classifier/Labram training.

## Pipeline usage

Script: `scripts/run_zuna_augmentation.py`

```bash
python3 scripts/run_zuna_augmentation.py \
  --input-fif-dir data/raw_samples/ds005815 \
  --work-dir runs/zuna_ds005815 \
  --gpu-device 0
```

This script runs:

1. `zuna.preprocessing()`
2. `zuna.inference()`
3. `zuna.pt_to_fif()`

## Leakage safeguards

- Apply ZUNA augmentation **only to training split**.
- Keep validation/test on non-augmented originals for primary metrics.
- Track augmentation provenance in manifests.

## Caveat

ZUNA outputs are imputed signals; treat them as model-generated approximations, not direct measurements.
