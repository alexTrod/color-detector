# Training Playbook

## Baseline classifier

Script: `scripts/train_baseline.py`

```bash
python3 scripts/train_baseline.py
```

Outputs:

- `runs/baseline_metrics.json`

Approach:

- Features: per-channel mean/std per epoch.
- Model: logistic regression with standardization.
- Split: group-aware subject split.

## Labram fine-tuning scaffold

Script: `scripts/train_labram.py`

```bash
python3 scripts/train_labram.py --epochs 10 --batch-size 32 --lr 1e-4 --device cpu
```

Outputs:

- `runs/labram_metrics.json`

Current implementation uses a fallback head scaffold and keeps clear replacement points for loading a real Labram backbone checkpoint.

## End-to-end orchestration

```bash
python3 scripts/run_pipeline.py
```
