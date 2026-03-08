#!/usr/bin/env python3
"""Train a baseline classifier on preprocessed EEG epochs."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EPOCH_INDEX = ROOT / "data/manifests/epoch_index.csv"
OUT_METRICS = ROOT / "runs/baseline_metrics.json"


def featurize(epochs: np.ndarray) -> np.ndarray:
    # Channel-agnostic features so variable montages can be combined.
    flat = epochs.reshape(epochs.shape[0], -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True)
    var = flat.var(axis=1, keepdims=True)
    rms = np.sqrt((flat**2).mean(axis=1, keepdims=True))
    ptp = (flat.max(axis=1, keepdims=True) - flat.min(axis=1, keepdims=True))
    return np.hstack([mean, std, var, rms, ptp])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-index", default=str(EPOCH_INDEX))
    parser.add_argument("--out-metrics", default=str(OUT_METRICS))
    args = parser.parse_args()

    df = pd.read_csv(args.epoch_index)
    X_all, y_all, g_all = [], [], []
    for row in df.itertuples(index=False):
        epochs_path = ROOT / row.epochs_file if not Path(row.epochs_file).is_absolute() else Path(row.epochs_file)
        epochs = np.load(epochs_path)
        X = featurize(epochs)
        X_all.append(X)
        y_all.extend([row.modality] * X.shape[0])
        g_all.extend([f"{row.dataset_id}::{row.subject_id}"] * X.shape[0])

    X = np.vstack(X_all)
    y = np.array(y_all)
    groups = np.array(g_all)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, n_jobs=-1)),
        ]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    out_metrics = Path(args.out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved baseline metrics to {out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
