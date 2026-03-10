#!/usr/bin/env python3
"""Compare multiple previously-used modality classifiers on one fixed split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from scripts.train_baseline import featurize
    from scripts.train_labram import (
        ROOT,
        build_erp_summary_features,
        extract_labram_features,
        load_data,
        resolve_checkpoint_path,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_baseline import featurize  # type: ignore
    from train_labram import (  # type: ignore
        ROOT,
        build_erp_summary_features,
        extract_labram_features,
        load_data,
        resolve_checkpoint_path,
    )

SEED = 42


def split_indices(X: np.ndarray, y_enc: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_samples = len(y_enc)
    n_groups = len(set(groups.tolist()))
    if n_samples < 10 or n_groups < 2:
        split = ShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
        return next(split.split(X, y_enc))
    split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    return next(split.split(X, y_enc, groups))


def run_compare(
    epoch_index_path: Path,
    out_csv: Path,
    out_json: Path,
    checkpoint: str = "",
    batch_size: int = 16,
    device: str = "cpu",
) -> dict:
    X, y, groups, channel_names = load_data(epoch_index_path)
    keep = np.array([lbl in {"auditory", "visual"} for lbl in y], dtype=bool)
    X = X[keep]
    y = y[keep]
    groups = groups[keep]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    train_idx, test_idx = split_indices(X, y_enc, groups)

    raw_features = featurize(X[:, :, 0, :])
    erp_features = build_erp_summary_features(X)
    checkpoint_path = resolve_checkpoint_path(checkpoint)
    labram_features, checkpoint_load = extract_labram_features(
        X,
        channel_names=channel_names,
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=batch_size,
    )
    combo_features = np.concatenate([labram_features, erp_features], axis=1)

    model_defs: list[tuple[str, np.ndarray, LogisticRegression]] = [
        (
            "raw_logreg_l2_c1",
            raw_features,
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED),
        ),
        (
            "erp_logreg_l2_c1",
            erp_features,
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED),
        ),
        (
            "labram_logreg_l2_c1",
            labram_features,
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED),
        ),
        (
            "labram_erp_logreg_l2_c1",
            combo_features,
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED),
        ),
        (
            "labram_erp_logreg_liblinear_c10",
            combo_features,
            LogisticRegression(
                max_iter=3000,
                C=10.0,
                class_weight="balanced",
                solver="liblinear",
                random_state=SEED,
            ),
        ),
    ]

    rows: list[dict] = []
    for name, feats, clf in model_defs:
        pipe = make_pipeline(StandardScaler(), clf)
        pipe.fit(feats[train_idx], y_enc[train_idx])
        pred = pipe.predict(feats[test_idx])
        true = y_enc[test_idx]
        rows.append(
            {
                "model_name": name,
                "accuracy": float((pred == true).mean()),
                "macro_f1": float(f1_score(true, pred, average="macro")),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(by=["accuracy", "macro_f1"], ascending=[False, False]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    best = df.iloc[0].to_dict()
    summary = {
        "epoch_index": str(epoch_index_path),
        "labels": list(le.classes_),
        "n_samples": int(len(y_enc)),
        "n_groups": int(len(set(groups.tolist()))),
        "checkpoint": str(checkpoint_path),
        "checkpoint_load": checkpoint_load,
        "channel_names": channel_names,
        "results_csv": str(out_csv),
        "best_model": best,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-index", type=str, default=str(ROOT / "data/manifests/epoch_index.csv"))
    parser.add_argument("--out-csv", type=str, default=str(ROOT / "runs/modality_model_comparison.csv"))
    parser.add_argument("--out-json", type=str, default=str(ROOT / "runs/modality_model_comparison.json"))
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    summary = run_compare(
        epoch_index_path=Path(args.epoch_index),
        out_csv=Path(args.out_csv),
        out_json=Path(args.out_json),
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

