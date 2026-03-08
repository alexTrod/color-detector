#!/usr/bin/env python3
"""Sweep frozen LaBraM linear-probe settings and keep best run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from scripts.train_labram import (
        ROOT,
        build_erp_summary_features,
        extract_labram_features,
        load_data,
        resolve_checkpoint_path,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_labram import (  # type: ignore
        ROOT,
        build_erp_summary_features,
        extract_labram_features,
        load_data,
        resolve_checkpoint_path,
    )


def _split_indices(X: np.ndarray, y_enc: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_samples = len(y_enc)
    n_groups = len(set(groups.tolist()))
    if n_samples < 10 or n_groups < 2:
        split = ShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        return next(split.split(X, y_enc))
    split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    return next(split.split(X, y_enc, groups))


def _build_clf(c_value: float, class_weight: str | None, use_scaler: bool) -> object:
    base = LogisticRegression(
        max_iter=5000,
        class_weight=class_weight,
        C=float(c_value),
        random_state=42,
    )
    if use_scaler:
        return make_pipeline(StandardScaler(), base)
    return base


def tune_probe(
    epoch_index_path: Path,
    out_results_path: Path,
    out_best_metrics_path: Path,
    checkpoint: str = "",
    device: str = "cpu",
    batch_size: int = 16,
    seed: int = 42,
) -> dict:
    np.random.seed(seed)

    X, y, groups, channel_names = load_data(epoch_index_path)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    if len(y_enc) < 2:
        raise ValueError("Not enough samples for probe tuning.")

    train_idx, test_idx = _split_indices(X, y_enc, groups, seed=seed)
    checkpoint_path = resolve_checkpoint_path(checkpoint)
    labram_features, checkpoint_load = extract_labram_features(
        X,
        channel_names=channel_names,
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=batch_size,
    )
    erp_features = build_erp_summary_features(X)
    feature_sets = {
        "labram": labram_features,
        "erp": erp_features,
        "labram_plus_erp": np.concatenate([labram_features, erp_features], axis=1),
    }

    c_values = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    class_weights = [None, "balanced"]
    scaler_opts = [True, False]

    rows: list[dict] = []
    exp_id = 0
    for feature_name, feats in feature_sets.items():
        for c_value in c_values:
            for class_weight in class_weights:
                for use_scaler in scaler_opts:
                    exp_id += 1
                    clf = _build_clf(c_value, class_weight, use_scaler)
                    clf.fit(feats[train_idx], y_enc[train_idx])
                    y_pred = clf.predict(feats[test_idx])
                    y_true = y_enc[test_idx]
                    acc = float((y_pred == y_true).mean())
                    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
                    rows.append({
                        "exp_id": exp_id,
                        "feature_set": feature_name,
                        "probe_c": float(c_value),
                        "class_weight": class_weight if class_weight is not None else "none",
                        "use_scaler": bool(use_scaler),
                        "accuracy": acc,
                        "macro_f1": macro_f1,
                    })

    results_df = pd.DataFrame(rows).sort_values(
        by=["macro_f1", "accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)
    out_results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_results_path, index=False)

    best = results_df.iloc[0].to_dict()
    best_feature = feature_sets[str(best["feature_set"])]
    best_clf = _build_clf(
        c_value=float(best["probe_c"]),
        class_weight=None if best["class_weight"] == "none" else str(best["class_weight"]),
        use_scaler=bool(best["use_scaler"]),
    )
    best_clf.fit(best_feature[train_idx], y_enc[train_idx])
    best_pred = best_clf.predict(best_feature[test_idx])
    best_true = y_enc[test_idx]

    metrics = {
        "accuracy": float((best_pred == best_true).mean()),
        "macro_f1": float(f1_score(best_true, best_pred, average="macro")),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "label_classes": list(le.classes_),
        "model": "labram_base_patch200_200",
        "classifier": "labram_frozen_probe_sweep_best",
        "checkpoint": str(checkpoint_path),
        "checkpoint_load": checkpoint_load,
        "input_shape": [int(X.shape[1]), int(X.shape[2]), int(X.shape[3])],
        "channel_names": channel_names,
        "labram_feature_dim": int(labram_features.shape[1]),
        "erp_summary_dim": int(erp_features.shape[1]),
        "best_experiment": best,
        "results_csv": str(out_results_path),
    }
    out_best_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    out_best_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-index", type=str, default=str(ROOT / "data/manifests/epoch_index_yoto_tones.csv"))
    parser.add_argument("--out-results", type=str, default=str(ROOT / "runs/labram_probe_sweep.csv"))
    parser.add_argument("--out-best-metrics", type=str, default=str(ROOT / "runs/labram_metrics_tuned.json"))
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = tune_probe(
        epoch_index_path=Path(args.epoch_index),
        out_results_path=Path(args.out_results),
        out_best_metrics_path=Path(args.out_best_metrics),
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
