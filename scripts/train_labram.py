#!/usr/bin/env python3
"""Fine-tune an actual LaBraM model on harmonized EEG epochs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
EPOCH_INDEX = ROOT / "data/manifests/epoch_index.csv"
OUT_METRICS = ROOT / "runs/labram_metrics.json"
LABRAM_REPO = ROOT / "vendor/LaBraM"

if str(LABRAM_REPO) not in sys.path:
    sys.path.insert(0, str(LABRAM_REPO))

from modeling_finetune import labram_base_patch200_200  # noqa: E402


class EpochDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_for_labram(epochs: np.ndarray, patch_size: int = 200, n_patches: int = 6) -> np.ndarray:
    """Convert [N, C, T] epochs into LaBraM input shape [N, 1, A, T_patch]."""
    target_len = patch_size * n_patches
    if epochs.shape[2] < target_len:
        pad = target_len - epochs.shape[2]
        epochs = np.pad(epochs, ((0, 0), (0, 0), (0, pad)))
    elif epochs.shape[2] > target_len:
        epochs = epochs[:, :, :target_len]

    # Mixed montages are reduced to one virtual channel for compatibility.
    epochs = epochs.mean(axis=1, keepdims=True)
    epochs = epochs.reshape(epochs.shape[0], 1, n_patches, patch_size)
    return epochs.astype(np.float32, copy=False)


def load_data(epoch_index_path: Path):
    df = pd.read_csv(epoch_index_path)
    if "stimulus_id" in df.columns and "epoch_idx" in df.columns:
        return _load_data_long_format(df)
    return _load_data_per_recording(df)


def _load_data_long_format(df: pd.DataFrame):
    X_list, y_list, g_list = [], [], []
    for epochs_file, grp in df.groupby("epochs_file"):
        arr = np.load(epochs_file)
        for row in grp.itertuples(index=False):
            epoch_idx = int(getattr(row, "epoch_idx", 0))
            stim_id = getattr(row, "stimulus_id", None)
            if stim_id is None:
                continue
            x = arr[epoch_idx : epoch_idx + 1]
            x = prepare_for_labram(x)
            X_list.append(x)
            y_list.append(stim_id)
            subj = getattr(row, "subject_id", "unknown")
            ds = getattr(row, "dataset_id", "unknown")
            g_list.append(f"{ds}::{subj}")
    X = np.concatenate(X_list, axis=0) if X_list else np.empty((0, 1, 6, 200), dtype=np.float32)
    y = np.array(y_list)
    g = np.array(g_list)
    return X, y, g


def _load_data_per_recording(df: pd.DataFrame):
    X_all, y_all, g_all = [], [], []
    for row in df.itertuples(index=False):
        epochs = np.load(row.epochs_file)
        epochs = prepare_for_labram(epochs)
        X_all.append(epochs)
        y_all.extend([row.modality] * len(epochs))
        g_all.extend([f"{row.dataset_id}::{row.subject_id}"] * len(epochs))
    X = np.concatenate(X_all, axis=0)
    y = np.array(y_all)
    g = np.array(g_all)
    return X, y, g


def train_labram(
    epoch_index_path: Path = EPOCH_INDEX,
    out_metrics_path: Path = OUT_METRICS,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cpu",
    checkpoint: str = "",
    write_json: bool = True,
) -> dict:
    X, y, groups = load_data(epoch_index_path)
    if len(y) < 2:
        print(f"Not enough samples ({len(y)}); need at least 2.")
        return {"accuracy": None, "n_train": 0, "n_test": 0, "label_classes": []}
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_samples = len(y_enc)
    if n_samples < 10 or len(set(groups)) < 2:
        split = ShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(split.split(X, y_enc))
    else:
        split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(split.split(X, y_enc, groups))

    train_ds = EpochDataset(X[train_idx], y_enc[train_idx])
    test_ds = EpochDataset(X[test_idx], y_enc[test_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = labram_base_patch200_200(
        num_classes=len(le.classes_),
        EEG_size=1200,
        init_values=0.1,
    ).to(device)
    input_chans = [0, 1]  # cls + FP1 placeholder index

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        current = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"Loaded checkpoint weights: matched={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, input_chans=input_chans), yb)
            loss.backward()
            opt.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, input_chans=input_chans).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += len(yb)

    metrics = {
        "accuracy": float(correct / max(total, 1)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "label_classes": list(le.classes_),
        "model": "labram_base_patch200_200",
        "checkpoint": checkpoint or None,
        "input_shape": [int(X.shape[1]), int(X.shape[2]), int(X.shape[3])],
        "note": "Uses official LaBraM architecture. For best quality, pass a pretrained checkpoint and dataset-specific channel mapping.",
    }
    if write_json:
        out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        out_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved Labram metrics to {out_metrics_path}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to pretrained LaBraM checkpoint (.pth)")
    parser.add_argument("--epoch-index", type=str, default=str(EPOCH_INDEX))
    parser.add_argument("--out-metrics", type=str, default=str(OUT_METRICS))
    args = parser.parse_args()
    train_labram(
        epoch_index_path=Path(args.epoch_index),
        out_metrics_path=Path(args.out_metrics),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        checkpoint=args.checkpoint,
        write_json=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
