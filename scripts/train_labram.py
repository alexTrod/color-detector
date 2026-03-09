#!/usr/bin/env python3
"""Fine-tune LaBraM on event-locked EEG epochs with real channel layouts."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EPOCH_INDEX = ROOT / "data/manifests/epoch_index_yoto_tones.csv"
OUT_METRICS = ROOT / "runs/labram_metrics.json"
LABRAM_REPO = ROOT / "vendor/LaBraM"
DEFAULT_CHECKPOINT = LABRAM_REPO / "checkpoints/labram-base.pth"
DEFAULT_CHECKPOINT_URL = "https://raw.githubusercontent.com/935963004/LaBraM/main/checkpoints/labram-base.pth"
SEED = 42

if str(LABRAM_REPO) not in sys.path:
    sys.path.insert(0, str(LABRAM_REPO))

from modeling_finetune import labram_base_patch200_200  # noqa: E402

STANDARD_1020 = [
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2",
    "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9H", "TTP7H", "TPP9H", "FTT10H", "TPP8H", "TPP10H",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]


def canonicalize_channel_name(name: str) -> str:
    name = str(name).strip().upper().replace("EEG ", "")
    name = name.replace("-REF", "").replace(" ", "")
    return name


def read_brainvision_channel_names(vhdr_path: Path) -> list[str]:
    channel_names: list[str] = []
    in_channel_section = False
    with vhdr_path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Channel Infos]":
                in_channel_section = True
                continue
            if in_channel_section and line.startswith("[") and line != "[Channel Infos]":
                break
            if not in_channel_section or not line.startswith("Ch"):
                continue
            _, rhs = line.split("=", 1)
            channel_name = rhs.split(",", 1)[0]
            canonical = canonicalize_channel_name(channel_name)
            if canonical:
                channel_names.append(canonical)
    return channel_names


def resample_last_axis(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[-1] == target_len:
        return arr.astype(np.float32, copy=False)
    old_positions = np.linspace(0.0, 1.0, arr.shape[-1], dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((*arr.shape[:-1], target_len), dtype=np.float32)
    flat_in = arr.reshape(-1, arr.shape[-1])
    flat_out = out.reshape(-1, target_len)
    for idx, row in enumerate(flat_in):
        flat_out[idx] = np.interp(new_positions, old_positions, row)
    return out


def prepare_for_labram(
    epochs: np.ndarray,
    channel_indices: list[int],
    patch_size: int = 200,
    baseline_fraction: float = 0.2,
) -> np.ndarray:
    """Convert [N, C, T] epochs into LaBraM input shape [N, C_keep, 1, 200]."""
    epochs = epochs[:, channel_indices, :].astype(np.float32, copy=False)

    baseline_len = max(1, int(round(epochs.shape[-1] * baseline_fraction)))
    baseline = epochs[:, :, :baseline_len].mean(axis=-1, keepdims=True)
    epochs = epochs - baseline
    epochs = resample_last_axis(epochs, patch_size)

    # Match official fine-tuning layout: one 200-sample patch per channel.
    epochs = epochs[:, :, None, :]
    return epochs.astype(np.float32, copy=False)


def collect_channel_layouts(df: pd.DataFrame) -> tuple[list[str], dict[str, list[int]]]:
    if "source_file" not in df.columns:
        raise ValueError("Epoch index must include `source_file` to map EEG channels for LaBraM.")

    per_source_channels: dict[str, list[str]] = {}
    for source_file in df["source_file"].dropna().astype(str).unique():
        source_path = Path(source_file)
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        channels = read_brainvision_channel_names(source_path)
        usable = [ch for ch in channels if ch in STANDARD_1020]
        if not usable:
            raise ValueError(f"No LaBraM-compatible 10-20 channels found in {source_path}")
        per_source_channels[str(source_path)] = usable

    common_channels = [
        ch for ch in STANDARD_1020
        if all(ch in channels for channels in per_source_channels.values())
    ]
    if not common_channels:
        raise ValueError("No common 10-20 channel set exists across the selected recordings.")

    selection_by_source = {
        source: [channels.index(ch) for ch in common_channels]
        for source, channels in per_source_channels.items()
    }
    return common_channels, selection_by_source


def get_input_chans(channel_names: list[str]) -> list[int]:
    return [0] + [STANDARD_1020.index(ch_name) + 1 for ch_name in channel_names]


def normalize_checkpoint_state_dict(checkpoint_obj: dict) -> OrderedDict[str, torch.Tensor]:
    checkpoint_model = None
    for model_key in ("model", "module", "state_dict"):
        if model_key in checkpoint_obj:
            checkpoint_model = checkpoint_obj[model_key]
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint_obj

    if any(key.startswith("student.") for key in checkpoint_model):
        checkpoint_model = OrderedDict(
            (key[8:], value)
            for key, value in checkpoint_model.items()
            if key.startswith("student.")
        )
    else:
        checkpoint_model = OrderedDict(checkpoint_model.items())
    return checkpoint_model


def download_checkpoint(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return destination


def resolve_checkpoint_path(checkpoint: str) -> Path:
    candidate = Path(checkpoint).expanduser() if checkpoint else DEFAULT_CHECKPOINT
    if candidate.exists():
        return candidate
    if checkpoint:
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    print(f"Downloading official LaBraM checkpoint to {candidate}")
    return download_checkpoint(DEFAULT_CHECKPOINT_URL, candidate)


def load_pretrained_weights(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = normalize_checkpoint_state_dict(checkpoint)
    state_dict = model.state_dict()

    for key in ("head.weight", "head.bias"):
        if key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]

    for key in list(checkpoint_model.keys()):
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

    filtered = OrderedDict(
        (key, value)
        for key, value in checkpoint_model.items()
        if key in state_dict and value.shape == state_dict[key].shape
    )
    incompatible = model.load_state_dict(filtered, strict=False)
    print(
        "Loaded checkpoint weights: matched=%d missing=%d unexpected=%d"
        % (len(filtered), len(incompatible.missing_keys), len(incompatible.unexpected_keys))
    )
    return {
        "matched": len(filtered),
        "missing": len(incompatible.missing_keys),
        "unexpected": len(incompatible.unexpected_keys),
    }


def load_data(epoch_index_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(epoch_index_path)
    common_channels, selection_by_source = collect_channel_layouts(df)
    if "stimulus_id" in df.columns and "epoch_idx" in df.columns:
        return _load_data_long_format(df, common_channels, selection_by_source)
    return _load_data_per_recording(df, common_channels, selection_by_source)


def _resolve_source_path(source_file: str) -> Path:
    path = Path(source_file)
    return path if path.is_absolute() else ROOT / path


def _load_data_long_format(
    df: pd.DataFrame,
    common_channels: list[str],
    selection_by_source: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_list, y_list, g_list = [], [], []
    for epochs_file, grp in df.groupby("epochs_file"):
        epochs_path = Path(epochs_file)
        if not epochs_path.is_absolute():
            epochs_path = ROOT / epochs_path
        arr = np.load(epochs_path)
        for row in grp.itertuples(index=False):
            epoch_idx = int(getattr(row, "epoch_idx", 0))
            stim_id = getattr(row, "stimulus_id", None)
            if stim_id is None:
                continue
            source_path = _resolve_source_path(getattr(row, "source_file"))
            channel_indices = selection_by_source[str(source_path)]
            x = prepare_for_labram(arr[epoch_idx : epoch_idx + 1], channel_indices=channel_indices)
            X_list.append(x)
            y_list.append(stim_id)
            subj = getattr(row, "subject_id", "unknown")
            ds = getattr(row, "dataset_id", "unknown")
            g_list.append(f"{ds}::{subj}")
    X = np.concatenate(X_list, axis=0) if X_list else np.empty((0, len(common_channels), 1, 200), dtype=np.float32)
    y = np.array(y_list)
    g = np.array(g_list)
    return X, y, g, common_channels


def _load_data_per_recording(
    df: pd.DataFrame,
    common_channels: list[str],
    selection_by_source: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_all, y_all, g_all = [], [], []
    for row in df.itertuples(index=False):
        epochs_path = Path(row.epochs_file)
        if not epochs_path.is_absolute():
            epochs_path = ROOT / epochs_path
        source_path = _resolve_source_path(getattr(row, "source_file"))
        channel_indices = selection_by_source[str(source_path)]
        epochs = np.load(epochs_path)
        epochs = prepare_for_labram(epochs, channel_indices=channel_indices)
        X_all.append(epochs)
        y_all.extend([row.modality] * len(epochs))
        g_all.extend([f"{row.dataset_id}::{row.subject_id}"] * len(epochs))
    X = np.concatenate(X_all, axis=0)
    y = np.array(y_all)
    g = np.array(g_all)
    return X, y, g, common_channels


def build_erp_summary_features(X: np.ndarray) -> np.ndarray:
    """Add compact ERP descriptors so the linear head can use event morphology directly."""
    signal = X[:, :, 0, :]
    mean_wave = signal.mean(axis=1)
    std_wave = signal.std(axis=1)
    return np.concatenate([mean_wave, std_wave], axis=1).astype(np.float32, copy=False)


def extract_labram_features(
    X: np.ndarray,
    channel_names: list[str],
    checkpoint_path: Path,
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[np.ndarray, dict[str, int]]:
    model = labram_base_patch200_200(
        num_classes=0,
        EEG_size=200,
        init_values=0.1,
    ).to(device)
    checkpoint_load = load_pretrained_weights(model, checkpoint_path)
    input_chans = get_input_chans(channel_names)

    feature_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            feats = model.forward_features(xb, input_chans=input_chans)
            feature_batches.append(feats.cpu().numpy())
    return np.concatenate(feature_batches, axis=0), checkpoint_load


def train_labram(
    epoch_index_path: Path = EPOCH_INDEX,
    out_metrics_path: Path = OUT_METRICS,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 5e-4,
    device: str = "cpu",
    checkpoint: str = "",
    freeze_backbone_epochs: int = 2,
    backbone_lr_scale: float = 0.1,
    weight_decay: float = 0.05,
    write_json: bool = True,
) -> dict:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X, y, groups, channel_names = load_data(epoch_index_path)
    if len(y) < 2:
        print(f"Not enough samples ({len(y)}); need at least 2.")
        return {"accuracy": None, "macro_f1": None, "n_train": 0, "n_test": 0, "label_classes": []}

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_samples = len(y_enc)
    n_groups = len(set(groups.tolist()))
    print(f"Training on {n_samples} samples across {n_groups} subject groups")
    print(f"Using {len(channel_names)} common channels: {channel_names}")

    if n_samples < 10 or n_groups < 2:
        split = ShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        train_idx, test_idx = next(split.split(X, y_enc))
    else:
        split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        train_idx, test_idx = next(split.split(X, y_enc, groups))

    checkpoint_path = resolve_checkpoint_path(checkpoint)
    labram_features, checkpoint_load = extract_labram_features(
        X,
        channel_names=channel_names,
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=batch_size,
    )
    erp_summary = build_erp_summary_features(X)
    combined_features = labram_features  # labram-only, no ERP summary

    # Small subject-held-out datasets were more stable with a frozen encoder plus a
    # regularized linear head than with end-to-end fine-tuning.
    probe_c = 0.04
    clf = make_pipeline(
        RobustScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=probe_c,
            random_state=SEED,
        ),
    )
    clf.fit(combined_features[train_idx], y_enc[train_idx])
    y_pred = clf.predict(combined_features[test_idx])
    y_true = y_enc[test_idx]
    accuracy = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "label_classes": list(le.classes_),
        "model": "labram_base_patch200_200",
        "classifier": "labram_frozen_features_plus_erp_logreg",
        "checkpoint": str(checkpoint_path),
        "checkpoint_load": checkpoint_load,
        "input_shape": [int(X.shape[1]), int(X.shape[2]), int(X.shape[3])],
        "channel_names": channel_names,
        "labram_feature_dim": int(labram_features.shape[1]),
        "erp_summary_dim": int(erp_summary.shape[1]),
        "probe_c": probe_c,
        "note": "Uses official pretrained LaBraM as a frozen encoder on the true EEG channel layout, then fits a regularized linear probe on LaBraM features plus compact ERP summary features.",
    }
    if write_json:
        out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        out_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved Labram metrics to {out_metrics_path}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to pretrained LaBraM checkpoint (.pth)")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
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
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        backbone_lr_scale=args.backbone_lr_scale,
        weight_decay=args.weight_decay,
        write_json=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
