#!/usr/bin/env python3
"""Preprocess color EEG datasets (Hajonides, Bae & Luck, Chauhan) from .mat to epoch index + .npy."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/raw_samples"
OUT_EPOCH_DIR = ROOT / "data/processed/epochs_color"
OUT_INDEX = ROOT / "data/manifests/epoch_index_color.csv"

# Target time samples for alignment with YOTO-style pipeline (will be resampled to 200 in LaBraM)
TARGET_SAMPLES = 250
TARGET_SFREQ = 250


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        try:
            import yaml
            return yaml.safe_load(f) or {}
        except ImportError:
            return {}


def _infer_epochs_and_labels(mat: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """From a loaded .mat dict, return (epochs [N,C,T], labels [N]) or None."""
    # Bae & Luck: nested data.eeg and data.ColorWheelID / ColorBinID
    if "data" in mat:
        d = mat["data"]
        if hasattr(d, "shape") and d.size >= 1:
            try:
                inner = d.flat[0] if hasattr(d, "flat") else d[0, 0]
                if hasattr(inner, "eeg"):
                    eeg = np.asarray(inner.eeg, dtype=np.float32)
                    if eeg.ndim == 3:
                        n = eeg.shape[0]
                        for label_name in ("ColorWheelID", "ColorBinID", "TargetColorAngle"):
                            if hasattr(inner, label_name):
                                lab = np.asarray(getattr(inner, label_name), dtype=np.float64).ravel()
                                if lab.size >= n:
                                    return eeg, lab[:n]
                        return eeg, np.arange(n, dtype=np.int32)
            except Exception:
                pass

    # Strip MATLAB metadata
    skip = {"__header__", "__version__", "__globals__"}
    keys = [k for k in mat if k not in skip and not k.startswith("_")]
    if not keys:
        return None

    # Find 3D array (epochs) and 1D/2D labels
    data_arr = None
    label_arr = None
    n_epochs = None
    for k in keys:
        v = mat[k]
        if not hasattr(v, "shape"):
            continue
        if v.ndim == 3:
            # (N,C,T) or (C,T,N)
            if data_arr is None:
                data_arr = (k, v)
                n_epochs = v.shape[0] if v.shape[0] < v.shape[-1] else v.shape[-1]
        elif v.ndim == 1 or (v.ndim == 2 and min(v.shape) == 1):
            flat = np.asarray(v).ravel()
            if flat.size > 1 and (np.issubdtype(flat.dtype, np.integer) or np.issubdtype(flat.dtype, np.floating)):
                if label_arr is None or (n_epochs and flat.size == n_epochs) or flat.size > (label_arr[1].size if label_arr else 0):
                    label_arr = (k, flat)

    if data_arr is None:
        return None
    name_d, arr = data_arr
    # Normalize to (N, C, T)
    if arr.shape[0] < arr.shape[-1]:
        # likely (C, T, N)
        arr = np.transpose(arr, (2, 0, 1))
    epochs = np.asarray(arr, dtype=np.float32)
    n_epochs = epochs.shape[0]
    if label_arr is None or label_arr[1].size != n_epochs:
        # Prefer label with same length as epochs
        if label_arr is not None and label_arr[1].ndim == 2 and label_arr[1].shape[0] == n_epochs:
            labels = np.asarray(label_arr[1][:, 0]).ravel()
        else:
            labels = np.arange(n_epochs, dtype=np.int32)
    else:
        labels = np.asarray(label_arr[1]).ravel()[:n_epochs]
    return epochs, labels


def _hue_to_stimulus_id(hue_val: float | int) -> str:
    """Map numeric hue to stimulus_id string (e.g. 12-bin or 48-bin)."""
    if isinstance(hue_val, (int, np.integer)):
        return f"hue_{int(hue_val)}"
    return f"hue_{int(round(float(hue_val)))}"


def _loadmat(path: Path) -> dict:
    """Load .mat file; try scipy then h5py (v7.3)."""
    try:
        from scipy.io import loadmat
        return loadmat(str(path), squeeze_me=False, struct_as_record=False)
    except OSError:
        try:
            import h5py
            raw = {}
            with h5py.File(path, "r") as f:
                for k, v in f.items():
                    if isinstance(v, h5py.Dataset):
                        raw[k] = np.array(v)
            return raw
        except ImportError:
            raise
        except Exception:
            raise


def process_hajonides_mat(mat_path: Path, dataset_id: str, subject_id: str) -> tuple[np.ndarray, list[str]]:
    """Load one Hajonides .mat; return (epochs [N,C,T], stimulus_ids)."""
    raw = _loadmat(mat_path)
    out = _infer_epochs_and_labels(raw)
    if out is None:
        raise ValueError(f"Could not infer epochs/labels from {mat_path}")
    epochs, labels = out
    # Resample to TARGET_SAMPLES if needed
    if epochs.shape[-1] != TARGET_SAMPLES:
        old_t = epochs.shape[-1]
        new_t = TARGET_SAMPLES
        out_arr = np.empty((epochs.shape[0], epochs.shape[1], new_t), dtype=np.float32)
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                out_arr[i, j, :] = np.interp(
                    np.linspace(0, 1, new_t), np.linspace(0, 1, old_t), epochs[i, j, :]
                )
        epochs = out_arr
    stimulus_ids = [_hue_to_stimulus_id(l) for l in labels]
    return epochs, stimulus_ids


def process_baeluck_mat(mat_path: Path, dataset_id: str, subject_id: str) -> tuple[np.ndarray, list[str]]:
    """Load one Bae & Luck .mat; return (epochs [N,C,T], stimulus_ids)."""
    raw = _loadmat(mat_path)
    # Bae & Luck: top-level "data" (1,1) struct with .eeg and .ColorWheelID
    if "data" in raw:
        d = raw["data"]
        try:
            inner = d[0, 0] if d.shape == (1, 1) else d.flat[0]
            eeg = np.asarray(inner.eeg, dtype=np.float32)
            if eeg.ndim == 3:
                n = eeg.shape[0]
                lab = None
                for name in ("ColorWheelID", "ColorBinID", "TargetColorAngle"):
                    if hasattr(inner, name):
                        lab = np.asarray(getattr(inner, name)).ravel()[:n]
                        break
                if lab is None:
                    lab = np.arange(n, dtype=np.int32)
                out = (eeg, lab)
            else:
                out = None
        except Exception:
            out = None
    else:
        out = None
    if out is None:
        out = _infer_epochs_and_labels(raw)
    if out is None:
        raise ValueError(f"Could not infer epochs/labels from {mat_path}")
    epochs, labels = out
    if epochs.shape[-1] != TARGET_SAMPLES:
        old_t = epochs.shape[-1]
        new_t = TARGET_SAMPLES
        out = np.empty((epochs.shape[0], epochs.shape[1], new_t), dtype=np.float32)
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                out[i, j, :] = np.interp(
                    np.linspace(0, 1, new_t), np.linspace(0, 1, old_t), epochs[i, j, :]
                )
        epochs = out
    stimulus_ids = [f"color_{int(l)}" if np.issubdtype(np.asarray(l).dtype, np.integer) else f"color_{l}" for l in labels]
    return epochs, stimulus_ids


def process_chauhan_file(file_path: Path, dataset_id: str, subject_id: str) -> tuple[np.ndarray, list[str]] | None:
    """Load Chauhan .mat/.set if present; same interface."""
    if file_path.suffix.lower() != ".mat":
        return None
    raw = _loadmat(file_path)
    out = _infer_epochs_and_labels(raw)
    if out is None:
        return None
    epochs, labels = out
    if epochs.shape[-1] != TARGET_SAMPLES:
        old_t = epochs.shape[-1]
        new_t = TARGET_SAMPLES
        out_arr = np.empty((epochs.shape[0], epochs.shape[1], new_t), dtype=np.float32)
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                out_arr[i, j, :] = np.interp(
                    np.linspace(0, 1, new_t), np.linspace(0, 1, old_t), epochs[i, j, :]
                )
        epochs = out_arr
    stimulus_ids = [_hue_to_stimulus_id(l) for l in labels]
    return epochs, stimulus_ids


def run_preprocess_color(
    raw_root: Path = RAW_ROOT,
    out_dir: Path = OUT_EPOCH_DIR,
    out_index: Path = OUT_INDEX,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []

    # Hajonides
    hajonides_dir = raw_root / "hajonides_j289e"
    if hajonides_dir.exists():
        for mat_path in sorted(hajonides_dir.glob("*.mat")):
            m = re.search(r"S(\d+)", mat_path.name, re.I)
            subject_id = f"sub-{m.group(1)}" if m else f"sub-{mat_path.stem}"
            try:
                epochs, stimulus_ids = process_hajonides_mat(mat_path, "hajonides_j289e", subject_id)
            except Exception as e:
                print(f"Skip {mat_path.name}: {e}")
                continue
            stem = f"hajonides_j289e__{subject_id}__{mat_path.stem}"
            out_path = out_dir / f"{stem}.npy"
            np.save(out_path, epochs)
            for epoch_idx, stim_id in enumerate(stimulus_ids):
                index_rows.append({
                    "dataset_id": "hajonides_j289e",
                    "subject_id": subject_id,
                    "source_file": str(mat_path.resolve()),
                    "epochs_file": str(out_path.resolve()),
                    "epoch_idx": epoch_idx,
                    "stimulus_id": stim_id,
                    "n_channels": int(epochs.shape[1]),
                    "n_samples": int(epochs.shape[2]),
                })
            print(f"hajonides {subject_id}: {epochs.shape[0]} epochs -> {out_path.name}")

    # Bae & Luck
    baeluck_dir = raw_root / "baeluck_jnwut"
    if baeluck_dir.exists():
        for mat_path in sorted(baeluck_dir.glob("*.mat")):
            m = re.search(r"_(\d+)\.mat$", mat_path.name)
            subject_id = f"sub-{m.group(1)}" if m else f"sub-{mat_path.stem}"
            try:
                epochs, stimulus_ids = process_baeluck_mat(mat_path, "baeluck_jnwut", subject_id)
            except Exception as e:
                print(f"Skip {mat_path.name}: {e}")
                continue
            stem = f"baeluck_jnwut__{subject_id}__{mat_path.stem}"
            out_path = out_dir / f"{stem}.npy"
            np.save(out_path, epochs)
            for epoch_idx, stim_id in enumerate(stimulus_ids):
                index_rows.append({
                    "dataset_id": "baeluck_jnwut",
                    "subject_id": subject_id,
                    "source_file": str(mat_path.resolve()),
                    "epochs_file": str(out_path.resolve()),
                    "epoch_idx": epoch_idx,
                    "stimulus_id": stim_id,
                    "n_channels": int(epochs.shape[1]),
                    "n_samples": int(epochs.shape[2]),
                })
            print(f"baeluck {subject_id}: {epochs.shape[0]} epochs -> {out_path.name}")

    # Chauhan
    chauhan_dir = raw_root / "chauhan_v9ewj"
    if chauhan_dir.exists():
        for mat_path in sorted(chauhan_dir.glob("*.mat")):
            subject_id = f"sub-{mat_path.stem}"
            out_tup = process_chauhan_file(mat_path, "chauhan_v9ewj", subject_id)
            if out_tup is None:
                continue
            epochs, stimulus_ids = out_tup
            stem = f"chauhan_v9ewj__{subject_id}__{mat_path.stem}"
            out_path = out_dir / f"{stem}.npy"
            np.save(out_path, epochs)
            for epoch_idx, stim_id in enumerate(stimulus_ids):
                index_rows.append({
                    "dataset_id": "chauhan_v9ewj",
                    "subject_id": subject_id,
                    "source_file": str(mat_path.resolve()),
                    "epochs_file": str(out_path.resolve()),
                    "epoch_idx": epoch_idx,
                    "stimulus_id": stim_id,
                    "n_channels": int(epochs.shape[1]),
                    "n_samples": int(epochs.shape[2]),
                })
            print(f"chauhan {subject_id}: {epochs.shape[0]} epochs -> {out_path.name}")

    if index_rows:
        out_index.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(index_rows).to_csv(out_index, index=False)
        out_index.with_suffix(".json").write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
        print(f"Saved {len(index_rows)} epoch rows to {out_index}")
    else:
        print("No color epochs produced. Ensure data/raw_samples/{hajonides_j289e,baeluck_jnwut,chauhan_v9ewj} exist and contain .mat files.")
    return {
        "n_epochs": len(index_rows),
        "out_index": str(out_index),
        "out_dir": str(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess color EEG .mat into epoch index + .npy")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_EPOCH_DIR)
    parser.add_argument("--out-index", type=Path, default=OUT_INDEX)
    args = parser.parse_args()
    run_preprocess_color(raw_root=args.raw_root, out_dir=args.out_dir, out_index=args.out_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
