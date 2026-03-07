from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import welch


def _is_long_format(df: pd.DataFrame) -> bool:
    return "stimulus_id" in df.columns and "epoch_idx" in df.columns


def summarise_epoch_index(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "datasets": 0,
            "subjects": 0,
            "recordings": 0,
            "total_epochs": 0,
            "channels_min": 0,
            "channels_max": 0,
            "samples_per_epoch": 0,
        }
    if _is_long_format(df):
        recordings = int(df["epochs_file"].nunique())
        total_epochs = len(df)
    else:
        recordings = len(df)
        total_epochs = int(pd.to_numeric(df["n_epochs"], errors="coerce").fillna(0).sum())
    return {
        "datasets": int(df["dataset_id"].nunique()),
        "subjects": int(df["subject_id"].nunique()),
        "recordings": recordings,
        "total_epochs": total_epochs,
        "channels_min": int(pd.to_numeric(df["n_channels"], errors="coerce").fillna(0).min()),
        "channels_max": int(pd.to_numeric(df["n_channels"], errors="coerce").fillna(0).max()),
        "samples_per_epoch": int(pd.to_numeric(df["n_samples"], errors="coerce").fillna(0).mode().iloc[0]) if "n_samples" in df.columns else 0,
    }


def dataset_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    for col in ("n_epochs", "n_channels", "n_samples"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    if _is_long_format(df):
        out = (
            work.groupby("dataset_id", as_index=False)
            .agg(
                subjects=("subject_id", "nunique"),
                recordings=("epochs_file", "nunique"),
                total_epochs=("epoch_idx", "count"),
                avg_channels=("n_channels", "mean"),
                avg_samples=("n_samples", "mean"),
            )
            .sort_values("total_epochs", ascending=False)
        )
    else:
        out = (
            work.groupby("dataset_id", as_index=False)
            .agg(
                subjects=("subject_id", "nunique"),
                recordings=("source_file", "count"),
                total_epochs=("n_epochs", "sum"),
                avg_channels=("n_channels", "mean"),
                avg_samples=("n_samples", "mean"),
            )
            .sort_values("total_epochs", ascending=False)
        )
    return out


def describe_epoch_tensor(epoch_data: np.ndarray) -> pd.DataFrame:
    n_epochs, n_channels, _ = epoch_data.shape
    rows: list[dict[str, float | int]] = []
    for channel_idx in range(n_channels):
        channel_values = epoch_data[:, channel_idx, :].reshape(n_epochs, -1)
        rows.append(
            {
                "channel": int(channel_idx),
                "mean": float(channel_values.mean()),
                "std": float(channel_values.std()),
                "min": float(channel_values.min()),
                "max": float(channel_values.max()),
                "peak_to_peak": float(channel_values.max() - channel_values.min()),
            }
        )
    return pd.DataFrame(rows)


def compute_psd(channel_signal: np.ndarray, sfreq: float = 256.0) -> tuple[np.ndarray, np.ndarray]:
    freqs, power = welch(channel_signal, fs=sfreq, nperseg=min(len(channel_signal), 512))
    return freqs, power


def compare_tensor_stats(original: np.ndarray, augmented: np.ndarray) -> pd.DataFrame:
    # Compare only common channel span to support shape differences.
    common_channels = min(original.shape[1], augmented.shape[1])
    orig = original[:, :common_channels, :]
    aug = augmented[:, :common_channels, :]

    rows: list[dict[str, float | int]] = []
    for channel_idx in range(common_channels):
        o = orig[:, channel_idx, :]
        a = aug[:, channel_idx, :]
        rows.append(
            {
                "channel": channel_idx,
                "orig_std": float(o.std()),
                "aug_std": float(a.std()),
                "delta_std": float(a.std() - o.std()),
                "orig_peak_to_peak": float(o.max() - o.min()),
                "aug_peak_to_peak": float(a.max() - a.min()),
                "delta_peak_to_peak": float((a.max() - a.min()) - (o.max() - o.min())),
            }
        )
    return pd.DataFrame(rows)
